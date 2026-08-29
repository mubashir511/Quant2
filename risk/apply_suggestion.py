from dataclasses import dataclass, field
from typing import Callable

from ai.portfolio_suggest import AllocationEntry
from data.mt5_source import AccountSummary, ContractSpec, MarketAsset, PendingOrder, Position


@dataclass
class PlannedOrder:
    symbol: str
    action: str  # "open" | "increase" | "reduce" | "close" | "hold" | "infeasible" | "cancel" | "amend_pending" | "amend_position"
    side: str  # "buy" or "sell"
    volume: float
    order_type: str  # "limit" | "market" | "none"
    price: float | None
    stop_loss: float | None
    take_profit: float | None = None
    tickets_to_close: list[tuple[int, float]] = field(default_factory=list)
    reason: str = ""
    pending_tickets_to_cancel: list[int] = field(default_factory=list)  # for "cancel" / "amend_pending"
    position_tickets_to_amend: list[int] = field(default_factory=list)  # for "amend_position"


def _round_down_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    steps = int(value / step + 1e-9)
    return round(steps * step, 8)


def _pending_order_side(order: PendingOrder) -> str:
    return "buy" if order.order_type.startswith("buy") else "sell"


def _within_tolerance(a: float | None, b: float | None, tolerance_pct: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    reference = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / reference * 100 <= tolerance_pct


def compute_rebalance_plan(
    positions: list[Position],
    account: AccountSummary,
    allocation: dict[str, AllocationEntry],
    get_spec: Callable[[str], ContractSpec | None],
    market_prices: dict[str, MarketAsset],
    price_sanity_band_pct: float = 5.0,
    pending_orders: list[PendingOrder] | None = None,
    amend_tolerance_pct: float = 0.05,
) -> list[PlannedOrder]:
    """Pure, deterministic diff between what's currently held and the
    AI's target allocation — no network calls, no AI reasoning here, only
    arithmetic (mirrors risk/rebalance.py's "pure function" pattern).

    Position size is derived from REAL RISK, not margin-affordability:
    `pct` means "this many % of equity is what's genuinely at risk if the
    stop is hit" (matching what the AI is explicitly told to assume when
    it computes its own "aggregate heat" — see ai/portfolio_suggest.py's
    and ai/ftmo_suggest.py's shared "pct = capital at risk" instruction).
    See this function's own git history / project memory for why margin-
    based sizing was replaced with this: it silently multiplied real
    exposure by an instrument's leverage ratio, confirmed live to cause a
    real account-terminating loss.

    Both directions are supported now, each `AllocationEntry.side` value
    ("buy" or "sell") driving three distinct cases per symbol:
    - **Same direction as currently held** (or currently flat): resized
      via open/increase/reduce/hold, exactly as before.
    - **Genuine flip** (held long, target short, or vice versa): the held
      side is closed IN FULL as its own `PlannedOrder`, then the target
      side is sized completely fresh (never netted against what was just
      closed) — two separate orders, since a broker can't atomically net
      a long against a short on the same ticket.
    - **Genuinely mixed** (both a buy and a sell open on the same symbol
      at once — a real hedge-mode anomaly this system never intentionally
      creates): still an unconditional full close of everything, same as
      the old long-only behavior, just no longer conflated with a clean
      single-direction short that legitimately matches its own target.

    A symbol held but missing from `allocation` entirely is treated as a
    0% target (close it) — matches the prompt's explicit instruction that
    every held symbol must appear in the final JSON even at 0, with this
    as the safety-net default in case that instruction is ever missed.

    Any proposed limit-entry price is clamped to within
    `price_sanity_band_pct` of the live reference price for that
    direction — the ASK for a buy, the BID for a sell, since those are
    the two different real prices a limit order actually executes
    against. A missing or wrong-side stop-loss (below entry for a buy,
    above entry for a sell) now makes the whole position `infeasible`
    rather than silently falling back to an unstopped, unbounded-risk
    order — sizing by risk has nothing to size against without a real
    stop, and a real stop is a mandatory field in the AI's own output
    schema, so its absence here is a genuine gap worth surfacing, not
    something to quietly paper over. A take-profit on the wrong side of
    entry (below entry for a buy, above for a sell) is still just
    dropped (noted in `reason`), since it doesn't affect sizing or risk.

    `pending_orders` (added 2026-08-23, direct user request — letting
    both Claude and Copilot re-assess an already-suggested position, not
    just propose fresh ones) makes this function aware of outstanding,
    not-yet-filled GTC limit orders, which it was completely blind to
    before: a stale one the fresh target no longer wants gets cancelled
    (`action="cancel"`), one whose price/stop/target changed gets
    replaced (`action="amend_pending"` — cancel + reopen, safe since an
    unfilled order has no realized exposure yet). `amend_tolerance_pct`
    is how close (as a %) an existing pending order's or position's own
    numbers must be to the fresh target before treating them as
    unchanged, rather than thrashing on trivial/rounding differences.
    An already-HELD position whose stop/target changed while its size
    didn't gets `action="amend_position"` instead of `"hold"` — a true
    in-place amend, deliberately never a close-then-reopen (see
    data/mt5_execution.py::modify_position_sltp's own docstring for why:
    FTMO's own daily-loss/max-loss tracking keys off REALIZED P&L, so a
    close purely to re-stamp a stop would prematurely realize floating
    P&L and distort those exact compliance numbers).

    IMPORTANT invariant this function must never break: `all_symbols`
    below stays EXACTLY `set(by_symbol) | allocation keys` — pending-
    order symbols are never added to it. A symbol whose only footprint
    is a still-unfilled Pending-Setup order (fired by Copilot mid-cycle,
    deliberately excluded from the allocation dict — see
    ai/copilot_execution.py::_build_carried_forward_allocation's own
    docstring) must stay completely invisible here, exactly as before;
    widening `all_symbols` to include it would give it `entry=None ->
    pct=0`, and the new "no longer wanted -> cancel" rule would then
    self-cancel that Pending Setup order the instant Copilot places it.
    """
    plans: list[PlannedOrder] = []

    by_symbol: dict[str, list[Position]] = {}
    for p in positions:
        by_symbol.setdefault(p.symbol, []).append(p)

    pending_by_symbol: dict[str, list[PendingOrder]] = {}
    for o in (pending_orders or []):
        pending_by_symbol.setdefault(o.symbol, []).append(o)

    all_symbols = set(by_symbol) | {s for s in allocation if s.upper() != "CASH"}

    for symbol in sorted(all_symbols):
        symbol_positions = by_symbol.get(symbol, [])
        symbol_pending = pending_by_symbol.get(symbol, [])
        entry = allocation.get(symbol)
        pct = entry.pct if entry is not None else 0.0

        sell_positions = [p for p in symbol_positions if p.side == "sell"]
        buy_positions = [p for p in symbol_positions if p.side == "buy"]

        # Genuinely mixed: both directions open on this symbol at once —
        # a real anomaly, not something this pipeline's own plans ever
        # intentionally create. Unconditional full close, independent of
        # whatever the target says; a fresh single-direction target can
        # be applied once flat, on the next run.
        if sell_positions and buy_positions:
            tickets = [(p.ticket, p.volume) for p in symbol_positions]
            total_volume = sum(p.volume for p in symbol_positions)
            plans.append(
                PlannedOrder(
                    symbol=symbol,
                    action="close",
                    side="sell",  # descriptive only — close/reduce execution keys off the real Position.side, not this field
                    volume=total_volume,
                    order_type="market",
                    price=None,
                    stop_loss=None,
                    tickets_to_close=tickets,
                    reason=(
                        "Mixed-direction position (both a long and a short "
                        "open on this symbol at once) — closing fully; a "
                        "fresh single-direction target can be applied once "
                        "flat."
                    ),
                )
            )
            continue

        held_side = "sell" if sell_positions else ("buy" if buy_positions else None)
        held_positions = sell_positions or buy_positions
        held_lots = sum(p.volume for p in held_positions)

        target_side = None
        if entry is not None and pct > 0:
            target_side = entry.side if entry.side in ("buy", "sell") else "buy"

        spec = get_spec(symbol)
        if spec is None or spec.margin_initial <= 0 or spec.trade_contract_size <= 0:
            # A missing/malformed spec can't size a fresh position (sizing
            # needs trade_contract_size as a real divisor — see below), but
            # closing what's already held needs no spec data at all (just
            # the held tickets/volume), so do that regardless rather than
            # leaving a position open indefinitely with no way to manage it.
            if held_lots > 0:
                plans.append(
                    PlannedOrder(
                        symbol=symbol,
                        action="close",
                        side=("sell" if held_side == "buy" else "buy"),
                        volume=held_lots,
                        order_type="market",
                        price=None,
                        stop_loss=None,
                        tickets_to_close=[(p.ticket, p.volume) for p in held_positions],
                        reason="No usable contract spec available for this symbol — closing held position.",
                    )
                )
            elif pct > 0:
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="infeasible",
                        side=target_side or "buy", volume=0.0,
                        order_type="none", price=None, stop_loss=None,
                        reason="No usable contract spec available for this symbol.",
                    )
                )
            continue

        # Genuine flip: single direction held, target wants the other —
        # close it in full now, then fall through and size the target
        # side completely fresh below (never netted against this close).
        if held_side is not None and target_side is not None and held_side != target_side:
            plans.append(
                PlannedOrder(
                    symbol=symbol,
                    action="close",
                    side=("sell" if held_side == "buy" else "buy"),
                    volume=held_lots,
                    order_type="market",
                    price=None,
                    stop_loss=None,
                    tickets_to_close=[(p.ticket, p.volume) for p in held_positions],
                    reason=(
                        f"Currently held {held_side}; target flips to "
                        f"{target_side} — closing the existing position "
                        f"fully before a fresh {target_side} is sized."
                    ),
                )
            )
            held_side, held_lots, held_positions = None, 0.0, []

        if pct <= 0:
            target_lots = 0.0
            clamped_price = None
            stop_loss = None
            take_profit = None
            price_note = ""
        else:
            # Resolve price/stop up front — sizing needs them, and reusing
            # the exact same clamped values for the order itself (below)
            # guarantees what was sized is what gets sent.
            asset = market_prices.get(symbol)
            proposed_price = entry.price if entry is not None else None
            clamped_price = None
            price_note = ""
            if asset is not None and proposed_price is not None:
                ref_price = asset.bid if target_side == "sell" else asset.ask
                band = price_sanity_band_pct / 100
                lo, hi = ref_price * (1 - band), ref_price * (1 + band)
                clamped_price = min(max(proposed_price, lo), hi)
                if clamped_price != proposed_price:
                    ref_label = "bid" if target_side == "sell" else "ask"
                    price_note = (
                        f" (clamped from {proposed_price:.4f} to stay within "
                        f"{price_sanity_band_pct:.0f}% of the live {ref_label} {ref_price:.4f})"
                    )

            stop_loss = entry.stop_loss if entry is not None else None
            if clamped_price is not None and stop_loss is not None:
                # Buy: stop must be BELOW entry. Sell: stop must be ABOVE
                # entry. Order matters — this must run before stop_distance
                # is ever computed below, since sizing trusts the sign is
                # already correct for target_side by that point.
                wrong_side = (
                    stop_loss >= clamped_price
                    if target_side == "buy"
                    else stop_loss <= clamped_price
                )
                if wrong_side:
                    stop_loss = None
                    price_note += " (suggested stop was on the wrong side of entry, dropped)"

            take_profit = entry.take_profit if entry is not None else None
            if clamped_price is not None and take_profit is not None:
                wrong_side_tp = (
                    take_profit <= clamped_price
                    if target_side == "buy"
                    else take_profit >= clamped_price
                )
                if wrong_side_tp:
                    take_profit = None
                    price_note += " (suggested take-profit was on the wrong side of entry, dropped)"

            if clamped_price is None or stop_loss is None:
                reason = (
                    "No live quote or suggested price available to size "
                    "this position by risk."
                    if clamped_price is None
                    else "No valid stop-loss available to size this "
                    "position by risk (missing, or on the wrong side of "
                    "entry) — refusing to send an unstopped, unbounded-"
                    f"risk order.{price_note}"
                )
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="infeasible", side=target_side, volume=0.0,
                        order_type="none", price=clamped_price, stop_loss=None,
                        reason=reason,
                    )
                )
                continue

            # abs() is safe here specifically because the wrong-side check
            # above already guarantees stop_loss's sign relative to
            # clamped_price is correct for target_side by this point.
            stop_distance = abs(clamped_price - stop_loss)
            risk_dollars = pct / 100 * account.equity
            target_lots = _round_down_to_step(
                risk_dollars / (stop_distance * spec.trade_contract_size), spec.volume_step
            )

        if pct > 0 and target_lots < spec.volume_min:
            plans.append(
                PlannedOrder(
                    symbol=symbol, action="infeasible", side=target_side, volume=0.0,
                    order_type="none", price=None, stop_loss=None,
                    reason=(
                        f"Risking {pct:.1f}% of equity against this stop "
                        f"distance can't afford even the minimum "
                        f"{spec.volume_min:g}-lot for this instrument."
                    ),
                )
            )
            continue

        if held_side is None and symbol_pending:
            # Nothing is filled yet for this symbol, but at least one
            # pending order already rests on it. `entry` is guaranteed
            # non-None here per this function's own invariant above
            # (all_symbols only ever contains held or allocation-key
            # symbols) — a symbol reached with held_side is None can
            # only be here via a real allocation key.
            all_tickets = [o.ticket for o in symbol_pending]

            if pct <= 0:
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="cancel",
                        side=_pending_order_side(symbol_pending[0]),
                        volume=sum(o.volume for o in symbol_pending),
                        order_type="none", price=None, stop_loss=None,
                        pending_tickets_to_cancel=all_tickets,
                        reason=(
                            "Mega session no longer wants this instrument "
                            "(pct: 0) — cancelling the outstanding pending order."
                        ),
                    )
                )
                continue

            # More than one resting order on the same symbol is an
            # anomaly this pipeline's own code never intentionally
            # creates — never resolve that to "hold"; always collapse it
            # to one clean, freshly-sized order at the target's terms.
            single_order = symbol_pending[0] if len(symbol_pending) == 1 else None
            unchanged = (
                single_order is not None
                and _pending_order_side(single_order) == target_side
                and abs(single_order.volume - target_lots) < spec.volume_step / 2
                and _within_tolerance(single_order.price_open, clamped_price, amend_tolerance_pct)
                and _within_tolerance(single_order.sl, stop_loss, amend_tolerance_pct)
                and _within_tolerance(single_order.tp, take_profit, amend_tolerance_pct)
            )
            if unchanged:
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="hold", side=target_side, volume=single_order.volume,
                        order_type="none", price=None, stop_loss=None,
                        reason="Outstanding pending order already matches today's target — nothing to change.",
                    )
                )
            else:
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="amend_pending", side=target_side, volume=target_lots,
                        order_type="limit", price=clamped_price, stop_loss=stop_loss, take_profit=take_profit,
                        pending_tickets_to_cancel=all_tickets,
                        reason=(
                            f"Mega session updated this pending order's terms — "
                            f"cancelling ticket(s) {all_tickets} and replacing "
                            f"with the new price/stop/target.{price_note}"
                        ),
                    )
                )
            continue

        delta = target_lots - held_lots

        if abs(delta) < spec.volume_step / 2:
            # Compares only held_positions[0]'s own sl/tp, not every
            # ticket individually — a KNOWN, named limitation (found on
            # self-review 2026-08-24): if this symbol ever accumulates
            # multiple tickets on the same side (e.g. an original open
            # plus a later top-up), a PARTIAL amend failure below could
            # leave one ticket un-amended forever, since a later poll's
            # aggregate check here would see whichever ticket MT5 happens
            # to return first and, if THAT one already matches, stop
            # retrying — never separately verifying the others. Still a
            # net improvement over the pre-existing behavior (which
            # never checked sl/tp consistency here AT ALL), and this
            # multi-ticket-per-symbol case is rare in practice for how
            # this pipeline actually opens positions; a full per-ticket
            # amend-and-verify would need real design work, not a quick
            # fix bolted on here.
            if held_lots > 0 and pct > 0 and (
                not _within_tolerance(held_positions[0].sl, stop_loss, amend_tolerance_pct)
                or not _within_tolerance(held_positions[0].tp, take_profit, amend_tolerance_pct)
            ):
                resolved_stop = stop_loss if stop_loss is not None else held_positions[0].sl
                resolved_target = take_profit if take_profit is not None else held_positions[0].tp
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="amend_position", side=held_side, volume=held_lots,
                        order_type="none", price=None,
                        stop_loss=resolved_stop, take_profit=resolved_target,
                        position_tickets_to_amend=[p.ticket for p in held_positions],
                        reason=(
                            f"Stop/target updated per today's mega session: {entry.reason}"
                            if entry is not None and entry.reason
                            else "Stop/target updated per today's mega session."
                        ),
                    )
                )
            else:
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="hold",
                        side=held_side or target_side or "buy", volume=held_lots,
                        order_type="none", price=None, stop_loss=None,
                        reason="Already at target allocation.",
                    )
                )

            if symbol_pending:
                # A stray/superseded pending order (e.g. a leftover
                # top-up attempt) resting on a symbol that's already at
                # its target size — cancel it as its own separate entry,
                # same "two PlannedOrders per symbol" pattern the
                # genuine-flip branch above already establishes.
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="cancel",
                        side=_pending_order_side(symbol_pending[0]),
                        volume=sum(o.volume for o in symbol_pending),
                        order_type="none", price=None, stop_loss=None,
                        pending_tickets_to_cancel=[o.ticket for o in symbol_pending],
                        reason=(
                            "Position is already at target size — cancelling a "
                            "stray/superseded pending order for the same symbol."
                        ),
                    )
                )
            continue

        if delta > 0:
            action = "open" if held_lots == 0 else "increase"
            plans.append(
                PlannedOrder(
                    symbol=symbol, action=action, side=target_side, volume=delta,
                    order_type="limit", price=clamped_price, stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Risking {pct:.1f}% of equity to this stop.{price_note}",
                )
            )
        else:
            reduce_volume = -delta
            tickets: list[tuple[int, float]] = []
            remaining = reduce_volume
            for p in held_positions:
                if remaining <= 1e-9:
                    break
                take = min(p.volume, remaining)
                tickets.append((p.ticket, take))
                remaining -= take

            action = "close" if target_lots == 0 else "reduce"
            plans.append(
                PlannedOrder(
                    symbol=symbol, action=action,
                    side=("sell" if held_side == "buy" else "buy"),
                    volume=reduce_volume,
                    order_type="market", price=None, stop_loss=None,
                    tickets_to_close=tickets,
                    reason=(
                        f"Risking {pct:.1f}% of equity to this stop, "
                        f"reducing from {held_lots:g} lots."
                    ),
                )
            )

    return plans


def pct_for_target_lots(
    symbol: str,
    target_lots: float,
    entry_price: float,
    stop_loss: float,
    account_equity: float,
    get_spec: Callable[[str], ContractSpec | None],
) -> float | None:
    """The exact mathematical inverse of compute_rebalance_plan's own
    risk-based sizing formula above (`target_lots =
    risk_dollars / (stop_distance * spec.trade_contract_size)`, where
    `risk_dollars = pct / 100 * account.equity`) — solved here for the
    `pct` that reproduces a given `target_lots` at a given stop distance.

    Exists for the Execution Clerk's tactical-defense check (added
    2026-08-27): tightening a position's stop while reusing its OLD `pct`
    would silently change its lot size, since the same risk-% divided by
    a SMALLER stop distance yields MORE lots — the opposite of "same
    size, tighter stop." Calling this with the position's own currently-
    held lot count (or a fraction of it, for a partial close) and the
    NEW stop distance recovers the `pct` that actually reproduces the
    intended lot count once compute_rebalance_plan sizes it.

    `entry_price` must be the same reference price compute_rebalance_plan
    will itself clamp-and-reuse as `clamped_price` for this symbol (its
    own suggested/held entry price, not a fresh live market price) —
    passing a different price here would solve for a `pct` that no
    longer reproduces `target_lots` once the real sizing pass runs.

    Returns None (never zero or a guess) when there's nothing sound to
    divide by: no usable spec, a non-positive stop distance, or non-
    positive equity — mirrors compute_rebalance_plan's own refusal to
    silently size against a broken input."""
    if target_lots <= 0:
        return 0.0
    if account_equity <= 0:
        return None
    spec = get_spec(symbol)
    if spec is None or spec.trade_contract_size <= 0:
        return None
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        return None
    risk_dollars = target_lots * stop_distance * spec.trade_contract_size
    return risk_dollars / account_equity * 100


def compute_aggregate_heat_pct(allocation: dict[str, AllocationEntry]) -> float:
    """Total % of equity at risk across a whole target allocation if
    every stop were hit at once — extracted from app.py's own
    _compute_aggregate_heat_pct (originally written for the capital-at-
    risk chart and the FTMO pre-execution heat gate) so the unattended
    hourly Copilot execution job can compute the identical number for
    its own merged target mix rather than a second, possibly-diverging
    calculation. Only meaningful for real leveraged MT5 instruments
    (pct_is_risk=True territory) — sums entry.pct directly, since that
    already equals each position's real % of equity at risk to its
    stop. A CASH entry, or any entry missing a price/stop_loss (nothing
    concrete enough yet to size or risk), contributes zero."""
    total = 0.0
    for symbol, entry in allocation.items():
        if symbol == "CASH" or entry.price is None or entry.stop_loss is None:
            continue
        total += entry.pct
    return total


def check_execution_safety_gates(
    is_demo: bool,
    allow_live_execution: bool,
    trading_permitted: bool,
    trading_blocked_reason: str = "",
    ftmo_heat_blocked: bool = False,
    ftmo_heat_blocked_reason: str = "",
) -> tuple[bool, str]:
    """The three real-money safety gates that must pass before ANY
    PlannedOrder from compute_rebalance_plan is actually sent to MT5 —
    extracted from app.py's "Confirm and Execute" dialog (where it first
    lived, inline) so this exact logic can never drift between two
    independently-maintained copies once a second caller (the unattended
    hourly Copilot execution job) needs it too. Pure — no Streamlit, no
    MT5 imports — callers own fetching the live inputs and rendering any
    resulting message.

    Same priority order as the original inline version, checked in this
    exact sequence (first failure wins, matching what a human reviewing
    the dialog would see first):
    1. `is_demo or allow_live_execution` — refuses to run at all against
       what looks like a real account unless explicitly overridden (see
       config.ALLOW_LIVE_EXECUTION's own docstring for why this defaults
       to blocked).
    2. `ftmo_heat_blocked` — FTMO-only: would this plan's aggregate heat
       breach the account's real remaining daily-loss headroom (see
       risk/ftmo_rules.py::would_breach_daily_loss_headroom). Always
       False for PMEX, which has no daily-loss rule to check against.
    3. `trading_permitted` — real MT5-level permission (the terminal's
       own AutoTrading toggle / account-level trade_allowed), distinct
       from the two policy checks above.

    Returns (True, "") if every gate passes; otherwise (False, reason)
    with the reason for whichever gate failed first — the same messages
    the dialog has always shown, just computed in one place now."""
    if not (is_demo or allow_live_execution):
        return False, (
            "Execution blocked: the connected account's server doesn't "
            "look like a demo account and ALLOW_LIVE_EXECUTION isn't "
            "set. Refusing to place real orders on what may be a live "
            "account."
        )
    if ftmo_heat_blocked:
        return False, ftmo_heat_blocked_reason
    if not trading_permitted:
        return False, f"Execution blocked: {trading_blocked_reason}"
    return True, ""
