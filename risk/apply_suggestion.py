from dataclasses import dataclass, field
from typing import Callable

from ai.portfolio_suggest import AllocationEntry
from data.mt5_source import AccountSummary, ContractSpec, MarketAsset, Position


@dataclass
class PlannedOrder:
    symbol: str
    action: str  # "open" | "increase" | "reduce" | "close" | "hold" | "infeasible"
    side: str  # "buy" or "sell"
    volume: float
    order_type: str  # "limit" | "market" | "none"
    price: float | None
    stop_loss: float | None
    take_profit: float | None = None
    tickets_to_close: list[tuple[int, float]] = field(default_factory=list)
    reason: str = ""


def _round_down_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    steps = int(value / step + 1e-9)
    return round(steps * step, 8)


def compute_rebalance_plan(
    positions: list[Position],
    account: AccountSummary,
    allocation: dict[str, AllocationEntry],
    get_spec: Callable[[str], ContractSpec | None],
    market_prices: dict[str, MarketAsset],
    price_sanity_band_pct: float = 5.0,
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
    """
    plans: list[PlannedOrder] = []

    by_symbol: dict[str, list[Position]] = {}
    for p in positions:
        by_symbol.setdefault(p.symbol, []).append(p)

    all_symbols = set(by_symbol) | {s for s in allocation if s.upper() != "CASH"}

    for symbol in sorted(all_symbols):
        symbol_positions = by_symbol.get(symbol, [])
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

        delta = target_lots - held_lots

        if abs(delta) < spec.volume_step / 2:
            plans.append(
                PlannedOrder(
                    symbol=symbol, action="hold",
                    side=held_side or target_side or "buy", volume=held_lots,
                    order_type="none", price=None, stop_loss=None,
                    reason="Already at target allocation.",
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
