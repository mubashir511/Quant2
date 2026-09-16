from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    now: datetime | None = None,
    max_pending_order_age_hours: float = 24.0,
    min_stop_distance_pct: float = 0.1,
    held_position_size_tolerance_pct: float = 25.0,
    is_pre_weekend: bool = False,
    weekend_tradable_symbols: set[str] | None = None,
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

    When entry-price clamping (above) actually moves the entry, the
    stop-loss is now RE-DERIVED to preserve the original suggestion's own
    risk DISTANCE from the new, clamped entry, rather than reusing the
    stale absolute stop price — real incident, 2026-09-10: a suggested
    XAUUSD entry/stop pair (an intentional $45/1.38x-ATR risk) got
    clamped closer and closer to the live market as gold fell all day
    while the absolute stop price stayed fixed, silently shrinking the
    real risk to $3.48, then $0.65 — the second fill was stopped out in
    5 seconds by ordinary noise, not a real adverse move. `min_stop_
    distance_pct` (default 0.1%, see config.MIN_STOP_DISTANCE_PCT's own
    comment) is the last-resort backstop for whatever still slips
    through this re-derivation (e.g. the original suggested distance was
    already this tight): a resulting stop distance below this % of the
    sizing price makes the position `infeasible` (or, for an already-
    held position, falls back to its own real current stop — same
    pattern the missing/wrong-side stop case already uses) instead of
    risking a near-guaranteed-instant stop-out.

    `held_position_size_tolerance_pct` (default 25%, see config.
    HELD_POSITION_SIZE_TOLERANCE_PCT's own comment) is a second, separate
    safety net — real incident, 2026-09-11: a held NVDA position's own
    revised stop was reported at pct=0.02%, a razor-thin 17% short of the
    0.0242% that exact distance actually needed to reach even ONE whole
    share. Every poll since then failed outright to `infeasible`,
    silently leaving the REAL position on its OLD stop/target for hours
    — a beneficial, clearly-intended stop-tightening blocked by an
    arithmetic miss, not a deliberate reduction. When an already-held
    position's own freshly-implied (pre-rounding) lot count sits within
    this % of what's ALREADY held, this now keeps the CURRENT held
    volume rather than failing the whole revision — the real fix is
    handing the mega session the exact rate to use instead of guessing
    (see ai/ftmo_suggest.py::format_ftmo_held_position_sizing_rates) —
    this tolerance is only the backstop for whatever still slips through.

    `pending_orders` (added 2026-08-23, direct user request — letting
    both Claude and the Execution Clerk re-assess an already-suggested
    position, not just propose fresh ones) makes this function aware of outstanding,
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

    `max_pending_order_age_hours` (default 24 — "at most one trading day",
    the same ceiling the AI's own entry-timing instructions use) enforces
    a real incident's fix: every entry is placed as a resting GTC limit
    order (see data/mt5_execution.py::open_position), which never expires
    on its own. If the mega session that proposed it doesn't run again
    (this account's mega session is manual/on-demand, not scheduled — it
    can go days between runs), nothing previously re-examined an
    unfilled order sitting untouched that whole time, because an
    unchanged target still reads as "nothing to change" regardless of
    how stale the price/technical read behind it has become. Confirmed
    live: a same-session-only silver entry from one evening's session
    sat as a resting limit order for 23+ hours (no next-day session ever
    ran) and finally filled the following day, in the middle of an
    unrelated flash-crash the original thesis never saw — a fill against
    conditions that had nothing to do with why the order was placed. Any
    still-unfilled order older than this ceiling now gets cancelled
    unconditionally, independent of whether the current `allocation`
    target still matches its terms — a fresh mega session is what's
    meant to re-propose it with re-validated levels, not a limit order
    quietly waiting out however many days it takes to get touched.

    `is_pre_weekend`/`weekend_tradable_symbols` (added 2026-09-12, real
    incident: an INTC pending limit order and a USDCHF one both survived
    into a weekend because the only existing cancel trigger — a fresh
    mega session superseding an old one — never fired in time, and once
    it tried, the market was already closed and the cancel silently
    failed) add a THIRD, independent condition, a sibling to the age-
    ceiling check above, not a replacement for it: when `is_pre_weekend`
    is True, every symbol with a resting pending order that is NOT in
    `weekend_tradable_symbols` (crypto, which genuinely trades through
    the weekend — everything else on this account does not) gets
    cancelled unconditionally, regardless of how close it is to filling
    or how old it is. This function stays pure (no MT5/network calls) —
    the caller (ai/clerk_execution.py) is responsible for deciding
    whether it's actually pre-weekend right now and for computing which
    symbols are weekend-tradable (via data/mt5_source.py::
    get_symbol_category), then passing both in as plain data. Cancelling
    doesn't free MT5 margin (this function's sizing model has none, per
    the REAL RISK note above) — it frees the resting order's own share
    of aggregate heat, letting a later mega session redeploy that
    risk-budget into a weekend-tradable instrument if it judges that
    worthwhile. Default False/None — every existing caller/test is
    unaffected.

    IMPORTANT invariant this function must never break: `all_symbols`
    below stays EXACTLY `set(by_symbol) | allocation keys` — pending-
    order symbols are never added to it. A symbol whose only footprint
    is a still-unfilled Pending-Setup order (fired by the Clerk
    mid-cycle, deliberately excluded from the allocation dict — see
    ai/clerk_execution.py::_build_carried_forward_allocation's own
    docstring) must stay completely invisible here, exactly as before;
    widening `all_symbols` to include it would give it `entry=None ->
    pct=0`, and the new "no longer wanted -> cancel" rule would then
    self-cancel that Pending Setup order the instant the Clerk places it.
    """
    plans: list[PlannedOrder] = []
    effective_now = now if now is not None else datetime.now(timezone.utc)

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
                # A LIMIT order can only ever sit on ONE side of the live
                # market: a buy limit must be <= the current ask, a sell
                # limit must be >= the current bid -- MT5 rejects the
                # other side outright with "Invalid price". Real incident
                # found live 2026-09-07: this clamp used to allow BOTH
                # directions symmetrically (ref_price * (1 +/- band) on
                # both sides regardless of target_side), so whenever the
                # AI's proposed entry had already been overtaken by live
                # price in the FAVORABLE direction (price ran toward the
                # target before execution), the "clamped" result still
                # landed on the wrong side of the spread -- three real
                # immediate-allocation orders (a buy priced above the
                # live ask, a sell priced below the live bid) failed to
                # place this exact way, silently losing the trade. The
                # favorable side is now capped hard at the live reference
                # price itself -- the tightest price still valid for that
                # order type -- while the unfavorable side keeps the
                # original band, still defending against a genuinely
                # stale/unreasonable proposed price there.
                if target_side == "buy":
                    lo, hi = ref_price * (1 - band), ref_price
                else:
                    lo, hi = ref_price, ref_price * (1 + band)
                clamped_price = min(max(proposed_price, lo), hi)
                if clamped_price != proposed_price:
                    ref_label = "bid" if target_side == "sell" else "ask"
                    if clamped_price == ref_price:
                        price_note = (
                            f" (clamped from {proposed_price:.4f} to the live "
                            f"{ref_label} {ref_price:.4f} -- a {target_side} limit "
                            "can't be priced on the wrong side of the current market)"
                        )
                    else:
                        price_note = (
                            f" (clamped from {proposed_price:.4f} to stay within "
                            f"{price_sanity_band_pct:.0f}% of the live {ref_label} {ref_price:.4f})"
                        )

            # sizing_price anchors every stop/target/risk-distance check
            # below; clamped_price itself stays reserved for what an
            # actual NEW order leg would be priced at (must be live-
            # market-valid). For an ALREADY-HELD position in the matching
            # direction, sizing_price is the position's own real, already-
            # filled price_open instead of the live-tracking clamped_price
            # — a real, severe incident found live 2026-09-08: once a
            # limit order fills, price naturally sits right around/through
            # that level afterward, so clamped_price's new hard cap (see
            # its own comment above) kept re-tracking the live ask/bid on
            # EVERY poll for an already-filled position, which for a
            # TIGHT-stop instrument (a few pips) swings stop_distance —
            # and therefore target_lots — by a large percentage from
            # ordinary tick noise alone. That fed a real EURUSD position
            # into a same-poll "increase"/"reduce" thrash: 9 separate
            # tickets opened and torn down via dozens of tiny partial
            # fills/closes inside 3.5 hours, each leg paying its own real
            # spread/commission for zero strategic reason. Anchoring to
            # the position's own fixed price_open makes stop_distance (and
            # everything sized from it) stable across polls again, exactly
            # like it already was before clamped_price's own live-tracking
            # fix was added.
            sizing_price = (
                held_positions[0].price_open
                if held_lots > 0 and held_side == target_side
                else clamped_price
            )

            stop_loss = entry.stop_loss if entry is not None else None
            # Real incident, 2026-09-10: a suggested XAUUSD entry (4415.00,
            # stop 4370.00 — an intentional 1.38x-ATR/$45 risk) sat
            # resting while gold fell all day. Each re-price clamped the
            # ENTRY down toward the live market (see the clamp above),
            # but the ABSOLUTE stop price above stayed fixed at 4370.00
            # — so the intended $45 risk silently shrank to $3.48, then
            # $0.65, entirely by accident of where the clamp happened to
            # land, never a deliberate choice by anyone. The second fill
            # was stopped out in 5 seconds — ordinary noise on an
            # instrument moving $4-7/minute that hour, not a real adverse
            # move. Re-derive the ORIGINAL intended risk DISTANCE (not
            # the stale absolute price) from `proposed_price`/`entry.
            # stop_loss` and re-apply that same distance from wherever
            # `sizing_price` actually ended up — this preserves the
            # mega session's own real ATR-multiple reasoning regardless
            # of how far the entry got clamped by the time it filled.
            # Only fires when sizing_price is genuinely the live-tracking
            # clamped_price (a fresh order) and actually differs from
            # what was proposed — an already-held position's own tactical
            # amend (sizing_price == entry.price == position.price_open,
            # see this function's own docstring) is untouched: its
            # stop_loss is already the deliberately-computed final value,
            # not something to rescale.
            if (
                sizing_price is not None
                and stop_loss is not None
                and proposed_price is not None
                and sizing_price != proposed_price
            ):
                original_distance = abs(proposed_price - stop_loss)
                rederived_stop = (
                    sizing_price - original_distance
                    if target_side == "buy"
                    else sizing_price + original_distance
                )
                if rederived_stop != stop_loss:
                    price_note += (
                        f" (stop re-derived from {stop_loss:.4f} to {rederived_stop:.4f} to "
                        f"preserve the original {original_distance:.4f} risk distance after "
                        "the entry above was clamped)"
                    )
                    stop_loss = rederived_stop

            if sizing_price is not None and stop_loss is not None:
                # Buy: stop must be BELOW entry. Sell: stop must be ABOVE
                # entry. Order matters — this must run before stop_distance
                # is ever computed below, since sizing trusts the sign is
                # already correct for target_side by that point.
                wrong_side = (
                    stop_loss >= sizing_price
                    if target_side == "buy"
                    else stop_loss <= sizing_price
                )
                if wrong_side:
                    stop_loss = None
                    price_note += " (suggested stop was on the wrong side of entry, dropped)"
                elif sizing_price != 0 and abs(sizing_price - stop_loss) / sizing_price * 100 < min_stop_distance_pct:
                    # Last-resort safety net, not a substitute for the
                    # re-derivation above (which already fixes the common
                    # case where clamping is WHY the distance shrank) —
                    # catches whatever still slips through, e.g. the
                    # ORIGINAL suggested distance already being this
                    # tight. Real incident this guards against directly:
                    # a stop just $0.65 from entry on gold, an instrument
                    # moving several dollars a minute — mathematically
                    # almost guaranteed to be clipped by ordinary noise
                    # within seconds, not a real, deliberate risk choice.
                    price_note += (
                        f" (stop distance {abs(sizing_price - stop_loss):.4f} is under the "
                        f"{min_stop_distance_pct:.2f}% minimum sane distance from entry "
                        f"{sizing_price:.4f} — dropped rather than risking a near-instant stop-out)"
                    )
                    stop_loss = None

            take_profit = entry.take_profit if entry is not None else None
            if sizing_price is not None and take_profit is not None:
                wrong_side_tp = (
                    take_profit <= sizing_price
                    if target_side == "buy"
                    else take_profit >= sizing_price
                )
                if wrong_side_tp:
                    take_profit = None
                    price_note += " (suggested take-profit was on the wrong side of entry, dropped)"

            if sizing_price is None or stop_loss is None:
                if held_lots > 0 and held_side == target_side:
                    # Already holding a real, filled position in the same
                    # direction — a stop coming out unusable (missing, or
                    # wrong-side/degenerate against sizing_price) must never
                    # make an ALREADY-HELD position "infeasible": no new
                    # capital is being risked here, and sizing_price/stop_loss
                    # are only used below to size `target_lots` for the
                    # hold-vs-amend comparison against the position's own
                    # real numbers, never to place a new order.
                    stop_loss = held_positions[0].sl
                    take_profit = take_profit if take_profit is not None else held_positions[0].tp
                else:
                    reason = (
                        "No live quote or suggested price available to size "
                        "this position by risk."
                        if sizing_price is None
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

            # Same reasoning as sizing_price, one level further, but two-
            # tiered: an ALREADY-HELD position that's already correctly
            # sized against its OWN current stop must not get resized
            # just because a fresh, stop-ONLY update comes in (e.g.
            # tactical defense tightening a stop to lock in gains) —
            # "same %-risk, new distance" mathematically implies a
            # different lot count, misreading a pure stop move as a real
            # reallocation and forcing a spurious increase/reduce instead
            # of the intended in-place amend. But a genuine, COORDINATED
            # mega-session revision (pct AND stop_loss both deliberately
            # updated together, e.g. to keep the same real lot count at a
            # tighter stop) must still size off the FRESH stop — that's
            # what the mega session actually intended, and the position's
            # OLD stop is exactly what's being superseded. The
            # distinguishing test: does the OLD stop, combined with
            # TODAY's pct, already explain the currently-held size? If
            # so, nothing about the real risk decision has changed this
            # poll — hold that basis. If not, the fresh combination is
            # the real, intentional one — trust it.
            sizing_stop_loss = stop_loss
            if held_lots > 0 and held_side == target_side and held_positions[0].sl is not None:
                held_stop_distance = abs(sizing_price - held_positions[0].sl)
                if held_stop_distance > 0:
                    old_basis_lots = _round_down_to_step(
                        (pct / 100 * account.equity) / (held_stop_distance * spec.trade_contract_size),
                        spec.volume_step,
                    )
                    if abs(old_basis_lots - held_lots) < spec.volume_step / 2:
                        sizing_stop_loss = held_positions[0].sl

            if sizing_stop_loss == sizing_price:
                # Zero stop-distance can't size at all (division by zero
                # below) — the held-position fallback just above can still
                # produce this if a real position was somehow opened with
                # no real stop distance; treat it the same as "no valid
                # stop" rather than crashing on stop_distance == 0.
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="infeasible", side=target_side, volume=0.0,
                        order_type="none", price=clamped_price, stop_loss=None,
                        reason="Zero stop distance — can't size a position by risk against no real stop.",
                    )
                )
                continue

            # abs() is safe here specifically because the wrong-side check
            # above already guarantees stop_loss's sign relative to
            # clamped_price is correct for target_side by this point.
            stop_distance = abs(sizing_price - sizing_stop_loss)
            risk_dollars = pct / 100 * account.equity
            raw_target_lots = risk_dollars / (stop_distance * spec.trade_contract_size)
            target_lots = _round_down_to_step(raw_target_lots, spec.volume_step)

            # Real incident, 2026-09-11: a held NVDA position's own
            # revised stop was reported at pct=0.02%, a razor-thin 17%
            # short of what that exact distance actually needed to reach
            # even ONE whole share — raw_target_lots came out to 0.83,
            # rounding DOWN to 0 and failing the whole revision to
            # "infeasible" every poll for hours, while the real position
            # sat on its old, unwanted stop/target the entire time with
            # no error visible anywhere. When this is genuinely an
            # already-held position (same side) and the RAW, pre-
            # rounding lot count is close to what's already held, this
            # is a rounding-precision miss, not a deliberate resize —
            # keep the real current volume and let the stop/target
            # change still apply, rather than silently blocking it.
            if held_lots > 0 and held_side == target_side and raw_target_lots > 0:
                if abs(raw_target_lots - held_lots) / held_lots * 100 <= held_position_size_tolerance_pct:
                    target_lots = held_lots

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

            setup_times = [o.time_setup for o in symbol_pending if o.time_setup is not None]
            if setup_times:
                oldest_age_hours = (effective_now - min(setup_times)).total_seconds() / 3600
                if oldest_age_hours > max_pending_order_age_hours:
                    plans.append(
                        PlannedOrder(
                            symbol=symbol, action="cancel",
                            side=_pending_order_side(symbol_pending[0]),
                            volume=sum(o.volume for o in symbol_pending),
                            order_type="none", price=None, stop_loss=None,
                            pending_tickets_to_cancel=all_tickets,
                            reason=(
                                f"Cancelling: this entry has rested unfilled for "
                                f"{oldest_age_hours:.1f}h, past the "
                                f"{max_pending_order_age_hours:g}h same-session "
                                "ceiling — the technical/momentum read it was "
                                "based on can no longer be assumed valid against "
                                "current conditions. A fresh mega session should "
                                "re-propose it with re-validated levels if the "
                                "idea still holds."
                            ),
                        )
                    )
                    continue

            if is_pre_weekend and symbol not in (weekend_tradable_symbols or set()):
                plans.append(
                    PlannedOrder(
                        symbol=symbol, action="cancel",
                        side=_pending_order_side(symbol_pending[0]),
                        volume=sum(o.volume for o in symbol_pending),
                        order_type="none", price=None, stop_loss=None,
                        pending_tickets_to_cancel=all_tickets,
                        reason=(
                            "Cancelling ahead of the weekend: this instrument's "
                            "market won't be open again until Monday, and an "
                            "unmanaged resting order over 2+ closed days risks a "
                            "bad gapped fill nobody can react to — freeing this "
                            "risk-budget for the mega session to redeploy, e.g. "
                            "into a weekend-tradable instrument, if it judges "
                            "that worthwhile."
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

        if delta > 0 and held_lots > 0:
            # Never auto-"increase" an already-held position — real,
            # severe incident found live 2026-09-08: this account's MT5
            # terminal runs in HEDGING mode, which cannot net an
            # "increase" into the existing ticket the way a netting
            # account would — every increase opens a genuinely NEW,
            # separate position. SymbolSettlement.order_ticket tracks
            # only ONE ticket per symbol, so the new ticket silently
            # overwrote the record pointing at the old one, orphaning it
            # from all further review (invalidation checks, tactical
            # defense) — exactly the "doesn't look at all the symbols
            # sitting in review" and "sl/tp not respected" reports this
            # closes: one EURUSD position quietly stopped being watched
            # at all the moment a second ticket appeared on the same
            # symbol. Leaving an already-held position alone here is a
            # deliberate, conservative policy choice: a genuine reason to
            # size up belongs to a fresh mega-session decision (or a
            # manual "Apply Suggestion" the user reviews first), not an
            # automatic action that fragments the account's own tracking
            # of what it holds.
            plans.append(
                PlannedOrder(
                    symbol=symbol, action="hold", side=held_side, volume=held_lots,
                    order_type="none", price=None, stop_loss=None,
                    reason=(
                        "Already holding this symbol in the target direction — "
                        "not auto-increasing (this account's hedging-mode "
                        "terminal would open a genuinely separate, untracked "
                        "position rather than grow this one). A real reason to "
                        "size up needs a fresh mega-session decision or a "
                        "manually-reviewed Apply Suggestion, not an automatic "
                        f"top-up.{price_note}"
                    ),
                )
            )
        elif delta > 0:
            plans.append(
                PlannedOrder(
                    symbol=symbol, action="open", side=target_side, volume=delta,
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
    hourly Clerk execution job can compute the identical number for
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
    hourly Clerk execution job) needs it too. Pure — no Streamlit, no
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
