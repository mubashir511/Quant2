"""Entry-mode resolution: how a drafted entry may be sent to the broker.

Why (2026-09-25 position-hunting review): every entry was a pullback LIMIT. Measured on this account's real
M5 bars, a limit N M5 ATRs below the market fills within 3 hours only 79% / 67% / 55% / 46% / 34% / 30% of
the time at 1.0 / 1.6 / 2.3 / 3.0 / 4.0 / 4.5 ATRs, the win rate of a FILLED limit is ~25% at every depth
(waiting buys no edge), and the TP level was reached WITHOUT a fill 24% / 34% / 63-70% of the time at
1.6 / 2.3 / 4-4.5 ATRs. Murphy's "breakout dilemma" and O'Neil's "buy the pivot, not the cheapest price"
say the same thing: a dynamic market often does not give the second chance a pullback plan needs.

Two more order types therefore exist, chosen by Claude per trade in its `entry_mode` field and RE-VERIFIED
here from a live quote (nothing in this module trusts Claude's arithmetic):

- "limit"  the existing behaviour (default).
- "stop"   a BUY STOP above / SELL STOP below the market at a breakout trigger. If the trigger has already
           been passed, the breakout is entered at the market ONLY while it is not extended
           (config.STOP_ENTRY_MAX_EXTENSION_ATR x M5 ATR past the trigger; O'Neil: do not chase), else the
           entry is rejected.
- "market" an immediate deal at the live ask/bid with SL/TP attached, only when the live price is within
           config.MARKET_ENTRY_MAX_SLIPPAGE_ATR x M5 ATR of the planned entry on the adverse side, the spread
           is a small share of the stop distance, and no market entry was already used for the symbol today.

Every check that fails DOWNGRADES to a plain limit (or rejects when the trade is dead) — the safe direction.
`config.NEW_ENTRY_KINDS_ENABLED` is the kill switch: off, every entry resolves to "limit".

Pure: no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import config

ENTRY_MODES = ("limit", "stop", "market")

# MEASURED 2026-09-25 on 20 real symbols x 6000 M5 bars: the share of placements whose price was touched within
# 3 hours, one direction, by distance in M5 ATRs. (1.0 -> 79%, 1.6 -> 67%, 2.0 -> 60%, 2.3 -> 55%, 3.0 -> 46%,
# 3.5 -> 39%, 4.0 -> 34%, 4.5 -> 30%, 5.0 -> 26%.)
_FILL_ODDS = ((0.0, 100.0), (1.0, 79.0), (1.6, 67.0), (2.0, 60.0), (2.3, 55.0), (3.0, 46.0), (3.5, 39.0), (4.0, 34.0), (4.5, 30.0), (5.0, 26.0))


def fill_odds_pct(distance_atr: float) -> float:
    """Measured probability (0-100) that a resting limit `distance_atr` M5 ATRs from the price is touched within
    3 hours — linear between the measured points, held at the last measured value beyond 5 ATRs."""
    d = max(distance_atr, 0.0)
    for (d0, p0), (d1, p1) in zip(_FILL_ODDS, _FILL_ODDS[1:]):
        if d <= d1:
            return p0 + (p1 - p0) * (d - d0) / (d1 - d0)
    return _FILL_ODDS[-1][1]


def normalise_entry_mode(value: object) -> str:
    """Anything that is not exactly one of ENTRY_MODES (case-insensitive) is the safe default, 'limit'."""
    if isinstance(value, str) and value.strip().lower() in ENTRY_MODES:
        return value.strip().lower()
    return "limit"


@dataclass
class EntryModeDecision:
    mode: str  # final mode: "limit" | "stop" | "market" | "reject"
    price: float | None  # the price to use for that mode (limit price / stop trigger / live market price)
    reason: str  # one plain-language sentence for the entry's reason / the log
    downgraded: bool = False  # True when the requested mode was not honoured
    max_deviation_price: float | None = None  # market only: the most the price may move between this check and the send


def resolve_entry_mode(
    *,
    requested: str,
    side: str,
    planned_price: float,
    stop_loss: float,
    take_profit: float | None,
    bid: float,
    ask: float,
    atr: float | None,
    min_stop_distance_price: float = 0.0,
    market_entry_already_used: bool = False,
) -> EntryModeDecision:
    """Resolve `requested` against the live quote. `atr` is the M5 ATR in price units; `min_stop_distance_
    price` is the broker's minimum distance for a pending stop order (price units)."""
    requested = normalise_entry_mode(requested)
    if requested == "limit":
        return EntryModeDecision("limit", planned_price, "")
    if not config.NEW_ENTRY_KINDS_ENABLED:
        return EntryModeDecision("limit", planned_price, f"entry_mode {requested!r} disabled (kill switch) - kept as a limit", True)
    if not atr or atr <= 0 or not bid or not ask or ask < bid:
        return EntryModeDecision("limit", planned_price, f"entry_mode {requested!r} needs a live quote and M5 ATR - kept as a limit", True)

    is_buy = side == "buy"
    live = ask if is_buy else bid
    spread = ask - bid
    stop_distance = (live - stop_loss) if is_buy else (stop_loss - live)
    if stop_distance <= 0:
        return EntryModeDecision("reject", None, "the live price is already through the stop - the setup is dead", True)
    if take_profit is not None and ((live >= take_profit) if is_buy else (live <= take_profit)):
        return EntryModeDecision("reject", None, "the live price is already at/through the target - the setup is dead", True)

    def _market(cap_atr: float, label: str) -> EntryModeDecision:
        adverse = (live - planned_price) if is_buy else (planned_price - live)  # > 0 = the market is worse than planned
        if adverse > cap_atr * atr:
            return EntryModeDecision(
                "limit", planned_price,
                f"{label}: the live price is {adverse / atr:.2f} M5 ATR worse than the plan (cap {cap_atr:g}) - not chasing, kept as a limit",
                True,
            )
        if spread > config.MARKET_ENTRY_MAX_SPREAD_STOP_FRACTION * stop_distance:
            return EntryModeDecision(
                "limit", planned_price,
                f"{label}: spread {spread:.5g} is {spread / stop_distance * 100:.0f}% of the {stop_distance:.5g} stop distance "
                f"(cap {config.MARKET_ENTRY_MAX_SPREAD_STOP_FRACTION * 100:.0f}%) - kept as a limit",
                True,
            )
        if market_entry_already_used:
            return EntryModeDecision("limit", planned_price, f"{label}: a market entry was already used for this symbol today - kept as a limit", True)
        return EntryModeDecision(
            "market", live,
            f"{label}: market entry at {live:.5g} ({adverse / atr:+.2f} M5 ATR vs the plan, spread {spread / stop_distance * 100:.0f}% of the stop)",
            False,
            max_deviation_price=config.MARKET_ENTRY_SEND_DEVIATION_ATR * atr,
        )

    if requested == "market":
        return _market(config.MARKET_ENTRY_MAX_SLIPPAGE_ATR, "entry_mode market")

    # requested == "stop": a breakout trigger.
    beyond = (planned_price - ask) if is_buy else (bid - planned_price)  # > 0 = the trigger is still ahead of price
    if beyond > 0:
        minimum = max(min_stop_distance_price, spread) * 1.02
        trigger = planned_price
        note = ""
        if beyond < minimum:
            trigger = (ask + minimum) if is_buy else (bid - minimum)
            note = f" (trigger nudged from {planned_price:.5g} to the broker's minimum distance)"
            beyond = minimum
        if beyond > config.STOP_ENTRY_MAX_DISTANCE_ATR * atr:
            return EntryModeDecision(
                "reject", None,
                f"entry_mode stop: the trigger is {beyond / atr:.1f} M5 ATR from the market (cap {config.STOP_ENTRY_MAX_DISTANCE_ATR:g}) - "
                "too far to be a same-session breakout entry",
                True,
            )
        return EntryModeDecision("stop", trigger, f"entry_mode stop: {'buy' if is_buy else 'sell'} stop at {trigger:.5g}, {beyond / atr:.2f} M5 ATR from the market{note}")
    extension = -beyond
    if extension > config.STOP_ENTRY_MAX_EXTENSION_ATR * atr:
        return EntryModeDecision(
            "reject", None,
            f"entry_mode stop: the breakout trigger {planned_price:.5g} was already passed and price is {extension / atr:.2f} M5 ATR beyond it "
            f"(cap {config.STOP_ENTRY_MAX_EXTENSION_ATR:g}) - extended, not chasing",
            True,
        )
    decision = _market(config.STOP_ENTRY_MAX_EXTENSION_ATR, "entry_mode stop (trigger already passed, not extended)")
    return decision
