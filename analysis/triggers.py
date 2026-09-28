"""Structured entry triggers for Pending Setups: a deterministic Python check instead of a local-LLM opinion.

Why (2026-09-25 plan W5): a resting limit does not always fill, and the free-text `trigger_condition` a Pending Setup
carries is judged by a small local model every few minutes - slow, noisy and unable to read a candle series exactly. A
structured trigger names a level and a kind; this module evaluates it on the COMPLETED M5 bars plus the live quote, with no
look-ahead (the still-forming bar is never in `bars`), and a fired trigger becomes a MARKET entry that still passes every
existing Clerk guard (entry-mode caps, stale re-anchor, cost gate, pre-close guard, kill switch).

Kinds (each measured or book-grounded, see Books/rule_ledger.md):
  range_break  - price traded through `level` in the direction of the trade (buy: above, sell: below) during the last
                 `within_bars` completed bars or right now, is not extended past it by more than
                 config.STOP_ENTRY_MAX_EXTENSION_ATR x ATR (O'Neil / Schwager Rule 8: do not chase) and has not fallen
                 back more than TRIGGER_FALLBACK_ATR x ATR under it (a failed break is not a break). A touch is enough: on
                 M5 a close filter measured worse (net +0.049 vs +0.077, ledger #10).
  reclaim      - a spring / upthrust: a completed bar pierced `level` against the trade (buy: low below, sell: high above)
                 and a LATER completed bar closed back on the trade's side of it; live price is still on that side and not
                 extended (Schwager: a close beyond the opposite extreme of a spike negates the failed signal).
  close_beyond - the latest completed bar closed beyond `level` in the trade's direction and price is not extended
                 (Murphy's close filter; slower than range_break, kept for setups that explicitly want confirmation).

Pure functions; nothing here talks to MT5.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config

KINDS = ("range_break", "reclaim", "close_beyond")
DEFAULT_WITHIN_BARS = {"range_break": 6, "reclaim": 8, "close_beyond": 1}
MAX_WITHIN_BARS = 24
TRIGGER_FALLBACK_ATR = 0.25


@dataclass
class TriggerResult:
    fired: bool
    reason: str


def normalise_trigger(raw: object) -> dict | None:
    """A clean {"kind", "level", "within_bars"} dict from whatever the model wrote, or None when it is not a valid structured
    trigger (the setup then keeps the legacy free-text path)."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind.strip().lower() not in KINDS:
        return None
    kind = kind.strip().lower()
    level = raw.get("level")
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not level > 0:
        return None
    within = raw.get("within_bars", DEFAULT_WITHIN_BARS[kind])
    if isinstance(within, bool) or not isinstance(within, (int, float)) or within < 1:
        within = DEFAULT_WITHIN_BARS[kind]
    return {"kind": kind, "level": float(level), "within_bars": int(min(within, MAX_WITHIN_BARS))}


def evaluate_trigger(
    trigger: dict, side: str, bars: pd.DataFrame | None, bid: float | None, ask: float | None, atr: float | None
) -> TriggerResult:
    """`bars`: COMPLETED M5 bars, oldest first, with High/Low/Close. Never fires on missing data."""
    clean = normalise_trigger(trigger)
    if clean is None:
        return TriggerResult(False, "not a valid structured trigger")
    if side not in ("buy", "sell"):
        return TriggerResult(False, f"unknown side {side!r}")
    if bars is None or not {"High", "Low", "Close"}.issubset(bars.columns) or len(bars) == 0:
        return TriggerResult(False, "no completed M5 bars to evaluate on")
    if not atr or atr <= 0 or bid is None or ask is None or bid <= 0 or ask <= 0:
        return TriggerResult(False, "no live quote / ATR to evaluate on")

    kind, level, within = clean["kind"], clean["level"], clean["within_bars"]
    recent = bars.tail(within)
    price = ask if side == "buy" else bid  # the side that would be paid
    sign = 1.0 if side == "buy" else -1.0
    beyond = sign * (price - level)  # > 0: through the level in the trade's direction
    # The fired entry is a market order the Clerk's guard measures from the trigger level against
    # MARKET_ENTRY_MAX_SLIPPAGE_ATR, so a trigger must not fire in a band that guard would then refuse (and downgrade to a
    # resting limit that may never fill - the very problem structured triggers exist to avoid).
    cap_atr = min(config.STOP_ENTRY_MAX_EXTENSION_ATR, config.MARKET_ENTRY_MAX_SLIPPAGE_ATR)
    cap = cap_atr * atr
    label = f"{kind} {level:.5g}"

    if beyond > cap:
        return TriggerResult(False, f"{label}: price is {beyond / atr:.2f} ATR past the level (cap {cap_atr:g}) - extended, not chasing")

    if kind == "range_break":
        touched = bool((recent["High"] >= level).any()) if side == "buy" else bool((recent["Low"] <= level).any())
        touched = touched or beyond >= 0
        if not touched:
            return TriggerResult(False, f"{label}: not yet traded through ({-beyond / atr:.2f} ATR away)")
        if beyond < -TRIGGER_FALLBACK_ATR * atr:
            return TriggerResult(False, f"{label}: broke but fell back {-beyond / atr:.2f} ATR inside - failed break")
        return TriggerResult(True, f"{label}: traded through within {within} bars, price {beyond / atr:+.2f} ATR from the level")

    if kind == "reclaim":
        closes = recent["Close"].to_numpy(dtype=float)
        pierced = (recent["Low"].to_numpy(dtype=float) < level) if side == "buy" else (recent["High"].to_numpy(dtype=float) > level)
        first = int(pierced.argmax()) if pierced.any() else None
        if first is None:
            return TriggerResult(False, f"{label}: no bar pierced the level in the last {within} bars")
        later = closes[first + 1:]
        reclaimed = bool(((sign * (later - level)) >= 0).any()) if len(later) else False
        if not reclaimed:
            return TriggerResult(False, f"{label}: pierced but no later close back on the trade's side")
        if beyond < 0:
            return TriggerResult(False, f"{label}: reclaimed but live price is back under the level")
        return TriggerResult(True, f"{label}: pierced then closed back inside, price {beyond / atr:+.2f} ATR from the level")

    last_close = float(recent["Close"].iloc[-1])
    if sign * (last_close - level) <= 0:
        return TriggerResult(False, f"{label}: last completed close {last_close:.5g} is not beyond the level")
    if beyond < 0:
        return TriggerResult(False, f"{label}: closed beyond but live price is back under the level")
    return TriggerResult(True, f"{label}: last completed M5 close {last_close:.5g} beyond the level")
