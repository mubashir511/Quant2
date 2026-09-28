"""Playbook selector: which ENTRY a with-the-trend trade should use, from measured numbers instead of habit.

Why (2026-09-25 plan W3): 205,758 aligned M5 opportunities (closed H4+D1 trend, 20 symbols, up to 120k bars each) showed
the entry ORDER matters more than the level picked - gross R per fill: a resting limit at the nearest reaction level
-0.03..-0.05, a market order +0.02..+0.03, a BREAKOUT STOP at the recent range extreme +0.04..+0.07 (best in every ADX
regime, most at ADX >= 30). With the stop at the opposite side of the range (clipped 1.5-4 ATR) the breakout earned
+0.086 gross and +0.077 net on the cheaper half of the symbols (fixed 2 ATR stop: +0.056 / +0.004). Volume expansion
(tick volume 1.5-3x its norm) and a rising ADX made it better; a close-beyond-trigger filter and time/soft stops did not.

Attribution, PAIRED at the same 34,407 breakout moments (deep M5, all 20 symbols, both time halves): net R per opportunity
market with a 2 ATR stop -0.196 (cheaper half -0.041) -> market with the structure stop -0.082 (+0.028) -> breakout stop
with the structure stop -0.035 (+0.058). So the STRUCTURE STOP is worth ~+0.07..0.11R (wider = fewer wick-outs and a lower
spread drag; it helps any entry) and waiting for the breakout order another ~+0.03..0.05R; a breakout stop with a fixed 2 ATR
stop is only +0.02..0.04R better than market. The advice below therefore always carries the structure stop.
Schwager (Rule 11: "use market orders rather than limit orders"), Bulkowski (pullbacks hurt performance) and O'Neil (buy
the pivot, do not chase past it) say the same in words; see Books/rule_ledger.md.

This module is pure and deterministic: it turns numbers already measured elsewhere (ATR, ADX, the 24-bar range, live
bid/ask) into ONE concrete proposal - order type, trigger, structure stop, 2R target, and which filters pass. It never
sends anything; Claude may adopt or override it with a reason, and analysis.entry_mode / the Clerk re-verify whatever is
finally chosen. Measured only for entries WITH the higher-timeframe trend (tier A of analysis.position_hunter); for other
tiers the advice is still computed but flagged as unmeasured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config

NAME_BREAKOUT = "TREND-BREAKOUT"


@dataclass
class RangeRead:
    """The last `bars` COMPLETED M5 bars' extremes (the still-forming bar is excluded, exactly like the study's
    shift(1)), and how busy the latest 12 bars are versus the 100 before them (tick volume)."""

    high: float
    low: float
    bars: int
    vol_ratio: float | None
    last_close: float


def range_read(ohlc: pd.DataFrame | None, bars: int | None = None) -> RangeRead | None:
    """None without High/Low/Close or fewer than `bars` + 1 completed bars. `ohlc` is oldest-first and its LAST row is
    treated as the forming bar."""
    bars = bars or config.PLAYBOOK_RANGE_BARS
    if ohlc is None or not {"High", "Low", "Close"}.issubset(ohlc.columns):
        return None
    done = ohlc.dropna(subset=["High", "Low", "Close"]).iloc[:-1]
    if len(done) < bars:
        return None
    window = done.tail(bars)
    vol_ratio = None
    if "Volume" in done.columns and len(done) >= 112:
        recent = float(done["Volume"].tail(12).mean())
        prior = float(done["Volume"].iloc[-112:-12].mean())
        vol_ratio = recent / prior if prior > 0 else None
    return RangeRead(
        high=float(window["High"].max()), low=float(window["Low"].min()), bars=bars,
        vol_ratio=vol_ratio, last_close=float(done["Close"].iloc[-1]),
    )


@dataclass
class PlaybookAdvice:
    name: str
    side: str  # "buy" / "sell"
    status: str  # ACTIVE | WAIT | EXTENDED | NOT_SUPPORTED
    order: str | None = None  # "stop" | "market" (the entry_mode to request) | None
    entry: float | None = None
    stop: float | None = None
    stop_atr: float | None = None
    target: float | None = None  # 2R from `entry`, advisory (the trade-zone / Level Map targets still apply)
    trigger_ahead_atr: float | None = None
    filters: list[tuple[str, bool, str]] = field(default_factory=list)  # (name, passed, text)
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        """One prompt-ready line of concrete numbers; every number is measured, none is a suggestion to invent."""
        verb = "buy" if self.side == "buy" else "sell"
        if self.status == "NOT_SUPPORTED":
            return f"{self.name}: not supported right now - " + "; ".join(self.notes)
        if self.status == "WAIT":
            return f"{self.name}: wait - " + "; ".join(self.notes)
        if self.status == "EXTENDED":
            return f"{self.name}: breakout already extended - " + "; ".join(self.notes)
        order_text = (
            f"{verb}-STOP at {self.entry:.5g} ({self.trigger_ahead_atr:.2f} M5 ATR from the market; the {config.PLAYBOOK_RANGE_BARS}-bar range extreme)"
            if self.order == "stop"
            else f"{verb.upper()} at the market ~{self.entry:.5g} (through the {config.PLAYBOOK_RANGE_BARS}-bar range extreme, not extended)"
        )
        filters = ", ".join(("OK " if ok else "WEAK ") + text for _name, ok, text in self.filters)
        return (
            f"{self.name}: {order_text}; structure stop {self.stop:.5g} ({self.stop_atr:.1f} M5 ATR, opposite side of the range); "
            f"2R target {self.target:.5g}; filters: {filters}. Measured on this account's data (aligned trades): breakout stop "
            f"{config.PLAYBOOK_MEASURED_GROSS:+.3f}R gross, {config.PLAYBOOK_MEASURED_NET_CHEAP:+.3f}R net on the cheaper half of symbols "
            "(negative net on costly ones) - request it with entry_mode " + (f"\"{self.order}\"" if self.order else "-") + "."
        )


def advise(
    side: str,
    bid: float,
    ask: float,
    atr: float | None,
    adx: float | None,
    adx_change_12: float | None,
    rng: RangeRead | None,
) -> PlaybookAdvice:
    """The breakout playbook for `side` from the live quote, the M5 ATR/ADX and the completed-bar range. Status:
    ACTIVE (order + numbers), WAIT (trigger farther than config.PLAYBOOK_MAX_TRIGGER_AHEAD_ATR), EXTENDED (already through
    the trigger by more than config.STOP_ENTRY_MAX_EXTENSION_ATR - O'Neil/Schwager Rule 8: do not chase), NOT_SUPPORTED
    (missing data, or ADX below config.PLAYBOOK_MIN_ADX where the breakout was not measured)."""
    advice = PlaybookAdvice(NAME_BREAKOUT, side, "NOT_SUPPORTED")
    if not atr or atr <= 0 or rng is None or not bid or not ask or ask < bid:
        advice.notes.append("missing M5 ATR, range or live quote")
        return advice
    if adx is None:
        advice.notes.append("no M5 ADX (needs High/Low history)")
        return advice
    if adx < config.PLAYBOOK_MIN_ADX:
        advice.notes.append(f"M5 ADX {adx:.0f} < {config.PLAYBOOK_MIN_ADX:g}: below the configured minimum trend strength")
        return advice

    is_buy = side == "buy"
    live = ask if is_buy else bid
    trigger = rng.high if is_buy else rng.low
    opposite = rng.low if is_buy else rng.high
    ahead = ((trigger - live) if is_buy else (live - trigger)) / atr  # > 0: trigger still ahead of the price
    if ahead > config.PLAYBOOK_MAX_TRIGGER_AHEAD_ATR:
        advice.status = "WAIT"
        advice.trigger_ahead_atr = ahead
        advice.notes.append(
            f"the {rng.bars}-bar range extreme {trigger:.5g} is {ahead:.1f} M5 ATR away (measured only within {config.PLAYBOOK_MAX_TRIGGER_AHEAD_ATR:g}); arm a range_break trigger instead of a resting order"
        )
        return advice
    if ahead < -config.STOP_ENTRY_MAX_EXTENSION_ATR:
        advice.status = "EXTENDED"
        advice.trigger_ahead_atr = ahead
        advice.notes.append(
            f"price is {-ahead:.2f} M5 ATR beyond the range extreme {trigger:.5g} (cap {config.STOP_ENTRY_MAX_EXTENSION_ATR:g}); wait for a secondary consolidation (Schwager Rule 8) or re-hunt"
        )
        return advice

    entry = trigger if ahead > 0 else live
    stop_distance = min(max(abs(entry - opposite), config.PLAYBOOK_STOP_MIN_ATR * atr), config.PLAYBOOK_STOP_MAX_ATR * atr)
    advice.status = "ACTIVE"
    advice.order = "stop" if ahead > 0 else "market"
    advice.entry = entry
    advice.trigger_ahead_atr = max(ahead, 0.0)
    advice.stop = entry - stop_distance if is_buy else entry + stop_distance
    advice.stop_atr = stop_distance / atr
    advice.target = entry + config.PLAYBOOK_TARGET_RR * stop_distance if is_buy else entry - config.PLAYBOOK_TARGET_RR * stop_distance
    strength_note = " (>40: strongest bucket)" if adx > 40 else " (<20: weakest bucket - still net positive on the cheaper half of symbols in the deep-history test)" if adx < 20 else ""
    advice.filters.append(("trend strength", adx >= 20, f"ADX {adx:.0f}{strength_note}"))
    if adx_change_12 is not None:
        rising = adx_change_12 > 0
        advice.filters.append(("ADX slope", rising, f"ADX {'rising' if rising else 'falling'} ({adx_change_12:+.1f} over 12 bars)"))
    if rng.vol_ratio is not None:
        ok = rng.vol_ratio >= config.PLAYBOOK_MIN_VOLUME_RATIO
        advice.filters.append(("volume", ok, f"tick volume {rng.vol_ratio:.1f}x its norm" + ("" if ok else f" (< {config.PLAYBOOK_MIN_VOLUME_RATIO:g}x: the weakest measured bucket)")))
    return advice
