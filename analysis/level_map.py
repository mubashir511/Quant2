"""Level Map: EVERY plausible reversal level near the price, nearest first, with measured odds - not one "strongest" level.

Why (2026-09-25 plan W4, the SOLUSD fix): on 2026-09-24 the engine listed only clustered multi-touch zones and capped them per
side, so the minor lows price had actually just turned up from (~116.3) were never candidates, and a deeper zone (115.85) was
chosen instead - price dropped, bounced from the earlier level and never reached it. The fix is a candidate list that also
carries UNBROKEN REACTION LOWS/HIGHS (fractal swings price turned from and has not since traded through) beside the zones,
range extremes, previous-day/session extremes and the Fibonacci retracements of the last leg.

What the books say and what our data says (Books/rule_ledger.md #17, #29): the books rank levels (thick multi-touch base, volume
node, polarity flip, 38-62% retracement, round number), but on 20 symbols x up to 200k M5 bars NO kind or feature of level
(touches, liquidity pool, round number, higher-timeframe confluence, recency, volume, Fibonacci, range extreme) filled-and-held
better than a RANDOM price at the same distance (-0.008..-0.040 vs random). What does drive the outcome is DISTANCE: fill odds fall
from ~93% within 1 ATR to ~62% at 3-5 ATR. So this map does not claim a strongest level. It lists the candidates nearest-first
with their measured fill odds, and says plainly which ones are actual reaction points (something the price did), leaving the
choice - and the duty to explain why a farther candidate beat a nearer one - to the caller.

Stop buffer: levels that later held were still pierced by a median 0.5, P75 1.3 and P90 2.7 ATR (Schwager ch. 4: stops cluster just
beyond a range and a small poke takes them), so a stop must sit BEYOND the level by a buffer, not at it.

Pure and deterministic; the caller passes the M5 bars it already fetched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config
from analysis.chart_structure import find_swing_points
from analysis.entry_mode import fill_odds_pct

REACTION_SWING_WINDOW = 3  # bars each side of a fractal reaction point on M5
MERGE_TOLERANCE_ATR = 0.15  # candidates closer than this are ONE level with several reasons


@dataclass
class LevelCandidate:
    price: float
    distance_atr: float  # always >= 0: how far the level is from the price, on the side a resting order would be
    fill_odds_pct: float
    kinds: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    is_reaction: bool = False  # something the price actually did here (an unbroken reaction low/high)
    too_close: bool = False  # inside the noise/spread band: not a real resting-order location


@dataclass
class LevelMap:
    side: str  # "buy" -> supports below the price, "sell" -> resistances above it
    price: float
    atr: float
    candidates: list[LevelCandidate]

    def text(self) -> str:
        role = "supports below" if self.side == "buy" else "resistances above"
        head = (
            f"  LEVEL MAP ({self.side}, {role} {self.price:.5g}, M5 ATR {self.atr:.5g}) - nearest first. Measured: no level kind "
            f"outperforms a random price at the same distance (Books/rule_ledger.md #29); distance sets the fill odds. Put the stop "
            f"BEYOND the level (pokes: median {config.LEVEL_PENETRATION_MEDIAN_ATR:g}, P75 {config.LEVEL_PENETRATION_P75_ATR:g}, "
            f"P90 {config.LEVEL_PENETRATION_P90_ATR:g} ATR). If you choose a level that is NOT the nearest real candidate, say why:"
        )
        if not self.candidates:
            return head + "\n    (no candidate within range)"
        rows = []
        for i, c in enumerate(self.candidates, 1):
            tag = "REACTION POINT " if c.is_reaction else ""
            close = " [inside the noise band - not a resting-order spot]" if c.too_close else ""
            rows.append(
                f"    {i}. {c.price:.5g} ({c.distance_atr:.1f} ATR away, fill odds ~{c.fill_odds_pct:.0f}%) "
                f"{tag}{' + '.join(c.kinds)}" + (f" - {'; '.join(c.notes)}" if c.notes else "") + close
            )
        return head + "\n" + "\n".join(rows)


def _add(raw: list[tuple[float, str, str | None, bool]], price: float | None, kind: str, note: str | None = None, reaction: bool = False) -> None:
    if price is not None and price > 0:
        raw.append((float(price), kind, note, reaction))


def _reaction_points(bars: pd.DataFrame, atr: float, side: str, min_bounce_atr: float) -> list[tuple[float, str, str | None, bool]]:
    """Unbroken fractal lows (buy) / highs (sell) the price turned from by at least `min_bounce_atr` ATR."""
    if bars is None or not {"High", "Low"}.issubset(bars.columns):
        return []
    swing_highs, swing_lows = find_swing_points(bars.reset_index(drop=True), window=REACTION_SWING_WINDOW)
    highs = bars["High"].to_numpy(dtype=float)
    lows = bars["Low"].to_numpy(dtype=float)
    out: list[tuple[float, str, str | None, bool]] = []
    n = len(bars)
    if side == "buy":
        for sp in swing_lows:
            after = slice(sp.index + 1, n)
            if lows[after].size == 0 or lows[after].min() < sp.price:
                continue  # traded through since: broken
            bounce = (highs[sp.index + 1: min(n, sp.index + 13)].max() - sp.price) / atr if sp.index + 1 < n else 0.0
            if bounce >= min_bounce_atr:
                out.append((sp.price, "reaction low", f"price turned up {bounce:.1f} ATR from it {n - 1 - sp.index} bars ago, not undercut since", True))
    else:
        for sp in swing_highs:
            after = slice(sp.index + 1, n)
            if highs[after].size == 0 or highs[after].max() > sp.price:
                continue
            bounce = (sp.price - lows[sp.index + 1: min(n, sp.index + 13)].min()) / atr if sp.index + 1 < n else 0.0
            if bounce >= min_bounce_atr:
                out.append((sp.price, "reaction high", f"price turned down {bounce:.1f} ATR from it {n - 1 - sp.index} bars ago, not exceeded since", True))
    return out


def build_level_map(
    side: str,
    price: float | None,
    atr: float | None,
    bars: pd.DataFrame | None = None,
    sr_levels=None,
    fibonacci=None,
    intraday_levels=None,
    range_read=None,
) -> LevelMap | None:
    """The Level Map for one side. None when there is no price/ATR to measure distance in. `bars`: the M5 bars (oldest first,
    the last may be the forming one - its low/high still count as traded prices) used to find reaction points."""
    if side not in ("buy", "sell") or not price or not atr or atr <= 0:
        return None
    raw: list[tuple[float, str, str | None, bool]] = []

    if sr_levels is not None:
        for lv in (sr_levels.support_levels if side == "buy" else sr_levels.resistance_levels):
            notes = [f"{lv.touches} touch" + ("" if lv.touches == 1 else "es")]
            if getattr(lv, "is_liquidity_pool", False):
                notes.append("equal-lows pool" if side == "buy" else "equal-highs pool")
            if getattr(lv, "low", None) and getattr(lv, "high", None) and lv.high > lv.low:
                notes.append(f"band {lv.low:.5g}-{lv.high:.5g}")
            _add(raw, lv.price, "S/R zone", ", ".join(notes))
    raw.extend(_reaction_points(bars, atr, side, config.LEVEL_MAP_MIN_BOUNCE_ATR))
    if range_read is not None:
        _add(raw, range_read.low if side == "buy" else range_read.high, f"{range_read.bars}-bar range {'low' if side == 'buy' else 'high'}")
    if intraday_levels is not None:
        _add(raw, intraday_levels.prev_day_low if side == "buy" else intraday_levels.prev_day_high, "previous-day " + ("low" if side == "buy" else "high"))
        _add(raw, intraday_levels.day_low if side == "buy" else intraday_levels.day_high, "session " + ("low" if side == "buy" else "high"))
    if fibonacci is not None:
        for name, level_price in (fibonacci.levels or {}).items():
            if name in ("38.2%", "50.0%", "61.8%"):
                _add(raw, level_price, f"Fibonacci {name}")

    on_side = [r for r in raw if (r[0] < price if side == "buy" else r[0] > price)]
    on_side = [r for r in on_side if abs(r[0] - price) / atr <= config.LEVEL_MAP_MAX_DISTANCE_ATR]
    on_side.sort(key=lambda r: abs(r[0] - price))

    merged: list[LevelCandidate] = []
    for lvl_price, kind, note, reaction in on_side:
        distance = abs(lvl_price - price) / atr
        target = merged[-1] if merged and abs(lvl_price - merged[-1].price) / atr <= MERGE_TOLERANCE_ATR else None
        if target is None:
            merged.append(LevelCandidate(
                price=lvl_price, distance_atr=distance, fill_odds_pct=fill_odds_pct(distance),
                kinds=[kind], notes=[note] if note else [], is_reaction=reaction,
                too_close=distance < config.LEVEL_MAP_MIN_DISTANCE_ATR,
            ))
        else:
            if kind not in target.kinds:
                target.kinds.append(kind)
            if note:
                target.notes.append(note)
            target.is_reaction = target.is_reaction or reaction
    return LevelMap(side=side, price=float(price), atr=float(atr), candidates=merged[: config.LEVEL_MAP_MAX_CANDIDATES])
