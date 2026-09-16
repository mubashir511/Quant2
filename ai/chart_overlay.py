"""Exports a plain-text overlay file that mql5/Quant2ChartOverlay.mq5
(an Expert Advisor running INSIDE the MT5 terminal) reads and draws
directly onto a chart as horizontal levels, shaded zones, Fibonacci
retracements, trendlines, and short text notes — instead of the long
prompt-style reasoning this account's decisions are normally expressed
in.

Direct user request 2026-09-04: "I usually not read long prompts and
reasoning, I am comfortable to read lines, levels, regions and some
notes on the mt5 terminal screen." The Python MetaTrader5 package has
no chart-drawing API at all (confirmed live: no ObjectCreate/Chart*
functions exist on it) — only a native MQL5 program running inside the
terminal can draw chart objects, which is why this is a two-language
feature: this module computes and writes WHAT to draw, in a tiny
line-oriented text format the EA can parse with plain StringSplit (no
MQL5 JSON library needed); the EA owns HOW to draw it.

v2 (2026-09-04, direct user feedback against the v1 drawing): v1 drew a
solid-filled zone (no MT5 rectangle-fill transparency exists, so a
solid color hid the candles under it), two trendlines anchored via a
chart-timeframe-dependent bar offset while the underlying price math
assumed the ANALYSIS timeframe's own bar spacing (H4/H1) — a real unit
mismatch that made them look "completely random" on an M5 chart — and
tooltip-only labels that never actually showed as visible text. v2:
zones are hollow (border-only, so nothing is ever hidden), every
level/line ships enough real-time information (seconds, not bars) for
the EA to anchor correctly regardless of the chart's own period, and
carries a real visible label instead of a hover-only tooltip. Also adds
GOLDENZONE (the 38.2-61.8% pullback retracement band this account's own
setup classifier already uses to detect a pullback setup — see
analysis/setup_classifier.py's FIB_GOLDEN_ZONE, reused here rather than
inventing a second number) and switches the Fibonacci ladder to MT5's
own native OBJ_FIBO tool instead of a single hand-picked "nearest
level" line.

v5 (2026-09-04, direct user feedback again): dropped the standalone
GOLDENZONE line — the EA now fills and labels the band between every
pair of adjacent Fibonacci ratios itself (including naming the
38.2-61.8% golden zone specifically), so a separate GOLDENZONE
rectangle only ever duplicated two of those bands' own price range with
a second, differently-labeled tint stacked on top — exactly the
"multiple hatched regions... don't know what represents what" confusion
reported live. Also dropped H4's own independent Fibonacci ladder
(kept H4's S/R zones/trendline) since showing H4's and H1's retracement
at once looked like "2 fibonacci level systems," not two independent,
additive reads the way separate S/R levels are.

Added 2026-09-04 (same day, later request): full-height vertical
regime-transition markers ("top to bottom straight lines to indicate
accumulation, distribution regions, trending and sideways regimes").
Unlike everything else in this file, this needed genuinely new analysis
first — TechnicalStats.market_regime (analysis/technical.py) was only
ever a single snapshot of the LATEST window, with nothing recording
WHEN the current regime began or what came before it. See
analysis.technical.compute_regime_segments (added alongside this) for
the rolling-window walk that builds real segment history from the
exact same, already-calibrated classification, plus the Wyckoff-style
accumulation/distribution inference (a sideways range immediately
following a down-leg vs. an up-leg) neither market_regime nor
chart_structure.py had a value for on their own.

Pure cosmetic/debugging aid: this module itself makes no MT5/analysis
calls of its own — it only turns already-computed inputs into text and
writes a file, and a write failure here must never interrupt whatever
real logic called it (see write_chart_overlay's own try/except
contract). Called once per Clerk poll, per symbol already being checked
that poll (tactical/watched/pending-setup candidates). Every input here
comes from that poll's own already-cached _cached_technical_context
call EXCEPT h1_regime_segments/h1_bar_count — regime-segment HISTORY
needs the raw H1 closes, which TechnicalStats/ChartStructureSnapshot
never retain, so the caller (ai/clerk_execution.py's
_export_chart_overlay) does fetch H1 price history again for that one
value; see its own docstring for why that's the sole exception.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from analysis.chart_structure import ChartStructureSnapshot
from analysis.technical import RegimeSegment, TechnicalStats
from data.mt5_source import PendingOrder, Position

logger = logging.getLogger(__name__)

OVERLAY_FILENAME = "quant2_overlay.txt"
# Same 1.5-2x band this account's own mega-session prompt already treats
# as the ceiling for a realistic same-session reward distance (see
# ai/ftmo_suggest.py's "NUMERIC self-check" instruction) — the ZONE line
# below draws exactly the range a target is supposed to stay inside,
# so a target sitting outside the shaded band is visually obvious
# without reading a single word of reasoning.
_ATR_ZONE_MULTIPLE = 1.75
_MAX_SR_LEVELS_PER_SIDE = 3
_MAX_NOTE_CHARS = 180
# Real elapsed seconds per bar for each analysis timeframe this account
# uses — needed to convert a "N bars back" concept (computed against the
# ANALYSIS timeframe: H4 bars for h4_structure, H1 bars for h1_structure)
# into a real time offset the EA can anchor correctly regardless of
# what period the chart it's drawn on is actually set to (the v1 bug:
# anchoring by chart bar-offset silently assumed the chart's own period
# matched the analysis timeframe, which produced nonsense trendlines on
# an M5 chart built from H4/H1-scale price math).
_TIMEFRAME_SECONDS = {"H1": 3600, "H4": 14400}
_TREND_LOOKBACK_BARS = 20

# Human labels per raw compute_regime_segments regime value, and the
# VISUAL category each collapses into — 5 categories now (uptrend/
# downtrend split apart, direct user request 2026-09-05: "use different
# color for downtrend and uptrend markets" — an earlier round had merged
# both into one "trending" swatch, direction only in the label text,
# which the user has now asked to undo), not the underlying 7 raw
# labels. choppy_up/choppy_down still fold into their own direction's
# category (a choppy up-leg is still visually "uptrend", just noisier)
# rather than getting a 6th/7th color of their own.
_REGIME_LABELS = {
    "trending_up": "Uptrend begins",
    "trending_down": "Downtrend begins",
    "choppy_up": "Choppy up-leg begins",
    "choppy_down": "Choppy down-leg begins",
    "sideways": "Sideways range begins",
    "accumulation": "Accumulation begins - range after a downtrend, watch for a reversal up",
    "distribution": "Distribution begins - range after an uptrend, watch for a reversal down",
}
_REGIME_CATEGORY = {
    "trending_up": "uptrend",
    "trending_down": "downtrend",
    "choppy_up": "uptrend",
    "choppy_down": "downtrend",
    "sideways": "sideways",
    "accumulation": "accumulation",
    "distribution": "distribution",
}
# Only the most recent transitions are actionable "right now" — a long
# H1 history can hold many old regime changes that would just clutter
# the chart with vertical lines far off the left edge.
_MAX_REGIME_MARKERS = 8


def _clean(text: str) -> str:
    """Strips characters that would break the pipe-delimited line format
    or span multiple lines in the file — a note is display text, never
    parsed further, so this just needs to stay a single, well-formed
    line, not preserve exact original formatting."""
    return " ".join(text.replace("|", "/").split())


@dataclass
class OverlaySymbolInputs:
    """Everything build_overlay_lines needs for one symbol, gathered by
    the caller from data it already has this poll — deliberately not a
    fetch itself, so this module never adds new MT5/analysis cost on top
    of whatever the caller already computed for its own real decision."""

    symbol: str
    h4_structure: ChartStructureSnapshot | None = None
    h1_structure: ChartStructureSnapshot | None = None
    h1_stats: TechnicalStats | None = None
    position: Position | None = None
    pending_order: PendingOrder | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    side: str | None = None
    note: str | None = None
    h1_regime_segments: list[RegimeSegment] | None = None
    h1_bar_count: int | None = None


def _sr_level_lines(symbol: str, structure: ChartStructureSnapshot | None, timeframe: str) -> list[str]:
    if structure is None or structure.sr_levels is None:
        return []
    # Real gap found on independent audit, fixed 2026-09-10: this used to
    # hardcode the flat SR_CLUSTER_TOLERANCE_PCT module constant as every
    # level's own exported zone-width, regardless of instrument. Since
    # analysis.chart_structure.compute_chart_structure now scales this
    # tolerance by the instrument's own ATR% for a fast mover (confirmed
    # live: 0.443% on NVDA, 0.9% on INTC, both several times the flat
    # 0.3% default), the terminal overlay was silently drawing a
    # NARROWER zone than what Python actually used to decide these were
    # "the same level" — a real mismatch between the computation and
    # what actually renders on the user's own MT5 chart (see mql5/
    # Quant2ChartOverlay.mq5's own DrawLevelZoneFill, which uses this
    # exact value numerically, not just cosmetically). sr_levels.
    # tolerance_pct is the REAL value compute_sr_levels actually used.
    tolerance_pct = structure.sr_levels.tolerance_pct
    lines = []
    for level in structure.sr_levels.resistance_levels[:_MAX_SR_LEVELS_PER_SIDE]:
        pool = " [LIQUIDITY POOL]" if level.is_liquidity_pool else ""
        lines.append(
            f"LEVEL|{symbol}|{level.price:g}|resistance|{timeframe} Resistance ({level.touches}x){pool} "
            f"- sellers defended here before, watch for rejection|{tolerance_pct:g}"
        )
    for level in structure.sr_levels.support_levels[:_MAX_SR_LEVELS_PER_SIDE]:
        pool = " [LIQUIDITY POOL]" if level.is_liquidity_pool else ""
        lines.append(
            f"LEVEL|{symbol}|{level.price:g}|support|{timeframe} Support ({level.touches}x){pool} "
            f"- buyers defended here before, watch for bounce|{tolerance_pct:g}"
        )
    return lines


def _fibo_lines(symbol: str, structure: ChartStructureSnapshot | None, timeframe: str) -> list[str]:
    """Native OBJ_FIBO anchor data (the EA draws MT5's own built-in
    Fibonacci tool, not a hand-picked single level). The EA fills the
    band between every pair of adjacent ratios itself, each with its
    own label (including calling out the 38.2-61.8% golden zone this
    account's own setup classifier treats as the real pullback-entry
    region) — no separate GOLDENZONE line is needed here; an earlier
    version sent one, but it drew a rectangle covering the exact same
    price range as two of the EA's own fib bands, stacking two
    overlapping, ambiguously-labeled tints on top of each other, which
    is exactly what direct user feedback flagged as confusing "multiple
    hatched regions... don't know what represents what.\""""
    if structure is None or structure.fibonacci is None:
        return []
    fib = structure.fibonacci
    bar_seconds = _TIMEFRAME_SECONDS.get(timeframe, 3600)
    high_seconds_ago = fib.swing_high_bars_ago * bar_seconds
    low_seconds_ago = fib.swing_low_bars_ago * bar_seconds
    return [
        f"FIBO|{symbol}|{fib.swing_high:g}|{fib.swing_low:g}|{int(fib.high_is_more_recent)}"
        f"|{high_seconds_ago}|{low_seconds_ago}|{timeframe} retracement from the last swing "
        f"- watch these ratios for a bounce or a break"
    ]


def _trendline_lines(symbol: str, structure: ChartStructureSnapshot | None, timeframe: str) -> list[str]:
    if structure is None or structure.trendlines is None:
        return []
    bar_seconds = _TIMEFRAME_SECONDS.get(timeframe, 3600)
    seconds_back = _TREND_LOOKBACK_BARS * bar_seconds
    lines = []
    for kind, trendline in (
        ("resistance", structure.trendlines.resistance_trendline),
        ("support", structure.trendlines.support_trendline),
    ):
        if trendline is None:
            continue
        price_now = trendline.current_price_on_line
        price_back = price_now - trendline.slope_per_bar * _TREND_LOOKBACK_BARS
        watch_for = "a break above" if kind == "resistance" else "a break below"
        lines.append(
            f"TREND|{symbol}|{price_back:g}|{price_now:g}|{seconds_back}|{kind}"
            f"|{timeframe} {kind} trendline ({trendline.direction}) - watching for {watch_for} or a bounce off it"
        )
    return lines


def _pattern_notes(symbol: str, structure: ChartStructureSnapshot | None, timeframe: str) -> list[str]:
    if structure is None or not structure.patterns:
        return []
    return [f"NOTE|{symbol}|{timeframe} pattern: {_clean(p.name)}" for p in structure.patterns]


def _regime_vline_lines(
    symbol: str, segments: list[RegimeSegment] | None, bar_count: int | None
) -> list[str]:
    """Full-height vertical markers at each real H1 regime TRANSITION —
    see analysis.technical.compute_regime_segments for how these are
    computed. `bar_count` is the length of the same H1 price series the
    segments were computed from — needed to turn each segment's own
    bar-index-in-that-series into a real elapsed-seconds-ago figure
    (same reasoning as _fibo_lines/_trendline_lines: the EA anchors by
    real time, not by the chart's own bar offset, since the chart period
    can differ from the H1 analysis timeframe)."""
    if not segments or not bar_count:
        return []
    lines = []
    bar_seconds = _TIMEFRAME_SECONDS["H1"]
    for seg in segments[-_MAX_REGIME_MARKERS:]:
        label = _REGIME_LABELS.get(seg.regime)
        category = _REGIME_CATEGORY.get(seg.regime)
        if label is None or category is None:
            continue
        bars_ago = (bar_count - 1) - seg.start_index
        if bars_ago < 0:
            continue
        seconds_ago = bars_ago * bar_seconds
        lines.append(f"VLINE|{symbol}|{seconds_ago}|{category}|{label}")
    return lines


def _atr_zone_line(symbol: str, h1_stats: TechnicalStats | None) -> list[str]:
    if h1_stats is None or h1_stats.atr is None or h1_stats.last_price is None:
        return []
    band = h1_stats.atr * _ATR_ZONE_MULTIPLE
    low = h1_stats.last_price - band
    high = h1_stats.last_price + band
    return [
        f"ZONE|{symbol}|{low:g}|{high:g}|Realistic move zone ({_ATR_ZONE_MULTIPLE:g}x H1 ATR) "
        f"- a target outside this band is stretching for a move this session probably won't reach"
    ]


_MAX_LOG_CHARS = 400


def _strategy_briefing(inputs: OverlaySymbolInputs) -> str:
    """One real, decision-oriented sentence for the EA to Print() into
    the MT5 Experts log — direct user request 2026-09-04: "give more
    strategic updates like a commander giving instructions... not just
    post health status logs." Built entirely from data this module
    already has (the account's own real reason/thesis text, side,
    entry/stop/target) — not a fabricated narration, the same facts the
    on-chart NOTE shows, just given the Experts log's own longer line
    budget instead of an on-chart label's tight space."""
    status = None
    if inputs.position is not None:
        p = inputs.position
        status = f"HOLDING {p.side.upper()} @ {p.price_open:g}"
        if p.sl is not None:
            status += f", stop {p.sl:g}"
        if p.tp is not None:
            status += f", target {p.tp:g}"
    elif inputs.pending_order is not None:
        o = inputs.pending_order
        side = "BUY" if o.order_type.startswith("buy") else "SELL"
        status = f"WATCHING resting {side} limit @ {o.price_open:g}"
    elif inputs.entry_price is not None:
        side_label = (inputs.side or "buy").upper()
        status = f"WATCHING for {side_label} @ {inputs.entry_price:g} (not yet placed)"

    parts = [p for p in (status, inputs.note) if p]
    if not parts:
        return ""
    return _clean(" -- ".join(parts))[:_MAX_LOG_CHARS]


def build_overlay_lines(inputs: OverlaySymbolInputs) -> list[str]:
    """Pure function: turns already-computed analysis + MT5 state for
    ONE symbol into this module's line-oriented overlay format. No I/O,
    no MT5 calls — trivially testable, and the actual file write
    (write_chart_overlay below) is a thin, separately-testable wrapper
    around calling this once per symbol and writing the result."""
    symbol = inputs.symbol
    lines: list[str] = []

    lines += _sr_level_lines(symbol, inputs.h4_structure, "H4")
    lines += _sr_level_lines(symbol, inputs.h1_structure, "H1")
    # H1 only, deliberately — direct user feedback 2026-09-04: showing
    # BOTH H4's and H1's own independent Fibonacci ladder at once read
    # as "2 fibonacci level systems" and was confusing, not additive
    # the way two independent S/R levels are. H1 is kept as the one
    # shown since it's the timeframe closer to current price action on
    # an intraday chart; H4's own S/R zones/trendline are unaffected.
    lines += _fibo_lines(symbol, inputs.h1_structure, "H1")
    lines += _trendline_lines(symbol, inputs.h4_structure, "H4")
    lines += _trendline_lines(symbol, inputs.h1_structure, "H1")
    lines += _pattern_notes(symbol, inputs.h4_structure, "H4")
    lines += _pattern_notes(symbol, inputs.h1_structure, "H1")
    lines += _atr_zone_line(symbol, inputs.h1_stats)
    lines += _regime_vline_lines(symbol, inputs.h1_regime_segments, inputs.h1_bar_count)

    if inputs.position is not None:
        p = inputs.position
        if p.sl is not None:
            lines.append(f"STOP|{symbol}|{p.sl:g}")
        if p.tp is not None:
            lines.append(f"TARGET|{symbol}|{p.tp:g}")
        lines.append(f"ENTRY|{symbol}|{p.price_open:g}|{p.side}")
    elif inputs.pending_order is not None:
        o = inputs.pending_order
        side = "buy" if o.order_type.startswith("buy") else "sell"
        if o.sl is not None:
            lines.append(f"STOP|{symbol}|{o.sl:g}")
        if o.tp is not None:
            lines.append(f"TARGET|{symbol}|{o.tp:g}")
        lines.append(f"ENTRY|{symbol}|{o.price_open:g}|{side} (pending)")
    elif inputs.entry_price is not None:
        # A Pending Setup with no real MT5 order placed yet — still
        # worth drawing so the level is visible before it ever triggers.
        if inputs.stop_loss is not None:
            lines.append(f"STOP|{symbol}|{inputs.stop_loss:g}")
        if inputs.take_profit is not None:
            lines.append(f"TARGET|{symbol}|{inputs.take_profit:g}")
        side_label = inputs.side or "buy"
        lines.append(f"ENTRY|{symbol}|{inputs.entry_price:g}|{side_label} (watching)")

    if inputs.note:
        lines.append(f"NOTE|{symbol}|{_clean(inputs.note)[:_MAX_NOTE_CHARS]}")

    briefing = _strategy_briefing(inputs)
    if briefing:
        lines.append(f"LOG|{symbol}|{briefing}")

    return lines


def write_chart_overlay(lines_by_symbol: dict[str, list[str]], commondata_path: str | Path) -> bool:
    """Writes every symbol's lines into ONE combined file in the
    terminal's COMMON Files folder (FILE_COMMON in MQL5 terms) — shared
    across every terminal/login profile on this machine, so the EA finds
    it regardless of which account is currently connected. The EA itself
    filters to its own chart's _Symbol when reading, so one file covers
    every symbol this account ever watches; each write fully replaces
    the prior content rather than appending, so a symbol no longer being
    watched simply stops appearing next refresh instead of leaving stale
    lines behind forever.

    Returns True on a successful write, False on any failure — logged,
    never raised, since this is a cosmetic side effect: a failure here
    (a locked file, a missing folder) must never interrupt whatever real
    Clerk/mega-session logic called it."""
    try:
        files_dir = Path(commondata_path) / "Files"
        files_dir.mkdir(parents=True, exist_ok=True)
        target = files_dir / OVERLAY_FILENAME
        all_lines = [line for symbol in sorted(lines_by_symbol) for line in lines_by_symbol[symbol]]
        tmp = target.with_suffix(".tmp")
        tmp.write_text("\n".join(all_lines) + ("\n" if all_lines else ""), encoding="utf-8")
        os.replace(tmp, target)
        return True
    except OSError as e:
        logger.warning("write_chart_overlay: failed to write overlay file: %s", e)
        return False
