"""Conservative, deterministic candlestick-shape detection — Nison's
Japanese Candlestick Charting Techniques is the canonical source for
every definition below (the standard reference, not an invented
convention), added 2026-08-26 direct user request after tracing a real,
recurring complaint (misreading pump/dump swings and sideways-vs-
breakout moments) to two gaps: analysis/technical.py's market_regime
having no short-window signal to catch an in-progress move (see
MOMENTUM_WINDOW there), and this codebase having ZERO candlestick-
pattern detection despite the user owning Nison's own book.

Same "no invented thresholds" discipline analysis/chart_structure.py's
own detect_chart_patterns already applies: only shapes with a hard,
mechanical, textbook definition are attempted. Deliberately excluded
this round: three white soldiers/three black crows (a subjective
"comparable body size across 3 bars" judgment with no hard %), harami/
harami cross (near-duplicate of engulfing's own logic), piercing line/
dark cloud cover (Nison gives only a "penetrates well into" rule of
thumb with no hard %, not a hard threshold to encode confidently). All
three are real candidates for a future round once a specific % is
sourced (see the NotebookLM query in the approved plan for this
feature) — not a permanent exclusion.

Only ever looks at the LATEST closed bar (plus however many prior bars
each specific pattern's own definition needs) — deliberately NOT
analysis/chart_structure.py's fractal swing-point machinery, which
requires a CONFIRMED pivot (only knowable once price has already moved
past it) before recognizing anything. Requiring a confirmed pivot first
would recreate the exact "wakes up late" lag this whole feature exists
to fix, just one layer down — a candlestick reversal is by definition a
call made on the most recent bar(s), not a lagging confirmation.

Returns [] whenever the given `history` lacks real Open/High/Low/Close
(e.g. PSX's EOD source, which has no High/Low at all — see
analysis/technical.py's own comment on that gap) or has too few bars —
never raises, matching every other detector in this project's "not
enough data yet is a normal, expected outcome" convention."""

import pandas as pd

from analysis.chart_structure import ChartPattern

# A bar's real body is at most this fraction of its own High-Low range
# to count as a doji ("the opening and closing prices are the same, or
# almost the same" — Nison). Expressed as a ratio, not an absolute
# price, so it auto-scales per instrument/timeframe the same way this
# project's other structural thresholds already do (see
# chart_structure.py's own SR_CLUSTER_TOLERANCE_PCT). Purely a "flag
# indecision, don't infer direction" signal — never treated as
# directional by analysis/setup_classifier.py.
DOJI_BODY_TO_RANGE_MAX = 0.1

# Hammer/hanging man (Nison): "the lower shadow should be at least twice
# the height of the real body" and "little or no upper shadow."
# Shooting star/inverted hammer is the exact mirror. Both mechanically
# identical shapes; which NAME applies depends only on the preceding
# trend (see _prior_trend) — part of the pattern's own definition, not
# an interpretive add-on.
HAMMER_LOWER_SHADOW_MIN_RATIO = 2.0
HAMMER_UPPER_SHADOW_MAX_RATIO = 0.1
STAR_UPPER_SHADOW_MIN_RATIO = 2.0
STAR_LOWER_SHADOW_MAX_RATIO = 0.1

# Morning/evening star's middle "star" candle must have a real body no
# larger than this fraction of its own range — a small, indecisive body
# sandwiched between two long ones, per Nison's own description.
STAR_BODY_TO_RANGE_MAX = 0.3

# Real bug found on self-audit: Nison's own definition requires the
# FIRST and THIRD candles to be genuinely LONG (that size contrast with
# the tiny middle "star" is the whole visual/definitional point), but an
# earlier version of this module never actually checked their size at
# all — only their direction (bearish/bullish) and that the third
# closes back past the first's own midpoint. Confirmed live this let a
# trivial 3-tiny-body flat sequence fire as a "morning star" even though
# none of the three bars was remotely long. Fixed by requiring each
# flanking candle's own body to be at least this many times the star's
# body — a relative measure (consistent with this file's other ratio-
# based thresholds, e.g. HAMMER_LOWER_SHADOW_MIN_RATIO) rather than a
# fresh absolute number, since "long" is meaningful only relative to the
# star it's supposed to dwarf.
STAR_FLANKING_BODY_MIN_RATIO = 2.0

# Confirmed 2026-08-27 directly against the user's own copy of Nison's
# book via NotebookLM (real cited text, not paraphrased from memory):
# the star candle (candle 2) gapping away from the first candle's real
# body is the "critical requirement" — but Nison's own text explicitly
# says the SECOND gap (between the star and the third candle) "is rare"
# and its absence "does not weaken the pattern's power." Even the
# first, "critical" gap essentially never happens on this account's
# continuously-traded Forex/Metals/Crypto/Indices CFDs on H1/H4 bars
# (equities-only assumption baked into Nison's own examples) — relaxed
# BOTH relationships here to "the star's own body sits at/beyond the
# midpoint of the first candle's real body" instead of a literal price
# gap for either one, a deliberate, now textually-cross-checked
# adaptation for a gap-less market, not an oversight.
REQUIRE_STAR_GAP = False

# How many bars back _prior_trend looks to judge "was this a downtrend
# or an uptrend going into this candle" — short and mechanical by
# design (a simple close-to-close comparison, not a confirmed swing
# pivot) for the same "don't recreate the lagging-confirmation problem"
# reason explained in this module's own docstring.
CANDLESTICK_TREND_CONTEXT_BARS = 5

# Minimum bars needed for the deepest pattern here (a 3-candle star,
# whose middle candle itself needs CANDLESTICK_TREND_CONTEXT_BARS of
# its own prior context) plus 3 bars of headroom.
CANDLESTICK_MIN_BARS = CANDLESTICK_TREND_CONTEXT_BARS + 3


def _body(bar: pd.Series) -> float:
    return abs(float(bar["Close"]) - float(bar["Open"]))


def _range(bar: pd.Series) -> float:
    return float(bar["High"]) - float(bar["Low"])


def _upper_shadow(bar: pd.Series) -> float:
    return float(bar["High"]) - max(float(bar["Open"]), float(bar["Close"]))


def _lower_shadow(bar: pd.Series) -> float:
    return min(float(bar["Open"]), float(bar["Close"])) - float(bar["Low"])


def _is_doji(bar: pd.Series) -> bool:
    rng = _range(bar)
    if rng <= 0:
        return False
    return _body(bar) / rng <= DOJI_BODY_TO_RANGE_MAX


def _prior_trend(closes: pd.Series, idx: int) -> str | None:
    """"up"/"down" if the close just before `idx` has genuinely moved
    away from the close CANDLESTICK_TREND_CONTEXT_BARS further back;
    None if there isn't enough history yet, or the two are exactly
    equal (no real context either way — deliberately not a coin-flip
    default)."""
    reference_idx = idx - 1 - CANDLESTICK_TREND_CONTEXT_BARS
    if idx < 1 or reference_idx < 0:
        return None
    recent_close = float(closes.iloc[idx - 1])
    reference_close = float(closes.iloc[reference_idx])
    if recent_close > reference_close:
        return "up"
    if recent_close < reference_close:
        return "down"
    return None


def _detect_doji(bars: pd.DataFrame, idx: int) -> ChartPattern | None:
    bar = bars.iloc[idx]
    if not _is_doji(bar):
        return None
    rng = _range(bar)
    body_pct = (_body(bar) / rng * 100) if rng > 0 else 0.0
    return ChartPattern(
        name="doji",
        detail=(
            f"Doji: real body is only {body_pct:.1f}% of this bar's own high-low range — "
            "genuine indecision between buyers and sellers, not a directional signal on its own."
        ),
    )


def _detect_engulfing(bars: pd.DataFrame, idx: int) -> ChartPattern | None:
    """Real bug found on self-audit, confirmed against real live H1 data
    across 6 FTMO instruments (300 bars each): a continuously-traded
    market's next bar opens at essentially the SAME price the prior bar
    closed at (median difference exactly 0.0, a few pips of tick noise
    either way) — this account never has the real overnight-style GAP
    Nison's own equity-market examples implicitly have between sessions.
    An earlier version of this function required cur_open to gap
    STRICTLY beyond prev_close (and cur_close strictly beyond
    prev_open), which on gap-less data only fires when that few-pips-of-
    noise happens to land the right way — confirmed live this missed
    roughly 20-40% of the genuinely-engulfing bodies a `<=`/`>=` (body
    fully contains, gap or no gap) check below correctly catches. The
    substance of "engulfing" is the current body containing the prior
    one, not a literal price gap — a real gap was never the actual
    defining feature for a continuously-traded instrument."""
    if idx < 1:
        return None
    prev, cur = bars.iloc[idx - 1], bars.iloc[idx]
    prev_open, prev_close = float(prev["Open"]), float(prev["Close"])
    cur_open, cur_close = float(cur["Open"]), float(cur["Close"])

    if prev_close < prev_open and cur_close > cur_open and cur_open <= prev_close and cur_close >= prev_open:
        return ChartPattern(
            name="bullish_engulfing",
            detail=(
                f"Bullish engulfing: this bar's real body ({cur_open:.5f} -> {cur_close:.5f}) fully "
                f"engulfs the prior bearish bar's body ({prev_open:.5f} -> {prev_close:.5f}) — a classic "
                "2-bar reversal shape (Nison)."
            ),
        )
    if prev_close > prev_open and cur_close < cur_open and cur_open >= prev_close and cur_close <= prev_open:
        return ChartPattern(
            name="bearish_engulfing",
            detail=(
                f"Bearish engulfing: this bar's real body ({cur_open:.5f} -> {cur_close:.5f}) fully "
                f"engulfs the prior bullish bar's body ({prev_open:.5f} -> {prev_close:.5f}) — a classic "
                "2-bar reversal shape (Nison)."
            ),
        )
    return None


def _detect_hammer_family(bars: pd.DataFrame, idx: int) -> ChartPattern | None:
    bar = bars.iloc[idx]
    if _is_doji(bar):
        # Excluded rather than double-labeled as a dragonfly/gravestone-
        # doji variant — that's a real, separate pattern this first
        # slice deliberately doesn't attempt (see module docstring).
        return None
    body = _body(bar)
    rng = _range(bar)
    if rng <= 0:
        return None
    lower = _lower_shadow(bar)
    upper = _upper_shadow(bar)
    trend = _prior_trend(bars["Close"], idx)
    if trend is None:
        return None

    if lower >= HAMMER_LOWER_SHADOW_MIN_RATIO * body and upper <= HAMMER_UPPER_SHADOW_MAX_RATIO * rng:
        if trend == "down":
            return ChartPattern(
                name="hammer",
                detail=(
                    f"Hammer: lower shadow ({lower:.5f}) is {lower / body:.1f}x the real body "
                    f"({body:.5f}) with a negligible upper shadow, after a recent downtrend — a "
                    "classic bullish reversal shape (Nison); a complete signal on its own, does "
                    "NOT require confirmation on a following bar."
                ),
            )
        return ChartPattern(
            name="hanging_man",
            detail=(
                f"Hanging man: lower shadow ({lower:.5f}) is {lower / body:.1f}x the real body "
                f"({body:.5f}) with a negligible upper shadow, after a recent uptrend — a classic "
                "bearish reversal shape (Nison); needs bearish confirmation on the next bar — at "
                "minimum a lower open, ideally a close beneath this bar's own real body."
            ),
        )

    if upper >= STAR_UPPER_SHADOW_MIN_RATIO * body and lower <= STAR_LOWER_SHADOW_MAX_RATIO * rng:
        if trend == "up":
            return ChartPattern(
                name="shooting_star",
                detail=(
                    f"Shooting star: upper shadow ({upper:.5f}) is {upper / body:.1f}x the real "
                    f"body ({body:.5f}) with a negligible lower shadow, after a recent uptrend — a "
                    "bearish reversal shape (Nison), though a comparatively minor one — Nison "
                    "himself rates it below the bearish engulfing/evening star, not pivotal "
                    "resistance on its own."
                ),
            )
        return ChartPattern(
            name="inverted_hammer",
            detail=(
                f"Inverted hammer: upper shadow ({upper:.5f}) is {upper / body:.1f}x the real body "
                f"({body:.5f}) with a negligible lower shadow, after a recent downtrend — a classic "
                "bullish reversal shape (Nison); needs bullish confirmation on the next bar — "
                "ideally a close above this bar's own real body."
            ),
        )
    return None


def _detect_star_pattern(bars: pd.DataFrame, idx: int) -> ChartPattern | None:
    if idx < 2:
        return None
    first, star, third = bars.iloc[idx - 2], bars.iloc[idx - 1], bars.iloc[idx]
    star_range = _range(star)
    star_body = _body(star)
    if star_range <= 0 or star_body / star_range > STAR_BODY_TO_RANGE_MAX:
        return None

    first_open, first_close = float(first["Open"]), float(first["Close"])
    third_open, third_close = float(third["Open"]), float(third["Close"])
    first_body = _body(first)
    third_body = _body(third)
    # Both flanking candles must genuinely dwarf the star — see
    # STAR_FLANKING_BODY_MIN_RATIO's own comment. Multiplying rather
    # than dividing avoids a division-by-zero guard entirely (a
    # star_body of exactly 0 just means any positive flanking body
    # already clears it, which is the correct outcome).
    if first_body < STAR_FLANKING_BODY_MIN_RATIO * star_body or third_body < STAR_FLANKING_BODY_MIN_RATIO * star_body:
        return None

    first_mid = (first_open + first_close) / 2
    star_low = min(float(star["Open"]), float(star["Close"]))
    star_high = max(float(star["Open"]), float(star["Close"]))

    # REQUIRE_STAR_GAP is False (see its own module comment) — the
    # star's position relative to the first candle's own body midpoint
    # substitutes for a literal price gap on these gap-less instruments.
    if first_close < first_open and third_close > third_open and star_high <= first_mid and third_close > first_mid:
        return ChartPattern(
            name="morning_star",
            detail=(
                f"Morning star: a long bearish bar, a small-bodied star sitting at/below its "
                f"midpoint ({first_mid:.5f}), then a bullish bar closing back above that midpoint "
                f"at {third_close:.5f} — a classic 3-bar bullish reversal (Nison)."
            ),
        )
    if first_close > first_open and third_close < third_open and star_low >= first_mid and third_close < first_mid:
        return ChartPattern(
            name="evening_star",
            detail=(
                f"Evening star: a long bullish bar, a small-bodied star sitting at/above its "
                f"midpoint ({first_mid:.5f}), then a bearish bar closing back below that midpoint "
                f"at {third_close:.5f} — a classic 3-bar bearish reversal (Nison)."
            ),
        )
    return None


def detect_candlestick_patterns(history: pd.DataFrame) -> list[ChartPattern]:
    """Real, computed candlestick-shape detection on the LATEST closed
    bar of `history` — [] whenever there's no real OHLC data or too few
    bars, never raises, same "not enough data yet is normal" contract as
    analysis/chart_structure.py::find_swing_points. Multiple patterns
    can legitimately fire for the same bar (e.g. a doji that also
    completes a morning star) — returns everything that genuinely
    applies, not a forced single verdict, matching setup_classifier.py's
    own established convention."""
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(history.columns) or len(history) < CANDLESTICK_MIN_BARS:
        return []

    bars = history.reset_index(drop=True)
    idx = len(bars) - 1
    patterns: list[ChartPattern] = []

    doji = _detect_doji(bars, idx)
    if doji is not None:
        patterns.append(doji)

    engulfing = _detect_engulfing(bars, idx)
    if engulfing is not None:
        patterns.append(engulfing)

    hammer_family = _detect_hammer_family(bars, idx)
    if hammer_family is not None:
        patterns.append(hammer_family)

    star = _detect_star_pattern(bars, idx)
    if star is not None:
        patterns.append(star)

    return patterns
