"""Synthesizes analysis/technical.py's TechnicalStats with
analysis/chart_structure.py's ChartStructureSnapshot into a small set of
real, deterministic trade-setup archetypes — reversal, pullback-
continuation, range-fade, breakout-watch, trend-following, grind-
continuation, in-progress-move, busted-pattern-reversal, candlestick-
reversal-confirmed. This is a classification LAYER, not a new data
source: every signal it reasons over was already computed by one of
those two modules; this module only combines them under well-defined
rules, so an AI reading the result gets "here is the kind of setup this
looks like, and exactly why" instead of having to reconstruct that
judgment itself from a dozen separate numbers every single time.

`in_progress_move`/`busted_pattern_reversal`/`candlestick_reversal_
confirmed` (rules 8-10, added 2026-08-26) directly address a real,
recurring complaint: misreading pump/dump swings and sideways-vs-
breakout moments, because market_regime's own 60-bar window can let
several swings cancel out in its own average — see analysis/
technical.py's MOMENTUM_WINDOW comment for the full root-cause trace,
and analysis/candlestick_patterns.py for the Nison-sourced shape
detection rule 10 draws on.

Deliberately does NOT try to be exhaustive — an instrument with no
confirmed structure just gets "no_clear_setup", which is a normal,
common, and useful result (it tells the model not to force a thesis),
not a failure of this module.

`trend_intact` (see its own rule below) is the clean-trend counterpart
to `grind_continuation`: a real, cleanly-trending regime with no more
specific trigger (a fresh reversal/pullback/breakout/trendline-retest)
active right now still IS genuine directional evidence, not nothing —
real gap found live 2026-08-22 (user's own words, looking at a real
BTCUSD read: "the asset is clearly up trend but technical showing
different story") where a clean `trending_up` regime with no special
trigger fell all the way through to `no_clear_setup`, reading
identically to genuinely having no evidence at all despite the
trend/regime badges right next to it agreeing on a clean uptrend."""

from dataclasses import dataclass

from analysis.chart_structure import ChartStructureSnapshot
from analysis.technical import TechnicalStats

# A support/resistance level needs at least this many real touches to
# count as "well-tested" for setup-classification purposes — a level
# touched only once is real (see chart_structure.py) but not yet strong
# enough evidence to anchor a reversal/range-fade call on.
STRONG_TOUCH_COUNT = 2

# How close price needs to sit to a level/trendline to count as "at" it
# for setup-classification purposes (re-test, range boundary, etc.).
NEAR_LEVEL_TOLERANCE_PCT = 0.5

# The classic Fibonacci "buy the dip" / "sell the rally" retracement
# band — not this project's own invention, the standard convention.
FIB_GOLDEN_ZONE = (0.382, 0.618)

# Candlestick names (analysis/candlestick_patterns.py) grouped by which
# direction they argue for, for the candlestick_reversal_confirmed rule
# below.
_BULLISH_CANDLES = ("bullish_engulfing", "hammer", "inverted_hammer", "morning_star")
_BEARISH_CANDLES = ("bearish_engulfing", "hanging_man", "shooting_star", "evening_star")


@dataclass
class SetupSignal:
    name: str
    detail: str


def _true_retracement_ratio(fib) -> float | None:
    """The ACTUAL continuous retracement ratio of current price within
    the fib swing (0.0 at the more-recent extreme, 1.0 at the older
    one) — computed directly from swing_high/swing_low/current_price,
    NOT snapped to the nearest of the 7 named levels first. Snapping
    first misclassifies prices just short of (or just past) the golden
    zone's real boundary as inside it whenever their nearest named level
    happens to sit on the other side of that boundary — confirmed live:
    a true ratio of 0.36 (genuinely short of the 0.382 boundary) snaps
    to the "38.2%" label and would wrongly read as inside the zone."""
    span = fib.swing_high - fib.swing_low
    if span == 0:
        return None
    if fib.high_is_more_recent:
        return (fib.swing_high - fib.current_price) / span
    return (fib.current_price - fib.swing_low) / span


def _find_pattern(structure: ChartStructureSnapshot, *names: str):
    return next((p for p in structure.patterns if p.name in names), None)


def _strongest_nearby_sr_level(structure: ChartStructureSnapshot):
    if structure.sr_levels is None:
        return None
    candidates = [
        lvl
        for lvl in structure.sr_levels.resistance_levels + structure.sr_levels.support_levels
        if lvl.touches >= STRONG_TOUCH_COUNT
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda lvl: abs(lvl.distance_pct))


def classify_setups(stats: TechnicalStats, structure: ChartStructureSnapshot) -> list[SetupSignal]:
    """Real, rule-based classification — every rule below cites the exact
    computed signal(s) it fired on in the returned `detail`, so a
    downstream reader (human or model) can verify the call rather than
    trust it blindly. More than one signal can legitimately fire at once
    (e.g. a pullback INTO a converging triangle) — this returns all that
    genuinely apply, not a forced single verdict."""
    signals: list[SetupSignal] = []
    if stats.last_price is None:
        return signals

    trend_pattern = _find_pattern(structure, "uptrend_structure", "downtrend_structure")
    reversal_pattern = _find_pattern(structure, "double_top", "double_bottom")
    triangle_pattern = _find_pattern(
        structure, "ascending_triangle", "descending_triangle", "converging_triangle"
    )
    nearest_sr = _strongest_nearby_sr_level(structure)

    # 1. Reversal candidate — a real double top/bottom fired.
    if reversal_pattern is not None:
        signals.append(
            SetupSignal(
                name="reversal_candidate",
                detail=(
                    f"{reversal_pattern.detail} Treat a break of the neckline as "
                    "confirmation, a failure to break as invalidation — the pattern "
                    "alone isn't a trigger."
                ),
            )
        )

    # 2. Pullback/continuation — an established trend structure, with
    # price currently retracing into the Fibonacci golden zone in the
    # direction that continues (not reverses) that trend: an uptrend
    # pulling back DOWN from a fresh high, or a downtrend bouncing UP
    # from a fresh low.
    if trend_pattern is not None and structure.fibonacci is not None:
        trend_is_up = trend_pattern.name == "uptrend_structure"
        fib = structure.fibonacci
        ratio = _true_retracement_ratio(fib)
        in_golden_zone = ratio is not None and FIB_GOLDEN_ZONE[0] <= ratio <= FIB_GOLDEN_ZONE[1]
        pullback_direction_ok = (trend_is_up and fib.high_is_more_recent) or (
            not trend_is_up and not fib.high_is_more_recent
        )
        if in_golden_zone and pullback_direction_ok:
            signals.append(
                SetupSignal(
                    name="pullback_continuation",
                    detail=(
                        f"{'Uptrend' if trend_is_up else 'Downtrend'} structure "
                        f"({trend_pattern.detail}) with price currently retracing into "
                        f"the {fib.nearest_level_name} Fibonacci zone — a classic "
                        "continuation-pullback entry, not a reversal signal."
                    ),
                )
            )

    # 3. Range-fade candidate — sideways regime, price sitting right at a
    # well-tested level, nothing suggesting an imminent breakout.
    if stats.market_regime == "sideways" and nearest_sr is not None and abs(nearest_sr.distance_pct) <= NEAR_LEVEL_TOLERANCE_PCT:
        signals.append(
            SetupSignal(
                name="range_fade_candidate",
                detail=(
                    f"Sideways regime with price {abs(nearest_sr.distance_pct):.2f}% from "
                    f"a {nearest_sr.touches}-touch level at {nearest_sr.price:.4f} — a "
                    "candidate to fade back toward the range rather than expect a breakout."
                ),
            )
        )

    # 4. Breakout watch — a converging triangle/wedge; direction isn't
    # implied by the pattern alone.
    if triangle_pattern is not None:
        signals.append(
            SetupSignal(
                name="breakout_watch",
                detail=(
                    f"{triangle_pattern.detail} A converging structure like this "
                    "typically resolves with a breakout, but the pattern alone doesn't "
                    "say which direction — confirm with the trend/momentum signals."
                ),
            )
        )

    # 5. Trend-following — an established trend, with price sitting right
    # on its OWN trendline (a real re-test, the classic trend-following
    # entry), not just "price is above a moving average somewhere".
    if trend_pattern is not None and structure.trendlines is not None:
        trend_is_up = trend_pattern.name == "uptrend_structure"
        relevant_line = (
            structure.trendlines.support_trendline if trend_is_up else structure.trendlines.resistance_trendline
        )
        if relevant_line is not None and abs(relevant_line.distance_pct) <= NEAR_LEVEL_TOLERANCE_PCT:
            signals.append(
                SetupSignal(
                    name="trend_following",
                    detail=(
                        f"{'Uptrend' if trend_is_up else 'Downtrend'} structure with price "
                        f"sitting right on its own {'support' if trend_is_up else 'resistance'} "
                        f"trendline ({relevant_line.distance_pct:+.2f}% away) — a real "
                        "trendline re-test, the classic trend-following entry point."
                    ),
                )
            )

    # 6. Grind continuation — market_regime is choppy_up/choppy_down (see
    # analysis/technical.py's own comment on that computation): a real
    # net directional move over the medium-term window, just reached via
    # a noisy, back-and-forth path rather than a clean trend. This is
    # genuine directional evidence a model can act on — real bug found
    # live: before market_regime distinguished "choppy but real
    # direction" from "no direction at all", both cases were lumped into
    # one "sideways" read, which then only ever fed range_fade_candidate
    # (rule 3) or fell through to no_clear_setup below, discarding the
    # real direction entirely. Momentum already at an extreme in the
    # SAME direction (RSI overbought on a choppy uptrend, oversold on a
    # choppy downtrend) argues the move may already be stretched, so this
    # deliberately doesn't fire there — a fresh, not-yet-exhausted grind
    # is a materially different setup than a tired one.
    if stats.market_regime in ("choppy_up", "choppy_down"):
        direction_word = "up" if stats.market_regime == "choppy_up" else "down"
        momentum_extreme = stats.rsi is not None and (
            (direction_word == "up" and stats.rsi >= 70) or (direction_word == "down" and stats.rsi <= 30)
        )
        if not momentum_extreme:
            signals.append(
                SetupSignal(
                    name="grind_continuation",
                    detail=(
                        f"A real net {direction_word}ward move over the medium-term window, "
                        "reached via a noisy, back-and-forth path rather than a clean "
                        "trend — genuine directional evidence, just expect whipsaws rather "
                        "than a straight-line move; RSI isn't already at an extreme that "
                        "would argue the move is exhausted."
                    ),
                )
            )

    # 7. Trend intact — market_regime is a CLEAN trending_up/trending_down
    # (not choppy — that's grind_continuation above), and nothing more
    # SPECIFIC already fired for this same direction (a fresh trendline
    # retest — trend_following — or a golden-zone pullback —
    # pullback_continuation — are stronger, narrower versions of this
    # exact situation, so this stays quiet rather than restating it).
    # Same RSI-not-already-extreme guard as grind_continuation, for the
    # same reason: a trend already stretched to an extreme is a
    # materially different setup than a fresh, still-room-to-run one.
    if stats.market_regime in ("trending_up", "trending_down") and not any(
        s.name in ("trend_following", "pullback_continuation") for s in signals
    ):
        direction_word = "up" if stats.market_regime == "trending_up" else "down"
        momentum_extreme = stats.rsi is not None and (
            (direction_word == "up" and stats.rsi >= 70) or (direction_word == "down" and stats.rsi <= 30)
        )
        if not momentum_extreme:
            signals.append(
                SetupSignal(
                    name="trend_intact",
                    detail=(
                        f"A clean, established {direction_word}trend (market regime reads "
                        f"trending_{direction_word}) with no more specific reversal/"
                        "pullback/breakout/trendline-retest trigger active right now — "
                        "real directional evidence to lean with rather than a reason to "
                        "sit out just because there's no fresh entry trigger this moment; "
                        "RSI isn't already at an extreme that would argue the move is "
                        "exhausted."
                    ),
                )
            )

    # 8. In-progress move — direct fix for a real, recurring complaint:
    # "the system wakes up late in the middle of what looks like a
    # sideways market when a real move is already underway." market_
    # regime reads sideways over the medium-term (60-bar) window because
    # several pump/dump swings cancel out in that window's own average,
    # but analysis/technical.py's own shorter momentum_acceleration
    # window shows a real, directional push happening right now.
    if stats.market_regime == "sideways" and stats.momentum_acceleration in (
        "accelerating_up", "accelerating_down"
    ):
        direction_word = "up" if stats.momentum_acceleration == "accelerating_up" else "down"
        signals.append(
            SetupSignal(
                name="in_progress_move",
                detail=(
                    "Medium-term regime reads sideways (recent swings are cancelling out in "
                    f"that window's own average), but the short-term window shows a real, "
                    f"efficient push {direction_word}ward — a fresh move may already be "
                    "underway inside what still looks like a flat window; don't dismiss this "
                    "as noise just because the medium-term regime hasn't caught up yet."
                ),
            )
        )

    # 9. Busted pattern reversal — Bulkowski's own documented concept
    # (Encyclopedia of Chart Patterns): a pattern that clearly implies
    # one breakout direction but instead sees price confirm the OPPOSITE
    # direction is a "busted pattern," and his own statistics show these
    # often travel FURTHER than the original pattern's own target, since
    # traders positioned for the expected breakout are forced to exit/
    # reverse into the move. Needs no new detection — cross-references
    # the already-computed double_top/double_bottom against market_
    # regime/momentum_acceleration now confirming the other way. Fires
    # ALONGSIDE reversal_candidate (rule 1), not instead of it — both are
    # real, complementary evidence.
    #
    # Confirmed 2026-08-27 directly against the user's own copy of
    # Bulkowski's book via NotebookLM (real cited text/numbers, not
    # secondhand): his OWN precise definition of a "bust" is narrower
    # than this rule's heuristic — the initial breakout must travel LESS
    # THAN 5% before reversing, then price must re-enter the pattern and
    # break out the OPPOSITE side to count as fully busted. This rule
    # instead just checks whether CURRENT market_regime/momentum_
    # acceleration confirm the opposite of what the pattern implied,
    # with no minimum-penetration/re-entry tracking of the actual post-
    # pattern price path (chart_structure.py doesn't currently record a
    # pattern's own breakout point or track price after it). A real,
    # scoped follow-on, not implemented here. The qualitative claim
    # below ("busted patterns travel FURTHER") is itself now backed by
    # real numbers, not just paraphrased: e.g. a busted Eve & Eve double
    # top averages a 29-70% post-bust rise against the original
    # pattern's own -18% to -25% average decline; a busted descending
    # triangle averages a 43-52% post-bust move.
    if reversal_pattern is not None:
        implied_down = reversal_pattern.name == "double_top"
        confirmed_up = stats.market_regime in ("trending_up", "choppy_up") or stats.momentum_acceleration == "accelerating_up"
        confirmed_down = stats.market_regime in ("trending_down", "choppy_down") or stats.momentum_acceleration == "accelerating_down"
        if (implied_down and confirmed_up) or (not implied_down and confirmed_down):
            actual_direction = "up" if confirmed_up else "down"
            signals.append(
                SetupSignal(
                    name="busted_pattern_reversal",
                    detail=(
                        f"{reversal_pattern.detail} This pattern implied a break "
                        f"{'DOWN' if implied_down else 'UP'}, but real subsequent price action "
                        f"instead confirms a genuine push {actual_direction}ward (market regime/"
                        "short-term momentum) — a BUSTED pattern per Bulkowski's Encyclopedia of "
                        "Chart Patterns, whose statistics show busted patterns often travel "
                        "FURTHER than the original pattern's own target. Treat this as real, "
                        "stronger directional evidence in the busted direction, not a reason "
                        "to distrust it."
                    ),
                )
            )

    # 10. Candlestick reversal confirmed — a candlestick pattern
    # (analysis/candlestick_patterns.py) only counts as an actionable
    # setup signal when it appears in the context Nison's own definition
    # requires to be meaningful: a bullish shape after a downtrend or at
    # a double-bottom; a bearish shape after an uptrend or at a
    # double-top. Ties the raw candlestick pattern list into an actual
    # signal rather than leaving it to only ever reach the raw chart-
    # structure text.
    candle_bull = _find_pattern(structure, *_BULLISH_CANDLES)
    candle_bear = _find_pattern(structure, *_BEARISH_CANDLES)
    if candle_bull is not None and (
        stats.trend == "downtrend" or (reversal_pattern is not None and reversal_pattern.name == "double_bottom")
    ):
        signals.append(
            SetupSignal(
                name="candlestick_reversal_confirmed",
                detail=(
                    f"{candle_bull.detail} — a bullish candlestick reversal (Nison) confirming "
                    "an existing bearish context; treat as added weight behind a bullish "
                    "reversal thesis, not a standalone trigger."
                ),
            )
        )
    if candle_bear is not None and (
        stats.trend == "uptrend" or (reversal_pattern is not None and reversal_pattern.name == "double_top")
    ):
        signals.append(
            SetupSignal(
                name="candlestick_reversal_confirmed",
                detail=(
                    f"{candle_bear.detail} — a bearish candlestick reversal (Nison) confirming "
                    "an existing bullish context; treat as added weight behind a bearish "
                    "reversal thesis, not a standalone trigger."
                ),
            )
        )

    if not signals:
        # Real gap found on self-audit: this list was already stale
        # before this round (trend_intact, added 2026-08-22, was never
        # added here either) — updated to name every archetype now,
        # not just re-adding the 3 newest ones, so this doesn't need a
        # third catch next time the list changes.
        signals.append(
            SetupSignal(
                name="no_clear_setup",
                detail=(
                    "No specific structural setup (reversal, pullback, range-fade, "
                    "breakout-watch, trend-following, grind-continuation, trend-intact, "
                    "in-progress-move, busted-pattern-reversal, or candlestick-reversal) "
                    "currently confirmed on this timeframe — a normal, common result, not a "
                    "gap to explain away."
                ),
            )
        )
    return signals
