"""Synthesizes analysis/technical.py's TechnicalStats with
analysis/chart_structure.py's ChartStructureSnapshot into a small set of
real, deterministic trade-setup archetypes — reversal, pullback-
continuation, range-fade, breakout-watch, trend-following. This is a
classification LAYER, not a new data source: every signal it reasons over
was already computed by one of those two modules; this module only
combines them under well-defined rules, so an AI reading the result gets
"here is the kind of setup this looks like, and exactly why" instead of
having to reconstruct that judgment itself from a dozen separate numbers
every single time.

Deliberately does NOT try to be exhaustive — an instrument with no
confirmed structure just gets "no_clear_setup", which is a normal,
common, and useful result (it tells the model not to force a thesis),
not a failure of this module."""

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

    if not signals:
        signals.append(
            SetupSignal(
                name="no_clear_setup",
                detail=(
                    "No specific structural setup (reversal, pullback, range-fade, "
                    "breakout-watch, or trendline re-test) currently confirmed on this "
                    "timeframe — a normal, common result, not a gap to explain away."
                ),
            )
        )
    return signals
