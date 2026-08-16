from dataclasses import dataclass

import numpy as np
import pandas as pd

# Fractal swing-point definition: a bar is a swing high/low if its own
# High/Low is the STRICT, UNIQUE extreme among itself and `window` bars on
# each side (a "5-bar fractal" at the default window=2). Strict+unique
# avoids flat-top/flat-bottom ambiguity (a tie doesn't register as either
# bar being "the" pivot) rather than picking one arbitrarily. This is the
# one building block every other function in this module is built from —
# getting it right matters more than any single downstream computation.
SWING_WINDOW = 2

# How far back (in bars) these functions look for swing structure — a
# fixed bar count, not a fixed calendar period, so the same window means
# "the same number of recent turning points" whether it's fed H1, H4, or
# daily bars (data.mt5_source.fetch_mt5_price_history's own three
# timeframes, or PMEX/PSX's daily history) — genuinely generic, the same
# way analysis/technical.py's own compute_technical_stats is.
STRUCTURE_LOOKBACK = 90

FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)

# Two swing prices within this % of each other are treated as "the same"
# level for clustering (support/resistance touch-counting) and for
# double-top/double-bottom matching — real quotes are never exactly
# equal, so some tolerance is unavoidable; this is deliberately tight
# (a genuine re-test, not "roughly the same area").
SR_CLUSTER_TOLERANCE_PCT = 0.3
SR_MAX_LEVELS_PER_SIDE = 3

TRENDLINE_MIN_POINTS = 3
# A trendline's slope, projected across the whole lookback window, that
# moves price by less than this % is called "flat" rather than
# rising/falling — avoids labeling near-zero numerical noise as a real
# directional trendline.
TRENDLINE_FLAT_THRESHOLD_PCT = 1.0

# A double-top/bottom's intervening pullback (the swing low between two
# comparable highs, or swing high between two comparable lows) must be at
# least this far from both peaks/troughs to count as a genuine reversal
# in between, not just noise sitting almost at the same level.
DOUBLE_PATTERN_MIN_PULLBACK_PCT = 0.5


@dataclass
class SwingPoint:
    index: int  # bar position within the DataFrame passed in, 0-based, oldest-first
    price: float


def find_swing_points(
    history: pd.DataFrame, window: int = SWING_WINDOW
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """Fractal swing-high/swing-low detection over the FULL `history`
    given (callers slice to their own lookback window first) — see the
    module-level SWING_WINDOW comment for the exact definition. Returns
    (swing_highs, swing_lows), each chronological (oldest first). Needs
    High/Low columns and at least 2*window+1 rows; ([], []) otherwise —
    never raises, matching this project's "not enough data yet" is a
    normal, expected outcome convention."""
    if not {"High", "Low"}.issubset(history.columns) or len(history) < 2 * window + 1:
        return [], []

    highs = history["High"].reset_index(drop=True)
    lows = history["Low"].reset_index(drop=True)
    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for i in range(window, len(history) - window):
        high_neighborhood = highs.iloc[i - window : i + window + 1]
        if highs.iloc[i] == high_neighborhood.max() and (high_neighborhood == highs.iloc[i]).sum() == 1:
            swing_highs.append(SwingPoint(index=i, price=float(highs.iloc[i])))

        low_neighborhood = lows.iloc[i - window : i + window + 1]
        if lows.iloc[i] == low_neighborhood.min() and (low_neighborhood == lows.iloc[i]).sum() == 1:
            swing_lows.append(SwingPoint(index=i, price=float(lows.iloc[i])))

    return swing_highs, swing_lows


def _tail_reset(history: pd.DataFrame, lookback: int) -> pd.DataFrame:
    tail = history.tail(lookback) if len(history) > lookback else history
    return tail.reset_index(drop=True)


@dataclass
class FibonacciLevels:
    swing_high: float
    swing_low: float
    high_is_more_recent: bool  # True: last move was UP (retracing down from the high); False: last move was DOWN
    levels: dict[str, float]  # "0.0%".."100.0%" -> price, in FIB_RATIOS order
    current_price: float
    nearest_level_name: str
    nearest_level_price: float
    distance_to_nearest_pct: float  # (current_price - nearest_level_price) / current_price * 100


def compute_fibonacci_levels(
    history: pd.DataFrame,
    lookback: int = STRUCTURE_LOOKBACK,
    swing_points: tuple[list[SwingPoint], list[SwingPoint]] | None = None,
) -> FibonacciLevels | None:
    """Standard Fibonacci retracement between the most recent confirmed
    swing high and swing low within the last `lookback` bars. Direction
    (which extreme is the 0% anchor) follows the standard charting
    convention: retracement is measured back from whichever extreme came
    LAST — a fresh high retraces DOWN toward the prior low (0% at the
    high), a fresh low retraces UP toward the prior high (0% at the low).

    Falls back to the window's own raw High/Low (no fractal confirmation
    yet) when there isn't a confirmed swing on both sides — the same
    "not enough structure yet, use the wider raw range instead" degrade
    analysis/technical.py's own support/resistance already uses, rather
    than returning None just because a fractal pivot hasn't confirmed.

    `swing_points`, if given, is used as-is instead of calling
    find_swing_points again — lets compute_chart_structure() compute it
    ONCE and share it across all four reads rather than recomputing the
    same fractal scan five separate times for one symbol/timeframe
    (confirmed: that's exactly what happened before this param existed)."""
    window = _tail_reset(history, lookback)
    if window.empty or "Close" not in window.columns:
        return None
    current_price = float(window["Close"].iloc[-1])

    swing_highs, swing_lows = swing_points if swing_points is not None else find_swing_points(window)
    if swing_highs and swing_lows:
        recent_high = max(swing_highs, key=lambda p: p.index)
        recent_low = max(swing_lows, key=lambda p: p.index)
    elif {"High", "Low"}.issubset(window.columns):
        high_idx = int(window["High"].idxmax())
        low_idx = int(window["Low"].idxmin())
        recent_high = SwingPoint(index=high_idx, price=float(window["High"].iloc[high_idx]))
        recent_low = SwingPoint(index=low_idx, price=float(window["Low"].iloc[low_idx]))
    else:
        return None

    if recent_high.price == recent_low.price:
        return None  # degenerate — no real range to retrace

    high_is_more_recent = recent_high.index > recent_low.index
    span = recent_high.price - recent_low.price

    levels: dict[str, float] = {}
    for ratio in FIB_RATIOS:
        name = f"{ratio * 100:.1f}%"
        if high_is_more_recent:
            levels[name] = recent_high.price - ratio * span  # descends from the fresh high toward the old low
        else:
            levels[name] = recent_low.price + ratio * span  # ascends from the fresh low toward the old high

    nearest_name = min(levels, key=lambda name: abs(current_price - levels[name]))
    nearest_price = levels[nearest_name]

    return FibonacciLevels(
        swing_high=recent_high.price,
        swing_low=recent_low.price,
        high_is_more_recent=high_is_more_recent,
        levels=levels,
        current_price=current_price,
        nearest_level_name=nearest_name,
        nearest_level_price=nearest_price,
        distance_to_nearest_pct=(current_price - nearest_price) / current_price * 100 if current_price else 0.0,
    )


def _cluster_prices(prices: list[float], tolerance_pct: float) -> list[tuple[float, int]]:
    """Greedy 1D clustering: walk sorted prices, growing a cluster while
    the next price is within `tolerance_pct` of the cluster's own FIRST
    (lowest) point — a simple, deterministic, order-independent (sorted
    first) way to group real swing-point "touches" of the same level
    without needing k-means or a fixed level count decided in advance.

    Anchored to the cluster's first point specifically, NOT a running
    mean that shifts as points are added — a mean-based anchor lets a
    chain of small, individually-in-tolerance steps "walk" the cluster
    arbitrarily far (confirmed live: a sequence of 0.05%-apart points
    chained into one cluster spanning ~0.5% total under a 0.3% mean-
    based tolerance), silently overstating how tightly the real touches
    actually agree. Anchoring to the fixed first point instead guarantees
    every cluster's own total span never exceeds tolerance_pct.

    Returns (cluster_mean, touch_count) pairs, unsorted (callers sort by
    whatever they need — touch count, distance)."""
    if not prices:
        return []
    sorted_prices = sorted(prices)
    clusters: list[list[float]] = [[sorted_prices[0]]]
    for price in sorted_prices[1:]:
        cluster_anchor = clusters[-1][0]
        within_tolerance = cluster_anchor != 0 and (price - cluster_anchor) / cluster_anchor * 100 <= tolerance_pct
        if within_tolerance:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    return [(sum(c) / len(c), len(c)) for c in clusters]


@dataclass
class SRLevel:
    price: float
    touches: int  # how many real swing points clustered into this level — a real strength measure, not assumed
    distance_pct: float  # (level - current_price) / current_price * 100 — positive above price, negative below


@dataclass
class SRLevelsResult:
    resistance_levels: list[SRLevel]  # above current price, strongest (most touches) first
    support_levels: list[SRLevel]  # below current price, strongest first


def compute_sr_levels(
    history: pd.DataFrame,
    lookback: int = STRUCTURE_LOOKBACK,
    tolerance_pct: float = SR_CLUSTER_TOLERANCE_PCT,
    max_levels: int = SR_MAX_LEVELS_PER_SIDE,
    swing_points: tuple[list[SwingPoint], list[SwingPoint]] | None = None,
) -> SRLevelsResult | None:
    """Multi-level support/resistance from real swing-point clustering —
    a genuinely different, richer read than analysis/technical.py's own
    single rolling min/max range: a level with 4 confirmed touches is
    real, tested evidence a specific price has repeatedly mattered,
    which a single high/low over the window can't distinguish from a
    level that was only ever touched once. None if there's no confirmed
    swing structure at all in the window (a fresh instrument, or one
    that's been in a pure straight line with no real pivots).

    `swing_points`, if given, is reused instead of recomputed — see
    compute_fibonacci_levels' own docstring for the full rationale."""
    window = _tail_reset(history, lookback)
    if window.empty or "Close" not in window.columns:
        return None
    current_price = float(window["Close"].iloc[-1])

    swing_highs, swing_lows = swing_points if swing_points is not None else find_swing_points(window)
    all_prices = [p.price for p in swing_highs] + [p.price for p in swing_lows]
    if not all_prices:
        return None

    clusters = _cluster_prices(all_prices, tolerance_pct)
    levels = [
        SRLevel(
            price=price,
            touches=touches,
            distance_pct=(price - current_price) / current_price * 100 if current_price else 0.0,
        )
        for price, touches in clusters
    ]

    resistance = sorted(
        (level for level in levels if level.price > current_price),
        key=lambda level: (-level.touches, level.distance_pct),
    )[:max_levels]
    support = sorted(
        (level for level in levels if level.price < current_price),
        key=lambda level: (-level.touches, -level.distance_pct),
    )[:max_levels]

    return SRLevelsResult(resistance_levels=resistance, support_levels=support)


@dataclass
class Trendline:
    slope_per_bar: float
    direction: str  # "rising" / "falling" / "flat"
    current_price_on_line: float  # the fitted line's value extrapolated to the LATEST bar
    price_vs_line: str  # "above" / "below" / "at"
    distance_pct: float  # (current_close - current_price_on_line) / current_close * 100
    bars_since_last_point: int  # how far the line's own rightmost fitted point sits from the latest bar


def _fit_trendline(points: list[SwingPoint], last_index: int, current_close: float) -> Trendline | None:
    if len(points) < TRENDLINE_MIN_POINTS or current_close == 0:
        return None

    xs = np.array([p.index for p in points], dtype=float)
    ys = np.array([p.price for p in points], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    current_price_on_line = float(slope * last_index + intercept)
    # A trendline fit from points that are old relative to the current
    # bar gets extrapolated further to reach it — real, if usually small
    # (confirmed live: typically 3-12 bars on active FTMO instruments) —
    # and unlike a hard cutoff that would silently drop an otherwise-
    # useful trendline, surfacing the real number lets a reader judge
    # how much to trust an unusually stale one instead.
    bars_since_last_point = last_index - int(xs.max())

    # Direction judged by the line's OWN implied % move across the fitted
    # span, not the raw per-bar slope — a per-bar threshold would mean
    # something different for a $1 forex pair than a $60,000 crypto pair,
    # while a % move across the span is comparable across instruments.
    span_bars = xs.max() - xs.min()
    implied_pct_move = (
        (slope * span_bars) / current_price_on_line * 100 if current_price_on_line else 0.0
    )
    if implied_pct_move > TRENDLINE_FLAT_THRESHOLD_PCT:
        direction = "rising"
    elif implied_pct_move < -TRENDLINE_FLAT_THRESHOLD_PCT:
        direction = "falling"
    else:
        direction = "flat"

    distance_pct = (current_close - current_price_on_line) / current_close * 100
    if abs(distance_pct) < 0.01:
        price_vs_line = "at"
    else:
        price_vs_line = "above" if current_close > current_price_on_line else "below"

    return Trendline(
        slope_per_bar=float(slope),
        direction=direction,
        current_price_on_line=current_price_on_line,
        price_vs_line=price_vs_line,
        distance_pct=distance_pct,
        bars_since_last_point=bars_since_last_point,
    )


@dataclass
class TrendlineAnalysis:
    resistance_trendline: Trendline | None  # least-squares fit through swing HIGHS
    support_trendline: Trendline | None  # least-squares fit through swing LOWS


def compute_trendlines(
    history: pd.DataFrame,
    lookback: int = STRUCTURE_LOOKBACK,
    swing_points: tuple[list[SwingPoint], list[SwingPoint]] | None = None,
) -> TrendlineAnalysis | None:
    """Real trendlines fit through actual swing points (least-squares,
    not just connecting the two most recent points, which is overly
    sensitive to a single outlier pivot) — needs at least
    TRENDLINE_MIN_POINTS confirmed swing highs/lows respectively; either
    side is None on its own if it doesn't have enough, rather than the
    whole result being None just because one side lacks structure.

    `swing_points`, if given, is reused instead of recomputed — see
    compute_fibonacci_levels' own docstring for the full rationale."""
    window = _tail_reset(history, lookback)
    if window.empty or "Close" not in window.columns:
        return None
    current_close = float(window["Close"].iloc[-1])
    last_index = len(window) - 1

    swing_highs, swing_lows = swing_points if swing_points is not None else find_swing_points(window)
    resistance = _fit_trendline(swing_highs, last_index, current_close)
    support = _fit_trendline(swing_lows, last_index, current_close)
    if resistance is None and support is None:
        return None
    return TrendlineAnalysis(resistance_trendline=resistance, support_trendline=support)


@dataclass
class ChartPattern:
    name: str  # "double_top" / "double_bottom" / "uptrend_structure" / "downtrend_structure" /
    #             "ascending_triangle" / "descending_triangle" / "converging_triangle"
    detail: str  # one plain-language sentence with the real numbers behind the call


def _detect_double_pattern(
    extreme_points: list[SwingPoint],
    opposing_points: list[SwingPoint],
    is_top: bool,
    tolerance_pct: float,
) -> ChartPattern | None:
    """Shared logic for double-top (extreme_points=highs, opposing=lows)
    and double-bottom (extreme_points=lows, opposing=highs): the two
    MOST RECENT extreme points must sit within `tolerance_pct` of each
    other, with a genuine opposing pivot between them (not just adjacent
    noise) that sits meaningfully away from both — a real pullback/
    bounce in between, not two peaks/troughs with nothing separating
    them structurally."""
    if len(extreme_points) < 2:
        return None
    recent_two = sorted(extreme_points, key=lambda p: p.index)[-2:]
    first, second = recent_two
    avg_price = (first.price + second.price) / 2
    if avg_price == 0 or abs(first.price - second.price) / avg_price * 100 > tolerance_pct:
        return None

    between = [p for p in opposing_points if first.index < p.index < second.index]
    if not between:
        return None
    neckline = min(between, key=lambda p: p.price) if is_top else max(between, key=lambda p: p.price)

    pullback_pct = abs(avg_price - neckline.price) / avg_price * 100
    if pullback_pct < DOUBLE_PATTERN_MIN_PULLBACK_PCT:
        return None

    kind = "double_top" if is_top else "double_bottom"
    peak_word = "peaks" if is_top else "troughs"
    return ChartPattern(
        name=kind,
        detail=(
            f"Two {peak_word} at {first.price:.4f} and {second.price:.4f} "
            f"(within {tolerance_pct:.1f}% of each other), separated by a real "
            f"{'pullback' if is_top else 'bounce'} to {neckline.price:.4f} "
            f"({pullback_pct:.2f}% away) — neckline at {neckline.price:.4f}."
        ),
    )


def _detect_trend_structure(swing_highs: list[SwingPoint], swing_lows: list[SwingPoint]) -> ChartPattern | None:
    """Real Dow-theory-style trend structure: the two most recent swing
    highs AND the two most recent swing lows both need to agree on
    direction (both higher, or both lower) — a genuinely different,
    stricter check than any single moving-average read, since it's
    about the actual SHAPE of recent price action, not just where price
    sits relative to an average."""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    highs = sorted(swing_highs, key=lambda p: p.index)[-2:]
    lows = sorted(swing_lows, key=lambda p: p.index)[-2:]

    higher_highs = highs[1].price > highs[0].price
    higher_lows = lows[1].price > lows[0].price
    lower_highs = highs[1].price < highs[0].price
    lower_lows = lows[1].price < lows[0].price

    if higher_highs and higher_lows:
        return ChartPattern(
            name="uptrend_structure",
            detail=(
                f"Higher highs ({highs[0].price:.4f} -> {highs[1].price:.4f}) AND higher lows "
                f"({lows[0].price:.4f} -> {lows[1].price:.4f}) — a genuine uptrend structure, "
                "not just price above a moving average."
            ),
        )
    if lower_highs and lower_lows:
        return ChartPattern(
            name="downtrend_structure",
            detail=(
                f"Lower highs ({highs[0].price:.4f} -> {highs[1].price:.4f}) AND lower lows "
                f"({lows[0].price:.4f} -> {lows[1].price:.4f}) — a genuine downtrend structure."
            ),
        )
    return None


def _detect_triangle(trendlines: TrendlineAnalysis) -> ChartPattern | None:
    """Triangle/wedge from real trendline slopes — deliberately only
    flags a pattern when the two lines are genuinely CONVERGING
    (resistance flat-or-falling while support flat-or-rising, and not
    both simply flat): a parallel channel or two diverging lines is a
    real, different structure, not a triangle, and isn't flagged as one."""
    r, s = trendlines.resistance_trendline, trendlines.support_trendline
    if r is None or s is None:
        return None

    resistance_ok = r.direction in ("falling", "flat")
    support_ok = s.direction in ("rising", "flat")
    if not (resistance_ok and support_ok) or (r.direction == "flat" and s.direction == "flat"):
        return None

    if r.direction == "flat" and s.direction == "rising":
        name = "ascending_triangle"
    elif r.direction == "falling" and s.direction == "flat":
        name = "descending_triangle"
    else:
        name = "converging_triangle"

    return ChartPattern(
        name=name,
        detail=(
            f"Resistance trendline {r.direction} ({r.slope_per_bar:+.5f}/bar), support trendline "
            f"{s.direction} ({s.slope_per_bar:+.5f}/bar) — converging, consistent with a {name.replace('_', ' ')}."
        ),
    )


def detect_chart_patterns(
    history: pd.DataFrame,
    lookback: int = STRUCTURE_LOOKBACK,
    swing_points: tuple[list[SwingPoint], list[SwingPoint]] | None = None,
) -> list[ChartPattern]:
    """Conservative, deterministic pattern detection — deliberately
    limited to patterns with an unambiguous, testable definition (double
    top/bottom, trend structure, triangle/wedge via trendline
    convergence). Head-and-shoulders, flags/pennants, and cup-and-handle
    are NOT attempted here: they require subjective judgment calls
    (shoulder symmetry, flagpole proportion) that don't have one
    universally agreed, mechanically checkable definition — a shaky
    detector presented as confident output would be worse than no
    detector at all. Returns only patterns actually found — [] is a
    completely normal result, not a failure.

    `swing_points`, if given, is reused instead of recomputed — see
    compute_fibonacci_levels' own docstring for the full rationale (also
    threaded into the internal compute_trendlines call below, so the
    triangle check doesn't trigger its own separate fractal rescan)."""
    window = _tail_reset(history, lookback)
    if window.empty:
        return []

    swing_highs, swing_lows = swing_points if swing_points is not None else find_swing_points(window)
    patterns: list[ChartPattern] = []

    double_top = _detect_double_pattern(swing_highs, swing_lows, is_top=True, tolerance_pct=SR_CLUSTER_TOLERANCE_PCT)
    if double_top:
        patterns.append(double_top)
    double_bottom = _detect_double_pattern(
        swing_lows, swing_highs, is_top=False, tolerance_pct=SR_CLUSTER_TOLERANCE_PCT
    )
    if double_bottom:
        patterns.append(double_bottom)

    trend_structure = _detect_trend_structure(swing_highs, swing_lows)
    if trend_structure:
        patterns.append(trend_structure)

    trendlines = compute_trendlines(window, lookback=len(window), swing_points=(swing_highs, swing_lows))
    if trendlines is not None:
        triangle = _detect_triangle(trendlines)
        if triangle:
            patterns.append(triangle)

    return patterns


@dataclass
class ChartStructureSnapshot:
    """The four reads above for one symbol on one timeframe, bundled
    together as this module's own public API for "give me everything" —
    callers that want all four (e.g. ai/ftmo_suggest.py, per-timeframe)
    use compute_chart_structure() instead of calling each function
    separately. Every field is independently optional — a symbol can
    have real Fibonacci levels but no confirmed trendline yet, for
    instance — never fabricated to fill a gap."""

    fibonacci: FibonacciLevels | None
    sr_levels: SRLevelsResult | None
    trendlines: TrendlineAnalysis | None
    patterns: list[ChartPattern]


def compute_chart_structure(history: pd.DataFrame, lookback: int = STRUCTURE_LOOKBACK) -> ChartStructureSnapshot:
    """The four reads above, all sharing ONE fractal swing-point scan —
    confirmed live that calling each function separately (as this used
    to) reran find_swing_points on the identical window five separate
    times for one symbol/timeframe (once each for Fibonacci/S/R/
    trendlines, plus twice more inside detect_chart_patterns's own body
    and its internal compute_trendlines call) — real, multiplied-by-
    every-symbol-and-both-timeframes redundant work for no correctness
    benefit."""
    window = _tail_reset(history, lookback)
    swing_points = find_swing_points(window)
    return ChartStructureSnapshot(
        fibonacci=compute_fibonacci_levels(window, lookback=lookback, swing_points=swing_points),
        sr_levels=compute_sr_levels(window, lookback=lookback, swing_points=swing_points),
        trendlines=compute_trendlines(window, lookback=lookback, swing_points=swing_points),
        patterns=detect_chart_patterns(window, lookback=lookback, swing_points=swing_points),
    )


# Cross-timeframe agreement needs a slightly wider tolerance than same-
# timeframe touch clustering (SR_CLUSTER_TOLERANCE_PCT) — two different
# timeframes' own swing points were never going to land on the exact same
# tick the way repeated touches on ONE timeframe's own price series might.
MTF_CONFLUENCE_TOLERANCE_PCT = 0.5


@dataclass
class ConfluenceZone:
    higher_tf_price: float
    lower_tf_price: float
    avg_price: float
    kind: str  # "resistance" (above current price) or "support" (below)
    distance_pct: float  # (avg_price - current_price) / current_price * 100


def find_mtf_confluence(
    higher_tf_levels: list[float],
    lower_tf_levels: list[float],
    current_price: float,
    tolerance_pct: float = MTF_CONFLUENCE_TOLERANCE_PCT,
) -> list[ConfluenceZone]:
    """Real price zones where a level from a HIGHER timeframe and a level
    from a LOWER timeframe independently agree (within `tolerance_pct`) —
    a level two different, independently-computed timeframes both flag
    is genuinely stronger evidence than either alone, the same logic
    same-timeframe touch clustering already applies, extended ACROSS
    timeframes. Callers pass in whatever candidate levels matter to them
    (typically each timeframe's own S/R + Fibonacci prices combined) —
    this function is deliberately generic over plain price lists, not
    tied to any specific pair of timeframes or level source.

    De-duplicates adjacent zones that are themselves within tolerance of
    each other (a real, if uncommon, artifact when several candidate
    levels on one side all cluster near the same small area) so the
    result reads as distinct zones, not near-duplicates of the same one."""
    zones = []
    for higher_price in higher_tf_levels:
        for lower_price in lower_tf_levels:
            avg = (higher_price + lower_price) / 2
            if avg == 0:
                continue
            if abs(higher_price - lower_price) / avg * 100 <= tolerance_pct:
                zones.append(
                    ConfluenceZone(
                        higher_tf_price=higher_price,
                        lower_tf_price=lower_price,
                        avg_price=avg,
                        kind="resistance" if avg > current_price else "support",
                        distance_pct=(avg - current_price) / current_price * 100 if current_price else 0.0,
                    )
                )

    zones.sort(key=lambda z: z.avg_price)
    merged: list[ConfluenceZone] = []
    for zone in zones:
        if merged and merged[-1].avg_price != 0 and (
            abs(zone.avg_price - merged[-1].avg_price) / merged[-1].avg_price * 100 <= tolerance_pct
        ):
            continue
        merged.append(zone)
    return merged
