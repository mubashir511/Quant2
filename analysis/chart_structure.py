from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.technical import VOLUME_TREND_BASELINE_WINDOW, compute_atr
from analysis.timeframe_profiles import TimeframeProfile, resolve_thresholds

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

# ATR-scaled S/R clustering tolerance — added 2026-09-09, real weakness
# found on independent user review: SR_CLUSTER_TOLERANCE_PCT above is ONE
# flat number applied identically to a slow FX pair (H1 atr_pct typically
# 0.03-0.20%, see analysis.technical.VELOCITY_FAST_THRESHOLD_PCT's own
# real comparison) and a genuinely fast mover (metals/equities/crypto,
# 0.3-1.5%+/hour). On a fast instrument, two swing points that are really
# the SAME re-test of one real level can sit further apart in raw %
# terms than 0.3%, so the flat tolerance silently SPLITS one real
# liquidity zone into several separate low-touch "levels" instead of
# correctly recognizing them as repeated tests of the same zone —
# understating exactly the kind of level a pullback entry should be
# targeting. Scaling the tolerance by THIS window's own ATR%, floored at
# the original flat value (never TIGHTER than before — a slow FX pair's
# behavior is unchanged), fixes this without touching slow-instrument
# behavior at all. A principled first cut, not yet validated across many
# more days/instruments — same "ship a value, then recalibrate against
# real data" process analysis.technical.VELOCITY_FAST_THRESHOLD_PCT went
# through before this one.
SR_TOLERANCE_ATR_MULTIPLE = 0.5

# "Equal highs/lows" liquidity-pool tag — added 2026-09-09, direct user
# request to distinguish a real, tight cluster of near-identical swing
# points (the textbook signature of an obvious resting-stop/pending-order
# concentration: multiple traders' own stops or entries sit just beyond
# the same visible double-top/bottom) from a broader structural zone that
# merely has several touches spread across the wider ATR-scaled tolerance
# above. Expressed as a FRACTION of whatever tolerance is actually in use
# for a given window (not a second flat %) so it stays "tight relative to
# THIS instrument's own zone width" on both a slow FX pair and a fast
# mover, rather than reintroducing the exact one-flat-number problem this
# file just fixed for the broader tolerance.
SR_LIQUIDITY_POOL_TOLERANCE_FRACTION = 0.35
SR_LIQUIDITY_POOL_MIN_TOUCHES = 2

# Recency-weighted touch strength — added 2026-09-20, direct user
# challenge after reviewing real trades: raw `touches` treats a touch
# from 90 bars ago identically to one from the last bar, so a level that
# hasn't mattered in weeks can still outrank one price is actively
# respecting right now. Exponential decay: weight = 0.5**(bars_ago /
# SR_RECENCY_HALF_LIFE_BARS) — a touch this many bars old counts for
# half the weight of a fresh one. 30 bars = 1/3 of STRUCTURE_LOOKBACK
# (90): a touch at the very edge of the lookback window contributes
# ~1/8 the weight of one on the latest bar, while a touch from 10 bars
# ago still counts close to full. A principled first cut, not yet
# validated across many more days/instruments — same "ship a value,
# then recalibrate against real data" process SR_TOLERANCE_ATR_MULTIPLE
# and analysis.technical.VELOCITY_FAST_THRESHOLD_PCT already went
# through before being fully trusted at the boundary.
SR_RECENCY_HALF_LIFE_BARS = 30.0

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
    # How many bars back (from the window's own last bar) each swing
    # extreme sits — added 2026-09-04 for ai/chart_overlay.py, which
    # needs REAL time anchors to draw a native MT5 OBJ_FIBO tool
    # correctly rather than guessing; every existing field above is a
    # price only, with no notion of WHEN either extreme occurred.
    # Defaulted to 0 so every caller that already constructs this
    # dataclass without knowing about bar timing (including this
    # project's own test fixtures) keeps working unchanged — only
    # compute_fibonacci_levels itself sets real values.
    swing_high_bars_ago: int = 0
    swing_low_bars_ago: int = 0


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

    last_bar_index = len(window) - 1
    return FibonacciLevels(
        swing_high=recent_high.price,
        swing_low=recent_low.price,
        high_is_more_recent=high_is_more_recent,
        levels=levels,
        current_price=current_price,
        nearest_level_name=nearest_name,
        nearest_level_price=nearest_price,
        distance_to_nearest_pct=(current_price - nearest_price) / current_price * 100 if current_price else 0.0,
        swing_high_bars_ago=last_bar_index - recent_high.index,
        swing_low_bars_ago=last_bar_index - recent_low.index,
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
class _LevelCluster:
    """Internal result of _cluster_swing_points — one real S/R band, not
    yet turned into the public SRLevel (that still needs distance_pct/
    is_liquidity_pool computed against the current price, which this
    function has no reason to know about)."""

    mean_price: float
    low: float
    high: float
    touches: int
    weighted_score: float


def _cluster_swing_points(
    points: list[SwingPoint],
    tolerance_pct: float,
    last_index: int,
    half_life_bars: float = SR_RECENCY_HALF_LIFE_BARS,
) -> list[_LevelCluster]:
    """Same greedy, anchor-to-first-point clustering discipline as
    _cluster_prices (see its own docstring for why a running mean isn't
    used) — duplicated here rather than extending that function in place
    because _cluster_prices has its own, separate, directly-tested
    `list[float] -> (mean, count)` contract (used as-is for the tight
    liquidity-pool sub-pass) that this function's own return shape would
    break. Same "duplicate a small, tested control-flow block rather
    than touch its tested return shape" precedent analysis.backtest.py's
    own _compute_favorable_excursion already sets for _simulate_trades's
    entry/ATR/stop-distance setup.

    Walks real SwingPoint objects (not bare floats) so each resulting
    cluster can report its own real low/high band and a recency-
    weighted strength score, not just a mean and a raw count."""
    if not points:
        return []
    sorted_points = sorted(points, key=lambda p: p.price)
    clusters: list[list[SwingPoint]] = [[sorted_points[0]]]
    for point in sorted_points[1:]:
        anchor = clusters[-1][0].price
        within_tolerance = anchor != 0 and (point.price - anchor) / anchor * 100 <= tolerance_pct
        if within_tolerance:
            clusters[-1].append(point)
        else:
            clusters.append([point])

    result = []
    for cluster in clusters:
        prices = [p.price for p in cluster]
        weighted_score = sum(0.5 ** ((last_index - p.index) / half_life_bars) for p in cluster)
        result.append(
            _LevelCluster(
                mean_price=sum(prices) / len(prices),
                low=min(prices),
                high=max(prices),
                touches=len(cluster),
                weighted_score=weighted_score,
            )
        )
    return result


def atr_scaled_sr_tolerance(window: pd.DataFrame, floor_pct: float = SR_CLUSTER_TOLERANCE_PCT) -> float:
    """The real S/R clustering tolerance to use for THIS window — see
    SR_TOLERANCE_ATR_MULTIPLE's own module-level comment for the full
    reasoning. max(floor_pct, SR_TOLERANCE_ATR_MULTIPLE * this window's
    own atr_pct): never tighter than the original flat convention, only
    ever wider for an instrument whose own bar-to-bar range genuinely
    justifies it. Falls back to floor_pct outright when ATR isn't
    computable (not enough rows, or no High/Low/Close columns) — same
    "never fabricate a stat from missing data" rule as every other
    ATR-derived read in this codebase (analysis.technical.compute_atr,
    ai.curiosity._velocity_tier_for)."""
    atr = compute_atr(window)
    if atr is None or window.empty or "Close" not in window.columns:
        return floor_pct
    current_price = float(window["Close"].iloc[-1])
    if not current_price:
        return floor_pct
    atr_pct = atr / current_price * 100
    return max(floor_pct, SR_TOLERANCE_ATR_MULTIPLE * atr_pct)


@dataclass
class SRLevel:
    price: float
    touches: int  # how many real swing points clustered into this level — a real strength measure, not assumed
    distance_pct: float  # (level - current_price) / current_price * 100 — positive above price, negative below
    # Added 2026-09-09 — see SR_LIQUIDITY_POOL_TOLERANCE_FRACTION's own
    # module-level comment. True when a genuinely TIGHT sub-cluster of
    # SR_LIQUIDITY_POOL_MIN_TOUCHES+ swing points sits within this level's
    # own zone — a real equal-highs/equal-lows signature, distinct from
    # (and stronger evidence than) the broader `touches` count alone.
    # Default False so any existing direct SRLevel(...) construction
    # elsewhere (including this project's own tests) keeps working
    # unchanged.
    is_liquidity_pool: bool = False
    # Real cluster band (min/max of the actual swing prices clustered
    # into this level), not just the mean — added 2026-09-20, direct
    # user challenge: a level was always a single point even though the
    # clustering tolerance that produces it already implies a real band
    # width. None for any SRLevel not built via the new _cluster_swing_
    # points path (including this project's own pre-existing direct
    # constructions and tests) — never a fabricated band around a bare
    # price.
    low: float | None = None
    high: float | None = None
    # Recency-weighted touch strength — see SR_RECENCY_HALF_LIFE_BARS'
    # own module-level comment. A SEPARATE number from `touches` (not a
    # replacement — `touches` stays the raw, honest count it always
    # was), on its own scale, so a caller can compare "how many times"
    # vs "how strongly, weighted by when." None for any SRLevel not
    # built via the new clustering path, same "never fabricate" rule.
    weighted_score: float | None = None


@dataclass
class SRLevelsResult:
    resistance_levels: list[SRLevel]  # above current price, strongest (most touches) first
    support_levels: list[SRLevel]  # below current price, strongest first
    # The REAL tolerance_pct actually used to cluster these levels — added
    # 2026-09-09, real gap found on independent audit: compute_chart_
    # structure() passes an ATR-scaled tolerance (see atr_scaled_sr_
    # tolerance) that can be several times wider than the flat
    # SR_CLUSTER_TOLERANCE_PCT default for a fast mover, but nothing
    # previously told a consumer what value was actually used — ai/
    # chart_overlay.py was found hardcoding the flat module constant when
    # exporting each level's own zone-width to the real MT5 terminal
    # overlay, silently showing a NARROWER zone than what Python actually
    # computed. Defaults to the flat constant so any existing direct
    # SRLevelsResult(...) construction (including this project's own
    # tests) keeps working unchanged.
    tolerance_pct: float = SR_CLUSTER_TOLERANCE_PCT


def compute_sr_levels(
    history: pd.DataFrame,
    lookback: int = STRUCTURE_LOOKBACK,
    tolerance_pct: float = SR_CLUSTER_TOLERANCE_PCT,
    max_levels: int = SR_MAX_LEVELS_PER_SIDE,
    swing_points: tuple[list[SwingPoint], list[SwingPoint]] | None = None,
    liquidity_pool_tolerance_pct: float | None = None,
    half_life_bars: float = SR_RECENCY_HALF_LIFE_BARS,
    rank_by: str = "weighted_score",
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
    compute_fibonacci_levels' own docstring for the full rationale.

    `liquidity_pool_tolerance_pct` (added 2026-09-09, defaults to
    `tolerance_pct * SR_LIQUIDITY_POOL_TOLERANCE_FRACTION`) runs a SECOND,
    tighter clustering pass over the same swing prices to flag each
    resulting level's own `is_liquidity_pool` — see SRLevel's own field
    comment and SR_LIQUIDITY_POOL_TOLERANCE_FRACTION's module-level
    comment for the reasoning.

    `rank_by` (added 2026-09-20, direct user challenge, default
    "weighted_score") decides which of the two SRLevel strength fields
    ranking/truncation to `max_levels` uses — "weighted_score" lets a
    level with fewer but MORE RECENT touches legitimately outrank a
    stale multi-touch one (see SR_RECENCY_HALF_LIFE_BARS' own comment
    for why this matters); "touches" reproduces the original raw-count
    ranking for any caller that explicitly wants it."""
    if rank_by not in ("weighted_score", "touches"):
        raise ValueError(f"rank_by must be 'weighted_score' or 'touches', got {rank_by!r}")
    # Found on self-review — no real caller ever overrides this (always
    # the SR_RECENCY_HALF_LIFE_BARS default), but a zero half-life
    # crashes on division by zero, and a negative one silently INVERTS
    # the decay direction (older touches would outscore recent ones) with
    # no error at all — a real, if latent, footgun for any future caller.
    if half_life_bars <= 0:
        raise ValueError(f"half_life_bars must be positive, got {half_life_bars!r}")
    window = _tail_reset(history, lookback)
    if window.empty or "Close" not in window.columns:
        return None
    current_price = float(window["Close"].iloc[-1])
    last_index = len(window) - 1

    swing_highs, swing_lows = swing_points if swing_points is not None else find_swing_points(window)
    all_points = swing_highs + swing_lows
    if not all_points:
        return None

    if liquidity_pool_tolerance_pct is None:
        liquidity_pool_tolerance_pct = tolerance_pct * SR_LIQUIDITY_POOL_TOLERANCE_FRACTION
    all_prices = [p.price for p in all_points]
    tight_clusters = _cluster_prices(all_prices, liquidity_pool_tolerance_pct)
    liquidity_pool_prices = [price for price, touches in tight_clusters if touches >= SR_LIQUIDITY_POOL_MIN_TOUCHES]

    def _is_liquidity_pool(level_price: float) -> bool:
        return any(
            level_price != 0
            and abs(level_price - pool_price) / level_price * 100 <= liquidity_pool_tolerance_pct
            for pool_price in liquidity_pool_prices
        )

    clusters = _cluster_swing_points(all_points, tolerance_pct, last_index=last_index, half_life_bars=half_life_bars)
    levels = [
        SRLevel(
            price=cluster.mean_price,
            touches=cluster.touches,
            distance_pct=(cluster.mean_price - current_price) / current_price * 100 if current_price else 0.0,
            is_liquidity_pool=_is_liquidity_pool(cluster.mean_price),
            low=cluster.low,
            high=cluster.high,
            weighted_score=cluster.weighted_score,
        )
        for cluster in clusters
    ]

    rank_key = (lambda level: level.weighted_score) if rank_by == "weighted_score" else (lambda level: level.touches)
    resistance = sorted(
        (level for level in levels if level.price > current_price),
        key=lambda level: (-rank_key(level), level.distance_pct),
    )[:max_levels]
    support = sorted(
        (level for level in levels if level.price < current_price),
        key=lambda level: (-rank_key(level), -level.distance_pct),
    )[:max_levels]

    return SRLevelsResult(resistance_levels=resistance, support_levels=support, tolerance_pct=tolerance_pct)


# --- Volume-confirmed breakouts + liquidity sweeps (real vs. fake) ------
# Added 2026-09-20, direct user challenge after reviewing real trades:
# zero volume confirmation existed anywhere in this file's pattern/S/R
# detection — a close beyond a level fired identically whether it came
# on real conviction or on thin, directionless noise. Reuses
# VOLUME_TREND_BASELINE_WINDOW from analysis.technical (the SAME 20-bar
# baseline _compute_volume_trend_pct already uses) rather than a second,
# silently-different "above average volume" definition.
#
# MT5's own `Volume` column for FX/CFDs is real tick-count (quote
# frequency), NOT genuine traded volume — already an accepted limitation
# analysis.technical._compute_volume_trend_pct itself inherits. A real,
# useful proxy for participation, but every field/detail text below says
# "tick-volume-confirmed," never bare "volume-confirmed," to avoid
# overclaiming what this account's own data actually is.

# A breakout bar's own volume must be at least this many times the
# recent 20-bar baseline to count as CONFIRMED — Bulkowski/Murphy's own
# qualitative "above-average volume" breakout requirement (already cited
# in data/book_wisdom.py), turned into one concrete number. A principled
# first cut, not yet recalibrated against real MT5 history — same
# disclosed-placeholder status as SR_TOLERANCE_ATR_MULTIPLE above.
BREAKOUT_VOLUME_CONFIRM_RATIO = 1.3

# Minimum real close-through distance past a level's own band edge to
# count as a breakout AT ALL, not noise sitting almost exactly on the
# line.
BREAKOUT_MIN_CLOSE_THROUGH_PCT = 0.1

# A wick through a level's band that closes back inside within this many
# bars counts as a genuine liquidity sweep/stop-hunt — a real breakout
# that fails much later is a different, slower phenomenon this codebase
# already covers via setup_classifier.py's own busted_pattern_reversal.
SWEEP_MAX_BARS_TO_CLOSE_BACK = 3


@dataclass
class BreakoutEvent:
    level_price: float
    direction: str  # "up" (broke resistance) / "down" (broke support)
    close_through_pct: float  # signed % the close sits beyond the level's own band edge, in the breakout direction
    volume_ratio: float | None  # this bar's volume / recent 20-bar baseline average; None if no real Volume column
    volume_confirmed: bool  # True only when volume_ratio is real AND >= BREAKOUT_VOLUME_CONFIRM_RATIO
    bars_ago: int


@dataclass
class LiquiditySweepEvent:
    level_price: float
    direction: str  # "swept_above" (wicked above resistance, closed back below) / "swept_below" (mirror, support)
    wick_penetration_pct: float  # how far the wick went past the level's own band edge, %
    bars_ago: int  # how many bars ago the sweep's own wick bar occurred
    volume_ratio: float | None  # informational — a sweep on above-average tick-volume is a stronger stop-hunt signature


def _volume_ratio_at(volume: pd.Series | None, bar_pos: int) -> float | None:
    """This bar's real volume divided by the mean of the
    VOLUME_TREND_BASELINE_WINDOW bars strictly BEFORE it — never
    including the bar itself, so a genuinely huge breakout bar can't
    dilute its own baseline. None (never fabricated) when there's no
    real Volume column, or not enough prior history to form a baseline."""
    if volume is None or bar_pos < VOLUME_TREND_BASELINE_WINDOW:
        return None
    baseline = volume.iloc[bar_pos - VOLUME_TREND_BASELINE_WINDOW : bar_pos].mean()
    if not baseline:
        return None
    return float(volume.iloc[bar_pos]) / float(baseline)


def detect_breakouts(
    history: pd.DataFrame,
    sr_levels: SRLevelsResult | None,
    lookback_bars: int = 5,
    min_close_through_pct: float = BREAKOUT_MIN_CLOSE_THROUGH_PCT,
) -> list[BreakoutEvent]:
    """Real closes beyond an S/R level's own band edge (Phase 1's
    SRLevel.low/.high — not the bare mean price) within the last
    `lookback_bars` bars, each tagged with real tick-volume confirmation.
    Reports the geometric fact (a real close-through happened) even when
    volume_confirmed is False — never silently drops a real breakout
    just because it lacked volume support; the caller decides how much
    weight to give an unconfirmed one. [] (never fabricated) when there's
    no real S/R structure to test against, or High/Low/Close is missing."""
    if sr_levels is None or not {"High", "Low", "Close"}.issubset(history.columns) or history.empty:
        return []
    closes = history["Close"].reset_index(drop=True)
    volume = history["Volume"].reset_index(drop=True) if "Volume" in history.columns else None
    last_pos = len(closes) - 1
    events: list[BreakoutEvent] = []

    sided_levels = [(level, True) for level in sr_levels.resistance_levels] + [
        (level, False) for level in sr_levels.support_levels
    ]
    for level, is_resistance in sided_levels:
        if is_resistance:
            band_edge = level.high if level.high is not None else level.price
        else:
            band_edge = level.low if level.low is not None else level.price
        if band_edge == 0:
            continue

        def _close_through_pct(bar_pos: int) -> float:
            close = float(closes.iloc[bar_pos])
            return (close - band_edge) / band_edge * 100 if is_resistance else (band_edge - close) / band_edge * 100

        # Real bug found on self-review, 2026-09-21: walking backward and
        # stopping at the FIRST (i.e. today's) qualifying bar meant a
        # SUSTAINED breakout — price has closed beyond the level for
        # several bars running — always reported bars_ago=0 using
        # TODAY's volume, silently discarding the ORIGINAL breakout bar's
        # own real volume signature (e.g. a genuine 5x-volume breakout 3
        # bars ago would be reported as "not volume-confirmed" just
        # because today's own volume happens to be ordinary). Require
        # today's close to still clear the threshold, then walk backward
        # to find where this streak actually began, and report THAT
        # bar's own close-through/volume — the real breakout bar.
        if _close_through_pct(last_pos) < min_close_through_pct:
            continue
        breakout_bar_pos = last_pos
        min_bar_pos = max(0, last_pos - min(lookback_bars, len(closes)) + 1)
        for bar_pos in range(last_pos - 1, min_bar_pos - 1, -1):
            if _close_through_pct(bar_pos) < min_close_through_pct:
                break
            breakout_bar_pos = bar_pos

        volume_ratio = _volume_ratio_at(volume, breakout_bar_pos)
        events.append(
            BreakoutEvent(
                level_price=level.price,
                direction="up" if is_resistance else "down",
                close_through_pct=_close_through_pct(breakout_bar_pos),
                volume_ratio=volume_ratio,
                volume_confirmed=volume_ratio is not None and volume_ratio >= BREAKOUT_VOLUME_CONFIRM_RATIO,
                bars_ago=last_pos - breakout_bar_pos,
            )
        )

    return events


def detect_liquidity_sweeps(
    history: pd.DataFrame,
    sr_levels: SRLevelsResult | None,
    lookback_bars: int = SWEEP_MAX_BARS_TO_CLOSE_BACK + 2,
) -> list[LiquiditySweepEvent]:
    """A real wick through an S/R level's own band edge that CLOSES BACK
    INSIDE within SWEEP_MAX_BARS_TO_CLOSE_BACK bars — the classic
    liquidity-sweep/stop-hunt signature price-action traders watch for:
    a level gets taken out just far enough to trigger resting stops/
    orders, then reverses. Deliberately distinct from a real breakout
    that later fails much slower (already covered by setup_classifier.
    py's own busted_pattern_reversal) — this only fires on a FAST,
    few-bar round trip. [] (never fabricated) when there's no real S/R
    structure to test against, or High/Low/Close is missing."""
    if sr_levels is None or not {"High", "Low", "Close"}.issubset(history.columns):
        return []
    highs = history["High"].reset_index(drop=True)
    lows = history["Low"].reset_index(drop=True)
    closes = history["Close"].reset_index(drop=True)
    volume = history["Volume"].reset_index(drop=True) if "Volume" in history.columns else None
    last_pos = len(closes) - 1
    events: list[LiquiditySweepEvent] = []

    sided_levels = [(level, True) for level in sr_levels.resistance_levels] + [
        (level, False) for level in sr_levels.support_levels
    ]
    for level, is_resistance in sided_levels:
        band_edge = (level.high if level.high is not None else level.price) if is_resistance else (
            level.low if level.low is not None else level.price
        )
        if band_edge == 0:
            continue
        for offset in range(min(lookback_bars, len(closes))):
            wick_pos = last_pos - offset
            if wick_pos < 0:
                break
            wicked_through = (
                float(highs.iloc[wick_pos]) > band_edge if is_resistance else float(lows.iloc[wick_pos]) < band_edge
            )
            if not wicked_through:
                continue
            # Closed back inside on the SAME bar (k=0, a single-bar spike
            # wick — the most common real sweep shape) or on any of the
            # next SWEEP_MAX_BARS_TO_CLOSE_BACK bars — only need ONE real
            # close back inside within the window, not every bar in it
            # (a later, UNRELATED wick breaking back out again a few bars
            # on shouldn't retroactively disqualify an earlier, already-
            # genuine reversal).
            closed_back_inside_within_window = any(
                (float(closes.iloc[wick_pos + k]) < band_edge if is_resistance else float(closes.iloc[wick_pos + k]) > band_edge)
                for k in range(0, SWEEP_MAX_BARS_TO_CLOSE_BACK + 1)
                if wick_pos + k <= last_pos
            )
            if not closed_back_inside_within_window:
                continue
            extreme = float(highs.iloc[wick_pos]) if is_resistance else float(lows.iloc[wick_pos])
            wick_penetration_pct = abs(extreme - band_edge) / band_edge * 100
            events.append(
                LiquiditySweepEvent(
                    level_price=level.price,
                    direction="swept_above" if is_resistance else "swept_below",
                    wick_penetration_pct=wick_penetration_pct,
                    bars_ago=offset,
                    volume_ratio=_volume_ratio_at(volume, wick_pos),
                )
            )
            break  # only the most recent qualifying sweep per level

    return events


@dataclass
class Trendline:
    slope_per_bar: float
    direction: str  # "rising" / "falling" / "flat"
    current_price_on_line: float  # the fitted line's value extrapolated to the LATEST bar
    price_vs_line: str  # "above" / "below" / "at"
    distance_pct: float  # (current_close - current_price_on_line) / current_close * 100
    bars_since_last_point: int  # how far the line's own rightmost fitted point sits from the latest bar


def _fit_trendline(
    points: list[SwingPoint],
    last_index: int,
    current_close: float,
    flat_threshold_pct: float = TRENDLINE_FLAT_THRESHOLD_PCT,
) -> Trendline | None:
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
    if implied_pct_move > flat_threshold_pct:
        direction = "rising"
    elif implied_pct_move < -flat_threshold_pct:
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
    flat_threshold_pct: float = TRENDLINE_FLAT_THRESHOLD_PCT,
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
    resistance = _fit_trendline(swing_highs, last_index, current_close, flat_threshold_pct)
    support = _fit_trendline(swing_lows, last_index, current_close, flat_threshold_pct)
    if resistance is None and support is None:
        return None
    return TrendlineAnalysis(resistance_trendline=resistance, support_trendline=support)


@dataclass
class ChartPattern:
    name: str  # "double_top" / "double_bottom" / "uptrend_structure" / "downtrend_structure" /
    #             "ascending_triangle" / "descending_triangle" / "converging_triangle" /
    #             (candlestick, see analysis/candlestick_patterns.py) "doji" / "bullish_engulfing" /
    #             "bearish_engulfing" / "hammer" / "hanging_man" / "shooting_star" /
    #             "inverted_hammer" / "morning_star" / "evening_star"
    detail: str  # one plain-language sentence with the real numbers behind the call


def _detect_double_pattern(
    extreme_points: list[SwingPoint],
    opposing_points: list[SwingPoint],
    is_top: bool,
    tolerance_pct: float,
    min_pullback_pct: float = DOUBLE_PATTERN_MIN_PULLBACK_PCT,
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
    if pullback_pct < min_pullback_pct:
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
    tolerance_pct: float = SR_CLUSTER_TOLERANCE_PCT,
    min_pullback_pct: float = DOUBLE_PATTERN_MIN_PULLBACK_PCT,
    trendline_flat_pct: float = TRENDLINE_FLAT_THRESHOLD_PCT,
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

    double_top = _detect_double_pattern(
        swing_highs, swing_lows, is_top=True, tolerance_pct=tolerance_pct, min_pullback_pct=min_pullback_pct
    )
    if double_top:
        patterns.append(double_top)
    double_bottom = _detect_double_pattern(
        swing_lows, swing_highs, is_top=False, tolerance_pct=tolerance_pct, min_pullback_pct=min_pullback_pct
    )
    if double_bottom:
        patterns.append(double_bottom)

    trend_structure = _detect_trend_structure(swing_highs, swing_lows)
    if trend_structure:
        patterns.append(trend_structure)

    trendlines = compute_trendlines(
        window, lookback=len(window), swing_points=(swing_highs, swing_lows), flat_threshold_pct=trendline_flat_pct
    )
    if trendlines is not None:
        triangle = _detect_triangle(trendlines)
        if triangle:
            patterns.append(triangle)

    return patterns


# --- Break of structure (BOS) / change of character (CHOCH) -------------
# Added 2026-09-20, direct user challenge after reviewing real trades:
# telling a genuine trend from an early reversal was left entirely to
# the model, with no mechanical help beyond the lagging regime/momentum
# reads in analysis/technical.py. This is real Dow-theory structure-
# break logic (Murphy's own reversal criteria: a trend isn't confirmed
# broken until price actually takes out a real prior swing point against
# it) — a pure SEQUENCING/COMPARISON pass over data find_swing_points and
# _detect_trend_structure already compute, not a new detection method.
STRUCTURE_BREAK_LOOKBACK_BARS = 10


@dataclass
class StructureBreak:
    kind: str  # "BOS" (break of structure, WITH the prevailing trend) or "CHOCH" (change of character, first break AGAINST it)
    direction: str  # "bullish" (closed above a prior swing high) / "bearish" (closed below a prior swing low)
    broken_level: float  # the real prior swing high/low price that was broken
    break_price: float  # the real close that confirmed the break
    bars_ago: int


def detect_structure_breaks(
    history: pd.DataFrame,
    swing_points: tuple[list[SwingPoint], list[SwingPoint]] | None = None,
    lookback_bars: int = STRUCTURE_BREAK_LOOKBACK_BARS,
) -> list[StructureBreak]:
    """Real breaks of the most recent CONFIRMED swing high/low, each
    labeled BOS or CHOCH using the SAME trend-structure read
    _detect_trend_structure already computes (reused, not re-derived a
    second, possibly-inconsistent way) — a break in the prevailing
    trend's own direction is a BOS (continuation); a break AGAINST an
    established trend is a CHOCH (the first real structural evidence a
    reversal may be starting, not just a lagging regime/momentum read
    catching up later). At most one bullish and one bearish break is
    reported (the most recent qualifying one each) — a real close beyond
    the SAME swing point on multiple later bars is still one structural
    fact, not a new one each bar. [] (never fabricated) without at least
    2 confirmed swing highs AND 2 confirmed swing lows (the same
    precondition _detect_trend_structure itself already requires to
    judge trend direction at all).

    Unlike compute_sr_levels/compute_trendlines/detect_chart_patterns,
    this takes `history` AS THE WINDOW already (no separate `lookback`
    slicing param) — every caller here already has a `window` in hand
    (compute_chart_structure) with `swing_points` computed against that
    SAME window, and re-slicing here independently would desynchronize
    swing_points' own 0-based indices from a freshly re-sliced frame."""
    window = history.reset_index(drop=True)
    if window.empty or "Close" not in window.columns:
        return []
    swing_highs, swing_lows = swing_points if swing_points is not None else find_swing_points(window)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return []

    trend = _detect_trend_structure(swing_highs, swing_lows)
    trend_name = trend.name if trend is not None else None  # "uptrend_structure" / "downtrend_structure" / None

    closes = window["Close"].reset_index(drop=True)
    last_pos = len(closes) - 1
    last_high = max(swing_highs, key=lambda p: p.index)
    last_low = max(swing_lows, key=lambda p: p.index)

    events: list[StructureBreak] = []
    for level, direction in ((last_high, "bullish"), (last_low, "bearish")):
        min_bar_pos = max(level.index + 1, last_pos - min(lookback_bars, len(closes)) + 1)
        if min_bar_pos > last_pos:
            continue  # no real bar exists after this swing point's own formation, within the window

        def _broke(bar_pos: int) -> bool:
            close = float(closes.iloc[bar_pos])
            return close > level.price if direction == "bullish" else close < level.price

        if not _broke(last_pos):
            continue  # not currently broken -- a since-reverted wick-through isn't a lasting structure break (see detect_liquidity_sweeps for that case)

        # Real bug found on self-review, 2026-09-21: reporting bars_ago
        # from wherever the scan first found ANY broken bar (starting
        # from today) meant a SUSTAINED break — price closes beyond the
        # level today, and has for many bars running — always reported
        # bars_ago=0, since today's own close always satisfies "broke".
        # That silently misrepresented an old, already-known structural
        # fact as brand new every single time this was recomputed (e.g.
        # every Clerk tactical poll). Walk backward from today instead to
        # find the actual TRANSITION bar — the most recent point price
        # crossed INTO its current broken state — and report bars_ago
        # relative to that, bounded by the same lookback window.
        transition_pos = last_pos
        for bar_pos in range(last_pos - 1, min_bar_pos - 1, -1):
            if not _broke(bar_pos):
                break
            transition_pos = bar_pos

        # No established trend at all (trend_name is None) defaults
        # to BOS, not CHOCH — CHOCH specifically means a break AGAINST
        # an established trend; there's nothing to "change" from yet
        # if no trend structure was confirmed in the first place.
        against_trend = (direction == "bullish" and trend_name == "downtrend_structure") or (
            direction == "bearish" and trend_name == "uptrend_structure"
        )
        kind = "CHOCH" if against_trend else "BOS"
        events.append(
            StructureBreak(
                kind=kind, direction=direction, broken_level=level.price,
                break_price=float(closes.iloc[transition_pos]), bars_ago=last_pos - transition_pos,
            )
        )

    return events


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
    # Real vs. fake — added 2026-09-20, direct user challenge. Defaulted
    # to empty lists (never None) so any existing direct
    # ChartStructureSnapshot(...) construction (including this project's
    # own pre-Phase-2 tests) keeps working unchanged.
    breakouts: list[BreakoutEvent] = field(default_factory=list)
    liquidity_sweeps: list[LiquiditySweepEvent] = field(default_factory=list)
    # Trend vs. reversal — added 2026-09-20, direct user challenge. Same
    # additive/defaulted-empty convention as breakouts/liquidity_sweeps
    # above.
    structure_breaks: list[StructureBreak] = field(default_factory=list)


def compute_chart_structure(
    history: pd.DataFrame, lookback: int = STRUCTURE_LOOKBACK, profile: TimeframeProfile | None = None
) -> ChartStructureSnapshot:
    """The four reads above, all sharing ONE fractal swing-point scan —
    confirmed live that calling each function separately (as this used
    to) reran find_swing_points on the identical window five separate
    times for one symbol/timeframe (once each for Fibonacci/S/R/
    trendlines, plus twice more inside detect_chart_patterns's own body
    and its internal compute_trendlines call) — real, multiplied-by-
    every-symbol-and-both-timeframes redundant work for no correctness
    benefit.

    Candlestick patterns (analysis/candlestick_patterns.py, added
    2026-08-26) are concatenated into the same `patterns` list —
    imported here, inside the function body rather than at module top,
    specifically to avoid a circular import: that module imports
    `ChartPattern` FROM this one, so importing it back at this file's
    own top level (before `ChartPattern` is even defined) would fail."""
    from analysis.candlestick_patterns import detect_candlestick_patterns

    # `profile` (intraday decision-tier upgrade, 2026-09-24): None keeps
    # every pre-existing flat threshold exactly; an M5/M15 profile swaps
    # the lookback and expresses the flat-percent thresholds as multiples
    # of this window's own ATR% (see analysis/timeframe_profiles.py).
    if profile is not None:
        lookback = profile.structure_lookback
    window = _tail_reset(history, lookback)
    swing_points = find_swing_points(window)
    atr_for_profile = compute_atr(window)
    window_close = float(window["Close"].iloc[-1]) if not window.empty and "Close" in window.columns else 0.0
    window_atr_pct = (atr_for_profile / window_close * 100) if atr_for_profile is not None and window_close else None
    if profile is not None:
        thresholds = resolve_thresholds(
            profile,
            window_atr_pct,
            default_trendline_flat_pct=TRENDLINE_FLAT_THRESHOLD_PCT,
            default_double_pullback_pct=DOUBLE_PATTERN_MIN_PULLBACK_PCT,
            default_breakout_close_through_pct=BREAKOUT_MIN_CLOSE_THROUGH_PCT,
        )
        sr_floor_pct = profile.sr_tolerance_floor_pct
        half_life_bars = profile.sr_half_life_bars
    else:
        thresholds = None
        sr_floor_pct = SR_CLUSTER_TOLERANCE_PCT
        half_life_bars = SR_RECENCY_HALF_LIFE_BARS
    # ATR-scaled, not the flat SR_CLUSTER_TOLERANCE_PCT default — see
    # SR_TOLERANCE_ATR_MULTIPLE's own module-level comment. Computed once
    # here and threaded into compute_sr_levels below, same "share one
    # real computation across this bundling function" discipline this
    # function already applies to the swing-point scan itself.
    sr_tolerance_pct = atr_scaled_sr_tolerance(window, floor_pct=sr_floor_pct)
    sr_levels = compute_sr_levels(
        window, lookback=lookback, swing_points=swing_points, tolerance_pct=sr_tolerance_pct, half_life_bars=half_life_bars
    )
    return ChartStructureSnapshot(
        fibonacci=compute_fibonacci_levels(window, lookback=lookback, swing_points=swing_points),
        sr_levels=sr_levels,
        trendlines=compute_trendlines(
            window, lookback=lookback, swing_points=swing_points,
            **({"flat_threshold_pct": thresholds.trendline_flat_pct} if thresholds else {}),
        ),
        patterns=(
            detect_chart_patterns(
                window, lookback=lookback, swing_points=swing_points,
                **(
                    {
                        "tolerance_pct": sr_tolerance_pct,
                        "min_pullback_pct": thresholds.double_pullback_pct,
                        "trendline_flat_pct": thresholds.trendline_flat_pct,
                    }
                    if thresholds
                    else {}
                ),
            )
            + detect_candlestick_patterns(window)
        ),
        # Real vs. fake — reuses the SAME sr_levels just computed above,
        # not a second recomputation.
        breakouts=detect_breakouts(
            window, sr_levels,
            **({"min_close_through_pct": thresholds.breakout_close_through_pct} if thresholds else {}),
        ),
        liquidity_sweeps=detect_liquidity_sweeps(window, sr_levels),
        # Trend vs. reversal — reuses the SAME swing_points just computed
        # above, not a second fractal rescan.
        structure_breaks=detect_structure_breaks(window, swing_points=swing_points),
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
