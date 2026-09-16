import pandas as pd
import pytest

from analysis.chart_structure import (
    SR_CLUSTER_TOLERANCE_PCT,
    SR_TOLERANCE_ATR_MULTIPLE,
    _cluster_prices,
    atr_scaled_sr_tolerance,
    compute_chart_structure,
    compute_fibonacci_levels,
    compute_sr_levels,
    compute_trendlines,
    detect_chart_patterns,
    find_mtf_confluence,
    find_swing_points,
)


def _zigzag(extrema: list[float], seg_len: int) -> list[float]:
    """Piecewise-linear path through `extrema`, `seg_len` bars per leg —
    each value in `extrema` lands at EXACTLY bar index i*seg_len, giving
    full control over where swing points end up so test expectations can
    be computed by hand (or, here, cross-checked live once and then
    encoded), not guessed at."""
    prices: list[float] = []
    for i in range(len(extrema) - 1):
        start, end = extrema[i], extrema[i + 1]
        for step in range(seg_len):
            prices.append(start + (end - start) * step / seg_len)
    prices.append(extrema[-1])
    return prices


def _make_history(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(closes), freq="h")
    close = pd.Series(closes, index=index)
    return pd.DataFrame(
        {"Open": close, "High": close + 0.001, "Low": close - 0.001, "Close": close, "Volume": [100.0] * len(close)},
        index=index,
    )


# --- find_swing_points ---


def test_find_swing_points_locates_a_clean_single_peak_and_trough():
    history = _make_history(_zigzag([100, 150, 90, 140], 10))
    swing_highs, swing_lows = find_swing_points(history)

    assert len(swing_highs) == 1
    assert swing_highs[0].index == 10
    assert swing_highs[0].price == pytest.approx(150.001)
    assert len(swing_lows) == 1
    assert swing_lows[0].index == 20
    assert swing_lows[0].price == pytest.approx(89.999)


def test_find_swing_points_empty_without_high_low_columns():
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    assert find_swing_points(df) == ([], [])


def test_find_swing_points_empty_when_too_few_rows():
    history = _make_history(_zigzag([100, 150], 1))  # only 2 rows
    assert find_swing_points(history, window=2) == ([], [])


def test_find_swing_points_does_not_register_a_tied_flat_top():
    # Two adjacent bars share the exact same (unique-within-window) max —
    # neither should register, since neither is the SOLE local extremum.
    closes = [100.0, 101.0, 105.0, 105.0, 101.0, 100.0]
    history = _make_history(closes)
    swing_highs, _ = find_swing_points(history, window=2)
    assert swing_highs == []


# --- compute_fibonacci_levels ---


def test_fibonacci_retraces_up_from_a_confirmed_low_toward_a_confirmed_high():
    # Confirmed swing high (150) at index 10, confirmed swing low (90) at
    # index 20 — the LOW is more recent, so retracement is measured UP
    # from the low (0%) toward the high (100%), the standard convention
    # for the last real move being a decline.
    history = _make_history(_zigzag([100, 150, 90, 140], 10))
    fib = compute_fibonacci_levels(history)

    assert fib is not None
    assert fib.high_is_more_recent is False
    assert fib.levels["0.0%"] == pytest.approx(89.999)
    assert fib.levels["50.0%"] == pytest.approx(120.0, abs=0.01)
    assert fib.levels["100.0%"] == pytest.approx(150.001)
    # current_price is the series' last close (140.0, the final zigzag point).
    assert fib.current_price == pytest.approx(140.0)
    assert fib.nearest_level_name == "78.6%"
    assert fib.nearest_level_price == pytest.approx(137.16, abs=0.01)


def test_fibonacci_falls_back_to_raw_range_without_a_confirmed_swing_on_both_sides():
    # A pure, single monotonic rise never reverses — no confirmed swing
    # low exists at all, so this falls back to the window's own raw
    # High/Low rather than returning None just because no fractal has
    # confirmed yet.
    history = _make_history(_zigzag([100, 150], 20))
    fib = compute_fibonacci_levels(history)

    assert fib is not None
    assert fib.swing_low == pytest.approx(99.999)
    assert fib.swing_high == pytest.approx(150.001)
    # The high sits at the very last bar (the raw idxmax), so it's "more
    # recent" than the low at the very first bar.
    assert fib.high_is_more_recent is True
    assert fib.nearest_level_name == "0.0%"  # current price IS the high


def test_fibonacci_none_on_empty_history():
    assert compute_fibonacci_levels(pd.DataFrame(columns=["Close"])) is None


def test_fibonacci_none_when_swing_high_equals_swing_low():
    # A perfectly flat instrument (no High/Low offset at all) — the raw
    # High and Low fallback would be numerically identical, a degenerate
    # "range" with nothing real to retrace.
    index = pd.date_range("2026-01-01", periods=10, freq="h")
    flat = pd.Series([100.0] * 10, index=index)
    history = pd.DataFrame(
        {"Open": flat, "High": flat, "Low": flat, "Close": flat, "Volume": [100.0] * 10}, index=index
    )
    assert compute_fibonacci_levels(history) is None


# --- _cluster_prices ---


def test_cluster_prices_never_lets_a_cluster_drift_beyond_tolerance():
    # Regression: a chain of small, individually-in-tolerance steps used
    # to "walk" one cluster's total span well past the stated tolerance
    # when anchored to a shifting running mean (confirmed live: ~0.5%
    # total span under a nominal 0.3% tolerance). Anchoring to each
    # cluster's fixed first point must keep every cluster's real span
    # within tolerance_pct, no matter how many points chain together.
    prices = [100.0]
    p = 100.0
    for _ in range(40):
        p = p * 1.0005  # 0.05% steps — individually well within a 0.3% tolerance
        prices.append(p)

    clusters = _cluster_prices(prices, tolerance_pct=0.3)

    # Re-derive each cluster's real min/max from the original prices list
    # in the same sorted order _cluster_prices itself walks, to check the
    # true span rather than trusting the reported mean alone.
    sorted_prices = sorted(prices)
    idx = 0
    for mean, count in clusters:
        members = sorted_prices[idx : idx + count]
        span_pct = (max(members) - min(members)) / min(members) * 100
        assert span_pct <= 0.3 + 1e-9
        idx += count


def test_cluster_prices_groups_tight_touches_and_separates_wide_ones():
    clusters = _cluster_prices([100.0, 100.1, 100.2, 110.0], tolerance_pct=0.3)
    assert len(clusters) == 2
    tight = next(c for c in clusters if c[1] == 3)
    assert tight[0] == pytest.approx(100.1, abs=0.05)


def test_cluster_prices_empty_input():
    assert _cluster_prices([], tolerance_pct=0.3) == []


# --- compute_sr_levels ---


def test_sr_levels_clusters_multiple_real_touches_into_one_level_each():
    # Three swing highs clustered near 150 (well within SR_CLUSTER_TOLERANCE_PCT
    # of each other), two swing lows clustered near 100.
    history = _make_history(_zigzag([120, 150.0, 100.0, 150.1, 100.05, 149.95, 105], 8))
    sr = compute_sr_levels(history)

    assert sr is not None
    assert len(sr.resistance_levels) == 1
    assert sr.resistance_levels[0].touches == 3
    assert sr.resistance_levels[0].price == pytest.approx(150.02, abs=0.05)
    assert len(sr.support_levels) == 1
    assert sr.support_levels[0].touches == 2
    assert sr.support_levels[0].price == pytest.approx(100.02, abs=0.05)


def test_sr_levels_none_without_any_swing_structure():
    history = _make_history(_zigzag([100, 150], 20))  # one pure monotonic rise, no reversal
    assert compute_sr_levels(history) is None


def test_sr_levels_touches_beat_a_higher_untested_level_for_ranking():
    # A well-tested level (3 touches, farther from price) should rank
    # ahead of a once-touched level (1 touch, nearer to price) — real
    # evidence of a level mattering beats mere proximity.
    history = _make_history(_zigzag([120, 150.0, 100.0, 150.1, 100.05, 149.95, 160, 105], 8))
    sr = compute_sr_levels(history)
    assert sr.resistance_levels[0].touches >= sr.resistance_levels[-1].touches


def test_sr_levels_result_echoes_back_the_real_tolerance_actually_used():
    # Real bug found on independent audit, fixed 2026-09-10: downstream
    # consumers (ai/chart_overlay.py) had no way to know what tolerance
    # was actually used to cluster a given result — SRLevelsResult must
    # carry it, not assume the flat SR_CLUSTER_TOLERANCE_PCT default.
    history = _make_history(_zigzag([120, 150.0, 100.0, 150.1, 100.05, 149.95, 105], 8))
    default = compute_sr_levels(history)
    assert default.tolerance_pct == pytest.approx(SR_CLUSTER_TOLERANCE_PCT)
    scaled = compute_sr_levels(history, tolerance_pct=0.9)
    assert scaled.tolerance_pct == pytest.approx(0.9)


# --- atr_scaled_sr_tolerance (added 2026-09-09) ---


def test_atr_scaled_sr_tolerance_stays_at_the_floor_for_a_genuinely_calm_instrument():
    # A tiny, steady drift gives a real but small ATR% -- well under the
    # flat floor -- so the ATR-scaled tolerance must never come back
    # TIGHTER than the original flat convention.
    closes = [100.0 + i * 0.01 for i in range(20)]
    history = _make_history(closes)
    assert atr_scaled_sr_tolerance(history) == pytest.approx(SR_CLUSTER_TOLERANCE_PCT)


def test_atr_scaled_sr_tolerance_widens_for_a_genuinely_fast_mover():
    # A real, large bar-to-bar range (~2% of price) must produce a WIDER
    # tolerance than the flat floor -- the whole point of this function,
    # real incident it targets: a flat 0.3% tolerance splitting one real
    # liquidity zone on a fast mover into several separate low-touch
    # "levels" (see SR_TOLERANCE_ATR_MULTIPLE's own module comment).
    closes = [100.0 + (i % 2) * 2.0 for i in range(20)]  # oscillates 100.0/102.0 every bar
    history = _make_history(closes)
    tolerance = atr_scaled_sr_tolerance(history)
    assert tolerance > SR_CLUSTER_TOLERANCE_PCT
    assert tolerance == pytest.approx(0.98, abs=0.15)  # ~0.5 * ~1.96% atr_pct


def test_atr_scaled_sr_tolerance_floor_when_atr_unavailable():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    assert atr_scaled_sr_tolerance(empty) == SR_CLUSTER_TOLERANCE_PCT


def test_compute_chart_structure_wires_atr_scaled_tolerance_into_sr_levels():
    # Regression: sr_tolerance_pct must actually be THREADED into
    # compute_sr_levels' own call inside compute_chart_structure, not
    # just computed and discarded.
    import analysis.chart_structure as chart_structure_module

    history = _make_history(_zigzag([105, 100, 120, 110, 150, 133], 8))
    captured: dict = {}
    real_compute_sr_levels = chart_structure_module.compute_sr_levels

    def _capturing_compute_sr_levels(*args, **kwargs):
        captured["tolerance_pct"] = kwargs.get("tolerance_pct")
        return real_compute_sr_levels(*args, **kwargs)

    chart_structure_module.compute_sr_levels = _capturing_compute_sr_levels
    try:
        chart_structure_module.compute_chart_structure(history)
    finally:
        chart_structure_module.compute_sr_levels = real_compute_sr_levels

    assert captured["tolerance_pct"] == pytest.approx(atr_scaled_sr_tolerance(history))


# --- SRLevel.is_liquidity_pool (equal highs/lows, added 2026-09-09) ---


def test_sr_levels_flags_a_tight_equal_highs_cluster_as_a_liquidity_pool():
    from analysis.chart_structure import SwingPoint

    # Two swing highs sitting almost exactly on top of each other (a real
    # equal-highs signature) plus a third, more loosely-spread high that
    # clears the BROAD 0.3% tolerance (so all three still merge into one
    # reported level) but NOT the tighter liquidity-pool sub-tolerance
    # (0.3% * SR_LIQUIDITY_POOL_TOLERANCE_FRACTION=0.35 = 0.105%) on its
    # own — the tight 200.0/200.02 pair alone must still be enough to
    # flag the whole merged level as a liquidity pool.
    history = _make_history([150.0] * 5)
    swing_highs = [
        SwingPoint(index=0, price=200.0),
        SwingPoint(index=1, price=200.02),  # 0.01% from 200.0 -- a real equal-high
        SwingPoint(index=2, price=200.5),   # 0.25% from 200.0 -- broad-tolerance touch only
    ]
    swing_lows = [SwingPoint(index=3, price=100.0)]  # single touch -- never a liquidity pool alone

    sr = compute_sr_levels(history, swing_points=(swing_highs, swing_lows))

    assert sr is not None
    assert len(sr.resistance_levels) == 1
    assert sr.resistance_levels[0].touches == 3
    assert sr.resistance_levels[0].is_liquidity_pool is True

    assert len(sr.support_levels) == 1
    assert sr.support_levels[0].touches == 1
    assert sr.support_levels[0].is_liquidity_pool is False


def test_sr_levels_no_liquidity_pool_when_touches_are_genuinely_spread_out():
    from analysis.chart_structure import SwingPoint

    # Three swing highs each ~0.15% apart from their neighbor -- close
    # enough to merge under the broad 0.3% tolerance, but no PAIR of them
    # is within the tight liquidity-pool sub-tolerance of each other.
    history = _make_history([150.0] * 5)
    swing_highs = [
        SwingPoint(index=0, price=200.0),
        SwingPoint(index=1, price=200.3),   # 0.15% from 200.0
        SwingPoint(index=2, price=200.6),   # 0.15% from 200.3, 0.30% from 200.0
    ]
    swing_lows = [SwingPoint(index=3, price=100.0)]  # single touch -- never a liquidity pool alone

    sr = compute_sr_levels(history, swing_points=(swing_highs, swing_lows))

    assert sr is not None
    assert sr.resistance_levels[0].touches == 3
    assert sr.resistance_levels[0].is_liquidity_pool is False
    assert sr.support_levels[0].is_liquidity_pool is False


# --- compute_trendlines ---


def test_trendlines_none_side_with_fewer_than_min_points_stays_none():
    # Only 2 confirmed swing lows (TRENDLINE_MIN_POINTS=3) — the support
    # side must come back None rather than fit a line through 2 points.
    history = _make_history(_zigzag([100, 150, 110, 150.05, 130, 149.95, 145], 8))
    trendlines = compute_trendlines(history)
    assert trendlines is not None
    assert trendlines.support_trendline is None
    assert trendlines.resistance_trendline is not None


def test_trendlines_detects_a_genuinely_rising_support_line():
    history = _make_history(_zigzag([100, 150, 105, 150.05, 120, 149.95, 140, 150.02, 145], 8))
    trendlines = compute_trendlines(history)
    assert trendlines.support_trendline is not None
    assert trendlines.support_trendline.direction == "rising"
    assert trendlines.support_trendline.slope_per_bar > 0
    assert trendlines.resistance_trendline.direction == "flat"


# --- detect_chart_patterns ---


def test_detects_double_top_with_a_real_intervening_pullback():
    history = _make_history(_zigzag([100, 150, 130, 149.8, 100], 10))
    patterns = detect_chart_patterns(history)
    names = [p.name for p in patterns]
    assert "double_top" in names


def test_detects_double_bottom_with_a_real_intervening_bounce():
    history = _make_history(_zigzag([150, 100, 120, 100.2, 150], 10))
    patterns = detect_chart_patterns(history)
    names = [p.name for p in patterns]
    assert "double_bottom" in names


def test_no_double_top_when_peaks_are_too_far_apart_in_price():
    # Second peak (170) is nowhere near the first (150) — must NOT be
    # mistaken for a double top just because both are "highs".
    history = _make_history(_zigzag([100, 150, 130, 170, 100], 10))
    patterns = detect_chart_patterns(history)
    assert "double_top" not in [p.name for p in patterns]


def test_no_double_top_when_intervening_pullback_is_too_shallow():
    # Two comparable peaks, but the dip between them barely moves —
    # not a real reversal in between, so no pattern should fire.
    history = _make_history(_zigzag([100, 150, 149.9, 150.05, 100], 10))
    patterns = detect_chart_patterns(history)
    assert "double_top" not in [p.name for p in patterns]


def test_detects_uptrend_structure_from_higher_highs_and_higher_lows():
    history = _make_history(_zigzag([100, 120, 110, 140, 125, 160], 8))
    patterns = detect_chart_patterns(history)
    assert "uptrend_structure" in [p.name for p in patterns]
    assert "downtrend_structure" not in [p.name for p in patterns]


def test_detects_downtrend_structure_from_lower_highs_and_lower_lows():
    history = _make_history(_zigzag([160, 140, 150, 110, 130, 90], 8))
    patterns = detect_chart_patterns(history)
    assert "downtrend_structure" in [p.name for p in patterns]
    assert "uptrend_structure" not in [p.name for p in patterns]


def test_detects_ascending_triangle_from_flat_resistance_and_rising_support():
    history = _make_history(_zigzag([100, 150, 105, 150.05, 120, 149.95, 140, 150.02, 145], 8))
    patterns = detect_chart_patterns(history)
    assert "ascending_triangle" in [p.name for p in patterns]


def test_no_pattern_flagged_for_a_flat_directionless_chop():
    # Small, non-alternating-in-a-meaningful-way noise — nothing here
    # should be confidently called a pattern.
    closes = [100.0, 100.1, 99.9, 100.05, 99.95, 100.02, 99.98, 100.0] * 5
    history = _make_history(closes)
    patterns = detect_chart_patterns(history)
    assert patterns == [] or all(p.name not in ("double_top", "double_bottom") for p in patterns)


def test_detect_chart_patterns_empty_list_not_none_on_empty_history():
    assert detect_chart_patterns(pd.DataFrame(columns=["Close", "High", "Low"])) == []


# --- compute_chart_structure (the bundling API) ---


def test_compute_chart_structure_bundles_all_four_reads():
    history = _make_history(_zigzag([100, 150, 90, 140], 10))
    snapshot = compute_chart_structure(history)
    assert snapshot.fibonacci is not None
    assert snapshot.sr_levels is not None
    assert isinstance(snapshot.patterns, list)


def test_compute_chart_structure_degrades_cleanly_on_empty_history():
    snapshot = compute_chart_structure(pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
    assert snapshot.fibonacci is None
    assert snapshot.sr_levels is None
    assert snapshot.trendlines is None
    assert snapshot.patterns == []


def test_compute_chart_structure_only_scans_for_swing_points_once():
    # Regression: compute_chart_structure used to call each of the four
    # sub-functions independently, and detect_chart_patterns' own body
    # (plus its internal compute_trendlines call) added two more — five
    # calls to find_swing_points on the identical window for one result.
    import analysis.chart_structure as chart_structure_module

    history = _make_history(_zigzag([105, 100, 120, 110, 150, 133], 8))
    call_count = 0
    real_find_swing_points = chart_structure_module.find_swing_points

    def _counting_find_swing_points(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_find_swing_points(*args, **kwargs)

    chart_structure_module.find_swing_points = _counting_find_swing_points
    try:
        chart_structure_module.compute_chart_structure(history)
    finally:
        chart_structure_module.find_swing_points = real_find_swing_points

    assert call_count == 1


def test_compute_chart_structure_still_produces_correct_results_after_sharing_swing_points():
    # The refactor to share one swing-point scan must produce IDENTICAL
    # results to calling each function independently with its own scan.
    # compute_chart_structure's own `patterns` also concatenates
    # candlestick patterns (analysis/candlestick_patterns.py, added
    # 2026-08-26) — included here via its own independent call too, not
    # just the swing-based detect_chart_patterns.
    from analysis.candlestick_patterns import detect_candlestick_patterns

    history = _make_history(_zigzag([105, 100, 120, 110, 150, 133], 8))
    shared = compute_chart_structure(history)
    independent_fib = compute_fibonacci_levels(history)
    # compute_chart_structure passes its own ATR-scaled tolerance (see
    # atr_scaled_sr_tolerance) into compute_sr_levels, not the flat
    # default — reuse that same real value here too, so this test keeps
    # checking "shared computation == independent computation" rather
    # than accidentally asserting two DIFFERENT tolerances produce
    # identical results (they don't, for a genuinely fast instrument).
    independent_sr = compute_sr_levels(history, tolerance_pct=atr_scaled_sr_tolerance(history))
    independent_trendlines = compute_trendlines(history)
    independent_patterns = detect_chart_patterns(history) + detect_candlestick_patterns(history)

    assert shared.fibonacci == independent_fib
    assert shared.sr_levels == independent_sr
    assert shared.trendlines == independent_trendlines
    assert shared.patterns == independent_patterns


# --- find_mtf_confluence ---


def test_find_mtf_confluence_matches_close_levels_across_timeframes():
    zones = find_mtf_confluence(
        higher_tf_levels=[150.0, 100.0], lower_tf_levels=[150.05, 105.0], current_price=120.0
    )
    assert len(zones) == 1
    assert zones[0].avg_price == pytest.approx(150.025)
    assert zones[0].kind == "resistance"


def test_find_mtf_confluence_rejects_levels_too_far_apart():
    # 100 vs 105 is 4.9% apart, well outside the default 0.5% tolerance.
    zones = find_mtf_confluence(higher_tf_levels=[100.0], lower_tf_levels=[105.0], current_price=90.0)
    assert zones == []


def test_find_mtf_confluence_labels_support_vs_resistance_by_current_price():
    zones = find_mtf_confluence(
        higher_tf_levels=[90.0], lower_tf_levels=[90.02], current_price=120.0
    )
    assert zones[0].kind == "support"
    assert zones[0].distance_pct < 0


def test_find_mtf_confluence_empty_with_no_candidate_levels():
    assert find_mtf_confluence([], [], current_price=100.0) == []
