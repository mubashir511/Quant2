import pandas as pd
import pytest

from analysis.chart_structure import (
    SR_CLUSTER_TOLERANCE_PCT,
    SR_TOLERANCE_ATR_MULTIPLE,
    SWEEP_MAX_BARS_TO_CLOSE_BACK,
    SRLevel,
    SRLevelsResult,
    SwingPoint,
    _cluster_prices,
    _cluster_swing_points,
    atr_scaled_sr_tolerance,
    compute_chart_structure,
    compute_fibonacci_levels,
    compute_sr_levels,
    compute_trendlines,
    detect_breakouts,
    detect_chart_patterns,
    detect_liquidity_sweeps,
    detect_structure_breaks,
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


def test_cluster_swing_points_never_lets_a_cluster_drift_beyond_tolerance():
    # Same anchor-to-first-point discipline as _cluster_prices' own test
    # above, applied to the new SwingPoint-based clustering.
    points = [SwingPoint(index=i, price=p) for i, p in enumerate([100.0, 100.09, 100.18, 100.27])]
    clusters = _cluster_swing_points(points, tolerance_pct=0.3, last_index=10)
    assert len(clusters) == 1
    assert clusters[0].low == 100.0
    assert clusters[0].high == pytest.approx(100.27)


def test_cluster_swing_points_weighted_score_decays_with_bars_ago():
    fresh = _cluster_swing_points([SwingPoint(index=10, price=100.0)], tolerance_pct=0.3, last_index=10, half_life_bars=30.0)
    stale = _cluster_swing_points([SwingPoint(index=10, price=100.0)], tolerance_pct=0.3, last_index=40, half_life_bars=30.0)
    assert fresh[0].weighted_score == pytest.approx(1.0)
    assert stale[0].weighted_score == pytest.approx(0.5)


def test_cluster_swing_points_empty_input():
    assert _cluster_swing_points([], tolerance_pct=0.3, last_index=10) == []


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


def test_sr_levels_touches_beat_a_higher_untested_level_for_ranking_when_rank_by_touches():
    # A well-tested level (2 touches, farther from price) should rank
    # ahead of a once-touched level (1 touch, nearer to price) under the
    # explicit rank_by="touches" back-compat path — real evidence of a
    # level mattering beats mere proximity. re-verified 2026-09-20 after
    # switching the DEFAULT ranking to weighted_score (see
    # test_sr_levels_weighted_score_favors_recent_touches_over_stale_
    # ones below for the new default's own behavior on this exact
    # fixture) — this old assertion only still holds when touches is
    # explicitly requested.
    history = _make_history(_zigzag([120, 150.0, 100.0, 150.1, 100.05, 149.95, 160, 105], 8))
    sr = compute_sr_levels(history, rank_by="touches")
    assert sr.resistance_levels[0].touches >= sr.resistance_levels[-1].touches


def test_sr_levels_rank_by_rejects_an_unknown_value():
    history = _make_history(_zigzag([120, 150.0, 100.0, 150.1, 100.05, 149.95, 160, 105], 8))
    with pytest.raises(ValueError):
        compute_sr_levels(history, rank_by="bogus")


def test_sr_levels_rejects_a_non_positive_half_life():
    # Real footgun found on self-review: half_life_bars=0 crashes on
    # division by zero, and a negative value silently INVERTS the decay
    # direction (stale touches would outscore recent ones) with no error
    # at all — no real caller passes anything but the default today, but
    # both must fail loudly, not silently misbehave or crash obscurely.
    history = _make_history(_zigzag([120, 150.0, 100.0, 150.1, 100.05, 149.95, 160, 105], 8))
    with pytest.raises(ValueError):
        compute_sr_levels(history, half_life_bars=0.0)
    with pytest.raises(ValueError):
        compute_sr_levels(history, half_life_bars=-10.0)


def test_sr_levels_weighted_score_favors_recent_touches_over_stale_ones():
    # Same fixture as the touches-ranking test above, but under the new
    # DEFAULT ranking (weighted_score): the 160 level is touched only
    # ONCE, but it's the MOST RECENT swing in the whole window, while the
    # 150-cluster's 2 touches sit much earlier — real gap found 2026-09-20,
    # direct user challenge: a stale multi-touch level was outranking a
    # level price is actively respecting right now. This is the test that
    # proves recency weighting actually changes an outcome, not just
    # attaches an unused extra number.
    history = _make_history(_zigzag([120, 150.0, 100.0, 150.1, 100.05, 149.95, 160, 105], 8))
    sr = compute_sr_levels(history)  # default rank_by="weighted_score"
    assert sr.resistance_levels[0].price == pytest.approx(160.0, abs=0.5)


def test_sr_levels_band_reflects_real_min_max_of_clustered_touches():
    # The two real swing highs near 150.0 and 150.1 cluster into one
    # level (149.95 is a monotonic waypoint between 100.05 and 160 in
    # this zigzag, not itself a confirmed swing point, so it never joins
    # this cluster — confirmed against the real computed swing points,
    # not assumed) — the resulting band's low/high must be the real
    # min/max of those two swing prices, not just their mean.
    history = _make_history(_zigzag([120, 150.0, 100.0, 150.1, 100.05, 149.95, 160, 105], 8))
    sr = compute_sr_levels(history)
    clustered = next(lvl for lvl in sr.resistance_levels if lvl.touches >= 2)
    assert clustered.low == pytest.approx(150.0, abs=0.01)
    assert clustered.high == pytest.approx(150.1, abs=0.01)
    assert clustered.low < clustered.high


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


# --- detect_breakouts / detect_liquidity_sweeps: real vs. fake, added ----
# 2026-09-20 direct user challenge after reviewing real trades -----------


def _ohlcv(closes: list[float], highs=None, lows=None, volumes=None) -> pd.DataFrame:
    n = len(closes)
    index = pd.date_range("2026-01-01", periods=n, freq="h")
    highs = closes if highs is None else highs
    lows = closes if lows is None else lows
    volumes = [100.0] * n if volumes is None else volumes
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": volumes}, index=index
    )


def _resistance_sr(band_low=100.0, band_high=101.0) -> SRLevelsResult:
    return SRLevelsResult(
        resistance_levels=[SRLevel(price=100.5, touches=2, distance_pct=0.5, low=band_low, high=band_high)],
        support_levels=[],
    )


def _support_sr(band_low=99.0, band_high=100.0) -> SRLevelsResult:
    return SRLevelsResult(
        resistance_levels=[],
        support_levels=[SRLevel(price=99.5, touches=2, distance_pct=-0.5, low=band_low, high=band_high)],
    )


def test_detect_breakouts_confirms_a_genuine_high_volume_close_through_resistance():
    closes = [95.0] * 20 + [101.5]
    volumes = [100.0] * 20 + [250.0]  # 2.5x the 20-bar baseline
    history = _ohlcv(closes, volumes=volumes)
    events = detect_breakouts(history, _resistance_sr())
    assert len(events) == 1
    assert events[0].direction == "up"
    assert events[0].volume_confirmed is True
    assert events[0].volume_ratio == pytest.approx(2.5)


def test_detect_breakouts_does_not_confirm_a_low_volume_close_through():
    closes = [95.0] * 20 + [101.5]
    volumes = [100.0] * 20 + [110.0]  # only 1.1x baseline — below the 1.3x confirm threshold
    history = _ohlcv(closes, volumes=volumes)
    events = detect_breakouts(history, _resistance_sr())
    assert len(events) == 1  # the real close-through is still reported...
    assert events[0].volume_confirmed is False  # ...just not confirmed


def test_detect_breakouts_none_when_no_volume_column():
    closes = [95.0] * 20 + [101.5]
    history = _ohlcv(closes).drop(columns=["Volume"])
    events = detect_breakouts(history, _resistance_sr())
    assert len(events) == 1
    assert events[0].volume_ratio is None
    assert events[0].volume_confirmed is False


def test_detect_breakouts_ignores_a_close_that_never_clears_the_band_edge():
    closes = [95.0] * 20 + [100.5]  # inside the 100-101 band, not through it
    history = _ohlcv(closes)
    assert detect_breakouts(history, _resistance_sr()) == []


def test_detect_breakouts_empty_without_sr_levels():
    history = _ohlcv([95.0] * 21)
    assert detect_breakouts(history, None) == []


def test_detect_breakouts_empty_history_does_not_crash():
    # Real regression found on self-review: the transition-walk fix above
    # (test_detect_breakouts_credits_the_original_breakout_bars_volume_
    # not_todays) unconditionally checked the LAST bar before any length
    # guard, which raised IndexError on a genuinely empty (but correctly
    # columned) history — a real SRLevel with no matching price data.
    empty_history = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": [], "Volume": []})
    assert detect_breakouts(empty_history, _resistance_sr()) == []


def test_detect_breakouts_support_side_bets_down():
    closes = [105.0] * 20 + [98.5]  # closes below the 99-100 support band
    history = _ohlcv(closes, volumes=[100.0] * 20 + [200.0])
    events = detect_breakouts(history, _support_sr())
    assert len(events) == 1
    assert events[0].direction == "down"
    assert events[0].volume_confirmed is True


def test_detect_breakouts_credits_the_original_breakout_bars_volume_not_todays():
    # Real bug found on self-review: walking backward and stopping at the
    # FIRST (i.e. today's) qualifying bar meant a SUSTAINED breakout —
    # price closes beyond the level for several bars running — always
    # reported bars_ago=0 using TODAY's volume, silently discarding the
    # ORIGINAL breakout bar's own real volume signature. A genuine 5x
    # breakout 3 bars ago, followed by 3 more bars still beyond the level
    # but at ordinary volume, must still be credited to the REAL
    # breakout bar (bars_ago=3, volume confirmed) — not misreported as an
    # unconfirmed breakout using today's ordinary volume.
    closes = [95.0] * 20 + [101.5, 101.6, 101.7, 101.8]
    volumes = [100.0] * 20 + [500.0, 100.0, 100.0, 100.0]
    history = _ohlcv(closes, volumes=volumes)
    events = detect_breakouts(history, _resistance_sr(), lookback_bars=5)
    assert len(events) == 1
    assert events[0].bars_ago == 3
    assert events[0].volume_ratio == pytest.approx(5.0)
    assert events[0].volume_confirmed is True


def test_detect_liquidity_sweeps_flags_a_same_bar_wick_and_close_back_inside():
    # A single spike bar: High wicks through resistance, but the SAME
    # bar's own Close reverts back inside — the most common real sweep
    # shape (a fast stop-hunt, not a slow multi-bar failure).
    closes = [95.0] * 20 + [100.5]
    highs = [95.0] * 20 + [102.0]  # wicks well above the 101.0 band edge
    history = _ohlcv(closes, highs=highs)
    events = detect_liquidity_sweeps(history, _resistance_sr())
    assert len(events) == 1
    assert events[0].direction == "swept_above"
    assert events[0].wick_penetration_pct > 0


def test_detect_liquidity_sweeps_flags_a_wick_that_closes_back_inside_a_few_bars_later():
    closes = [95.0] * 20 + [101.5, 100.8]  # wick bar closes beyond, next bar reverts inside
    highs = [95.0] * 20 + [102.0, 100.8]
    history = _ohlcv(closes, highs=highs)
    events = detect_liquidity_sweeps(history, _resistance_sr(), lookback_bars=5)
    assert len(events) == 1
    assert events[0].bars_ago == 1  # the wick bar itself, one bar before the latest


def test_detect_liquidity_sweeps_does_not_fire_on_a_real_breakout_that_holds():
    # Close stays beyond the band edge for MORE than SWEEP_MAX_BARS_TO_
    # CLOSE_BACK bars — a real, held breakout, not a fast sweep.
    hold_bars = SWEEP_MAX_BARS_TO_CLOSE_BACK + 3
    closes = [95.0] * 20 + [101.5] * hold_bars
    highs = [95.0] * 20 + [102.0] * hold_bars
    history = _ohlcv(closes, highs=highs)
    events = detect_liquidity_sweeps(history, _resistance_sr(), lookback_bars=hold_bars + 2)
    assert events == []


def test_detect_liquidity_sweeps_empty_without_sr_levels():
    history = _ohlcv([95.0] * 21)
    assert detect_liquidity_sweeps(history, None) == []


def test_compute_chart_structure_wires_breakouts_and_sweeps():
    # Regression test, same style as the existing atr-scaled-tolerance
    # wiring test — proves compute_chart_structure actually threads
    # breakouts/liquidity_sweeps through, not just the four original reads.
    history = _make_history(_zigzag([120, 150.0, 100.0, 150.1, 100.05, 149.95, 160, 105], 8))
    snapshot = compute_chart_structure(history)
    assert isinstance(snapshot.breakouts, list)
    assert isinstance(snapshot.liquidity_sweeps, list)


# --- detect_structure_breaks: BOS/CHOCH, added 2026-09-20 direct user ----
# challenge ("differential between trend and reversal") ------------------


def test_detect_structure_breaks_labels_a_break_with_the_prevailing_trend_as_bos():
    # A real, confirmed uptrend (higher highs 140->160, higher lows
    # 110->130), then a fresh close (165) above the latest confirmed
    # swing high (160) — a break WITH the trend's own direction.
    history = _make_history(_zigzag([100, 90, 120, 95, 140, 110, 160, 130, 165], 8))
    events = detect_structure_breaks(history)
    assert len(events) == 1
    assert events[0].kind == "BOS"
    assert events[0].direction == "bullish"
    assert events[0].broken_level == pytest.approx(160.001, abs=0.01)


def test_detect_structure_breaks_labels_the_first_counter_trend_break_as_choch():
    # Same confirmed uptrend as above, but this time price dives straight
    # through the latest confirmed swing LOW (110) instead — a break
    # AGAINST the established trend's own direction, the first real
    # structural evidence of a possible reversal.
    history = _make_history(_zigzag([100, 90, 120, 95, 140, 110, 160, 90], 8))
    events = detect_structure_breaks(history)
    assert len(events) == 1
    assert events[0].kind == "CHOCH"
    assert events[0].direction == "bearish"
    assert events[0].broken_level == pytest.approx(109.999, abs=0.01)


def test_detect_structure_breaks_empty_with_no_confirmed_swings():
    history = _make_history(_zigzag([100, 150], 20))  # one pure monotonic rise, no reversal at all
    assert detect_structure_breaks(history) == []


def test_detect_structure_breaks_empty_when_price_never_breaks_the_last_swing():
    # A real uptrend where the latest close stays comfortably INSIDE the
    # existing structure — nothing has actually broken yet.
    history = _make_history(_zigzag([100, 90, 120, 95, 140, 110, 160, 130, 140], 8))
    assert detect_structure_breaks(history) == []


def test_detect_structure_breaks_reports_the_real_crossing_bar_not_todays():
    # Real bug found on self-review: walking backward and stopping at the
    # FIRST (i.e. today's) bar that satisfied "broke" meant a SUSTAINED
    # break — price has closed beyond the swing low for several bars
    # running, not just today — always reported bars_ago=0, silently
    # misrepresenting an old, already-known structural fact as brand new
    # every time this was recomputed. Same confirmed uptrend as the CHOCH
    # test above, but this time price breaks the swing low (90) and stays
    # below it for several more bars — bars_ago must point at the ACTUAL
    # crossing bar, not just "today."
    history = _make_history(_zigzag([100, 90, 120, 95, 140, 110, 160, 90], 8) + [89.0, 88.0, 87.0])
    events = detect_structure_breaks(history)
    assert len(events) == 1
    assert events[0].kind == "CHOCH"
    # The real crossing bar (first close below the 109.999 swing low) is
    # 5 bars before the series' own last bar — verified empirically, not
    # guessed — and must NOT be reported as bars_ago=0 just because price
    # has stayed below that level ever since.
    assert events[0].bars_ago == 5
    assert events[0].break_price == pytest.approx(107.5, abs=0.01)  # the crossing bar's own close, not today's (87.0)
