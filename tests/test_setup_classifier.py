import pandas as pd
import pytest

from analysis.chart_structure import (
    ChartPattern,
    ChartStructureSnapshot,
    FibonacciLevels,
    SRLevel,
    SRLevelsResult,
    Trendline,
    TrendlineAnalysis,
    compute_chart_structure,
)
from analysis.setup_classifier import _true_retracement_ratio, classify_setups
from analysis.technical import TechnicalStats, compute_technical_stats


def _zigzag(extrema: list[float], seg_len: int) -> list[float]:
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


def _stats(**overrides) -> TechnicalStats:
    defaults = dict(
        last_price=100.0, sma20=None, pct_vs_sma20=None, trend=None,
        change_1m_pct=None, change_3m_pct=None, change_6m_pct=None,
        volatility_annualized_pct=None, support=None, resistance=None,
        range_width_pct=None, market_regime=None, atr=None, atr_pct=None,
        rsi=None, volume_trend_pct=None,
    )
    return TechnicalStats(**{**defaults, **overrides})


def _empty_structure(**overrides) -> ChartStructureSnapshot:
    defaults = dict(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])
    return ChartStructureSnapshot(**{**defaults, **overrides})


def _trendline(**overrides) -> Trendline:
    defaults = dict(
        slope_per_bar=0.0, direction="flat", current_price_on_line=100.0,
        price_vs_line="at", distance_pct=0.0, bars_since_last_point=0,
    )
    return Trendline(**{**defaults, **overrides})


def _fib(**overrides) -> FibonacciLevels:
    defaults = dict(
        swing_high=150.0, swing_low=110.0, high_is_more_recent=True,
        levels={}, current_price=130.0, nearest_level_name="50.0%",
        nearest_level_price=130.0, distance_to_nearest_pct=0.0,
    )
    return FibonacciLevels(**{**defaults, **overrides})


# --- _true_retracement_ratio ---


def test_true_retracement_ratio_excludes_a_price_snapping_to_a_zone_level_but_genuinely_outside_it():
    # Regression: current_price corresponds to a TRUE ratio of 0.36 (just
    # short of the golden zone's real 0.382 boundary), but its NEAREST
    # named level is "38.2%" — snapping to that label first used to
    # misclassify this as inside the golden zone.
    span = 150.0 - 110.0
    price_at_ratio_036 = 150.0 - 0.36 * span
    fib = _fib(swing_high=150.0, swing_low=110.0, high_is_more_recent=True, current_price=price_at_ratio_036)
    ratio = _true_retracement_ratio(fib)
    assert ratio == pytest.approx(0.36)
    assert not (0.382 <= ratio <= 0.618)


def test_true_retracement_ratio_matches_direction_for_low_more_recent():
    span = 150.0 - 110.0
    price_at_ratio_050 = 110.0 + 0.5 * span
    fib = _fib(swing_high=150.0, swing_low=110.0, high_is_more_recent=False, current_price=price_at_ratio_050)
    assert _true_retracement_ratio(fib) == pytest.approx(0.5)


def test_true_retracement_ratio_none_for_degenerate_zero_span():
    fib = _fib(swing_high=100.0, swing_low=100.0, current_price=100.0)
    assert _true_retracement_ratio(fib) is None


def test_pullback_continuation_does_not_fire_for_a_retracement_short_of_the_true_golden_zone():
    span = 150.0 - 110.0
    price_at_ratio_036 = 150.0 - 0.36 * span
    structure = _empty_structure(
        patterns=[ChartPattern(name="uptrend_structure", detail="higher highs and higher lows")],
        fibonacci=_fib(
            swing_high=150.0, swing_low=110.0, high_is_more_recent=True,
            current_price=price_at_ratio_036, nearest_level_name="38.2%",
        ),
    )
    names = [s.name for s in classify_setups(_stats(last_price=price_at_ratio_036), structure)]
    assert "pullback_continuation" not in names


# --- end-to-end, via real constructed price series ---


def test_pullback_continuation_from_a_real_uptrend_retracing_into_the_golden_zone():
    # Confirmed uptrend structure (2 higher highs, 2 higher lows), with
    # the most recent confirmed pivot a HIGH and current price retracing
    # into the 38.2% Fibonacci zone measured back from it — hand-verified
    # against the live code before encoding here.
    history = _make_history(_zigzag([105, 100, 120, 110, 150, 133], 8))
    stats = compute_technical_stats(history["Close"], history=history)
    structure = compute_chart_structure(history)
    names = [s.name for s in classify_setups(stats, structure)]
    assert "pullback_continuation" in names


def test_reversal_and_range_fade_can_coexist_on_a_real_choppy_double_top():
    history = _make_history(_zigzag([105, 100, 110, 100.05, 110, 105, 100.08], 11))
    stats = compute_technical_stats(history["Close"], history=history)
    structure = compute_chart_structure(history)
    names = [s.name for s in classify_setups(stats, structure)]
    assert "reversal_candidate" in names
    assert "range_fade_candidate" in names


# --- individual rules, via directly-constructed inputs (isolating each rule) ---


def test_no_signals_return_no_clear_setup():
    signals = classify_setups(_stats(), _empty_structure())
    assert len(signals) == 1
    assert signals[0].name == "no_clear_setup"


def test_empty_when_last_price_is_none():
    stats = _stats(last_price=None)
    assert classify_setups(stats, _empty_structure()) == []


def test_breakout_watch_fires_on_any_triangle_pattern():
    structure = _empty_structure(
        patterns=[ChartPattern(name="ascending_triangle", detail="flat resistance, rising support")]
    )
    names = [s.name for s in classify_setups(_stats(), structure)]
    assert "breakout_watch" in names


def test_trend_following_fires_when_price_sits_on_its_own_trend_supporting_trendline():
    structure = _empty_structure(
        patterns=[ChartPattern(name="uptrend_structure", detail="higher highs and higher lows")],
        trendlines=TrendlineAnalysis(
            resistance_trendline=None,
            support_trendline=_trendline(
                slope_per_bar=0.5, direction="rising", current_price_on_line=100.2,
                price_vs_line="above", distance_pct=0.1,
            ),
        ),
    )
    names = [s.name for s in classify_setups(_stats(last_price=100.3), structure)]
    assert "trend_following" in names


def test_trend_following_does_not_fire_when_price_is_far_from_the_trendline():
    structure = _empty_structure(
        patterns=[ChartPattern(name="uptrend_structure", detail="higher highs and higher lows")],
        trendlines=TrendlineAnalysis(
            resistance_trendline=None,
            support_trendline=_trendline(
                slope_per_bar=0.5, direction="rising", current_price_on_line=90.0,
                price_vs_line="above", distance_pct=11.1,  # far outside NEAR_LEVEL_TOLERANCE_PCT
            ),
        ),
    )
    names = [s.name for s in classify_setups(_stats(last_price=100.0), structure)]
    assert "trend_following" not in names


def test_downtrend_uses_the_resistance_trendline_not_the_support_one():
    structure = _empty_structure(
        patterns=[ChartPattern(name="downtrend_structure", detail="lower highs and lower lows")],
        trendlines=TrendlineAnalysis(
            resistance_trendline=_trendline(
                slope_per_bar=-0.5, direction="falling", current_price_on_line=99.9,
                price_vs_line="below", distance_pct=-0.1,
            ),
            support_trendline=_trendline(
                slope_per_bar=-0.5, direction="falling", current_price_on_line=50.0,
                price_vs_line="above", distance_pct=100.0,  # would wrongly fire if the wrong line were used
            ),
        ),
    )
    names = [s.name for s in classify_setups(_stats(last_price=100.0), structure)]
    assert "trend_following" in names


def test_range_fade_requires_both_sideways_regime_and_proximity_to_a_strong_level():
    strong_level_far = SRLevelsResult(
        resistance_levels=[SRLevel(price=110.0, touches=3, distance_pct=10.0)], support_levels=[]
    )
    # Sideways but too far from the level — must not fire.
    names_far = [s.name for s in classify_setups(_stats(market_regime="sideways"), _empty_structure(sr_levels=strong_level_far))]
    assert "range_fade_candidate" not in names_far

    strong_level_near = SRLevelsResult(
        resistance_levels=[SRLevel(price=100.2, touches=3, distance_pct=0.2)], support_levels=[]
    )
    names_near = [s.name for s in classify_setups(_stats(market_regime="sideways"), _empty_structure(sr_levels=strong_level_near))]
    assert "range_fade_candidate" in names_near

    # Trending (not sideways) even though near the same strong level — must not fire.
    names_trending = [s.name for s in classify_setups(_stats(market_regime="trending_up"), _empty_structure(sr_levels=strong_level_near))]
    assert "range_fade_candidate" not in names_trending


def test_range_fade_ignores_a_weakly_touched_level():
    weak_level = SRLevelsResult(
        resistance_levels=[SRLevel(price=100.1, touches=1, distance_pct=0.1)], support_levels=[]
    )
    names = [s.name for s in classify_setups(_stats(market_regime="sideways"), _empty_structure(sr_levels=weak_level))]
    assert "range_fade_candidate" not in names
