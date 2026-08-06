import pandas as pd

from analysis.technical import compute_technical_stats


def make_series(n_days: int, start: float = 100.0, daily_drift: float = 0.0) -> pd.Series:
    values = [start + i * daily_drift for i in range(n_days)]
    index = pd.date_range("2026-01-01", periods=n_days, freq="D")
    return pd.Series(values, index=index)


def test_returns_all_none_for_empty_series():
    stats = compute_technical_stats(pd.Series(dtype=float))
    assert stats.last_price is None
    assert stats.trend is None


def test_uptrend_detected_when_price_above_sma20():
    prices = make_series(30, start=100.0, daily_drift=1.0)  # steadily rising
    stats = compute_technical_stats(prices)
    assert stats.trend == "uptrend"
    assert stats.pct_vs_sma20 > 1.0


def test_downtrend_detected_when_price_below_sma20():
    prices = make_series(30, start=100.0, daily_drift=-1.0)  # steadily falling
    stats = compute_technical_stats(prices)
    assert stats.trend == "downtrend"
    assert stats.pct_vs_sma20 < -1.0


def test_flat_trend_for_constant_price():
    prices = make_series(30, start=100.0, daily_drift=0.0)
    stats = compute_technical_stats(prices)
    assert stats.trend == "flat"
    assert stats.volatility_annualized_pct == 0.0


def test_no_sma_or_trend_when_history_shorter_than_window():
    prices = make_series(10)
    stats = compute_technical_stats(prices)
    assert stats.sma20 is None
    assert stats.trend is None
    assert stats.last_price is not None


def test_change_windows_omitted_when_not_enough_history():
    # Only 40 days: 1M change should compute, 3M/6M should not be fabricated.
    prices = make_series(40, start=100.0, daily_drift=0.5)
    stats = compute_technical_stats(prices)
    assert stats.change_1m_pct is not None
    assert stats.change_3m_pct is None
    assert stats.change_6m_pct is None


def test_change_windows_present_with_enough_history():
    prices = make_series(200, start=100.0, daily_drift=0.1)
    stats = compute_technical_stats(prices)
    assert stats.change_1m_pct is not None
    assert stats.change_3m_pct is not None
    assert stats.change_6m_pct is not None


def test_support_resistance_none_when_shorter_than_range_window():
    prices = make_series(59, start=100.0, daily_drift=1.0)
    stats = compute_technical_stats(prices)
    assert stats.support is None
    assert stats.resistance is None
    assert stats.market_regime is None


def test_support_resistance_and_trending_regime_for_monotonic_rise():
    # Every day moves +1 with no reversals: the window's low/high are its
    # first/last values, and the move is perfectly directional.
    prices = make_series(60, start=100.0, daily_drift=1.0)
    stats = compute_technical_stats(prices)
    assert stats.support == 100.0
    assert stats.resistance == 159.0
    assert stats.market_regime == "trending_up"
    assert round(stats.range_width_pct, 1) == round((159.0 - 100.0) / 159.0 * 100, 1)


def test_trending_down_regime_for_monotonic_decline():
    prices = make_series(60, start=100.0, daily_drift=-1.0)
    stats = compute_technical_stats(prices)
    assert stats.market_regime == "trending_down"


def test_sideways_regime_for_choppy_oscillation():
    # Alternates +5/-5 around a base: large total path, near-zero net move.
    values = [100.0 + (5 if i % 2 == 0 else -5) for i in range(60)]
    prices = pd.Series(values, index=pd.date_range("2026-01-01", periods=60, freq="D"))
    stats = compute_technical_stats(prices)
    assert stats.market_regime == "sideways"
