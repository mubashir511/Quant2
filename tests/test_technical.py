import pandas as pd
import pytest

from analysis.technical import compute_technical_stats


def make_series(n_days: int, start: float = 100.0, daily_drift: float = 0.0) -> pd.Series:
    values = [start + i * daily_drift for i in range(n_days)]
    index = pd.date_range("2026-01-01", periods=n_days, freq="D")
    return pd.Series(values, index=index)


def make_history(
    n_days: int, start: float = 100.0, daily_drift: float = 0.0, volumes: list[float] | None = None
) -> pd.DataFrame:
    prices = make_series(n_days, start, daily_drift)
    return pd.DataFrame(
        {
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices,
            "Volume": volumes if volumes is not None else [1000.0] * n_days,
        },
        index=prices.index,
    )


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


def test_rsi_none_when_shorter_than_window():
    prices = make_series(10, start=100.0, daily_drift=1.0)
    stats = compute_technical_stats(prices)
    assert stats.rsi is None


def test_rsi_high_for_steady_uptrend_with_no_down_days():
    # Every day is a gain, no losing days at all: RSI should sit at its
    # ceiling (100), not just "high".
    prices = make_series(20, start=100.0, daily_drift=1.0)
    stats = compute_technical_stats(prices)
    assert stats.rsi == 100.0


def test_rsi_low_for_steady_downtrend_with_no_up_days():
    prices = make_series(20, start=100.0, daily_drift=-1.0)
    stats = compute_technical_stats(prices)
    assert stats.rsi == 0.0


def test_rsi_mid_range_for_choppy_oscillation():
    values = [100.0 + (5 if i % 2 == 0 else -5) for i in range(20)]
    prices = pd.Series(values, index=pd.date_range("2026-01-01", periods=20, freq="D"))
    stats = compute_technical_stats(prices)
    assert 30 < stats.rsi < 70


def test_atr_and_volume_trend_none_without_history():
    prices = make_series(30, start=100.0, daily_drift=1.0)
    stats = compute_technical_stats(prices)
    assert stats.atr is None
    assert stats.atr_pct is None
    assert stats.volume_trend_pct is None


def test_atr_computed_from_history():
    history = make_history(30, start=100.0, daily_drift=1.0)
    stats = compute_technical_stats(history["Close"], history=history)
    # True Range accounts for the gap from the prior close, not just the
    # day's own High-Low spread (1.0 here) — with a steady +1.0/day drift,
    # High[t] - prev_close = 1.5, which dominates the day's own 1.0 range.
    assert stats.atr == 1.5
    assert stats.atr_pct == stats.atr / stats.last_price * 100


def test_atr_none_when_history_shorter_than_window():
    history = make_history(10, start=100.0, daily_drift=1.0)
    stats = compute_technical_stats(history["Close"], history=history)
    assert stats.atr is None


def test_volume_trend_positive_when_recent_volume_spikes():
    volumes = [1000.0] * 15 + [2000.0] * 5  # last 5 days double the baseline
    history = make_history(20, start=100.0, daily_drift=0.0, volumes=volumes)
    stats = compute_technical_stats(history["Close"], history=history)
    assert stats.volume_trend_pct is not None
    assert stats.volume_trend_pct > 0


def test_volume_trend_none_when_no_volume_column():
    history = make_history(30, start=100.0, daily_drift=1.0).drop(columns=["Volume"])
    stats = compute_technical_stats(history["Close"], history=history)
    assert stats.volume_trend_pct is None


def test_periods_per_year_defaults_to_daily_trading_days():
    import random

    random.seed(42)
    prices = pd.Series([100.0 + random.uniform(-1, 1) for _ in range(30)])
    default_call = compute_technical_stats(prices)
    explicit_daily = compute_technical_stats(prices, periods_per_year=252)
    assert default_call.volatility_annualized_pct == explicit_daily.volatility_annualized_pct


def test_periods_per_year_scales_volatility_correctly_for_a_different_bar_frequency():
    # Regression: compute_technical_stats used to always scale by
    # sqrt(TRADING_DAYS_PER_YEAR) regardless of the actual bar frequency
    # it was fed — confirmed live this understated real annualized
    # volatility by ~2.5x-6x for H4/H1 data (a live EURUSD H1 read came
    # back LESS volatile than its own daily figure). The ratio between
    # two periods_per_year values must translate directly into the ratio
    # between their reported volatility, since the underlying per-bar
    # std is identical for the same price series either way.
    import random

    random.seed(7)
    prices = pd.Series([100.0 + random.uniform(-1, 1) for _ in range(30)])
    daily = compute_technical_stats(prices, periods_per_year=252)
    hourly_scale = compute_technical_stats(prices, periods_per_year=24 * 252)

    expected_ratio = (24 * 252) ** 0.5 / 252**0.5
    actual_ratio = hourly_scale.volatility_annualized_pct / daily.volatility_annualized_pct
    assert actual_ratio == pytest.approx(expected_ratio)
