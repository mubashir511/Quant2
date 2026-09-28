from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from analysis.technical import (
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_WINDOW,
    VELOCITY_FAST_THRESHOLD_PCT,
    classify_rsi_tier,
    classify_velocity_tier,
    compute_atr,
    compute_regime_segments,
    compute_rolling_rsi,
    compute_technical_stats,
    detect_rsi_divergence,
)


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


# --- compute_rolling_rsi: promoted from analysis.backtest._rolling_rsi
# 2026-09-20 (same "public because a second module needs the identical
# baseline" move compute_atr itself already went through) -------------


def test_compute_rolling_rsi_matches_the_single_point_reading_at_the_series_end():
    # The rolling series' own LAST value must agree with _compute_rsi's
    # single-point reading on the identical price history — same
    # formula, just computed at every point instead of only the latest.
    prices = make_series(30, start=100.0, daily_drift=1.0)
    rolling = compute_rolling_rsi(prices)
    stats = compute_technical_stats(prices)
    assert rolling.iloc[-1] == pytest.approx(stats.rsi)


def test_compute_rolling_rsi_is_nan_before_a_real_windows_worth_of_history():
    prices = make_series(RSI_WINDOW, start=100.0, daily_drift=1.0)
    rolling = compute_rolling_rsi(prices)
    assert rolling.iloc[: RSI_WINDOW - 1].isna().all()


def test_compute_rolling_rsi_full_uptrend_reads_100_throughout():
    prices = make_series(30, start=100.0, daily_drift=1.0)
    rolling = compute_rolling_rsi(prices)
    assert (rolling.dropna() == 100.0).all()


# --- detect_rsi_divergence: added 2026-09-20, direct user challenge -----
# ("differential between trend and reversal") ----------------------------


@dataclass
class _SwingPointLike:
    """A minimal, duck-typed stand-in for analysis.chart_structure.
    SwingPoint — detect_rsi_divergence deliberately doesn't import the
    real class (would create a circular import), so its own tests
    shouldn't either; this proves the duck-typing contract actually
    works with any `.index`/`.price` object, not just the real one."""

    index: int
    price: float


def _bearish_divergence_prices() -> pd.Series:
    # Strong, pure uptrend to a first high (RSI pinned at 100 — no down
    # bars at all), a real pullback, then a choppy but net-positive climb
    # to a genuinely HIGHER second high with real down-bars mixed in —
    # weaker momentum even though price itself made a new high.
    prices = [100.0 + i * 1.0 for i in range(21)]  # bar0=100 -> bar20=120
    prices += [120.0 - i * 2.0 for i in range(1, 6)]  # pullback to 110 at bar 25
    val = 110.0
    for i in range(30):
        val += 2.2 if i % 2 == 0 else -1.5
        prices.append(val)
    return pd.Series(prices)


def _bullish_divergence_prices() -> pd.Series:
    # Mirror of the bearish fixture: strong pure downtrend, a bounce,
    # then a choppy but net-negative decline to a genuinely LOWER second
    # low with real up-bars mixed in — weaker downward momentum.
    prices = [100.0 - i * 1.0 for i in range(21)]  # bar0=100 -> bar20=80
    prices += [80.0 + i * 2.0 for i in range(1, 6)]  # bounce to 90 at bar 25
    val = 90.0
    for i in range(30):
        val += -2.2 if i % 2 == 0 else 1.5
        prices.append(val)
    return pd.Series(prices)


def test_detect_rsi_divergence_flags_bearish_divergence_on_a_higher_price_high_with_lower_rsi_high():
    prices = _bearish_divergence_prices()
    highs = [_SwingPointLike(index=20, price=prices.iloc[20]), _SwingPointLike(index=len(prices) - 1, price=prices.iloc[-1])]
    signal = detect_rsi_divergence(prices, swing_highs=highs)
    assert signal is not None
    assert signal.kind == "bearish"
    assert signal.rsi_at_prior > signal.rsi_at_recent


def test_detect_rsi_divergence_flags_bullish_divergence_on_a_lower_price_low_with_higher_rsi_low():
    prices = _bullish_divergence_prices()
    lows = [_SwingPointLike(index=20, price=prices.iloc[20]), _SwingPointLike(index=len(prices) - 1, price=prices.iloc[-1])]
    signal = detect_rsi_divergence(prices, swing_lows=lows)
    assert signal is not None
    assert signal.kind == "bullish"
    assert signal.rsi_at_recent > signal.rsi_at_prior


def test_detect_rsi_divergence_none_without_two_confirmed_swings():
    prices = _bearish_divergence_prices()
    signal = detect_rsi_divergence(prices, swing_highs=[_SwingPointLike(index=20, price=prices.iloc[20])])
    assert signal is None


def test_detect_rsi_divergence_none_when_price_and_rsi_both_make_new_highs_together():
    # A real, confirmed (not diverging) new high — RSI agrees with price,
    # so this must NOT be flagged.
    prices = make_series(30, start=100.0, daily_drift=1.0)  # pure uptrend, no pullback at all
    highs = [_SwingPointLike(index=15, price=prices.iloc[15]), _SwingPointLike(index=29, price=prices.iloc[29])]
    assert detect_rsi_divergence(prices, swing_highs=highs) is None


def test_detect_rsi_divergence_none_with_no_swing_points_given():
    prices = _bearish_divergence_prices()
    assert detect_rsi_divergence(prices) is None


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


def test_compute_atr_public_direct_call_matches_compute_technical_stats():
    # compute_atr was promoted from a private _compute_atr (2026-09-05,
    # for ai/curiosity.py's own reuse) — same known scenario as
    # test_atr_computed_from_history above, called directly instead of
    # through compute_technical_stats, confirming the public entry point
    # gives the identical real number.
    history = make_history(30, start=100.0, daily_drift=1.0)
    assert compute_atr(history) == 1.5


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


# --- momentum_acceleration (direct fix for "wakes up late in a sideways market") ---


def test_momentum_acceleration_none_below_window():
    prices = pd.Series([100.0] * 10)  # below MOMENTUM_WINDOW=12
    stats = compute_technical_stats(prices)
    assert stats.momentum_acceleration is None


def test_momentum_acceleration_accelerating_up_on_a_fresh_sharp_ramp():
    # A long flat run followed by a short, sharp up-move in just the
    # last 12 bars — the real scenario this signal exists to catch.
    flat = [100.0] * 50
    ramp = [100.0 + i * 1.0 for i in range(1, 13)]
    stats = compute_technical_stats(pd.Series(flat + ramp))
    assert stats.momentum_acceleration == "accelerating_up"


def test_momentum_acceleration_accelerating_down_on_a_fresh_sharp_drop():
    flat = [100.0] * 50
    drop = [100.0 - i * 1.0 for i in range(1, 13)]
    stats = compute_technical_stats(pd.Series(flat + drop))
    assert stats.momentum_acceleration == "accelerating_down"


def test_momentum_acceleration_stable_when_genuinely_flat():
    prices = pd.Series([100.0] * 60)
    stats = compute_technical_stats(prices)
    assert stats.momentum_acceleration == "stable"


def _phase(n: int, start: float, drift: float) -> list[float]:
    return [start + i * drift for i in range(n)]


def test_regime_segments_empty_below_two_windows():
    prices = pd.Series(_phase(39, 100.0, 0.0))
    assert compute_regime_segments(prices, window=20) == []


def test_regime_segments_single_sideways_segment_stays_plain_sideways():
    # Flat for exactly 2 windows: nothing precedes it, so it can't be
    # relabeled accumulation/distribution — stays plain "sideways".
    prices = pd.Series(_phase(40, 100.0, 0.0))
    segments = compute_regime_segments(prices, window=20)
    assert len(segments) == 1
    assert segments[0].regime == "sideways"
    assert segments[0].start_index == 19


def test_regime_segments_down_then_flat_infers_accumulation():
    prices = pd.Series(_phase(40, 500.0, -2.0) + _phase(40, 500.0 - 39 * 2.0, 0.0))
    segments = compute_regime_segments(prices, window=20)
    regimes = [s.regime for s in segments]
    assert regimes[0] == "trending_down"
    assert "accumulation" in regimes
    assert "distribution" not in regimes


def test_regime_segments_up_then_flat_infers_distribution():
    prices = pd.Series(_phase(40, 100.0, 2.0) + _phase(40, 100.0 + 39 * 2.0, 0.0))
    segments = compute_regime_segments(prices, window=20)
    regimes = [s.regime for s in segments]
    assert regimes[0] == "trending_up"
    assert "distribution" in regimes
    assert "accumulation" not in regimes


def test_regime_segments_never_repeats_the_same_regime_back_to_back():
    down = _phase(40, 500.0, -2.0)
    flat1 = _phase(40, down[-1], 0.0)
    up = _phase(40, flat1[-1], 2.0)
    flat2 = _phase(40, up[-1], 0.0)
    prices = pd.Series(down + flat1 + up + flat2)
    segments = compute_regime_segments(prices, window=20)
    for prev, nxt in zip(segments, segments[1:]):
        assert prev.regime != nxt.regime


def test_regime_segments_full_sequence_down_accumulation_up_distribution():
    down = _phase(40, 500.0, -2.0)
    flat1 = _phase(40, down[-1], 0.0)
    up = _phase(40, flat1[-1], 2.0)
    flat2 = _phase(40, up[-1], 0.0)
    prices = pd.Series(down + flat1 + up + flat2)
    segments = compute_regime_segments(prices, window=20)
    regimes = [s.regime for s in segments]
    assert regimes == ["trending_down", "accumulation", "trending_up", "distribution"]
    starts = [s.start_index for s in segments]
    assert starts == sorted(starts)


# --- classify_velocity_tier: real 2026-09-08/09 incident-driven feature ---


def test_classify_velocity_tier_none_without_a_real_atr_pct_reading():
    # Never guess a tier from missing data — same convention as every
    # other stat in this file.
    assert classify_velocity_tier(None) is None


def test_classify_velocity_tier_slow_below_threshold():
    # Real USDCAD reading the day this was built: ~0.10%/hour.
    assert classify_velocity_tier(0.10) == "slow"
    assert classify_velocity_tier(0.20) == "slow"


def test_classify_velocity_tier_fast_at_and_above_threshold():
    # Real XAUUSD reading the day this was built: ~0.31-0.35%/hour.
    assert classify_velocity_tier(VELOCITY_FAST_THRESHOLD_PCT) == "fast"
    assert classify_velocity_tier(0.33) == "fast"
    assert classify_velocity_tier(0.85) == "fast"  # equities-range


def test_classify_velocity_tier_boundary_is_inclusive_on_the_fast_side():
    just_below = VELOCITY_FAST_THRESHOLD_PCT - 0.001
    just_above = VELOCITY_FAST_THRESHOLD_PCT + 0.001
    assert classify_velocity_tier(just_below) == "slow"
    assert classify_velocity_tier(just_above) == "fast"


# --- classify_rsi_tier: real 2026-09-09 incident (RSI 11/15 mislabeled "overbought") ---


def test_classify_rsi_tier_none_without_a_reading():
    assert classify_rsi_tier(None) is None


def test_classify_rsi_tier_oversold_below_30():
    # The exact real readings a local model repeatedly mislabeled
    # "overbought" on a live USDCAD position across three consecutive polls.
    assert classify_rsi_tier(11) == "oversold"
    assert classify_rsi_tier(15) == "oversold"
    assert classify_rsi_tier(29.9) == "oversold"


def test_classify_rsi_tier_overbought_above_70():
    assert classify_rsi_tier(71) == "overbought"
    assert classify_rsi_tier(90) == "overbought"


def test_classify_rsi_tier_neutral_between_30_and_70_inclusive():
    assert classify_rsi_tier(30) == "neutral"
    assert classify_rsi_tier(50) == "neutral"
    assert classify_rsi_tier(70) == "neutral"


def test_classify_rsi_tier_uses_the_documented_convention_constants():
    assert classify_rsi_tier(RSI_OVERSOLD) == "neutral"  # boundary itself is not yet oversold
    assert classify_rsi_tier(RSI_OVERSOLD - 0.01) == "oversold"
    assert classify_rsi_tier(RSI_OVERBOUGHT) == "neutral"  # boundary itself is not yet overbought
    assert classify_rsi_tier(RSI_OVERBOUGHT + 0.01) == "overbought"


# --- classify_m5_velocity_tier: the M5 counterpart, re-based 2026-09-24 when M5 became the only decision timeframe ---


def test_classify_m5_velocity_tier_separates_the_measured_real_symbol_clusters():
    from analysis.technical import VELOCITY_FAST_THRESHOLD_M5_PCT, classify_m5_velocity_tier

    assert classify_m5_velocity_tier(None) is None
    # Median M5 ATR%, measured 2026-09-24 on the 20 real Market Watch symbols: every "slow" one read
    # 0.018-0.038, every "fast" one >= 0.100 (XAUUSD 0.100, XCUUSD 0.102, BTCUSD 0.126, ...).
    for slow in (0.018, 0.022, 0.038):
        assert classify_m5_velocity_tier(slow) == "slow"
    for fast in (0.100, 0.126, 0.842):
        assert classify_m5_velocity_tier(fast) == "fast"
    assert classify_m5_velocity_tier(VELOCITY_FAST_THRESHOLD_M5_PCT) == "fast"


# --- ADX (2026-09-25, playbook selector) ----------------------------------------------------------------------

def _ohlc_from_close(close, spread=0.1):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"High": close + spread, "Low": close - spread, "Close": close})


def test_adx_is_high_in_a_clean_trend_and_low_in_a_range():
    from analysis.technical import compute_adx

    trend = _ohlc_from_close(100 + np.arange(200) * 0.5)
    rng = np.random.default_rng(3)
    chop = _ohlc_from_close(100 + np.sin(np.arange(200) * 0.9) * 0.5 + rng.normal(0, 0.05, 200))
    adx_trend, _ = compute_adx(trend)
    adx_chop, _ = compute_adx(chop)
    assert adx_trend > 40 and adx_chop < 25 and adx_trend > adx_chop + 20


def test_adx_change_is_positive_when_a_trend_is_strengthening_and_negative_when_it_fades():
    from analysis.technical import compute_adx

    flat_then_trend = np.concatenate([np.full(120, 100.0) + np.random.default_rng(1).normal(0, 0.05, 120), 100 + np.arange(1, 41) * 0.6])
    trend_then_flat = np.concatenate([100 + np.arange(120) * 0.6, np.full(40, 100 + 119 * 0.6) + np.random.default_rng(2).normal(0, 0.05, 40)])
    _, rising = compute_adx(_ohlc_from_close(flat_then_trend))
    _, fading = compute_adx(_ohlc_from_close(trend_then_flat))
    assert rising > 5 and fading < -5


def test_adx_needs_high_low_and_enough_bars_and_is_bounded():
    from analysis.technical import compute_adx, compute_adx_series

    assert compute_adx(pd.DataFrame({"Close": np.arange(200.0)})) == (None, None)
    assert compute_adx(_ohlc_from_close(np.arange(30.0))) == (None, None)
    series = compute_adx_series(*(_ohlc_from_close(100 + np.cumsum(np.random.default_rng(4).normal(0, 1, 500)))[c] for c in ("High", "Low", "Close")))
    valid = series[~np.isnan(series)]
    assert valid.min() >= 0 and valid.max() <= 100


def test_technical_stats_carry_adx_when_history_has_high_low():
    from analysis.technical import compute_technical_stats

    history = _ohlc_from_close(100 + np.arange(200) * 0.5)
    stats = compute_technical_stats(history["Close"], history)
    assert stats.adx is not None and stats.adx > 40 and stats.adx_change_12 is not None
    assert compute_technical_stats(history["Close"]).adx is None  # closes only: no ADX, never guessed
