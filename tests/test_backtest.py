import math

import pandas as pd
import pytest

from analysis.backtest import (
    backtest_beta_stability,
    backtest_momentum_persistence,
    backtest_rsi_reaction,
    backtest_support_resistance_reaction,
    backtest_volatility_regime,
    compute_beta,
)


def _make_series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=dates)


def _cyclical_series(cycles: int, up_days: int = 20, down_days: int = 10) -> pd.Series:
    """6 repeating cycles of a clean rally then a clean pullback — RSI
    reliably crosses overbought during the rally and oversold during the
    pullback, at exactly the same relative offset every cycle, so the
    backtest result is fully hand-predictable rather than incidental."""
    prices = [100.0]
    for _ in range(cycles):
        for _ in range(up_days):
            prices.append(prices[-1] * 1.015)
        for _ in range(down_days):
            prices.append(prices[-1] * 0.97)
    return _make_series(prices)


def test_backtest_rsi_reaction_none_on_too_short_series():
    ob, os_ = backtest_rsi_reaction(_make_series([100.0] * 10))
    assert ob is None
    assert os_ is None


def test_backtest_rsi_reaction_detects_reliable_reversal_pattern():
    # Empirically calibrated: this exact cycle shape puts every RSI
    # overbought/oversold episode's start day + 16 trading days squarely
    # inside the opposite phase, for every one of the 6 cycles.
    series = _cyclical_series(cycles=6)
    overbought, oversold = backtest_rsi_reaction(series, forward_days=16)

    assert overbought.occurrences == 6
    assert overbought.avg_forward_return_pct < 0
    assert overbought.reversal_rate_pct == pytest.approx(100.0)

    assert oversold.occurrences == 5
    assert oversold.avg_forward_return_pct > 0
    assert oversold.reversal_rate_pct == pytest.approx(100.0)


def test_backtest_rsi_reaction_collapses_consecutive_extreme_days_into_one_episode():
    # A single long rally holds RSI >= 70 for many consecutive days —
    # this must count as ONE episode, not one per day, or the average
    # would be dominated by near-duplicate overlapping samples.
    series = _cyclical_series(cycles=6, up_days=25)  # longer rally, more consecutive overbought days
    overbought, _ = backtest_rsi_reaction(series, forward_days=16, min_occurrences=5)
    assert overbought.occurrences == 6  # still one per cycle, not one per overbought day


def test_backtest_rsi_reaction_none_below_min_occurrences():
    # Only 2 cycles — fewer real episodes than the default min_occurrences.
    series = _cyclical_series(cycles=2)
    overbought, oversold = backtest_rsi_reaction(series, forward_days=16, min_occurrences=5)
    assert overbought is None
    assert oversold is None


def test_compute_beta_identical_series_is_one():
    series = _make_series([100.0 + i + (i % 5) for i in range(120)])
    assert compute_beta(series, series) == pytest.approx(1.0)


def test_compute_beta_none_without_enough_observations():
    series = _make_series([100.0, 101.0, 102.0])
    assert compute_beta(series, series) is None


def test_backtest_beta_stability_stable_when_self_compared():
    series = _make_series([100.0 * (1.001**i) + (i % 7) for i in range(400)])
    result = backtest_beta_stability(series, series)
    assert result.beta_3m == pytest.approx(1.0, abs=1e-6)
    assert result.beta_1y == pytest.approx(1.0, abs=1e-6)
    assert result.stable is True


def test_backtest_beta_stability_unstable_when_windows_disagree():
    # Stock moves in lockstep with the index for the first half, then
    # moves at triple amplitude for the second half — a real beta shift,
    # not noise, so the two windows must disagree by a wide margin.
    index_returns = [0.01 if i % 2 == 0 else -0.01 for i in range(300)]
    index_prices = [100.0]
    for r in index_returns:
        index_prices.append(index_prices[-1] * (1 + r))

    stock_prices = [100.0]
    for i, r in enumerate(index_returns):
        multiplier = 1.0 if i < 150 else 3.0
        stock_prices.append(stock_prices[-1] * (1 + r * multiplier))

    result = backtest_beta_stability(_make_series(stock_prices), _make_series(index_prices))
    assert result.beta_full_history is not None
    assert result.beta_3m is not None
    assert abs(result.beta_3m - result.beta_full_history) > 0.5
    assert result.stable is False


def test_backtest_momentum_persistence_none_on_short_series():
    assert backtest_momentum_persistence(_make_series([100.0] * 30)) is None


def test_backtest_momentum_persistence_detects_mean_reversion():
    # A strong trailing move is always immediately erased by an equal
    # and opposite move — textbook mean reversion, not incidental.
    prices = [100.0]
    up = True
    for _ in range(20):
        step = 1.30 if up else 1 / 1.30
        for _ in range(21):
            prices.append(prices[-1] * (step ** (1 / 21)))
        up = not up
    result = backtest_momentum_persistence(_make_series(prices))
    assert result is not None
    assert result.correlation < -0.15
    assert result.interpretation == "mean_reverting"


def test_backtest_momentum_persistence_detects_persistence():
    # A perfectly smooth, constant-rate exponential curve is actually a
    # bad test case here: every 21-day trailing return is identical by
    # construction, so there's no real variance for a correlation to
    # measure at all (dominated by float noise, not signal). Real
    # persistence needs distinct, multi-window REGIMES: alternating
    # fast/slow growth blocks, each much longer than one sample window,
    # so most trailing/forward pairs fall entirely inside one regime and
    # trailing return closely predicts forward return.
    rates = [0.006, 0.0005] * 3
    prices = [100.0]
    for rate in rates:
        for _ in range(105):
            prices.append(prices[-1] * (1 + rate))
    result = backtest_momentum_persistence(_make_series(prices))
    assert result is not None
    assert result.correlation > 0.15
    assert result.interpretation == "persistent"


def test_backtest_volatility_regime_none_on_too_short_series():
    assert backtest_volatility_regime(_make_series([100.0] * 15)) is None


def test_backtest_volatility_regime_detects_contraction_before_expansion():
    # 15 cycles of a quiet phase (tiny back-and-forth noise, low rolling
    # volatility) followed by a sharp breakout move (large absolute
    # change) — a real "coiled spring" pattern by construction, so the
    # average absolute move following LOW-vol episodes should clearly
    # exceed the average following HIGH-vol episodes (which mostly land
    # back in the next quiet phase).
    prices = [100.0]
    for _ in range(15):
        for i in range(15):
            prices.append(prices[-1] * (1.001 if i % 2 == 0 else 0.999))
        for _ in range(10):
            prices.append(prices[-1] * 1.05)
    result = backtest_volatility_regime(_make_series(prices), forward_days=10, min_episodes=8)
    assert result is not None
    assert result.low_vol_avg_abs_move_pct > result.high_vol_avg_abs_move_pct
    assert result.low_vol_episodes >= 8
    assert result.high_vol_episodes >= 8


def test_backtest_support_resistance_reaction_none_on_too_short_series():
    assert backtest_support_resistance_reaction(_make_series([100.0] * 30)) is None


def test_backtest_support_resistance_reaction_detects_clean_bounce():
    # A short-period oscillation (12 days) relative to the 60-day S/R
    # window and a 5-day forward check — price passes through the "near
    # support/resistance" zone quickly (not a slow multi-day approach
    # that would be caught mid-move), so each touch is a genuine test of
    # a level that's already established, immediately followed by a
    # clean reversal.
    n_cycles, period = 40, 12
    prices = [100.0 + 5.0 * math.sin(2 * math.pi * k / period) for k in range(n_cycles * period)]
    result = backtest_support_resistance_reaction(_make_series(prices), forward_days=5, min_tests=5)
    assert result is not None
    assert result.support_hold_rate_pct == pytest.approx(100.0)
    assert result.resistance_reject_rate_pct == pytest.approx(100.0)
    assert result.support_tests >= 5
    assert result.resistance_tests >= 5
