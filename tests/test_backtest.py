import math

import pandas as pd
import pytest

from analysis.backtest import (
    _simulate_trades,
    _summarize_trade_results,
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


def _make_ohlc(closes: pd.Series, pad_pct: float = 0.001) -> pd.DataFrame:
    """Wraps a bare Close series into a minimal OHLC frame — every
    directional backtest below now needs real High/Low to derive an
    ATR-based stop/target from (see analysis/backtest.py's own
    _rolling_atr). `pad_pct` is a small, constant intrabar band around
    each close, kept tight relative to the fixture's own real price
    moves so True Range stays dominated by genuine day-to-day
    volatility, not this padding."""
    high = closes * (1 + pad_pct)
    low = closes * (1 - pad_pct)
    return pd.DataFrame(
        {"Open": closes, "High": high, "Low": low, "Close": closes, "Volume": [100.0] * len(closes)},
        index=closes.index,
    )


def _cyclical_series(
    cycles: int, up_days: int = 20, up_rate: float = 0.015, down_days: int = 10, down_rate: float = 0.03
) -> pd.Series:
    """Repeating cycles of a clean rally then a clean pullback — RSI
    reliably crosses overbought/oversold at the same relative offset
    every cycle, so occurrence counts are fully hand-predictable."""
    prices = [100.0]
    for _ in range(cycles):
        for _ in range(up_days):
            prices.append(prices[-1] * (1 + up_rate))
        for _ in range(down_days):
            prices.append(prices[-1] * (1 - down_rate))
    return _make_series(prices)


# --- _simulate_trades / _summarize_trade_results: the shared engine ---


def test_simulate_trades_same_bar_stop_and_target_resolves_to_stop():
    # A real, unresolvable ambiguity without intrabar tick data: the very
    # next bar's range touches BOTH the long's stop (98.5 = 100 - 1.5*1.0)
    # AND its target (103.0 = 100 + 3.0*1.0) at once. _simulate_trades'
    # own documented convention is "stop wins" — never assume the more
    # favorable outcome — verified directly here rather than only
    # incidentally through a larger fixture.
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    ohlc = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 103.5, 100.0],
            "Low": [100.0, 98.0, 100.0],
            "Close": [100.0, 100.0, 100.0],
            "Volume": [100.0, 100.0, 100.0],
        },
        index=dates,
    )
    atr = pd.Series([1.0, 1.0, 1.0], index=dates)
    results = _simulate_trades(
        ohlc, [0], "buy", atr, stop_atr_multiple=1.5, target_atr_multiple=3.0, max_holding_bars=5
    )
    assert results == [(-1.0, "loss")]


def test_simulate_trades_marks_timeout_to_market_and_excludes_it_from_win_rate():
    # Neither stop (98.5) nor target (103.0) ever touched within the
    # 2-bar holding window — a real "timeout", marked to market at its
    # final close (100.5) rather than discarded, and explicitly NOT
    # counted as a win despite a positive mark-to-market R (a timeout is
    # genuinely undecided, not the same claim as a real target hit).
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    ohlc = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 100.5, 100.5],
            "Low": [100.0, 99.5, 99.5],
            "Close": [100.0, 100.5, 100.5],
            "Volume": [100.0, 100.0, 100.0],
        },
        index=dates,
    )
    atr = pd.Series([1.0, 1.0, 1.0], index=dates)
    results = _simulate_trades(
        ohlc, [0], "buy", atr, stop_atr_multiple=1.5, target_atr_multiple=3.0, max_holding_bars=2
    )
    assert len(results) == 1
    r_multiple, outcome = results[0]
    assert outcome == "timeout"
    assert r_multiple == pytest.approx(0.5 / 1.5)

    summary = _summarize_trade_results(results)
    assert summary.trades == 1
    assert summary.timeouts == 1
    assert summary.wins == 0
    assert summary.losses == 0
    assert summary.win_rate_pct is None  # nothing definitively resolved yet
    assert summary.avg_r_multiple == pytest.approx(0.5 / 1.5)


def test_backtest_rsi_reaction_none_on_too_short_series():
    ob, os_ = backtest_rsi_reaction(_make_ohlc(_make_series([100.0] * 10)))
    assert ob is None
    assert os_ is None


def test_backtest_rsi_reaction_simulated_short_wins_cleanly_after_overbought():
    # Empirically verified against the real running code (this project's
    # own established practice — see analysis/chart_structure.py's test
    # suite for the precedent): a 14-day rally (up_days matches
    # _RSI_WINDOW, so RSI only crosses 70 right at the LAST up day, not
    # partway through with more upside still to come) immediately
    # followed by a sharp 5-day, 6%/day decline. The simulated SHORT
    # entered at each overbought episode's start never sees the stop
    # (1.5x ATR above entry) before the target (3x ATR below it, i.e.
    # +2.0R) — a real, clean, hand-verifiable winning pattern, not a
    # coincidence of "return N days later" the way the old measurement
    # was vulnerable to (a rally that kept extending past the RSI
    # signal before finally reversing used to still get counted as a
    # "reversal" by that measurement despite a real stop being hit
    # first — confirmed by directly running the old-shaped fixture
    # through the new engine during development).
    series = _cyclical_series(cycles=6, up_days=14, up_rate=0.02, down_days=5, down_rate=0.06)
    overbought, oversold = backtest_rsi_reaction(_make_ohlc(series), max_holding_bars=10)

    assert overbought.trades == 6
    assert overbought.wins == 6
    assert overbought.losses == 0
    assert overbought.win_rate_pct == pytest.approx(100.0)
    assert overbought.avg_r_multiple == pytest.approx(2.0)
    # This particular fixture's down-phase is too short/shallow to ever
    # push RSI to oversold — genuinely no signal on that side, not a bug.
    assert oversold is None


def test_backtest_rsi_reaction_simulated_long_wins_cleanly_after_oversold():
    # Mirror shape for the LONG side: a short, sharp rally (RSI never
    # reaches overbought) followed by a longer, steady decline that
    # reliably pushes RSI to oversold, then a clean, fast recovery.
    series = _cyclical_series(cycles=6, up_days=6, up_rate=0.03, down_days=10, down_rate=0.04)
    overbought, oversold = backtest_rsi_reaction(_make_ohlc(series), max_holding_bars=10)

    assert overbought is None
    assert oversold.trades == 5
    assert oversold.wins == 5
    assert oversold.losses == 0
    assert oversold.win_rate_pct == pytest.approx(100.0)
    assert oversold.avg_r_multiple == pytest.approx(2.0)


def test_backtest_rsi_reaction_collapses_consecutive_extreme_days_into_one_episode():
    # A single long rally holds RSI >= 70 for many consecutive days —
    # this must count as ONE episode (one simulated trade), not one per
    # day, or the sample would be inflated with near-duplicate,
    # non-independent trades all sharing the same real outcome.
    series = _cyclical_series(cycles=6, up_days=25)  # longer rally, more consecutive overbought days
    overbought, _ = backtest_rsi_reaction(_make_ohlc(series), max_holding_bars=10, min_occurrences=5)
    assert overbought.trades == 6  # still one per cycle, not one per overbought day


def test_backtest_rsi_reaction_none_below_min_occurrences():
    # Only 2 cycles of the verified-winning shape above — fewer real
    # episodes than the default min_occurrences.
    series = _cyclical_series(cycles=2, up_days=14, up_rate=0.02, down_days=5, down_rate=0.06)
    overbought, oversold = backtest_rsi_reaction(_make_ohlc(series), max_holding_bars=10, min_occurrences=5)
    assert overbought is None
    assert oversold is None


def test_backtest_rsi_reaction_none_when_no_high_low_in_the_feed():
    # PSX's own EOD feed (data/psx_source.py) has no High/Low at all —
    # a real ATR-based stop/target genuinely can't be derived, so this
    # must degrade to None rather than silently fabricate one, same
    # convention analysis.technical._compute_atr already uses.
    series = _cyclical_series(cycles=6, up_days=14, up_rate=0.02, down_days=5, down_rate=0.06)
    bare = pd.DataFrame({"Open": series, "Close": series, "Volume": [100.0] * len(series)}, index=series.index)
    overbought, oversold = backtest_rsi_reaction(bare, max_holding_bars=10)
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
    assert backtest_support_resistance_reaction(_make_ohlc(_make_series([100.0] * 30))) is None


def test_backtest_support_resistance_reaction_detects_clean_bounce():
    # A short-period oscillation (12 days) relative to the 60-day S/R
    # window and a 5-bar max hold — price passes through the "near
    # support/resistance" zone quickly (not a slow multi-day approach
    # that would be caught mid-move), so each touch is a genuine test of
    # a level that's already established, immediately followed by a
    # clean reversal toward the opposite extreme — verified empirically
    # to clear the simulated ATR-based target well before any stop.
    n_cycles, period = 40, 12
    prices = [100.0 + 5.0 * math.sin(2 * math.pi * k / period) for k in range(n_cycles * period)]
    ohlc = _make_ohlc(_make_series(prices), pad_pct=0.0005)
    result = backtest_support_resistance_reaction(ohlc, max_holding_bars=5, min_tests=5)
    assert result is not None
    assert result.support_win_rate_pct == pytest.approx(100.0)
    assert result.resistance_win_rate_pct == pytest.approx(100.0)
    assert result.support_tests >= 5
    assert result.resistance_tests >= 5
    assert result.support_avg_r_multiple > 0
    assert result.resistance_avg_r_multiple > 0


def test_backtest_support_resistance_reaction_none_when_no_high_low_in_the_feed():
    n_cycles, period = 40, 12
    prices = [100.0 + 5.0 * math.sin(2 * math.pi * k / period) for k in range(n_cycles * period)]
    series = _make_series(prices)
    bare = pd.DataFrame({"Open": series, "Close": series, "Volume": [100.0] * len(series)}, index=series.index)
    assert backtest_support_resistance_reaction(bare, max_holding_bars=5, min_tests=5) is None
