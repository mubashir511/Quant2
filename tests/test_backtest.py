import math

import pandas as pd
import pytest

from analysis.backtest import (
    ChartPatternBacktest,
    FavorableExcursionStats,
    RSIReactionBacktest,
    SupportResistanceBacktest,
    _compute_favorable_excursion,
    _simulate_trades,
    _summarize_trade_results,
    backtest_beta_stability,
    backtest_chart_pattern_reaction,
    backtest_momentum_persistence,
    backtest_rsi_reaction,
    backtest_support_resistance_reaction,
    backtest_volatility_regime,
    classify_backtest_favorability,
    compute_beta,
)
from analysis.chart_structure import detect_chart_patterns


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


def _zigzag(extrema: list[float], seg_len: int) -> list[float]:
    """Piecewise-linear path through `extrema`, `seg_len` bars per leg —
    same helper as tests/test_chart_structure.py's own _zigzag, kept as
    an independent copy since this file doesn't otherwise depend on that
    one. Gives full control over exactly where swing points land, so a
    double-top/double-bottom shape can be verified deterministically
    rather than hoped for from noisy data."""
    prices: list[float] = []
    for i in range(len(extrema) - 1):
        start, end = extrema[i], extrema[i + 1]
        for step in range(seg_len):
            prices.append(start + (end - start) * step / seg_len)
    prices.append(extrema[-1])
    return prices


def _double_bottom_series(n_cycles: int, seg_len: int = 12) -> pd.Series:
    """Chains `n_cycles` real "W" double-bottom shapes back to back: peak
    (150) -> T1 (100) -> neckline bounce (120) -> T2 (100.2, within
    analysis.chart_structure.SR_CLUSTER_TOLERANCE_PCT of T1 by
    construction) -> breakout rally back to 150, which becomes the next
    cycle's own starting peak. Mirrors the exact shape
    tests/test_chart_structure.py::test_detects_double_bottom_with_a_real_
    intervening_bounce already verifies detect_chart_patterns confirms as
    a real double_bottom, just repeated. seg_len=12/lookback=40 (used by
    the tests below) was empirically verified against the real running
    backtest_chart_pattern_reaction to produce a clean, all-winning
    simulated long — see that test's own comment."""
    extrema = [150.0]
    for _ in range(n_cycles):
        extrema += [100.0, 120.0, 100.2, 150.0]
    return _make_series(_zigzag(extrema, seg_len))


def _double_top_series(n_cycles: int, seg_len: int = 12) -> pd.Series:
    """Mirror of _double_bottom_series for an "M" double-top shape:
    trough (100) -> P1 (150) -> pullback (130) -> P2 (149.8, within
    tolerance of P1) -> decline back to 100."""
    extrema = [100.0]
    for _ in range(n_cycles):
        extrema += [150.0, 130.0, 149.8, 100.0]
    return _make_series(_zigzag(extrema, seg_len))


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


# --- _compute_favorable_excursion: the decoupled magnitude measurement ---


def _linear_grid_ohlc(n_bars: int, step: float) -> pd.DataFrame:
    """A perfectly linear price ramp (High == Low == Close == price at
    every bar) — `step` per bar (positive: rallies; negative: declines).
    Gives a fully hand-computable "best price reached" for any forward
    window: it's always exactly the window's own last bar, `step *
    window_len` away from the entry."""
    prices = _make_series([100.0 + step * i for i in range(n_bars)])
    return pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [100.0] * n_bars},
        index=prices.index,
    )


def test_compute_favorable_excursion_measures_the_real_peak_past_the_fixed_target():
    # A straight +2.0/bar rally, 5 entries spaced 5 bars apart, each with
    # a full 5-bar forward window ahead of it (30 bars total, last entry
    # at position 20). Every entry's own forward window is identical by
    # construction: +2.0/bar for 5 bars = +10.0 total, so with atr=1.0
    # and stop_atr_multiple=1.5 (stop_distance=1.5), every entry's
    # excursion is EXACTLY 10.0/1.5 ≈ 6.67R — clearly past the flat 2.0R
    # this module's own fixed reward:risk convention would cap a win at,
    # proving this figure isn't just a relabeling of that capped number.
    ohlc = _linear_grid_ohlc(n_bars=30, step=2.0)
    atr = pd.Series([1.0] * 30, index=ohlc.index)
    result = _compute_favorable_excursion(
        ohlc, entry_positions=[0, 5, 10, 15, 20], side="buy", atr=atr, stop_atr_multiple=1.5, horizon_bars=5,
    )
    assert result is not None
    assert result.sample_size == 5
    assert result.horizon_bars == 5
    assert result.avg_r == pytest.approx(10.0 / 1.5)
    assert result.median_r == pytest.approx(10.0 / 1.5)


def test_compute_favorable_excursion_short_side_measures_the_low_not_the_high():
    # Mirror of the above with a straight decline — a sell's favorable
    # direction is DOWN, so this must read off window_low, not window_high.
    ohlc = _linear_grid_ohlc(n_bars=30, step=-2.0)
    atr = pd.Series([1.0] * 30, index=ohlc.index)
    result = _compute_favorable_excursion(
        ohlc, entry_positions=[0, 5, 10, 15, 20], side="sell", atr=atr, stop_atr_multiple=1.5, horizon_bars=5,
    )
    assert result is not None
    assert result.avg_r == pytest.approx(10.0 / 1.5)
    assert result.median_r == pytest.approx(10.0 / 1.5)


def test_compute_favorable_excursion_never_negative_even_when_price_never_recovers():
    # A buy entry on a straight DECLINE — price never once trades back
    # above the entry price within the forward window, so the real
    # "best favorable price reached" is the entry price itself (the
    # trade never had a favorable moment) — must read exactly 0.0, never
    # negative, since price genuinely WAS at entry_price at time zero.
    ohlc = _linear_grid_ohlc(n_bars=30, step=-2.0)
    atr = pd.Series([1.0] * 30, index=ohlc.index)
    result = _compute_favorable_excursion(
        ohlc, entry_positions=[0, 5, 10, 15, 20], side="buy", atr=atr, stop_atr_multiple=1.5, horizon_bars=5,
    )
    assert result is not None
    assert result.avg_r == pytest.approx(0.0)
    assert result.median_r == pytest.approx(0.0)


def test_compute_favorable_excursion_drops_entries_too_close_to_the_end_of_history():
    # Same fixture as the first test above (5 valid entries with a full
    # 5-bar window each), plus a 6th entry at position 28 — only 1 bar of
    # real history remains ahead of it (28+5=33 >= 30, the series length)
    # — this entry must be EXCLUDED from the sample entirely, not counted
    # with a truncated/partial window (which would silently understate
    # its true excursion) — a stricter rule than _simulate_trades' own
    # lenient partial-window-for-timeout acceptance.
    ohlc = _linear_grid_ohlc(n_bars=30, step=2.0)
    atr = pd.Series([1.0] * 30, index=ohlc.index)
    result = _compute_favorable_excursion(
        ohlc, entry_positions=[0, 5, 10, 15, 20, 28], side="buy", atr=atr, stop_atr_multiple=1.5, horizon_bars=5,
    )
    assert result is not None
    assert result.sample_size == 5  # the near-end entry at 28 was dropped, not the other 5


def test_compute_favorable_excursion_none_below_min_sample():
    # Only 4 valid entries — one short of _MIN_EXCURSION_SAMPLE (5).
    ohlc = _linear_grid_ohlc(n_bars=30, step=2.0)
    atr = pd.Series([1.0] * 30, index=ohlc.index)
    result = _compute_favorable_excursion(
        ohlc, entry_positions=[0, 5, 10, 15], side="buy", atr=atr, stop_atr_multiple=1.5, horizon_bars=5,
    )
    assert result is None


def test_compute_favorable_excursion_median_differs_from_mean_on_a_skewed_sample():
    # 6 entries every 2 bars, horizon_bars=1 (each entry's window is
    # exactly the single next bar): 5 of them peak 1.0 above entry, the
    # 6th (an outlier) peaks 20.0 above entry. With atr=1.0 and
    # stop_atr_multiple=1.0 (stop_distance=1.0), R-multiples are exactly
    # [1, 1, 1, 1, 1, 20] — even sample size, so median is the mean of
    # the two middle sorted values (both 1.0), clearly BELOW the mean
    # (pulled up hard by the one outlier) — exactly the skew this
    # feature exists to report honestly via the median, not just a mean
    # that one big trend run can dominate.
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    close = [100.0] * 12
    high = [100.0] * 12
    for entry_pos, peak in [(0, 101.0), (2, 101.0), (4, 101.0), (6, 101.0), (8, 101.0), (10, 120.0)]:
        high[entry_pos + 1] = peak
    ohlc = pd.DataFrame(
        {"Open": close, "High": high, "Low": close, "Close": close, "Volume": [100.0] * 12}, index=dates
    )
    atr = pd.Series([1.0] * 12, index=dates)
    result = _compute_favorable_excursion(
        ohlc, entry_positions=[0, 2, 4, 6, 8, 10], side="buy", atr=atr, stop_atr_multiple=1.0, horizon_bars=1,
    )
    assert result is not None
    assert result.sample_size == 6
    assert result.avg_r == pytest.approx(25.0 / 6.0)
    assert result.median_r == pytest.approx(1.0)
    assert result.median_r < result.avg_r


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
    # convention analysis.technical.compute_atr already uses.
    series = _cyclical_series(cycles=6, up_days=14, up_rate=0.02, down_days=5, down_rate=0.06)
    bare = pd.DataFrame({"Open": series, "Close": series, "Volume": [100.0] * len(series)}, index=series.index)
    overbought, oversold = backtest_rsi_reaction(bare, max_holding_bars=10)
    assert overbought is None
    assert oversold is None


def test_backtest_rsi_reaction_attaches_excursion_stats():
    # Reuses the clean-winner overbought fixture above, but with a small
    # excursion_horizon_bars override (5, matching the fixture's own
    # down_days) so the real magnitude of the SHARP 6%/day decline shows
    # up within a compact window — empirically verified to clearly
    # exceed the parent backtest's own fixed-target avg_r_multiple (2.0),
    # proving this is genuinely new evidence, not a relabeling of it.
    series = _cyclical_series(cycles=6, up_days=14, up_rate=0.02, down_days=5, down_rate=0.06)
    overbought, _ = backtest_rsi_reaction(_make_ohlc(series), max_holding_bars=10, excursion_horizon_bars=5)
    assert overbought.excursion is not None
    assert overbought.excursion.sample_size == 6
    assert overbought.excursion.horizon_bars == 5
    assert overbought.excursion.median_r > overbought.avg_r_multiple
    assert overbought.excursion.avg_r > overbought.avg_r_multiple


def test_backtest_rsi_reaction_excursion_none_when_sample_too_thin_for_the_horizon():
    # Same fixture, but excursion_horizon_bars=30 — long enough that
    # fewer than _MIN_EXCURSION_SAMPLE entries near the end of this
    # 115-bar series keep a real, full 30-bar look-forward window
    # (empirically verified: all 6 entries drop out at this horizon) —
    # while the win-rate/avg-R figures (driven by the unrelated,
    # unchanged max_holding_bars=10) still resolve completely normally,
    # proving the two gates are genuinely independent of each other.
    series = _cyclical_series(cycles=6, up_days=14, up_rate=0.02, down_days=5, down_rate=0.06)
    overbought, _ = backtest_rsi_reaction(_make_ohlc(series), max_holding_bars=10, excursion_horizon_bars=30)
    assert overbought.excursion is None
    assert overbought.trades == 6
    assert overbought.win_rate_pct == pytest.approx(100.0)
    assert overbought.avg_r_multiple == pytest.approx(2.0)


# --- backtest_chart_pattern_reaction ---


def test_backtest_chart_pattern_reaction_none_on_too_short_series():
    db, dt = backtest_chart_pattern_reaction(_make_ohlc(_make_series([100.0] * 10)))
    assert db is None
    assert dt is None


def test_backtest_chart_pattern_reaction_none_when_no_high_low_in_the_feed():
    # Same convention as backtest_rsi_reaction's own equivalent test — no
    # High/Low means no real ATR-based stop/target can be derived, so
    # this must degrade to None rather than fabricate one, even though
    # the series itself has plenty of real double-bottom shapes in it.
    series = _double_bottom_series(n_cycles=6)
    bare = pd.DataFrame({"Open": series, "Close": series, "Volume": [100.0] * len(series)}, index=series.index)
    db, dt = backtest_chart_pattern_reaction(bare, lookback=40, max_holding_bars=10)
    assert db is None
    assert dt is None


def test_backtest_chart_pattern_reaction_simulated_long_wins_cleanly_after_double_bottom():
    # Empirically verified against the real running code (this project's
    # own established practice — see tests/test_chart_structure.py's own
    # test suite for the precedent): 6 chained double-bottom "W" cycles
    # (see _double_bottom_series), scanned with a reduced lookback=40 (so
    # the fixture doesn't need STRUCTURE_LOOKBACK=90 bars of warm-up per
    # cycle) and the default 5-bar scan step. Each breakout rally back to
    # 150 is long/steep enough that even the worst-case few-bar entry lag
    # (see ChartPatternBacktest's own docstring) still has runway left to
    # hit the 2:1 target before max_holding_bars — a real, clean,
    # hand-verifiable winning pattern, not a coincidence.
    series = _double_bottom_series(n_cycles=6, seg_len=12)
    double_bottom, double_top = backtest_chart_pattern_reaction(
        _make_ohlc(series), lookback=40, max_holding_bars=10
    )

    assert double_bottom.trades == 10
    assert double_bottom.wins == 10
    assert double_bottom.losses == 0
    assert double_bottom.win_rate_pct == pytest.approx(100.0)
    assert double_bottom.avg_r_multiple == pytest.approx(2.0)
    # This fixture never forms a real double_top shape — genuinely no
    # signal on that side, and this also validates the per-side
    # independent gating (a real double_bottom result doesn't require a
    # double_top result to also exist).
    assert double_top is None


def test_backtest_chart_pattern_reaction_simulated_short_wins_cleanly_after_double_top():
    # Mirror of the double-bottom test above using the "M" shape.
    series = _double_top_series(n_cycles=6, seg_len=12)
    double_bottom, double_top = backtest_chart_pattern_reaction(
        _make_ohlc(series), lookback=40, max_holding_bars=10
    )

    assert double_bottom is None
    assert double_top.trades == 10
    assert double_top.wins == 10
    assert double_top.losses == 0
    assert double_top.win_rate_pct == pytest.approx(100.0)
    assert double_top.avg_r_multiple == pytest.approx(2.0)


def test_backtest_chart_pattern_reaction_collapses_persisting_pattern_into_one_occurrence():
    # Reuses the 6-cycle double-bottom fixture above. Without collapsing
    # an ongoing, still-detected pattern across consecutive scan steps
    # into one occurrence, the raw number of scan steps where
    # detect_chart_patterns' own output contains "double_bottom" is
    # nearly 2.5x the real, deduped trade count (verified directly here
    # by independently re-walking the same scan with detect_chart_patterns
    # itself, not just asserted) — real evidence the dedup is doing
    # something, not a no-op.
    series = _double_bottom_series(n_cycles=6, seg_len=12)
    ohlc = _make_ohlc(series)
    lookback = 40
    max_holding_bars = 10

    raw_present_steps = 0
    for t in range(lookback, len(ohlc) - max_holding_bars, 5):
        names = {p.name for p in detect_chart_patterns(ohlc.iloc[: t + 1], lookback=lookback)}
        if "double_bottom" in names:
            raw_present_steps += 1

    double_bottom, _ = backtest_chart_pattern_reaction(ohlc, lookback=lookback, max_holding_bars=max_holding_bars)
    assert double_bottom.trades == 10
    assert double_bottom.trades < raw_present_steps


def test_backtest_chart_pattern_reaction_none_below_min_occurrences():
    # Only 2 cycles of the verified-winning shape above — fewer real
    # occurrences than the default min_occurrences.
    series = _double_bottom_series(n_cycles=2, seg_len=12)
    double_bottom, double_top = backtest_chart_pattern_reaction(
        _make_ohlc(series), lookback=40, max_holding_bars=10
    )
    assert double_bottom is None
    assert double_top is None


def test_backtest_chart_pattern_reaction_attaches_excursion_stats():
    # Reuses the 6-cycle double-bottom fixture — empirically verified to
    # attach a real, populated excursion reading at the default 90-bar
    # horizon. sample_size <= trades is an explicit, testable invariant:
    # the longer excursion horizon can only DROP entries too close to
    # the tail of history (it needs a full 90-bar window, not just the
    # 10-bar max_holding_bars the win/loss simulation needs), never add
    # any beyond what the pattern scan itself found.
    series = _double_bottom_series(n_cycles=6, seg_len=12)
    double_bottom, _ = backtest_chart_pattern_reaction(_make_ohlc(series), lookback=40, max_holding_bars=10)
    assert double_bottom.excursion is not None
    assert double_bottom.excursion.sample_size <= double_bottom.trades


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


def test_backtest_support_resistance_reaction_attaches_per_side_excursion_stats():
    # Reuses the clean-bounce sine-wave fixture — the 480-bar series
    # (40 cycles x 12-day period) comfortably clears the default 90-bar
    # excursion horizon for real support AND resistance touches, so both
    # sides should independently come back populated, each with a
    # sensible sample size.
    n_cycles, period = 40, 12
    prices = [100.0 + 5.0 * math.sin(2 * math.pi * k / period) for k in range(n_cycles * period)]
    ohlc = _make_ohlc(_make_series(prices), pad_pct=0.0005)
    result = backtest_support_resistance_reaction(ohlc, max_holding_bars=5, min_tests=5)
    assert result is not None
    assert result.support_excursion is not None
    assert result.support_excursion.sample_size >= 5
    assert result.resistance_excursion is not None
    assert result.resistance_excursion.sample_size >= 5


# --- classify_backtest_favorability ---


def _rsi_bt(avg_r: float, excursion: FavorableExcursionStats | None = None) -> RSIReactionBacktest:
    return RSIReactionBacktest(
        "overbought", 70.0, 10, 5, 5, 0, 50.0, avg_r, 1.5, 3.0, 10, excursion=excursion,
    )


def _sr_bt(
    support_avg_r: float,
    resistance_avg_r: float,
    support_excursion: FavorableExcursionStats | None = None,
    resistance_excursion: FavorableExcursionStats | None = None,
) -> SupportResistanceBacktest:
    return SupportResistanceBacktest(
        10, 5, 5, 0, 50.0, support_avg_r, 10, 5, 5, 0, 50.0, resistance_avg_r, 1.5, 3.0, 10,
        support_excursion=support_excursion, resistance_excursion=resistance_excursion,
    )


def _pattern_bt(avg_r: float, excursion: FavorableExcursionStats | None = None) -> ChartPatternBacktest:
    return ChartPatternBacktest(
        "double_bottom", 10, 5, 5, 0, 50.0, avg_r, 1.5, 3.0, 10, excursion=excursion,
    )


def _excursion(median_r: float) -> FavorableExcursionStats:
    return FavorableExcursionStats(sample_size=10, avg_r=median_r, median_r=median_r, horizon_bars=90)


def test_classify_backtest_favorability_all_favorable_buy_returns_supported_with_excursion():
    # Oversold RSI (avg_r=0.5) is the best of the two favorable long-side
    # backtests (double_bottom avg_r=0.2) — its own excursion must win.
    favorability, excursion_median_r = classify_backtest_favorability(
        "buy",
        rsi_overbought_backtest=None,
        rsi_oversold_backtest=_rsi_bt(0.5, excursion=_excursion(1.2)),
        support_resistance_backtest=None,
        double_bottom_backtest=_pattern_bt(0.2, excursion=_excursion(0.8)),
        double_top_backtest=None,
    )
    assert favorability == "supported"
    assert excursion_median_r == pytest.approx(1.2)


def test_classify_backtest_favorability_all_unfavorable_sell_returns_contradicted_with_no_excursion():
    favorability, excursion_median_r = classify_backtest_favorability(
        "sell",
        rsi_overbought_backtest=_rsi_bt(-0.3),
        rsi_oversold_backtest=None,
        support_resistance_backtest=None,
        double_bottom_backtest=None,
        double_top_backtest=_pattern_bt(-0.1),
    )
    assert favorability == "contradicted"
    assert excursion_median_r is None


def test_classify_backtest_favorability_genuine_mix_returns_mixed_with_no_excursion():
    favorability, excursion_median_r = classify_backtest_favorability(
        "buy",
        rsi_overbought_backtest=None,
        rsi_oversold_backtest=_rsi_bt(0.5, excursion=_excursion(2.0)),
        support_resistance_backtest=_sr_bt(support_avg_r=-0.4, resistance_avg_r=0.0),
        double_bottom_backtest=None,
        double_top_backtest=None,
    )
    assert favorability == "mixed"
    assert excursion_median_r is None


def test_classify_backtest_favorability_no_relevant_backtests_returns_none_none():
    favorability, excursion_median_r = classify_backtest_favorability(
        "buy",
        rsi_overbought_backtest=None,
        rsi_oversold_backtest=None,
        support_resistance_backtest=None,
        double_bottom_backtest=None,
        double_top_backtest=None,
    )
    assert favorability is None
    assert excursion_median_r is None


def test_classify_backtest_favorability_all_neutral_returns_none_none():
    # Every relevant avg_r_multiple rounds to a dead-flat 0.00R — "no
    # real edge either way", genuinely different from real disagreement,
    # so this must degrade like missing data, not collapse into "mixed".
    favorability, excursion_median_r = classify_backtest_favorability(
        "buy",
        rsi_overbought_backtest=None,
        rsi_oversold_backtest=_rsi_bt(0.001),
        support_resistance_backtest=None,
        double_bottom_backtest=_pattern_bt(-0.001),
        double_top_backtest=None,
    )
    assert favorability is None
    assert excursion_median_r is None


def test_classify_backtest_favorability_best_avg_r_multiples_own_excursion_wins_even_when_it_is_none():
    # The best-avg_r_multiple favorable backtest (0.5) has NO excursion
    # data; a weaker favorable backtest (0.2) DOES — must NOT borrow it.
    favorability, excursion_median_r = classify_backtest_favorability(
        "buy",
        rsi_overbought_backtest=None,
        rsi_oversold_backtest=_rsi_bt(0.5, excursion=None),
        support_resistance_backtest=None,
        double_bottom_backtest=_pattern_bt(0.2, excursion=_excursion(3.0)),
        double_top_backtest=None,
    )
    assert favorability == "supported"
    assert excursion_median_r is None


def test_classify_backtest_favorability_ignores_the_wrong_side_backtest():
    # A strongly unfavorable OVERBOUGHT (short-side) backtest must never
    # be consulted for a BUY position — only the long-side one counts.
    favorability, excursion_median_r = classify_backtest_favorability(
        "buy",
        rsi_overbought_backtest=_rsi_bt(-5.0),
        rsi_oversold_backtest=_rsi_bt(0.5, excursion=_excursion(1.0)),
        support_resistance_backtest=None,
        double_bottom_backtest=None,
        double_top_backtest=_pattern_bt(-5.0),
    )
    assert favorability == "supported"
    assert excursion_median_r == pytest.approx(1.0)


def test_classify_backtest_favorability_support_resistance_uses_the_correct_side_field():
    # support_avg_r_multiple drives a "buy" read, resistance_avg_r_multiple
    # drives a "sell" read — never the other side's field.
    sr = _sr_bt(support_avg_r=0.6, resistance_avg_r=-0.6, support_excursion=_excursion(4.0))
    buy_favorability, buy_excursion = classify_backtest_favorability(
        "buy", None, None, sr, None, None,
    )
    sell_favorability, sell_excursion = classify_backtest_favorability(
        "sell", None, None, sr, None, None,
    )
    assert buy_favorability == "supported"
    assert buy_excursion == pytest.approx(4.0)
    assert sell_favorability == "contradicted"
    assert sell_excursion is None
