import numpy as np
import pandas as pd
import pytest

from tools.studies.common import closed_bucket_flag, simulate_trade
from tools.studies.studies import cost_drag_by_stop_study, exit_rule_study, htf_alignment_study


def _frame(n=6000, seed=3, drift=0.0, freq="5min"):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(drift, 0.15, n))
    high = close + np.abs(rng.normal(0.08, 0.04, n))
    low = close - np.abs(rng.normal(0.08, 0.04, n))
    idx = pd.date_range("2026-01-01", periods=n, freq=freq)
    return pd.DataFrame({"High": high, "Low": low, "Close": close}, index=idx)


def _brute_force_flag(close: pd.Series, t: pd.Timestamp, bucket: str, window: int) -> float:
    """The flag using ONLY data a trader standing at time t could have: buckets whose end is <= t."""
    known = close[close.index < t]
    buckets = known.resample(bucket).last().dropna()
    delta = pd.tseries.frequencies.to_offset(bucket)
    buckets = buckets[buckets.index + delta <= t]  # drop the bucket t itself sits in
    if len(buckets) < window:
        return np.nan
    sma = buckets.rolling(window).mean().iloc[-1]
    return float(buckets.iloc[-1] > sma)


@pytest.mark.parametrize("bucket,window", [("1h", 5), ("4h", 10), ("1D", 3)])
def test_closed_bucket_flag_never_sees_its_own_buckets_future_close(bucket, window):
    close = _frame(n=2500)["Close"]
    flag = closed_bucket_flag(close, bucket, sma_window=window)
    for t in close.index[::37]:
        expected = _brute_force_flag(close, t, bucket, window)
        got = flag.loc[t]
        assert (np.isnan(expected) and np.isnan(got)) or expected == got, (t, expected, got)


def test_a_spike_in_the_last_bar_of_a_bucket_cannot_flip_the_flag_inside_that_bucket():
    """The exact leak of the first alignment study: the bucket's own final close must not reach earlier bars."""
    close = _frame(n=1200, drift=0.0)["Close"].copy()
    bucket_start = close.index[12 * 50]  # a 1h bucket boundary
    inside = close.index[(close.index >= bucket_start) & (close.index < bucket_start + pd.Timedelta("1h"))]
    baseline = closed_bucket_flag(close, "1h", 5).loc[inside]
    spiked = close.copy()
    spiked.loc[inside[-1]] = close.max() + 500  # the bucket's LAST bar explodes upward
    after = closed_bucket_flag(spiked, "1h", 5).loc[inside]
    assert (baseline.to_numpy()[:-1] == after.to_numpy()[:-1]).all() or np.isnan(baseline.to_numpy()).all()
    # ...and only once the bucket has closed does the next bucket's first bar see it.
    next_bar = inside[-1] + pd.Timedelta("5min")
    assert closed_bucket_flag(spiked, "1h", 5).loc[next_bar] == 1.0


def test_the_flag_needs_a_datetime_index():
    with pytest.raises(TypeError):
        closed_bucket_flag(pd.Series([1.0, 2.0, 3.0]), "1h")


def test_the_flag_is_nan_until_enough_closed_buckets_exist():
    close = _frame(n=400)["Close"]
    flag = closed_bucket_flag(close, "1h", 20)
    assert flag.iloc[: 12 * 20].isna().all()  # fewer than 20 CLOSED hourly buckets exist: no flag, never a guess
    assert flag.iloc[-1] in (0.0, 1.0)  # ~33 hours in: the SMA(20) of closed buckets is computable


# ---- simulate_trade ---------------------------------------------------------------------------------------

def _path(highs, lows, closes):
    return np.array(highs, float), np.array(lows, float), np.array(closes, float)


def test_simulate_trade_resolves_stop_target_and_the_stop_first_convention():
    hi, lo, cl = _path([100, 100.5, 102.5, 104], [100, 98.9, 100, 103], [100, 100, 102, 104])
    assert simulate_trade(hi, lo, cl, 0, 1, 1.0, 2.0, 10) == pytest.approx(-1.0)  # bar 1 low 98.9 touches the stop at 99
    hi, lo, cl = _path([100, 101.2, 102.5], [100, 100.2, 101], [100, 101, 102])
    assert simulate_trade(hi, lo, cl, 0, 1, 1.0, 2.0, 10) == pytest.approx(2.0)  # target 102 reached
    both = _path([100, 103], [100, 98], [100, 100])
    assert simulate_trade(*both, 0, 1, 1.0, 2.0, 10) == pytest.approx(-1.0)  # same bar touches both: stop first


def test_simulate_trade_exit_rules():
    # +1R reached (101.1), then price falls back through the entry, then trades on.
    hi, lo, cl = _path([100, 101.1, 100.4, 99.5], [100, 100.3, 99.9, 98.5], [100, 101, 100.1, 99])
    assert simulate_trade(hi, lo, cl, 0, 1, 1.0, 3.0, 10, "base") == pytest.approx(-1.0)        # back down to the stop
    assert simulate_trade(hi, lo, cl, 0, 1, 1.0, 3.0, 10, "be@1R") == pytest.approx(0.0)        # stopped at breakeven
    assert simulate_trade(hi, lo, cl, 0, 1, 1.0, 3.0, 10, "partial@1R") == pytest.approx(0.5)   # +0.5R banked, rest at breakeven
    trail = simulate_trade(hi, lo, cl, 0, 1, 1.0, 3.0, 10, "trail@1R", trail_atr=0.5)
    assert trail == pytest.approx(0.6)  # trailed to 101.1 - 0.5 = 100.6, hit on bar 2 (low 99.9): +0.6R


def test_simulate_trade_sell_mirrors_and_marks_a_timeout_to_market():
    hi, lo, cl = _path([100, 100.2, 100.1], [100, 99.6, 99.7], [100, 99.8, 99.5])
    assert simulate_trade(hi, lo, cl, 0, -1, 1.0, 2.0, 2) == pytest.approx(0.5)  # unresolved: marked at 99.5


# ---- the three studies (structure + sanity on synthetic data) ---------------------------------------------

def test_htf_alignment_study_reports_groups_years_and_symbol_counts():
    frames = {f"S{i}": _frame(n=9000, seed=i, drift=0.002 * (i % 2), freq="1h") for i in range(3)}
    out = htf_alignment_study(frames, warmup=700, step=6)
    assert set(out["pooled"]) == {"aligned", "mixed", "counter"}
    assert all(out["pooled"][g]["n"] > 50 for g in ("aligned", "counter"))
    assert out["symbols_tested"] == 3 and 0 <= out["symbols_aligned_beats_counter"] <= 3
    assert out["by_year"]  # at least one year with enough of both groups


def test_htf_alignment_study_finds_the_effect_when_it_really_exists():
    """A persistent-trend synthetic market: entries WITH the closed higher-timeframe trend must beat those against it."""
    rng = np.random.default_rng(11)
    n = 12000
    regime = np.repeat(rng.choice([-1.0, 1.0], size=n // 400 + 1), 400)[:n]
    close = 100 + np.cumsum(regime * 0.03 + rng.normal(0, 0.1, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    frame = pd.DataFrame({"High": close + 0.1, "Low": close - 0.1, "Close": close}, index=idx)
    out = htf_alignment_study({"TRENDY": frame}, warmup=700, step=3)
    assert out["pooled"]["aligned"]["mean_r"] > out["pooled"]["counter"]["mean_r"]


def test_exit_rule_study_is_paired_and_charges_cost():
    frames = {"A": _frame(n=4000, seed=1), "B": _frame(n=4000, seed=2)}
    free = exit_rule_study(frames, hold=48, step=12)
    costly = exit_rule_study(frames, hold=48, step=12, cost_by_symbol={"A": 0.05, "B": 0.05})
    assert set(free["rules"]) == {"base", "be@1R", "partial@1R", "trail@1R", "partial+trail@1R"}
    assert free["rules"]["base"]["n"] == free["rules"]["trail@1R"]["n"] > 100  # the SAME entries for every rule
    assert costly["rules"]["base"]["mean_net_r"] < free["rules"]["base"]["mean_net_r"]
    assert set(free["beats_base_on_symbols"]) == {"be@1R", "partial@1R", "trail@1R", "partial+trail@1R"} and free["symbols"] == 2


def test_cost_drag_falls_as_the_stop_widens_and_scales_with_the_cost():
    frames = {"A": _frame(n=1500)}
    out = cost_drag_by_stop_study(frames, {"A": 0.02})
    d = out["per_symbol"]["A"]
    assert d[2.0] > d[4.0] > d[6.0] > 0
    assert d[2.0] == pytest.approx(d[4.0] * 2, rel=1e-6) and out["pooled"][2.0] == pytest.approx(d[2.0])
    assert cost_drag_by_stop_study(frames, {"A": 0.04})["per_symbol"]["A"][2.0] == pytest.approx(d[2.0] * 2)


def test_partial_plus_trail_banks_half_then_trails_the_rest_from_breakeven():
    hi, lo, cl = _path([100, 101.1, 102.4, 101.0], [100, 100.3, 101.2, 100.9], [100, 101, 102.2, 101.0])
    # +1R reached on bar 1 (best 101.1): bank 0.5, stop -> entry, trail 0.5 behind the best. Bar 2 makes 102.4 (trail 101.9);
    # bar 3's low 100.9 hits it: remaining half exits at 101.9 = +1.9R -> 0.5 + 0.5 * 1.9 = 1.45R
    assert simulate_trade(hi, lo, cl, 0, 1, 1.0, 3.0, 10, "partial+trail@1R", trail_atr=0.5) == pytest.approx(1.45)
    # a fall straight back through the entry after +1R keeps the banked half: +0.5R (the remainder stops at breakeven)
    hi2, lo2, cl2 = _path([100, 101.1, 100.4, 99.5], [100, 100.3, 99.9, 98.5], [100, 101, 100.1, 99])
    assert simulate_trade(hi2, lo2, cl2, 0, 1, 1.0, 3.0, 10, "partial+trail@1R", trail_atr=1.5) == pytest.approx(0.5)


def test_closed_htf_directions_use_only_closed_buckets():
    from analysis.htf_flags import closed_htf_directions

    index = pd.date_range("2025-01-01", periods=24 * 12 * 60, freq="5min")  # 60 days
    rising = pd.Series(np.linspace(100, 200, len(index)), index=index)
    assert closed_htf_directions(rising) == {"h4": "up", "d1": "up"}
    falling = pd.Series(np.linspace(200, 100, len(index)), index=index)
    assert closed_htf_directions(falling) == {"h4": "down", "d1": "down"}
    assert closed_htf_directions(rising.iloc[:50]) == {"h4": None, "d1": None}  # too short for an SMA of buckets
    assert closed_htf_directions(None) == {"h4": None, "d1": None}
    # a spike in the still-forming last day cannot change the closed-bucket read
    spiked = rising.copy()
    spiked.iloc[-3:] = 1.0
    assert closed_htf_directions(spiked)["d1"] == "up"
