import numpy as np
import pandas as pd
import pytest

import config
from analysis.backtest import classify_backtest_favorability
from analysis.edge_stats import (
    CONTRADICTED,
    NO_INFORMATION,
    SUPPORTED,
    EdgeBaseline,
    classify_edge,
    compute_null_baseline,
)


def _random_walk(n=3000, seed=1, drift=0.0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(drift, 0.2, n))
    high = close + np.abs(rng.normal(0.1, 0.05, n))
    low = close - np.abs(rng.normal(0.1, 0.05, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close}, index=idx)


_KW = dict(stop_atr_multiple=2.0, target_atr_multiple=4.0, max_holding_bars=48)


def test_null_baseline_on_a_random_walk_is_near_zero_gross_and_has_a_real_sample():
    baseline = compute_null_baseline(_random_walk(), "buy", **_KW)
    assert baseline is not None
    assert baseline.trades >= config.EDGE_BASELINE_MIN_TRADES
    assert abs(baseline.mean_r) < 0.25  # no drift, no cost: nothing to earn
    assert baseline.std_r > 0.3
    assert 0 < baseline.win_rate_pct < 100


def test_null_baseline_charges_the_real_cost():
    free = compute_null_baseline(_random_walk(), "buy", **_KW)
    costly = compute_null_baseline(_random_walk(), "buy", round_trip_cost_pct=0.05, **_KW)
    assert costly.mean_r < free.mean_r - 0.05


def test_null_baseline_is_none_on_thin_history_or_missing_high_low():
    assert compute_null_baseline(_random_walk(n=120), "buy", **_KW) is None
    assert compute_null_baseline(_random_walk().drop(columns=["High", "Low"]), "buy", **_KW) is None
    assert compute_null_baseline(_random_walk(), "flat", **_KW) is None


def test_null_baseline_thins_evenly_when_over_the_entry_cap(monkeypatch):
    monkeypatch.setattr(config, "EDGE_BASELINE_MAX_ENTRIES", 150)
    monkeypatch.setattr(config, "EDGE_BASELINE_MIN_TRADES", 50)
    baseline = compute_null_baseline(_random_walk(n=6000), "sell", **_KW)
    assert baseline is not None and baseline.trades <= 150


_BASE = EdgeBaseline("buy", -0.05, 1.2, 800, 27.0)


def test_supported_needs_a_positive_net_result_a_real_z_and_enough_trades():
    verdict = classify_edge(0.30, 100, _BASE)  # z = 0.35 / (1.2 / 10) = +2.9
    assert verdict.verdict == SUPPORTED and verdict.z == pytest.approx(2.9167, abs=1e-3)
    assert classify_edge(0.30, 10, _BASE).verdict == NO_INFORMATION  # below the sample floor
    # Clears the baseline but is still net-negative: not "supported".
    assert classify_edge(-0.01, 400, EdgeBaseline("buy", -0.30, 1.2, 800, 20.0)).verdict == NO_INFORMATION


def test_a_null_result_is_no_information_not_contradicted():
    """The whole point: 'a bit below zero' is what random entries score."""
    verdict = classify_edge(-0.06, 200, _BASE)
    assert verdict.verdict == NO_INFORMATION and abs(verdict.z) < 1


def test_contradicted_needs_a_clearly_worse_result_on_a_real_sample():
    assert classify_edge(-0.60, 100, _BASE).verdict == CONTRADICTED  # z = -0.55 / 0.12 = -4.6
    assert classify_edge(-0.60, 25, _BASE).verdict == NO_INFORMATION  # too few trades to condemn


def test_missing_or_degenerate_baseline_never_promotes_to_a_claim():
    for baseline in (None, EdgeBaseline("buy", 0.0, 0.0, 800, 30.0)):
        verdict = classify_edge(0.5, 500, baseline)
        assert verdict.verdict == NO_INFORMATION and verdict.z is None
    assert classify_edge(None, 100, _BASE).verdict == NO_INFORMATION


# ---- classify_backtest_favorability in verdict mode -------------------------------------------------

def _rsi(avg_r):
    from analysis.backtest import RSIReactionBacktest

    return RSIReactionBacktest(
        condition="oversold", threshold=30.0, trades=100, wins=0, losses=0, timeouts=0, win_rate_pct=30.0,
        avg_r_multiple=avg_r, stop_atr_multiple=2.0, target_atr_multiple=4.0, max_holding_bars=48,
    )


def test_favorability_verdict_mode_ignores_the_sign_of_a_null_result():
    # Legacy sign logic calls -0.05R "contradicted"; verdict mode says NO-INFORMATION -> unknown.
    assert classify_backtest_favorability("buy", None, _rsi(-0.05), None, None, None)[0] == "contradicted"
    assert classify_backtest_favorability(
        "buy", None, _rsi(-0.05), None, None, None, edge_verdicts={"rsi": NO_INFORMATION}
    ) == (None, None)


def test_favorability_verdict_mode_reports_supported_and_contradicted():
    assert classify_backtest_favorability(
        "buy", None, _rsi(0.4), None, None, None, edge_verdicts={"rsi": SUPPORTED}
    )[0] == "supported"
    assert classify_backtest_favorability(
        "buy", None, _rsi(-0.6), None, None, None, edge_verdicts={"rsi": CONTRADICTED}
    )[0] == "contradicted"


def test_a_verdict_for_the_wrong_side_backtest_is_not_used():
    # A sell reads the overbought backtest (None here) - the oversold one must not leak in.
    assert classify_backtest_favorability(
        "sell", None, _rsi(0.4), None, None, None, edge_verdicts={"rsi": SUPPORTED}
    ) == (None, None)


# ---- ftmo_suggest wiring ----------------------------------------------------------------------------

def test_intraday_edge_verdicts_use_the_side_matching_baseline_and_backtests():
    from ai.ftmo_suggest import IntradayBacktests, intraday_edge_verdicts

    bt = IntradayBacktests(
        rsi_oversold_backtest=_rsi(0.3),
        null_baseline_buy=_BASE,
        null_baseline_sell=EdgeBaseline("sell", 5.0, 1.0, 800, 90.0),
    )
    assert intraday_edge_verdicts(bt, "buy")["rsi"].verdict == SUPPORTED
    assert "rsi" not in intraday_edge_verdicts(bt, "sell")  # sells read the overbought backtest (absent)
    assert intraday_edge_verdicts(IntradayBacktests(rsi_oversold_backtest=_rsi(0.3)), "buy")["rsi"].verdict == NO_INFORMATION


def test_formatted_evidence_prints_the_baseline_and_the_verdict():
    from types import SimpleNamespace

    from ai.ftmo_suggest import IntradayBacktests, _format_intraday_backtests

    bt = IntradayBacktests(rsi_oversold_backtest=_rsi(0.3), null_baseline_buy=_BASE)
    text = _format_intraday_backtests(SimpleNamespace(intraday_backtests=bt))
    assert "random-entry baseline (long" in text
    assert "SUPPORTED (z +2.9 vs random -0.05R, n=100)" in text
    assert "NO-INFORMATION" in text  # explained in the header
    assert "NOT a reason to exclude" in text


def test_win_rate_floor_is_only_lowered_for_a_supported_setup():
    from ai.clerk_execution import _side_relevant_win_rate
    from ai.ftmo_suggest import IntradayBacktests

    supported = IntradayBacktests(rsi_oversold_backtest=_rsi(0.3), null_baseline_buy=_BASE)
    unproven = IntradayBacktests(rsi_oversold_backtest=_rsi(0.3))  # no baseline -> no lowering
    null_result = IntradayBacktests(rsi_oversold_backtest=_rsi(-0.06), null_baseline_buy=_BASE)
    assert _side_relevant_win_rate("buy", supported) == 30.0
    assert _side_relevant_win_rate("buy", unproven) is None
    assert _side_relevant_win_rate("buy", null_result) is None
