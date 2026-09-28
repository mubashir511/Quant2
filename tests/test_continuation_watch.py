"""Tests for analysis/continuation_watch.py — Continuation Watch, phase 1 (the deterministic filter only).

`test_calibration_against_every_real_closed_trade_this_account_has_ever_had` is the actual acceptance test
the user asked for: run the filter over every real M5 bar sequence this account's own trade history produced
(tests/fixtures/continuation_watch_real_trades.json, captured live from MT5 — see that file's own generation
comment below) and confirm it fires on ONLY the two real trades that motivated this feature."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import config
from analysis.continuation_watch import evaluate_continuation

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "continuation_watch_real_trades.json"


def _atr(bars_before: list) -> float:
    """Wilder-style ATR(14) from OHLC rows [iso_time, open, high, low, close], oldest first — same math
    analysis.continuation_watch's own callers will use in the live pipeline."""
    df = pd.DataFrame(bars_before, columns=["time", "Open", "High", "Low", "Close"])
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-14:].mean())


def _bars_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["time", "Open", "High", "Low", "Close"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def test_calibration_against_every_real_closed_trade_this_account_has_ever_had():
    cases = json.loads(FIXTURE_PATH.read_text())
    expected_pass = {"SOLUSD_win1", "XAGUSD_win2"}
    expected_fail = {"WHEATc_tiny", "EURUSD_tiny"}
    assert set(cases) == expected_pass | expected_fail  # this IS the account's whole closed-trade history

    results = {}
    for name, case in cases.items():
        atr_at_close = _atr(case["before_bars"])
        exit_time = datetime.fromisoformat(case["exit_time_utc"])
        after = _bars_df(case["after_bars"])
        # "now" = well past the decision window, so a market that never produced enough bars in time is
        # correctly read as a closed-out no, exactly as the live job will see it hours/days later.
        now = exit_time + timedelta(hours=6)
        verdict = evaluate_continuation(case["side"], case["exit_price"], exit_time, atr_at_close, after, now)
        results[name] = verdict

    for name in expected_pass:
        assert results[name].passed, f"{name} should have passed: {results[name].reason}"
    for name in expected_fail:
        assert not results[name].passed, f"{name} should NOT have passed: {results[name].reason}"

    # The real numbers this filter was calibrated against — pinned so a future change to the thresholds
    # or the math is forced to look at this exact evidence again, not just at the pass/fail booleans.
    assert results["SOLUSD_win1"].move_atr == pytest.approx(4.51, abs=0.05)
    assert results["XAGUSD_win2"].move_atr == pytest.approx(20.98, abs=0.5)
    # WHEAT.c: the market didn't trade again for ~4.5h — the 15-minute window closes with zero real bars.
    assert results["WHEATc_tiny"].bars_seen == 0 and results["WHEATc_tiny"].move_atr is None
    # EURUSD: the weekend gap — without the timeliness gate this looks like an +11x ATR "continuation";
    # WITH it, it's rejected before the (stale, unrelated) price is ever even measured.
    assert results["EURUSD_tiny"].bars_seen == 0 and results["EURUSD_tiny"].move_atr is None


# --- synthetic edge cases (deliberately not sourced from real trades, to test the boundaries directly) ------

def _bars(start: datetime, rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = [start + timedelta(minutes=5 * i) for i in range(len(rows))]
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=pd.DatetimeIndex(idx))


def test_passes_on_a_clean_fast_continuation():
    close = datetime(2026, 1, 1, 12, 0)
    bars = _bars(close, [(100, 101, 100, 100.8), (100.8, 102, 100.7, 101.9), (101.9, 104, 101.8, 103.9)])
    v = evaluate_continuation("buy", 100.0, close, atr_at_close=1.0, bars_since_close=bars, now=close + timedelta(minutes=20))
    assert v.passed and v.move_atr == pytest.approx(3.9, abs=0.01) and v.bars_seen == 3


def test_fails_when_the_move_is_too_small():
    close = datetime(2026, 1, 1, 12, 0)
    bars = _bars(close, [(100, 100.3, 99.9, 100.1), (100.1, 100.4, 100.0, 100.2), (100.2, 100.5, 100.1, 100.3)])
    v = evaluate_continuation("buy", 100.0, close, atr_at_close=1.0, bars_since_close=bars, now=close + timedelta(minutes=20))
    assert not v.passed and "only" in v.reason and v.move_atr == pytest.approx(0.3, abs=0.01)


def test_fails_when_price_reversed_instead_of_continuing():
    close = datetime(2026, 1, 1, 12, 0)
    bars = _bars(close, [(100, 100.1, 99.0, 99.2), (99.2, 99.3, 98.0, 98.1), (98.1, 98.2, 97.0, 97.1)])
    v = evaluate_continuation("buy", 100.0, close, atr_at_close=1.0, bars_since_close=bars, now=close + timedelta(minutes=20))
    assert not v.passed and v.move_atr < 0


def test_a_sell_measures_the_move_in_the_right_direction():
    close = datetime(2026, 1, 1, 12, 0)
    bars = _bars(close, [(100, 100.1, 97, 97.5), (97.5, 97.6, 95, 95.2), (95.2, 95.3, 93, 93.1)])
    v = evaluate_continuation("sell", 100.0, close, atr_at_close=1.0, bars_since_close=bars, now=close + timedelta(minutes=20))
    assert v.passed and v.move_atr == pytest.approx(6.9, abs=0.01)


def test_still_waiting_is_not_the_same_as_rejected():
    # Only 1 of the required 3 bars has arrived, and the 15-minute deadline hasn't passed yet -> the caller
    # must be able to tell "check again shortly" apart from "this trade is done, stop watching it".
    close = datetime(2026, 1, 1, 12, 0)
    bars = _bars(close, [(100, 105, 100, 104.9)])
    v = evaluate_continuation("buy", 100.0, close, atr_at_close=1.0, bars_since_close=bars, now=close + timedelta(minutes=6))
    assert not v.passed and v.bars_seen == 1 and v.deadline_passed is False and "still waiting" in v.reason


def test_the_deadline_expiring_with_too_few_bars_is_a_permanent_no():
    close = datetime(2026, 1, 1, 12, 0)
    bars = _bars(close, [(100, 105, 100, 104.9)])
    v = evaluate_continuation("buy", 100.0, close, atr_at_close=1.0, bars_since_close=bars, now=close + timedelta(minutes=16))
    assert not v.passed and v.deadline_passed is True and "window has closed" in v.reason


def test_no_atr_never_crashes_and_is_a_clean_reject():
    close = datetime(2026, 1, 1, 12, 0)
    bars = _bars(close, [(100, 105, 100, 104.9)] * 3)
    v = evaluate_continuation("buy", 100.0, close, atr_at_close=None, bars_since_close=bars, now=close + timedelta(minutes=20))
    assert not v.passed and v.move_atr is None and "ATR" in v.reason


def test_config_defaults_match_the_calibration():
    assert config.CONTINUATION_MIN_BARS == 3
    assert config.CONTINUATION_MAX_WAIT_MINUTES == 15
    assert config.CONTINUATION_MIN_ATR_MOVE == 2.0
