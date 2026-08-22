from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import config
from ai.mega_analysis import (
    _write_state,
    is_due,
    next_run_utc,
    read_state,
    run_scheduled_mega_analysis,
)


@pytest.fixture(autouse=True)
def _fixed_schedule(tmp_path):
    # Fixed, test-local trigger/grace/state-file values — deliberately
    # different from production defaults so these tests stay isolated
    # from config.py ever changing the real schedule.
    state_path = tmp_path / "mega_analysis_state.json"
    with (
        patch.object(config, "MEGA_ANALYSIS_TRIGGER_HOUR_UTC", 10),
        patch.object(config, "MEGA_ANALYSIS_TRIGGER_MINUTE_UTC", 0),
        patch.object(config, "MEGA_ANALYSIS_GRACE_MINUTES", 20),
        patch.object(config, "MEGA_ANALYSIS_STATE_FILE", str(state_path)),
    ):
        yield state_path


def _utc(y, m, d, h, mi):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def test_is_due_false_before_trigger_time():
    now = _utc(2026, 8, 21, 9, 59)
    assert is_due(now, state={}) is False


def test_is_due_true_at_trigger_instant():
    now = _utc(2026, 8, 21, 10, 0)
    assert is_due(now, state={}) is True


def test_is_due_true_within_grace_window():
    now = _utc(2026, 8, 21, 10, 15)
    assert is_due(now, state={}) is True


def test_is_due_false_after_grace_window_pc_was_off():
    # PC turns on well after the window (e.g. it was off/asleep through
    # 10:00-10:20) — today is treated as missed, not run late.
    now = _utc(2026, 8, 21, 14, 0)
    assert is_due(now, state={}) is False


def test_is_due_false_when_already_run_today():
    now = _utc(2026, 8, 21, 10, 5)
    state = {"last_run_date_utc": "2026-08-21"}
    assert is_due(now, state=state) is False


def test_is_due_true_again_next_day_after_yesterdays_run():
    now = _utc(2026, 8, 22, 10, 5)
    state = {"last_run_date_utc": "2026-08-21"}
    assert is_due(now, state=state) is True


def test_next_run_utc_is_today_trigger_when_not_yet_due():
    now = _utc(2026, 8, 21, 6, 0)
    assert next_run_utc(now, state={}) == _utc(2026, 8, 21, 10, 0)


def test_next_run_utc_is_tomorrow_when_already_ran_today():
    now = _utc(2026, 8, 21, 11, 0)
    state = {"last_run_date_utc": "2026-08-21"}
    assert next_run_utc(now, state=state) == _utc(2026, 8, 22, 10, 0)


def test_next_run_utc_is_tomorrow_when_todays_window_already_passed():
    # No run recorded, but we're well past today's trigger — the next
    # opportunity is tomorrow, not "run immediately since we're overdue".
    now = _utc(2026, 8, 21, 20, 0)
    assert next_run_utc(now, state={}) == _utc(2026, 8, 22, 10, 0)


def test_read_state_empty_dict_when_file_missing(_fixed_schedule):
    assert read_state() == {}


def test_write_state_then_read_state_round_trips(_fixed_schedule):
    _write_state("success")
    state = read_state()
    assert state["last_status"] == "success"
    assert state["last_run_date_utc"] == datetime.now(timezone.utc).date().isoformat()


def test_write_state_failure_does_not_clear_a_prior_success_date(_fixed_schedule):
    _write_state("success")
    prior_date = read_state()["last_run_date_utc"]
    _write_state("error", "boom")
    state = read_state()
    assert state["last_status"] == "error"
    # A later failed attempt (e.g. tomorrow's run crashing) must not erase
    # the fact that a real prior success happened, since last_run_date_utc
    # is the field is_due()/next_run_utc() actually key off of.
    assert state["last_run_date_utc"] == prior_date


def test_run_scheduled_mega_analysis_records_success(_fixed_schedule):
    with patch("ai.mega_analysis.run_mega_analysis", return_value="a real suggestion"):
        run_scheduled_mega_analysis()
    assert read_state()["last_status"] == "success"


def test_run_scheduled_mega_analysis_records_cli_failure_not_success(_fixed_schedule):
    from ai.claude_cli import CLI_MISSING_MESSAGE

    with patch("ai.mega_analysis.run_mega_analysis", return_value=CLI_MISSING_MESSAGE):
        run_scheduled_mega_analysis()
    state = read_state()
    assert state["last_status"] != "success"
    assert "last_run_date_utc" not in state or state["last_run_date_utc"] is None


def test_run_scheduled_mega_analysis_records_exception_not_success(_fixed_schedule):
    with patch("ai.mega_analysis.run_mega_analysis", side_effect=RuntimeError("MT5 unreachable")):
        run_scheduled_mega_analysis()
    state = read_state()
    assert state["last_status"] == "error"
    assert "MT5 unreachable" in state["last_detail"]


def test_run_scheduled_mega_analysis_records_timeout_not_success(_fixed_schedule):
    # The real incident this guards against: a hang deep inside a
    # dependency (yfinance's own cookie/crumb negotiation, confirmed
    # live) that never raises at all — it just blocks forever, which a
    # plain try/except cannot catch. Simulated here by making the
    # wrapped call outlast a deliberately tiny timeout.
    import time as time_module

    def _hangs():
        time_module.sleep(1)
        return "should never get here"

    with (
        patch("ai.mega_analysis.run_mega_analysis", side_effect=_hangs),
        patch("ai.mega_analysis._RUN_TIMEOUT_SECONDS", 0.05),
    ):
        run_scheduled_mega_analysis()
    state = read_state()
    assert state["last_status"] == "timeout"
    assert "last_run_date_utc" not in state or state["last_run_date_utc"] is None
