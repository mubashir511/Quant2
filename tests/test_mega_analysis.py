from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import config
from ai.claude_cli import CLI_MISSING_MESSAGE
from ai.mega_analysis import (
    _write_progress,
    _write_state,
    is_due,
    next_run_utc,
    read_mega_analysis_enabled,
    read_mega_analysis_trigger,
    read_progress,
    read_state,
    run_scheduled_mega_analysis,
    set_mega_analysis_enabled,
    set_mega_analysis_trigger,
)


@pytest.fixture(autouse=True)
def _fixed_schedule(tmp_path):
    # Fixed, test-local trigger/grace/state-file values — deliberately
    # different from production defaults so these tests stay isolated
    # from config.py ever changing the real schedule.
    state_path = tmp_path / "mega_analysis_state.json"
    progress_path = tmp_path / "mega_analysis_progress.json"
    suggestion_path = tmp_path / "mega_analysis_latest_suggestion.json"
    enabled_path = tmp_path / "mega_analysis_enabled.json"
    trigger_path = tmp_path / "mega_analysis_trigger.json"
    with (
        patch.object(config, "MEGA_ANALYSIS_TRIGGER_HOUR_UTC", 10),
        patch.object(config, "MEGA_ANALYSIS_TRIGGER_MINUTE_UTC", 0),
        patch.object(config, "MEGA_ANALYSIS_GRACE_MINUTES", 20),
        patch.object(config, "MEGA_ANALYSIS_STATE_FILE", str(state_path)),
        patch.object(config, "MEGA_ANALYSIS_PROGRESS_FILE", str(progress_path)),
        patch.object(config, "MEGA_ANALYSIS_LATEST_SUGGESTION_FILE", str(suggestion_path)),
        patch.object(config, "MEGA_ANALYSIS_ENABLED_FILE", str(enabled_path)),
        patch.object(config, "MEGA_ANALYSIS_TRIGGER_FILE", str(trigger_path)),
        # Fixed regardless of any real MEGA_ANALYSIS_MODEL env override
        # (e.g. a temporary manual-test "haiku" override in .env) — tests
        # must stay deterministic and independent of that.
        patch.object(config, "MEGA_ANALYSIS_MODEL", "sonnet"),
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


def test_read_mega_analysis_enabled_true_when_file_missing(_fixed_schedule):
    # Opt-out, not opt-in — direct user request 2026-08-23: a missing/
    # fresh-install file must never silently disable the daily run.
    assert read_mega_analysis_enabled() is True


def test_read_mega_analysis_enabled_true_on_corrupt_file(_fixed_schedule, tmp_path):
    corrupt_path = tmp_path / "corrupt_enabled.json"
    corrupt_path.write_text("{not valid json")
    with patch.object(config, "MEGA_ANALYSIS_ENABLED_FILE", str(corrupt_path)):
        assert read_mega_analysis_enabled() is True


def test_set_mega_analysis_enabled_false_then_read_round_trips(_fixed_schedule):
    set_mega_analysis_enabled(False)
    assert read_mega_analysis_enabled() is False


def test_set_mega_analysis_enabled_true_then_read_round_trips(_fixed_schedule):
    set_mega_analysis_enabled(False)
    set_mega_analysis_enabled(True)
    assert read_mega_analysis_enabled() is True


def test_read_mega_analysis_trigger_falls_back_to_config_when_file_missing(_fixed_schedule):
    # _fixed_schedule patches MEGA_ANALYSIS_TRIGGER_HOUR_UTC/MINUTE_UTC to 10:00.
    assert read_mega_analysis_trigger() == (10, 0)


def test_set_mega_analysis_trigger_then_read_round_trips(_fixed_schedule):
    set_mega_analysis_trigger(16, 45)
    assert read_mega_analysis_trigger() == (16, 45)


def test_read_mega_analysis_trigger_falls_back_on_corrupt_file(_fixed_schedule, tmp_path):
    corrupt_path = tmp_path / "corrupt_trigger.json"
    corrupt_path.write_text("{not valid json")
    with patch.object(config, "MEGA_ANALYSIS_TRIGGER_FILE", str(corrupt_path)):
        assert read_mega_analysis_trigger() == (10, 0)


def test_read_mega_analysis_trigger_falls_back_on_out_of_range_value(_fixed_schedule, tmp_path):
    import json

    bad_path = tmp_path / "bad_trigger.json"
    bad_path.write_text(json.dumps({"hour_utc": 25, "minute_utc": 0}))
    with patch.object(config, "MEGA_ANALYSIS_TRIGGER_FILE", str(bad_path)):
        assert read_mega_analysis_trigger() == (10, 0)


def test_next_run_utc_uses_the_overridden_trigger_time(_fixed_schedule):
    set_mega_analysis_trigger(16, 30)
    now = _utc(2026, 8, 21, 8, 0)
    assert next_run_utc(now, state={}) == _utc(2026, 8, 21, 16, 30)


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


# --- read_progress / _write_progress (live status while a run is happening) ---


def test_read_progress_empty_dict_when_file_missing(_fixed_schedule):
    assert read_progress() == {}


def test_write_progress_then_read_progress_round_trips(_fixed_schedule):
    _write_progress("Analyzing 5/18 — USDCAD")
    progress = read_progress()
    assert progress["message"] == "Analyzing 5/18 — USDCAD"
    assert progress["updated_utc"]  # a real timestamp was recorded


def test_write_progress_overwrites_the_previous_message(_fixed_schedule):
    _write_progress("first message")
    _write_progress("second message")
    assert read_progress()["message"] == "second message"


def test_read_progress_empty_dict_on_corrupt_file(_fixed_schedule, tmp_path):
    corrupt_path = tmp_path / "corrupt_progress.json"
    corrupt_path.write_text("{not valid json")
    with patch.object(config, "MEGA_ANALYSIS_PROGRESS_FILE", str(corrupt_path)):
        assert read_progress() == {}


@patch("ai.mega_analysis.suggest_ftmo_portfolio", return_value="a real suggestion")
@patch("ai.mega_analysis.build_ftmo_summary", return_value="summary text")
@patch("ai.mega_analysis.analyze_ftmo_assets", return_value=[])
@patch("ai.mega_analysis.fetch_ftmo_status")
@patch("ai.mega_analysis.get_open_positions", return_value=[])
@patch("ai.mega_analysis.get_market_watch", return_value=["EURUSD"])
@patch("ai.mega_analysis.get_account_summary")
@patch("ai.mega_analysis.connect")
def test_run_mega_analysis_broadcasts_live_progress_without_a_caller_supplied_callback(
    mock_connect, mock_account, mock_watch, mock_positions, mock_status,
    mock_analyze, mock_summary, mock_suggest, _fixed_schedule,
):
    # Direct user request 2026-08-22: the unattended run (which never
    # passes on_stage/on_audit_progress — see run_scheduled_mega_analysis)
    # must still show live status the same way the manual button does,
    # not just log silently. This is the real gap: before this fix,
    # _notify only reached logger.info + an optional caller callback,
    # which the scheduled path never supplies.
    from ai.mega_analysis import run_mega_analysis

    run_mega_analysis()

    progress = read_progress()
    assert progress["message"] == "Running Claude sonnet + the full audit-model pool..."


@patch("ai.mega_analysis.suggest_ftmo_portfolio", return_value="a real suggestion")
@patch("ai.mega_analysis.build_ftmo_summary", return_value="summary text")
@patch("ai.mega_analysis.analyze_ftmo_assets", return_value=[])
@patch("ai.mega_analysis.fetch_ftmo_status")
@patch("ai.mega_analysis.get_open_positions", return_value=[])
@patch("ai.mega_analysis.get_market_watch", return_value=["EURUSD"])
@patch("ai.mega_analysis.get_account_summary")
@patch("ai.mega_analysis.connect")
def test_run_mega_analysis_excludes_copilot_from_the_audit_pool(
    mock_connect, mock_account, mock_watch, mock_positions, mock_status,
    mock_analyze, mock_summary, mock_suggest, _fixed_schedule,
):
    # Direct user request 2026-08-22: Copilot has a separate, dedicated
    # role elsewhere and shouldn't also spend its own request budget on
    # this unattended run's audit pool. The manual "Suggest Portfolio
    # Mix" button (app.py) keeps calling suggest_ftmo_portfolio without
    # this override, which defaults to include_copilot=True.
    from ai.mega_analysis import run_mega_analysis

    run_mega_analysis()

    assert mock_suggest.call_args.kwargs["include_copilot"] is False


@patch("ai.mega_analysis.suggest_ftmo_portfolio")
@patch("ai.mega_analysis.build_ftmo_summary", return_value="summary text")
@patch("ai.mega_analysis.analyze_ftmo_assets", return_value=[])
@patch("ai.mega_analysis.fetch_ftmo_status")
@patch("ai.mega_analysis.get_open_positions", return_value=[])
@patch("ai.mega_analysis.get_market_watch", return_value=["EURUSD"])
@patch("ai.mega_analysis.get_account_summary")
@patch("ai.mega_analysis.connect")
def test_run_mega_analysis_broadcasts_audit_progress_too(
    mock_connect, mock_account, mock_watch, mock_positions, mock_status,
    mock_analyze, mock_summary, mock_suggest, _fixed_schedule,
):
    from ai.mega_analysis import run_mega_analysis

    def _fake_suggest(summary, **kwargs):
        kwargs["on_audit_progress"]("gpt-4: analyzing...")
        return "a real suggestion"

    mock_suggest.side_effect = _fake_suggest

    run_mega_analysis()

    assert read_progress()["message"] == "gpt-4: analyzing..."
