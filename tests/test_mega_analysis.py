from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import config
from ai.claude_cli import CLI_MISSING_MESSAGE
from ai.mega_analysis import (
    _write_current_activity,
    _write_state,
    _write_step,
    is_due,
    mega_session_is_live,
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


# --- read_progress / _write_step / _write_current_activity (live status while a run is happening) ---


def test_read_progress_empty_dict_when_file_missing(_fixed_schedule):
    assert read_progress() == {}


def test_write_step_then_read_progress_round_trips(_fixed_schedule):
    _write_step("Analyzing 5/18 — USDCAD")
    progress = read_progress()
    assert progress["steps"] == ["Analyzing 5/18 — USDCAD"]
    assert progress["current_activity"] is None
    assert progress["updated_utc"]  # a real timestamp was recorded


def test_write_step_appends_rather_than_overwrites(_fixed_schedule):
    # Direct user request 2026-08-25: the live display should show every
    # discrete step so far, not just the latest one replacing the last —
    # this is the behavior that makes that possible.
    _write_step("first message")
    _write_step("second message")
    assert read_progress()["steps"] == ["first message", "second message"]


def test_write_current_activity_overwrites_rather_than_appends(_fixed_schedule):
    # Real bug found live 2026-08-27: the audit-model pool's own
    # per-second retry/countdown status was being appended as a brand
    # new entry every tick, ballooning one real run's progress log to
    # 246+ near-duplicate lines within minutes. current_activity must
    # replace, not accumulate, no matter how many times it's called.
    _write_current_activity("0/10 models completed, 10 retrying, 0 gave up")
    _write_current_activity("0/10 models completed, 1 in progress, 9 retrying, 0 gave up")
    progress = read_progress()
    assert progress["current_activity"] == "0/10 models completed, 1 in progress, 9 retrying, 0 gave up"


def test_write_current_activity_does_not_touch_steps(_fixed_schedule):
    _write_step("Running Claude sonnet + the full audit-model pool...")
    _write_current_activity("3/10 models completed, 2 in progress, 5 gave up")
    progress = read_progress()
    assert progress["steps"] == ["Running Claude sonnet + the full audit-model pool..."]
    assert progress["current_activity"] == "3/10 models completed, 2 in progress, 5 gave up"


def test_write_step_clears_stale_current_activity(_fixed_schedule):
    # Real bug caught before shipping (2026-08-27): once a noisy phase
    # (the instrument scan, or the audit pool) hands off to a new
    # discrete milestone, its last current_activity value must not
    # linger and keep rendering as "live" — the new step is what's live
    # now.
    _write_current_activity("Analyzing 18/18 — ETHUSD (D1/H4/H1/monthly + chart structure + trading cost)")
    _write_step("Analyzed all 18 instruments.")
    progress = read_progress()
    assert progress["current_activity"] is None
    assert progress["steps"] == ["Analyzed all 18 instruments."]


def test_reset_progress_clears_a_prior_runs_state(_fixed_schedule):
    from ai.mega_analysis import _reset_progress

    _write_step("leftover from a previous run")
    _write_current_activity("leftover activity")
    _reset_progress()
    progress = read_progress()
    assert progress["steps"] == []
    assert progress["current_activity"] is None


def test_read_progress_empty_dict_on_corrupt_file(_fixed_schedule, tmp_path):
    corrupt_path = tmp_path / "corrupt_progress.json"
    corrupt_path.write_text("{not valid json")
    with patch.object(config, "MEGA_ANALYSIS_PROGRESS_FILE", str(corrupt_path)):
        assert read_progress() == {}


# --- mega_session_is_live ---


def test_mega_session_is_live_true_when_progress_is_fresh_and_newer_than_last_attempt():
    now = datetime.now(timezone.utc).isoformat()
    progress = {"updated_utc": now}
    state = {"last_attempt_utc": "2020-01-01T00:00:00+00:00"}
    assert mega_session_is_live(progress, state) is True


def test_mega_session_is_live_false_when_no_progress():
    assert mega_session_is_live({}, {}) is False


def test_mega_session_is_live_false_when_progress_is_stale():
    old = "2020-01-01T00:00:00+00:00"
    progress = {"updated_utc": old}
    assert mega_session_is_live(progress, {}) is False


def test_mega_session_is_live_false_when_progress_is_older_than_last_completed_attempt():
    now = datetime.now(timezone.utc).isoformat()
    progress = {"updated_utc": now}
    state = {"last_attempt_utc": now}  # a completed attempt at least as new as the progress marker
    assert mega_session_is_live(progress, state) is False


def test_mega_session_is_live_true_after_a_long_silent_gap_under_the_real_run_ceiling():
    # Real bug found live 2026-08-27: a flat 5-minute staleness cutoff
    # wrongly marked a genuinely still-running session as dead, because
    # a single step (e.g. Claude's own multi-minute web-research call)
    # can go quiet for longer than that with nothing new to report. 20
    # minutes of silence must still read as live — well past the old
    # 300-second cutoff, comfortably under the real _RUN_TIMEOUT_SECONDS
    # ceiling.
    old_but_within_run_budget = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    progress = {"updated_utc": old_but_within_run_budget}
    state = {"last_attempt_utc": "2020-01-01T00:00:00+00:00"}
    assert mega_session_is_live(progress, state) is True


@patch("ai.mega_analysis.suggest_ftmo_portfolio", return_value="a real suggestion")
@patch("ai.mega_analysis.build_ftmo_summary", return_value="summary text")
@patch("ai.mega_analysis.analyze_ftmo_assets", return_value=[])
@patch("ai.mega_analysis.fetch_ftmo_status")
@patch("ai.mega_analysis.get_pending_orders", return_value=[])
@patch("ai.mega_analysis.get_open_positions", return_value=[])
@patch("ai.mega_analysis.get_market_watch", return_value=["EURUSD"])
@patch("ai.mega_analysis.get_account_summary")
@patch("ai.mega_analysis.connect")
def test_run_mega_analysis_broadcasts_live_progress_without_a_caller_supplied_callback(
    mock_connect, mock_account, mock_watch, mock_positions, mock_pending_orders, mock_status,
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
    assert "Running Claude sonnet + the full audit-model pool..." in progress["steps"]


@patch("ai.mega_analysis.suggest_ftmo_portfolio", return_value="a real suggestion")
@patch("ai.mega_analysis.build_ftmo_summary", return_value="summary text")
@patch("ai.mega_analysis.analyze_ftmo_assets")
@patch("ai.mega_analysis.fetch_ftmo_status")
@patch("ai.mega_analysis.get_pending_orders", return_value=[])
@patch("ai.mega_analysis.get_open_positions", return_value=[])
@patch("ai.mega_analysis.get_market_watch", return_value=["EURUSD", "GBPUSD"])
@patch("ai.mega_analysis.get_account_summary")
@patch("ai.mega_analysis.connect")
def test_run_mega_analysis_reports_per_instrument_progress_as_current_activity_not_steps(
    mock_connect, mock_account, mock_watch, mock_positions, mock_pending_orders, mock_status,
    mock_analyze, mock_summary, mock_suggest, _fixed_schedule,
):
    # Direct user request 2026-08-27, after seeing a mock-up render each
    # instrument as its own permanent checkmarked line: "just use one
    # message only dynamically update it 18 times... and also update
    # the symbol names as well simultaneously" — per-instrument progress
    # must overwrite current_activity, the same self-superseding-status
    # category as the audit pool, not grow the permanent steps list by
    # one entry per instrument.
    from ai.mega_analysis import run_mega_analysis

    def _fake_analyze(assets, on_progress=None):
        on_progress("Analyzing 1/2 — EURUSD (D1/H4/H1/monthly + chart structure + trading cost)")
        on_progress("Analyzing 2/2 — GBPUSD (D1/H4/H1/monthly + chart structure + trading cost)")
        return []

    mock_analyze.side_effect = _fake_analyze

    run_mega_analysis()

    progress = read_progress()
    assert not any("EURUSD" in step or "GBPUSD" in step for step in progress["steps"])
    assert "Analyzed all 2 instruments." in progress["steps"]
    # The last per-instrument message must not linger as a stale
    # current_activity once the scan hands off to the next milestone.
    assert progress["current_activity"] is None


@patch("ai.mega_analysis.suggest_ftmo_portfolio", return_value="a real suggestion")
@patch("ai.mega_analysis.build_ftmo_summary", return_value="summary text")
@patch("ai.mega_analysis.analyze_ftmo_assets", return_value=[])
@patch("ai.mega_analysis.fetch_ftmo_status")
@patch("ai.mega_analysis.get_pending_orders")
@patch("ai.mega_analysis.get_open_positions", return_value=[])
@patch("ai.mega_analysis.get_market_watch", return_value=["EURUSD"])
@patch("ai.mega_analysis.get_account_summary")
@patch("ai.mega_analysis.connect")
def test_run_mega_analysis_fetches_and_passes_pending_orders_into_build_ftmo_summary(
    mock_connect, mock_account, mock_watch, mock_positions, mock_pending_orders, mock_status,
    mock_analyze, mock_summary, mock_suggest, _fixed_schedule,
):
    # Real gap this proves closed: build_ftmo_summary previously had zero
    # visibility into outstanding pending limit orders at all.
    from ai.mega_analysis import run_mega_analysis
    from data.mt5_source import PendingOrder

    sentinel_orders = [
        PendingOrder(symbol="EURUSD", volume=1.0, order_type="buy limit", price_open=1.09, sl=1.08, tp=None, ticket=777)
    ]
    mock_pending_orders.return_value = sentinel_orders

    run_mega_analysis()

    mock_summary.assert_called_once()
    assert mock_summary.call_args.kwargs["pending_orders"] == sentinel_orders


@patch("ai.mega_analysis.suggest_ftmo_portfolio", return_value="a real suggestion")
@patch("ai.mega_analysis.build_ftmo_summary", return_value="summary text")
@patch("ai.mega_analysis.analyze_ftmo_assets", return_value=[])
@patch("ai.mega_analysis.fetch_ftmo_status")
@patch("ai.mega_analysis.get_pending_orders", return_value=[])
@patch("ai.mega_analysis.get_open_positions", return_value=[])
@patch("ai.mega_analysis.get_market_watch", return_value=["EURUSD"])
@patch("ai.mega_analysis.get_account_summary")
@patch("ai.mega_analysis.connect")
def test_run_mega_analysis_excludes_copilot_from_the_audit_pool(
    mock_connect, mock_account, mock_watch, mock_positions, mock_pending_orders, mock_status,
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
@patch("ai.mega_analysis.get_pending_orders", return_value=[])
@patch("ai.mega_analysis.get_open_positions", return_value=[])
@patch("ai.mega_analysis.get_market_watch", return_value=["EURUSD"])
@patch("ai.mega_analysis.get_account_summary")
@patch("ai.mega_analysis.connect")
def test_run_mega_analysis_broadcasts_audit_progress_too(
    mock_connect, mock_account, mock_watch, mock_positions, mock_pending_orders, mock_status,
    mock_analyze, mock_summary, mock_suggest, _fixed_schedule,
):
    from ai.mega_analysis import run_mega_analysis

    def _fake_suggest(summary, **kwargs):
        kwargs["on_audit_progress"]("gpt-4: analyzing...")
        return "a real suggestion"

    mock_suggest.side_effect = _fake_suggest

    run_mega_analysis()

    assert read_progress()["current_activity"] == "gpt-4: analyzing..."
