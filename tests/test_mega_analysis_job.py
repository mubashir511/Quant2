from unittest.mock import patch

import pytest

import config
import mega_analysis_job

# The PID-liveness-aware lock mechanism itself (pid_is_alive/acquire_lock/
# release_lock) was extracted into job_lock.py once a second unattended
# job (copilot_execution_job.py) needed the exact same logic — see
# tests/test_job_lock.py for its own dedicated coverage. This file keeps
# only what's specific to mega_analysis_job.py: main()'s lock lifecycle
# around a due run, and the inline post-success execution-check wiring.


@pytest.fixture(autouse=True)
def _fixed_toggle_files(tmp_path):
    # Real bug found while adding the enable/disable toggle (2026-08-23):
    # main() now checks read_mega_analysis_enabled() before is_due(), and
    # every OLDER test here left config.MEGA_ANALYSIS_ENABLED_FILE
    # pointed at its real project-root default — which, mid-session, the
    # user's own live app had already toggled to {"enabled": false} by
    # actually using the new feature, silently breaking every test that
    # expects a due run to actually fire. Isolated to a tmp path here so
    # these tests never depend on real, live application state again.
    # MEGA_ANALYSIS_TRIGGER_FILE (the later trigger-time picker) gets the
    # same treatment proactively, before it can bite the same way.
    with (
        patch.object(config, "MEGA_ANALYSIS_ENABLED_FILE", str(tmp_path / "mega_analysis_enabled.json")),
        patch.object(config, "MEGA_ANALYSIS_TRIGGER_FILE", str(tmp_path / "mega_analysis_trigger.json")),
    ):
        yield


# --- main(): the lock's actual lifecycle around a real due run ---


@patch("mega_analysis_job._run_inline_execution_check_if_successful")
@patch("mega_analysis_job.run_scheduled_mega_analysis")
@patch("mega_analysis_job.is_due", return_value=True)
def test_main_acquires_and_releases_lock_around_a_due_run(mock_is_due, mock_run, mock_inline, tmp_path, monkeypatch):
    lock = tmp_path / "mega_analysis.lock"
    monkeypatch.setattr(mega_analysis_job, "_LOCK_PATH", lock)
    lock_held_during_run = {}

    def _check_lock_held(*args, **kwargs):
        lock_held_during_run["exists"] = lock.exists()

    mock_run.side_effect = _check_lock_held

    mega_analysis_job.main()

    assert lock_held_during_run["exists"] is True  # held for the real duration of the run
    assert not lock.exists()  # released once the run (successfully) finishes
    mock_run.assert_called_once()


@patch("mega_analysis_job._run_inline_execution_check_if_successful")
@patch("mega_analysis_job.run_scheduled_mega_analysis")
@patch("mega_analysis_job.is_due", return_value=True)
def test_main_skips_the_run_when_the_lock_is_already_held(mock_is_due, mock_run, mock_inline, tmp_path, monkeypatch):
    import os

    lock = tmp_path / "mega_analysis.lock"
    lock.write_text(str(os.getpid()))  # a fresh, genuinely-alive lock from "another invocation"
    monkeypatch.setattr(mega_analysis_job, "_LOCK_PATH", lock)

    mega_analysis_job.main()

    mock_run.assert_not_called()
    assert lock.read_text() == str(os.getpid())  # the other invocation's own lock, left alone


@patch("mega_analysis_job._run_inline_execution_check_if_successful")
@patch("mega_analysis_job.run_scheduled_mega_analysis")
@patch("mega_analysis_job.is_due", return_value=True)
def test_main_releases_the_lock_even_if_the_run_raises(mock_is_due, mock_run, mock_inline, tmp_path, monkeypatch):
    # run_scheduled_mega_analysis is documented to never raise (it
    # catches everything internally) — this is defense-in-depth via the
    # `finally` block, not a scenario expected to happen in practice.
    lock = tmp_path / "mega_analysis.lock"
    monkeypatch.setattr(mega_analysis_job, "_LOCK_PATH", lock)
    mock_run.side_effect = RuntimeError("unexpected")

    try:
        mega_analysis_job.main()
    except RuntimeError:
        pass

    assert not lock.exists()


@patch("mega_analysis_job._run_inline_execution_check_if_successful")
@patch("mega_analysis_job.run_scheduled_mega_analysis")
@patch("mega_analysis_job.is_due", return_value=False)
def test_main_never_touches_the_lock_when_not_due(mock_is_due, mock_run, mock_inline, tmp_path, monkeypatch):
    lock = tmp_path / "mega_analysis.lock"
    monkeypatch.setattr(mega_analysis_job, "_LOCK_PATH", lock)

    mega_analysis_job.main()

    mock_run.assert_not_called()
    mock_inline.assert_not_called()
    assert not lock.exists()


# --- main(): the user-controlled enable/disable toggle (2026-08-23) ---


@patch("mega_analysis_job._run_inline_execution_check_if_successful")
@patch("mega_analysis_job.run_scheduled_mega_analysis")
@patch("mega_analysis_job.is_due", return_value=True)
@patch("mega_analysis_job.read_mega_analysis_enabled", return_value=False)
def test_main_skips_before_even_checking_is_due_when_disabled(
    mock_enabled, mock_is_due, mock_run, mock_inline, tmp_path, monkeypatch
):
    lock = tmp_path / "mega_analysis.lock"
    monkeypatch.setattr(mega_analysis_job, "_LOCK_PATH", lock)

    mega_analysis_job.main()

    mock_is_due.assert_not_called()  # the toggle is checked first, not just the lock
    mock_run.assert_not_called()
    mock_inline.assert_not_called()
    assert not lock.exists()


@patch("mega_analysis_job._run_inline_execution_check_if_successful")
@patch("mega_analysis_job.run_scheduled_mega_analysis")
@patch("mega_analysis_job.is_due", return_value=True)
@patch("mega_analysis_job.read_mega_analysis_enabled", return_value=True)
def test_main_still_runs_when_enabled_and_due(mock_enabled, mock_is_due, mock_run, mock_inline, tmp_path, monkeypatch):
    lock = tmp_path / "mega_analysis.lock"
    monkeypatch.setattr(mega_analysis_job, "_LOCK_PATH", lock)

    mega_analysis_job.main()

    mock_run.assert_called_once()


# --- inline "immediate execution" wiring after a successful run ---


@patch("mega_analysis_job.run_scheduled_mega_analysis")
@patch("mega_analysis_job.is_due", return_value=True)
def test_main_fires_inline_execution_check_after_a_successful_run(mock_is_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "mega_analysis.lock"
    monkeypatch.setattr(mega_analysis_job, "_LOCK_PATH", lock)
    # The inline path acquires its own, separate execution lock (shared
    # with copilot_execution_job.py's own standalone poll — see the
    # module docstring) — pointed at a tmp_path file so this test never
    # touches the real project-root lock file.
    monkeypatch.setattr(mega_analysis_job, "EXECUTION_LOCK_PATH", tmp_path / "copilot_execution.lock")

    state_before = {"last_attempt_utc": "2026-08-22T00:00:00+00:00"}
    state_after = {
        "last_attempt_utc": "2026-08-23T14:00:00+00:00",
        "last_status": "success",
        "last_run_date_utc": mega_analysis_job.datetime.now(mega_analysis_job.timezone.utc).date().isoformat(),
    }
    read_state_calls = {"count": 0}

    def _fake_read_state():
        read_state_calls["count"] += 1
        return state_before if read_state_calls["count"] == 1 else state_after

    with (
        patch("mega_analysis_job.read_state", side_effect=_fake_read_state),
        patch("mega_analysis_job.run_with_timeout") as mock_run_with_timeout,
    ):
        mega_analysis_job.main()

    mock_run_with_timeout.assert_called_once()
    args, kwargs = mock_run_with_timeout.call_args
    assert args[0] is mega_analysis_job.run_copilot_execution_check


@patch("mega_analysis_job.run_scheduled_mega_analysis")
@patch("mega_analysis_job.is_due", return_value=True)
def test_main_does_not_fire_inline_execution_check_after_a_cli_failure(mock_is_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "mega_analysis.lock"
    monkeypatch.setattr(mega_analysis_job, "_LOCK_PATH", lock)

    state_before = {"last_attempt_utc": "2026-08-22T00:00:00+00:00"}
    state_after = {
        "last_attempt_utc": "2026-08-23T14:00:00+00:00",
        "last_status": "cli_failed",
        "last_run_date_utc": None,
    }
    read_state_calls = {"count": 0}

    def _fake_read_state():
        read_state_calls["count"] += 1
        return state_before if read_state_calls["count"] == 1 else state_after

    with (
        patch("mega_analysis_job.read_state", side_effect=_fake_read_state),
        patch("mega_analysis_job.run_with_timeout") as mock_run_with_timeout,
    ):
        mega_analysis_job.main()

    mock_run_with_timeout.assert_not_called()


@patch("mega_analysis_job.run_scheduled_mega_analysis")
@patch("mega_analysis_job.is_due", return_value=True)
def test_main_does_not_fire_inline_execution_check_when_nothing_new_ran(mock_is_due, mock_run, tmp_path, monkeypatch):
    # A prior success from an EARLIER poll (same attempt timestamp before
    # and after) must not re-trigger the inline check — only a run that
    # genuinely just happened should.
    lock = tmp_path / "mega_analysis.lock"
    monkeypatch.setattr(mega_analysis_job, "_LOCK_PATH", lock)

    same_state = {
        "last_attempt_utc": "2026-08-23T14:00:00+00:00",
        "last_status": "success",
        "last_run_date_utc": mega_analysis_job.datetime.now(mega_analysis_job.timezone.utc).date().isoformat(),
    }

    with (
        patch("mega_analysis_job.read_state", return_value=same_state),
        patch("mega_analysis_job.run_with_timeout") as mock_run_with_timeout,
    ):
        mega_analysis_job.main()

    mock_run_with_timeout.assert_not_called()


@patch("mega_analysis_job.run_scheduled_mega_analysis")
@patch("mega_analysis_job.is_due", return_value=True)
def test_main_swallows_an_exception_from_the_inline_execution_check(mock_is_due, mock_run, tmp_path, monkeypatch):
    # A hang/exception in the bonus inline path must never break this
    # job's own lock release or propagate out of main().
    lock = tmp_path / "mega_analysis.lock"
    monkeypatch.setattr(mega_analysis_job, "_LOCK_PATH", lock)
    monkeypatch.setattr(mega_analysis_job, "EXECUTION_LOCK_PATH", tmp_path / "copilot_execution.lock")

    state_before = {"last_attempt_utc": "2026-08-22T00:00:00+00:00"}
    state_after = {
        "last_attempt_utc": "2026-08-23T14:00:00+00:00",
        "last_status": "success",
        "last_run_date_utc": mega_analysis_job.datetime.now(mega_analysis_job.timezone.utc).date().isoformat(),
    }
    read_state_calls = {"count": 0}

    def _fake_read_state():
        read_state_calls["count"] += 1
        return state_before if read_state_calls["count"] == 1 else state_after

    with (
        patch("mega_analysis_job.read_state", side_effect=_fake_read_state),
        patch("mega_analysis_job.run_with_timeout", side_effect=RuntimeError("boom")),
    ):
        mega_analysis_job.main()  # must not raise

    assert not lock.exists()  # still released cleanly
    assert not (tmp_path / "copilot_execution.lock").exists()  # execution lock also released


# --- real concurrency bug found on audit: two independent processes ---
# (this job's own inline pass, and copilot_execution_job.py's standalone
# hourly poll) must never run run_copilot_execution_check() at once


@patch("mega_analysis_job.run_scheduled_mega_analysis")
@patch("mega_analysis_job.is_due", return_value=True)
def test_inline_execution_check_skips_gracefully_when_lock_already_held(mock_is_due, mock_run, tmp_path, monkeypatch):
    import os

    lock = tmp_path / "mega_analysis.lock"
    monkeypatch.setattr(mega_analysis_job, "_LOCK_PATH", lock)
    execution_lock = tmp_path / "copilot_execution.lock"
    # Simulate the standalone hourly job (a genuinely different, alive
    # process) already holding the shared execution lock right now.
    execution_lock.write_text(str(os.getpid()))
    monkeypatch.setattr(mega_analysis_job, "EXECUTION_LOCK_PATH", execution_lock)

    state_before = {"last_attempt_utc": "2026-08-22T00:00:00+00:00"}
    state_after = {
        "last_attempt_utc": "2026-08-23T14:00:00+00:00",
        "last_status": "success",
        "last_run_date_utc": mega_analysis_job.datetime.now(mega_analysis_job.timezone.utc).date().isoformat(),
    }
    read_state_calls = {"count": 0}

    def _fake_read_state():
        read_state_calls["count"] += 1
        return state_before if read_state_calls["count"] == 1 else state_after

    with (
        patch("mega_analysis_job.read_state", side_effect=_fake_read_state),
        patch("mega_analysis_job.run_with_timeout") as mock_run_with_timeout,
    ):
        mega_analysis_job.main()  # must not raise, must not block

    # Never attempted the guarded work at all -- lost the race gracefully.
    mock_run_with_timeout.assert_not_called()
    # The other process's own lock is left completely untouched.
    assert execution_lock.read_text() == str(os.getpid())
    # And this job's OWN lock is still released as normal.
    assert not lock.exists()
