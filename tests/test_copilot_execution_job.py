from unittest.mock import patch

import pytest

import config
import copilot_execution_job

# The PID-liveness lock mechanism itself lives in job_lock.py and is
# covered by tests/test_job_lock.py — this file only exercises what's
# specific to copilot_execution_job.py: its own due-check gate and the
# lock lifecycle around main().


@pytest.fixture(autouse=True)
def _fixed_enabled_file(tmp_path):
    # Same real-file-collision lesson as test_mega_analysis_job.py's own
    # fixture (2026-08-23): main() now checks read_copilot_execution_
    # enabled() before is_execution_due(), and a real, live toggle click
    # in the running app would otherwise silently break every test here
    # that expects a due run to actually fire.
    with patch.object(config, "COPILOT_EXECUTION_ENABLED_FILE", str(tmp_path / "copilot_execution_enabled.json")):
        yield


@patch("copilot_execution_job.run_copilot_execution_check")
@patch("copilot_execution_job.is_execution_due", return_value=True)
def test_main_acquires_and_releases_lock_around_a_due_run(mock_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "copilot_execution.lock"
    monkeypatch.setattr(copilot_execution_job, "_LOCK_PATH", lock)
    lock_held_during_run = {}

    def _check_lock_held(*args, **kwargs):
        lock_held_during_run["exists"] = lock.exists()

    with patch("copilot_execution_job.run_with_timeout", side_effect=lambda fn, *a, **kw: fn()):
        mock_run.side_effect = _check_lock_held
        copilot_execution_job.main()

    assert lock_held_during_run["exists"] is True
    assert not lock.exists()
    mock_run.assert_called_once()


@patch("copilot_execution_job.run_copilot_execution_check")
@patch("copilot_execution_job.is_execution_due", return_value=True)
def test_main_skips_the_run_when_the_lock_is_already_held(mock_due, mock_run, tmp_path, monkeypatch):
    import os

    lock = tmp_path / "copilot_execution.lock"
    lock.write_text(str(os.getpid()))
    monkeypatch.setattr(copilot_execution_job, "_LOCK_PATH", lock)

    copilot_execution_job.main()

    mock_run.assert_not_called()
    assert lock.read_text() == str(os.getpid())


@patch("copilot_execution_job.run_copilot_execution_check")
@patch("copilot_execution_job.is_execution_due", return_value=False)
def test_main_never_touches_the_lock_when_not_due(mock_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "copilot_execution.lock"
    monkeypatch.setattr(copilot_execution_job, "_LOCK_PATH", lock)

    copilot_execution_job.main()

    mock_run.assert_not_called()
    assert not lock.exists()


@patch("copilot_execution_job.run_copilot_execution_check")
@patch("copilot_execution_job.is_execution_due", return_value=True)
@patch("copilot_execution_job.read_copilot_execution_enabled", return_value=False)
def test_main_skips_before_even_checking_is_due_when_disabled(mock_enabled, mock_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "copilot_execution.lock"
    monkeypatch.setattr(copilot_execution_job, "_LOCK_PATH", lock)

    copilot_execution_job.main()

    mock_due.assert_not_called()  # the toggle is a fast-path checked first
    mock_run.assert_not_called()
    assert not lock.exists()


@patch("copilot_execution_job.run_copilot_execution_check")
@patch("copilot_execution_job.is_execution_due", return_value=True)
def test_main_releases_lock_even_if_run_with_timeout_raises(mock_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "copilot_execution.lock"
    monkeypatch.setattr(copilot_execution_job, "_LOCK_PATH", lock)

    with patch("copilot_execution_job.run_with_timeout", side_effect=RuntimeError("boom")):
        copilot_execution_job.main()  # must not raise -- caught and logged internally

    assert not lock.exists()


@patch("copilot_execution_job.run_copilot_execution_check")
@patch("copilot_execution_job.is_execution_due", return_value=True)
def test_main_logs_timeout_without_raising(mock_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "copilot_execution.lock"
    monkeypatch.setattr(copilot_execution_job, "_LOCK_PATH", lock)

    with patch("copilot_execution_job.run_with_timeout", return_value=copilot_execution_job._TIMEOUT_SENTINEL):
        copilot_execution_job.main()  # must not raise

    assert not lock.exists()
