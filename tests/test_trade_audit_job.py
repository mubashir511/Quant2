from unittest.mock import patch

import pytest

import config
import trade_audit_job

# The PID-liveness lock mechanism itself lives in job_lock.py and is
# covered by tests/test_job_lock.py — this file only exercises what's
# specific to trade_audit_job.py: its own due-check gate and the lock
# lifecycle around main(), mirroring tests/test_clerk_execution_job.py
# exactly.


@pytest.fixture(autouse=True)
def _fixed_enabled_file(tmp_path):
    with patch.object(config, "TRADE_AUDIT_ENABLED_FILE", str(tmp_path / "trade_audit_enabled.json")):
        yield


@patch("trade_audit_job.run_trade_audit_check")
@patch("trade_audit_job.is_trade_audit_due", return_value=True)
def test_main_acquires_and_releases_lock_around_a_due_run(mock_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "trade_audit.lock"
    monkeypatch.setattr(trade_audit_job, "_LOCK_PATH", lock)
    lock_held_during_run = {}

    def _check_lock_held(*args, **kwargs):
        lock_held_during_run["exists"] = lock.exists()

    with patch("trade_audit_job.run_with_timeout", side_effect=lambda fn, *a, **kw: fn()):
        mock_run.side_effect = _check_lock_held
        trade_audit_job.main()

    assert lock_held_during_run["exists"] is True
    assert not lock.exists()
    mock_run.assert_called_once()


@patch("trade_audit_job.run_trade_audit_check")
@patch("trade_audit_job.is_trade_audit_due", return_value=True)
def test_main_skips_the_run_when_the_lock_is_already_held(mock_due, mock_run, tmp_path, monkeypatch):
    import os

    lock = tmp_path / "trade_audit.lock"
    lock.write_text(str(os.getpid()))
    monkeypatch.setattr(trade_audit_job, "_LOCK_PATH", lock)

    trade_audit_job.main()

    mock_run.assert_not_called()
    assert lock.read_text() == str(os.getpid())


@patch("trade_audit_job.run_trade_audit_check")
@patch("trade_audit_job.is_trade_audit_due", return_value=False)
def test_main_never_touches_the_lock_when_not_due(mock_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "trade_audit.lock"
    monkeypatch.setattr(trade_audit_job, "_LOCK_PATH", lock)

    trade_audit_job.main()

    mock_run.assert_not_called()
    assert not lock.exists()


@patch("trade_audit_job.run_trade_audit_check")
@patch("trade_audit_job.is_trade_audit_due", return_value=True)
@patch("trade_audit_job.read_trade_audit_enabled", return_value=False)
def test_main_skips_before_even_checking_is_due_when_disabled(mock_enabled, mock_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "trade_audit.lock"
    monkeypatch.setattr(trade_audit_job, "_LOCK_PATH", lock)

    trade_audit_job.main()

    mock_due.assert_not_called()  # the toggle is a fast-path checked first
    mock_run.assert_not_called()
    assert not lock.exists()


@patch("trade_audit_job.run_trade_audit_check")
@patch("trade_audit_job.is_trade_audit_due", return_value=True)
def test_main_releases_lock_even_if_run_with_timeout_raises(mock_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "trade_audit.lock"
    monkeypatch.setattr(trade_audit_job, "_LOCK_PATH", lock)

    with patch("trade_audit_job.run_with_timeout", side_effect=RuntimeError("boom")):
        trade_audit_job.main()  # must not raise -- caught and logged internally

    assert not lock.exists()


@patch("trade_audit_job.run_trade_audit_check")
@patch("trade_audit_job.is_trade_audit_due", return_value=True)
def test_main_logs_timeout_without_raising(mock_due, mock_run, tmp_path, monkeypatch):
    lock = tmp_path / "trade_audit.lock"
    monkeypatch.setattr(trade_audit_job, "_LOCK_PATH", lock)

    with patch("trade_audit_job.run_with_timeout", return_value=trade_audit_job._TIMEOUT_SENTINEL):
        trade_audit_job.main()  # must not raise

    assert not lock.exists()
