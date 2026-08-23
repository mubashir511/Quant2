import os
import subprocess
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import job_lock


# --- pid_is_alive: real liveness checks, not just mocked-away plumbing ---


def test_pid_is_alive_true_for_a_real_running_process():
    # This test process's own PID is unambiguously alive right now.
    assert job_lock.pid_is_alive(os.getpid()) is True


def test_pid_is_alive_false_for_a_pid_that_has_exited():
    # Real bug this guards against: os.kill(pid, 0) does NOT work for a
    # liveness check on Windows — it raises WinError 87 ("the parameter
    # is incorrect") regardless of whether the PID exists, since 0 isn't
    # a signal Windows' os.kill implementation accepts (confirmed
    # empirically before writing this fix). Spawns a real short-lived
    # process and waits for it to actually exit, so this PID is
    # guaranteed dead rather than guessed at.
    proc = subprocess.Popen(["cmd", "/c", "exit"])
    proc.wait(timeout=10)
    assert job_lock.pid_is_alive(proc.pid) is False


def test_pid_is_alive_none_when_the_check_itself_fails():
    with patch("subprocess.run", side_effect=OSError("tasklist not found")):
        assert job_lock.pid_is_alive(12345) is None


# --- acquire_lock / release_lock ---


def test_acquire_lock_succeeds_when_no_lock_exists(tmp_path):
    lock = tmp_path / "some.lock"

    assert job_lock.acquire_lock(lock, 90 * 60) is True
    assert lock.exists()
    assert lock.read_text().strip().isdigit()


def test_acquire_lock_fails_when_the_holding_pid_is_confirmed_alive(tmp_path):
    lock = tmp_path / "some.lock"
    lock.write_text(str(os.getpid()))  # this test process is genuinely alive

    # Real user-facing bug this guards against: a still-running attempt
    # must NOT be treated as "safe to restart" — see this module's own
    # docstring for the live incident (every daily run since the
    # scheduler was set up got reset to instrument #1 every 15 minutes,
    # never once completing).
    assert job_lock.acquire_lock(lock, 90 * 60) is False
    assert lock.read_text() == str(os.getpid())  # untouched, not overwritten


def test_acquire_lock_succeeds_immediately_when_the_holding_pid_is_confirmed_dead(tmp_path):
    # Real bug this guards against, found live the same day: a PURE
    # age-based staleness check left a lock pointing at an already-dead
    # PID (killed externally mid-run, so it never reached release_lock)
    # blocking recovery for up to the whole staleness window, since the
    # file's own mtime doesn't change just because its owner died. A
    # confirmed-dead PID must be taken over immediately, regardless of
    # the lock's age.
    lock = tmp_path / "some.lock"
    lock.write_text("999999")
    # Fresh mtime deliberately — proves this is a REAL liveness check,
    # not just the age-based fallback happening to also say "take over".
    with patch("job_lock.pid_is_alive", return_value=False) as mock_alive:
        assert job_lock.acquire_lock(lock, 90 * 60) is True
    mock_alive.assert_called_once_with(999999)
    assert lock.read_text().strip() != "999999"  # taken over by this process's own PID


def test_acquire_lock_falls_back_to_age_and_blocks_when_liveness_check_unavailable(tmp_path):
    lock = tmp_path / "some.lock"
    lock.write_text("12345")

    with patch("job_lock.pid_is_alive", return_value=None):
        assert job_lock.acquire_lock(lock, 90 * 60) is False
    assert lock.read_text() == "12345"


def test_acquire_lock_falls_back_to_age_and_takes_over_a_stale_one_when_liveness_check_unavailable(tmp_path):
    lock = tmp_path / "some.lock"
    lock.write_text("12345")
    stale_after_seconds = 90 * 60
    stale_time = (
        datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds + 60)
    ).timestamp()
    os.utime(lock, (stale_time, stale_time))

    with patch("job_lock.pid_is_alive", return_value=None):
        assert job_lock.acquire_lock(lock, stale_after_seconds) is True
    assert lock.read_text().strip() != "12345"


def test_acquire_lock_falls_back_to_age_when_lock_content_is_not_a_pid(tmp_path):
    # Corrupt/empty lock content (e.g. a write that got cut off mid-way)
    # must degrade to the age-based fallback, not crash or silently
    # treat it as "no lock at all".
    lock = tmp_path / "some.lock"
    lock.write_text("not-a-pid")

    assert job_lock.acquire_lock(lock, 90 * 60) is False  # fresh mtime -> still treated as held


def test_release_lock_removes_the_file(tmp_path):
    lock = tmp_path / "some.lock"
    job_lock.acquire_lock(lock, 90 * 60)
    assert lock.exists()

    job_lock.release_lock(lock)
    assert not lock.exists()


def test_release_lock_is_a_no_op_when_nothing_to_remove(tmp_path):
    lock = tmp_path / "some.lock"
    job_lock.release_lock(lock)  # must not raise
