"""Shared PID-liveness-aware lock-file mechanism for this project's
unattended scheduled jobs (mega_analysis_job.py, copilot_execution_job.py)
— extracted from mega_analysis_job.py, where it first shipped after a
real live incident (see acquire_lock's own docstring), so a second
scheduled job doesn't copy-paste a second, independently-drifting copy
of logic that already has real bug history.
"""

import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def pid_is_alive(pid: int) -> bool | None:
    """True/False from a real liveness check via `tasklist` (Windows-
    native, no extra dependency — confirmed empirically to correctly
    report both a live and a just-terminated PID, unlike
    `os.kill(pid, 0)`, which raises WinError 87 on Windows regardless of
    whether the PID exists). None if the check itself couldn't be
    performed at all (tasklist missing/erroring/timing out) — callers
    must fall back to a different signal in that case, not guess."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Could not check liveness of PID %d via tasklist: %s", pid, e)
        return None
    return str(pid) in result.stdout


def acquire_lock(lock_path: Path, stale_after_seconds: float) -> bool:
    """True if this invocation now holds the lock (this process's own PID
    written to `lock_path`). False — and logs exactly why — if an
    existing lock's own recorded PID is confirmed still alive; the
    caller must not proceed with the guarded work in that case.

    A lock whose PID is confirmed DEAD is taken over immediately,
    regardless of the lock file's own age — the real incident this fixes
    (mega_analysis_job.py, 2026-08-22): a time-only staleness check left
    a dead-owner lock blocking recovery for up to `stale_after_seconds`
    needlessly, since a killed process's lock file doesn't update its
    own mtime just because its owner died. The age-based check only
    runs as a fallback when PID liveness itself can't be determined."""
    if lock_path.exists():
        held_pid: int | None = None
        try:
            held_pid = int(lock_path.read_text().strip())
        except (OSError, ValueError):
            held_pid = None

        if held_pid is not None:
            alive = pid_is_alive(held_pid)
            if alive is True:
                logger.warning(
                    "Lock file %s is held by PID %d, which is still running — skipping this "
                    "poll rather than restarting the guarded work from scratch.",
                    lock_path, held_pid,
                )
                return False
            if alive is False:
                logger.warning(
                    "Lock file %s is held by PID %d, which is no longer running (killed "
                    "externally mid-run, not a graceful exit) — taking over immediately.",
                    lock_path, held_pid,
                )
                lock_path.write_text(str(os.getpid()))
                return True
            # alive is None: the liveness check itself failed — fall through
            # to the age-based fallback below rather than guessing either way.

        age_seconds = time.time() - lock_path.stat().st_mtime
        if age_seconds < stale_after_seconds:
            logger.warning(
                "Lock file %s is only %.1f minutes old (stale threshold %.0f minutes) and its "
                "PID's liveness couldn't be confirmed either way — treating it as still held; "
                "skipping this poll rather than restarting the guarded work from scratch.",
                lock_path, age_seconds / 60, stale_after_seconds / 60,
            )
            return False
        logger.warning(
            "Lock file %s is %.1f minutes old — past the staleness threshold, treating the "
            "prior holder as dead and taking over.",
            lock_path, age_seconds / 60,
        )
    lock_path.write_text(str(os.getpid()))
    return True


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("Could not remove lock file %s: %s", lock_path, e)
