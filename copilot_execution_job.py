"""Unattended entry point for the FTMO Copilot execution check — the
"clerk/executioner" half of the boardroom architecture (see
ai/copilot_execution.py's own module docstring for the full design).

Invoked by the "Quant2CopilotExecution" Windows Scheduled Task, polling
every few minutes at the OS level (a poll, not a one-shot trigger — same
reasoning as mega_analysis_job.py: polling sidesteps local-timezone/DST
drift and naturally implements "the PC was off through this window,
skip it" as a side effect of the grace-window check in
ai.copilot_execution.is_execution_due). Exits almost immediately, with
no MT5 connection attempt at all, on every poll that isn't currently due
— the actual check only runs once per
config.COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES-sized window (default
15 minutes; originally hourly, tightened on direct user request
2026-08-23 since the check itself is cheap). The OS-level Scheduled Task
poll cadence itself is unrelated and unchanged by that constant — it
only needs to be frequent enough to land inside whatever window this
constant defines.

Uses the same PID-liveness-aware lock file mechanism as
mega_analysis_job.py (see job_lock.py, extracted from that file once
this second unattended job needed the exact same logic) so an
interrupting poll can never mistake a still-running check for "time to
start over." The lock path/staleness constants themselves live in
ai/copilot_execution.py (EXECUTION_LOCK_PATH/EXECUTION_LOCK_STALE_AFTER_
SECONDS), not here — mega_analysis_job.py's own inline "run right after
a successful mega session" pass acquires the literal same lock before
calling run_copilot_execution_check() too, so the two independent call
sites can never run that function concurrently (a real race found on
audit — see ai.copilot_execution's own comment next to those constants).
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from ai.copilot_execution import (
    EXECUTION_LOCK_PATH,
    EXECUTION_LOCK_STALE_AFTER_SECONDS,
    is_execution_due,
    read_copilot_execution_enabled,
    read_execution_state,
    run_copilot_execution_check,
)
from job_lock import acquire_lock, release_lock
from utils import run_with_timeout

_LOG_PATH = Path(__file__).resolve().parent / "copilot_execution_log.txt"
_LOCK_PATH = EXECUTION_LOCK_PATH
_LOCK_STALE_AFTER_SECONDS = EXECUTION_LOCK_STALE_AFTER_SECONDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("copilot_execution_job")

_TIMEOUT_SENTINEL = object()


def main() -> None:
    now_utc = datetime.now(timezone.utc)
    state = read_execution_state()

    # Fast-path only — the authoritative gate lives inside
    # run_copilot_execution_check itself, so this toggle is respected
    # uniformly across all three of its call sites. Checked here too,
    # before is_execution_due/the lock, purely so a disabled clerk
    # doesn't touch the lock file on every single OS-level poll.
    if not read_copilot_execution_enabled():
        return

    if not is_execution_due(now_utc, state=state):
        return

    if not acquire_lock(_LOCK_PATH, _LOCK_STALE_AFTER_SECONDS):
        return
    try:
        logger.info("Due — running the Copilot execution check (now=%s UTC).", now_utc)
        try:
            result = run_with_timeout(
                run_copilot_execution_check,
                config.COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS,
                default=_TIMEOUT_SENTINEL,
                catch_exceptions=False,
            )
            if result is _TIMEOUT_SENTINEL:
                logger.error(
                    "Copilot execution check timed out after %.0f minutes",
                    config.COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS / 60,
                )
        except Exception:
            logger.exception("Copilot execution check failed")
        logger.info("Run finished; see this log and %s for the outcome.", read_execution_state())
    finally:
        release_lock(_LOCK_PATH)


if __name__ == "__main__":
    main()
