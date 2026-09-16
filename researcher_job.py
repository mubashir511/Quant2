"""Unattended entry point for Researcher — the third, independent agent
standing alongside the Mega Session (mega_analysis_job.py) and Clerk
(clerk_execution_job.py). See ai/researcher.py's own module docstring
for the full design; this is Phase 1, standalone — nothing here is
wired into either existing job yet.

Same skeleton as the other two jobs: invoked by its own Windows
Scheduled Task (not yet registered — a manual, out-of-repo step, same
as the other two), polling every few minutes at the OS level. Exits
almost immediately, with no MT5 connection attempt at all, on every poll
that isn't currently due — the actual research run only happens once per
day, at config.RESEARCHER_TRIGGER_HOUR_UTC:RESEARCHER_TRIGGER_MINUTE_UTC
(default 12:30 UTC — about an hour before the US cash-equity-index open
at 13:30 UTC, the same real anchor already used by Mega Session's own
trigger), within a config.RESEARCHER_GRACE_MINUTES window.

Uses the same PID-liveness-aware lock file mechanism as the other two
jobs (job_lock.py) so an interrupting poll can never mistake a still-
running run for "time to start over." The lock path/staleness constants
live in ai/researcher.py (RESEARCHER_LOCK_PATH/RESEARCHER_LOCK_STALE_
AFTER_SECONDS), not here, matching ai.clerk_execution's own convention.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from ai.researcher import (
    RESEARCHER_LOCK_PATH,
    RESEARCHER_LOCK_STALE_AFTER_SECONDS,
    is_researcher_due,
    read_researcher_enabled,
    read_researcher_state,
    run_researcher_check,
)
from job_lock import acquire_lock, release_lock
from utils import run_with_timeout

_LOG_PATH = Path(__file__).resolve().parent / "researcher_log.txt"
_LOCK_PATH = RESEARCHER_LOCK_PATH
_LOCK_STALE_AFTER_SECONDS = RESEARCHER_LOCK_STALE_AFTER_SECONDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("researcher_job")

_TIMEOUT_SENTINEL = object()


def main() -> None:
    now_utc = datetime.now(timezone.utc)
    state = read_researcher_state()

    # Fast-path only, same reasoning as clerk_execution_job.py's own
    # identical check — purely so a disabled Researcher doesn't touch
    # the lock file on every single OS-level poll.
    if not read_researcher_enabled():
        return

    if not is_researcher_due(now_utc, state=state):
        return

    if not acquire_lock(_LOCK_PATH, _LOCK_STALE_AFTER_SECONDS):
        return
    try:
        logger.info("Due — running Researcher (now=%s UTC).", now_utc)
        try:
            result = run_with_timeout(
                run_researcher_check,
                config.RESEARCHER_RUN_TIMEOUT_SECONDS,
                default=_TIMEOUT_SENTINEL,
                catch_exceptions=False,
            )
            if result is _TIMEOUT_SENTINEL:
                logger.error(
                    "Researcher run timed out after %.0f minutes",
                    config.RESEARCHER_RUN_TIMEOUT_SECONDS / 60,
                )
        except Exception:
            logger.exception("Researcher run failed")
        logger.info("Run finished; see this log and %s for the outcome.", read_researcher_state())
    finally:
        release_lock(_LOCK_PATH)


if __name__ == "__main__":
    main()
