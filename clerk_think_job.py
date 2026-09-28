"""Unattended THINKING job of the Execution Clerk: the slow, model-driven half, kept apart from the fast acting half.

Runs every minute from the tray but does real work only when a thinking pass is due (a new Mega suggestion has no verdicts yet, or the
review interval has passed): `run_clerk_execution_check(think=True)` builds the same lists an acting poll does, asks the local model
about each free-text pending trigger, each held position's invalidation condition and each tactical review, and stores the verdicts
(ai/clerk_thinking.py) - with NO order, settlement, journal or desk-status side effects. It has its own lock and never takes the
execution lock, so the acting passes (clerk_fast_job.py, clerk_execution_job.py) keep placing, amending and closing orders in seconds
while it thinks for minutes.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from ai import clerk_thinking
from ai.clerk_execution import (
    read_clerk_execution_enabled,
    read_clerk_execution_interval_minutes,
    read_latest_suggestion,
    run_clerk_execution_check,
)
from ai.mega_analysis import mega_session_is_live, read_progress, read_state
from job_lock import acquire_lock, release_lock
from utils import run_with_timeout

_LOG_PATH = Path(__file__).resolve().parent / "clerk_think_log.txt"
_LOCK_PATH = Path(__file__).resolve().parent / "clerk_think.lock"
_LOCK_STALE_AFTER_SECONDS = 30 * 60
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_PATH, encoding="utf-8")],
)
logger = logging.getLogger("clerk_think_job")
_TIMEOUT = object()


def main() -> None:
    if not config.CLERK_THINK_SPLIT_ENABLED or not read_clerk_execution_enabled():
        return
    if mega_session_is_live(read_progress(), read_state()):
        return
    suggestion = read_latest_suggestion()
    if not suggestion:
        return
    now = datetime.now(timezone.utc)
    if not clerk_thinking.is_think_due(
        clerk_thinking.load_cache(), suggestion.get("generated_utc"), now, read_clerk_execution_interval_minutes()
    ):
        return
    if not acquire_lock(_LOCK_PATH, _LOCK_STALE_AFTER_SECONDS):
        return
    try:
        logger.info("Thinking pass starting.")
        result = run_with_timeout(
            lambda: run_clerk_execution_check(think=True),
            config.CLERK_EXECUTION_RUN_TIMEOUT_SECONDS, default=_TIMEOUT, catch_exceptions=False,
        )
        if result is _TIMEOUT:
            logger.error("Thinking pass timed out after %.0f minutes", config.CLERK_EXECUTION_RUN_TIMEOUT_SECONDS / 60)
    except Exception:
        logger.exception("Thinking pass failed")
    finally:
        release_lock(_LOCK_PATH)


if __name__ == "__main__":
    main()
