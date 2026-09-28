"""Unattended entry point for Trade Audit — the fourth, independent
agent-like role in this app's automation family, alongside Mega Session
(mega_analysis_job.py), Clerk (clerk_execution_job.py), and Researcher
(researcher_job.py). See ai/trade_audit.py's own module docstring for
the full design; this is the job-runner shell only, same skeleton as
the other three.

Trigger: once daily, in a config.TRADE_AUDIT_GRACE_MINUTES-wide window
starting config.TRADE_AUDIT_SETTLEMENT_BUFFER_MINUTES after the real
US-forex/FTMO daily session close (5:00 PM America/New_York local time),
computed FRESH EACH DAY via zoneinfo — DST-aware, unlike every other job
in this family; see ai/trade_audit.py::_ny_close_trigger_utc and
config.py's own TRADE_AUDIT_TRIGGER_HOUR_LOCAL comment for why.

Uses the same PID-liveness-aware lock file mechanism as the other three
jobs (job_lock.py); lock path/staleness constants live in
ai/trade_audit.py, matching the other three jobs' own convention.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from ai.trade_audit import (
    TRADE_AUDIT_LOCK_PATH,
    TRADE_AUDIT_LOCK_STALE_AFTER_SECONDS,
    is_trade_audit_due,
    read_trade_audit_enabled,
    read_trade_audit_state,
    run_trade_audit_check,
)
from job_lock import acquire_lock, release_lock
from utils import run_with_timeout

_LOG_PATH = Path(__file__).resolve().parent / "trade_audit_log.txt"
_LOCK_PATH = TRADE_AUDIT_LOCK_PATH
_LOCK_STALE_AFTER_SECONDS = TRADE_AUDIT_LOCK_STALE_AFTER_SECONDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("trade_audit_job")

_TIMEOUT_SENTINEL = object()


def main() -> None:
    now_utc = datetime.now(timezone.utc)
    state = read_trade_audit_state()

    # Fast-path only, same reasoning as the other three jobs' own
    # identical check — purely so a disabled Trade Audit doesn't touch
    # the lock file on every single OS-level poll.
    if not read_trade_audit_enabled():
        return

    if not is_trade_audit_due(now_utc, state=state):
        return

    if not acquire_lock(_LOCK_PATH, _LOCK_STALE_AFTER_SECONDS):
        return
    try:
        logger.info("Due — running Trade Audit (now=%s UTC).", now_utc)
        try:
            result = run_with_timeout(
                run_trade_audit_check,
                config.TRADE_AUDIT_RUN_TIMEOUT_SECONDS,
                default=_TIMEOUT_SENTINEL,
                catch_exceptions=False,
            )
            if result is _TIMEOUT_SENTINEL:
                logger.error(
                    "Trade Audit run timed out after %.0f minutes",
                    config.TRADE_AUDIT_RUN_TIMEOUT_SECONDS / 60,
                )
        except Exception:
            logger.exception("Trade Audit run failed")
        logger.info("Run finished; see this log and %s for the outcome.", read_trade_audit_state())
    finally:
        release_lock(_LOCK_PATH)


if __name__ == "__main__":
    main()
