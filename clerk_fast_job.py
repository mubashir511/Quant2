"""Unattended one-minute FAST LANE of the Execution Clerk (ai/clerk_execution.py run_clerk_execution_check(fast=True)).

The full poll (clerk_execution_job.py) runs every review interval AND then waits for the local model, so between two polls a
Mega entry that failed, a structured trigger that fired or a stop that should ratchet can wait a whole candle or more. This job runs
the same deterministic pipeline every minute with NO model call. Cheap when there is nothing to do: it exits before connecting to MT5
unless the Clerk is on, no Mega session is running, the Mega suggestion has something for the fast lane, the full poll is not due (it
will run right now and would only be delayed by taking the lock) and no other process holds the execution lock.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from ai.clerk_execution import (
    EXECUTION_LOCK_PATH,
    EXECUTION_LOCK_STALE_AFTER_SECONDS,
    _load_settlement,
    is_execution_due,
    read_clerk_execution_enabled,
    read_latest_suggestion,
    run_clerk_execution_check,
)
from ai.mega_analysis import mega_session_is_live, read_progress, read_state
from job_lock import acquire_lock_wait, release_lock

_LOG_PATH = Path(__file__).resolve().parent / "clerk_fast_log.txt"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_PATH, encoding="utf-8")],
)
logger = logging.getLogger("clerk_fast_job")


def fast_lane_has_work(suggestion: dict | None, settlement: dict) -> bool:
    """True when the latest Mega suggestion gives the fast lane something to do: an immediate target with risk, a structured-trigger
    Pending Setup, or anything already resting/filled that the deterministic guards and breakers watch."""
    if not suggestion:
        return False
    if any((e or {}).get("pct", 0) > 0 for s, e in (suggestion.get("immediate_allocation") or {}).items() if s.upper() != "CASH"):
        return True
    if any(p.get("trigger") for p in suggestion.get("pending_setups") or []):
        return True
    return any(rec.get("state") in ("order_placed", "filled") for rec in (settlement.get("settled") or {}).values())


def main() -> None:
    if not config.CLERK_FAST_LANE_ENABLED or not read_clerk_execution_enabled():
        return
    now_utc = datetime.now(timezone.utc)
    if mega_session_is_live(read_progress(), read_state()) or is_execution_due(now_utc):
        return
    if not fast_lane_has_work(read_latest_suggestion(), _load_settlement()):
        return
    if not acquire_lock_wait(EXECUTION_LOCK_PATH, EXECUTION_LOCK_STALE_AFTER_SECONDS, config.CLERK_LOCK_WAIT_SECONDS):
        return
    try:
        run_clerk_execution_check(fast=True)
    except Exception:
        logger.exception("Fast-lane pass failed")
    finally:
        release_lock(EXECUTION_LOCK_PATH)


if __name__ == "__main__":
    main()
