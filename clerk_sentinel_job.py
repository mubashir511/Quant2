"""Unattended one-minute entry point for the Sentinel (ai/sentinel.py) - the fast, LLM-free profit-trail ratchet.

Started every minute by the tray's own timer (quant_app_tray.ps1), exactly like clerk_execution_job.py. Cheap by design:
it exits immediately, with no MT5 connection at all, when the Sentinel is off, the Clerk is off, a mega session is running,
the settlement file tracks no filled position, or the Clerk's own execution lock is held (a Clerk poll is in flight and it
manages stops itself). While it runs it HOLDS that same lock, so a Clerk poll can never overlap a Sentinel tick.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from ai.clerk_execution import (
    EXECUTION_LOCK_PATH,
    EXECUTION_LOCK_STALE_AFTER_SECONDS,
    _load_settlement,
    read_clerk_execution_enabled,
)
from ai.mega_analysis import mega_session_is_live, read_progress, read_state
from ai.sentinel import run_sentinel_check
from job_lock import acquire_lock_wait, release_lock

_LOG_PATH = Path(__file__).resolve().parent / "clerk_sentinel_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_PATH, encoding="utf-8")],
)
logger = logging.getLogger("clerk_sentinel_job")


def _has_filled_position(settlement: dict) -> bool:
    return any(rec.get("state") == "filled" for rec in (settlement.get("settled") or {}).values())


def main() -> None:
    if not config.SENTINEL_ENABLED or not read_clerk_execution_enabled():
        return
    if mega_session_is_live(read_progress(), read_state()):
        return
    if not _has_filled_position(_load_settlement()):
        return
    if not acquire_lock_wait(EXECUTION_LOCK_PATH, EXECUTION_LOCK_STALE_AFTER_SECONDS, config.CLERK_LOCK_WAIT_SECONDS):
        return  # a Clerk poll is in flight - it manages stops itself
    try:
        from data.mt5_source import connect

        connect(login=config.FTMO_MT5_LOGIN, password=config.FTMO_MT5_PASSWORD, server=config.FTMO_MT5_SERVER)
        summary = run_sentinel_check(datetime.now(timezone.utc))
        if summary["actions"]:
            logger.info("Sentinel tick: %s", summary["actions"])
    except Exception:
        logger.exception("Sentinel tick failed")
    finally:
        release_lock(EXECUTION_LOCK_PATH)


if __name__ == "__main__":
    main()
