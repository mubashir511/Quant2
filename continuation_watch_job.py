"""Unattended one-minute entry point for Continuation Watch (ai/continuation_hunter.py) — phase 2.

Started every minute by the tray's own timer (quant_app_tray.ps1), same pattern as clerk_think_job.py. Its own
lock (continuation_watch.lock), separate from the Clerk's execution lock — LOG-ONLY today (config.
CONTINUATION_WATCH_LOG_ONLY, default True) never places an order or touches the settlement file, so it never
needs to coordinate with a Clerk poll the way the Sentinel does. Cheap when nothing is being watched: no MT5
connection at all unless there's at least one symbol in continuation_watch_state.json.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from ai.clerk_execution import read_clerk_execution_enabled
from ai.continuation_hunter import load_state, run_continuation_watch_check
from ai.mega_analysis import mega_session_is_live, read_progress, read_state
from job_lock import acquire_lock, release_lock

_LOG_PATH = Path(__file__).resolve().parent / "continuation_watch_log.txt"
_LOCK_PATH = Path(__file__).resolve().parent / "continuation_watch.lock"
_LOCK_STALE_AFTER_SECONDS = 10 * 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_PATH, encoding="utf-8")],
)
logger = logging.getLogger("continuation_watch_job")


def main() -> None:
    if not config.CONTINUATION_WATCH_ENABLED or not read_clerk_execution_enabled():
        return
    if mega_session_is_live(read_progress(), read_state()):
        return
    if not (load_state().get("watching") or {}):
        return  # nothing to check right now — no MT5 call, no lock
    if not acquire_lock(_LOCK_PATH, _LOCK_STALE_AFTER_SECONDS):
        return
    try:
        from data.mt5_source import connect

        connect(login=config.FTMO_MT5_LOGIN, password=config.FTMO_MT5_PASSWORD, server=config.FTMO_MT5_SERVER)
        summary = run_continuation_watch_check(datetime.now(timezone.utc))
        if summary["checked"]:
            logger.info("Continuation watch tick: %s", summary)
    except Exception:
        logger.exception("Continuation watch tick failed")
    finally:
        release_lock(_LOCK_PATH)


if __name__ == "__main__":
    main()
