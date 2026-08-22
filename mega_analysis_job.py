"""Unattended entry point for the daily FTMO "mega market analysis."

Invoked by the "Quant2MegaAnalysis" Windows Scheduled Task every 15
minutes (a poll, not a one-shot trigger — see ai/mega_analysis.py's own
docstrings for why: polling in plain UTC sidesteps both this machine's
local-timezone/DST quirks and Windows Task Scheduler's local-time-trigger
drift entirely, and naturally implements "skip a day the PC was off for,
don't run it late" as a side effect of the grace-window check). Exits
almost immediately, with no MT5 connection attempt at all, on every poll
that isn't currently due — the actual analysis only runs once a day.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from ai.mega_analysis import is_due, next_run_utc, read_state, run_scheduled_mega_analysis

_LOG_PATH = Path(__file__).resolve().parent / "mega_analysis_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("mega_analysis_job")


def main() -> None:
    now_utc = datetime.now(timezone.utc)
    state = read_state()

    if not is_due(now_utc, state=state):
        logger.info(
            "Not due (now=%s UTC, last_run_date_utc=%s, next=%s UTC) — exiting.",
            now_utc.strftime("%Y-%m-%d %H:%M:%S"),
            state.get("last_run_date_utc", "never"),
            next_run_utc(now_utc, state=state).strftime("%Y-%m-%d %H:%M:%S"),
        )
        return

    logger.info("Due — starting the FTMO mega market analysis (now=%s UTC).", now_utc)
    run_scheduled_mega_analysis()
    logger.info("Run finished; see this log and %s for the outcome.", read_state())


if __name__ == "__main__":
    main()
