"""Unattended entry point for the daily FTMO "mega market analysis."

Invoked by the "Quant2MegaAnalysis" Windows Scheduled Task every 15
minutes (a poll, not a one-shot trigger — see ai/mega_analysis.py's own
docstrings for why: polling in plain UTC sidesteps both this machine's
local-timezone/DST quirks and Windows Task Scheduler's local-time-trigger
drift entirely, and naturally implements "skip a day the PC was off for,
don't run it late" as a side effect of the grace-window check). Exits
almost immediately, with no MT5 connection attempt at all, on every poll
that isn't currently due — the actual analysis only runs once a day.

Real bug found live 2026-08-22 (user reported the scheduler "did not run
again today"): checked the real logs (mega_analysis_log.txt/
mega_analysis_bat_output.txt) against mega_analysis_state.json, which
turned out not to exist at all — meaning zero successful runs had EVER
been recorded since this scheduler was set up. The log showed the SAME
pattern on both days it had fired: "Due — starting..." begins, gets
partway through analyzing the 18 instruments (as far as #15/18 one run,
as little as #3/18 another), then a literal interrupt character appears
in the log at almost exactly the 15-minute mark, immediately followed by
the NEXT poll's own "Due — starting..." restarting the whole analysis
from instrument #1 again — never a timeout, never an exception, never a
completion. Windows Task Scheduler's own `MultipleInstances=IgnoreNew`
is configured (confirmed via Get-ScheduledTask) and SHOULD make exactly
this impossible, but its own operational event log was disabled on this
machine, so there's no definitive OS-level trace of what's actually
terminating the process — a real, currently-unresolved question, not
something this fix claims to explain away.

What this DOES fix, structurally, regardless of the exact external
cause: a simple lock file makes a second, interrupting invocation impossible
to mistake for "time to start over" — if a run is still within a generous
staleness window, a new poll skips instead of restarting from scratch.
This can't stop whatever is actually killing the original process, but it
does stop the "reset to instrument #1 every 15 minutes, forever" failure
mode, and makes a stuck lock itself a visible, diagnosable symptom (the
lock file's own timestamp) instead of an invisible one.

Confirmed live the same day, via a real manual test run: the lock does
stop the restart-every-15-minutes pattern (a poll correctly skipped while
the run was genuinely still active), but the run was then killed AGAIN
later — this time well past the per-instrument analysis, with no state
ever written — leaving a lock file pointing at a PID that no longer
existed. A pure time-based staleness check (the original version of this
function) would have kept treating that as "still running" for up to 90
more minutes, since the file's own mtime doesn't change just because its
owner died. Fixed by checking the recorded PID's REAL liveness first
(via `tasklist`, confirmed empirically against both a live and a
just-killed PID before relying on it — `os.kill(pid, 0)` does NOT work
for this on Windows, it raises WinError 87 regardless of whether the PID
exists, since 0 isn't a signal Windows' os.kill implementation accepts)
— a dead PID is taken over immediately, not after waiting out a timer.
The original age-based check is kept as a fallback for when the
liveness check itself can't be performed (e.g. tasklist unavailable),
not removed.

The lock-file mechanism itself (job_lock.py::acquire_lock/release_lock)
was later extracted into a shared module once a second unattended job
(copilot_execution_job.py) needed the exact same PID-liveness logic —
this file now just supplies its own lock path/staleness window to that
shared implementation rather than keeping a second, independently-
drifting private copy.

2026-08-23 update — inline "immediate execution" wiring: direct user
request ("if the mega session instructed for any trade immediately in
the analysis, that trade will be immediately executed by the copilot")
— rather than making a same-session trade wait for the next standalone
Copilot poll (originally up to ~59 minutes away when that poll was
hourly; now bounded by config.COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES,
tightened to 15 on direct user request 2026-08-23 since the check
itself is cheap), a successful run here also triggers one inline
Copilot execution-check pass immediately afterward, in the same
process. Gated strictly on `last_status == "success"` AND
`last_run_date_utc == today` (never after a timeout/error/cli_failed —
see ai.mega_analysis's own _RUN_TIMEOUT_SECONDS/run_with_timeout
docstring for why an abandoned timed-out thread must never race a fresh
inline call), and wrapped in its own try/except plus a separate, small
run_with_timeout ceiling (COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS) so a
hang or exception in the execution-check can never propagate out of
main() or make this job overrun. The standalone poll
(copilot_execution_job.py) continues independently afterward for
ongoing conditional-setup watching regardless of whether this inline
call ran.

Real concurrency bug found on a post-build audit, fixed the same day:
this inline call used to invoke run_copilot_execution_check() directly
with no lock protection at all — a genuinely different OS process
(copilot_execution_job.py's own standalone poll) could acquire
ITS OWN separate lock and start running the exact same function
concurrently, since nothing here ever competed for that lock. Fixed by
acquiring the literal same EXECUTION_LOCK_PATH (ai/copilot_execution.py)
before calling it, and skipping gracefully (not blocking/waiting) if the
standalone job already holds it — this inline pass is a lower-latency
bonus, not a guarantee, so losing the race just means the next standalone
poll (imminent regardless) picks up the same "immediate" trade instead.

2026-08-23 update — user-controlled enable/disable toggle: a toggle in
app.py's Mega Market Analysis box (default ON) writes
config.MEGA_ANALYSIS_ENABLED_FILE via ai.mega_analysis.
set_mega_analysis_enabled; this job checks read_mega_analysis_enabled()
first, before even the is_due() check, and exits immediately if the
user has turned it off — same "worst case is a missed poll, never a
silent extra run" caution as every other check in this function. The
toggle is also mutually exclusive with the manual "Suggest Portfolio
Mix" button for FTMO (app.py disables that button while this is
enabled, and vice versa) — direct user request, so only one of the two
trigger paths is ever armed at a time.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from ai.copilot_execution import EXECUTION_LOCK_PATH, EXECUTION_LOCK_STALE_AFTER_SECONDS, run_copilot_execution_check
from ai.mega_analysis import is_due, next_run_utc, read_mega_analysis_enabled, read_state, run_scheduled_mega_analysis
from job_lock import acquire_lock, release_lock
from utils import run_with_timeout

_LOG_PATH = Path(__file__).resolve().parent / "mega_analysis_log.txt"
_LOCK_PATH = Path(__file__).resolve().parent / "mega_analysis.lock"
# Fallback only, used when a real PID-liveness check can't be performed —
# comfortably past run_mega_analysis's own _RUN_TIMEOUT_SECONDS (55 min)
# ceiling, so a lock still fresher than this genuinely could be a real,
# still-running attempt.
_LOCK_STALE_AFTER_SECONDS = 90 * 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(_LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("mega_analysis_job")

_INLINE_EXECUTION_TIMEOUT_SENTINEL = object()


def _run_inline_execution_check_if_successful(state_before: dict, now_utc: datetime) -> None:
    """Fires one immediate Copilot execution-check pass right after a
    genuinely successful mega-analysis run, so a "feasible right now"
    trade doesn't sit idle waiting for the next standalone poll boundary.
    Never lets a hang or exception here propagate — this is a bonus,
    lower-latency path alongside the standalone poll, not a
    replacement for it, so a failure here must not affect this job's own
    lock/exit behavior."""
    state_after = read_state()
    ran_today_successfully = (
        state_after.get("last_status") == "success"
        and state_after.get("last_run_date_utc") == now_utc.date().isoformat()
        # Only fire on a run that actually happened just now — never on
        # an already-stale prior success this same poll didn't produce
        # (relevant if is_due's own dedup ever changes; belt-and-braces).
        and state_after.get("last_attempt_utc") != state_before.get("last_attempt_utc")
    )
    if not ran_today_successfully:
        return

    if not acquire_lock(EXECUTION_LOCK_PATH, EXECUTION_LOCK_STALE_AFTER_SECONDS):
        logger.info(
            "Skipping the inline Copilot execution-check pass — the standalone "
            "poll already holds the execution lock right now; it will pick up any "
            "immediately-feasible trade on its own next run instead."
        )
        return

    logger.info("Mega analysis succeeded — running one inline Copilot execution-check pass.")
    try:
        result = run_with_timeout(
            run_copilot_execution_check,
            config.COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS,
            default=_INLINE_EXECUTION_TIMEOUT_SENTINEL,
            catch_exceptions=False,
        )
        if result is _INLINE_EXECUTION_TIMEOUT_SENTINEL:
            logger.error(
                "Inline Copilot execution-check timed out after %.0f minutes.",
                config.COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS / 60,
            )
    except Exception:
        logger.exception("Inline Copilot execution-check raised an exception.")
    finally:
        release_lock(EXECUTION_LOCK_PATH)


def main() -> None:
    now_utc = datetime.now(timezone.utc)
    state = read_state()

    if not read_mega_analysis_enabled():
        logger.info("Automated mega analysis is disabled via the app's toggle — exiting.")
        return

    if not is_due(now_utc, state=state):
        logger.info(
            "Not due (now=%s UTC, last_run_date_utc=%s, next=%s UTC) — exiting.",
            now_utc.strftime("%Y-%m-%d %H:%M:%S"),
            state.get("last_run_date_utc", "never"),
            next_run_utc(now_utc, state=state).strftime("%Y-%m-%d %H:%M:%S"),
        )
        return

    if not acquire_lock(_LOCK_PATH, _LOCK_STALE_AFTER_SECONDS):
        return
    try:
        logger.info("Due — starting the FTMO mega market analysis (now=%s UTC).", now_utc)
        run_scheduled_mega_analysis()
        logger.info("Run finished; see this log and %s for the outcome.", read_state())
        _run_inline_execution_check_if_successful(state, now_utc)
    finally:
        release_lock(_LOCK_PATH)


if __name__ == "__main__":
    main()
