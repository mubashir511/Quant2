import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE
from ai.ftmo_suggest import analyze_ftmo_assets, build_ftmo_summary, fetch_ftmo_status, suggest_ftmo_portfolio
from data.mt5_source import connect, get_account_summary, get_market_watch, get_open_positions
from utils import run_with_timeout

logger = logging.getLogger(__name__)

# A hard ceiling on the WHOLE unattended run, not just its individual
# network calls — confirmed live as genuinely necessary: a hang inside
# yfinance's own cookie/crumb negotiation (see utils.run_with_timeout's
# own docstring for the incident) silently killed a full day's scheduled
# run with no exception ever raised, so run_scheduled_mega_analysis's own
# try/except never even saw it — a hang is not a crash, and its own
# docstring's "never raises" claim only ever covered the latter. Set
# comfortably above the pipeline's own known real duration (Claude Sonnet
# + the full audit-model pool + Copilot has taken up to ~35 minutes per
# this project's own history) but under the Windows Scheduled Task's own
# 1-hour ExecutionTimeLimit, so THIS timeout fires first and gets a
# chance to log a real error and write a real failure state before
# Windows would otherwise just silently terminate the process.
_RUN_TIMEOUT_SECONDS = 55 * 60


def _trigger_time_utc(for_date: date) -> datetime:
    return datetime(
        for_date.year,
        for_date.month,
        for_date.day,
        config.MEGA_ANALYSIS_TRIGGER_HOUR_UTC,
        config.MEGA_ANALYSIS_TRIGGER_MINUTE_UTC,
        tzinfo=timezone.utc,
    )


def read_state() -> dict:
    """Best-effort: the marker mega_analysis_job.py writes after every
    attempt. `{}` (not an exception) on anything missing/unreadable —
    both callers (the countdown display, is_due's own dedup check) treat
    an absent/corrupt state file the same as "never run," which is the
    correct, safe default (worst case: one extra run attempt, never a
    silently skipped day)."""
    path = Path(config.MEGA_ANALYSIS_STATE_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(status: str, detail: str = "") -> None:
    prior = read_state()
    now = datetime.now(timezone.utc)
    payload = {
        "last_attempt_utc": now.isoformat(),
        "last_status": status,
        "last_detail": detail,
        "last_run_date_utc": (
            now.date().isoformat() if status == "success" else prior.get("last_run_date_utc")
        ),
    }
    Path(config.MEGA_ANALYSIS_STATE_FILE).write_text(json.dumps(payload, indent=2))


def next_run_utc(now_utc: datetime, state: dict | None = None) -> datetime:
    """The next scheduled mega-analysis instant, in UTC. Today's trigger
    if it hasn't happened yet, we haven't already recorded a successful
    run for today, and we're still before it; tomorrow's trigger
    otherwise (covers: today's run already happened, today's window has
    already passed with no run, or it's simply not today's trigger day
    yet in a way that still resolves to "today")."""
    state = state if state is not None else read_state()
    today = now_utc.date()
    ran_today = state.get("last_run_date_utc") == today.isoformat()
    today_trigger = _trigger_time_utc(today)
    if not ran_today and now_utc < today_trigger:
        return today_trigger
    return _trigger_time_utc(today + timedelta(days=1))


def is_due(now_utc: datetime, state: dict | None = None) -> bool:
    """True only inside the grace window right after today's trigger
    instant, and only if today hasn't already produced a successful run.
    A poll landing outside this window (the PC was off/asleep through
    the whole window) means today's slot is treated as missed and
    skipped — by explicit design, not run late — since the next poll to
    find `now_utc` back in-window won't be until tomorrow's trigger."""
    state = state if state is not None else read_state()
    today = now_utc.date()
    if state.get("last_run_date_utc") == today.isoformat():
        return False
    today_trigger = _trigger_time_utc(today)
    grace_end = today_trigger + timedelta(minutes=config.MEGA_ANALYSIS_GRACE_MINUTES)
    return today_trigger <= now_utc <= grace_end


def run_mega_analysis(
    on_stage: Callable[[str], None] | None = None,
    on_audit_progress: Callable[[str], None] | None = None,
) -> str:
    """Runs the exact same FTMO Portfolio Suggestion pipeline the manual
    "Suggest Portfolio Mix" button does for FTMO (app.py's FTMO branch) —
    Claude Sonnet draft + the full AUDIT_MODELS pool + Copilot — always
    against the FTMO account explicitly (never whatever account happens
    to be connected from a browser session elsewhere), fetching fresh
    data and saving the result via save_record=True so it surfaces
    through the app's existing cold-start loader on the next page view,
    with zero coupling to Streamlit session_state (this can and does run
    from a plain, non-Streamlit process).

    Returns the raw suggestion text. A CLI-layer failure (missing CLI /
    a run that failed outright) comes back as that same error string
    rather than raising — callers (mega_analysis_job.py) check for it via
    CLI_MISSING_MESSAGE/CLI_FAILED_PREFIX, the same contract app.py's
    button handler already uses, and must NOT record that as a
    successful run. A connection/data failure raises MT5ConnectionError/
    RuntimeError instead, same "never silently swallow a real failure"
    convention as everywhere else in this project."""

    def _notify(message: str) -> None:
        logger.info(message)
        if on_stage:
            on_stage(message)

    _notify("Connecting to the FTMO MT5 account...")
    connect(login=config.FTMO_MT5_LOGIN, password=config.FTMO_MT5_PASSWORD, server=config.FTMO_MT5_SERVER)

    _notify("Fetching FTMO account and market data...")
    account = get_account_summary()
    assets = get_market_watch()
    if not assets:
        raise RuntimeError(
            "No instruments are visible in this FTMO account's MT5 Market Watch."
        )
    positions = get_open_positions()
    status = fetch_ftmo_status(account)

    _notify(f"Analyzing {len(assets)} instruments...")
    analyses = analyze_ftmo_assets(assets, on_progress=_notify)
    summary = build_ftmo_summary(account, assets, status, positions=positions, analyses=analyses)

    _notify("Running Claude Sonnet + the full audit-model pool + Copilot...")
    audit_notify = on_audit_progress if on_audit_progress is not None else (lambda text: logger.info(text))
    return suggest_ftmo_portfolio(
        summary,
        model="sonnet",
        save_record=True,
        on_stage=_notify,
        on_audit_progress=audit_notify,
    )


_TIMEOUT_SENTINEL = object()


def run_scheduled_mega_analysis() -> None:
    """The unattended entry point (called from mega_analysis_job.py):
    runs the pipeline and records the outcome to MEGA_ANALYSIS_STATE_FILE
    — success only on a real, complete suggestion; a CLI-layer failure,
    a hang past _RUN_TIMEOUT_SECONDS, or a real exception all record a
    non-"success" status (so the marker never falsely claims today is
    done) without being treated as anything other than a real failure.
    Never hangs the caller past _RUN_TIMEOUT_SECONDS and never raises —
    this is the unattended top of the call stack, and this project has
    already been burned once by trusting "a crash here would just mean a
    job that silently stopped" — a genuine HANG (no exception ever
    raised, just blocked forever) slips straight past a plain try/except
    and needs its own explicit guard, which is exactly what run_with_
    timeout provides here."""
    try:
        suggestion = run_with_timeout(
            run_mega_analysis, _RUN_TIMEOUT_SECONDS, default=_TIMEOUT_SENTINEL, catch_exceptions=False
        )
    except Exception as e:
        logger.exception("Mega analysis run failed")
        _write_state("error", str(e))
        return

    if suggestion is _TIMEOUT_SENTINEL:
        logger.error("Mega analysis run timed out after %.0f minutes", _RUN_TIMEOUT_SECONDS / 60)
        _write_state("timeout", f"No result after {_RUN_TIMEOUT_SECONDS / 60:.0f} minutes")
        return

    if suggestion == CLI_MISSING_MESSAGE or suggestion.startswith(CLI_FAILED_PREFIX):
        logger.error("Mega analysis run failed at the CLI layer: %s", suggestion)
        _write_state("cli_failed", suggestion[:500])
        return

    logger.info("Mega analysis run succeeded.")
    _write_state("success")
