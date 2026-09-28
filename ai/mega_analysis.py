import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE
from ai.ftmo_suggest import (
    analyze_ftmo_assets,
    build_ftmo_summary,
    fetch_ftmo_status,
    read_latest_suggestion,
    suggest_ftmo_portfolio,
)
from data.mt5_source import (
    connect,
    get_account_summary,
    get_last_tick_epochs,
    get_market_watch,
    get_open_positions,
    get_pending_orders,
    is_symbol_tradable_now,
)
from utils import run_with_timeout

# Re-exported so existing callers (ai/clerk_execution.py, app.py) that
# import read_latest_suggestion from this module keep working unchanged
# — the function itself moved to ai/ftmo_suggest.py (see that module's
# own docstring for why: it's written by ANY successful
# suggest_ftmo_portfolio() call now, not only this module's own
# run_mega_analysis, so it belongs next to that function instead).
__all__ = ["read_latest_suggestion"]

logger = logging.getLogger(__name__)

# A hard ceiling on the WHOLE unattended run, not just its individual
# network calls — confirmed live as genuinely necessary: a hang inside
# yfinance's own cookie/crumb negotiation (see utils.run_with_timeout's
# own docstring for the incident) silently killed a full day's scheduled
# run with no exception ever raised, so run_scheduled_mega_analysis's own
# try/except never even saw it — a hang is not a crash, and its own
# docstring's "never raises" claim only ever covered the latter. Set
# comfortably above the pipeline's own known real duration (Claude Sonnet
# + the full audit-model pool has taken up to ~35 minutes per this
# project's own history, measured back when Copilot still ran as an
# extra 11th audit voice here too — see run_mega_analysis's own
# docstring for why it no longer does; kept as-is since it's still a
# safe, conservative ceiling) but under the Windows Scheduled Task's own
# 1-hour ExecutionTimeLimit, so THIS timeout fires first and gets a
# chance to log a real error and write a real failure state before
# Windows would otherwise just silently terminate the process.
_RUN_TIMEOUT_SECONDS = 55 * 60


def read_mega_analysis_trigger() -> tuple[int, int]:
    """(hour_utc, minute_utc) for the daily trigger — user-overridable via
    app.py's time picker (direct user request 2026-08-23). Falls back to
    config.MEGA_ANALYSIS_TRIGGER_HOUR_UTC/MINUTE_UTC (the env-var-
    configured default) on a missing file, a corrupt one, or one holding
    an out-of-range hour/minute — same "never let bad state silently
    break the daily run" posture as every other read_* helper here."""
    path = Path(config.MEGA_ANALYSIS_TRIGGER_FILE)
    default = (config.MEGA_ANALYSIS_TRIGGER_HOUR_UTC, config.MEGA_ANALYSIS_TRIGGER_MINUTE_UTC)
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text())
        hour = int(data["hour_utc"])
        minute = int(data["minute_utc"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return default
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return default
    return hour, minute


def set_mega_analysis_trigger(hour_utc: int, minute_utc: int) -> None:
    """Persists the user's chosen trigger time so mega_analysis_job.py —
    a totally separate OS process — sees the same choice app.py's time
    picker just recorded."""
    try:
        Path(config.MEGA_ANALYSIS_TRIGGER_FILE).write_text(
            json.dumps({"hour_utc": hour_utc, "minute_utc": minute_utc})
        )
    except OSError as e:
        logger.warning("Could not write trigger-time file %s: %s", config.MEGA_ANALYSIS_TRIGGER_FILE, e)


def _trigger_time_utc(for_date: date) -> datetime:
    hour, minute = read_mega_analysis_trigger()
    return datetime(for_date.year, for_date.month, for_date.day, hour, minute, tzinfo=timezone.utc)


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


def read_progress() -> dict:
    """Best-effort read of the unattended run's own LIVE, in-progress
    status (see config.MEGA_ANALYSIS_PROGRESS_FILE's own comment for why
    this is a separate file from read_state's completed-attempt marker)
    — `{}` on anything missing/unreadable, same safe-default convention
    as read_state, since every consumer already treats an empty dict as
    "nothing live to show" correctly.

    Shape as of 2026-08-27: `{"steps": [str, ...], "current_activity":
    str | None, "updated_utc": str}` — two different kinds of status,
    kept separate because they behave completely differently:

    `steps` is a growing list of genuinely discrete, one-time
    milestones (connecting, analyzing instrument N/18, starting the
    audit pool...) — bounded to a few dozen entries per run, so
    appending is fine (this is the 2026-08-25 fix: the OLD single-
    `"message"` shape made the display flicker, one line replacing the
    last, instead of reading as a steady step-by-step log).

    `current_activity` is a single, frequently-OVERWRITTEN slot for
    whatever's happening RIGHT NOW that updates far more often than
    once per milestone — namely the audit-model pool's own per-second,
    per-model retry/countdown status. A real run confirmed live
    2026-08-27: appending that instead of overwriting it ballooned the
    old single `messages` list to 246+ near-duplicate entries within
    minutes (the manual "Suggest Portfolio Mix" button never had this
    problem — it already renders this exact same on_audit_progress
    callback into a single `st.empty()` placeholder it overwrites, not
    an accumulating log; this mirrors that for the file-backed
    unattended path). `_reset_progress` clears both at the start of
    each new run so a stale prior run's status never bleeds into a
    fresh one's display."""
    path = Path(config.MEGA_ANALYSIS_PROGRESS_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def mega_session_is_live(progress: dict, state: dict) -> bool:
    """True iff `progress` (from read_progress()) reflects a mega
    session that's still genuinely in flight, not a stale leftover from
    a run that already finished or crashed. Shared by app.py's own
    live-progress display and ai.clerk_execution's pre-execution
    check (previously two independent copies of this exact heuristic —
    consolidated here 2026-08-27 after fixing a real bug in both at
    once: see below).

    Two signals, both required:

    1. `progress` must be NEWER than the last COMPLETED attempt
       recorded in `state` — this is what correctly detects "still
       running" even during a long silent stretch with zero new writes
       (e.g. Claude's own multi-minute web-research call), since
       `state` only advances once the run actually finishes.
    2. A generous absolute age ceiling, so a genuinely crashed/orphaned
       process that never got to write a real completion state at all
       doesn't read as "live" forever. Set to _RUN_TIMEOUT_SECONDS —
       the same hard ceiling the scheduled run itself is bounded by, so
       this can never time out a run that's still legitimately within
       its own allowed budget.

    Confirmed too short before at a flat 300 seconds (5 minutes): a
    real run's single "Claude is researching..." step and its
    audit-pool phase can each individually run longer than that with
    no new progress write, which wrongly hid the whole live-progress
    UI mid-run (found live 2026-08-27) — and, in ai.clerk_execution's
    copy, could in principle have let the execution-check job act
    mid-run instead of correctly waiting."""
    progress_ts = progress.get("updated_utc")
    if not progress_ts:
        return False
    try:
        progress_dt = datetime.fromisoformat(progress_ts)
    except ValueError:
        return False
    age_seconds = (datetime.now(timezone.utc) - progress_dt).total_seconds()
    last_attempt = state.get("last_attempt_utc")
    newer_than_last_attempt = last_attempt is None or progress_ts > last_attempt
    return 0 <= age_seconds < _RUN_TIMEOUT_SECONDS and newer_than_last_attempt


def _reset_progress() -> None:
    """Clears the live-progress log at the start of a new run — called
    once, before the first _notify, so app.py's live-progress display
    never shows a leftover step (or leftover current_activity) from the
    PRIOR run alongside the new one's."""
    payload = {"steps": [], "current_activity": None, "updated_utc": datetime.now(timezone.utc).isoformat()}
    try:
        Path(config.MEGA_ANALYSIS_PROGRESS_FILE).write_text(json.dumps(payload))
    except OSError as e:
        logger.warning("Could not reset progress file %s: %s", config.MEGA_ANALYSIS_PROGRESS_FILE, e)


def _write_step(message: str) -> None:
    """Best-effort — a failure to write this purely-cosmetic live-status
    file must never break the real analysis it's reporting on, so unlike
    _write_state (whose job IS to reliably record the outcome), this
    swallows its own I/O errors rather than letting them propagate.

    Appends to the run's own `steps` log (see read_progress's own
    docstring for the shape/rationale) — for genuinely discrete,
    one-time milestones only; a status that repeats/updates many times
    a minute belongs in _write_current_activity instead, not here.

    Also CLEARS `current_activity` back to None. A discrete milestone
    means whatever noisy sub-status was live a moment ago (the
    instrument-scan's "N/18 — SYMBOL", the audit pool's per-model
    countdown) has now genuinely concluded — without this, its last
    value would linger and keep rendering as "live" even after the run
    has moved on to a completely different phase (a real bug caught
    while wiring up the instrument-scan's own current_activity use,
    2026-08-27, before it ever shipped).

    Read-modify-write against whatever's on disk right now (falling
    back to an empty list if the file is missing/corrupt) rather than
    keeping the list in memory, since this, _write_current_activity,
    and _reset_progress are the only writers and nothing else needs to
    stay in sync with them across the single-process pipeline that
    calls this."""
    prior = read_progress()
    steps = prior.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    steps.append(message)
    payload = {
        "steps": steps,
        "current_activity": None,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        Path(config.MEGA_ANALYSIS_PROGRESS_FILE).write_text(json.dumps(payload))
    except OSError as e:
        logger.warning("Could not write progress file %s: %s", config.MEGA_ANALYSIS_PROGRESS_FILE, e)


def _write_current_activity(text: str) -> None:
    """Best-effort, same swallow-I/O-errors convention as _write_step.

    OVERWRITES the single `current_activity` slot rather than appending
    — for high-frequency, self-superseding status only (right now: the
    audit-model pool's own per-second, per-model retry/countdown text).
    Appending this instead of overwriting it is exactly what ballooned
    a real run's progress log to 246+ near-duplicate entries within
    minutes (confirmed live 2026-08-27) — see read_progress's docstring."""
    prior = read_progress()
    payload = {
        "steps": prior.get("steps", []),
        "current_activity": text,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        Path(config.MEGA_ANALYSIS_PROGRESS_FILE).write_text(json.dumps(payload))
    except OSError as e:
        logger.warning("Could not write progress file %s: %s", config.MEGA_ANALYSIS_PROGRESS_FILE, e)


def read_mega_analysis_enabled() -> bool:
    """Whether the unattended scheduled mega analysis is currently
    enabled — defaults to True (enabled) on a missing/corrupt file, same
    "worst case is one extra harmless check" philosophy as read_state's
    own safe default, just inverted: here the safe default is to keep
    running rather than to skip, since a silently-disabled daily run is
    the failure mode direct user request 2026-08-23 was designed to
    avoid ("keep toggle enabled by default")."""
    path = Path(config.MEGA_ANALYSIS_ENABLED_FILE)
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return bool(data.get("enabled", True))


def set_mega_analysis_enabled(enabled: bool) -> None:
    """Persists the toggle so mega_analysis_job.py — a totally separate
    OS process — sees the same choice app.py's toggle just recorded."""
    try:
        Path(config.MEGA_ANALYSIS_ENABLED_FILE).write_text(json.dumps({"enabled": enabled}))
    except OSError as e:
        logger.warning("Could not write enabled-flag file %s: %s", config.MEGA_ANALYSIS_ENABLED_FILE, e)


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


def select_tradable_assets(
    assets: list,
    keep_symbols: set[str],
    now_utc: datetime,
    tick_epochs: dict[str, int | None],
    calendar_fn: Callable[[str, datetime], bool] | None = None,
    max_tick_age_minutes: float | None = None,
) -> tuple[list, list[tuple[str, str]]]:
    """(tradable assets, [(symbol, why skipped)]) - the FIRST thing a Mega session does, so no data fetch, chart read,
    backtest or model token is spent on an instrument whose market is closed right now.

    An instrument is skipped when (a) the weekly market calendar says it is closed, (b) it has no live tick at all, or (c) its
    last tick is more than `max_tick_age_minutes` older than the FRESHEST tick in the pool (a session that ended for the day: US
    stocks after the close, an exchange lunch break, a paused feed). Comparing against the pool's freshest tick instead of the
    wall clock keeps this independent of the broker's UTC offset. A symbol with an open position or a resting order is always
    kept (the session must still manage it). Never raises: an unreadable tick falls back to the calendar alone.
    """
    calendar_fn = calendar_fn or is_symbol_tradable_now  # resolved at call time (so tests/patches of the module name apply)
    max_age = (config.MEGA_TRADABLE_MAX_TICK_AGE_MINUTES if max_tick_age_minutes is None else max_tick_age_minutes) * 60
    known = [t for t in tick_epochs.values() if t]
    freshest = max(known) if known else None
    tradable, skipped = [], []
    for asset in assets:
        symbol = asset.symbol
        if symbol in keep_symbols:
            tradable.append(asset)
            continue
        try:
            calendar_open = calendar_fn(symbol, now_utc)
        except Exception:  # noqa: BLE001
            calendar_open = True
        if not calendar_open:
            skipped.append((symbol, "market closed (weekly calendar)"))
            continue
        tick = tick_epochs.get(symbol)
        if tick is None and freshest is not None:
            skipped.append((symbol, "no live tick"))
            continue
        if tick is not None and freshest is not None and freshest - tick > max_age:
            skipped.append((symbol, f"no fresh tick (last tick {(freshest - tick) / 60:.0f} min behind the freshest market)"))
            continue
        tradable.append(asset)
    return tradable, skipped


def run_mega_analysis(
    on_stage: Callable[[str], None] | None = None,
    on_audit_progress: Callable[[str], None] | None = None,
    on_analyses: Callable[[list], None] | None = None,
    model: str | None = None,
) -> str:
    """Runs the exact same FTMO Portfolio Suggestion pipeline the manual
    "Suggest Portfolio Mix" button does for FTMO (app.py's FTMO branch) —
    Claude Sonnet draft + the full AUDIT_MODELS pool — always against the
    FTMO account explicitly (never whatever account happens to be
    connected from a browser session elsewhere), fetching fresh data and
    saving the result via save_record=True so it surfaces through the
    app's existing cold-start loader on the next page view, with zero
    coupling to Streamlit session_state (this can and does run from a
    plain, non-Streamlit process).

    Passes include_copilot=False explicitly below (matching suggest_
    ftmo_portfolio's own default now) — Copilot has a separate, dedicated
    role in the audit pool it must not also spend its own request budget
    on here; that role is entirely unrelated to the Execution Clerk (the
    short-interval clerk/executioner, see ai/clerk_execution.py, which
    stopped using Copilot at all as of 2026-08-30 — see that module's
    own docstring). This is no longer scheduled-run-specific: an earlier
    version of this exclusion only covered this path, but the manual
    "Suggest Portfolio Mix" button (app.py) showed Copilot still
    auditing FTMO suggestions there too, so suggest_ftmo_portfolio's own
    default flipped to exclude it everywhere for FTMO — this explicit
    pass is now redundant with that default, kept for clarity. PMEX and
    PSX are untouched — only FTMO's relationship with Copilot's audit
    role changed.

    Returns the raw suggestion text. A CLI-layer failure (missing CLI /
    a run that failed outright) comes back as that same error string
    rather than raising — callers (mega_analysis_job.py) check for it via
    CLI_MISSING_MESSAGE/CLI_FAILED_PREFIX, the same contract app.py's
    button handler already uses, and must NOT record that as a
    successful run. A connection/data failure raises MT5ConnectionError/
    RuntimeError instead, same "never silently swallow a real failure"
    convention as everywhere else in this project.

    suggest_ftmo_portfolio() itself writes the derived
    MEGA_ANALYSIS_LATEST_SUGGESTION_FILE artifact on any genuine
    success — the sole source of truth ai.clerk_execution's clerk
    reads — unconditionally, regardless of caller; this function doesn't
    need to (and no longer does) trigger that separately.

    Every stage/audit-progress message is ALSO broadcast to
    read_progress()'s own file (see _write_step/_write_current_activity)
    regardless of whether a caller supplied on_stage/on_audit_progress —
    direct user request 2026-08-22: the unattended run should show the
    same live step-by-step status the manual "Suggest Portfolio Mix"
    button already shows, the only real difference being WHAT triggers
    it, not whether it's visible while running. app.py's live-progress
    display is what actually renders this; this function's own job is
    just to keep the file honestly up to date. Calls _reset_progress()
    first, before anything else, so this run's own step log starts
    empty rather than carrying over whatever the previous run last left
    behind.

    `on_analyses`, if given, is called once with the real per-instrument
    analyses list right after it's computed — added so the manual
    "Suggest Portfolio Mix" button (app.py) can call this SAME function
    instead of re-implementing the whole pipeline inline, which is what
    silently caused it to never write to this function's own progress
    file at all: a real, reported bug where the manual run showed no
    live status in the shared Mega Session section, and didn't pause the
    Clerk's own periodic check the way a scheduled run correctly does,
    since mega_session_is_live() has nothing to go on without these
    writes. This callback lets the button still keep the `analyses`
    value it separately needs for its own session_state afterward.

    `model`, if given, overrides config.MEGA_ANALYSIS_MODEL for this one
    call — the manual button lets a user pick a model (e.g. a cheap/fast
    "haiku" test run) without touching that config value; None (the
    default) preserves the exact original scheduled-run behavior."""
    _reset_progress()

    def _notify(message: str) -> None:
        logger.info(message)
        _write_step(message)
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
    pending_orders = get_pending_orders()
    status = fetch_ftmo_status(account)

    # Before ANY per-instrument work: which of the Market Watch symbols can actually trade right now? Everything else is dropped
    # here, saving the MT5 fetches, chart/backtest reads and (mostly) the model tokens. Held / resting-order symbols always stay.
    closed_note = ""
    if config.MEGA_TRADABLE_FILTER_ENABLED:
        _notify(f"Checking which of the {len(assets)} instruments are tradable right now...")
        keep = {p.symbol for p in positions} | {o.symbol for o in pending_orders}
        try:
            tick_epochs = get_last_tick_epochs([a.symbol for a in assets])
        except Exception:  # noqa: BLE001 - fall back to the calendar alone rather than block the session
            logger.warning("Could not read last-tick times; filtering by the market calendar only.", exc_info=True)
            tick_epochs = {}
        tradable, skipped = select_tradable_assets(assets, keep, datetime.now(timezone.utc), tick_epochs)
        if not tradable:
            raise RuntimeError(
                "No instrument is tradable right now (every market is closed"
                + (": " + "; ".join(f"{s} - {why}" for s, why in skipped) if skipped else "")
                + "). Nothing was analysed and no model tokens were spent; try again when a market is open."
            )
        if skipped:
            _notify(
                f"{len(tradable)} of {len(assets)} instruments are tradable now; skipping {len(skipped)} closed: "
                + ", ".join(s for s, _ in skipped)
            )
            closed_note = (
                "MARKETS CLOSED RIGHT NOW - NOT ANALYSED THIS RUN (no data is printed for them; do not propose new entries "
                "for them): " + "; ".join(f"{s} ({why})" for s, why in skipped) + ".\n\n"
            )
            assets = tradable
        else:
            _notify(f"All {len(assets)} instruments are tradable now.")

    def _notify_instrument_progress(message: str) -> None:
        # Deliberately writes to current_activity, NOT a new step — one
        # instrument finishing isn't independently worth a permanent
        # line the way "Connecting..." or "Running Claude..." are; it's
        # the same self-superseding-status category as the audit pool's
        # own per-model countdown (see _notify_audit below), just at a
        # calmer cadence. Direct user request 2026-08-27, after seeing a
        # mock-up render each instrument as its own checkmarked line:
        # "just use one message only dynamically update it 18 times...
        # and also update the symbol names as well simultaneously" —
        # one line updates in place through 1/18, 2/18, ... 18/18 with
        # the current symbol named, instead of the log growing by one
        # entry per instrument.
        logger.info(message)
        _write_current_activity(message)
        if on_stage:
            on_stage(message)

    _notify(f"Analyzing {len(assets)} instruments...")
    analyses = analyze_ftmo_assets(assets, on_progress=_notify_instrument_progress)
    _notify(f"Analyzed all {len(assets)} instruments.")
    if on_analyses:
        on_analyses(analyses)
    summary = closed_note + build_ftmo_summary(
        account, assets, status, positions=positions, analyses=analyses, pending_orders=pending_orders
    )

    # config.MEGA_ANALYSIS_MODEL defaults to "sonnet" (the real, intended
    # production model) — overridable via the MEGA_ANALYSIS_MODEL env var,
    # or per-call via the `model` parameter above (the manual button's own
    # model picker), purely for a cheap/fast manual test run (e.g. "haiku")
    # without touching this file, so there's a single obvious place to
    # remember to unset the override afterward rather than a hardcoded
    # value here that's easy to forget mid-test.
    resolved_model = model or config.MEGA_ANALYSIS_MODEL
    _notify(f"Running Claude {resolved_model} + the full audit-model pool...")

    def _notify_audit(text: str) -> None:
        logger.info(text)
        _write_current_activity(text)
        if on_audit_progress:
            on_audit_progress(text)

    return suggest_ftmo_portfolio(
        summary,
        model=resolved_model,
        save_record=True,
        on_stage=_notify,
        on_audit_progress=_notify_audit,
        include_copilot=False,
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
