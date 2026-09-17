import logging
import secrets
import threading
import time
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE
from ai.ftmo_suggest import (
    LONG_TERM_ALIGNMENT_SHORT_MESSAGES,
    FtmoAssetAnalysis,
    analyze_ftmo_asset_live,
    classify_long_term_alignment,
    fetch_ftmo_status,
    format_ftmo_trade_cost,
)
from ai.clerk_execution import (
    EXECUTION_LOCK_PATH,
    EXECUTION_LOCK_STALE_AFTER_SECONDS,
    execution_check_is_live,
    read_clerk_execution_enabled,
    read_clerk_execution_interval_minutes,
    read_execution_progress,
    read_execution_state,
    read_settlement,
    read_tactical_defense_enabled,
    run_clerk_execution_check,
    set_clerk_execution_enabled,
    set_clerk_execution_interval_minutes,
    set_tactical_defense_enabled,
)
from job_lock import acquire_lock, release_lock
from utils import run_with_timeout
from ai.researcher import (
    RESEARCHER_LOCK_PATH,
    RESEARCHER_LOCK_STALE_AFTER_SECONDS,
    is_researcher_due,
    latest_research_report,
    next_researcher_check_utc,
    parse_researcher_sentiment,
    read_researcher_enabled,
    read_researcher_state,
    run_researcher_check,
)
from ai.mega_analysis import (
    _write_state as _write_mega_analysis_state,
    is_due as mega_analysis_is_due,
    mega_session_is_live,
    next_run_utc,
    read_latest_suggestion,
    read_mega_analysis_enabled,
    read_mega_analysis_trigger,
    read_progress,
    read_state,
    run_mega_analysis,
    run_scheduled_mega_analysis,
    set_mega_analysis_enabled,
    set_mega_analysis_trigger,
)
from ai.portfolio_suggest import (
    AUDIT_MODELS,
    AllocationEntry,
    AssetAnalysis,
    analyze_assets,
    build_portfolio_summary,
    parse_final_allocation,
    strip_allocation_block,
    strip_pending_setups_block,
    strip_leading_process_narration,
    suggest_portfolio,
)
from ai.psx_suggest import (
    analyze_psx_assets,
    build_psx_summary,
    compute_sector_allocation,
    compute_sector_performance,
    suggest_psx_portfolio,
)
from analysis.setup_classifier import classify_setups
from data.mt5_execution import (
    OrderResult,
    cancel_pending_order,
    close_position,
    modify_position_sltp,
    open_position,
)
from data.mt5_source import (
    MT5ConnectionError,
    get_contract_spec,
    get_server_time_offset,
    group_closed_trades,
)
from data.psx_source import PSX_INDICES, PSXAsset, PSXConnectionError, get_psx_market_watch
from risk.apply_suggestion import (
    check_execution_safety_gates,
    compute_aggregate_heat_pct,
    compute_rebalance_plan,
)
from risk.ftmo_rules import DEFAULT_HEADROOM_FRACTION, FtmoStatus, would_breach_daily_loss_headroom

logger = logging.getLogger(__name__)

# A fixed-order categorical palette (validated colorblind-safe adjacent
# pairs, per this project's dataviz conventions) — used everywhere a
# chart distinguishes categories (allocation slices, sectors), instead of
# Plotly's default rainbow cycling, so the same few hues mean the same
# thing across every chart on the page.
_CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
# The same blue/red pair, used instead for its OTHER job here: a
# diverging (positive/negative) polarity encoding, not category identity
# — gains vs. losses, not "sector A vs. sector B".
_GAIN_COLOR = "#2a78d6"
_LOSS_COLOR = "#e34948"


def _render_pie(names: list[str], values: list[float], title: str) -> None:
    # Slice text is percent-only (not the full name) — the always-present
    # horizontal legend below already carries identity, and cramming both
    # onto every wedge is exactly what caused labels to wrap and collide
    # with each other/the title on a narrow pool of many small sectors.
    # "inside" + radial orientation keeps text from spilling past the
    # chart's own bounding box into whatever sits next to it.
    fig = px.pie(
        names=names,
        values=values,
        hole=0.4,
        color_discrete_sequence=_CATEGORICAL_COLORS,
    )
    fig.update_traces(
        textinfo="percent",
        textposition="inside",
        insidetextorientation="radial",
        textfont_size=14,
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    )
    fig.update_layout(
        title=title,
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.1, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_ai_report(text: str) -> None:
    """Renders AI-generated markdown text safely. Streamlit auto-renders
    anything between a pair of literal '$' characters as LaTeX (confirmed
    on the installed 1.60) — these reports are full of dollar amounts like
    "$7.21" and "$300", so two or more unescaped '$' in a row silently
    triggered broken math rendering (garbled text, stray '|' fragments)
    instead of showing the dollar figures as plain text."""
    st.markdown(text.replace("$", "\\$"))


def _color_pnl(val: float) -> str:
    """Shared green/red/inherit styling for a P&L column — used by both
    the Open Positions and Trade History tables (previously two identical
    closures defined inline at each call site)."""
    color = "green" if val > 0 else "red" if val < 0 else "inherit"
    return f"color: {color}"


def _sync_pull_from_file(
    widget_key: str,
    last_synced_key: str,
    file_value,
    widget_to_comparable=lambda v: v,
    comparable_to_widget=lambda v: v,
) -> None:
    """Cross-device/cross-tab live-sync guardrail (direct user request
    2026-09-02, after adding remote/mobile access via Tailscale: "add
    live sync between mobile/PC devices with some guard rails"). This is
    the READ-side counterpart of the multi-tab write guard already used
    by every "instant sync to file" control in this app (the mega-
    analysis trigger-time picker, the Clerk enable/tactical-defense
    toggles, the Clerk review-interval picker, the mega-analysis enable
    toggle) — see _render_mega_analysis_countdown's own 2026-08-24
    comment for that fix's full history: comparing a widget's value
    against the FILE's CURRENT value (instead of against what THIS
    session itself last saw) let an idle second tab "helpfully" revert a
    fresh change made from elsewhere within about a second.

    That fix solved the WRITE side (a stale tab never writes its own
    stale value back) but left the READ side unsolved: a control seeded
    from the file only ONCE per session never picks up someone else's
    later change at all, which is exactly the "my phone's change isn't
    showing on the PC" report that motivated this function. The
    guardrail: call this BEFORE the widget is (re-)instantiated each
    rerun. If THIS session already has a not-yet-written local edit
    sitting in the widget (its current value differs from what this
    session last synced), do nothing at all — an active local edit
    always wins and is never silently overwritten, mirroring the write
    guard's own priority exactly. Only once there's no pending local
    edit does a genuine external change (the file has moved since this
    session's own last sync) get pulled into this session's display.
    Idempotent — a no-op once already in sync, so it's safe to call on
    every single tick of a run_every fragment."""
    last_synced = st.session_state[last_synced_key]
    raw_widget_value = st.session_state.get(widget_key)
    if raw_widget_value is not None and widget_to_comparable(raw_widget_value) != last_synced:
        return  # a local edit is pending this session — the write path below owns it
    if file_value != last_synced:
        st.session_state[widget_key] = comparable_to_widget(file_value)
        st.session_state[last_synced_key] = file_value


@st.cache_data(ttl=300)
def _cached_server_offset_seconds() -> float | None:
    """The MT5 broker's real UTC offset, cached for 5 minutes — this
    countdown ticks every second (see _render_mega_analysis_countdown's
    own st.fragment), and re-fetching a live tick from MT5 that often for
    a slowly-changing display value would be a wasted round-trip every
    single second."""
    offset = get_server_time_offset()
    return offset.total_seconds() if offset is not None else None


def _format_last_run(iso_str: str | None) -> str:
    """Human-readable "last run"/"last check" timestamp for the Mega
    Analysis and Execution Clerk status captions — direct user request
    2026-08-30 ("current mechanical formate is not understandable"):
    these previously interpolated the raw stored ISO string (e.g.
    "2026-08-30T12:08:43.291205+00:00 UTC") straight into the caption.
    Renders this machine's own local time (what the user actually reads
    off their clock, same convention as _render_mega_analysis_countdown's
    "Scheduled for (local)" metric above) plus a relative "X ago"
    qualifier, since "how long ago did this run" is usually the more
    useful read at a glance than the exact clock instant. Falls back to
    the raw string on anything unparseable rather than hiding a genuine
    value."""
    if not iso_str:
        return "never"
    try:
        dt = datetime.fromisoformat(iso_str)
    except ValueError:
        return iso_str
    now = datetime.now(timezone.utc) if dt.tzinfo else datetime.now()
    elapsed_seconds = max(0, int((now - dt).total_seconds()))
    if elapsed_seconds < 60:
        relative = "just now"
    elif elapsed_seconds < 3600:
        relative = f"{elapsed_seconds // 60} min ago"
    elif elapsed_seconds < 86400:
        relative = f"{elapsed_seconds // 3600} hr ago"
    else:
        relative = f"{elapsed_seconds // 86400} day(s) ago"
    local_str = dt.astimezone().strftime("%b %d, %I:%M %p").replace(" 0", " ")
    return f"{local_str} ({relative})"


# Repo-root lock path, matching mega_analysis_job.py's own literal
# constant exactly (Path(__file__).resolve().parent — app.py lives in
# the same repo root) — so this in-app auto-trigger and that standalone
# Scheduled Task (if one is ever configured again on this machine) can
# never run the same mega analysis concurrently; whichever gets there
# first holds the real, cross-process, PID-liveness-aware lock.
_MEGA_ANALYSIS_LOCK_PATH = Path(__file__).resolve().parent / "mega_analysis.lock"
_MEGA_ANALYSIS_LOCK_STALE_AFTER_SECONDS = 90 * 60


def _fire_scheduled_job_once(fn: Callable[[], None], state: dict, lock_path: Path, lock_stale_seconds: float) -> None:
    """Fires fn() in a genuine background thread — never blocks the
    calling Streamlit fragment. Direct user report 2026-08-30/31: the
    Clerk section was "fading and staying stale" after its own countdown
    reached zero — root cause was this exact call running SYNCHRONOUSLY
    inside the run_every fragment, blocking that fragment's own script-
    execution thread for however long a real MT5+local-LLM check takes
    (theoretically up to CLERK_EXECUTION_RUN_TIMEOUT_SECONDS, 25 minutes)
    — long enough to desync the browser's own auto-rerun timer for that
    fragment, which then never visibly recovered even after the call
    itself finished successfully. Backgrounding it here means the
    fragment returns almost immediately every tick regardless of how
    long fn() takes; the existing progress-file mechanism (read_execution
    _progress/read_progress, already polled every tick) is what shows
    live status without blocking, exactly like it already does for a
    check triggered by a genuinely separate OS process.

    `state` (a plain dict backed by an @st.cache_resource — process-
    lifetime, shared across every open tab, NOT st.session_state) tracks
    whether THIS process already has a thread in flight for this
    specific job, so a fragment tick landing while the previous run is
    still going doesn't spawn a redundant second thread. `lock_path`/
    `lock_stale_seconds` is the SAME cross-process PID-liveness lock
    (job_lock.py) the standalone *_job.py scripts and the manual buttons
    already use — this can never race a real Scheduled Task run or
    another browser tab's own trigger for the same job."""
    existing_thread = state.get("thread")
    if existing_thread is not None and existing_thread.is_alive():
        return
    if not acquire_lock(lock_path, lock_stale_seconds):
        return

    def _worker() -> None:
        try:
            fn()
        except Exception:
            logger.exception("Background scheduled run failed")
        finally:
            release_lock(lock_path)

    thread = threading.Thread(target=_worker, daemon=True)
    state["thread"] = thread
    thread.start()


@st.cache_resource
def _mega_analysis_timer_state() -> dict:
    """Process-lifetime holder tracking whether an in-app-triggered mega
    analysis background thread is currently in flight — see
    _fire_scheduled_job_once's own docstring for why this is
    st.cache_resource rather than st.session_state. Unlike the Clerk's
    own _clerk_timer_state(), this doesn't need a next_check_utc: mega
    analysis's own due-check (ai.mega_analysis.is_due/next_run_utc) is
    already derived entirely from real persisted state and the clock,
    not from a marker only a live poll would ever advance — see this
    module's own docstring for the real 2026-08-22 incident that shaped
    that design."""
    return {"thread": None}


def _render_mega_analysis_countdown() -> None:
    """FTMO-only: shows the countdown to the next unattended daily "mega
    market analysis" (Claude Sonnet + the full audit pool — Copilot is
    deliberately excluded from FTMO's audit pool entirely now, not just
    here, see ai.ftmo_suggest.suggest_ftmo_portfolio's own docstring for
    why). Originally triggered ONLY by the Quant2MegaAnalysis Windows
    Scheduled Task (see mega_analysis_job.py) — but with no such task
    configured on this machine (direct user choice earlier this session:
    webapp-only, no independent scheduled tasks), that trigger simply
    never fired, and this countdown just silently reset to tomorrow every
    time it reached zero with nothing actually having run (direct user
    report 2026-08-30/31: "the mega session again did not run today just
    counter restart"). This fragment now fires run_scheduled_mega_
    analysis() itself, in the background (see _fire_scheduled_job_once),
    whenever read_mega_analysis_enabled() is on and ai.mega_analysis.
    is_due() says today's window is open — so the countdown genuinely
    drives a real run now, for as long as this app process stays
    running, matching the Windows-Scheduled-Task version's own trigger
    conditions exactly (same is_due/grace-window logic, same lock file)
    rather than a parallel, independently-drifting copy of them.
    Scheduling itself is anchored to plain UTC (see ai.mega_analysis's
    own docstrings for why local-time scheduling would silently drift
    across DST); this widget just converts that UTC instant to both this
    machine's local time (astimezone(), correctly DST-aware) and the
    broker's own server time (from a live tick, best-effort) purely for
    display, per the user's explicit request to stay aware of both
    clocks alongside UTC.

    The run's own LIVE step-by-step progress log (see
    ai.mega_analysis::_write_step/_write_current_activity) is rendered separately, by
    _render_mega_analysis_progress_log — direct user request 2026-08-25:
    it used to live here as a single-line st.info() right under this
    heading, but showing only the LATEST message (replacing the last one
    every second) read as messages randomly appearing and disappearing
    rather than a run steadily progressing. Moved to the bottom of the
    shared controls box, after the "Suggest Portfolio Mix" button, and
    changed to render the full accumulated list every tick instead of
    just the latest entry — see that function's own docstring."""
    now_utc = datetime.now(timezone.utc)
    state = read_state()
    mega_enabled = read_mega_analysis_enabled()

    # Real retry-storm bug found live 2026-08-31: is_due() only clears
    # once a run reaches status="success" (see ai.mega_analysis._write_
    # state's own docstring/comment) — a genuinely persistent failure
    # (that day: the `claude` CLI unreachable from this process) was
    # retried on nearly every single 1-second tick for the rest of the
    # grace window, each attempt doing a real MT5-connect-and-analyze
    # cycle, which is what actually produced "status messages dancing"/
    # a sluggish page, not a hang. is_due()'s own OS-level-poll design
    # assumed several MINUTES between attempts; this cooldown restores
    # that natural spacing for the in-app 1-second-tick trigger without
    # touching is_due()'s own semantics.
    last_attempt_str = state.get("last_attempt_utc")
    cooldown_elapsed = True
    if last_attempt_str:
        try:
            last_attempt_dt = datetime.fromisoformat(last_attempt_str)
            cooldown_elapsed = (
                (now_utc - last_attempt_dt).total_seconds()
                >= config.MEGA_ANALYSIS_RETRY_COOLDOWN_MINUTES * 60
            )
        except ValueError:
            cooldown_elapsed = True

    if mega_enabled and mega_analysis_is_due(now_utc, state=state) and cooldown_elapsed:
        _fire_scheduled_job_once(
            run_scheduled_mega_analysis,
            _mega_analysis_timer_state(),
            _MEGA_ANALYSIS_LOCK_PATH,
            _MEGA_ANALYSIS_LOCK_STALE_AFTER_SECONDS,
        )

    target = next_run_utc(now_utc, state=state)
    remaining_seconds = max(0, int((target - now_utc).total_seconds()))
    hours, rem = divmod(remaining_seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    # No own border here (direct user request 2026-08-23: merge this
    # visually with the manual "Suggest Portfolio Mix" controls into one
    # shared box) — the caller supplies the surrounding
    # st.container(border=True) instead.
    st.subheader(":material/schedule: Next Mega Market Analysis")
    # Direct user request 2026-08-23: significantly shortened (2-3 lines,
    # not the full design rationale — kept in this module's own docstring
    # above instead), and no longer names a specific model or trigger
    # time — both are user-changeable (the model via MEGA_ANALYSIS_MODEL,
    # the time via the picker below), so hardcoding either here would go
    # stale the moment either one is actually changed.
    st.caption(
        f"Unattended daily FTMO run, reviewed by up to {len(AUDIT_MODELS)} audit models "
        "before it's finalized. The Clerk handles execution separately, checking every "
        f"{config.CLERK_EXECUTION_CHECK_INTERVAL_MINUTES} min (see below)."
    )
    if not mega_enabled:
        # Direct user report 2026-09-05: this countdown used to render
        # UNCONDITIONALLY regardless of the toggle just above — so with
        # automated analysis switched off (the ONLY state that makes the
        # manual "Suggest Portfolio Mix" button clickable at all, see its
        # own disabled= condition), the metric still displayed a live,
        # ticking "Time remaining" as if a real automated run were
        # imminent, when that path is actually completely inert while
        # disabled. A manual click was what genuinely ran — a totally
        # separate, on-demand mechanism this countdown has no connection
        # to — but seeing "2m 30s remaining" tick down right next to it
        # made the manual run look like it had fired FROM this countdown,
        # early. Showing a plain "inactive" message instead of a number
        # makes that impossible to misread.
        st.info(
            ":material/pause_circle: Automated daily analysis is currently OFF — this "
            "countdown is inactive and nothing will fire from it. Use \"Suggest Portfolio "
            "Mix\" below to run it manually, or toggle automated analysis back on above."
        )
    else:
        broker_str = "unavailable (no live MT5 tick)"
        if not config.USE_MOCK_DATA:
            try:
                offset_seconds = _cached_server_offset_seconds()
            except MT5ConnectionError:
                offset_seconds = None
            if offset_seconds is not None:
                broker_time = target + pd.Timedelta(seconds=offset_seconds)
                broker_str = broker_time.strftime("%Y-%m-%d %H:%M") + " broker server time"

        col1, col2 = st.columns(2)
        col1.metric("Time remaining", f"{hours}h {minutes:02d}m {seconds:02d}s")
        col2.metric(
            "Scheduled for (local)",
            target.astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        )
        st.caption(f"Same instant in UTC: {target.strftime('%Y-%m-%d %H:%M')} · {broker_str}")

    # Direct user request 2026-08-23: let the user change the daily
    # trigger time from the UI instead of only via the
    # MEGA_ANALYSIS_TRIGGER_HOUR_UTC/MINUTE_UTC env vars. Seeded from the
    # persisted file into session_state only once per session (not every
    # 1-second fragment tick) so this widget's own interaction is never
    # clobbered by re-reading the file; the countdown above catches up on
    # the very next tick since this whole function reruns every second.
    #
    # Real bug found live 2026-08-24 (user: changed this to 14:00, it
    # reverted to 23:16): comparing the widget's value against the
    # FILE's CURRENT value is wrong once more than one browser tab/
    # session can be open at once. A second, stale tab's own widget
    # value never changed from ITS perspective, but this fragment reruns
    # every second regardless of interaction — so on its very next tick,
    # it would see "my widget value differs from the file" (because the
    # OTHER tab just updated the file) and "helpfully" write its own
    # stale value straight back, undoing the real change within a
    # second. Fixed by tracking what THIS session itself last wrote/saw
    # separately from the file's live value — a session only ever writes
    # when its OWN widget value changed relative to what IT last knew,
    # never merely because the file diverged from under it.
    _trigger_hour, _trigger_minute = read_mega_analysis_trigger()
    _file_trigger_pair = (_trigger_hour, _trigger_minute)
    if "mega_analysis_trigger_time" not in st.session_state:
        st.session_state["mega_analysis_trigger_time"] = dt_time(_trigger_hour, _trigger_minute)
        st.session_state["_mega_analysis_trigger_last_synced"] = _file_trigger_pair
    else:
        # Live cross-device sync (this fragment already ticks every
        # second) — pulls a change made from another tab/device into
        # THIS session's own display, but only when nothing typed here
        # is still pending; see _sync_pull_from_file's own docstring.
        _sync_pull_from_file(
            "mega_analysis_trigger_time",
            "_mega_analysis_trigger_last_synced",
            _file_trigger_pair,
            widget_to_comparable=lambda t: (t.hour, t.minute),
            comparable_to_widget=lambda pair: dt_time(*pair),
        )
    _new_trigger_time = st.time_input(
        "Change daily trigger time (UTC)", key="mega_analysis_trigger_time", step=300
    )
    _new_trigger_pair = (_new_trigger_time.hour, _new_trigger_time.minute)
    if _new_trigger_pair != st.session_state["_mega_analysis_trigger_last_synced"]:
        set_mega_analysis_trigger(*_new_trigger_pair)
        st.session_state["_mega_analysis_trigger_last_synced"] = _new_trigger_pair

    last_status = state.get("last_status")
    last_attempt = state.get("last_attempt_utc")
    # Compact caption-styled status instead of a full st.success/st.warning
    # alert box (direct user request 2026-08-23: shrink this and squeeze
    # the gap before the manual controls below it).
    if last_status == "success" and last_attempt:
        st.caption(f":green[✓ Last run ({_format_last_run(last_attempt)}): succeeded.]")
    elif last_status in ("error", "cli_failed", "timeout") and last_attempt:
        st.caption(
            f":orange[⚠ Last attempt ({_format_last_run(last_attempt)}) did NOT succeed: "
            f"{state.get('last_detail', '(no detail recorded)')}]"
        )


_render_mega_analysis_countdown = st.fragment(run_every=1)(_render_mega_analysis_countdown)


def _watch_for_external_mega_toggle_change() -> None:
    """Cross-device live-sync guardrail for the mega-analysis enable
    toggle specifically (direct user request 2026-09-02: "add live
    sync... with some guard rails"). Every OTHER synced control in this
    app already lives inside its own run_every fragment, so a plain
    _sync_pull_from_file call on each tick is enough — but THIS toggle
    was deliberately kept in the OUTER, non-fragment script (see its own
    call site's comment) so a local click updates the "Suggest Portfolio
    Mix" button's disabled= in the very same instant, which a fragment-
    scoped rerun can't do. That means nothing periodically reruns the
    outer script here on its own, so an external change (another
    device's toggle) would otherwise only ever show up the next time
    THIS tab happens to rerun for some unrelated reason.

    This tiny fragment's only job is noticing that gap and closing it:
    every couple of seconds, if this session has no pending local edit
    of its own AND the file has moved since this session's last sync,
    force a full st.rerun() (the default scope="app", not "fragment" —
    confirmed against the installed Streamlit version) so the outer
    script reruns and its own _sync_pull_from_file call (right where the
    toggle is rendered) picks up the fresh value together with the
    button it drives. This function itself never writes the toggle's
    value anywhere — it only ever triggers the rerun that lets the
    outer script's own existing logic do that safely."""
    if "mega_analysis_enabled_toggle" not in st.session_state:
        return  # outer script hasn't rendered the toggle yet this session
    last_synced = st.session_state["_mega_analysis_enabled_last_synced"]
    if st.session_state["mega_analysis_enabled_toggle"] != last_synced:
        return  # a local edit is pending — the outer script's own write path owns it
    if read_mega_analysis_enabled() != last_synced:
        st.rerun()


_watch_for_external_mega_toggle_change = st.fragment(run_every=2)(_watch_for_external_mega_toggle_change)


_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _spinner_frame() -> str:
    """One animated frame, chosen from the wall clock — this function is
    called from a fragment that reruns every second (see the st.fragment
    wrapper below), so indexing by whole seconds is enough to make the
    glyph visibly cycle without needing any state of its own."""
    return _SPINNER_FRAMES[int(time.time()) % len(_SPINNER_FRAMES)]


def _render_mega_analysis_progress_log() -> None:
    """FTMO-only: renders the unattended mega session's own live status —
    direct user request 2026-08-25, moved here (the bottom of the shared
    controls box, right after the "Suggest Portfolio Mix" button) from a
    single-line st.info() that used to sit at the TOP of the box.

    Redesigned 2026-08-27 after two real bugs found live in the same
    session:

    1. The whole log used to VANISH mid-run once 5 minutes passed with
       no new write — but a single step (e.g. Claude's own multi-minute
       web-research call) can legitimately run that long with nothing
       new to report. Fixed by switching to the shared
       ai.mega_analysis.mega_session_is_live check, whose real "still
       running" signal is comparing against the last COMPLETED attempt
       in state.json, not a short fixed age — see its own docstring.
    2. The display used to number every message 1, 2, 3... and one real
       run reached 246+ entries within minutes, because the audit-
       model-pool's own per-second retry/countdown status was being
       APPENDED as a new entry on every tick instead of overwritten.
       Fixed at the source (ai.mega_analysis._write_current_activity)
       by splitting the progress file into a small, bounded `steps`
       list (genuinely one-time milestones) plus a single
       `current_activity` slot that's replaced in place, mirroring the
       manual "Suggest Portfolio Mix" button's own st.status()/
       st.empty() pattern, which never had either bug.

    Renders like a CLI installer: completed steps get a plain checkmark
    (no numbering), and exactly one line — the current step if nothing
    more granular is available, or the live current_activity text once
    it exists — gets an animated spinner glyph in front of it, so
    there's always a visibly-moving indicator instead of a static list.
    Renders nothing at all (not even an empty container) when no run is
    currently live."""
    state = read_state()
    progress = read_progress()
    if not mega_session_is_live(progress, state):
        return

    steps: list[str] = progress.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return
    current_activity = progress.get("current_activity")

    st.divider()
    st.caption(":material/autorenew: Mega analysis running now — live progress")

    spinner = _spinner_frame()
    done_steps = steps if current_activity else steps[:-1]
    for step in done_steps:
        st.write(f":green[✓] {step}")

    if current_activity:
        lines = current_activity.split("\n", 1)
        st.write(f"{spinner} {lines[0]}")
        if len(lines) > 1 and lines[1].strip():
            with st.expander("Details"):
                st.text(lines[1])
    else:
        st.write(f"{spinner} {steps[-1]}")


_render_mega_analysis_progress_log = st.fragment(run_every=1)(_render_mega_analysis_progress_log)


@st.cache_resource
def _clerk_timer_state() -> dict:
    """Mutable holder for the Clerk's in-app periodic-check countdown —
    deliberately NOT st.session_state (2026-08-30 direct user correction:
    a plain browser page refresh opens a brand-new Streamlit session
    with fresh session_state, which was silently resetting the countdown
    on every refresh — a real bug, not a rendering quirk) and deliberately
    NOT a JSON file either (a real app restart SHOULD reset this, per
    that same correction: "only... on full app reset or restart or from
    the enable/disable toggle switch"). st.cache_resource is the one
    primitive that's actually shaped for this: its factory (this
    function) runs exactly once per PROCESS, and the same returned dict
    is handed to every session/tab that calls it afterward — so a page
    refresh (new session, same process) sees the SAME dict and the
    countdown keeps going, while a real restart (new process) re-runs
    this factory and starts fresh, and the toggle's own handler below
    still resets it explicitly on enable/disable."""
    return {"next_check_utc": None}


def _run_clerk_check_with_timeout() -> None:
    """The actual work _fire_scheduled_job_once runs on a background
    thread for the Clerk's in-app periodic trigger. Unlike
    run_scheduled_mega_analysis (which already wraps its own call in
    run_with_timeout internally), run_clerk_execution_check does NOT
    self-bound its own runtime — every other caller (the manual button,
    clerk_execution_job.py) wraps it in run_with_timeout itself, so this
    does too, just from the background thread now instead of the
    fragment's own thread."""
    timed_out = object()
    result = run_with_timeout(
        run_clerk_execution_check,
        config.CLERK_EXECUTION_RUN_TIMEOUT_SECONDS,
        default=timed_out,
        catch_exceptions=False,
    )
    if result is timed_out:
        logger.error(
            "Periodic Clerk check timed out after %.0f minutes",
            config.CLERK_EXECUTION_RUN_TIMEOUT_SECONDS / 60,
        )


def _render_clerk_execution_panel() -> None:
    """FTMO-only: shows the mega session's own Immediate Allocation AND
    Pending Setups (the "clerk/executioner" half of the boardroom
    architecture — see ai/clerk_execution.py's own module docstring),
    each with its real status (placed/failed-with-reason for immediate
    targets; unchecked/order placed/filled/closed-after-fill plus
    the Clerk's own reasoning text for Pending Setups), and a live-progress
    banner while a check is actually running — same standing "automated
    runs must be as visible as a manual button click" principle as
    _render_mega_analysis_countdown above, applied to this second
    unattended job. The Immediate Allocation table was added 2026-08-23
    direct user request ("I see position in 5 assets but only 2 assets
    are showing up in the clerk section") — this panel previously only
    ever rendered pending_setups, never the immediate_allocation targets
    ai.clerk_execution.run_clerk_execution_check actually tries to
    execute every poll, so a FAILED attempt (no settlement record ever
    gets created for those) was invisible here even though it was
    already in the log file. Also has its own enable/disable toggle, a
    "next review" timer, and a review-frequency picker (direct user
    request 2026-08-23) — all three widgets live inside this same
    st.fragment(run_every=1), which is safe here (unlike the mega
    analysis toggle) since nothing OUTSIDE this fragment needs to react
    to them; the whole panel simply redraws itself on its own next tick.

    2026-08-30: the "next review" timer stopped being purely decorative
    — this fragment now actually calls run_clerk_execution_check() when
    its own countdown (_clerk_timer_state(), process-lifetime, shared
    across every open tab — see that function's own docstring for why
    it's not st.session_state) reaches zero and the toggle above is on,
    so the Clerk genuinely re-checks on the configured cadence for as
    long as the app process is running, not only when a mega session
    happens to trigger it inline. Ticking at 1s (not 5s) to read as a
    live countdown the same way the mega-analysis countdown above does
    — direct user comparison."""
    suggestion = read_latest_suggestion()
    pending_setups = suggestion.get("pending_setups", [])

    now_utc = datetime.now(timezone.utc)
    exec_state = read_execution_state()
    # Content-change fingerprint for the two tables below (direct user
    # complaint 2026-09-02: this whole panel — including its tables —
    # visually "jerks" on every single 1-second tick, since Streamlit
    # just replaces the dataframe wholesale each time regardless of
    # whether anything in it actually changed). Only advances when a
    # NEW mega session was generated or a NEW Clerk check genuinely
    # completed — not on every tick — so giving each table a key derived
    # from it (see below) makes Streamlit only remount that table when
    # there's real new content to show, which is what lets the CSS fade
    # animation replay meaningfully instead of playing every second
    # regardless of whether anything changed (same proven technique
    # already used for the Watchlist's own up/down price-pulse
    # animation — see that CSS block's own key-generation comment).
    _clerk_content_gen = f"{suggestion.get('generated_utc', '')}_{exec_state.get('last_attempt_utc', '')}"
    exec_progress = read_execution_progress()
    clerk_check_live = execution_check_is_live(exec_progress, exec_state)
    live_message = None
    if clerk_check_live:
        live_message = exec_progress.get("message")
    # Direct user request 2026-08-31, after a real live incident: starting
    # a mega session while the Clerk's own periodic trigger could also
    # fire froze the entire Streamlit session (not just this panel) —
    # root-caused to concurrent MT5 access from the two background
    # threads (now independently prevented at the source by data.
    # mt5_source's own MT5_ACCESS_LOCK). Pausing the Clerk's countdown
    # while a mega session is live is a second, deliberate layer on top
    # of that lock, not a duplicate fix for the same bug — the user's
    # own reasoning: don't spend local-LLM/CPU capacity on Clerk checks
    # at the same moment the heavier mega session already needs it.
    mega_session_live = mega_session_is_live(read_progress(), read_state())

    with st.container(border=True, key="clerk_execution_panel"):
        heading_col, toggle_col, tactical_toggle_col = st.columns([3, 1, 1.3])
        with heading_col:
            st.subheader(":material/support_agent: Execution Clerk")
        with toggle_col:
            # Same multi-tab race fixed for the mega-analysis trigger-
            # time picker above (see its own comment for the full
            # explanation) — this panel is ALSO a run_every fragment, so
            # a second, stale tab would otherwise flip this back within
            # seconds of a real change made elsewhere. Compare against
            # this session's own last-synced value, never the file's.
            if "clerk_execution_enabled_toggle" not in st.session_state:
                st.session_state["clerk_execution_enabled_toggle"] = read_clerk_execution_enabled()
                st.session_state["_clerk_execution_enabled_last_synced"] = (
                    st.session_state["clerk_execution_enabled_toggle"]
                )
            else:
                # Live cross-device sync — this whole panel already
                # ticks every second and nothing outside it depends on
                # this toggle, so a plain pull-on-tick is enough (no
                # st.rerun() needed, unlike the mega-analysis toggle).
                _sync_pull_from_file(
                    "clerk_execution_enabled_toggle",
                    "_clerk_execution_enabled_last_synced",
                    read_clerk_execution_enabled(),
                )
            clerk_enabled = st.toggle("Enabled", key="clerk_execution_enabled_toggle")
            if clerk_enabled != st.session_state["_clerk_execution_enabled_last_synced"]:
                set_clerk_execution_enabled(clerk_enabled)
                st.session_state["_clerk_execution_enabled_last_synced"] = clerk_enabled
                # Direct user request 2026-08-30: re-enabling restarts the
                # "Next review" countdown fresh from this moment, rather
                # than resuming some stale reference from before it was
                # switched off — the reset below (which also runs every
                # tick while disabled) already guarantees this, this is
                # just the immediate case of flipping it on right now.
                _clerk_timer_state()["next_check_utc"] = None
        with tactical_toggle_col:
            # New, separately-staged authority (added 2026-08-27, direct
            # user request after a real gold position went from +$28 to
            # -$61 while the mega session hadn't run in days and the
            # clerk had zero authority to react — see ai/clerk_
            # execution.py's own docstring). Defaults OFF, unlike the
            # toggle above — this is fresh unattended authority over real
            # money, not yet proven live; the full pipeline still runs
            # and reports every poll even while off (shadow mode, see the
            # Tactical columns below), it just never touches a real
            # order until switched on. Same multi-tab-race-safe pattern.
            if "tactical_defense_enabled_toggle" not in st.session_state:
                st.session_state["tactical_defense_enabled_toggle"] = read_tactical_defense_enabled()
                st.session_state["_tactical_defense_enabled_last_synced"] = (
                    st.session_state["tactical_defense_enabled_toggle"]
                )
            else:
                # Live cross-device sync — same reasoning as the toggle above.
                _sync_pull_from_file(
                    "tactical_defense_enabled_toggle",
                    "_tactical_defense_enabled_last_synced",
                    read_tactical_defense_enabled(),
                )
            tactical_enabled = st.toggle("Tactical defense (DEFEND/EXIT)", key="tactical_defense_enabled_toggle")
            if tactical_enabled != st.session_state["_tactical_defense_enabled_last_synced"]:
                set_tactical_defense_enabled(tactical_enabled)
                st.session_state["_tactical_defense_enabled_last_synced"] = tactical_enabled

        st.caption(
            "Checks each Pending Setup against fresh live technicals and executes a "
            "confirmed one immediately — no human confirmation step."
        )
        if not tactical_enabled:
            st.caption(
                ":gray[Tactical defense off — still checks every open position and reports "
                "what it WOULD do (see the Tactical columns below), but never acts.]"
            )

        interval_minutes = read_clerk_execution_interval_minutes()

        # In-app periodic trigger, direct user request 2026-08-30: while
        # this fragment is alive (the webapp is open) and the toggle
        # above is on, actually re-check on the "Review every (min)"
        # cadence below — not just a passive countdown. Previously this
        # metric counted down toward next_execution_check_utc, a clock-
        # aligned instant meant for clerk_execution_job.py's own OS-level
        # Scheduled Task poll; with no such task configured on this
        # machine (direct user choice, "webapp-only, no independent
        # Clerk task"), nothing ever advanced that function's own
        # last_run_interval_utc marker, so it permanently read "due now"
        # — a real bug, not a rendering issue.
        #
        # _clerk_timer_state() (process-lifetime, shared across every
        # open tab — see its own docstring) drives this, NOT
        # st.session_state: a plain page refresh opens a brand-new
        # session with fresh session_state, which was silently resetting
        # this countdown on every refresh (a second real bug, direct
        # user report) — it must reset ONLY on the toggle above or an
        # actual app restart, never on a refresh. EXECUTION_LOCK_PATH
        # (the same lock the manual "Run mega session" button's own
        # inline call already uses below) is what keeps two tabs from
        # ever actually running a check at once, not this timer.
        timer_state = _clerk_timer_state()
        if not clerk_enabled:
            # Re-enabling later must start the countdown fresh from that
            # moment (direct user request), never resume a stale
            # reference from before it was switched off.
            timer_state["next_check_utc"] = None
            remaining_seconds = None
        elif mega_session_live:
            # Stop the countdown entirely while a mega session runs, per
            # the user's own explicit request — resetting to None (not
            # just holding it) so it restarts a fresh full interval once
            # the mega session ends, rather than resuming a countdown
            # that may have already elapsed while paused.
            timer_state["next_check_utc"] = None
            remaining_seconds = None
        elif clerk_check_live:
            # Real gap found live 2026-09-16, direct user report: unlike
            # the mega-session pause immediately above, this countdown
            # previously kept ticking (and re-firing _fire_scheduled_job_
            # once every time it hit zero) even while a Clerk check was
            # ALREADY genuinely running — that inner call is a harmless
            # no-op while the prior check's own background thread is
            # still alive, but `next_check_utc` still got unconditionally
            # reset to "now + interval" regardless, so the visible
            # countdown kept recycling on its own schedule with no
            # relationship to whether a real check was actually in
            # flight. With a short review interval (this account's own
            # is currently 1 minute) a single check that legitimately
            # takes longer than one interval to finish (multiple MT5/
            # LLM calls, or a slow/retrying Ollama call) made the
            # countdown visibly "reset and fire again" well before a
            # user would expect, since it was never actually synced to
            # real completion. Paused the same way mega_session_live is
            # — reset to None, not just held, so a fresh full interval
            # starts once this check genuinely finishes.
            timer_state["next_check_utc"] = None
            remaining_seconds = None
        else:
            if timer_state["next_check_utc"] is None:
                timer_state["next_check_utc"] = now_utc + timedelta(minutes=interval_minutes)
            next_check = timer_state["next_check_utc"]
            remaining_seconds = max(0, int((next_check - now_utc).total_seconds()))

        timer_col, freq_col = st.columns([1, 1])
        with timer_col:
            if mega_session_live:
                st.metric("Next review", "paused (mega session running)")
            elif clerk_check_live:
                st.metric("Next review", "running now")
            elif remaining_seconds is None:
                st.metric("Next review", "paused")
            else:
                rem_minutes, rem_seconds = divmod(remaining_seconds, 60)
                st.metric(
                    "Next review",
                    "due now" if remaining_seconds <= 0 else f"{rem_minutes}m {rem_seconds:02d}s",
                )
        with freq_col:
            # Range starts at 1 min (direct user request 2026-08-23).
            # Same multi-tab race fixed for the mega-analysis
            # trigger-time picker (see its own comment) — compare
            # against this session's own last-synced value, never the
            # file's current one, so a second stale tab can't flip this
            # back within seconds of a real change made elsewhere.
            if "clerk_execution_interval_select" not in st.session_state:
                st.session_state["clerk_execution_interval_select"] = interval_minutes
                st.session_state["_clerk_execution_interval_last_synced"] = interval_minutes
            else:
                # Live cross-device sync — same reasoning as the toggles
                # above. _freq_options below already includes whatever
                # this pulls in (it always includes the fresh file value).
                _sync_pull_from_file(
                    "clerk_execution_interval_select",
                    "_clerk_execution_interval_last_synced",
                    interval_minutes,
                )
            # Always includes the CURRENT file value AND this session's
            # own last-synced value (they can differ if another tab
            # changed it) so a custom/hand-edited interval never breaks
            # this selectbox — st.selectbox requires its key's stored
            # value to be one of the options offered.
            _freq_options = sorted({
                1, 2, 5, 10, 15, 30, 60, interval_minutes,
                st.session_state["_clerk_execution_interval_last_synced"],
            })
            new_interval = st.selectbox("Review every (min)", _freq_options, key="clerk_execution_interval_select")
            if new_interval != st.session_state["_clerk_execution_interval_last_synced"]:
                set_clerk_execution_interval_minutes(new_interval)
                st.session_state["_clerk_execution_interval_last_synced"] = new_interval
                # Re-base the countdown on the new interval immediately,
                # rather than finishing out a countdown sized for the old
                # one.
                _clerk_timer_state()["next_check_utc"] = None

        if clerk_enabled and not mega_session_live and remaining_seconds is not None and remaining_seconds <= 0:
            # Backgrounded (see _fire_scheduled_job_once's own docstring
            # for the real "fades and stays stale" bug this fixes,
            # 2026-08-30/31) — never blocks this fragment's own tick, so
            # the countdown/live-progress caption below keep updating
            # normally for however long the actual check takes. Advanced
            # immediately (not after completion) since nothing here waits
            # on it any more; a still-in-flight check from a previous
            # tick is separately guarded against inside the helper.
            _fire_scheduled_job_once(
                _run_clerk_check_with_timeout, timer_state, EXECUTION_LOCK_PATH, EXECUTION_LOCK_STALE_AFTER_SECONDS,
            )
            timer_state["next_check_utc"] = datetime.now(timezone.utc) + timedelta(minutes=interval_minutes)

        if interval_minutes < 5:
            st.caption(
                ":gray[The Windows Scheduled Task itself only polls every 5 min, so anything "
                "below that still checks at most every ~5 min in practice.]"
            )

        if not clerk_enabled:
            st.caption(":gray[Disabled — won't check or execute anything until re-enabled.]")

        if not suggestion:
            st.caption("No mega-analysis suggestion on file yet.")
            return

        settled = read_settlement().get("settled", {})
        last_verdicts = exec_state.get("last_verdicts", {})
        last_tactical_verdicts = exec_state.get("last_tactical_verdicts", {})
        # Added 2026-08-23 direct user request ("I see position in 5
        # assets but only 2 assets are showing up in the clerk section")
        # — this table previously only ever showed pending_setups; the
        # immediate_allocation targets (what compute_rebalance_plan
        # actually tries to execute every poll) were never displayed at
        # all, so a symbol that failed (no settlement record ever gets
        # created for a failed attempt) was invisible outside the log
        # file. last_execution_results covers every symbol in the latest
        # plan, success or failure.
        immediate_allocation = suggestion.get("immediate_allocation", {})
        last_execution_results = exec_state.get("last_execution_results", {})
        immediate_symbols = [s for s in immediate_allocation if s.upper() != "CASH"]
        st.caption("**Immediate allocation** — feasible right now per the mega session:")
        if not immediate_symbols:
            st.caption("Cash-only — no immediate-allocation targets from the latest mega session.")
        else:
            imm_rows = []
            for symbol in immediate_symbols:
                entry = immediate_allocation[symbol]
                rec = settled.get(symbol)
                result = last_execution_results.get(symbol)
                if rec is not None:
                    status = rec.get("state", "unknown")
                elif result is not None:
                    status = "placed" if result.get("success") else "failed"
                else:
                    status = "not yet checked"
                # Watch condition / verdict: added 2026-08-23 direct user
                # request — Claude's daily mega session can now write an
                # invalidation_condition per already-suggested position,
                # and the Clerk mechanically re-checks it every poll; since
                # this executes with no human confirmation, "why did this
                # fire" must be visible here, not just the log.
                invalidation_condition = entry.get("invalidation_condition")
                # Real bug found live 2026-08-25: last_verdicts is a flat
                # symbol-keyed dict shared with the UNRELATED Pending
                # Setups verdict mechanism — a symbol like BTCUSD that's
                # BOTH an immediate_allocation entry (pct: 0, no
                # invalidation_condition) AND separately re-listed as a
                # fresh Pending Setup gets a real last_verdicts entry from
                # THAT check, which then leaked into this row as if it
                # were an invalidation-check result for the immediate
                # allocation entry — misleading, since this entry was
                # never actually checked for invalidation at all. Only
                # show verdict/reasoning here when this symbol genuinely
                # HAS an invalidation_condition (i.e. was actually a
                # watched_positions candidate this poll).
                verdict = last_verdicts.get(symbol, {}) if invalidation_condition else {}
                if invalidation_condition:
                    verdict_text = (
                        ("INVALIDATED" if verdict.get("confirmed") else "still holds")
                        if symbol in last_verdicts else "not yet checked"
                    )
                elif entry.get("pct", 0) == 0:
                    # Real user question 2026-08-25: a pct: 0 row has no
                    # invalidation_condition (there's nothing left to
                    # watch once the mega session is closing/cancelling
                    # it) — this column used to just show "—" here, which
                    # read as an empty/broken entry rather than the real
                    # close/cancel instruction it is. `rec`'s state (still
                    # present in settlement, so its ORIGINAL nature is
                    # known) gives the precise wording when available;
                    # most closes/cancels were never Clerk-tracked in
                    # settlement to begin with (e.g. a manually-opened
                    # position, or one settled by an earlier mega-session
                    # cycle), so this falls back to an equally accurate,
                    # just less specific, description in that case.
                    if rec is not None and rec.get("state") == "order_placed":
                        verdict_text = "Cancelled pending order"
                    elif rec is not None and rec.get("state") in ("filled", "closed_after_fill"):
                        verdict_text = "Closed existing position"
                    else:
                        verdict_text = "Closed/cancelled by mega session"
                else:
                    verdict_text = "—"
                raw_text = verdict.get("raw_text", "") or ""
                reasoning = raw_text[:300] + ("…" if len(raw_text) > 300 else "") if raw_text else ""
                # Real user question 2026-08-25: a pct: 0 row (Claude
                # closing/cancelling an already-suggested position, see
                # AllocationEntry.reason) had every other column blank
                # (no watch condition/verdict — there's nothing to watch
                # once it's closing) and this table never showed `reason`
                # at all, so the row looked like a garbage/empty entry
                # rather than the real, deliberate instruction it is.
                mega_reason = entry.get("reason", "") or ""
                mega_reason_display = (
                    mega_reason[:300] + ("…" if len(mega_reason) > 300 else "") if mega_reason else "—"
                )
                # Tactical defense (DEFEND/EXIT) — added 2026-08-27, the
                # Clerk's own new short-term/"trend" authority, checked
                # independently of the invalidation-condition trip-wire
                # above (see ai/clerk_execution.py's own docstring for
                # the real gold-trade incident that motivated it). Every
                # already-filled position gets checked every poll, on or
                # off — the toggle above only gates whether a verdict is
                # actually ACTED on, so this stays visible even in shadow
                # mode, the same "as visible as a manual click" principle
                # this whole panel already follows for everything else.
                tactical = last_tactical_verdicts.get(symbol)
                if tactical is None:
                    tactical_verdict_display = "not yet checked" if status in ("filled",) else "—"
                    tactical_action_display = "—"
                    tactical_justification_display = "—"
                else:
                    tier = tactical.get("tier", "hold")
                    shadow_prefix = "[SHADOW] " if tactical.get("shadow_mode") else ""
                    skipped_reason = tactical.get("skipped_reason") or ""
                    if skipped_reason:
                        tactical_verdict_display = f"{shadow_prefix}{tier.upper()} (skipped)"
                    else:
                        tactical_verdict_display = f"{shadow_prefix}{tier.upper()}"
                    if skipped_reason:
                        # Found on audit: a genuine guardrail REJECTION
                        # (cooldown, never-widen-stop, wrong-side-of-
                        # price, sizing failure — see ai/clerk_
                        # execution.py::_validate_and_apply_tactical_
                        # verdict's own rejected_reason) and a VALID
                        # action merely pending on the disabled toggle
                        # used to look identical here ("Would: ..." for
                        # both) — during exactly the shadow-mode
                        # observation window this feature's rollout plan
                        # depends on, that could mask a DEFEND that would
                        # NEVER actually fire as if it were healthy and
                        # waiting. Show the real reason instead.
                        tactical_action_display = (
                            skipped_reason[:200] + ("…" if len(skipped_reason) > 200 else "")
                        )
                    elif tier == "exit":
                        tactical_action_display = "Full close" if tactical.get("applied") else "Would: full close"
                    elif tier == "defend":
                        parts = []
                        new_stop = tactical.get("new_stop_loss")
                        fraction = tactical.get("partial_close_fraction")
                        verb = "" if tactical.get("applied") else "Would: "
                        if new_stop is not None:
                            parts.append(f"Stop → {new_stop:g}")
                        if fraction is not None:
                            parts.append(f"{fraction:.0%} closed")
                        tactical_action_display = verb + ", ".join(parts) if parts else "—"
                    else:
                        tactical_action_display = "—"
                    tactical_justification = (
                        f"{tactical.get('rule_citation', '')} — {tactical.get('numbers_citation', '')}"
                        if tier in ("defend", "exit")
                        else ""
                    )
                    tactical_justification_display = (
                        tactical_justification[:300] + ("…" if len(tactical_justification) > 300 else "")
                        if tactical_justification.strip(" —")
                        else "—"
                    )
                imm_rows.append(
                    {
                        "Symbol": symbol,
                        "Side": entry.get("side"),
                        "Target %": entry.get("pct"),
                        "Reason": mega_reason_display,
                        "Status": status,
                        "Detail": result.get("detail", "") if result is not None else "",
                        "Watch condition": invalidation_condition or "—",
                        "Last invalidation verdict": verdict_text,
                        "Clerk's reasoning": reasoning,
                        "Tactical verdict": tactical_verdict_display,
                        "Tactical action": tactical_action_display,
                        "Tactical justification": tactical_justification_display,
                    }
                )
            st.dataframe(
                pd.DataFrame(imm_rows), hide_index=True, key=f"clerk_imm_table_{_clerk_content_gen}"
            )

        if not pending_setups:
            st.caption("No Pending Setups from the latest mega session — nothing to watch for.")
        else:
            st.caption("**Pending Setups** — conditional, watched until the next mega session:")
            rows = []
            for s in pending_setups:
                symbol = s.get("symbol", "?")
                rec = settled.get(symbol)
                # .get(..., "unknown") rather than direct indexing — this
                # file is entirely self-managed, but a display function
                # should degrade to a placeholder rather than crash the
                # whole page if it's ever hand-edited or half-written.
                status = rec.get("state", "unknown") if rec is not None else "unchecked"
                verdict = last_verdicts.get(symbol, {})
                raw_text = verdict.get("raw_text", "") or ""
                # Direct user request 2026-08-23: when the Clerk judges a
                # setup genuinely STALE (not just "hasn't triggered yet")
                # due to a real delay, the prompt asks it to lead with a
                # "STALE SETUP WARNING:" line — surfaced here verbatim
                # rather than buried in the log/state file only.
                reasoning = raw_text[:300] + ("…" if len(raw_text) > 300 else "") if raw_text else ""
                rows.append(
                    {
                        "Symbol": symbol,
                        "Side": s.get("side"),
                        "Status": status,
                        "Last verdict": (
                            "CONFIRMED" if verdict.get("confirmed") else "NOT_CONFIRMED"
                        ) if symbol in last_verdicts else "not yet checked",
                        "Trigger condition": s.get("trigger_condition"),
                        "Clerk's reasoning": reasoning,
                    }
                )
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, key=f"clerk_pending_table_{_clerk_content_gen}"
            )

        last_status = exec_state.get("last_status")
        last_attempt = exec_state.get("last_attempt_utc")
        # Compact caption-styled status instead of a full st.success/
        # st.warning alert box — direct user request 2026-08-25, same
        # "shrink this and squeeze the box" treatment already applied to
        # the Mega Market Analysis section's own last-run status above.
        #
        # Direct user request 2026-08-25: this used to be a SEPARATE blue
        # st.info() banner at the TOP of the box (right under the
        # heading) while a check was running, alongside this green/
        # orange status line at the bottom staying frozen on the
        # PREVIOUS result the whole time — two different messages in two
        # different places. Now there's exactly one status line, always
        # in this same spot: it shows the live blue "running now"
        # message in place of the last-check line while a check is
        # actually in progress, and automatically reverts to the normal
        # green/orange/gray line the moment exec_state's own
        # last_attempt_utc catches up (the same freshness check already
        # used above to compute live_message, so no separate state is
        # needed).
        if live_message:
            st.caption(f":blue[↻ Running now — {live_message}]")
        elif last_status == "success" and last_attempt:
            _exec_last_detail = exec_state.get("last_detail", "succeeded")
            # Real gap found live 2026-09-16: an Ollama-outage warning
            # (see ai.clerk_execution._detect_ollama_outage) gets folded
            # into this SAME "success" detail string when real orders
            # still went out fine this poll despite it — rendering it in
            # green would bury a real, ongoing issue inside a status
            # color that reads as "all clear."
            if "WARNING" in _exec_last_detail:
                st.caption(f":orange[⚠ Last check ({_format_last_run(last_attempt)}): {_exec_last_detail}.]")
            else:
                st.caption(f":green[✓ Last check ({_format_last_run(last_attempt)}): {_exec_last_detail}.]")
        elif last_status == "blocked" and last_attempt:
            st.caption(f":orange[⚠ Last check ({_format_last_run(last_attempt)}) executed nothing: {exec_state.get('last_detail', '')}]")
        elif last_status == "disabled" and last_attempt:
            st.caption(f"Last check ({_format_last_run(last_attempt)}): skipped — the clerk was disabled.")


_render_clerk_execution_panel = st.fragment(run_every=1)(_render_clerk_execution_panel)


def _render_allocation_chart(allocation: dict[str, AllocationEntry]) -> None:
    # Real user question 2026-08-25: a symbol Claude is explicitly
    # closing/cancelling (FTMO's position-reassessment feature — pct: 0
    # with a real `reason`, see ai/portfolio_suggest.py::AllocationEntry)
    # was showing up as a 0%-slice legend entry here, which reads as a
    # garbage/empty row since a pie chart has nothing meaningful to draw
    # for 0% and this chart never showed `reason` anyway. That data is
    # real and intentional — it belongs in the Immediate Allocation
    # table (now with its own Reason column), not in a proportional
    # chart, which only makes sense for what's actually held right now.
    held = {symbol: entry for symbol, entry in allocation.items() if entry.pct > 0}
    if not held:
        return
    _render_pie(
        list(held.keys()), [e.pct for e in held.values()], "Allocation by Instrument"
    )


def _render_sector_allocation_chart(
    allocation: dict[str, AllocationEntry], analyses: list
) -> None:
    sector_pct = compute_sector_allocation(allocation, analyses)
    if not sector_pct:
        return
    _render_pie(list(sector_pct.keys()), list(sector_pct.values()), "Allocation by Sector")


def _render_sector_performance_chart(all_assets: list[PSXAsset]) -> None:
    performance = compute_sector_performance(all_assets)
    if not performance:
        return
    # Reversed so the best-performing sector renders at the TOP of the
    # horizontal bar chart — Plotly draws categorical y-axes bottom-to-top
    # in the order given, so the worst performer needs to come first.
    ordered = list(reversed(performance))
    sectors = [p[0] for p in ordered]
    changes = [p[1] for p in ordered]
    counts = [p[2] for p in ordered]
    colors = [_GAIN_COLOR if c >= 0 else _LOSS_COLOR for c in changes]

    fig = go.Figure(
        go.Bar(
            x=changes,
            y=sectors,
            orientation="h",
            marker_color=colors,
            customdata=counts,
            hovertemplate="%{y}: %{x:+.2f}%% avg (%{customdata} symbols)<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="gray", line_width=1)
    fig.update_layout(
        title="Sector Performance Today — Whole PSX Market",
        xaxis_title="Average % change",
        margin=dict(l=20, r=20, t=50, b=20),
        height=max(380, 34 * len(sectors)),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_risk_return_scatter(analyses: list) -> None:
    rows = [
        {
            "symbol": a.symbol,
            "sector": a.sector_name,
            "volatility": a.stats.volatility_annualized_pct,
            "relative_strength": a.relative_strength_1m_pct,
        }
        for a in analyses
        if a.stats.volatility_annualized_pct is not None and a.relative_strength_1m_pct is not None
    ]
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="volatility",
        y="relative_strength",
        color="sector",
        text="symbol",
        color_discrete_sequence=_CATEGORICAL_COLORS,
        labels={
            "volatility": "Annualized Volatility (%)",
            "relative_strength": "1-Month Relative Strength vs. Index (%)",
        },
    )
    fig.update_traces(textposition="top center", marker=dict(size=13, line=dict(width=1, color="white")))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Risk vs. Relative Strength — Enriched Candidates",
        margin=dict(l=20, r=20, t=50, b=20),
        height=520,
        legend_title_text="Sector",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_capital_at_risk_chart(
    allocation: dict[str, AllocationEntry], pct_is_risk: bool = False
) -> None:
    """Visualizes the same "aggregate heat" figure the report's own risk-
    management reasoning already computes in prose — needs only the
    allocation block itself (price + stop_loss per symbol), so it works
    identically for every exchange without any exchange-specific data,
    modulo one real difference between them:

    `pct_is_risk` distinguishes PSX's allocation semantic from PMEX/
    FTMO's. PSX is a hypothetical, unleveraged, real-shares-equivalent
    simulation, where "pct" genuinely means "% of hypothetical capital's
    notional value" — for that, "% allocation x stop-loss distance %"
    correctly derives the real % of equity at risk (pct_is_risk=False,
    the default). PMEX/FTMO trade real leveraged MT5 instruments, where
    risk/apply_suggestion.py::compute_rebalance_plan() now sizes
    positions so "pct" directly EQUALS the real % of equity at risk to
    the stop (see that function's own docstring for the live-confirmed
    bug this fixed: the old margin-based sizing let a 44% "allocation" in
    ~50:1-leveraged gold actually risk 11.76% of equity — ~49x what the
    old pct x distance% formula believed, and FTMO force-closed the
    account's very first real trade as a direct result). For those two,
    pct_is_risk=True — multiplying by distance% again would silently
    reintroduce the exact same understatement this was built to fix."""
    rows = []
    for symbol, entry in allocation.items():
        if symbol == "CASH" or not entry.price or entry.stop_loss is None:
            continue
        distance_pct = abs(entry.price - entry.stop_loss) / entry.price * 100
        risk_pct = entry.pct if pct_is_risk else entry.pct * distance_pct / 100
        rows.append((symbol, risk_pct, distance_pct))
    if not rows:
        return
    # Ascending so the highest-risk position ends up at the TOP of the
    # horizontal bar (Plotly draws categorical y-axes bottom-to-top in
    # the order given), matching the sector-performance chart's convention.
    rows.sort(key=lambda r: r[1])
    symbols = [r[0] for r in rows]
    risk = [r[1] for r in rows]
    distances = [r[2] for r in rows]
    total_heat = sum(risk)

    fig = go.Figure(
        go.Bar(
            x=risk,
            y=symbols,
            orientation="h",
            marker_color=_LOSS_COLOR,
            customdata=distances,
            hovertemplate="%{y}: %{x:.2f}% of total capital at risk (stop is %{customdata:.1f}% away)<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Capital at Risk by Position — {total_heat:.1f}% total portfolio heat",
        xaxis_title="% of total capital at risk if stop is hit",
        margin=dict(l=20, r=20, t=50, b=20),
        height=max(360, 34 * len(symbols)),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_week52_position_chart(analyses: list) -> None:
    """A floating range-bar per candidate (52-week low -> high, per PSX's
    own published range) with a marker at today's price — the same
    week52_position_pct number already narrated in the report text, made
    visual across the whole enriched pool at once instead of read one
    symbol at a time."""
    rows = []
    for a in analyses:
        f = a.fundamentals
        if f is None or f.week52_low is None or f.week52_high is None or f.week52_high <= f.week52_low:
            continue
        rows.append((a.symbol, f.week52_low, f.week52_high, a.current, a.week52_position_pct))
    if not rows:
        return
    rows.sort(key=lambda r: r[4] if r[4] is not None else 0.0)
    symbols = [r[0] for r in rows]
    lows = [r[1] for r in rows]
    spans = [r[2] - r[1] for r in rows]
    currents = [r[3] for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=spans,
            y=symbols,
            base=lows,
            orientation="h",
            marker_color="rgba(148, 148, 148, 0.35)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=currents,
            y=symbols,
            mode="markers",
            marker=dict(size=13, color=_CATEGORICAL_COLORS[0], symbol="diamond", line=dict(width=1, color="white")),
            name="Current price",
            hovertemplate="%{y}: %{x:.2f} PKR today<extra></extra>",
        )
    )
    fig.update_layout(
        title="52-Week Range — Where Each Candidate Sits Today",
        xaxis_title="Price (PKR)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=max(360, 34 * len(symbols)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_instrument_chart(analysis: AssetAnalysis) -> None:
    if analysis.display_name is None or analysis.prices.empty:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=analysis.prices.index,
            y=analysis.prices.values,
            mode="lines",
            name=analysis.symbol,
            line=dict(color=_CATEGORICAL_COLORS[0], width=2),
        )
    )
    if analysis.stats.support is not None:
        fig.add_hline(
            y=analysis.stats.support, line_dash="dash", line_color=_GAIN_COLOR, annotation_text="support"
        )
    if analysis.stats.resistance is not None:
        fig.add_hline(
            y=analysis.stats.resistance,
            line_dash="dash",
            line_color=_LOSS_COLOR,
            annotation_text="resistance",
        )
    fig.update_layout(
        title=f"{analysis.symbol} ({analysis.display_name})",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

def _load_latest_saved_suggestion(records_dir: Path) -> tuple[str, datetime] | None:
    """Best-effort: the most recent saved records/*.md transcript's final
    answer plus its generation timestamp (parsed from the filename,
    which save_portfolio_session() always writes as
    "portfolio_suggestion_YYYY-MM-DD_HHMMSS.md") — lets the Portfolio
    Suggestion section show the last real result even across an app
    restart, before any button has been clicked this session. Never
    raises: returns None on any failure (missing dir, no files, an
    unexpected filename, an unreadable file, a missing section) since
    this is a nicer default, not a hard requirement to render the page."""
    if not records_dir.exists():
        return None
    files = sorted(records_dir.glob("portfolio_suggestion_*.md"), reverse=True)
    if not files:
        return None
    latest = files[0]
    try:
        generated_at = datetime.strptime(
            latest.stem.removeprefix("portfolio_suggestion_"), "%Y-%m-%d_%H%M%S"
        )
        content = latest.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    marker = "## Stage 3 — Claude's final revised suggestion\n\n"
    if marker not in content:
        return None
    return content.split(marker, 1)[1].strip(), generated_at


st.set_page_config(
    page_title="Quant Advisor", page_icon=":material/monitoring:", layout="wide"
)


@st.cache_resource
def _app_session_token() -> str:
    """A random token generated exactly ONCE per running server process
    (st.cache_resource is process-wide, shared by every session/device,
    not per-tab like st.session_state) — regenerates fresh only when the
    app itself is restarted. This is what lets a browser refresh skip
    the password prompt (see _require_password) while a real app
    restart still forces it again, matching the user's own exact
    request 2026-09-02: "ask for password when app is turned on for
    first time or restarted, not on every page refresh." """
    return secrets.token_urlsafe(24)


def _check_password() -> None:
    entered = st.session_state.get("_password_attempt", "")
    if entered and secrets.compare_digest(entered, config.QUANT2_WEBAPP_PASSWORD):
        st.session_state["_authenticated"] = True
        st.session_state["_password_wrong"] = False
        # Stashes the current process's own token into the URL so a
        # plain browser refresh (which reloads the same URL, query
        # params included, and starts a brand-new st.session_state)
        # still recognizes this browser as already-authenticated —
        # see _app_session_token's own docstring for why a restart
        # still forces a fresh prompt despite this.
        st.query_params["auth"] = _app_session_token()
    else:
        st.session_state["_password_wrong"] = True


def _require_password() -> None:
    """Login gate — direct user request 2026-09-02, added alongside
    Tailscale-based remote/mobile access. Streamlit has no built-in
    authentication of its own, and this app can place real orders on a
    live funded FTMO account, so mere reachability (even over a private
    Tailscale network — this is meant as a SECOND, independent layer
    behind that, not a replacement for it) must never be the only thing
    standing between a stranger and the account.

    Persists across a plain browser refresh via a URL query param
    (`?auth=<token>`), not st.session_state alone — real user complaint
    2026-09-02: a hard refresh starts a brand-new Streamlit session with
    empty session_state, so a session-only gate re-prompted on every
    single page reload on the PC, which felt broken rather than secure.
    The token itself is regenerated only on an actual app restart (see
    _app_session_token), so "stays logged in across refreshes" and "asks
    again after a restart" are both satisfied by the same mechanism.

    Security note: the token lives in the URL, not an HttpOnly cookie —
    on this app's private-Tailscale-only threat model that's an
    accepted, pragmatic tradeoff (no new dependency needed), but it does
    mean the exact post-login URL is itself sensitive for as long as the
    app keeps running — don't share that specific link.

    Fails OPEN (skips the gate entirely) when config.QUANT2_WEBAPP_
    PASSWORD is empty/unset — the default, matching this app's original
    local-only-no-auth behavior — never a silent bypass once a real
    password has actually been configured. Uses secrets.compare_digest
    for the actual password check so a wrong guess can't be timed to
    narrow down the real password character by character."""
    if not config.QUANT2_WEBAPP_PASSWORD:
        return
    if st.session_state.get("_authenticated"):
        return
    if secrets.compare_digest(st.query_params.get("auth", ""), _app_session_token()):
        st.session_state["_authenticated"] = True
        return

    st.title("Quant Advisor")
    st.text_input(
        "Password", type="password", key="_password_attempt", on_change=_check_password
    )
    if st.session_state.get("_password_wrong"):
        st.error("Incorrect password.")
    st.stop()


_require_password()


@st.cache_resource
def _configure_clerk_and_mega_logging() -> None:
    """Wires up persistent file logging for the Clerk and mega-session
    machinery when they run IN-PROCESS inside the webapp (a manual
    button click, or the auto-poll fragment) — added 2026-09-04, direct
    user request after a real incident (a USDCHF position oscillated
    across three separate tickets within about an hour, on a chart with
    no real trend to explain it) had to be reconstructed entirely from
    raw MT5 deal timestamps, because nothing durable ever recorded the
    Clerk's own actual reasoning: this module never called logging.
    basicConfig, so every logger.info/.warning call throughout ai/
    clerk_execution.py, risk/apply_suggestion.py, etc. reached Python's
    own "last resort" handler and vanished — INFO silently dropped
    entirely, WARNING+ only ever printed to a console nobody was
    watching, never saved anywhere. The standalone scheduled-job
    scripts (clerk_execution_job.py, mega_analysis_job.py) already log
    correctly to clerk_execution_log.txt/mega_analysis_log.txt — this
    gives the interactive, webapp-driven path (the ONLY path this
    account actually uses; scheduled tasks were explicitly declined)
    the exact same durable record, in the exact same two files, so
    "what did the Clerk actually decide, and why" is a log read from
    now on instead of an hour of MT5 forensics.

    Attaches directly to each module's own named logger (not the root
    logger via logging.basicConfig) so this can't fight over global
    logging config with anything else that might configure it later in
    the process, and doesn't affect log records from unrelated modules.
    @st.cache_resource makes this run exactly ONCE per running server
    process, not on every Streamlit script rerun (which happens on
    nearly every user interaction) — attaching a fresh FileHandler on
    every rerun would both duplicate every log line and leak file
    handles for the life of the process."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = Path(__file__).resolve().parent

    clerk_handler = logging.FileHandler(root / "clerk_execution_log.txt", encoding="utf-8")
    clerk_handler.setFormatter(formatter)
    for name in ("ai.clerk_execution", "risk.apply_suggestion", "ai.ollama_client", "data.mt5_source", "ai.chart_overlay"):
        target = logging.getLogger(name)
        target.setLevel(logging.INFO)
        target.addHandler(clerk_handler)

    mega_handler = logging.FileHandler(root / "mega_analysis_log.txt", encoding="utf-8")
    mega_handler.setFormatter(formatter)
    for name in ("ai.ftmo_suggest", "ai.portfolio_suggest", "ai.mega_analysis", "ai.psx_suggest"):
        target = logging.getLogger(name)
        target.setLevel(logging.INFO)
        target.addHandler(mega_handler)


_configure_clerk_and_mega_logging()

# Fluid typography: headings/descriptions/metrics scale continuously with
# viewport width via clamp(min, preferred-vw, max) instead of Streamlit's
# fixed default sizes, so a narrower window (not just a literal mobile
# device) shrinks text smoothly rather than clipping/truncating it or
# forcing horizontal scroll. Paired with `white-space: normal` +
# `overflow-wrap` overrides, since Streamlit's own default for metric
# values/labels is a single non-wrapping line with ellipsis truncation —
# the opposite of "shrink, then wrap to the next line" being asked for
# here. Streamlit's own st.columns already stacks vertically below its
# own breakpoint with no help needed; this only handles font sizing/
# wrapping *within* whatever column width results. data-testid selectors
# (confirmed against the installed 1.60 frontend bundle) are paired with
# bare h1/h2/h3 tag selectors as a fallback, since Streamlit's own docs
# guarantee st.title/header/subheader render those tags but don't
# guarantee the exact wrapper nesting around them stays stable release to
# release.
st.markdown(
    """
    <style>
    h1, [data-testid="stHeading"] h1 {
        font-size: clamp(1.3rem, 1.0rem + 1.5vw, 2.1rem) !important;
        line-height: 1.25 !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }
    h2, [data-testid="stHeading"] h2 {
        font-size: clamp(1.1rem, 0.9rem + 1vw, 1.55rem) !important;
        line-height: 1.25 !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }
    h3, [data-testid="stHeading"] h3 {
        font-size: clamp(1.0rem, 0.85rem + 0.75vw, 1.3rem) !important;
        line-height: 1.25 !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        font-size: clamp(0.72rem, 0.65rem + 0.3vw, 0.875rem) !important;
        line-height: 1.35 !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.05rem, 0.8rem + 1.3vw, 1.75rem) !important;
        line-height: 1.2 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: clamp(0.7rem, 0.6rem + 0.35vw, 0.875rem) !important;
        line-height: 1.3 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        overflow-wrap: break-word !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: clamp(0.65rem, 0.55rem + 0.3vw, 0.8rem) !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Quant Advisor")
st.caption(
    "Live MT5 account state, rule-based rebalance checks, and AI portfolio "
    "suggestions — one page per exchange, refreshed automatically below."
)

if config.USE_MOCK_DATA:
    st.caption("Using mock data (USE_MOCK_DATA=1)")
    from data.mock_source import (
        get_account_summary,
        get_current_bid_ask,
        get_current_price,
        get_history_deals,
        get_market_watch,
        get_open_positions,
        get_pending_orders,
        get_symbol_category,
    )
else:
    from data.mt5_source import (
        connect,
        get_account_summary,
        get_current_bid_ask,
        get_current_price,
        get_history_deals,
        get_market_watch,
        get_open_positions,
        get_pending_orders,
        get_symbol_category,
        is_trading_permitted,
    )

# The exchange choice drives which real account (if any) this page
# connects to below — declared here, before Open Positions, rather than
# further down where the Portfolio Suggestion section used to own it.
# Previously the Open Positions section ran unconditionally against
# PMEX's global config regardless of this dropdown (which appeared later
# in the script) — a latent bug that only became a real problem once a
# second live MT5 account (FTMO) existed to actually choose between.
selected_exchange = st.selectbox("Exchange", ["PMEX", "PSX", "FTMO"], index=2)
# Isolates every "Apply Suggestion"/rebalance-execution session_state key
# between PMEX and FTMO (same reasoning already used for PSX's own
# psx_-prefixed keys) so a stale FTMO plan can never look applicable
# while PMEX is selected, and vice versa. PSX has no execution avenue at
# all and keeps its own separate, hardcoded "psx_" keys below.
_execution_state_prefix = "ftmo_" if selected_exchange == "FTMO" else ""

positions = []
pending_orders = []
account = None
ftmo_status = None
if selected_exchange in ("PMEX", "FTMO"):
    try:
        if not config.USE_MOCK_DATA:
            if selected_exchange == "FTMO":
                # connect()'s login/password/server params fall back to
                # config.MT5_* (PMEX's own credentials) whenever they're
                # None — the right behavior for its many "no override"
                # call sites, but WRONG here: if FTMO_MT5_LOGIN is simply
                # unset, that fallback would silently connect this FTMO
                # branch to the PMEX account instead, undetectably (the
                # post-connect verification in connect() only checks
                # against whatever login was actually requested, which
                # would already have silently become PMEX's). Must fail
                # loudly here instead, before connect() is ever called.
                if not config.FTMO_MT5_LOGIN or not config.FTMO_MT5_SERVER:
                    raise MT5ConnectionError(
                        "FTMO account isn't configured yet — add FTMO_MT5_LOGIN, "
                        "FTMO_MT5_PASSWORD, and FTMO_MT5_SERVER to your .env file "
                        "(see .env.example) before selecting FTMO."
                    )
                connect(
                    login=config.FTMO_MT5_LOGIN,
                    password=config.FTMO_MT5_PASSWORD,
                    server=config.FTMO_MT5_SERVER,
                )
            else:
                connect()
        positions = get_open_positions()
        pending_orders = get_pending_orders()
        if selected_exchange == "FTMO":
            account = get_account_summary()
            ftmo_status = fetch_ftmo_status(account)
    except MT5ConnectionError as e:
        st.error(str(e))
        st.stop()

_account_overview_box = st.container(border=True)
with _account_overview_box:
    st.header(":material/dashboard: Account Overview", divider=True)
    st.caption(
        "Real-time snapshot of the connected MT5 account — open positions, "
        "working orders, and (for FTMO) compliance headroom, refreshed "
        "automatically at the interval below."
    )
    _refresh_choice = st.selectbox(
        "Auto-refresh rate",
        ["Off", "1s", "5s", "10s", "30s", "60s"],
        index=1,
        key="positions_refresh_choice",
        help="How often Open Positions / Pending Orders / the Positions "
        "Dashboard below refetch live prices from MT5. 1s keeps prices as "
        "current as possible; slower settings trade some freshness for "
        "fewer local API calls.",
    )


def _render_live_positions(selected_exchange: str) -> None:
    """Self-contained live view: fetches its own positions/pending
    orders/account fresh on every call — both the initial render and
    every periodic st.fragment(run_every=...) tick — using the MT5
    connection the top-level connect() above already established.
    Deliberately does NOT call connect() itself: MT5 supports exactly
    one connection per process, and repeated mt5.initialize() calls
    trigger a real server-side re-auth each time (confirmed live against
    the real FTMO terminal's own log) — polling every few seconds would
    hammer that unnecessarily when the existing connection is already
    live and simply reading fresh data doesn't need it.

    Deliberately independent of the top-level positions/account/
    ftmo_status variables that feed Portfolio Suggestion/execution below
    — this is a display-only section, so a
    periodic refetch here can never cause those higher-stakes sections to
    act on a stale-vs-fresh data mismatch. Renders inside the "Account
    Overview" bordered container established above — a fragment's
    periodic reruns only refill its own slot in the page, so it stays
    nested there across every tick, not just the first render."""
    if st.session_state.pop("_close_position_just_opened", False):
        # A row's "✕" button just fired (see _open_close_position_confirm)
        # — this function runs inside a run_every fragment, and a widget
        # interaction INSIDE a fragment only triggers a FRAGMENT-scoped
        # rerun by default, not a full-app one, but the `if session_
        # state[...]: _show_close_position_confirm_dialog()` check that
        # actually opens the dialog lives at the top level of the script,
        # outside this fragment — without forcing a full rerun here, that
        # check would never run again after the click and the dialog
        # would silently never open (same real bug, and same fix, as
        # _open_watchlist_detail's own docstring documents for the
        # watchlist tiles). Checked before the live data fetch below so
        # the about-to-be-discarded fetch on this tick is skipped
        # entirely. Popped, not just read, so this fires exactly once per
        # click, never again on a later refresh tick while the dialog
        # stays open.
        st.rerun()
    if st.session_state.pop("_pos_table_clear_selection", False):
        # Clears the dataframe's own row-selection state left over from
        # the click that just opened the confirm dialog (see below) — a
        # widget's session_state entry can't be reassigned in the SAME
        # run that already instantiated it (Streamlit raises
        # StreamlitAPIException: "cannot be modified after the widget...
        # is instantiated" — hit this live), so the reset has to happen
        # here, before pos_table_select's own st.dataframe call below
        # creates it fresh on this run. Without this, re-clicking the
        # same row later wouldn't fire on_select again, since Streamlit
        # only reruns on a SELECTION CHANGE.
        st.session_state["pos_table_select"] = {"selection": {"rows": []}}

    live_positions: list = []
    live_pending: list = []
    live_closed_trades: list = []
    live_account = None
    live_ftmo_status = None
    if selected_exchange in ("PMEX", "FTMO"):
        try:
            live_positions = get_open_positions()
            live_pending = get_pending_orders()
            live_deals = get_history_deals(datetime(2000, 1, 1))
            live_closed_trades = group_closed_trades(live_deals)
            live_account = get_account_summary()
            if selected_exchange == "FTMO":
                live_ftmo_status = fetch_ftmo_status(live_account, deals=live_deals)
        except MT5ConnectionError as e:
            st.error(f"Live refresh failed: {e}")
            return

    with st.container(border=True):
        st.subheader("Open Positions")
        st.caption(
            f"Live as of {datetime.now().strftime('%H:%M:%S')} — account "
            "balance plus every currently open position, with its live P&L. "
            "Merges what used to be a separate 'Positions Dashboard' "
            "section, since the two were showing the same positions twice."
        )
        # One-shot result banner for whatever the close-position confirm
        # dialog last did — popped so it shows exactly once, right after
        # the dialog closes and this rerun lands here (same pattern as
        # the Apply Suggestion execution-outcome banner elsewhere on this
        # page).
        _close_outcome = st.session_state.pop("_close_position_outcome", None)
        if _close_outcome is not None:
            _outcome_ok, _outcome_msg = _close_outcome
            (st.success if _outcome_ok else st.error)(_outcome_msg)

        if selected_exchange == "PSX":
            st.caption(
                "PSX is a hypothetical, research-only avenue — there's no "
                "live broker connection (K-Trade has no order-placement "
                "API), so there are no real positions or balance to show "
                "here."
            )
        else:
            if live_account is not None:
                b1, b2, b3 = st.columns(3)
                b1.metric("Balance", f"{live_account.balance:,.2f} {live_account.currency}")
                b2.metric("Equity", f"{live_account.equity:,.2f} {live_account.currency}")
                b3.metric("Free Margin", f"{live_account.free_margin:,.2f} {live_account.currency}")

            if live_positions:
                total_pnl = sum(p.profit for p in live_positions)
                winners = [p for p in live_positions if p.profit > 0]
                losers = [p for p in live_positions if p.profit < 0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Open Positions", len(live_positions))
                m2.metric("Floating P&L", f"{total_pnl:,.2f}")
                m3.metric("Winning", len(winners))
                m4.metric("Losing", len(losers))

                # Genuinely the SAME widget the Pending Orders table below
                # uses (st.dataframe, same construction: a plain
                # pd.DataFrame, hide_index=True, zero custom CSS) — not an
                # approximation. That's the only way to get truly
                # identical font/border/spacing/header rendering, since
                # st.dataframe draws its grid on a <canvas>, not styleable
                # HTML text. Trade-off: it can't hold a real button, so
                # closing works by clicking a row (single-row selection,
                # which shows Streamlit's own built-in checkbox indicator
                # on that row) rather than a dedicated "✕" — the click
                # immediately opens the same reconfirmation dialog used
                # elsewhere, and the selection is reset right after (via
                # the _pos_table_clear_selection flag handled at the top
                # of this function) so the same row can be clicked again.
                _pos_df = pd.DataFrame(
                    [
                        {
                            "Symbol": p.symbol,
                            "Side": p.side,
                            "Volume": p.volume,
                            "Open": p.price_open,
                            "Current Price": p.price_current,
                            "Stop": p.sl if p.sl is not None else "—",
                            "P&L": p.profit,
                        }
                        for p in live_positions
                    ]
                )
                st.caption("Click a row's checkbox to select it and close that position.")
                _pos_event = st.dataframe(
                    _pos_df.style.map(_color_pnl, subset=["P&L"]),
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="pos_table_select",
                )
                _pos_selected_rows = (
                    _pos_event.selection.rows if _pos_event is not None else []
                )
                if _pos_selected_rows:
                    _sel_pos = live_positions[_pos_selected_rows[0]]
                    st.session_state["_pos_table_clear_selection"] = True
                    _open_close_position_confirm(
                        _sel_pos.ticket,
                        _sel_pos.symbol,
                        _sel_pos.side,
                        _sel_pos.volume,
                        _sel_pos.price_open,
                        _sel_pos.profit,
                    )
                    st.rerun()
            else:
                st.write("No open positions.")

    if selected_exchange in ("PMEX", "FTMO") and live_pending:
        with st.container(border=True):
            st.subheader("Pending Orders")
            st.caption(
                "Working limit/stop orders not yet filled — these carry no "
                "P&L and don't show up in Open Positions until price "
                "actually reaches them. Current Price is fetched live per "
                "symbol so you can see how far each order still is from "
                "triggering."
            )
            pending_df = pd.DataFrame(
                [
                    {
                        "Symbol": o.symbol,
                        "Type": o.order_type,
                        "Volume": o.volume,
                        "Trigger Price": o.price_open,
                        "Current Price": get_current_price(o.symbol) or "—",
                        "Stop": o.sl if o.sl is not None else "—",
                        "Target": o.tp if o.tp is not None else "—",
                    }
                    for o in live_pending
                ]
            )
            st.dataframe(pending_df, hide_index=True)

    if selected_exchange == "FTMO" and live_ftmo_status is not None:
        with st.container(border=True):
            st.subheader("FTMO Compliance Status")
            st.caption(
                "Best-effort reconstruction from this account's own real "
                "MT5 trade history — NOT a certified mirror of FTMO's "
                "internal ledger. Cross-check against FTMO's own dashboard "
                "before relying on this near a hard limit."
            )
            heat_col1, heat_col2, heat_col3 = st.columns(3)
            heat_col1.metric(
                "Daily-Loss Headroom", f"{live_ftmo_status.daily_loss_headroom_pct:.2f}%",
                help="Remaining room before today's 3% max-daily-loss rule breaches.",
            )
            heat_col2.metric(
                "Trailing Max-Loss Headroom", f"{live_ftmo_status.max_loss_headroom_pct:.2f}%",
                help="Remaining room before the 10% trailing max-loss floor breaches.",
            )
            if live_ftmo_status.best_day_rule_pct is not None:
                heat_col3.metric(
                    "Best Day Rule",
                    f"{live_ftmo_status.best_day_rule_pct:.1f}%",
                    help="Single best day's profit as a % of total profit — must stay under 50%.",
                )
            else:
                heat_col3.metric(
                    "Best Day Rule", "n/a", help="No positive trading day on record yet"
                )

    if selected_exchange in ("PMEX", "FTMO"):
        with st.expander("Trade History", icon=":material/history:", expanded=False):
            st.caption(
                "Real closed trades from this account's own MT5 history, most "
                "recent first. **Net P&L** is the true realized result (every "
                "leg's profit, swap, and commission combined) — what actually "
                "moved the account balance. **Gross P&L** and **Change %** "
                "deliberately match the MT5 terminal's own \"Profit\"/\"Change\" "
                "columns instead (raw instrument profit only, before costs, as "
                "a % of the position's own notional value) — the two P&L "
                "columns are expected to differ by the position's total "
                "commission+swap, not a bug."
            )
            if live_closed_trades:
                realized_pnl = sum(t.profit for t in live_closed_trades)
                won = [t for t in live_closed_trades if t.profit > 0]
                lost = [t for t in live_closed_trades if t.profit < 0]
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Closed Trades", len(live_closed_trades))
                h2.metric("Realized P&L", f"{realized_pnl:,.2f}")
                h3.metric("Won", len(won))
                h4.metric("Lost", len(lost))

                # One get_contract_spec() call per DISTINCT symbol this
                # render, not per trade — several closed trades commonly
                # share a symbol (e.g. two AUDUSD round-trips), and this
                # is a live MT5 round-trip each time.
                spec_cache: dict[str, object] = {}
                history_rows = []
                for t in live_closed_trades:
                    if t.symbol not in spec_cache:
                        spec_cache[t.symbol] = get_contract_spec(t.symbol)
                    spec = spec_cache[t.symbol]
                    # Same formula MT5's own terminal uses for its "Change"
                    # column (confirmed live against this account's real
                    # trade history to 2 decimal places): raw (gross)
                    # profit as a % of the position's own notional value
                    # at entry. None (blank in the table) when this
                    # symbol's contract spec isn't available right now,
                    # rather than a fabricated/misleading 0%.
                    notional = t.open_price * t.volume * spec.trade_contract_size if spec is not None else 0.0
                    pct_change = (t.gross_profit / notional * 100) if notional else None
                    history_rows.append(
                        {
                            "Symbol": t.symbol,
                            "Side": t.side,
                            "Volume": t.volume,
                            "Open Price": t.open_price,
                            "Close Price": t.close_price,
                            "Opened At": t.opened_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "Closed At": t.closed_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "Duration": str(t.duration),
                            "Gross P&L": t.gross_profit,
                            "Net P&L": t.profit,
                            "Change %": pct_change,
                        }
                    )
                history_df = pd.DataFrame(history_rows)

                st.dataframe(
                    history_df.style
                    .map(_color_pnl, subset=["Gross P&L", "Net P&L", "Change %"])
                    .format({"Change %": "{:.2f}%"}, na_rep="—"),
                    hide_index=True,
                )
            else:
                st.write("No closed trades on record yet.")


def _open_close_position_confirm(
    ticket: int, symbol: str, side: str, volume: float, price_open: float, profit: float
) -> None:
    """st.button's on_click callback for a position row's close ("✕")
    button — just records which ticket was clicked, matching this app's
    existing "click records intent, a separate @st.dialog does the real
    work" pattern (see _open_watchlist_detail). The actual close only
    ever happens after explicit reconfirmation in
    _show_close_position_confirm_dialog below, never directly from this
    click — these snapshot values are shown immediately while the dialog
    opens, but the dialog itself re-fetches the live position fresh
    before actually closing anything, since price/P&L can move (or the
    position can already be closed via its own SL/TP) in the time
    between this click and the user confirming.

    Also sets a one-shot "just opened" flag — real bug found live
    2026-09-08 (same class as _open_watchlist_detail's own documented
    one): this button lives inside a run_every fragment, and a widget
    interaction INSIDE a fragment only triggers a FRAGMENT-scoped rerun
    by default, not a full-app one, but the `if session_state[...]:
    _show_close_position_confirm_dialog()` check that actually opens the
    dialog lives at the top level of the script, outside this fragment —
    without forcing a full rerun (see _render_live_positions's own
    handling of this exact flag), that check never ran again after the
    click and the dialog silently never opened, which is exactly why the
    "✕" button did nothing."""
    st.session_state["_close_position_ticket"] = ticket
    st.session_state["_close_position_symbol"] = symbol
    st.session_state["_close_position_side"] = side
    st.session_state["_close_position_volume"] = volume
    st.session_state["_close_position_price_open"] = price_open
    st.session_state["_close_position_profit"] = profit
    st.session_state["_close_position_just_opened"] = True


def _clear_close_position_confirm() -> None:
    """Shared by the dialog's own Cancel button AND @st.dialog's
    on_dismiss hook — same real-bug rationale as
    _clear_watchlist_detail's own docstring: without on_dismiss wired up,
    dismissing via the dialog's native X/Esc/outside-click would leave
    _close_position_ticket stuck set forever, since only an explicit
    button click would otherwise ever clear it."""
    st.session_state["_close_position_ticket"] = None


@st.dialog("Confirm Close Position", on_dismiss=_clear_close_position_confirm)
def _show_close_position_confirm_dialog() -> None:
    """Reconfirmation dialog for a manual position close — direct user
    request. Re-fetches the position fresh by ticket rather than trusting
    the snapshot the row's own button click captured, since price/P&L
    move continuously and the position may have already closed on its
    own (hit its real SL/TP) in the time between the click and this
    confirmation — closing a stale snapshot's "current price" would be
    misleading, and closing a ticket that no longer exists would just
    fail confusingly instead of explaining what actually happened."""
    ticket = st.session_state.get("_close_position_ticket")
    if not ticket:
        return
    symbol = st.session_state.get("_close_position_symbol")

    try:
        current_positions = get_open_positions()
    except MT5ConnectionError as e:
        st.error(f"Could not refresh live position data: {e}")
        if st.button("Close"):
            _clear_close_position_confirm()
            st.rerun()
        return

    matching = next((p for p in current_positions if p.ticket == ticket), None)
    if matching is None:
        st.warning(
            f"This position (ticket {ticket}, {symbol}) is no longer open — "
            "it likely already closed via its own stop-loss or take-profit."
        )
        if st.button("Close"):
            _clear_close_position_confirm()
            st.rerun()
        return

    st.markdown(
        f"**{matching.symbol}** — {matching.side} {matching.volume:g} lots @ "
        f"{matching.price_open:g}, current price {matching.price_current:g}"
    )
    st.markdown(
        f"Current P&L: <span style='{_color_pnl(matching.profit)}'>{matching.profit:+.2f}</span>",
        unsafe_allow_html=True,
    )
    st.warning(
        "This closes the position at the current market price immediately. "
        "This cannot be undone."
    )

    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button("Yes, close it", type="primary", icon=":material/close:"):
            try:
                result = close_position(matching)
            except MT5ConnectionError as e:
                result = OrderResult(False, None, str(e), None)
            st.session_state["_close_position_outcome"] = (
                (True, f"{matching.symbol} (ticket {matching.ticket}) closed successfully.")
                if result.success
                else (False, f"Failed to close {matching.symbol}: {result.comment}")
            )
            _clear_close_position_confirm()
            st.rerun()
    with cancel_col:
        if st.button("Cancel"):
            _clear_close_position_confirm()
            st.rerun()


if st.session_state.get("_close_position_ticket"):
    _show_close_position_confirm_dialog()


# Broker-provided top-level symbol-path categories (data.mt5_source.
# get_symbol_category) collapsed into the handful of asset classes this
# account's own pool actually spans — same grouping FTMO's own commission
# lookup already relies on (ai/ftmo_suggest.py::_ftmo_commission_pct_
# round_turn), reused here for consistency rather than inventing a
# second, possibly-diverging classification. Order here is also the
# grid's own display order.
_WATCHLIST_CATEGORY_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Forex", ("Forex", "Exotics")),
    ("Metals", ("Metals CFD",)),
    ("Indices", ("Cash CFD", "Cash II CFD")),
    ("Commodities", ("Cash III CFD", "Commodities")),
    ("Agriculture", ("Agriculture",)),
]


def _watchlist_group_label(raw_category: str) -> str:
    if raw_category.startswith("Crypto"):
        return "Crypto"
    for label, raw_values in _WATCHLIST_CATEGORY_GROUPS:
        if raw_category in raw_values:
            return label
    # Whatever's left in this account's pool that isn't forex/metals/
    # indices/commodities/agriculture/crypto is single-stock CFDs (e.g.
    # MSFT, NVDA) — "Equity" names that accurately instead of a catch-all
    # "Other".
    return "Equity"


@st.cache_data(ttl=3600)
def _cached_symbol_category(symbol: str) -> str:
    """A symbol's broker category never changes mid-session, so this is
    fetched once per symbol (per hour, generously) rather than every
    fast watchlist refresh tick — get_symbol_category itself is cheap
    (a single symbol_info() call, no tick fetch), but there's no reason
    to repeat even that ~18-21 times a second when it can't change."""
    return get_symbol_category(symbol)


def _watchlist_decimals(price: float) -> int:
    """Adaptive decimal precision for the watchlist grid's display only
    (not used for any real sizing/risk math, which always works off the
    live MT5 float directly) — this account's instruments span wildly
    different price magnitudes (EURUSD ~1.1, XAUUSD ~2000, BTCUSD
    ~60000+), and MT5's own per-symbol `digits` isn't fetched here to
    avoid an extra round-trip per symbol on every fast refresh tick."""
    if price >= 1000:
        return 2
    if price >= 10:
        return 3
    return 5


def _open_watchlist_detail(symbol: str, exchange: str, description: str) -> None:
    """st.button's on_click callback for a watchlist tile — just records
    which symbol was clicked; the actual dialog is opened further below
    by checking this same session_state key, matching this app's
    existing "Confirm Trade Execution" dialog's own open/close pattern.

    Also sets a one-shot "just opened" flag — see _render_watchlist_grid's
    own use of it for why: this button lives inside a run_every fragment,
    and a widget interaction INSIDE a fragment only triggers a FRAGMENT-
    scoped rerun by default, not a full-app one, but the `if session_
    state[...]: _show_watchlist_detail_dialog()` check that actually
    opens the dialog lives at the top level of the script, outside this
    fragment — without forcing a full rerun, that check would never run
    again after the click and the dialog would silently never open
    (confirmed live: this was a real bug, not a hypothetical)."""
    st.session_state["_watchlist_detail_symbol"] = symbol
    st.session_state["_watchlist_detail_exchange"] = exchange
    st.session_state["_watchlist_detail_description"] = description
    st.session_state["_watchlist_detail_just_opened"] = True


_WATCHLIST_TILE_STYLE = """
<style>
[class*="st-key-wl_tile_"] button {
    padding: 3px 10px;
    border-radius: 7px;
    font-size: clamp(0.68rem, 0.6rem + 0.25vw, 0.85rem);
    font-variant-numeric: tabular-nums;
    border: 1px solid rgba(128, 128, 128, 0.3);
    border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
    background: transparent;
    transition: transform 0.15s ease;
}
[class*="st-key-wl_tile_"] button:hover {
    transform: translateY(-1px);
    cursor: pointer;
}
/* A real price move gets a fresh DOM node (see the key_dir/gen comment
   at this tile's own build site), which is what lets a CSS `animation`
   (unlike `transition`) play correctly from a clean start every time.
   Graceful pulse, not a flash: rises to its peak color partway through
   rather than snapping to full intensity on frame 0, then eases back to
   the tile's normal neutral border — `forwards` holds that neutral end
   state after the animation finishes, since a symbol that goes quiet
   for a while should visually settle, not stay permanently tinted. */
[class*="st-key-wl_tile_up_"] button {
    animation: wl-pulse-up 1.6s ease-in-out forwards;
}
[class*="st-key-wl_tile_down_"] button {
    animation: wl-pulse-down 1.6s ease-in-out forwards;
}
@keyframes wl-pulse-up {
    0% { background-color: transparent; border-color: rgba(128, 128, 128, 0.3); }
    20% { background-color: rgba(22, 163, 74, 0.26); border-color: #16a34a; }
    100% { background-color: transparent; border-color: rgba(128, 128, 128, 0.3); }
}
@keyframes wl-pulse-down {
    0% { background-color: transparent; border-color: rgba(128, 128, 128, 0.3); }
    20% { background-color: rgba(220, 38, 38, 0.26); border-color: #dc2626; }
    100% { background-color: transparent; border-color: rgba(128, 128, 128, 0.3); }
}
/* Direct user complaint 2026-09-02: the Execution Clerk panel's own
   tables visually "jerk" — Streamlit just replaces them wholesale on
   every 1-second tick. Each table's own key (see _render_clerk_
   execution_panel's own _clerk_content_gen comment) only actually
   CHANGES when there's genuinely new content to show, which is what
   makes this fade animation replay meaningfully instead of playing
   every single second regardless — same key-remount technique as the
   wl-pulse rules above, applied as a one-shot fade-in instead of a
   repeating pulse. */
[class*="st-key-clerk_imm_table_"],
[class*="st-key-clerk_pending_table_"] {
    animation: clerk-fade-in 0.5s ease-out;
}
@keyframes clerk-fade-in {
    from { opacity: 0.25; }
    to { opacity: 1; }
}
</style>
"""


def _render_watchlist_grid(selected_exchange: str) -> None:
    """Live current-price tile grid for every instrument in this
    account's own MT5 Market Watch, grouped by asset class — a fast,
    whole-pool glance distinct from (and much cheaper than) Portfolio
    Suggestion's deep per-instrument analysis. Each tile is a real
    st.button (not raw HTML) so it's genuinely clickable — opens that
    symbol's live technical-dashboard popup (see
    _show_watchlist_detail_dialog below) — styled via CSS targeting its
    own `st-key-` class (see _WATCHLIST_TILE_STYLE) rather than
    st.button's default chrome.

    The flash-on-real-price-move animation still works despite buttons
    having STABLE identity across reruns (unlike the old raw-HTML
    version, which got a fresh DOM node every tick "for free"): each
    tile's `key` embeds a per-symbol "flash generation" counter that
    only increments on an actual up/down move, so the key — and
    therefore the button's DOM identity — only changes exactly on ticks
    where a real move happened, which is what makes the CSS `animation`
    restart correctly instead of playing once on load and never again
    (or worse, replaying every single tick regardless of whether the
    price moved).

    Previous prices/flash generations live in session_state, keyed per
    exchange so switching the exchange dropdown can never compare
    FTMO's price history against PMEX's (or vice versa) for a symbol
    name that happens to collide."""
    if st.session_state.pop("_watchlist_detail_just_opened", False):
        # A tile's on_click just fired (see _open_watchlist_detail) — a
        # click on a widget INSIDE this fragment only reruns the
        # fragment by default, but opening the dialog needs a real
        # full-app rerun (see that function's own docstring). Popped,
        # not just read, so this fires exactly once per click rather
        # than on every subsequent tick while the dialog stays open —
        # that would otherwise re-trigger st.rerun() every refresh
        # interval forever, an infinite-rerun loop.
        st.rerun()

    if st.session_state.get("_watchlist_detail_symbol"):
        # Confirmed live: interacting with ANY widget inside the "Asset
        # health" dialog — even just its own auto-refresh tick — forces
        # this ENTIRE fragment (and every other one on the page) to
        # re-execute too, despite st.dialog's own docs claiming reruns
        # stay scoped to the dialog. That made this grid's own full
        # get_market_watch() fetch across all 17-21 symbols + rebuild
        # the single biggest cost in every one of the dialog's own
        # refresh ticks — real lag the dialog-side optimizations alone
        # couldn't fix, since they never touched this fragment at all.
        # Skipping the expensive part here while a dialog is open (the
        # modal covers this grid visually anyway) removes that cost from
        # every forced re-execution; this resumes normal full-grid
        # refreshing automatically the instant the dialog closes (this
        # check goes False again on the very next tick).
        st.caption("Paused while an asset's details are open — resumes when you close it.")
        return

    try:
        assets = sorted(get_market_watch(), key=lambda a: a.symbol)
    except MT5ConnectionError as e:
        st.error(f"Live watchlist failed: {e}")
        return
    if not assets:
        st.write("No instruments visible in this account's MT5 Market Watch.")
        return

    prev_prices: dict[str, float] = st.session_state.setdefault(
        f"_watchlist_prev_prices_{selected_exchange}", {}
    )
    flash_gens: dict[str, int] = st.session_state.setdefault(
        f"_watchlist_flash_gen_{selected_exchange}", {}
    )
    # The direction AT THE TIME gen last bumped — deliberately separate
    # from the per-tick `direction` computed below. A symbol's price is
    # "flat" (unchanged) on almost every tick by definition, including
    # the tick immediately following a real move — if the key's own
    # direction segment tracked that raw per-tick value, the tile would
    # get a fresh DOM node TWICE per real move (once for the move
    # itself, once more reverting to "flat" the very next tick) instead
    # of once, and the CSS class would flip straight back to neutral
    # rather than fading there — confirmed live as a real, unintended
    # contributor to the reported jerkiness. Keying off the last REAL
    # direction instead keeps the tile's identity (and therefore its
    # fade-back-to-neutral keyframe, see _WATCHLIST_TILE_STYLE) stable
    # and correct across every tick in between.
    flash_last_dir: dict[str, str] = st.session_state.setdefault(
        f"_watchlist_flash_dir_{selected_exchange}", {}
    )

    # Grouped by asset class (see _WATCHLIST_CATEGORY_GROUPS) so a mixed
    # forex/metals/indices/crypto/agriculture/commodities pool the size of
    # this account's own reads as several short, scannable rows instead
    # of one long undifferentiated wall of 17-21 tiles.
    groups: dict[str, list[tuple]] = {}
    for a in assets:
        label = _watchlist_group_label(_cached_symbol_category(a.symbol))
        mid = (a.bid + a.ask) / 2
        prev = prev_prices.get(a.symbol)
        if prev is None or mid == prev:
            arrow = "•"
        else:
            moved_dir, arrow = ("up", "▲") if mid > prev else ("down", "▼")
            flash_gens[a.symbol] = flash_gens.get(a.symbol, 0) + 1
            flash_last_dir[a.symbol] = moved_dir
        prev_prices[a.symbol] = mid
        dp = _watchlist_decimals(mid)
        key_dir = flash_last_dir.get(a.symbol, "flat")
        groups.setdefault(label, []).append((a, mid, arrow, dp, key_dir, flash_gens.get(a.symbol, 0)))

    st.html(_WATCHLIST_TILE_STYLE)

    # Fixed display order, skipping any group this pool has nothing in,
    # rather than an alphabetical or first-seen order that would reshuffle
    # as symbols are added/removed from Market Watch.
    ordered_labels = ["Forex", "Metals", "Crypto", "Indices", "Commodities", "Agriculture", "Equity"]
    for label in ordered_labels:
        tiles = groups.pop(label, None)
        if not tiles:
            continue
        st.caption(f"**{label}** ({len(tiles)})")
        with st.container(horizontal=True, gap="small"):
            for a, mid, arrow, dp, key_dir, gen in tiles:
                st.button(
                    f"{a.symbol}  {mid:.{dp}f} {arrow}",
                    key=f"wl_tile_{key_dir}_{a.symbol}_{gen}",
                    on_click=_open_watchlist_detail,
                    args=(a.symbol, selected_exchange, a.description),
                )

    st.caption(
        f"Live · {len(assets)} instruments · updated {datetime.now():%H:%M:%S} "
        "· click a tile for its full live technical dashboard"
    )


if _refresh_choice != "Off":
    _render_live_positions = st.fragment(run_every=_refresh_choice)(_render_live_positions)
    _render_watchlist_grid = st.fragment(run_every=_refresh_choice)(_render_watchlist_grid)

with _account_overview_box:
    _render_live_positions(selected_exchange)

st.header(":material/monitoring: Watchlist", divider=True)
# Real bug found live 2026-08-21: st.expander's own on_change="rerun" state
# tracking does NOT survive a full-page rerun that a DIALOG triggers (e.g.
# clicking "Close" on the Asset health popup) — confirmed via a controlled
# AppTest repro. Streamlit's own expander deserializer is `ui_value if
# ui_value is not None else self.expanded` (elements/layouts.py); on a
# dialog-triggered rerun `ui_value` arrives as None (the same anomalous-
# rerun behavior already confirmed elsewhere in this dialog — see
# _render_watchlist_grid's own docstring), so it silently falls back to
# the LITERAL `expanded=` argument passed at the call site. Since that was
# a hardcoded `False`, the section would snap shut right after Close, which
# then made the grid's own st.fragment unreachable (never called again on
# any later rerun) — the likely source of the reported post-close
# lag/crash, an orphaned fragment whose container stopped being rendered
# while its own auto-refresh timer stayed armed.
#
# Fixed by feeding our OWN last-known-good state back in as the `expanded`
# fallback instead of a hardcoded literal, so a dialog-triggered rerun
# (ui_value=None) resolves to "stay open" rather than "reset to closed".
_watchlist_open_key = "_watchlist_section_open"
_watchlist_expander = st.expander(
    "Show live prices",
    icon=":material/query_stats:",
    expanded=st.session_state.get(_watchlist_open_key, False),
    on_change="rerun",
)
if _watchlist_expander.open is not None:
    st.session_state[_watchlist_open_key] = _watchlist_expander.open
if _watchlist_expander.open:
    with _watchlist_expander:
        if selected_exchange in ("PMEX", "FTMO"):
            st.caption(
                "Live current price for every instrument in this account's "
                "MT5 Market Watch, grouped by asset class — flashes green/red "
                "on an actual price move, refreshed at the same rate chosen "
                "above. Click any tile for its full live technical dashboard."
            )
            _render_watchlist_grid(selected_exchange)
        else:
            st.caption("No live MT5 feed for PSX — this grid is PMEX/FTMO only.")


def _list_researcher_symbols() -> list[str]:
    """Every symbol with at least one saved Researcher report, sorted
    alphabetically. Report filenames are "<SYMBOL>_<YYYY-MM-DD>_
    <HHMMSS>.md" (see ai.researcher.save_research_report) — the symbol
    itself never contains an underscore, so splitting on the FIRST one
    reliably recovers it even though the timestamp portion has its own
    underscore between the date and time."""
    records_dir = Path(config.RESEARCHER_RECORDS_DIR)
    if not records_dir.exists():
        return []
    symbols = {path.name.split("_", 1)[0] for path in records_dir.glob("*.md")}
    return sorted(symbols)


@st.cache_resource
def _researcher_timer_state() -> dict:
    """Process-lifetime holder so _fire_scheduled_job_once never spawns a
    redundant second in-process thread on back-to-back fragment ticks —
    same shape/reasoning as _clerk_timer_state(), except Researcher's
    actual due-ness decision is delegated entirely to ai.researcher.
    is_researcher_due (already built, already tested, driven by real
    persisted state) rather than a second, in-memory-only countdown —
    this dict holds nothing but the "thread" key _fire_scheduled_job_once
    itself manages."""
    return {}


def _run_researcher_check_with_timeout() -> None:
    """The actual work _fire_scheduled_job_once runs on a background
    thread for Researcher's in-app periodic trigger — same reasoning as
    _run_clerk_check_with_timeout: run_researcher_check does not self-
    bound its own runtime (every other caller wraps it in run_with_timeout
    itself), so this does too, just from the background thread."""
    timed_out = object()
    result = run_with_timeout(
        run_researcher_check, config.RESEARCHER_RUN_TIMEOUT_SECONDS, default=timed_out, catch_exceptions=False,
    )
    if result is timed_out:
        logger.error(
            "Periodic Researcher check timed out after %.0f minutes",
            config.RESEARCHER_RUN_TIMEOUT_SECONDS / 60,
        )


# Read-only DISPLAY (Phase 1 of the Researcher agent — see ai/
# researcher.py's own module docstring): what's rendered below only
# reads and shows whatever Researcher has already saved to disk, for a
# human to read. It does not feed into, and is not fed by, either the
# Mega Session's or Clerk's own decisions — that "Phase 2" wiring is a
# deliberately separate, not-yet-built step. The TRIGGER that actually
# produces those saved reports is a different matter, fixed here
# (direct user report 2026-09-14, "the researcher doesn't auto run after
# 1 hour"): ai/researcher.py's own run_researcher_check was fully built
# but never wired into anything that calls it automatically — no Windows
# Scheduled Task exists for ANY of this project's three agents (direct,
# deliberate user choice recorded elsewhere in this file: "webapp-only,
# no independent Clerk task"), so the in-app st.fragment(run_every=...)
# + _fire_scheduled_job_once pattern Mega Session/Clerk already use is
# the actual, sole trigger mechanism in this app — Researcher simply
# never had its own copy of that wiring. Every prior "hourly run" this
# session was a manual python -c invocation, never a real automatic one.
def _render_researcher_panel() -> None:
    now_utc = datetime.now(timezone.utc)
    if read_researcher_enabled():
        # _fire_scheduled_job_once's own acquire_lock call (cross-process,
        # PID-liveness-aware) is what actually prevents two overlapping
        # runs — this is purely the "is it time yet" decision, backed by
        # is_researcher_due's own real persisted state (the SAME check
        # researcher_job.py's own OS-level path would use, so the in-app
        # trigger and that standalone script can never both think it's
        # simultaneously due and double-fire within the same window).
        if is_researcher_due(now_utc):
            _fire_scheduled_job_once(
                _run_researcher_check_with_timeout,
                _researcher_timer_state(),
                RESEARCHER_LOCK_PATH,
                RESEARCHER_LOCK_STALE_AFTER_SECONDS,
            )

    st.header(":material/travel_explore: Researcher", divider=True)
    _researcher_open_key = "_researcher_section_open"
    _researcher_expander = st.expander(
        "Show research reports",
        icon=":material/travel_explore:",
        expanded=st.session_state.get(_researcher_open_key, False),
        on_change="rerun",
    )
    if _researcher_expander.open is not None:
        st.session_state[_researcher_open_key] = _researcher_expander.open
    if _researcher_expander.open:
        with _researcher_expander:
            st.caption(
                "Independent, local-only per-symbol research (real fetched "
                "headlines + a local model's synthesis and sentiment lean) — "
                "generated on its own schedule, separately from the Mega "
                "Session and Clerk above. Purely for you to read; it isn't "
                "consumed by either agent yet."
            )
            _researcher_state = read_researcher_state()
            _researcher_total_runs = _researcher_state.get("total_runs", 0)
            _researcher_last_attempt = _researcher_state.get("last_attempt_utc")
            if not read_researcher_enabled():
                _next_check_caption = "auto-run disabled."
            else:
                _next_check = next_researcher_check_utc(now_utc)
                _remaining = max(0, int((_next_check - now_utc).total_seconds()))
                if _remaining <= 0:
                    _next_check_caption = "next check due now."
                elif _remaining >= 3600:
                    _rem_hours, _rem_rest = divmod(_remaining, 3600)
                    _rem_minutes = _rem_rest // 60
                    _next_check_caption = f"next check in {_rem_hours}h {_rem_minutes}m."
                else:
                    _rem_minutes, _rem_seconds = divmod(_remaining, 60)
                    _next_check_caption = f"next check in {_rem_minutes}m {_rem_seconds:02d}s."
            if _researcher_last_attempt:
                st.caption(
                    f"Runs so far: {_researcher_total_runs} · last run "
                    f"({_format_last_run(_researcher_last_attempt)}): "
                    f"{_researcher_state.get('last_detail') or _researcher_state.get('last_status', '')} · "
                    f"{_next_check_caption}"
                )
            else:
                st.caption(f"Runs so far: {_researcher_total_runs} · {_next_check_caption}")
            _researcher_symbols = _list_researcher_symbols()
            if not _researcher_symbols:
                st.info(
                    "No research reports yet — run researcher_job.py (or call "
                    "ai.researcher.run_researcher_check() directly) to generate some."
                )
            else:
                for _symbol in _researcher_symbols:
                    _report_text = latest_research_report(_symbol)
                    if not _report_text:
                        continue
                    _sentiment = parse_researcher_sentiment(_report_text)
                    _emoji = _SENTIMENT_EMOJI.get(_sentiment, "❔")
                    # Collapsed by default (expanded=False, not tied to
                    # session state) — direct user request: every symbol's
                    # own dropdown starts closed regardless of whether the
                    # outer section itself is open.
                    with st.expander(f"{_emoji} {_symbol}", expanded=False):
                        st.markdown(_report_text)


# 30s: frequent enough to catch is_researcher_due's own once-daily
# trigger window comfortably (see RESEARCHER_GRACE_MINUTES) without
# re-running every second the way Clerk's own live-countdown-focused
# panel does — Researcher only fires once a day, so that level of tick
# rate would be pure overhead here.
_render_researcher_panel = st.fragment(run_every=30)(_render_researcher_panel)
_SENTIMENT_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}

if selected_exchange == "FTMO":
    _render_researcher_panel()


_SETUP_BADGE_COLORS = {
    "reversal_candidate": "orange",
    "pullback_continuation": "blue",
    "range_fade_candidate": "violet",
    "breakout_watch": "yellow",
    "trend_following": "green",
    # Real gap found on a self-audit re-check: this dict was never
    # updated when grind_continuation (analysis/setup_classifier.py) was
    # added — without an entry here, it fell back to the same gray as
    # "no_clear_setup" below, visually erasing the exact distinction
    # (real direction, just noisy) grind_continuation exists to surface.
    "grind_continuation": "blue",
    # New archetype added 2026-08-22 (see analysis/setup_classifier.py's
    # own module docstring) — same "don't let a new archetype silently
    # fall back to no_clear_setup's own gray" lesson as grind_continuation
    # above; green like trend_following since both are genuine trend
    # evidence, distinguished by their own badge text/tooltip instead
    # (trend_following = a fresh trendline retest, trend_intact = the
    # broader trend itself, no specific trigger active right now).
    "trend_intact": "green",
    # 3 new archetypes added 2026-08-26 (analysis/setup_classifier.py
    # rules 8-10) — same lesson as grind_continuation/trend_intact
    # above, added proactively this time rather than needing a second
    # user catch. in_progress_move is the direct fix for a real
    # complaint ("wakes up late in a sideways market") so it gets an
    # attention-grabbing color; busted_pattern_reversal shares
    # reversal_candidate's own color since it's explicitly a stronger
    # variant of that same signal, firing alongside it.
    "in_progress_move": "red",
    "busted_pattern_reversal": "orange",
    "candlestick_reversal_confirmed": "yellow",
    "no_clear_setup": "gray",
}
_TREND_BADGE_COLORS = {
    "uptrend": "green",
    "trending_up": "green",
    "downtrend": "red",
    "trending_down": "red",
    "flat": "gray",
    "sideways": "gray",
    # choppy_up/choppy_down (analysis/technical.py) mean "real net
    # direction over the window, but reached via a lot of back-and-forth
    # churn" — genuinely distinct from both a clean trend and no
    # direction at all, so distinct colors rather than reusing green/red/
    # gray, which would visually erase that distinction.
    "choppy_up": "blue",
    "choppy_down": "orange",
    # momentum_acceleration (analysis/technical.py, added 2026-08-26) is
    # a SEPARATE short-window signal from market_regime above — violet
    # rather than reusing green/red, so it never reads as if it were
    # just another market_regime value at a glance.
    "accelerating_up": "violet",
    "accelerating_down": "violet",
    "stable": "gray",
}


def _trend_badge(value: str | None) -> None:
    if not value:
        st.badge("Unknown", color="gray")
        return
    st.badge(value.replace("_", " ").title(), color=_TREND_BADGE_COLORS.get(value, "gray"))


def _render_long_term_alignment_banner(analysis: FtmoAssetAnalysis) -> None:
    """Real feature added live 2026-08-22 directly answering the user's
    own reported gap ("the analysis is missing the long/medium term
    trends and can throw us in a fake trend unguarded") — surfaces
    ai.ftmo_suggest.LONG_TERM_ALIGNMENT_SHORT_MESSAGES' one-line read
    prominently, with color coding matching its own three real outcomes:
    a structurally-backed move gets a positive/green treatment, a
    counter-trend spike gets an amber "trade it, but manage it tighter"
    treatment (never red/error — the user's own stated preference is
    that these are genuinely good, tradeable opportunities, not
    something to avoid), anything else (no real backdrop yet, H4 itself
    flat/unavailable) gets a neutral, informational treatment.

    Uses the SHORT message set, not the long AI-prompt formatter — real
    user feedback, live: the long version (multi-sentence reasoning
    meant for a model) was showing up verbatim in this banner, which
    read as "a long complex message" instead of a glanceable dashboard
    line. See ai.ftmo_suggest's own LONG_TERM_ALIGNMENT_SHORT_MESSAGES
    comment for the messaging standard this and future UI text on this
    page follow. Classifies once (real bug caught on a self-audit
    re-check: an earlier version called classify_long_term_alignment
    here for the color AND separately via format_long_term_alignment_
    short for the message text — same pure function, same input, called
    twice for no reason) and reads the message straight from the shared
    dict with the state already in hand."""
    state, _short, _agreeing, _opposing = classify_long_term_alignment(analysis)
    message = LONG_TERM_ALIGNMENT_SHORT_MESSAGES[state]
    if state == "structurally_backed":
        st.success(message, icon=":material/verified:")
    elif state == "counter_trend_spike":
        st.warning(message, icon=":material/bolt:")
    else:
        st.info(message, icon=":material/info:")


def _format_generic_trade_cost(analysis: FtmoAssetAnalysis) -> str:
    """PMEX-side counterpart to format_ftmo_trade_cost — real spread/
    swap only, no commission line, since FTMO's confirmed commission
    schedule doesn't apply to a different broker/account."""
    tc = analysis.trade_cost
    if tc is None:
        return "Trade cost unavailable for this symbol."
    parts = [f"Spread {tc.spread_pct_of_price:.4f}% of price"]
    if tc.swap_long_pct_per_day is not None:
        parts.append(f"swap {tc.swap_long_pct_per_day:+.4f}%/day held long")
    if tc.swap_short_pct_per_day is not None:
        parts.append(f"{tc.swap_short_pct_per_day:+.4f}%/day held short")
    return " · ".join(parts)


_ASSET_HEALTH_ACCENT = "#f0a020"  # amber — the one deliberately Bloomberg-coded accent in this dialog

_ASSET_HEALTH_STYLE = f"""
<style>
[data-testid="stDialog"] [data-testid="stMetricValue"],
[data-testid="stDialog"] [data-testid="stMetricLabel"] {{
    font-variant-numeric: tabular-nums;
}}
[data-testid="stDialog"] [data-testid="stMetricValue"] {{
    font-family: 'SF Mono', Consolas, Menlo, monospace;
}}
[data-testid="stDialog"] .wl-ticker {{
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    column-gap: clamp(10px, 2vw, 18px);
    row-gap: 2px;
    padding: clamp(8px, 1.2vw, 14px) clamp(10px, 1.5vw, 16px);
    border-radius: 8px;
    border: 1px solid {_ASSET_HEALTH_ACCENT}59;
    background: linear-gradient(180deg, {_ASSET_HEALTH_ACCENT}14, transparent);
    margin-bottom: 10px;
    /* This element is a brand-new DOM node every refresh (needed so the
       glow below reliably replays — see the watchlist tiles' own design
       note on why `transition` can't do this on a fresh node), so a
       soft glow ring here — present at full strength on frame 0, eased
       away to nothing — reads as "this just refreshed" on every single
       tick, not just a real price move, without a hard color pop. */
    animation: wl-ticker-glow 1s ease-out;
}}
[data-testid="stDialog"] .wl-ticker.up {{ animation: wl-ticker-glow-up 1.5s ease-out; }}
[data-testid="stDialog"] .wl-ticker.down {{ animation: wl-ticker-glow-down 1.5s ease-out; }}
@keyframes wl-ticker-glow {{
    0% {{ box-shadow: 0 0 0 3px {_ASSET_HEALTH_ACCENT}40; }}
    100% {{ box-shadow: 0 0 0 0 transparent; }}
}}
@keyframes wl-ticker-glow-up {{
    0% {{ box-shadow: 0 0 0 3px #16a34a55; }}
    100% {{ box-shadow: 0 0 0 0 transparent; }}
}}
@keyframes wl-ticker-glow-down {{
    0% {{ box-shadow: 0 0 0 3px #dc262655; }}
    100% {{ box-shadow: 0 0 0 0 transparent; }}
}}
[data-testid="stDialog"] .wl-ticker-symbol {{
    font-weight: 700;
    letter-spacing: 0.5px;
    font-size: clamp(0.8rem, 0.7rem + 0.3vw, 0.95rem);
    opacity: 0.85;
}}
[data-testid="stDialog"] .wl-ticker-price {{
    font-family: 'SF Mono', Consolas, Menlo, monospace;
    font-size: clamp(1.35rem, 1.05rem + 1.4vw, 2rem);
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}}
[data-testid="stDialog"] .wl-ticker-price.up {{ color: #16a34a; }}
[data-testid="stDialog"] .wl-ticker-price.down {{ color: #dc2626; }}
[data-testid="stDialog"] .wl-ticker-meta {{
    font-family: 'SF Mono', Consolas, Menlo, monospace;
    font-variant-numeric: tabular-nums;
    font-size: clamp(0.68rem, 0.6rem + 0.25vw, 0.8rem);
    opacity: 0.7;
}}
[data-testid="stDialog"] .wl-section-hdr {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    opacity: 0.65;
    margin: 12px 0 5px 0;
    padding-bottom: 3px;
    border-bottom: 1px solid {_ASSET_HEALTH_ACCENT}59;
}}
</style>
"""


def _render_ticker(symbol: str, mid: float, bid: float, ask: float, spread: float, dp: int, direction: str) -> None:
    # `direction` on the OUTER container too (not just the price span) —
    # that's what the graceful glow-ring animation targets (see
    # _ASSET_HEALTH_STYLE's own note on why it's a ring around the whole
    # ticker rather than a hard color pop on the number itself).
    st.html(
        f'<div class="wl-ticker {direction}">'
        f'<span class="wl-ticker-symbol">{symbol}</span>'
        f'<span class="wl-ticker-price {direction}">{mid:.{dp}f}</span>'
        f'<span class="wl-ticker-meta">BID {bid:.{dp}f} &nbsp;·&nbsp; ASK {ask:.{dp}f} '
        f"&nbsp;·&nbsp; SPREAD {spread:.{dp}f}</span>"
        f"</div>"
    )


def _section_header(text: str) -> None:
    st.html(f'<div class="wl-section-hdr">{text}</div>')


def _render_rsi_gauge(rsi: float | None, label: str) -> None:
    """A real gauge (not a number in a sentence) so overbought/oversold
    reads at a glance — colored zones match this project's own
    green=favorable/red=unfavorable convention (see the watchlist tiles'
    flash colors), not a generic rainbow. The needle/number itself is
    also colored by zone (green <30, amber neutral, red >70) rather than
    a fixed neutral blue, so the single most important read (is this
    extended?) doesn't need the zone bands to be decoded separately."""
    if rsi is None:
        st.caption(f"{label} RSI — not enough history yet")
        return
    bar_color = "#16a34a" if rsi < 30 else "#dc2626" if rsi > 70 else _ASSET_HEALTH_ACCENT
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rsi,
            title={"text": f"{label} RSI", "font": {"size": 12}},
            number={"font": {"size": 24, "color": bar_color}},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [30, 70], "tickfont": {"size": 9}},
                "bar": {"color": bar_color, "thickness": 0.35},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(22, 163, 74, 0.22)"},
                    {"range": [30, 70], "color": "rgba(128, 128, 128, 0.10)"},
                    {"range": [70, 100], "color": "rgba(220, 38, 38, 0.22)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=145,
        margin=dict(l=12, r=12, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        # Smoothly tweens the needle/number/bar between successive
        # Plotly.react() updates (what the stable key below enables)
        # instead of snapping straight to the new value — this is the
        # actual "make it feel smooth, not jerky" lever; the stable key
        # alone only stops the chart from being torn down and rebuilt,
        # it doesn't by itself animate the value change.
        transition={"duration": 500, "easing": "cubic-in-out"},
    )
    # A stable key (not the default auto-generated one, which is a hash
    # of the figure's own contents — different every tick since the RSI
    # value itself changes) is what lets the frontend update this
    # chart's data in place instead of unmounting/remounting the whole
    # Plotly widget on every refresh. Confirmed live as a real,
    # meaningful contributor to this dialog's reported jerky/delayed
    # updates: a full Plotly remount is comparatively expensive, and
    # was happening on literally every tick for all 3 charts here.
    st.plotly_chart(fig, width="stretch", key=f"wl_gauge_{label}")


def _render_sr_ladder(current_price: float, mn1_sr, d1_sr, h4_sr, h1_sr) -> None:
    """One combined horizontal price ladder for ALL FOUR timeframes' real
    support/resistance levels — Monthly on the top row (biggest
    markers) down to H1 on the bottom row (smallest), current price as
    an amber diamond. Doubles as an implicit multi-timeframe-confluence
    view: when markers from different timeframes visually cluster at
    the same price, that IS the confluence signal, read directly off
    the chart instead of a separate sentence explaining it. Redesigned
    live 2026-08-22 from an H4/H1-only ladder to all 4 timeframes —
    direct user request ("redesign the charts...with all H1, H4, D1
    and monthly and display combined charts") matching the D1/monthly
    chart-structure computation already added to ai/ftmo_suggest.py."""
    rows = [
        ("Monthly", mn1_sr, 0.45, 3.0),
        ("D1", d1_sr, 0.15, 2.4),
        ("H4", h4_sr, -0.15, 1.8),
        ("H1", h1_sr, -0.45, 1.2),
    ]
    if all(sr is None or (not sr.support_levels and not sr.resistance_levels) for _, sr, _, _ in rows):
        st.caption("No confirmed support/resistance levels yet on any timeframe.")
        return

    fig = go.Figure()

    def _add(levels, color, y, size_mult, label):
        for lvl in levels:
            fig.add_trace(
                go.Scatter(
                    x=[lvl.price], y=[y], mode="markers",
                    marker=dict(size=8 + min(lvl.touches, 8) * size_mult, color=color, opacity=0.85),
                    hovertemplate=f"{label} {lvl.price:.5g} · {lvl.touches} touches<extra></extra>",
                    showlegend=False,
                )
            )

    for label, sr, y, size_mult in rows:
        if sr is None:
            continue
        _add(sr.support_levels, "#16a34a", y, size_mult, f"{label} support")
        _add(sr.resistance_levels, "#dc2626", y, size_mult, f"{label} resistance")

    fig.add_trace(
        go.Scatter(
            x=[current_price], y=[0], mode="markers",
            marker=dict(size=16, color=_ASSET_HEALTH_ACCENT, symbol="diamond", line=dict(width=2, color="white")),
            hovertemplate=f"Current {current_price:.5g}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_yaxes(visible=False, range=[-1, 1], fixedrange=True)
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_layout(
        height=170,
        margin=dict(l=20, r=20, t=8, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        transition={"duration": 500, "easing": "cubic-in-out"},  # see _render_rsi_gauge's own note
    )
    st.plotly_chart(fig, width="stretch", key="wl_sr_ladder")  # stable key — see _render_rsi_gauge's own note
    st.caption("◆ current price · rows top→bottom: Monthly, D1, H4, H1 · marker size = touch count")


def _render_setup_badges(label: str, signals: list) -> None:
    """Setup archetypes as colored badges, not paragraphs — each
    signal's full reasoning (SetupSignal.detail) moves into the badge's
    own hover tooltip (st.badge's `help` param) instead of sitting on
    the page by default, available on demand without being the default
    view."""
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.caption(label)
        if not signals:
            st.badge("No clear setup", color="gray")
            return
        for s in signals:
            st.badge(s.name.replace("_", " ").title(), color=_SETUP_BADGE_COLORS.get(s.name, "gray"), help=s.detail)


def _render_trade_sim_metric(
    col, label: str, direction_word: str, *,
    win_rate_pct: float | None, avg_r_multiple: float,
    stop_atr_multiple: float, target_atr_multiple: float, max_holding_bars: int,
    wins: int, losses: int, timeouts: int, trades: int,
    round_trip_cost_pct: float = 0.0, swap_pct_per_day: float = 0.0, min_stop_distance_pct: float = 0.0,
) -> None:
    """Shared tile renderer for the RSI/support/resistance backtests,
    all three of which are now real trade simulations (see
    analysis/backtest.py's own top-of-file note) — headline number is
    the real WIN RATE, with the average realized R-multiple as the
    tile's colored delta (st.metric's own automatic red/green-by-sign
    coloring, real signal free of charge) instead of a bare number the
    reader has to interpret unaided. Takes plain fields rather than a
    dataclass instance since the caller has two different real shapes
    to feed it (RSIReactionBacktest directly; SupportResistanceBacktest
    split into its two independent support/resistance halves).

    `round_trip_cost_pct`/`swap_pct_per_day`/`min_stop_distance_pct`
    (added 2026-08-22, direct user request for "more realistic and
    dependable" results) disclose in the tooltip whenever real broker
    cost/guard-rail data was actually netted into avg_r_multiple —
    silently absent (all default to 0.0) for FTMO symbols where that
    real data isn't available, and always absent for PMEX/PSX today,
    which don't have this wired in yet."""
    win_rate = f"{win_rate_pct:.0f}%" if win_rate_pct is not None else "n/a"
    # st.metric only colors a string delta red when it starts with "-"
    # (confirmed against the installed version's own docstring) —
    # everything else, including "+0.00R avg" for a true breakeven
    # result, would otherwise render green/favorable by default. Force
    # gray for anything that rounds to a dead flat 0.00R so a breakeven
    # edge never gets misread as a positive one.
    delta_color = "off" if round(avg_r_multiple, 2) == 0.0 else "normal"
    # Two separate sentences, not one joined list — "net of real X" only
    # grammatically fits the cost/swap numbers; the guard-rail clause
    # describes a structural change to the stop, not something "netted",
    # so it gets its own sentence rather than being forced into that list
    # (a real wording bug caught live: joining all three under "net of
    # real" produced "net of real ... stop widened to..." — broken
    # grammar, not just redundant "real real" phrasing).
    cost_parts = []
    if round_trip_cost_pct > 0:
        cost_parts.append(f"{round_trip_cost_pct:.4f}% round-trip cost")
    if swap_pct_per_day:
        cost_parts.append(f"{swap_pct_per_day:+.4f}%/day swap")
    sentences = []
    if cost_parts:
        sentences.append("Net of real " + " and ".join(cost_parts) + ".")
    if min_stop_distance_pct > 0:
        # Overrides the stop_atr_multiple quoted below for any trade
        # whose ATR-based stop would've been tighter than this floor —
        # not every trade necessarily used the exact ATR multiple.
        sentences.append(f"Stop widened to broker min {min_stop_distance_pct:.3f}% where tighter.")
    cost_note = (" " + " ".join(sentences)) if sentences else ""
    col.metric(
        label,
        win_rate,
        delta=f"{avg_r_multiple:+.2f}R avg",
        delta_color=delta_color,
        help=(
            f"Simulated {direction_word} trade, {stop_atr_multiple:g}x-ATR stop / "
            f"{target_atr_multiple:g}x-ATR target, max {max_holding_bars} bars held "
            f"— {wins} wins, {losses} losses, {timeouts} timed out across "
            f"{trades} real historical episodes.{cost_note}"
        ),
    )


def _render_no_trade_sim_metric(col, label: str) -> None:
    col.metric(
        label, "—",
        help="Not enough real historical episodes, or no High/Low data in this feed to derive a real stop/target from",
    )


def _render_backtest_metrics(base: AssetAnalysis) -> None:
    """Real backtest results as metric tiles (with the full real-sample
    detail in each tile's hover tooltip) instead of four dense
    paragraphs — same underlying numbers analysis/backtest.py already
    computed, just legible at a glance.

    Redesigned 2026-08-22 (direct user feedback: "weak, ambiguous...
    not differentiating between winning and losing opportunity"): the
    RSI-reversal and support/resistance backtests are now real trade
    simulations — a genuine ATR-based stop/target walked forward bar by
    bar to an actual win/loss verdict, not "average return N days
    later" (which couldn't tell a clean winner from a trade that
    crashed hard and only recovered to positive by the measurement
    day). Support and resistance now get their OWN tiles (like RSI's
    existing overbought/oversold pair) rather than one combined "hold
    rate" string, since they're two independent simulated bets (long
    vs. short) with their own independent win rates. Momentum
    persistence and the volatility-regime check are left as-is — both
    are genuine statistical checks (does trailing return predict
    forward return; does low volatility precede bigger moves), not a
    directional trade with a natural stop/target, so a win/loss
    simulation doesn't cleanly apply to either."""
    ob, os_ = base.rsi_overbought_backtest, base.rsi_oversold_backtest
    mp = base.momentum_persistence_backtest
    c1, c2, c3 = st.columns(3)
    if ob is not None:
        _render_trade_sim_metric(
            c1, "RSI overbought → short win rate", "short",
            win_rate_pct=ob.win_rate_pct, avg_r_multiple=ob.avg_r_multiple,
            stop_atr_multiple=ob.stop_atr_multiple, target_atr_multiple=ob.target_atr_multiple,
            max_holding_bars=ob.max_holding_bars, wins=ob.wins, losses=ob.losses,
            timeouts=ob.timeouts, trades=ob.trades,
            round_trip_cost_pct=ob.round_trip_cost_pct, swap_pct_per_day=ob.swap_pct_per_day_used,
            min_stop_distance_pct=ob.min_stop_distance_pct,
        )
    else:
        _render_no_trade_sim_metric(c1, "RSI overbought → short win rate")
    if os_ is not None:
        _render_trade_sim_metric(
            c2, "RSI oversold → long win rate", "long",
            win_rate_pct=os_.win_rate_pct, avg_r_multiple=os_.avg_r_multiple,
            stop_atr_multiple=os_.stop_atr_multiple, target_atr_multiple=os_.target_atr_multiple,
            max_holding_bars=os_.max_holding_bars, wins=os_.wins, losses=os_.losses,
            timeouts=os_.timeouts, trades=os_.trades,
            round_trip_cost_pct=os_.round_trip_cost_pct, swap_pct_per_day=os_.swap_pct_per_day_used,
            min_stop_distance_pct=os_.min_stop_distance_pct,
        )
    else:
        _render_no_trade_sim_metric(c2, "RSI oversold → long win rate")
    c3.metric(
        "Momentum persistence",
        f"{mp.correlation:+.2f}" if mp is not None else "—",
        help=(
            f"{mp.interpretation.replace('_', ' ')} ({mp.sample_size} independent periods)"
            if mp is not None
            else "Not enough history to compute"
        ),
    )

    sr_bt, vr = base.support_resistance_backtest, base.volatility_regime_backtest
    c4, c5, c6 = st.columns(3)
    if sr_bt is not None:
        _render_trade_sim_metric(
            c4, "Support hold → long win rate", "long",
            win_rate_pct=sr_bt.support_win_rate_pct, avg_r_multiple=sr_bt.support_avg_r_multiple,
            stop_atr_multiple=sr_bt.stop_atr_multiple, target_atr_multiple=sr_bt.target_atr_multiple,
            max_holding_bars=sr_bt.max_holding_bars, wins=sr_bt.support_wins, losses=sr_bt.support_losses,
            timeouts=sr_bt.support_timeouts, trades=sr_bt.support_tests,
            round_trip_cost_pct=sr_bt.round_trip_cost_pct, swap_pct_per_day=sr_bt.support_swap_pct_per_day_used,
            min_stop_distance_pct=sr_bt.min_stop_distance_pct,
        )
        _render_trade_sim_metric(
            c5, "Resistance reject → short win rate", "short",
            win_rate_pct=sr_bt.resistance_win_rate_pct, avg_r_multiple=sr_bt.resistance_avg_r_multiple,
            stop_atr_multiple=sr_bt.stop_atr_multiple, target_atr_multiple=sr_bt.target_atr_multiple,
            max_holding_bars=sr_bt.max_holding_bars, wins=sr_bt.resistance_wins, losses=sr_bt.resistance_losses,
            timeouts=sr_bt.resistance_timeouts, trades=sr_bt.resistance_tests,
            round_trip_cost_pct=sr_bt.round_trip_cost_pct, swap_pct_per_day=sr_bt.resistance_swap_pct_per_day_used,
            min_stop_distance_pct=sr_bt.min_stop_distance_pct,
        )
    else:
        _render_no_trade_sim_metric(c4, "Support hold → long win rate")
        _render_no_trade_sim_metric(c5, "Resistance reject → short win rate")
    if vr is not None:
        c6.metric(
            "Move after low-vol vs high-vol",
            f"{vr.low_vol_avg_abs_move_pct:.2f}% / {vr.high_vol_avg_abs_move_pct:.2f}%",
            help=f"{vr.low_vol_episodes} low-vol and {vr.high_vol_episodes} high-vol episodes, {vr.forward_days}-day forward",
        )
    else:
        c6.metric("Move after low-vol vs high-vol", "—", help="Not enough history")


@st.cache_data(ttl=30)
def _cached_asset_live_analysis(symbol: str, description: str) -> FtmoAssetAnalysis:
    """The heavy part of analyze_ftmo_asset_live (D1/H4/H1 fetch +
    backtests + chart structure — several real MT5 round-trips plus real
    computation over up to 1500 bars) cached for 30s per symbol,
    deliberately keyed WITHOUT bid/ask — those tick essentially every
    refresh for an active symbol, which would defeat the cache almost
    entirely if included, and none of this function's own computation
    actually depends on the current bid/ask anyway (see below: the live
    price shown in the dialog always comes from a separate, always-fresh
    get_market_watch() lookup, never from this cached object's own
    stale placeholder bid/ask). This is what makes a fast, near-real-time
    refresh interval affordable: most ticks hit this cache and only pay
    for a fresh live-price lookup + re-rendering already-computed
    numbers, not the full multi-fetch analysis pipeline again.

    st.cache_data deep-copies its return value on every cache HIT (not
    just on the real compute), to protect the cached original from a
    caller mutating it — nothing in this dialog ever reads
    `.base.prices` (confirmed: it's only used by the "Instrument charts"
    expander elsewhere, an entirely different code path), so the up-to-
    1500-row D1 price series it would otherwise carry is pure dead
    weight being deep-copied on every single tick for no benefit.
    Dropped here, right before caching, so the copy that actually
    matters (the rest of the analysis) is the only thing paid for."""
    result = analyze_ftmo_asset_live(symbol, bid=0.0, ask=0.0, description=description)
    result.base.prices = pd.Series(dtype=float)
    return result


def _live_asset_dashboard(symbol: str, exchange: str, description: str) -> None:
    """The dialog's own body — a PLAIN function, deliberately not an
    st.fragment. Two different attempts at an auto-ticking st.fragment
    inside this dialog both caused real, live-reproduced breakage (one a
    runaway accumulating-rerun loop, the next a hard crash that took
    down the whole page's rendering, not just the dialog) — st.dialog
    already has its own fragment-like rerun behavior layered on top of
    the normal script model, and stacking an independently-run_every-
    ticking fragment on top of that is where it broke both times, not
    any one specific way the fragment happened to be wrapped. Rather
    than risk a third variant, this renders once per dialog open/rerun;
    _show_watchlist_detail_dialog's own "Refresh now" button (any widget
    interaction inside a dialog already reruns the dialog on its own,
    per st.dialog's documented fragment-like behavior) is what brings
    it up to date, backed by _cached_asset_live_analysis's 30s TTL so a
    refresh is cheap rather than re-running the full multi-fetch
    analysis pipeline every time.

    Structure follows this project's own "render stable UI before slow
    work" convention: every section header and layout slot is claimed
    up front, in final position, before the one real fetch+compute step
    (bid/ask + the cached technical read) runs — so headings/labels
    appear instantly and only the genuinely slow part shows a loading
    placeholder (st.skeleton), rather than the whole dialog sitting
    blank while it thinks."""
    # Every section header and layout slot claimed up front, in final
    # visual order, BEFORE the one real fetch+compute step runs — so
    # headings/labels appear instantly and only the genuinely slow part
    # (inside ticker_slot) shows a loading placeholder.
    ticker_slot = st.container()
    meta_row = st.columns(2)
    _section_header("Momentum")
    momentum_slot = st.container()
    _section_header("Setup read")
    setup_slot = st.container()
    _section_header("Historical backtests — this instrument's own real price history")
    backtest_slot = st.container()
    cost_slot = st.container()

    with ticker_slot:
        with st.skeleton(height=80):
            try:
                # Single-symbol tick, not the whole Market Watch —
                # confirmed live as a real, meaningful contributor to
                # this popup's reported update lag when every ~1s tick
                # was fetching all 17-21 account symbols just to use one.
                bid_ask = get_current_bid_ask(symbol)
            except MT5ConnectionError as e:
                st.error(f"Live data failed: {e}")
                return
            if bid_ask is None:
                st.warning("This symbol is no longer visible in Market Watch.")
                return
            bid, ask = bid_ask

            # Cached, heavy technical read (see
            # _cached_asset_live_analysis) — only recomputed once per
            # 30s regardless of how fast this dialog itself ticks.
            # Works identically for any exchange despite the "ftmo"
            # name in analyze_ftmo_asset_live; only the trade-cost line
            # below branches on `exchange`, since FTMO's own commission
            # schedule doesn't apply to PMEX.
            analysis = _cached_asset_live_analysis(symbol, description)
            mid = (bid + ask) / 2
            spread = ask - bid
            dp = _watchlist_decimals(mid)

            prev_prices: dict[str, float] = st.session_state.setdefault(
                f"_watchlist_prev_prices_{exchange}", {}
            )
            prev = prev_prices.get(symbol)
            direction = (
                "up" if prev is not None and mid > prev else "down" if prev is not None and mid < prev else ""
            )
            prev_prices[symbol] = mid

            _render_ticker(symbol, mid, bid, ask, spread, dp, direction)

    meta_row[0].metric("Category", _watchlist_group_label(_cached_symbol_category(symbol)))
    meta_row[1].metric("Updated", f"{datetime.now():%H:%M:%S}")

    with momentum_slot:
        # Real gap found live 2026-08-22 (user's own words: this
        # account's analysis "reads, displays and analyzes H1, H4 but
        # not the daily or monthly charts... this way the analysis is
        # missing the long/medium term trends and can throw us in a
        # fake trend unguarded") — the long-term alignment read is the
        # direct answer to that concern, so it's shown first and
        # prominently, not buried under the timeframe detail below it.
        _render_long_term_alignment_banner(analysis)

        # Redesigned live 2026-08-22 — direct user request ("the old
        # charts...only show H1, H4...redesign the charts and its
        # decision with all H1, H4, D1 and monthly and display combined
        # charts and results"). Previously Monthly/Daily got plain
        # st.metric numbers while only H4/H1 got real gauges + the S/R
        # ladder; all 4 timeframes now get the same treatment, ordered
        # long-to-short so the long-term backdrop (what the banner
        # above just talked about) reads first, near-term timing last —
        # matching HOLDING HORIZON, H4/H1 still primary for the actual
        # entry/stop/target, Monthly/D1 real context alongside them.
        timeframes = [
            ("Monthly", analysis.mn1_stats, analysis.mn1_structure),
            ("D1", analysis.base.stats, analysis.d1_structure),
            ("H4", analysis.h4_stats, analysis.h4_structure),
            ("H1", analysis.h1_stats, analysis.h1_structure),
        ]

        st.caption("Monthly · D1 · H4 · H1 — combined RSI read across every timeframe")
        gauge_cols = st.columns(4)
        for col, (label, stats, _structure) in zip(gauge_cols, timeframes):
            with col:
                _render_rsi_gauge(stats.rsi, label)

        _render_sr_ladder(
            mid,
            analysis.mn1_structure.sr_levels,
            analysis.d1_structure.sr_levels,
            analysis.h4_structure.sr_levels,
            analysis.h1_structure.sr_levels,
        )

        st.caption("Trend / regime and volatility per timeframe")
        detail_cols = st.columns(4)
        for col, (label, stats, _structure) in zip(detail_cols, timeframes):
            with col:
                # Real regression caught live 2026-08-22 (user's own
                # screenshot: "Monthly" showing "Downtrend" AND "Trending
                # Up" side by side with nothing distinguishing them reads
                # as a flat contradiction) — this used to say "{label}
                # trend / regime" before the 4-timeframe redesign
                # collapsed it down to just the bare timeframe label,
                # silently dropping the one thing that told a reader these
                # are two DIFFERENT reads (short-term trend vs. the
                # medium-term regime computation), not the same value
                # stated twice. They genuinely can disagree — different
                # formulas, different windows — that disagreement is real
                # signal (a recent pullback inside a longer uptrend, e.g.),
                # not a bug, but only once it's actually labeled as such.
                st.caption(f"{label} trend / regime")
                with st.container(horizontal=True, gap="small"):
                    _trend_badge(stats.trend)
                    _trend_badge(stats.market_regime)
                    # Short-window sibling added 2026-08-26 — catches a
                    # fresh pump/dump leg already underway even when
                    # market_regime above still reads sideways.
                    _trend_badge(stats.momentum_acceleration)
                st.metric(
                    "Volatility (ann.)",
                    f"{stats.volatility_annualized_pct:.1f}%"
                    if stats.volatility_annualized_pct is not None
                    else "—",
                )

    with setup_slot:
        # Same 4-timeframe expansion as momentum_slot above, same
        # ordering (long-to-short) — direct user request ("same with
        # the sub section 'setup read'").
        for label, stats, structure in timeframes:
            _render_setup_badges(label, classify_setups(stats, structure))

    with backtest_slot:
        _render_backtest_metrics(analysis.base)

    with cost_slot:
        cost_line = (
            format_ftmo_trade_cost(analysis) if exchange == "FTMO" else _format_generic_trade_cost(analysis)
        )
        st.caption(cost_line.strip())


_AUTO_TICK_KEY = "wl_auto_tick"


def _inject_autotick_script(interval_ms: int) -> None:
    """Drives the dialog's auto-refresh via a plain JS timer clicking a
    hidden, real st.button — NOT st.fragment(run_every=...). Two
    different attempts at an auto-ticking fragment nested inside this
    same dialog both caused real, live-reproduced breakage (see
    _live_asset_dashboard's own docstring) — st.dialog's own fragment-
    like rerun behavior apparently doesn't tolerate a second,
    independently-scheduled fragment stacked inside it in this
    Streamlit version, regardless of exactly how that fragment was
    wrapped. A real button click is a completely different, far more
    battle-tested code path (the same one Close/Refresh already use
    safely) — "any widget interaction inside a dialog reruns the
    dialog" is st.dialog's own documented behavior, so a JS-driven
    synthetic click on a real (if invisible) button gets the identical,
    already-proven rerun without going anywhere near st.fragment.

    Self-cleaning: the interval checks that the hidden button still
    exists in the DOM before each click and clears itself the first
    time it doesn't (i.e. once the dialog has actually closed and this
    content unmounted) — the timer replaced (via a stored id +
    clearInterval) on every dialog rerun keeps there from ever being
    more than one running at once, the same accumulating-timers failure
    mode the first fragment attempt hit."""
    st.html(
        f"""
        <style>
        [class*="st-key-{_AUTO_TICK_KEY}"] {{
            position: absolute; width: 0; height: 0; overflow: hidden;
            opacity: 0; pointer-events: none;
        }}
        </style>
        <script>
        (function() {{
            if (window.__wlAutoTick) {{ clearInterval(window.__wlAutoTick); }}
            window.__wlAutoTick = setInterval(function() {{
                var btn = document.querySelector('[class*="st-key-{_AUTO_TICK_KEY}"] button');
                if (btn) {{
                    btn.click();
                }} else {{
                    clearInterval(window.__wlAutoTick);
                    window.__wlAutoTick = null;
                }}
            }}, {interval_ms});
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def _clear_watchlist_detail() -> None:
    """Shared by the explicit Close button AND @st.dialog's own
    on_dismiss hook below — real bug found live 2026-08-21: without
    on_dismiss wired up, dismissing via the dialog's native X, Esc, or an
    outside click (its own docs: all three are just "dismiss", no way to
    tell them apart, and none of them are a widget click of ours) left
    `_watchlist_detail_symbol` stuck set forever, since only the explicit
    Close button ever cleared it. That stale flag doesn't just leave the
    dialog's own state confused — _render_watchlist_grid's own pause
    check (`if session_state.get("_watchlist_detail_symbol"): ... return`)
    reads the exact same key, so the watchlist grid was left permanently
    stuck on "Paused while an asset's details are open" with no way left
    to resume, since the one thing that would clear it (clicking Close)
    was no longer reachable once the dialog was gone. Streamlit's
    @st.dialog has a real, documented hook for exactly this
    (`on_dismiss`, a callable invoked — with an automatic rerun — on ANY
    dismissal, X/Esc/outside-click included, not just a widget inside the
    dialog), used below instead of a heuristic/timeout workaround."""
    st.session_state["_watchlist_detail_symbol"] = None


@st.dialog("Asset health", width="large", on_dismiss=_clear_watchlist_detail)
def _show_watchlist_detail_dialog() -> None:
    """Full live technical-dashboard popup for whichever watchlist tile
    was last clicked — see _open_watchlist_detail. Mirrors the existing
    "Confirm Trade Execution" dialog's own open/close contract: a
    session_state key drives whether this gets called at all (below).
    Both the explicit Close button AND any other dismissal (X/Esc/
    outside click, via on_dismiss above) clear that key — see
    _clear_watchlist_detail's own docstring for why both paths are
    needed, not just the button."""
    symbol = st.session_state.get("_watchlist_detail_symbol")
    exchange = st.session_state.get("_watchlist_detail_exchange")
    description = st.session_state.get("_watchlist_detail_description") or symbol
    if not symbol:
        return

    st.html(_ASSET_HEALTH_STYLE)
    refresh_note = (
        f"auto-refreshes every {_refresh_choice} while this stays open"
        if _refresh_choice != "Off"
        else "auto-refresh is off"
    )
    st.caption(f"{description} · live · {refresh_note}")

    _live_asset_dashboard(symbol, exchange, description)

    with st.container(horizontal=True, gap="small"):
        if st.button("Refresh now", icon=":material/refresh:"):
            _cached_asset_live_analysis.clear()
            st.rerun()
        if st.button("Close"):
            _clear_watchlist_detail()
            st.rerun()
        # Real (if invisible) widget — clicking it is what actually
        # drives the auto-refresh; see _inject_autotick_script's own
        # docstring for why this exists instead of st.fragment.
        st.button("tick", key=_AUTO_TICK_KEY)

    if _refresh_choice != "Off":
        _inject_autotick_script(int(_refresh_choice.rstrip("s")) * 1000)


if st.session_state.get("_watchlist_detail_symbol"):
    _show_watchlist_detail_dialog()

# Cold-start fill: if this browser session has never generated a
# suggestion for a given exchange yet, load the most recent saved
# records/*.md transcript's final answer as the default so the section
# below isn't empty right after an app restart. Deliberately does NOT
# populate "suggested_allocation"/mark the suggestion as
# session-generated — a disk-loaded suggestion's entry/stop/target
# prices could be hours or days stale, so "Apply Suggestion" must stay
# disabled until the user generates a fresh one this session (see the
# disabled= condition below).
for _prefix, _records_dir in (
    ("", Path(config.PORTFOLIO_RECORDS_DIR)),
    ("psx_", Path(config.PSX_RECORDS_DIR)),
    ("ftmo_", Path(config.FTMO_RECORDS_DIR)),
):
    if f"{_prefix}last_suggestion_text" not in st.session_state:
        _loaded = _load_latest_saved_suggestion(_records_dir)
        if _loaded is not None:
            _final_answer, _generated_at = _loaded
            _display_text = strip_pending_setups_block(strip_allocation_block(_final_answer))
            if len(_display_text) < 200 and len(_final_answer) > 200:
                _display_text = _final_answer
            _display_text = strip_leading_process_narration(_display_text)
            st.session_state[f"{_prefix}last_suggestion_text"] = _display_text
            st.session_state[f"{_prefix}last_suggestion_generated_at"] = _generated_at
            st.session_state[f"{_prefix}suggested_allocation"] = parse_final_allocation(_final_answer)
        else:
            st.session_state[f"{_prefix}last_suggestion_text"] = None

st.header(":material/insights: Portfolio Suggestion", divider=True)
_portfolio_controls_box = st.container(border=True)
with _portfolio_controls_box:
    # Direct user request 2026-08-23: the manual "Suggest Portfolio Mix"
    # controls now live inside the same box as the Mega Market Analysis
    # countdown for FTMO, rather than a separate box below it.
    mega_analysis_enabled = False
    if selected_exchange == "FTMO":
        _render_mega_analysis_countdown()
        # Outside the countdown's own st.fragment on purpose — a widget
        # inside a fragment only triggers a fragment-scoped rerun, which
        # would leave the "Suggest Portfolio Mix" disabled= state below
        # (defined in this outer, non-fragment script) stuck showing the
        # PREVIOUS value until some unrelated full rerun happened to
        # catch up. Placed here instead, a toggle click is a normal
        # full-script rerun, so the button's disabled state updates in
        # the same instant. Seeded from the persisted file only once per
        # session (not on every rerun) so a later change from THIS
        # widget's own interaction is never clobbered by the file value.
        # Same multi-tab race fixed for the mega-analysis trigger-time
        # picker (see its own comment in _render_mega_analysis_countdown
        # for the full explanation) — compare against this session's own
        # last-synced value, never the file's current one, so a second,
        # stale tab can't flip this back the next time IT reruns for any
        # unrelated reason (any click anywhere on that tab's page).
        if "mega_analysis_enabled_toggle" not in st.session_state:
            st.session_state["mega_analysis_enabled_toggle"] = read_mega_analysis_enabled()
            st.session_state["_mega_analysis_enabled_last_synced"] = (
                st.session_state["mega_analysis_enabled_toggle"]
            )
        else:
            # Live cross-device sync — this toggle lives in the OUTER
            # script (see the comment above for why), so a pull here
            # only actually shows up when a full rerun happens for some
            # reason. _watch_for_external_mega_toggle_change below is
            # what forces that rerun to happen within a couple of
            # seconds even with zero other interaction on this tab.
            _sync_pull_from_file(
                "mega_analysis_enabled_toggle",
                "_mega_analysis_enabled_last_synced",
                read_mega_analysis_enabled(),
            )
        mega_analysis_enabled = st.toggle(
            "Automated daily analysis",
            key="mega_analysis_enabled_toggle",
            help="Mutually exclusive with the manual button below — only one "
            "trigger path is armed at a time.",
        )
        if mega_analysis_enabled != st.session_state["_mega_analysis_enabled_last_synced"]:
            set_mega_analysis_enabled(mega_analysis_enabled)
            st.session_state["_mega_analysis_enabled_last_synced"] = mega_analysis_enabled
        _watch_for_external_mega_toggle_change()
    button_col, model_col, apply_col = st.columns([2, 1, 1])
    with button_col:
        suggest_clicked = st.button(
            "Suggest Portfolio Mix",
            disabled=selected_exchange == "FTMO" and mega_analysis_enabled,
            help=(
                "Automated daily analysis is on — toggle it off above to "
                "trigger this manually."
                if selected_exchange == "FTMO" and mega_analysis_enabled
                else None
            ),
        )
    with model_col:
        selected_model = st.selectbox(
            "Model", ["sonnet", "opus", "haiku"], index=0, label_visibility="collapsed"
        )
    with apply_col:
        # PSX has no execution avenue at all (K-Trade has no order-placement
        # API) — force-disabled regardless of session_state. PMEX/FTMO each
        # gate on their own exchange-scoped suggested_allocation key (see
        # _execution_state_prefix above) so switching the exchange dropdown
        # can never leave this looking clickable for the wrong account. Also
        # requires suggestion_generated_this_session, so a suggestion merely
        # loaded from a past saved record (cold-start fill above) can never
        # arm real execution against stale entry/stop/target prices.
        apply_clicked = st.button(
            "Apply Suggestion",
            disabled=(selected_exchange not in ("PMEX", "FTMO"))
            or not st.session_state.get(f"{_execution_state_prefix}suggested_allocation")
            or not st.session_state.get(f"{_execution_state_prefix}suggestion_generated_this_session"),
        )

    if selected_exchange == "FTMO":
        _render_mega_analysis_progress_log()

    # One-shot result banner for whatever the confirmation dialog (further
    # below) last executed — popped so it shows exactly once, right after
    # the dialog closes and this rerun lands here.
    _execution_outcomes = st.session_state.pop(
        f"{_execution_state_prefix}last_execution_outcomes", None
    )
    if _execution_outcomes:
        for _o, _result in _execution_outcomes:
            if _result.success:
                st.success(f"{_o.symbol} ({_o.action}): order placed, ticket {_result.ticket}")
            else:
                st.error(
                    f"{_o.symbol} ({_o.action}): failed — {_result.comment} "
                    f"(retcode {_result.retcode})"
                )

    if selected_exchange == "PSX":
        st.caption(
            "Hypothetical and research-only — there's no live K-Trade account "
            "connection (no broker API exists for it), so this builds an "
            "illustrative portfolio from PSX's own public market data plus "
            f"live web research, audited by up to {len(AUDIT_MODELS)} free "
            "models the same way as the PMEX suggestion, for you to consider "
            "and execute manually through your own broker. No positions, no "
            "automatic execution."
        )
        capital_col, index_col = st.columns([1, 1])
        with capital_col:
            hypothetical_capital = st.number_input(
                "Hypothetical capital (PKR)", min_value=0.0, value=1_000_000.0, step=50_000.0
            )
        with index_col:
            # Narrows the enrichment pool to one real PSX index instead of
            # always drawing from the full ~490-symbol market, by explicit
            # user request — so the model reasons over a focused, coherent
            # set of companies rather than the entire exchange.
            index_labels = list(PSX_INDICES.values())
            index_tags = list(PSX_INDICES.keys())
            selected_index_label = st.selectbox("PSX Index", index_labels, index=0)
            selected_index_tag = index_tags[index_labels.index(selected_index_label)]
    elif selected_exchange != "FTMO":
        st.caption(
            "Early-stage and discretionary — not a rule-backed recommendation, but "
            "it does factor in your open positions above. Claude drafts a "
            f"suggestion with live web research, up to {len(AUDIT_MODELS)} free "
            "models audit it (each retried for up to 10 min if unavailable, so "
            "more of them get to weigh in), then Claude revises. Can take up to "
            "~35 minutes and uses significant Claude Pro usage."
        )
    # FTMO's own description was removed here (direct user request
    # 2026-08-23) — the Mega Market Analysis section above already
    # explains the pipeline; no separate caption is shown for the
    # manual controls now that they share its box.

if selected_exchange == "FTMO":
    _render_clerk_execution_panel()

if suggest_clicked and selected_exchange == "PSX":
    error_message = None
    warning_message = None
    suggestion = None
    analyses = None

    with st.status("Building a PSX portfolio suggestion...", expanded=True) as status:
        try:
            st.write("Fetching PSX market data...")
            psx_assets = get_psx_market_watch()
        except PSXConnectionError as e:
            error_message = str(e)
        else:
            if not psx_assets:
                warning_message = "PSX Data Portal returned no symbols — try again shortly."
            else:
                analysis_placeholder = st.empty()

                def _on_analysis_progress(msg: str) -> None:
                    analysis_placeholder.markdown(msg)

                analyses = analyze_psx_assets(
                    psx_assets, index_tag=selected_index_tag, on_progress=_on_analysis_progress
                )
                psx_summary = build_psx_summary(
                    hypothetical_capital, psx_assets, analyses, index_tag=selected_index_tag
                )

                audit_placeholder = {"box": None}

                def _on_stage(msg: str) -> None:
                    st.write(msg)
                    if msg.startswith("Sending the draft to"):
                        audit_placeholder["box"] = st.empty()

                def _on_audit_progress(text: str) -> None:
                    box = audit_placeholder["box"]
                    if box is not None:
                        box.markdown(text)

                suggestion = suggest_psx_portfolio(
                    psx_summary,
                    on_stage=_on_stage,
                    on_audit_progress=_on_audit_progress,
                    model=selected_model,
                    save_record=True,
                )
                if suggestion == CLI_MISSING_MESSAGE or suggestion.startswith(CLI_FAILED_PREFIX):
                    error_message = suggestion

        if error_message:
            status.update(label="Failed", state="error")
        elif warning_message:
            status.update(label="No PSX data available", state="error")
        else:
            status.update(label="Suggestion ready", state="complete")

    # Own session_state namespace (psx_* rather than the PMEX keys below)
    # so a PSX result can never make "Apply Suggestion" look enabled for
    # a real MT5 account, and a stale PMEX allocation never bleeds into a
    # PSX run — see the "Apply Suggestion" disabled= condition above.
    st.session_state["psx_suggestion_error"] = error_message
    st.session_state["psx_suggestion_warning"] = warning_message
    if suggestion is not None and not error_message:
        allocation = parse_final_allocation(suggestion)
        display_text = strip_pending_setups_block(strip_allocation_block(suggestion))
        if len(display_text) < 200 and len(suggestion) > 200:
            display_text = suggestion
        display_text = strip_leading_process_narration(display_text)
        st.session_state["psx_last_suggestion_text"] = display_text
        st.session_state["psx_last_suggestion_generated_at"] = datetime.now()
        st.session_state["psx_last_suggestion_analyses"] = analyses
        st.session_state["psx_suggested_allocation"] = allocation
        # Kept alongside the enriched analyses so the sector-performance
        # chart (a whole-market view, not just the enriched pool) can
        # still redraw after the rerun below, the same reason analyses
        # itself is persisted rather than recomputed.
        st.session_state["psx_last_market_assets"] = psx_assets
    else:
        st.session_state["psx_last_suggestion_text"] = None
        st.session_state["psx_last_suggestion_analyses"] = None
    st.rerun()

elif suggest_clicked and selected_exchange == "FTMO":
    error_message = None
    warning_message = None
    already_running_message = None
    suggestion = None
    analyses = None

    with st.status("Building an FTMO portfolio suggestion...", expanded=True) as status:
        # Same lock run_scheduled_mega_analysis (via _fire_scheduled_job_
        # once) already uses — see _MEGA_ANALYSIS_LOCK_PATH's own comment
        # above. Wrapping the WHOLE run in it (not just the Clerk hand-off
        # below, which already had its own separate EXECUTION_LOCK_PATH)
        # is the actual fix for two real, reported bugs that turned out
        # to share one root cause: this branch used to call analyze_ftmo_
        # assets/build_ftmo_summary/suggest_ftmo_portfolio directly,
        # inline, completely bypassing run_mega_analysis() — so a manual
        # click never wrote to the shared progress file (see
        # ai.mega_analysis._write_step/_write_current_activity):
        # (1) no live status ever showed in the shared Mega Session
        # section the way a scheduled run's does, and (2) mega_session_
        # is_live() — what the Clerk panel's own "paused" display and
        # firing logic both depend on — had nothing to go on, so Clerk's
        # periodic check never actually paused for a manual run either.
        # Calling the SAME run_mega_analysis() the scheduled trigger uses
        # fixes both at once, for free — no separate UI code needed here.
        if not acquire_lock(_MEGA_ANALYSIS_LOCK_PATH, _MEGA_ANALYSIS_LOCK_STALE_AFTER_SECONDS):
            already_running_message = (
                "A mega session is already running (the daily schedule, or another "
                "tab's own click) — see its live progress in the Mega Market Analysis "
                "section above; this click will not start a second, overlapping run."
            )
        else:
            st.write(
                "Running now — live step-by-step progress is shown in the Mega "
                "Market Analysis section above, exactly like a scheduled run."
            )
            try:
                _captured_analyses: list = []
                suggestion = run_mega_analysis(model=selected_model, on_analyses=_captured_analyses.append)
                analyses = _captured_analyses[0] if _captured_analyses else None
            except MT5ConnectionError as e:
                error_message = str(e)
                # Real gap found live 2026-09-16: a manual click never
                # wrote to MEGA_ANALYSIS_STATE_FILE at all (only
                # run_scheduled_mega_analysis did), so the "Last run"
                # status shown elsewhere on this page kept reporting a
                # stale scheduled-run result — up to several days old —
                # even immediately after a real, successful manual run.
                # Mirrors run_scheduled_mega_analysis's own outcome
                # classification exactly, just for this second call site.
                _write_mega_analysis_state("error", error_message)
            except RuntimeError as e:
                # e.g. "No instruments are visible in this FTMO account's
                # MT5 Market Watch." — a data-availability issue, not a
                # connection failure, same distinction the old inline
                # check made before delegating this check to
                # run_mega_analysis() itself.
                warning_message = str(e)
                _write_mega_analysis_state("error", warning_message)
            finally:
                release_lock(_MEGA_ANALYSIS_LOCK_PATH)

            if suggestion is not None:
                if suggestion == CLI_MISSING_MESSAGE or suggestion.startswith(CLI_FAILED_PREFIX):
                    error_message = suggestion
                    _write_mega_analysis_state("cli_failed", suggestion[:500])
                else:
                    _write_mega_analysis_state("success")
                    def _on_stage(msg: str) -> None:
                        st.write(msg)

                    # Manual and scheduled runs differ only in what
                    # triggers them, never in what they produce or feed
                    # downstream (direct user correction 2026-08-23) —
                    # this mirrors mega_analysis_job.py's own inline
                    # wiring exactly, down to sharing the literal same
                    # lock, so a manual click and the standalone poll
                    # can never run the Clerk's execution-check at
                    # once. Skips gracefully (not blocking) if the
                    # standalone poll already holds the lock right now.
                    _on_stage("Handing off to the Clerk for an immediate execution-check...")
                    if acquire_lock(EXECUTION_LOCK_PATH, EXECUTION_LOCK_STALE_AFTER_SECONDS):
                        try:
                            # Same hard timeout ceiling and sentinel
                            # pattern as mega_analysis_job.py's own
                            # inline call — a hang here must never
                            # freeze this whole Streamlit session
                            # indefinitely.
                            _clerk_timed_out = object()
                            _clerk_result = run_with_timeout(
                                lambda: run_clerk_execution_check(on_stage=_on_stage),
                                config.CLERK_EXECUTION_RUN_TIMEOUT_SECONDS,
                                default=_clerk_timed_out,
                                catch_exceptions=False,
                            )
                            if _clerk_result is _clerk_timed_out:
                                _on_stage(
                                    "The Clerk's execution-check timed out after "
                                    f"{config.CLERK_EXECUTION_RUN_TIMEOUT_SECONDS / 60:.0f} minutes."
                                )
                        except Exception as e:
                            _on_stage(f"The Clerk's execution-check failed: {e}")
                        finally:
                            release_lock(EXECUTION_LOCK_PATH)
                    else:
                        _on_stage(
                            "The Clerk's execution check is already running right now — "
                            "it will pick this up on its own next run instead."
                        )

        if error_message:
            status.update(label="Failed", state="error")
        elif warning_message:
            status.update(label="No FTMO data available", state="error")
        elif already_running_message:
            status.update(label="Skipped — already running", state="complete")
        else:
            status.update(label="Suggestion ready", state="complete")

    # Own session_state namespace (ftmo_* rather than PMEX's unprefixed
    # keys or PSX's psx_* keys) — same isolation reasoning as PSX's own
    # comment below: a stale FTMO allocation can never make "Apply
    # Suggestion" look enabled against the wrong account, and vice versa.
    st.session_state["ftmo_suggestion_error"] = error_message
    st.session_state["ftmo_suggestion_warning"] = warning_message
    st.session_state["ftmo_suggestion_already_running"] = already_running_message
    if suggestion is not None and not error_message:
        allocation = parse_final_allocation(suggestion, require_side=True)
        display_text = strip_pending_setups_block(strip_allocation_block(suggestion))
        if len(display_text) < 200 and len(suggestion) > 200:
            display_text = suggestion
        display_text = strip_leading_process_narration(display_text)
        st.session_state["ftmo_last_suggestion_text"] = display_text
        st.session_state["ftmo_last_suggestion_generated_at"] = datetime.now(timezone.utc)
        st.session_state["ftmo_last_suggestion_analyses"] = analyses
        if allocation:
            st.session_state["ftmo_suggested_allocation"] = allocation
            st.session_state["ftmo_suggestion_generated_this_session"] = True
            st.session_state["ftmo_rebalance_plan"] = None
    else:
        st.session_state["ftmo_last_suggestion_text"] = None
        st.session_state["ftmo_last_suggestion_analyses"] = None
    st.rerun()

elif suggest_clicked:
    error_message = None
    warning_message = None
    suggestion = None
    analyses = None

    with st.status("Building a portfolio suggestion...", expanded=True) as status:
        try:
            st.write("Fetching account and market data...")
            account = get_account_summary()
            assets = get_market_watch()
        except MT5ConnectionError as e:
            error_message = str(e)
        else:
            if not assets:
                warning_message = (
                    "No instruments are visible in your MT5 Market Watch — "
                    "add some symbols there first."
                )
            else:
                # An updating placeholder, not one line per instrument —
                # real per-symbol work (Yahoo resolution + fetch +
                # backtests), previously with zero progress visibility.
                analysis_placeholder = st.empty()

                def _on_analysis_progress(msg: str) -> None:
                    analysis_placeholder.markdown(msg)

                analyses = analyze_assets(assets, on_progress=_on_analysis_progress)
                portfolio_summary = build_portfolio_summary(
                    account, assets, positions=positions, analyses=analyses
                )

                # A placeholder for the per-model audit sub-status gets
                # created the moment the "Sending the draft..." stage
                # message appears, so it renders directly under that line
                # — then on_audit_progress rewrites the SAME placeholder
                # in place (not a new st.write() line) as models complete,
                # retry, or give up, since that phase alone can run for
                # up to ~10 minutes per model.
                audit_placeholder = {"box": None}

                def _on_stage(msg: str) -> None:
                    st.write(msg)
                    if msg.startswith("Sending the draft to"):
                        audit_placeholder["box"] = st.empty()

                def _on_audit_progress(text: str) -> None:
                    box = audit_placeholder["box"]
                    if box is not None:
                        box.markdown(text)

                suggestion = suggest_portfolio(
                    portfolio_summary,
                    on_stage=_on_stage,
                    on_audit_progress=_on_audit_progress,
                    model=selected_model,
                    save_record=True,
                )
                if suggestion == CLI_MISSING_MESSAGE or suggestion.startswith(CLI_FAILED_PREFIX):
                    error_message = suggestion

        if error_message:
            status.update(label="Failed", state="error")
        elif warning_message:
            status.update(label="No instruments found", state="error")
        else:
            status.update(label="Suggestion ready", state="complete")

    # Persist the result into session_state and rerun, rather than
    # rendering directly here. The "Apply Suggestion" button above already
    # rendered for *this* run (with whatever session_state["suggested_
    # allocation"] held before this block ran) before this point is ever
    # reached — setting session_state now can't retroactively change an
    # already-drawn widget. Only a fresh run re-reads it, which is why the
    # button stayed disabled even once a suggestion had printed. The
    # rendering below is moved outside this `if suggest_clicked` block
    # (driven by session_state, not this run's local variables) so it
    # still appears after the rerun, whose own pass has suggest_clicked
    # back to False.
    st.session_state["suggestion_error"] = error_message
    st.session_state["suggestion_warning"] = warning_message
    if suggestion is not None and not error_message:
        allocation = parse_final_allocation(suggestion, require_side=True)
        display_text = strip_pending_setups_block(strip_allocation_block(suggestion))
        if len(display_text) < 200 and len(suggestion) > 200:
            # Stripping removed almost everything — likely the model used
            # more than one fenced block in a way that confused
            # extraction. Never silently hide the actual answer: show the
            # raw response instead.
            display_text = suggestion
        display_text = strip_leading_process_narration(display_text)
        st.session_state["last_suggestion_text"] = display_text
        st.session_state["last_suggestion_generated_at"] = datetime.now()
        st.session_state["last_suggestion_analyses"] = analyses
        if allocation:
            # Clearing any stale preview from a previous suggestion avoids
            # applying an old plan against a new suggestion.
            st.session_state["suggested_allocation"] = allocation
            st.session_state["suggestion_generated_this_session"] = True
            st.session_state["rebalance_plan"] = None
    else:
        st.session_state["last_suggestion_text"] = None
        st.session_state["last_suggestion_analyses"] = None
    st.rerun()

# Rendered from session_state (not gated on suggest_clicked) so it
# survives the rerun above and keeps showing after any later, unrelated
# button click on this page (e.g. clicking "Apply Suggestion" itself),
# and — via the cold-start fill above — even right after an app restart,
# before this session has generated anything itself. Each exchange has
# its own session_state namespace (see above), so switching the dropdown
# shows that exchange's own last result, if any.
with st.expander("Latest Suggestion", icon=":material/insights:", expanded=False):
    st.caption(
        "Freshly generated this session, or — right after a restart, "
        "before you've clicked \"Suggest Portfolio Mix\" yet — reloaded "
        "from the last saved run for this exchange."
    )
    if selected_exchange == "PSX":
        if st.session_state.get("psx_suggestion_error"):
            st.error(st.session_state["psx_suggestion_error"])
        elif st.session_state.get("psx_suggestion_warning"):
            st.warning(st.session_state["psx_suggestion_warning"])
        elif st.session_state.get("psx_last_suggestion_text"):
            _psx_generated_at = st.session_state.get("psx_last_suggestion_generated_at")
            if _psx_generated_at is not None:
                st.badge(
                    f"Generated {_psx_generated_at:%Y-%m-%d %H:%M:%S}",
                    icon=":material/schedule:",
                    color="blue",
                )
            psx_allocation = st.session_state.get("psx_suggested_allocation")
            psx_analyses = st.session_state.get("psx_last_suggestion_analyses") or []
            psx_market_assets = st.session_state.get("psx_last_market_assets") or []

            if psx_allocation:
                alloc_col, sector_col = st.columns(2)
                with alloc_col:
                    _render_allocation_chart(psx_allocation)
                with sector_col:
                    _render_sector_allocation_chart(psx_allocation, psx_analyses)
                _render_capital_at_risk_chart(psx_allocation)

            _render_ai_report(st.session_state["psx_last_suggestion_text"])

            if psx_market_assets:
                _render_sector_performance_chart(psx_market_assets)
            if psx_analyses:
                _render_risk_return_scatter(psx_analyses)
                _render_week52_position_chart(psx_analyses)

            psx_chartable = [
                a for a in psx_analyses if a.display_name is not None and not a.prices.empty
            ]
            if psx_chartable:
                with st.expander(f"Instrument charts ({len(psx_chartable)})"):
                    for a in psx_chartable:
                        _render_instrument_chart(a)
    elif selected_exchange == "FTMO":
        if st.session_state.get("ftmo_suggestion_error"):
            st.error(st.session_state["ftmo_suggestion_error"])
        elif st.session_state.get("ftmo_suggestion_warning"):
            st.warning(st.session_state["ftmo_suggestion_warning"])
        elif st.session_state.get("ftmo_suggestion_already_running"):
            st.info(st.session_state["ftmo_suggestion_already_running"])
        else:
            # Two candidate sources: THIS session's own manual click (if
            # any) vs the most recent saved records/*.md transcript, which
            # ANY successful run — manual OR automated, in any process —
            # always writes (see save_portfolio_session). Direct user
            # report 2026-09-05 (two rounds): (1) after a genuinely NEWER
            # automated run had completed, this panel kept showing an
            # OLDER manual click's own session-local suggestion instead —
            # a prior fix (2026-09-01) added a persisted-file fallback for
            # when session_state was EMPTY, but once session_state held
            # anything at all it unconditionally took priority forever
            # after, regardless of which was actually more recent; (2) the
            # first attempt at fixing that read the DERIVED, simplified
            # mega_analysis_latest_suggestion.json (built for Copilot's
            # execution job, see _write_latest_suggestion) and rendered
            # bare tables from it instead of the real report — that file
            # never carried the full prose or a chart-ready allocation
            # shape. This reuses _load_latest_saved_suggestion (the same
            # helper the cold-start loader below already uses) to read the
            # actual saved transcript instead, giving this fallback the
            # exact same rich rendering (allocation chart, capital-at-risk
            # chart, full AI report text) as a fresh manual click — not a
            # second, poorer-quality rendering path.
            _session_generated_at = st.session_state.get("ftmo_last_suggestion_generated_at")
            if _session_generated_at is not None and _session_generated_at.tzinfo is None:
                # Defensive: this can still be naive here — either a value
                # written by an OLDER version of this code (Streamlit
                # re-execs app.py fresh on every rerun, but session_state
                # itself persists across reruns within the same browser
                # session) from before that fix, or the cold-start
                # loader's own value (populated from _load_latest_saved_
                # suggestion, which parses a naive LOCAL timestamp from
                # the saved file's own NAME). Either way it represents
                # this SYSTEM's local time, not UTC — .astimezone() on a
                # naive datetime is documented Python behavior to
                # interpret it that way and convert correctly (unlike
                # .replace(tzinfo=...), which would mislabel a local value
                # as UTC without adjusting the actual time).
                _session_generated_at = _session_generated_at.astimezone(timezone.utc)

            _persisted_loaded = _load_latest_saved_suggestion(Path(config.FTMO_RECORDS_DIR))
            _persisted_generated_at = None
            if _persisted_loaded is not None:
                # generated_at here is parsed from the FILENAME, written
                # via naive local datetime.now() (save_portfolio_session)
                # — .astimezone() on a naive datetime is documented Python
                # behavior to interpret it as this SYSTEM's own local time
                # and convert correctly, making it comparable to the
                # UTC-aware session timestamp above.
                _persisted_generated_at = _persisted_loaded[1].astimezone(timezone.utc)

            _use_persisted = _persisted_loaded is not None and (
                not st.session_state.get("ftmo_last_suggestion_text")
                or _session_generated_at is None
                or (_persisted_generated_at is not None and _persisted_generated_at > _session_generated_at)
            )

            if not _use_persisted and st.session_state.get("ftmo_last_suggestion_text"):
                if _session_generated_at is not None:
                    st.badge(
                        f"Generated {_session_generated_at:%Y-%m-%d %H:%M:%S} UTC",
                        icon=":material/schedule:",
                        color="blue",
                    )
                ftmo_allocation = st.session_state.get("ftmo_suggested_allocation")
                if ftmo_allocation:
                    _render_allocation_chart(ftmo_allocation)
                    _render_capital_at_risk_chart(ftmo_allocation, pct_is_risk=True)
                _render_ai_report(st.session_state["ftmo_last_suggestion_text"])

                # FtmoAssetAnalysis wraps a PMEX-shape AssetAnalysis as
                # `.base` — unwrap it here so _render_instrument_chart
                # (built for that shape) works unchanged, same reuse as
                # PMEX's own chart loop.
                ftmo_analyses = st.session_state.get("ftmo_last_suggestion_analyses") or []
                ftmo_chartable = [
                    a.base
                    for a in ftmo_analyses
                    if a.base.display_name is not None and not a.base.prices.empty
                ]
                if ftmo_chartable:
                    with st.expander(f"Instrument charts ({len(ftmo_chartable)})"):
                        for a in ftmo_chartable:
                            _render_instrument_chart(a)
            elif _persisted_loaded is not None:
                # Direct user report 2026-09-01: after a genuinely
                # successful AUTOMATED mega session (the in-app auto-
                # trigger, running on a background thread — see
                # _fire_scheduled_job_once), this whole panel showed
                # nothing at all — a background thread has no browser
                # session to write ftmo_last_suggestion_text into, so an
                # automated run's real, successful, freshly-written
                # suggestion was invisible here even though the Execution
                # Clerk panel below (which reads its own persisted file
                # directly on every tick) and the real MT5 account were
                # both already correctly reflecting it.
                _final_answer, _ = _persisted_loaded
                _display_text = strip_pending_setups_block(strip_allocation_block(_final_answer))
                if len(_display_text) < 200 and len(_final_answer) > 200:
                    _display_text = _final_answer
                _display_text = strip_leading_process_narration(_display_text)
                _persisted_allocation = parse_final_allocation(_final_answer)

                if _persisted_generated_at is not None:
                    st.badge(
                        f"Generated {_persisted_generated_at:%Y-%m-%d %H:%M:%S} UTC",
                        icon=":material/schedule:",
                        color="blue",
                    )
                st.caption(
                    "This came from the automated daily mega session, not a manual click "
                    "in this browser tab — showing the saved suggestion directly. See the "
                    "Execution Clerk section below for live status and any orders it's "
                    "already acted on."
                )
                if _persisted_allocation:
                    _render_allocation_chart(_persisted_allocation)
                    _render_capital_at_risk_chart(_persisted_allocation, pct_is_risk=True)
                _render_ai_report(_display_text)
                # No instrument charts here — those need the live
                # FtmoAssetAnalysis objects (price history, chart
                # structure) a saved .md transcript never persists; only
                # a session that actually just ran the analysis has them.
    else:
        if st.session_state.get("suggestion_error"):
            # Covers both the MT5 connection error and the claude -p call itself
            # failing — show that clearly and stop, rather than rendering research
            # charts next to an easy-to-miss error (which is exactly what an
            # earlier "just charts, no suggestion" bug looked like).
            st.error(st.session_state["suggestion_error"])
        elif st.session_state.get("suggestion_warning"):
            st.warning(st.session_state["suggestion_warning"])
        elif st.session_state.get("last_suggestion_text"):
            _generated_at = st.session_state.get("last_suggestion_generated_at")
            if _generated_at is not None:
                st.badge(
                    f"Generated {_generated_at:%Y-%m-%d %H:%M:%S}",
                    icon=":material/schedule:",
                    color="blue",
                )
            allocation = st.session_state.get("suggested_allocation")
            if allocation:
                _render_allocation_chart(allocation)
                _render_capital_at_risk_chart(allocation, pct_is_risk=True)
            _render_ai_report(st.session_state["last_suggestion_text"])

            analyses = st.session_state.get("last_suggestion_analyses") or []
            chartable = [a for a in analyses if a.display_name is not None and not a.prices.empty]
            if chartable:
                with st.expander(f"Instrument charts ({len(chartable)})"):
                    for a in chartable:
                        _render_instrument_chart(a)

if apply_clicked:
    # The top-of-page connect() branch already connected to whichever
    # account matches the CURRENT selected_exchange earlier in this same
    # script run (Streamlit reruns top-to-bottom on every interaction),
    # so these fresh_* fetches below already reflect the right account
    # without a second connect() call here.
    allocation = st.session_state.get(f"{_execution_state_prefix}suggested_allocation")
    if allocation:
        try:
            fresh_account = get_account_summary()
            fresh_positions = get_open_positions()
            fresh_pending_orders = get_pending_orders()
            fresh_assets = get_market_watch()
        except MT5ConnectionError as e:
            st.error(str(e))
        else:
            market_prices = {a.symbol: a for a in fresh_assets}
            plan = compute_rebalance_plan(
                fresh_positions,
                fresh_account,
                allocation,
                get_contract_spec,
                market_prices,
                price_sanity_band_pct=config.PRICE_SANITY_BAND_PCT,
                pending_orders=fresh_pending_orders,
                amend_tolerance_pct=config.AMEND_TOLERANCE_PCT,
                max_pending_order_age_hours=config.MAX_PENDING_ORDER_AGE_HOURS,
                min_stop_distance_pct=config.MIN_STOP_DISTANCE_PCT,
                held_position_size_tolerance_pct=config.HELD_POSITION_SIZE_TOLERANCE_PCT,
            )
            if not plan:
                st.info("Nothing to do — the account already matches the suggested mix.")
            st.session_state[f"{_execution_state_prefix}rebalance_plan"] = plan
            st.session_state[f"{_execution_state_prefix}rebalance_plan_positions"] = fresh_positions

if st.session_state.get(f"{_execution_state_prefix}rebalance_plan"):
    _plan = st.session_state[f"{_execution_state_prefix}rebalance_plan"]
    _plan_positions = st.session_state.get(f"{_execution_state_prefix}rebalance_plan_positions", [])

    @st.dialog(f"Confirm Trade Execution — {selected_exchange}", width="large")
    def _show_apply_dialog(plan: list, plan_positions: list) -> None:
        """Final approval step before anything real is sent to MT5 —
        a modal instead of a table quietly sitting at the bottom of the
        page, so a trade can't be confirmed by an absent-minded scroll
        past it. Behaves like a fragment (per st.dialog's own semantics):
        clicking Confirm/Cancel here reruns only this dialog UNLESS
        st.rerun() is called explicitly, which forces the full-page
        rerun that actually closes it (the outer `if` above re-checking
        session_state and finding rebalance_plan cleared to None)."""
        st.caption("Nothing has been sent to MT5 yet — review, then confirm or cancel.")
        preview_df = pd.DataFrame(
            [
                {
                    "Symbol": o.symbol,
                    "Action": o.action,
                    "Side": o.side,
                    "Lots": o.volume,
                    "Order Type": o.order_type,
                    "Price": f"{o.price:.4f}" if o.price is not None else "—",
                    "Stop": f"{o.stop_loss:.4f}" if o.stop_loss is not None else "—",
                    "Target": f"{o.take_profit:.4f}" if o.take_profit is not None else "—",
                    "Reason": o.reason,
                }
                for o in plan
            ]
        )
        st.dataframe(preview_df, hide_index=True)

        # Account-aware, not a single global check: PMEX and FTMO each
        # have their own MT5_SERVER-shaped config value, and only the
        # one for the account actually connected right now (per
        # selected_exchange) is relevant — checking the wrong one could
        # either wrongly block a real PMEX demo run or wrongly allow one
        # against FTMO's real server string.
        current_server = (
            config.FTMO_MT5_SERVER if selected_exchange == "FTMO" else config.MT5_SERVER
        )
        is_demo = bool(current_server) and "demo" in current_server.lower()

        # FTMO-specific hard pre-execution check — PMEX has no daily-loss
        # rule to check against, so this never applies there. Computed
        # from the same allocation the capital-at-risk chart already
        # visualizes (see compute_aggregate_heat_pct), not from `plan`
        # itself (whose PlannedOrder objects don't carry a %-of-equity
        # figure).
        ftmo_heat_blocked = False
        ftmo_heat_blocked_reason = ""
        if selected_exchange == "FTMO" and ftmo_status is not None:
            current_allocation = (
                st.session_state.get(f"{_execution_state_prefix}suggested_allocation") or {}
            )
            planned_heat_pct = compute_aggregate_heat_pct(current_allocation)
            if would_breach_daily_loss_headroom(ftmo_status, planned_heat_pct):
                ftmo_heat_blocked = True
                allowed_pct = (
                    max(0.0, ftmo_status.daily_loss_headroom_pct) * DEFAULT_HEADROOM_FRACTION
                )
                ftmo_heat_blocked_reason = (
                    f"Execution blocked: this plan's aggregate heat "
                    f"({planned_heat_pct:.2f}% of equity at risk if every stop "
                    f"is hit) would exceed {DEFAULT_HEADROOM_FRACTION:.0%} of "
                    "this account's REAL remaining daily-loss headroom "
                    f"({ftmo_status.daily_loss_headroom_pct:.2f}%, so at most "
                    f"{allowed_pct:.2f}% of equity at risk is allowed right "
                    "now). Reduce position sizes or wait for headroom to "
                    "recover before executing — a daily-loss breach is "
                    "instant termination with zero grace period."
                )

        # Real MT5-level trading permission — distinct from account
        # CONFIGURATION (is_demo/ALLOW_LIVE_EXECUTION above are policy
        # choices this app makes; this is whether the terminal will even
        # attempt to send an order right now). Confirmed live this
        # account's own AutoTrading toggle was off, which silently fails
        # every order_send() with a bare retcode — checked here, before
        # offering "Confirm and Execute" at all, so the reason is obvious
        # up front instead of discovered one rejected order at a time.
        trading_permitted, trading_blocked_reason = (
            (True, "") if config.USE_MOCK_DATA else is_trading_permitted()
        )

        # See risk/apply_suggestion.py::check_execution_safety_gates'
        # own docstring for why this is a shared, pure function now
        # rather than inline logic here — the unattended Clerk
        # execution job needs the exact same three gates, and this logic
        # must never drift between two independently-maintained copies.
        execution_ok, block_reason = check_execution_safety_gates(
            is_demo=is_demo,
            allow_live_execution=config.ALLOW_LIVE_EXECUTION,
            trading_permitted=trading_permitted,
            trading_blocked_reason=trading_blocked_reason,
            ftmo_heat_blocked=ftmo_heat_blocked,
            ftmo_heat_blocked_reason=ftmo_heat_blocked_reason,
        )
        execution_blocked = not execution_ok
        if execution_blocked and block_reason:
            st.error(block_reason)

        # A Cancel button is always available, blocked or not — previously
        # only the unblocked branch rendered one, so a BLOCKED plan (the
        # heat gate especially) had no in-dialog way to dismiss itself and
        # kept re-triggering the same blocked dialog on every rerun; only
        # the dialog's own X/Esc worked, and even that left session_state
        # holding the stale plan.
        if execution_blocked:
            cancel_clicked = st.button("Cancel")
            confirm_clicked = False
        else:
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                confirm_clicked = st.button("Confirm and Execute", type="primary")
            with cancel_col:
                cancel_clicked = st.button("Cancel")

        if cancel_clicked:
            st.session_state[f"{_execution_state_prefix}rebalance_plan"] = None
            st.rerun()

        if confirm_clicked:
            outcomes: list[tuple] = []
            for o in plan:
                if o.action in ("open", "increase"):
                    try:
                        result = open_position(
                            o.symbol, o.side, o.volume, o.price, o.stop_loss, o.take_profit
                        )
                    except MT5ConnectionError as e:
                        result = OrderResult(False, None, str(e), None)
                    outcomes.append((o, result))
                elif o.action in ("reduce", "close"):
                    for ticket, ticket_volume in o.tickets_to_close:
                        matching = next(
                            (p for p in plan_positions if p.ticket == ticket), None
                        )
                        if matching is None:
                            outcomes.append(
                                (o, OrderResult(False, None, f"ticket {ticket} not found", None))
                            )
                            continue
                        try:
                            result = close_position(matching, volume=ticket_volume)
                        except MT5ConnectionError as e:
                            result = OrderResult(False, None, str(e), None)
                        outcomes.append((o, result))
                elif o.action == "cancel":
                    for ticket in o.pending_tickets_to_cancel:
                        try:
                            result = cancel_pending_order(ticket)
                        except MT5ConnectionError as e:
                            result = OrderResult(False, None, str(e), None)
                        outcomes.append((o, result))
                elif o.action == "amend_pending":
                    cancel_results = []
                    for ticket in o.pending_tickets_to_cancel:
                        try:
                            result = cancel_pending_order(ticket)
                        except MT5ConnectionError as e:
                            result = OrderResult(False, None, str(e), None)
                        cancel_results.append(result)
                        outcomes.append((o, result))
                    if all(r.success for r in cancel_results):
                        try:
                            result = open_position(
                                o.symbol, o.side, o.volume, o.price, o.stop_loss, o.take_profit
                            )
                        except MT5ConnectionError as e:
                            result = OrderResult(False, None, str(e), None)
                        outcomes.append((o, result))
                    else:
                        # Don't stack a fresh order on top of one that
                        # failed to cancel — same guard as the unattended
                        # execution loop's own amend_pending branch.
                        outcomes.append(
                            (o, OrderResult(False, None, "cancel failed, replacement order not sent", None))
                        )
                elif o.action == "amend_position":
                    for ticket in o.position_tickets_to_amend:
                        matching = next(
                            (p for p in plan_positions if p.ticket == ticket), None
                        )
                        if matching is None:
                            outcomes.append(
                                (o, OrderResult(False, None, f"ticket {ticket} not found", None))
                            )
                            continue
                        try:
                            result = modify_position_sltp(matching, o.stop_loss, o.take_profit)
                        except MT5ConnectionError as e:
                            result = OrderResult(False, None, str(e), None)
                        outcomes.append((o, result))
                # "hold"/"infeasible": nothing to execute.

            st.session_state[f"{_execution_state_prefix}rebalance_plan"] = None
            # Stored rather than shown here directly: this dialog is
            # about to close via the rerun below, and the outcome
            # banner (rendered near the Apply Suggestion button,
            # above) is what the user actually sees next.
            st.session_state[f"{_execution_state_prefix}last_execution_outcomes"] = outcomes
            st.rerun()

    _show_apply_dialog(_plan, _plan_positions)
