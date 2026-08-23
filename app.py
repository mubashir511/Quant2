from datetime import datetime, time as dt_time, timezone
from pathlib import Path

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
    analyze_ftmo_assets,
    build_ftmo_summary,
    classify_long_term_alignment,
    fetch_ftmo_status,
    format_ftmo_trade_cost,
    suggest_ftmo_portfolio,
)
from ai.copilot_execution import (
    EXECUTION_LOCK_PATH,
    EXECUTION_LOCK_STALE_AFTER_SECONDS,
    next_execution_check_utc,
    read_copilot_execution_enabled,
    read_copilot_execution_interval_minutes,
    read_execution_progress,
    read_execution_state,
    read_settlement,
    run_copilot_execution_check,
    set_copilot_execution_enabled,
    set_copilot_execution_interval_minutes,
)
from job_lock import acquire_lock, release_lock
from utils import run_with_timeout
from ai.mega_analysis import (
    next_run_utc,
    read_latest_suggestion,
    read_mega_analysis_enabled,
    read_mega_analysis_trigger,
    read_progress,
    read_state,
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
from data.mt5_execution import OrderResult, close_position, open_position
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
from risk.rebalance import evaluate_positions

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


@st.cache_data(ttl=300)
def _cached_server_offset_seconds() -> float | None:
    """The MT5 broker's real UTC offset, cached for 5 minutes — this
    countdown ticks every second (see _render_mega_analysis_countdown's
    own st.fragment), and re-fetching a live tick from MT5 that often for
    a slowly-changing display value would be a wasted round-trip every
    single second."""
    offset = get_server_time_offset()
    return offset.total_seconds() if offset is not None else None


def _render_mega_analysis_countdown() -> None:
    """FTMO-only: shows the countdown to the next unattended daily "mega
    market analysis" (Claude Sonnet + the full audit pool — Copilot is
    deliberately excluded from FTMO's audit pool entirely now, not just
    here, see ai.ftmo_suggest.suggest_ftmo_portfolio's own docstring for
    why — triggered by the Quant2MegaAnalysis Windows Scheduled Task —
    see mega_analysis_job.py — independent of whether this page is even
    open). Scheduling itself is anchored to plain UTC (see
    ai.mega_analysis's own docstrings for why local-time scheduling would
    silently drift across DST); this widget just converts that UTC
    instant to both this machine's local time (astimezone(), correctly
    DST-aware) and the broker's own server time (from a live tick,
    best-effort) purely for display, per the user's explicit request to
    stay aware of both clocks alongside UTC.

    Also shows the run's own LIVE progress while one is actually
    happening (see ai.mega_analysis::_write_progress) — direct user
    request 2026-08-22: the unattended run should be visible the same
    way the manual "Suggest Portfolio Mix" button already is, the only
    real difference being WHAT triggers it. "Live" is detected purely
    from timestamps (progress fresher than 5 minutes old AND newer than
    the last COMPLETED attempt) rather than an explicit start/stop flag
    — once run_scheduled_mega_analysis finishes and writes real state,
    that naturally becomes the newer of the two again, turning this back
    off with no separate cleanup step needed."""
    now_utc = datetime.now(timezone.utc)
    state = read_state()
    progress = read_progress()
    live_message = None
    progress_ts = progress.get("updated_utc")
    if progress_ts:
        try:
            progress_dt = datetime.fromisoformat(progress_ts)
        except ValueError:
            progress_dt = None
        if progress_dt is not None:
            age_seconds = (now_utc - progress_dt).total_seconds()
            last_attempt = state.get("last_attempt_utc")
            newer_than_last_attempt = last_attempt is None or progress_ts > last_attempt
            if 0 <= age_seconds < 300 and newer_than_last_attempt:
                live_message = progress.get("message")
    target = next_run_utc(now_utc, state=state)
    remaining_seconds = max(0, int((target - now_utc).total_seconds()))
    hours, rem = divmod(remaining_seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    broker_str = "unavailable (no live MT5 tick)"
    if not config.USE_MOCK_DATA:
        try:
            offset_seconds = _cached_server_offset_seconds()
        except MT5ConnectionError:
            offset_seconds = None
        if offset_seconds is not None:
            broker_time = target + pd.Timedelta(seconds=offset_seconds)
            broker_str = broker_time.strftime("%Y-%m-%d %H:%M") + " broker server time"

    # No own border here (direct user request 2026-08-23: merge this
    # visually with the manual "Suggest Portfolio Mix" controls into one
    # shared box) — the caller supplies the surrounding
    # st.container(border=True) instead.
    st.subheader(":material/schedule: Next Mega Market Analysis")
    if live_message:
        st.info(f":material/autorenew: Running now — {live_message}")
    # Direct user request 2026-08-23: significantly shortened (2-3 lines,
    # not the full design rationale — kept in this module's own docstring
    # above instead), and no longer names a specific model or trigger
    # time — both are user-changeable (the model via MEGA_ANALYSIS_MODEL,
    # the time via the picker below), so hardcoding either here would go
    # stale the moment either one is actually changed.
    st.caption(
        f"Unattended daily FTMO run, reviewed by up to {len(AUDIT_MODELS)} audit models "
        "before it's finalized. Copilot handles execution separately, checking every "
        f"{config.COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES} min (see below)."
    )
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
    _trigger_hour, _trigger_minute = read_mega_analysis_trigger()
    if "mega_analysis_trigger_time" not in st.session_state:
        st.session_state["mega_analysis_trigger_time"] = dt_time(_trigger_hour, _trigger_minute)
    _new_trigger_time = st.time_input(
        "Change daily trigger time (UTC)", key="mega_analysis_trigger_time", step=300
    )
    if (_new_trigger_time.hour, _new_trigger_time.minute) != (_trigger_hour, _trigger_minute):
        set_mega_analysis_trigger(_new_trigger_time.hour, _new_trigger_time.minute)

    last_status = state.get("last_status")
    last_attempt = state.get("last_attempt_utc")
    # Compact caption-styled status instead of a full st.success/st.warning
    # alert box (direct user request 2026-08-23: shrink this and squeeze
    # the gap before the manual controls below it).
    if last_status == "success" and last_attempt:
        st.caption(f":green[✓ Last run ({last_attempt} UTC): succeeded.]")
    elif last_status in ("error", "cli_failed", "timeout") and last_attempt:
        st.caption(
            f":orange[⚠ Last attempt ({last_attempt} UTC) did NOT succeed: "
            f"{state.get('last_detail', '(no detail recorded)')}]"
        )


_render_mega_analysis_countdown = st.fragment(run_every=1)(_render_mega_analysis_countdown)


def _render_copilot_execution_panel() -> None:
    """FTMO-only: shows the mega session's own Immediate Allocation AND
    Pending Setups (the "clerk/executioner" half of the boardroom
    architecture — see ai/copilot_execution.py's own module docstring),
    each with its real status (placed/failed-with-reason for immediate
    targets; unchecked/order placed/filled/closed-after-fill plus
    Copilot's own reasoning text for Pending Setups), and a live-progress
    banner while a check is actually running — same standing "automated
    runs must be as visible as a manual button click" principle as
    _render_mega_analysis_countdown above, applied to this second
    unattended job. The Immediate Allocation table was added 2026-08-23
    direct user request ("I see position in 5 assets but only 2 assets
    are showing up in the clerk section") — this panel previously only
    ever rendered pending_setups, never the immediate_allocation targets
    ai.copilot_execution.run_copilot_execution_check actually tries to
    execute every poll, so a FAILED attempt (no settlement record ever
    gets created for those) was invisible here even though it was
    already in the log file. Also has its own enable/disable toggle, a
    "next review" timer, and a review-frequency picker (direct user
    request 2026-08-23) — all three widgets live inside this same
    st.fragment(run_every=5), which is safe here (unlike the mega
    analysis toggle) since nothing OUTSIDE this fragment needs to react
    to them; the whole panel simply redraws itself on its own next tick."""
    suggestion = read_latest_suggestion()
    pending_setups = suggestion.get("pending_setups", [])

    now_utc = datetime.now(timezone.utc)
    exec_state = read_execution_state()
    exec_progress = read_execution_progress()
    live_message = None
    progress_ts = exec_progress.get("updated_utc")
    if progress_ts:
        try:
            progress_dt = datetime.fromisoformat(progress_ts)
        except ValueError:
            progress_dt = None
        if progress_dt is not None:
            age_seconds = (now_utc - progress_dt).total_seconds()
            last_attempt = exec_state.get("last_attempt_utc")
            newer_than_last_attempt = last_attempt is None or progress_ts > last_attempt
            if 0 <= age_seconds < 300 and newer_than_last_attempt:
                live_message = exec_progress.get("message")

    with st.container(border=True):
        heading_col, toggle_col = st.columns([3, 1])
        with heading_col:
            st.subheader(":material/support_agent: Copilot Execution Clerk")
        with toggle_col:
            if "copilot_execution_enabled_toggle" not in st.session_state:
                st.session_state["copilot_execution_enabled_toggle"] = read_copilot_execution_enabled()
            copilot_enabled = st.toggle("Enabled", key="copilot_execution_enabled_toggle")
            if copilot_enabled != read_copilot_execution_enabled():
                set_copilot_execution_enabled(copilot_enabled)

        if live_message:
            st.info(f":material/autorenew: Running now — {live_message}")
        st.caption(
            "Unattended: reads the mega session's own guidance, checks each Pending "
            "Setup against fresh live technicals via GitHub Copilot, and executes a "
            "confirmed one immediately — no human confirmation step."
        )

        interval_minutes = read_copilot_execution_interval_minutes()
        next_check = next_execution_check_utc(now_utc, state=exec_state)
        remaining_seconds = max(0, int((next_check - now_utc).total_seconds()))
        rem_minutes, rem_seconds = divmod(remaining_seconds, 60)

        timer_col, freq_col = st.columns([1, 1])
        with timer_col:
            st.metric(
                "Next review",
                "due now" if remaining_seconds <= 0 else f"{rem_minutes}m {rem_seconds:02d}s",
            )
        with freq_col:
            # Range starts at 1 min (direct user request 2026-08-23).
            # Always includes the CURRENT value too (whatever it is) so a
            # custom/hand-edited interval never breaks this selectbox —
            # st.selectbox requires its key's stored value to be one of
            # the options offered.
            _freq_options = sorted({1, 2, 5, 10, 15, 30, 60, interval_minutes})
            if "copilot_execution_interval_select" not in st.session_state:
                st.session_state["copilot_execution_interval_select"] = interval_minutes
            new_interval = st.selectbox("Review every (min)", _freq_options, key="copilot_execution_interval_select")
            if new_interval != interval_minutes:
                set_copilot_execution_interval_minutes(new_interval)
        if interval_minutes < 5:
            st.caption(
                ":gray[The Windows Scheduled Task itself only polls every 5 min, so anything "
                "below that still checks at most every ~5 min in practice.]"
            )

        if not copilot_enabled:
            st.caption(":gray[Disabled — won't check or execute anything until re-enabled.]")

        if not suggestion:
            st.caption("No mega-analysis suggestion on file yet.")
            return

        settled = read_settlement().get("settled", {})
        last_verdicts = exec_state.get("last_verdicts", {})
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
                imm_rows.append(
                    {
                        "Symbol": symbol,
                        "Side": entry.get("side"),
                        "Target %": entry.get("pct"),
                        "Status": status,
                        "Detail": result.get("detail", "") if result is not None else "",
                    }
                )
            st.dataframe(pd.DataFrame(imm_rows), hide_index=True)

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
                # Direct user request 2026-08-23: when Copilot judges a
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
                        "Copilot's reasoning": reasoning,
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        last_status = exec_state.get("last_status")
        last_attempt = exec_state.get("last_attempt_utc")
        if last_status == "success" and last_attempt:
            st.success(f"Last check ({last_attempt} UTC): {exec_state.get('last_detail', 'succeeded')}.")
        elif last_status == "blocked" and last_attempt:
            st.warning(f"Last check ({last_attempt} UTC) executed nothing: {exec_state.get('last_detail', '')}")
        elif last_status == "disabled" and last_attempt:
            st.caption(f"Last check ({last_attempt} UTC): skipped — the clerk was disabled.")


_render_copilot_execution_panel = st.fragment(run_every=5)(_render_copilot_execution_panel)


def _render_allocation_chart(allocation: dict[str, AllocationEntry]) -> None:
    _render_pie(
        list(allocation.keys()), [e.pct for e in allocation.values()], "Allocation by Instrument"
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
    ftmo_status variables that feed Rebalance Suggestions/Portfolio
    Suggestion/execution below — this is a display-only section, so a
    periodic refetch here can never cause those higher-stakes sections to
    act on a stale-vs-fresh data mismatch. Renders inside the "Account
    Overview" bordered container established above — a fragment's
    periodic reruns only refill its own slot in the page, so it stays
    nested there across every tick, not just the first render."""
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

                df = pd.DataFrame(
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

                st.dataframe(df.style.map(_color_pnl, subset=["P&L"]), hide_index=True)
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
        with st.container(border=True):
            st.subheader("Trade History")
            st.caption(
                "Real closed trades from this account's own MT5 history — "
                "net P&L per position (every leg's profit, swap, and "
                "commission combined), most recent first."
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

                history_df = pd.DataFrame(
                    [
                        {
                            "Symbol": t.symbol,
                            "Side": t.side,
                            "Volume": t.volume,
                            "Open Price": t.open_price,
                            "Close Price": t.close_price,
                            "Opened At": t.opened_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "Closed At": t.closed_at.strftime("%Y-%m-%d %H:%M:%S"),
                            "Duration": str(t.duration),
                            "P&L": t.profit,
                        }
                        for t in live_closed_trades
                    ]
                )

                st.dataframe(
                    history_df.style.map(_color_pnl, subset=["P&L"]), hide_index=True
                )
            else:
                st.write("No closed trades on record yet.")


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

st.header(":material/rule: Rebalance Suggestions", divider=True)
# FTMO gets its own, wider position-count/tighter per-symbol-exposure
# rules (config.FTMO_MAX_POSITION_COUNT/FTMO_MAX_SYMBOL_EXPOSURE_PCT) —
# this account is meant to genuinely diversify across several asset
# categories at once, unlike PMEX's narrower single-market book.
with st.container(border=True):
    st.caption(
        "Deterministic rule checks — stop-loss presence, position count, "
        "and symbol concentration — run against the live positions above. "
        "Independent of the AI reasoning below; nothing here requires a "
        "model call."
    )
    suggestions = evaluate_positions(
        positions,
        max_position_count=config.FTMO_MAX_POSITION_COUNT if selected_exchange == "FTMO" else None,
        max_symbol_exposure_pct=(
            config.FTMO_MAX_SYMBOL_EXPOSURE_PCT if selected_exchange == "FTMO" else None
        ),
    )
    if suggestions:
        for s in suggestions:
            st.warning(f"**[{s.action}] {s.symbol}** — {s.reason}")
    else:
        st.success("All positions are within the current rules.")

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
        if "mega_analysis_enabled_toggle" not in st.session_state:
            st.session_state["mega_analysis_enabled_toggle"] = read_mega_analysis_enabled()
        mega_analysis_enabled = st.toggle(
            "Automated daily analysis",
            key="mega_analysis_enabled_toggle",
            help="Mutually exclusive with the manual button below — only one "
            "trigger path is armed at a time.",
        )
        if mega_analysis_enabled != read_mega_analysis_enabled():
            set_mega_analysis_enabled(mega_analysis_enabled)
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
    _render_copilot_execution_panel()

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
    suggestion = None
    analyses = None

    with st.status("Building an FTMO portfolio suggestion...", expanded=True) as status:
        try:
            st.write("Fetching FTMO account and market data...")
            ftmo_account = get_account_summary()
            ftmo_assets = get_market_watch()
            # Recomputed fresh from this click's own account fetch, not
            # reused from the top-of-page snapshot — see fetch_ftmo_status's
            # own docstring for why a compliance-critical number shouldn't
            # silently reason over slightly stale data from earlier in the
            # same script run.
            fresh_ftmo_status = fetch_ftmo_status(ftmo_account)
        except MT5ConnectionError as e:
            error_message = str(e)
        else:
            if not ftmo_assets:
                warning_message = (
                    "No instruments are visible in this FTMO account's MT5 "
                    "Market Watch — add some symbols there first."
                )
            else:
                # An updating placeholder (not one st.write() line per
                # instrument) — this loop now does real per-symbol work
                # (Yahoo resolution, H4/H1 fetch, chart structure, real
                # trading cost) across up to 21+ instruments, previously
                # with zero progress visibility for the whole step.
                analysis_placeholder = st.empty()

                def _on_analysis_progress(msg: str) -> None:
                    analysis_placeholder.markdown(msg)

                analyses = analyze_ftmo_assets(ftmo_assets, on_progress=_on_analysis_progress)
                ftmo_summary_text = build_ftmo_summary(
                    ftmo_account, ftmo_assets, fresh_ftmo_status, positions=positions, analyses=analyses
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

                suggestion = suggest_ftmo_portfolio(
                    ftmo_summary_text,
                    on_stage=_on_stage,
                    on_audit_progress=_on_audit_progress,
                    model=selected_model,
                    save_record=True,
                )
                if suggestion == CLI_MISSING_MESSAGE or suggestion.startswith(CLI_FAILED_PREFIX):
                    error_message = suggestion
                else:
                    # Manual and scheduled runs differ only in what
                    # triggers them, never in what they produce or feed
                    # downstream (direct user correction 2026-08-23) —
                    # this mirrors mega_analysis_job.py's own inline
                    # wiring exactly, down to sharing the literal same
                    # lock, so a manual click and the standalone poll
                    # can never run Copilot's execution-check at
                    # once. Skips gracefully (not blocking) if the
                    # standalone poll already holds the lock right now.
                    _on_stage("Handing off to Copilot for an immediate execution-check...")
                    if acquire_lock(EXECUTION_LOCK_PATH, EXECUTION_LOCK_STALE_AFTER_SECONDS):
                        try:
                            # Same hard timeout ceiling and sentinel
                            # pattern as mega_analysis_job.py's own
                            # inline call — a hang here must never
                            # freeze this whole Streamlit session
                            # indefinitely.
                            _copilot_timed_out = object()
                            _copilot_result = run_with_timeout(
                                lambda: run_copilot_execution_check(on_stage=_on_stage),
                                config.COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS,
                                default=_copilot_timed_out,
                                catch_exceptions=False,
                            )
                            if _copilot_result is _copilot_timed_out:
                                _on_stage(
                                    "Copilot's execution-check timed out after "
                                    f"{config.COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS / 60:.0f} minutes."
                                )
                        except Exception as e:
                            _on_stage(f"Copilot's execution-check failed: {e}")
                        finally:
                            release_lock(EXECUTION_LOCK_PATH)
                    else:
                        _on_stage(
                            "Copilot's execution check is already running right now — "
                            "it will pick this up on its own next run instead."
                        )

        if error_message:
            status.update(label="Failed", state="error")
        elif warning_message:
            status.update(label="No FTMO data available", state="error")
        else:
            status.update(label="Suggestion ready", state="complete")

    # Own session_state namespace (ftmo_* rather than PMEX's unprefixed
    # keys or PSX's psx_* keys) — same isolation reasoning as PSX's own
    # comment below: a stale FTMO allocation can never make "Apply
    # Suggestion" look enabled against the wrong account, and vice versa.
    st.session_state["ftmo_suggestion_error"] = error_message
    st.session_state["ftmo_suggestion_warning"] = warning_message
    if suggestion is not None and not error_message:
        allocation = parse_final_allocation(suggestion, require_side=True)
        display_text = strip_pending_setups_block(strip_allocation_block(suggestion))
        if len(display_text) < 200 and len(suggestion) > 200:
            display_text = suggestion
        display_text = strip_leading_process_narration(display_text)
        st.session_state["ftmo_last_suggestion_text"] = display_text
        st.session_state["ftmo_last_suggestion_generated_at"] = datetime.now()
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
        elif st.session_state.get("ftmo_last_suggestion_text"):
            _ftmo_generated_at = st.session_state.get("ftmo_last_suggestion_generated_at")
            if _ftmo_generated_at is not None:
                st.badge(
                    f"Generated {_ftmo_generated_at:%Y-%m-%d %H:%M:%S}",
                    icon=":material/schedule:",
                    color="blue",
                )
            ftmo_allocation = st.session_state.get("ftmo_suggested_allocation")
            if ftmo_allocation:
                _render_allocation_chart(ftmo_allocation)
                _render_capital_at_risk_chart(ftmo_allocation, pct_is_risk=True)
            _render_ai_report(st.session_state["ftmo_last_suggestion_text"])

            # FtmoAssetAnalysis wraps a PMEX-shape AssetAnalysis as `.base` —
            # unwrap it here so _render_instrument_chart (built for that
            # shape) works unchanged, same reuse as PMEX's own chart loop.
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
        # rather than inline logic here — the unattended Copilot
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
                # "hold"/"infeasible": nothing to execute.

            st.session_state[f"{_execution_state_prefix}rebalance_plan"] = None
            # Stored rather than shown here directly: this dialog is
            # about to close via the rerun below, and the outcome
            # banner (rendered near the Apply Suggestion button,
            # above) is what the user actually sees next.
            st.session_state[f"{_execution_state_prefix}last_execution_outcomes"] = outcomes
            st.rerun()

    _show_apply_dialog(_plan, _plan_positions)
