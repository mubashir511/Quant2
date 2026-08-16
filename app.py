from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE
from ai.ftmo_suggest import analyze_ftmo_assets, build_ftmo_summary, suggest_ftmo_portfolio
from ai.narrate import build_summary, narrate
from ai.portfolio_suggest import (
    AUDIT_MODELS,
    AllocationEntry,
    AssetAnalysis,
    analyze_assets,
    build_portfolio_summary,
    parse_final_allocation,
    strip_allocation_block,
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
from data.mt5_execution import OrderResult, close_position, open_position
from data.mt5_source import MT5ConnectionError, get_contract_spec
from data.psx_source import PSX_INDICES, PSXAsset, PSXConnectionError, get_psx_market_watch
from risk.apply_suggestion import compute_rebalance_plan
from risk.ftmo_rules import FtmoStatus, compute_ftmo_status, would_breach_daily_loss_headroom
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


def _render_capital_at_risk_chart(allocation: dict[str, AllocationEntry]) -> None:
    """Visualizes the same "aggregate heat" figure the report's own risk-
    management reasoning already computes in prose (% allocation × stop-
    loss distance, summed across the mix) — needs only the allocation
    block itself (price + stop_loss per symbol), so it works identically
    for PMEX and PSX without any exchange-specific data."""
    rows = []
    for symbol, entry in allocation.items():
        if symbol == "CASH" or not entry.price or entry.stop_loss is None:
            continue
        distance_pct = abs(entry.price - entry.stop_loss) / entry.price * 100
        rows.append((symbol, entry.pct * distance_pct / 100, distance_pct))
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


def _compute_aggregate_heat_pct(allocation: dict[str, AllocationEntry]) -> float:
    """Same 'aggregate heat' math _render_capital_at_risk_chart visualizes
    (% allocation x stop-loss distance %, summed across the mix), pulled
    out as its own pure function so the FTMO pre-execution gate below can
    reuse the identical number the chart already shows for the same
    allocation, rather than a second, possibly-diverging computation."""
    total = 0.0
    for symbol, entry in allocation.items():
        if symbol == "CASH" or not entry.price or entry.stop_loss is None:
            continue
        distance_pct = abs(entry.price - entry.stop_loss) / entry.price * 100
        total += entry.pct * distance_pct / 100
    return total


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

def _fetch_ftmo_status(account_summary) -> FtmoStatus:
    """Real, freshly-computed FTMO compliance status for the given (also
    freshly-fetched) account summary — pulled out as its own function so
    both the top-of-page snapshot (for the compliance panel) and the
    Portfolio Suggestion click (which fetches its own newer account
    summary a moment later) can each get a status that matches the exact
    account data they're using, rather than the suggestion prompt
    silently reasoning over a slightly stale snapshot from earlier in
    the same script run — a real, if usually small, correctness gap for
    a compliance-critical number."""
    ftmo_deals = get_history_deals(datetime(2000, 1, 1))
    # See the top-of-page comment for why initial_balance is reconstructed
    # this way rather than read from a config constant. Excludes deals
    # with no symbol (MT5's own deposit/withdrawal/credit "balance"
    # operations, e.g. the Challenge's initial funding itself) — those
    # aren't trading P&L, and folding them in here would badly understate
    # initial_balance (see risk/ftmo_rules.py's own matching filter for
    # the full reasoning).
    initial_balance = account_summary.balance - sum(d.profit for d in ftmo_deals if d.symbol)
    return compute_ftmo_status(
        ftmo_deals,
        initial_balance=initial_balance,
        current_equity=account_summary.equity,
        current_balance=account_summary.balance,
    )


st.set_page_config(page_title="Quant2 Advisor", layout="wide")

st.title("Quant2 Advisor")

if st.button("Refresh"):
    st.rerun()

if config.USE_MOCK_DATA:
    st.caption("Using mock data (USE_MOCK_DATA=1)")
    from data.mock_source import (
        get_account_summary,
        get_history_deals,
        get_market_watch,
        get_open_positions,
    )
else:
    from data.mt5_source import (
        connect,
        get_account_summary,
        get_history_deals,
        get_market_watch,
        get_open_positions,
        is_trading_permitted,
    )

# The exchange choice drives which real account (if any) this page
# connects to below — declared here, before Open Positions, rather than
# further down where the Portfolio Suggestion section used to own it.
# Previously the Open Positions section ran unconditionally against
# PMEX's global config regardless of this dropdown (which appeared later
# in the script) — a latent bug that only became a real problem once a
# second live MT5 account (FTMO) existed to actually choose between.
selected_exchange = st.selectbox("Exchange", ["PMEX", "PSX", "FTMO"], index=0)
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
            ftmo_status = _fetch_ftmo_status(account)
    except MT5ConnectionError as e:
        st.error(str(e))
        st.stop()

st.header("Open Positions")
if selected_exchange == "PSX":
    st.caption(
        "PSX is a hypothetical, research-only avenue — there's no live "
        "broker connection (K-Trade has no order-placement API), so "
        "there are no real positions to show here."
    )
elif positions:
    df = pd.DataFrame(
        [
            {
                "Symbol": p.symbol,
                "Side": p.side,
                "Volume": p.volume,
                "Open": p.price_open,
                "Current": p.price_current,
                "Stop": p.sl if p.sl is not None else "—",
                "P&L": p.profit,
            }
            for p in positions
        ]
    )

    def _color_pnl(val):
        color = "green" if val > 0 else "red" if val < 0 else "inherit"
        return f"color: {color}"

    st.dataframe(df.style.map(_color_pnl, subset=["P&L"]), hide_index=True)
else:
    st.write("No open positions.")

if selected_exchange == "FTMO" and ftmo_status is not None:
    st.subheader("FTMO Compliance Status")
    st.caption(
        "Best-effort reconstruction from this account's own real MT5 "
        "trade history — NOT a certified mirror of FTMO's internal "
        "ledger. Cross-check against FTMO's own dashboard before relying "
        "on this near a hard limit."
    )
    heat_col1, heat_col2, heat_col3 = st.columns(3)
    heat_col1.metric("Daily-Loss Headroom", f"{ftmo_status.daily_loss_headroom_pct:.2f}%")
    heat_col2.metric("Trailing Max-Loss Headroom", f"{ftmo_status.max_loss_headroom_pct:.2f}%")
    if ftmo_status.best_day_rule_pct is not None:
        heat_col3.metric(
            "Best Day Rule", f"{ftmo_status.best_day_rule_pct:.1f}%", help="Must stay under 50%"
        )
    else:
        heat_col3.metric("Best Day Rule", "n/a", help="No positive trading day on record yet")

st.header("Rebalance Suggestions")
# FTMO gets its own, wider position-count/tighter per-symbol-exposure
# rules (config.FTMO_MAX_POSITION_COUNT/FTMO_MAX_SYMBOL_EXPOSURE_PCT) —
# this account is meant to genuinely diversify across several asset
# categories at once, unlike PMEX's narrower single-market book.
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

st.header("AI Narration")
if st.button("Get AI Narration"):
    with st.spinner("Asking the desk officer..."):
        summary = build_summary(positions, suggestions)
        commentary = narrate(summary)
    st.markdown(commentary)
else:
    st.caption("Click the button to get a plain-English narration via the local `claude` CLI.")

st.header("Portfolio Suggestion")
button_col, model_col, apply_col = st.columns([2, 1, 1])
with button_col:
    suggest_clicked = st.button("Suggest Portfolio Mix")
with model_col:
    selected_model = st.selectbox(
        "Model", ["sonnet", "opus", "haiku"], index=0, label_visibility="collapsed"
    )
with apply_col:
    # PSX has no execution avenue at all (K-Trade has no order-placement
    # API) — force-disabled regardless of session_state. PMEX/FTMO each
    # gate on their own exchange-scoped suggested_allocation key (see
    # _execution_state_prefix above) so switching the exchange dropdown
    # can never leave this looking clickable for the wrong account.
    apply_clicked = st.button(
        "Apply Suggestion",
        disabled=(selected_exchange not in ("PMEX", "FTMO"))
        or not st.session_state.get(f"{_execution_state_prefix}suggested_allocation"),
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
elif selected_exchange == "FTMO":
    st.caption(
        "Real FTMO 1-Stage Challenge account (same MT5 terminal as PMEX, "
        "fully separate balance/positions/history) — not a rule-backed "
        "recommendation, but it does factor in this account's own real "
        "compliance headroom above, its own open positions, and H1/H4/D1 "
        f"technical reads per instrument. Claude drafts a suggestion with "
        f"live web research, up to {len(AUDIT_MODELS)} free models audit "
        "it, then Claude revises. Can take up to ~35 minutes and uses "
        "significant Claude Pro usage."
    )
else:
    st.caption(
        "Early-stage and discretionary — not a rule-backed recommendation, but "
        "it does factor in your open positions above. Claude drafts a "
        f"suggestion with live web research, up to {len(AUDIT_MODELS)} free "
        "models audit it (each retried for up to 10 min if unavailable, so "
        "more of them get to weigh in), then Claude revises. Can take up to "
        "~35 minutes and uses significant Claude Pro usage."
    )

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
        display_text = strip_allocation_block(suggestion)
        if len(display_text) < 200 and len(suggestion) > 200:
            display_text = suggestion
        display_text = strip_leading_process_narration(display_text)
        st.session_state["psx_last_suggestion_text"] = display_text
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
            # reused from the top-of-page snapshot — see _fetch_ftmo_status's
            # own docstring for why a compliance-critical number shouldn't
            # silently reason over slightly stale data from earlier in the
            # same script run.
            fresh_ftmo_status = _fetch_ftmo_status(ftmo_account)
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
        allocation = parse_final_allocation(suggestion)
        display_text = strip_allocation_block(suggestion)
        if len(display_text) < 200 and len(suggestion) > 200:
            display_text = suggestion
        display_text = strip_leading_process_narration(display_text)
        st.session_state["ftmo_last_suggestion_text"] = display_text
        st.session_state["ftmo_last_suggestion_analyses"] = analyses
        if allocation:
            st.session_state["ftmo_suggested_allocation"] = allocation
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
        allocation = parse_final_allocation(suggestion)
        display_text = strip_allocation_block(suggestion)
        if len(display_text) < 200 and len(suggestion) > 200:
            # Stripping removed almost everything — likely the model used
            # more than one fenced block in a way that confused
            # extraction. Never silently hide the actual answer: show the
            # raw response instead.
            display_text = suggestion
        display_text = strip_leading_process_narration(display_text)
        st.session_state["last_suggestion_text"] = display_text
        st.session_state["last_suggestion_analyses"] = analyses
        if allocation:
            # Clearing any stale preview from a previous suggestion avoids
            # applying an old plan against a new suggestion.
            st.session_state["suggested_allocation"] = allocation
            st.session_state["rebalance_plan"] = None
    else:
        st.session_state["last_suggestion_text"] = None
        st.session_state["last_suggestion_analyses"] = None
    st.rerun()

# Rendered from session_state (not gated on suggest_clicked) so it
# survives the rerun above and keeps showing after any later, unrelated
# button click on this page (e.g. clicking "Apply Suggestion" itself).
# Each exchange has its own session_state namespace (see above), so
# switching the dropdown shows that exchange's own last result, if any.
if selected_exchange == "PSX":
    if st.session_state.get("psx_suggestion_error"):
        st.error(st.session_state["psx_suggestion_error"])
    elif st.session_state.get("psx_suggestion_warning"):
        st.warning(st.session_state["psx_suggestion_warning"])
    elif st.session_state.get("psx_last_suggestion_text"):
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

        st.markdown(st.session_state["psx_last_suggestion_text"])

        if psx_market_assets:
            _render_sector_performance_chart(psx_market_assets)
        if psx_analyses:
            _render_risk_return_scatter(psx_analyses)
            _render_week52_position_chart(psx_analyses)

        psx_chartable = [a for a in psx_analyses if a.display_name is not None and not a.prices.empty]
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
        ftmo_allocation = st.session_state.get("ftmo_suggested_allocation")
        if ftmo_allocation:
            _render_allocation_chart(ftmo_allocation)
            _render_capital_at_risk_chart(ftmo_allocation)
        st.markdown(st.session_state["ftmo_last_suggestion_text"])

        # FtmoAssetAnalysis wraps a PMEX-shape AssetAnalysis as `.base` —
        # unwrap it here so _render_instrument_chart (built for that
        # shape) works unchanged, same reuse as PMEX's own chart loop.
        ftmo_analyses = st.session_state.get("ftmo_last_suggestion_analyses") or []
        ftmo_chartable = [
            a.base for a in ftmo_analyses if a.base.display_name is not None and not a.base.prices.empty
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
        allocation = st.session_state.get("suggested_allocation")
        if allocation:
            _render_allocation_chart(allocation)
            _render_capital_at_risk_chart(allocation)
        st.markdown(st.session_state["last_suggestion_text"])

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
    plan = st.session_state[f"{_execution_state_prefix}rebalance_plan"]
    plan_positions = st.session_state.get(f"{_execution_state_prefix}rebalance_plan_positions", [])

    st.subheader("Rebalance Preview — nothing sent to MT5 yet")
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

    # Account-aware, not a single global check: PMEX and FTMO each have
    # their own MT5_SERVER-shaped config value, and only the one for the
    # account actually connected right now (per selected_exchange) is
    # relevant — checking the wrong one could either wrongly block a real
    # PMEX demo run or wrongly allow one against FTMO's real server string.
    current_server = config.FTMO_MT5_SERVER if selected_exchange == "FTMO" else config.MT5_SERVER
    is_demo = bool(current_server) and "demo" in current_server.lower()

    # FTMO-specific hard pre-execution check — PMEX has no daily-loss
    # rule to check against, so this never applies there. Computed from
    # the same allocation the capital-at-risk chart already visualizes
    # (see _compute_aggregate_heat_pct), not from `plan` itself (whose
    # PlannedOrder objects don't carry a %-of-equity figure).
    ftmo_heat_blocked = False
    if selected_exchange == "FTMO" and ftmo_status is not None:
        current_allocation = st.session_state.get(f"{_execution_state_prefix}suggested_allocation") or {}
        planned_heat_pct = _compute_aggregate_heat_pct(current_allocation)
        if would_breach_daily_loss_headroom(ftmo_status, planned_heat_pct):
            ftmo_heat_blocked = True
            allowed_pct = max(0.0, ftmo_status.daily_loss_headroom_pct) * 0.5
            st.error(
                f"Execution blocked: this plan's aggregate heat "
                f"({planned_heat_pct:.2f}% of equity at risk if every stop "
                "is hit) would exceed 50% of this account's REAL "
                f"remaining daily-loss headroom "
                f"({ftmo_status.daily_loss_headroom_pct:.2f}%, so at most "
                f"{allowed_pct:.2f}% of equity at risk is allowed right "
                "now). Reduce position sizes or wait for headroom to "
                "recover before executing — a daily-loss breach is "
                "instant termination with zero grace period."
            )

    # Real MT5-level trading permission — distinct from account
    # CONFIGURATION (is_demo/ALLOW_LIVE_EXECUTION above are policy
    # choices this app makes; this is whether the terminal will even
    # attempt to send an order right now). Confirmed live this account's
    # own AutoTrading toggle was off, which silently fails every
    # order_send() with a bare retcode — checked here, before offering
    # "Confirm and Execute" at all, so the reason is obvious up front
    # instead of discovered one rejected order at a time.
    trading_permitted, trading_blocked_reason = (
        (True, "") if config.USE_MOCK_DATA else is_trading_permitted()
    )

    if not (is_demo or config.ALLOW_LIVE_EXECUTION):
        st.error(
            "Execution blocked: the connected account's server doesn't "
            "look like a demo account and ALLOW_LIVE_EXECUTION isn't "
            "set. Refusing to place real orders on what may be a live "
            "account."
        )
    elif ftmo_heat_blocked:
        pass  # blocking message already shown above
    elif not trading_permitted:
        st.error(f"Execution blocked: {trading_blocked_reason}")
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
            for o, result in outcomes:
                if result.success:
                    st.success(f"{o.symbol} ({o.action}): order placed, ticket {result.ticket}")
                else:
                    st.error(
                        f"{o.symbol} ({o.action}): failed — {result.comment} "
                        f"(retcode {result.retcode})"
                    )
