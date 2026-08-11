import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE
from ai.narrate import build_summary, narrate
from ai.portfolio_suggest import (
    AUDIT_MODELS,
    AllocationEntry,
    AssetAnalysis,
    analyze_assets,
    build_portfolio_summary,
    parse_final_allocation,
    strip_allocation_block,
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

st.set_page_config(page_title="Quant2 Advisor", layout="wide")

st.title("Quant2 Advisor")

if st.button("Refresh"):
    st.rerun()

if config.USE_MOCK_DATA:
    st.caption("Using mock data (USE_MOCK_DATA=1)")
    from data.mock_source import get_account_summary, get_market_watch, get_open_positions
else:
    from data.mt5_source import connect, get_account_summary, get_market_watch, get_open_positions

try:
    if not config.USE_MOCK_DATA:
        connect()
    positions = get_open_positions()
except MT5ConnectionError as e:
    st.error(str(e))
    st.stop()

st.header("Open Positions")
if positions:
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

st.header("Rebalance Suggestions")
suggestions = evaluate_positions(positions)
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
exchange_col, button_col, model_col, apply_col = st.columns([1, 2, 1, 1])
with exchange_col:
    selected_exchange = st.selectbox(
        "Exchange", ["PMEX", "PSX"], index=0, label_visibility="collapsed"
    )
with button_col:
    suggest_clicked = st.button("Suggest Portfolio Mix")
with model_col:
    selected_model = st.selectbox(
        "Model", ["sonnet", "opus", "haiku"], index=0, label_visibility="collapsed"
    )
with apply_col:
    # PSX has no execution avenue at all (K-Trade has no order-placement
    # API) — force-disabled regardless of session_state, on top of
    # PMEX's own suggested_allocation gate, so switching the exchange
    # dropdown can never leave this looking clickable for a market it
    # doesn't apply to.
    apply_clicked = st.button(
        "Apply Suggestion",
        disabled=(selected_exchange != "PMEX") or not st.session_state.get("suggested_allocation"),
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
                analyses = analyze_psx_assets(psx_assets, index_tag=selected_index_tag)
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
                analyses = analyze_assets(assets)
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
    allocation = st.session_state.get("suggested_allocation")
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
            st.session_state["rebalance_plan"] = plan
            st.session_state["rebalance_plan_positions"] = fresh_positions

if st.session_state.get("rebalance_plan"):
    plan = st.session_state["rebalance_plan"]
    plan_positions = st.session_state.get("rebalance_plan_positions", [])

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
                "Reason": o.reason,
            }
            for o in plan
        ]
    )
    st.dataframe(preview_df, hide_index=True)

    is_demo = bool(config.MT5_SERVER) and "demo" in config.MT5_SERVER.lower()
    if not (is_demo or config.ALLOW_LIVE_EXECUTION):
        st.error(
            "Execution blocked: MT5_SERVER doesn't look like a demo account "
            "and ALLOW_LIVE_EXECUTION isn't set. Refusing to place real "
            "orders on what may be a live account."
        )
    else:
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            confirm_clicked = st.button("Confirm and Execute", type="primary")
        with cancel_col:
            cancel_clicked = st.button("Cancel")

        if cancel_clicked:
            st.session_state["rebalance_plan"] = None
            st.rerun()

        if confirm_clicked:
            outcomes: list[tuple] = []
            for o in plan:
                if o.action in ("open", "increase"):
                    try:
                        result = open_position(o.symbol, o.side, o.volume, o.price, o.stop_loss)
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

            st.session_state["rebalance_plan"] = None
            for o, result in outcomes:
                if result.success:
                    st.success(f"{o.symbol} ({o.action}): order placed, ticket {result.ticket}")
                else:
                    st.error(
                        f"{o.symbol} ({o.action}): failed — {result.comment} "
                        f"(retcode {result.retcode})"
                    )
