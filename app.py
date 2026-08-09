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
from data.mt5_execution import OrderResult, close_position, open_position
from data.mt5_source import MT5ConnectionError, get_contract_spec
from risk.apply_suggestion import compute_rebalance_plan
from risk.rebalance import evaluate_positions


def _render_allocation_chart(allocation: dict[str, AllocationEntry]) -> None:
    fig = px.pie(names=list(allocation.keys()), values=[e.pct for e in allocation.values()])
    fig.update_layout(title="Suggested Allocation", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _render_instrument_chart(analysis: AssetAnalysis) -> None:
    if analysis.display_name is None or analysis.prices.empty:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=analysis.prices.index, y=analysis.prices.values, mode="lines", name=analysis.symbol)
    )
    if analysis.stats.support is not None:
        fig.add_hline(y=analysis.stats.support, line_dash="dash", line_color="green", annotation_text="support")
    if analysis.stats.resistance is not None:
        fig.add_hline(
            y=analysis.stats.resistance, line_dash="dash", line_color="red", annotation_text="resistance"
        )
    fig.update_layout(
        title=f"{analysis.symbol} ({analysis.display_name})",
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.set_page_config(page_title="Quant2 Advisor", layout="centered")

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
st.caption(
    "Early-stage and discretionary — not a rule-backed recommendation, but "
    "it does factor in your open positions above. Claude drafts a "
    f"suggestion with live web research, up to {len(AUDIT_MODELS)} free "
    "models audit it (each retried for up to 10 min if unavailable, so "
    "more of them get to weigh in), then Claude revises. Can take up to "
    "~35 minutes and uses significant Claude Pro usage."
)
button_col, model_col, apply_col = st.columns([2, 1, 1])
with button_col:
    suggest_clicked = st.button("Suggest Portfolio Mix")
with model_col:
    selected_model = st.selectbox(
        "Model", ["sonnet", "opus", "haiku"], index=0, label_visibility="collapsed"
    )
with apply_col:
    apply_clicked = st.button(
        "Apply Suggestion", disabled=not st.session_state.get("suggested_allocation")
    )

if suggest_clicked:
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
