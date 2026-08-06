import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from ai.claude_cli import CLI_FAILED_MESSAGE, CLI_MISSING_MESSAGE
from ai.narrate import build_summary, narrate
from ai.portfolio_suggest import (
    AssetAnalysis,
    analyze_assets,
    build_portfolio_summary,
    parse_final_allocation,
    strip_allocation_block,
    suggest_portfolio,
)
from data.mt5_source import MT5ConnectionError
from risk.rebalance import evaluate_positions


def _render_allocation_chart(allocation: dict[str, float]) -> None:
    fig = px.pie(names=list(allocation.keys()), values=list(allocation.values()))
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
    "Early-stage: no sizing/diversification rules are wired up for this yet, "
    "so the AI is reasoning freely here — treat it as a discussion starter, "
    "not a rule-backed recommendation. It does not currently factor in any "
    "open positions above. This performs deep live web research on Claude "
    "Opus (real searches across many sources) and internally drafts, "
    "stress-tests, and revises before answering — there's no fixed length "
    "limit, so a click can take up to 15 minutes and uses noticeably more "
    "of your Claude Pro session than the other button above."
)
if st.button("Suggest Portfolio Mix"):
    with st.spinner(
        "Building a portfolio suggestion — fetching market/macro/FX data, "
        "then letting Claude Opus research the web and reason through a "
        "full recommendation. This can take up to 15 minutes..."
    ):
        try:
            account = get_account_summary()
            assets = get_market_watch()
        except MT5ConnectionError as e:
            st.error(str(e))
        else:
            if not assets:
                st.warning(
                    "No instruments are visible in your MT5 Market Watch — "
                    "add some symbols there first."
                )
            else:
                analyses = analyze_assets(assets)
                portfolio_summary = build_portfolio_summary(account, assets, analyses=analyses)
                suggestion = suggest_portfolio(portfolio_summary)

                if suggestion in (CLI_FAILED_MESSAGE, CLI_MISSING_MESSAGE):
                    # The claude -p call itself failed — show that clearly
                    # and stop, rather than rendering research charts next
                    # to an easy-to-miss one-line error (which is exactly
                    # what "just charts, no suggestion" looked like).
                    st.error(suggestion)
                else:
                    allocation = parse_final_allocation(suggestion)
                    if allocation:
                        _render_allocation_chart(allocation)

                    display_text = strip_allocation_block(suggestion)
                    if len(display_text) < 200 and len(suggestion) > 200:
                        # Stripping removed almost everything — likely the
                        # model used more than one fenced block in a way
                        # that confused extraction. Never silently hide
                        # the actual answer: show the raw response instead.
                        display_text = suggestion
                    st.markdown(display_text)

                    chartable = [a for a in analyses if a.display_name is not None and not a.prices.empty]
                    if chartable:
                        with st.expander(f"Instrument charts ({len(chartable)})"):
                            for a in chartable:
                                _render_instrument_chart(a)
