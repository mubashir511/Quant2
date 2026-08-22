import time
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from ai.portfolio_suggest import (
    AUDIT_INSTRUCTION,
    AUDIT_MODELS,
    AllocationEntry,
    AssetAnalysis,
    AuditResult,
    ModelAuditStatus,
    _detect_present_commodities,
    _fetch_current_pmex_price,
    _format_audit_progress,
    _run_audit_with_retry,
    _run_copilot_with_status,
    analyze_assets,
    build_audit_block,
    build_copilot_verification,
    build_past_audit_lessons,
    build_past_lessons,
    build_past_outcome_lessons,
    build_crop_supply_demand_context,
    build_enriched_asset_context,
    build_fx_context,
    build_macro_snapshot,
    build_multi_expiry_context,
    build_portfolio_summary,
    build_positions_context,
    build_research_directives,
    build_stage1_instruction,
    build_stage2_instruction,
    format_enriched_asset_context,
    get_openrouter_audit,
    parse_final_allocation,
    strip_allocation_block,
    strip_leading_process_narration,
    suggest_portfolio,
)
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE
from ai.copilot_cli import CLI_FAILED_PREFIX as COPILOT_FAILED_PREFIX
from ai.copilot_cli import CLI_MISSING_MESSAGE as COPILOT_MISSING_MESSAGE
from ai.openrouter_client import FAILED_MESSAGE as OPENROUTER_FAILED_MESSAGE
from ai.openrouter_client import MISSING_KEY_MESSAGE as OPENROUTER_MISSING_KEY_MESSAGE
from analysis.backtest import (
    MomentumPersistenceBacktest,
    RSIReactionBacktest,
    SupportResistanceBacktest,
    VolatilityRegimeBacktest,
)
from analysis.technical import compute_technical_stats
from data.macro_source import CountryIndicators, MarketIndicators
from data.mt5_source import ContractSpec
from data.mt5_source import AccountSummary, MarketAsset, Position


@pytest.fixture(autouse=True)
def _no_real_past_lessons():
    """build_past_lessons does real file I/O (reads this actual project's
    real records/ directory, which genuinely has saved sessions from real
    runs) and real network calls (re-prices every symbol in them) —
    without this, every suggest_portfolio()-calling test in this file
    would silently pick up real production data and make real Yahoo
    Finance calls, turning a normally-instant test suite into one that
    takes minutes and depends on live network state. Same "autouse
    fixture patches a real-environment dependency to empty by default,
    individual tests override it explicitly when they need to exercise
    it" pattern already used for MT5_SYMBOL_SPECS_CSV_PATH elsewhere in
    this project's test suite."""
    with patch("ai.portfolio_suggest.build_past_lessons", return_value=""):
        yield


def _ohlc(closes: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    """Builds a High/Low/Close/Volume frame for mocking
    fetch_price_history_ohlcv — High/Low bracket Close by a small fixed
    amount and Volume is constant, enough for ATR/RSI/volume-trend to
    compute without it mattering to tests that don't assert on them."""
    index = pd.date_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "High": [c + 0.5 for c in closes],
            "Low": [c - 0.5 for c in closes],
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=index,
    )


EMPTY_MARKET_INDICATORS = MarketIndicators(
    yield_3m_pct=None,
    yield_10y_pct=None,
    yield_30y_pct=None,
    yield_curve_10y_3m_spread=None,
    dxy=None,
    vix=None,
)


@patch("ai.portfolio_suggest.fetch_country_indicators", return_value=[])
@patch("ai.portfolio_suggest.fetch_market_indicators", return_value=EMPTY_MARKET_INDICATORS)
def test_build_portfolio_summary_includes_account(mock_market, mock_country):
    account = AccountSummary(balance=10000.0, equity=9900.0, free_margin=8500.0, currency="USD")
    # Symbol with no resolvable Yahoo ticker, so no network calls are made.
    assets = [MarketAsset("XYZ999", "Unmapped instrument", bid=1.0, ask=1.1)]
    summary = build_portfolio_summary(account, assets)
    assert "10000.00 USD" in summary
    assert "XYZ999" in summary


def test_build_positions_context_empty_when_no_positions():
    assert build_positions_context([]) == ""


def test_build_positions_context_describes_each_open_position():
    positions = [
        Position(
            symbol="GO10OZ", volume=2.0, side="buy", price_open=2000.0,
            price_current=2050.0, sl=1950.0, profit=100.0,
            opened_at=datetime.now(), ticket=555,
        ),
        Position(
            symbol="SV5OZ", volume=1.0, side="sell", price_open=30.0,
            price_current=30.0, sl=None, profit=0.0,
            opened_at=datetime.now(), ticket=556,
        ),
    ]
    text = build_positions_context(positions)
    assert "GO10OZ" in text
    assert "buy" in text
    assert "2.0" in text or "2 lots" in text.lower()
    assert "stop at 1950" in text
    assert "SV5OZ" in text
    assert "no stop set" in text


@patch("ai.portfolio_suggest.fetch_country_indicators", return_value=[])
@patch("ai.portfolio_suggest.fetch_market_indicators", return_value=EMPTY_MARKET_INDICATORS)
def test_build_portfolio_summary_includes_open_positions_when_given(mock_market, mock_country):
    account = AccountSummary(balance=10000.0, equity=9900.0, free_margin=8500.0, currency="USD")
    assets = [MarketAsset("XYZ999", "Unmapped instrument", bid=1.0, ask=1.1)]
    positions = [
        Position(
            symbol="XYZ999", volume=1.0, side="buy", price_open=1.0,
            price_current=1.05, sl=None, profit=5.0,
            opened_at=datetime.now(), ticket=1,
        )
    ]
    summary = build_portfolio_summary(account, assets, positions=positions)
    assert "Current Open Positions" in summary
    assert "XYZ999" in summary


@patch("ai.portfolio_suggest.fetch_country_indicators", return_value=[])
@patch("ai.portfolio_suggest.fetch_market_indicators", return_value=EMPTY_MARKET_INDICATORS)
def test_build_portfolio_summary_omits_positions_section_when_none_held(mock_market, mock_country):
    account = AccountSummary(balance=10000.0, equity=9900.0, free_margin=8500.0, currency="USD")
    assets = [MarketAsset("XYZ999", "Unmapped instrument", bid=1.0, ask=1.1)]
    summary = build_portfolio_summary(account, assets)
    assert "Current Open Positions" not in summary


@patch("ai.portfolio_suggest.fetch_country_indicators", return_value=[])
@patch("ai.portfolio_suggest.fetch_market_indicators", return_value=EMPTY_MARKET_INDICATORS)
def test_build_portfolio_summary_handles_no_visible_assets(mock_market, mock_country):
    account = AccountSummary(balance=10000.0, equity=10000.0, free_margin=10000.0, currency="USD")
    summary = build_portfolio_summary(account, [])
    assert "none visible" in summary.lower()


@patch("ai.portfolio_suggest.fetch_country_indicators", return_value=[])
@patch("ai.portfolio_suggest.fetch_market_indicators", return_value=EMPTY_MARKET_INDICATORS)
def test_build_macro_snapshot_reports_unavailable_when_all_missing(mock_market, mock_country):
    snapshot = build_macro_snapshot()
    assert "unavailable" in snapshot.lower()


@patch("ai.portfolio_suggest.fetch_country_indicators")
@patch("ai.portfolio_suggest.fetch_market_indicators")
def test_build_macro_snapshot_includes_yields_and_country_data(mock_market, mock_country):
    mock_market.return_value = MarketIndicators(
        yield_3m_pct=3.73,
        yield_10y_pct=4.63,
        yield_30y_pct=5.19,
        yield_curve_10y_3m_spread=0.90,
        dxy=99.91,
        vix=16.37,
    )
    mock_country.return_value = [
        CountryIndicators("United States", gdp_growth_pct=2.2, inflation_pct=2.9, unemployment_pct=4.2)
    ]

    snapshot = build_macro_snapshot()
    assert "3M 3.73%" in snapshot
    assert "10Y-3M spread +0.90pp" in snapshot
    assert "VIX" in snapshot
    assert "United States" in snapshot
    assert "GDP growth 2.2%" in snapshot


def test_build_enriched_asset_context_lists_unmapped_symbol_plainly():
    assets = [MarketAsset("XYZ999", "Unmapped instrument", bid=1.0, ask=1.1)]
    context = build_enriched_asset_context(assets)
    assert "XYZ999" in context
    # No fabricated stats for an instrument with zero price-history source
    # — but the absence must be stated explicitly, not silently omitted,
    # so the model can't mistake "no line" for "nothing notable."
    assert "not available" in context.lower()
    assert "verify" in context.lower()


@patch("ai.portfolio_suggest.fetch_recent_headlines")
@patch("ai.portfolio_suggest.fetch_price_history_ohlcv")
@patch("ai.portfolio_suggest.resolve_yahoo_ticker")
def test_build_enriched_asset_context_adds_technical_and_news_for_mapped_symbol(
    mock_resolve, mock_history, mock_headlines
):
    mock_resolve.return_value = ("Gold", "GC=F")
    mock_history.return_value = _ohlc([100.0 + i for i in range(30)])
    mock_headlines.return_value = ["Gold rallies on rate cut bets"]

    assets = [MarketAsset("GO10OZ", "Gold 10oz", bid=2000.0, ask=2000.5)]
    context = build_enriched_asset_context(assets)

    assert "GO10OZ" in context
    assert "technical:" in context
    assert "uptrend" in context
    assert "RSI" in context
    assert "ATR" in context
    assert "volume" in context.lower()
    assert "Gold rallies on rate cut bets" in context


@patch("ai.portfolio_suggest.fetch_recent_headlines", return_value=[])
@patch("ai.portfolio_suggest.fetch_price_history_ohlcv")
@patch("ai.portfolio_suggest.resolve_yahoo_ticker")
def test_build_enriched_asset_context_adds_pattern_with_enough_history(
    mock_resolve, mock_history, mock_headlines
):
    mock_resolve.return_value = ("Gold", "GC=F")
    mock_history.return_value = _ohlc([100.0 + i for i in range(60)])

    assets = [MarketAsset("GO10OZ", "Gold 10oz", bid=2000.0, ask=2000.5)]
    context = build_enriched_asset_context(assets)

    assert "pattern:" in context
    assert "support" in context
    assert "resistance" in context
    assert "trending_up" in context


@patch("ai.portfolio_suggest.resolve_yahoo_ticker")
def test_build_enriched_asset_context_respects_max_enriched_cap(mock_resolve):
    import config

    original_cap = config.MAX_ENRICHED_ASSETS
    config.MAX_ENRICHED_ASSETS = 1
    try:
        mock_resolve.return_value = ("Gold", "GC=F")
        with patch("ai.portfolio_suggest.fetch_price_history_ohlcv", return_value=_ohlc([])), \
             patch("ai.portfolio_suggest.fetch_recent_headlines", return_value=[]):
            assets = [
                MarketAsset("GO10OZ", "Gold 10oz", bid=2000.0, ask=2000.5),
                MarketAsset("SV5OZ", "Silver 5oz", bid=30.0, ask=30.05),
            ]
            build_enriched_asset_context(assets)
            # Only the first asset should have triggered a resolve attempt
            # up to the cap; resolve_yahoo_ticker is still called for the
            # second (to check), but fetch calls beyond the cap should not.
            assert mock_resolve.call_count == 1
    finally:
        config.MAX_ENRICHED_ASSETS = original_cap


@patch("ai.portfolio_suggest.backtest_support_resistance_reaction")
@patch("ai.portfolio_suggest.backtest_volatility_regime")
@patch("ai.portfolio_suggest.backtest_momentum_persistence")
@patch("ai.portfolio_suggest.backtest_rsi_reaction")
@patch("ai.portfolio_suggest.fetch_recent_headlines", return_value=[])
@patch("ai.portfolio_suggest.fetch_price_history_ohlcv")
@patch("ai.portfolio_suggest.resolve_yahoo_ticker")
def test_analyze_assets_wires_backtests_onto_the_analysis(
    mock_resolve, mock_history, mock_headlines, mock_rsi_bt, mock_momentum_bt, mock_vol_bt, mock_sr_bt
):
    mock_resolve.return_value = ("Gold", "GC=F")
    mock_history.return_value = _ohlc([100.0 + i for i in range(30)])
    overbought = RSIReactionBacktest("overbought", 70.0, 6, 1, 5, 0, 16.7, -0.4, 1.5, 3.0, 10)
    oversold = RSIReactionBacktest("oversold", 30.0, 5, 4, 1, 0, 80.0, 1.4, 1.5, 3.0, 10)
    mock_rsi_bt.return_value = (overbought, oversold)
    momentum_bt = MomentumPersistenceBacktest(0.4, 20, "persistent")
    mock_momentum_bt.return_value = momentum_bt
    vol_bt = VolatilityRegimeBacktest(5.0, 15, 10.0, 20, 10)
    mock_vol_bt.return_value = vol_bt
    sr_bt = SupportResistanceBacktest(10, 7, 3, 0, 70.0, 0.8, 8, 5, 3, 0, 62.5, 0.5, 1.5, 3.0, 10)
    mock_sr_bt.return_value = sr_bt

    assets = [MarketAsset("GO10OZ", "Gold 10oz", bid=2000.0, ask=2000.5)]
    a = analyze_assets(assets)[0]

    assert a.rsi_overbought_backtest is overbought
    assert a.rsi_oversold_backtest is oversold
    assert a.momentum_persistence_backtest is momentum_bt
    assert a.volatility_regime_backtest is vol_bt
    assert a.support_resistance_backtest is sr_bt
    mock_rsi_bt.assert_called_once()
    mock_momentum_bt.assert_called_once()
    mock_vol_bt.assert_called_once()
    mock_sr_bt.assert_called_once()
    # 5y, not the old 1y default — real multi-year history is what makes
    # these backtests (which need real historical episodes, not just a
    # recent window) usable at all.
    mock_history.assert_called_once_with("GC=F", period="5y")


@patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=None)
def test_analyze_assets_reports_per_symbol_progress(mock_resolve):
    assets = [
        MarketAsset("A", "Asset A", bid=1.0, ask=1.1),
        MarketAsset("B", "Asset B", bid=2.0, ask=2.1),
    ]
    calls = []
    analyze_assets(assets, on_progress=calls.append)
    assert len(calls) == 2
    assert "1/2" in calls[0] and "A" in calls[0]
    assert "2/2" in calls[1] and "B" in calls[1]


def test_analyze_assets_without_on_progress_still_works():
    assets = [MarketAsset("A", "Asset A", bid=1.0, ask=1.1)]
    with patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=None):
        results = analyze_assets(assets)
    assert len(results) == 1


def _analysis_with_stats(symbol="GOLD-DE26"):
    history = _ohlc([100.0 + i for i in range(60)])
    prices = history["Close"]
    stats = compute_technical_stats(prices, history=history)
    return AssetAnalysis(symbol, symbol, 2000.0, 2000.5, "Gold", prices, stats, [], None)


def test_format_enriched_asset_context_includes_backtest_evidence():
    analysis = _analysis_with_stats()
    analysis.rsi_overbought_backtest = RSIReactionBacktest("overbought", 70.0, 6, 5, 1, 0, 83.3, 1.5, 1.5, 3.0, 10)
    analysis.rsi_oversold_backtest = RSIReactionBacktest("oversold", 30.0, 5, 3, 2, 0, 60.0, 0.4, 1.5, 3.0, 10)
    analysis.momentum_persistence_backtest = MomentumPersistenceBacktest(0.42, 25, "persistent")
    analysis.volatility_regime_backtest = VolatilityRegimeBacktest(8.0, 15, 4.0, 20, 10)
    analysis.support_resistance_backtest = SupportResistanceBacktest(
        10, 7, 3, 0, 70.0, 0.8, 8, 6, 2, 0, 75.0, 0.9, 1.5, 3.0, 10
    )

    text = format_enriched_asset_context([analysis])
    assert "6 distinct past episodes" in text
    assert "5 wins, 1 losses" in text
    assert "win rate 83%" in text
    assert "0.42" in text and "persistent" in text
    assert "supports the 'coiled spring' reading" in text
    assert "buying off support" in text and "shorting off resistance" in text


def test_format_enriched_asset_context_discloses_real_execution_cost_when_present():
    # Direct user request 2026-08-22: backtest results should be "more
    # realistic and dependable" by accounting for real trade cost and
    # broker guard rails — this locks in that the prose formatter
    # actually surfaces it when a backtest carries real, nonzero cost/
    # guard-rail data (e.g. FTMO's own live TradeCost, threaded in via
    # ai/ftmo_suggest.py::_real_backtest_execution_kwargs), and stays
    # silent about it when there's nothing real to disclose (the
    # existing test above, using the all-default-zero fixtures, already
    # covers that silent case).
    analysis = _analysis_with_stats()
    analysis.rsi_overbought_backtest = RSIReactionBacktest(
        "overbought", 70.0, 6, 5, 1, 0, 83.3, 1.5, 1.5, 3.0, 10,
        min_stop_distance_pct=0.15, round_trip_cost_pct=0.045, swap_pct_per_day_used=0.0006,
    )
    analysis.support_resistance_backtest = SupportResistanceBacktest(
        10, 7, 3, 0, 70.0, 0.8, 8, 6, 2, 0, 75.0, 0.9, 1.5, 3.0, 10,
        min_stop_distance_pct=0.15, round_trip_cost_pct=0.045,
        support_swap_pct_per_day_used=-0.0075, resistance_swap_pct_per_day_used=0.0003,
    )

    text = format_enriched_asset_context([analysis])
    assert "net of real 0.0450% round-trip cost" in text
    assert "stop widened to the broker's own 0.150% minimum" in text
    assert "+0.0006%/day swap" in text  # RSI oversold's own swap side
    assert "-0.0075%/day swap" in text  # support's own swap side
    assert "+0.0003%/day swap" in text  # resistance's own swap side


def test_format_enriched_asset_context_flags_volatility_regime_contradiction():
    analysis = _analysis_with_stats()
    analysis.volatility_regime_backtest = VolatilityRegimeBacktest(4.0, 15, 8.0, 20, 10)
    text = format_enriched_asset_context([analysis])
    assert "CONTRADICTS the 'coiled spring' reading" in text


def test_format_enriched_asset_context_discloses_missing_backtest_evidence():
    analysis = _analysis_with_stats()  # backtest fields default to None
    text = format_enriched_asset_context([analysis])
    assert "not enough real historical episodes" in text
    assert "not enough history to compute" in text
    assert "not enough real historical tests of these levels" in text


def test_volume_caveat_is_broader_market_for_yahoo_sourced_data():
    # data_source defaults to "yahoo" — every PMEX/PSX entry, and FTMO's
    # own Yahoo-mapped ones (e.g. XAUUSD -> GC=F).
    analysis = _analysis_with_stats()
    assert analysis.data_source == "yahoo"
    text = format_enriched_asset_context([analysis])
    assert "not this account's own order flow" in text


def test_volume_caveat_is_this_accounts_own_feed_for_mt5_sourced_data():
    # FTMO's native-D1 backfill (ai/ftmo_suggest.py::_enrich_with_native_d1)
    # for symbols with no Yahoo mapping — real gap confirmed live: 16 of
    # 17 real FTMO Market Watch symbols. Volume there genuinely IS this
    # account's own MT5 feed, not a Yahoo broader-market proxy, so the
    # caveat must say something different from the yahoo case.
    analysis = _analysis_with_stats()
    analysis.data_source = "mt5"
    text = format_enriched_asset_context([analysis])
    assert "this account's own MT5 feed" in text
    assert "not this account's own order flow" not in text


_NO_AUDIT = AuditResult(block="", audit_available=False)


@patch("ai.portfolio_suggest.build_audit_block", return_value=_NO_AUDIT)
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_calls_run_claude_twice_with_draft_then_final(mock_run_claude, mock_audit):
    mock_run_claude.side_effect = ["draft suggestion text", "final answer"]
    result = suggest_portfolio("some summary")
    assert result == "final answer"
    assert mock_run_claude.call_count == 2
    draft_prompt = mock_run_claude.call_args_list[0].args[0]
    assert "some summary" in draft_prompt


@patch("ai.portfolio_suggest.build_audit_block", return_value=_NO_AUDIT)
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_resumes_stage1_session_for_a_lean_stage2_call(mock_run_claude, mock_audit):
    # The whole point of session continuation: stage 2 must reuse stage
    # 1's own session rather than resending the huge unchanging context
    # (the market/macro summary, _INSTRUCTION_HEAD, the draft itself) a
    # second time — live-verified separately that `claude -p --resume`
    # actually retains this without repeating it.
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_portfolio("some summary")
    draft_call, final_call = mock_run_claude.call_args_list
    session_id = draft_call.kwargs.get("session_id")
    assert session_id  # a real value was generated and passed
    assert final_call.kwargs.get("resume_session_id") == session_id
    assert "resume_session_id" not in draft_call.kwargs
    final_prompt = final_call.args[0]
    assert "some summary" not in final_prompt


@patch("ai.portfolio_suggest.build_audit_block", return_value=_NO_AUDIT)
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_grants_web_search_tools_on_both_calls(mock_run_claude, mock_audit):
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_portfolio("some summary")
    for call in mock_run_claude.call_args_list:
        assert call.kwargs["allowed_tools"] == ["WebSearch", "WebFetch"]


@patch("ai.portfolio_suggest.build_audit_block", return_value=_NO_AUDIT)
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_uses_configured_model_on_both_calls(mock_run_claude, mock_audit):
    import config

    mock_run_claude.side_effect = ["draft", "final"]
    suggest_portfolio("some summary")
    for call in mock_run_claude.call_args_list:
        assert call.kwargs["model"] == config.PORTFOLIO_SUGGESTION_MODEL


@patch("ai.portfolio_suggest.build_audit_block", return_value=_NO_AUDIT)
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_uses_explicit_model_override_on_both_calls(mock_run_claude, mock_audit):
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_portfolio("some summary", model="sonnet")
    for call in mock_run_claude.call_args_list:
        assert call.kwargs["model"] == "sonnet"


@patch("ai.portfolio_suggest.build_audit_block", return_value=_NO_AUDIT)
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_uses_separate_timeout_and_revision_timeout(mock_run_claude, mock_audit):
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_portfolio("some summary", timeout=111, revision_timeout=222)
    draft_call, final_call = mock_run_claude.call_args_list
    assert draft_call.kwargs["timeout"] == 111
    assert final_call.kwargs["timeout"] == 222


@patch("ai.portfolio_suggest.build_audit_block")
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_passes_audit_block_into_lean_revision_prompt(mock_run_claude, mock_audit):
    mock_run_claude.side_effect = ["my draft text", "final answer"]
    mock_audit.return_value = AuditResult(block="Audit block: XYZ", audit_available=True)
    suggest_portfolio("some summary")
    final_prompt = mock_run_claude.call_args_list[1].args[0]
    assert "Audit block: XYZ" in final_prompt
    # The draft itself is NOT repeated — it's already in the resumed
    # session's own history, per the whole point of this design.
    assert "my draft text" not in final_prompt
    mock_audit.assert_called_once_with(
        "some summary", "my draft text", on_progress=None, past_lessons=""
    )


_SAMPLE_CLI_FAILURE = f"{CLI_FAILED_PREFIX} (the `claude` CLI exited with code 1: boom)."


@patch("ai.portfolio_suggest.build_audit_block")
@patch("ai.portfolio_suggest.run_claude", return_value=_SAMPLE_CLI_FAILURE)
def test_suggest_portfolio_returns_cli_failed_message_immediately_without_second_call(
    mock_run_claude, mock_audit
):
    result = suggest_portfolio("some summary")
    assert result == _SAMPLE_CLI_FAILURE
    assert mock_run_claude.call_count == 1
    assert mock_audit.call_count == 0


@patch("ai.portfolio_suggest.build_audit_block")
@patch("ai.portfolio_suggest.run_claude", return_value=CLI_MISSING_MESSAGE)
def test_suggest_portfolio_returns_cli_missing_message_immediately_without_second_call(
    mock_run_claude, mock_audit
):
    result = suggest_portfolio("some summary")
    assert result == CLI_MISSING_MESSAGE
    assert mock_run_claude.call_count == 1
    assert mock_audit.call_count == 0


@patch("ai.portfolio_suggest.save_portfolio_session")
@patch("ai.portfolio_suggest.build_audit_block")
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_does_not_save_record_by_default(mock_run_claude, mock_audit, mock_save):
    mock_run_claude.side_effect = ["draft", "final"]
    mock_audit.return_value = AuditResult(block="", audit_available=True)
    suggest_portfolio("some summary")
    assert mock_save.call_count == 0


@patch("ai.portfolio_suggest.save_portfolio_session")
@patch("ai.portfolio_suggest.build_audit_block")
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_saves_record_when_requested(mock_run_claude, mock_audit, mock_save):
    mock_run_claude.side_effect = ["my draft text", "final answer"]
    mock_audit.return_value = AuditResult(block="Audit block: XYZ", audit_available=True)
    suggest_portfolio("some summary", model="sonnet", save_record=True)
    assert mock_save.call_count == 1
    record = mock_save.call_args.args[0]
    assert record.summary == "some summary"
    assert record.model == "sonnet"
    assert record.draft == "my draft text"
    assert record.audit_block == "Audit block: XYZ"
    assert record.audit_available is True
    assert record.final_answer == "final answer"


@patch("ai.portfolio_suggest.save_portfolio_session")
@patch("ai.portfolio_suggest.build_audit_block")
@patch("ai.portfolio_suggest.run_claude", return_value=_SAMPLE_CLI_FAILURE)
def test_suggest_portfolio_does_not_save_record_when_draft_fails(mock_run_claude, mock_audit, mock_save):
    suggest_portfolio("some summary", save_record=True)
    assert mock_save.call_count == 0


@patch("ai.portfolio_suggest.build_audit_block")
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_invokes_on_stage_callback_with_expected_message_sequence(
    mock_run_claude, mock_audit
):
    mock_run_claude.side_effect = ["draft", "final"]
    mock_audit.return_value = AuditResult(block="", audit_available=True)
    messages = []
    suggest_portfolio("some summary", on_stage=messages.append)
    assert len(messages) == 4
    assert "past-session context" in messages[0].lower()
    assert "drafting" in messages[1].lower()
    assert "audit" in messages[2].lower()
    assert "revising" in messages[3].lower()


@patch("ai.portfolio_suggest.build_audit_block")
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_on_stage_discloses_when_audit_unavailable(mock_run_claude, mock_audit):
    mock_run_claude.side_effect = ["draft", "final"]
    mock_audit.return_value = AuditResult(block="", audit_available=False)
    messages = []
    suggest_portfolio("some summary", on_stage=messages.append)
    assert "wasn't available" in messages[3] or "re-checking" in messages[3].lower()


@patch("ai.portfolio_suggest.build_audit_block")
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_uses_synthesize_role_when_audit_available_true(mock_run_claude, mock_audit):
    # build_stage2_instruction (the full, non-lean builder) is only used
    # by the fallback path now — the normal, successful path picks
    # between the two _CONTINUED role variants directly.
    mock_run_claude.side_effect = ["draft", "final"]
    mock_audit.return_value = AuditResult(block="", audit_available=True)
    suggest_portfolio("some summary")
    final_prompt = mock_run_claude.call_args_list[1].args[0]
    assert "WEIGH EACH AUDIT BY ITS SOURCE MODEL" in final_prompt
    assert "already above in this conversation" in final_prompt


@patch("ai.portfolio_suggest.build_audit_block")
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_uses_self_review_role_when_audit_available_false(mock_run_claude, mock_audit):
    mock_run_claude.side_effect = ["draft", "final"]
    mock_audit.return_value = AuditResult(block="", audit_available=False)
    suggest_portfolio("some summary")
    final_prompt = mock_run_claude.call_args_list[1].args[0]
    assert "independent audit that normally reviews it was not available this run" in final_prompt
    assert "WEIGH EACH AUDIT BY ITS SOURCE MODEL" not in final_prompt


@patch("ai.portfolio_suggest.build_audit_block", return_value=_NO_AUDIT)
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_falls_back_to_full_context_when_resume_fails(mock_run_claude, mock_audit):
    # A resumed session can fail for reasons unrelated to content quality
    # (expired/evicted session, a CLI without --resume support) — that
    # must never degrade the final answer, so a failed lean call triggers
    # one full-context retry with no session dependency at all.
    fallback_failure = f"{CLI_FAILED_PREFIX} (the `claude` CLI exited with code 1: no conversation found)."
    mock_run_claude.side_effect = ["draft suggestion text", fallback_failure, "final answer after retry"]
    result = suggest_portfolio("some summary")
    assert result == "final answer after retry"
    assert mock_run_claude.call_count == 3
    retry_call = mock_run_claude.call_args_list[2]
    assert "resume_session_id" not in retry_call.kwargs
    retry_prompt = retry_call.args[0]
    assert "some summary" in retry_prompt
    assert "draft suggestion text" in retry_prompt


@patch("ai.portfolio_suggest.resolve_yahoo_ticker")
def test_detect_present_commodities_collects_resolved_names(mock_resolve):
    def side_effect(symbol, description):
        return {"WHEAT-JY26": ("Wheat", "ZW=F"), "GO10OZ": ("Gold", "GC=F")}.get(symbol)

    mock_resolve.side_effect = side_effect
    assets = [
        MarketAsset("WHEAT-JY26", "Wheat", bid=1.0, ask=1.1),
        MarketAsset("GO10OZ", "Gold 10oz", bid=2000.0, ask=2000.5),
        MarketAsset("XYZ999", "Unmapped", bid=1.0, ask=1.1),
    ]
    assert _detect_present_commodities(assets) == {"Wheat", "Gold"}


@patch("ai.portfolio_suggest.fetch_crop_supply_demand_context", return_value="Wheat supply/demand context:\n- Exporter Russia...")
@patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=("Wheat", "ZW=F"))
def test_build_crop_supply_demand_context_calls_fetch_when_crop_present(mock_resolve, mock_fetch):
    assets = [MarketAsset("WHEAT-JY26", "Wheat", bid=1.0, ask=1.1)]
    context = build_crop_supply_demand_context(assets)
    assert "Wheat supply/demand context" in context
    mock_fetch.assert_called_once_with({"Wheat"})


@patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=("Gold", "GC=F"))
def test_build_crop_supply_demand_context_empty_when_no_crop_present(mock_resolve):
    assets = [MarketAsset("GO10OZ", "Gold 10oz", bid=2000.0, ask=2000.5)]
    assert build_crop_supply_demand_context(assets) == ""


@patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=("Crude Oil", "CL=F"))
def test_build_research_directives_includes_oil_chokepoints_when_energy_present(mock_resolve):
    assets = [MarketAsset("CL100BBL", "Crude Oil", bid=80.0, ask=80.1)]
    directives = build_research_directives(assets)
    assert "chokepoints" in directives.lower()
    assert "ADM" not in directives  # crop directive shouldn't appear


@patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=("Wheat", "ZW=F"))
def test_build_research_directives_includes_crop_traders_when_crop_present(mock_resolve):
    assets = [MarketAsset("WHEAT-JY26", "Wheat", bid=1.0, ask=1.1)]
    directives = build_research_directives(assets)
    assert "ADM" in directives
    assert "chokepoints" not in directives.lower()


@patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=None)
def test_build_research_directives_empty_when_nothing_resolves(mock_resolve):
    assets = [MarketAsset("XYZ999", "Unmapped instrument", bid=1.0, ask=1.1)]
    assert build_research_directives(assets) == ""


@patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=("Palladium", "PA=F"))
def test_build_research_directives_names_linked_countries_for_metal(mock_resolve):
    assets = [MarketAsset("PALDIUM100-SE26", "Palladium", bid=1300.0, ask=1305.0)]
    directives = build_research_directives(assets)
    assert "Palladium" in directives
    assert "Russia" in directives
    assert "South Africa" in directives


@patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=("S&P 500", "^GSPC"))
def test_build_research_directives_names_linked_country_for_index(mock_resolve):
    assets = [MarketAsset("SP500-SE26", "S&P 500", bid=6500.0, ask=6501.0)]
    directives = build_research_directives(assets)
    assert "S&P 500" in directives
    assert "United States" in directives


def test_build_multi_expiry_context_empty_when_no_shared_base():
    assets = [
        MarketAsset("MAIZELD-AU26", "Corn", bid=400.0, ask=400.5),
        MarketAsset("CRUDE1", "Crude Oil", bid=80.0, ask=80.1),
    ]
    assert build_multi_expiry_context(assets) == ""


def test_build_multi_expiry_context_detects_same_base_different_expiry():
    assets = [
        MarketAsset("MAIZELD-AU26", "Corn", bid=400.0, ask=400.5),
        MarketAsset("MAIZELD-JY26", "Corn", bid=410.0, ask=410.5),
    ]
    context = build_multi_expiry_context(assets)
    assert "MAIZELD" in context
    assert "MAIZELD-AU26" in context and "MAIZELD-JY26" in context
    assert "400.0" in context and "410.0" in context
    assert "roll yield" in context.lower()


def test_build_multi_expiry_context_does_not_confuse_lot_size_variants():
    # Different base names entirely (PLATINUM1 vs PLATINUM5) sharing the
    # same expiry suffix must NOT be grouped as a multi-expiry pair — they
    # differ in lot size, not contract month.
    assets = [
        MarketAsset("PLATINUM1-OC26", "Platinum", bid=1700.0, ask=1701.0),
        MarketAsset("PLATINUM5-OC26", "Platinum", bid=1700.0, ask=1701.0),
    ]
    assert build_multi_expiry_context(assets) == ""


def test_build_multi_expiry_context_ignores_symbols_without_expiry_suffix():
    assets = [
        MarketAsset("CRUDE1", "Crude Oil", bid=80.0, ask=80.1),
        MarketAsset("CRUDE10", "Crude Oil", bid=80.0, ask=80.1),
    ]
    assert build_multi_expiry_context(assets) == ""


@patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=("Gold", "GC=F"))
def test_build_research_directives_names_multiple_countries_for_gold(mock_resolve):
    assets = [MarketAsset("GO10OZ", "Gold 10oz", bid=2000.0, ask=2000.5)]
    directives = build_research_directives(assets)
    assert "China" in directives
    assert "Australia" in directives
    assert "India" in directives


def test_build_enriched_asset_context_shows_spread_for_unmapped_symbol():
    assets = [MarketAsset("XYZ999", "Unmapped instrument", bid=99.0, ask=100.0)]
    context = build_enriched_asset_context(assets)
    assert "execution:" in context
    assert "1.00%" in context  # (100-99)/100 * 100


@patch("ai.portfolio_suggest.fetch_recent_headlines", return_value=[])
@patch("ai.portfolio_suggest.fetch_price_history_ohlcv", return_value=_ohlc([]))
@patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=("Gold", "GC=F"))
def test_build_enriched_asset_context_shows_spread_for_mapped_symbol(
    mock_resolve, mock_history, mock_headlines
):
    assets = [MarketAsset("GO10OZ", "Gold 10oz", bid=1990.0, ask=2000.0)]
    context = build_enriched_asset_context(assets)
    assert "execution:" in context
    assert "0.50%" in context  # (2000-1990)/2000 * 100


def test_build_fx_context_empty_for_usd_account():
    account = AccountSummary(balance=10000.0, equity=10000.0, free_margin=10000.0, currency="USD")
    assert build_fx_context(account) == ""


@patch("ai.portfolio_suggest.fetch_fx_rate_to_usd", return_value=277.48)
def test_build_fx_context_states_rate_for_non_usd_account(mock_fetch):
    account = AccountSummary(balance=1000000.0, equity=1000000.0, free_margin=1000000.0, currency="PKR")
    context = build_fx_context(account)
    assert "PKR" in context
    assert "277.48" in context
    mock_fetch.assert_called_once_with("PKR")


@patch("ai.portfolio_suggest.fetch_fx_rate_to_usd", return_value=None)
def test_build_fx_context_still_flags_mismatch_when_rate_unavailable(mock_fetch):
    account = AccountSummary(balance=1000000.0, equity=1000000.0, free_margin=1000000.0, currency="PKR")
    context = build_fx_context(account)
    assert "PKR" in context
    assert context != ""


def test_system_instruction_covers_all_five_stress_test_failure_modes():
    lowered = build_stage1_instruction().lower()
    assert "fx" in lowered
    assert "roll yield" in lowered
    assert "contango" in lowered and "backwardation" in lowered
    assert "correlation" in lowered
    assert "execution" in lowered or "liquidity" in lowered
    assert "idle cash" in lowered


def test_system_instruction_hides_draft_stress_test_labels_from_output():
    # The reasoning steps must still happen, but not as visible headers —
    # and no hard word cap should remain (thoroughness over brevity).
    instruction = build_stage1_instruction()
    assert "DRAFT MIX" not in instruction
    assert "STRESS-TEST" not in instruction
    assert "REVISED MIX" not in instruction
    assert "do not print it as separate labeled" in instruction.lower()
    assert "no strict length limit" in instruction.lower()
    assert "under 1000 words" not in instruction.lower()


def test_system_instruction_requires_trailing_json_allocation_block():
    instruction = build_stage1_instruction()
    assert "```json" in instruction
    assert "CASH" in instruction


def test_system_instruction_mentions_short_side_support():
    instruction = build_stage1_instruction()
    assert '"side"' in instruction
    assert '"buy"' in instruction
    assert '"sell"' in instruction


@patch("ai.portfolio_suggest.fetch_recent_headlines", return_value=["Gold rallies on rate cut bets"])
@patch("ai.portfolio_suggest.fetch_price_history_ohlcv")
@patch("ai.portfolio_suggest.resolve_yahoo_ticker")
def test_analyze_assets_and_format_match_build_enriched_asset_context(
    mock_resolve, mock_history, mock_headlines
):
    mock_resolve.return_value = ("Gold", "GC=F")
    mock_history.return_value = _ohlc([100.0 + i for i in range(60)])
    assets = [MarketAsset("GO10OZ", "Gold 10oz", bid=2000.0, ask=2000.5)]

    analyses = analyze_assets(assets)
    assert len(analyses) == 1
    assert analyses[0].display_name == "Gold"
    assert not analyses[0].prices.empty

    # format_enriched_asset_context over the pre-computed analyses must
    # produce exactly what the legacy single-call wrapper produces —
    # the refactor should be a pure decomposition, not a behavior change.
    assert format_enriched_asset_context(analyses) == build_enriched_asset_context(assets)


def test_parse_final_allocation_extracts_trailing_json_block():
    # Bare-number shape (tolerated fallback — see docstring) still parses,
    # just with price/stop_loss left None.
    text = (
        "Some prose explaining the reasoning.\n\n"
        '```json\n{"GO10OZ": 15.0, "CL100BBL": 10, "CASH": 75}\n```'
    )
    allocation = parse_final_allocation(text)
    assert allocation == {
        "GO10OZ": AllocationEntry(pct=15.0),
        "CL100BBL": AllocationEntry(pct=10.0),
        "CASH": AllocationEntry(pct=75.0),
    }


def test_parse_final_allocation_extracts_object_shape_with_price_and_stop():
    text = (
        "Some prose explaining the reasoning.\n\n"
        '```json\n{"GO10OZ": {"pct": 15, "price": 2005.5, "stop_loss": 1950.0}, '
        '"CASH": 85}\n```'
    )
    allocation = parse_final_allocation(text)
    assert allocation == {
        "GO10OZ": AllocationEntry(pct=15.0, price=2005.5, stop_loss=1950.0),
        "CASH": AllocationEntry(pct=85.0),
    }


def test_parse_final_allocation_object_shape_tolerates_missing_price_and_stop():
    text = '```json\n{"GO10OZ": {"pct": 15}, "CASH": 85}\n```'
    allocation = parse_final_allocation(text)
    assert allocation["GO10OZ"] == AllocationEntry(pct=15.0, price=None, stop_loss=None)


def test_parse_final_allocation_extracts_side_buy_and_sell():
    text = (
        '```json\n{"GO10OZ": {"pct": 15, "price": 2005.5, "stop_loss": 1950.0, "side": "sell"}, '
        '"CASH": 85}\n```'
    )
    allocation = parse_final_allocation(text)
    assert allocation["GO10OZ"] == AllocationEntry(
        pct=15.0, price=2005.5, stop_loss=1950.0, side="sell"
    )


def test_parse_final_allocation_defaults_side_to_buy_when_absent():
    # Old-shape responses (and PSX's own prompt, which never emits this
    # field) must keep working exactly as before.
    text = '```json\n{"GO10OZ": {"pct": 15, "price": 2000.0}, "CASH": 85}\n```'
    allocation = parse_final_allocation(text)
    assert allocation["GO10OZ"].side == "buy"


def test_parse_final_allocation_side_is_case_insensitive():
    text = '```json\n{"GO10OZ": {"pct": 15, "price": 2000.0, "side": "SELL"}, "CASH": 85}\n```'
    allocation = parse_final_allocation(text)
    assert allocation["GO10OZ"].side == "sell"


def test_parse_final_allocation_none_when_side_is_invalid():
    # An invalid direction is a structural schema violation, not a
    # "missing, tolerate it" case — fails the whole parse rather than
    # silently guessing a direction.
    text = '```json\n{"GO10OZ": {"pct": 15, "price": 2000.0, "side": "long"}, "CASH": 85}\n```'
    assert parse_final_allocation(text) is None


def test_parse_final_allocation_require_side_true_fails_whole_parse_when_absent():
    # For FTMO/PMEX (require_side=True), an object-shape entry silently
    # missing "side" must fail the whole parse rather than default to
    # "buy" — a missing side on a symbol currently held SHORT would
    # otherwise silently look like a target to flip to long, closing a
    # real short and opening a real long on a silently-guessed field.
    text = '```json\n{"GO10OZ": {"pct": 15, "price": 2000.0, "stop_loss": 1950.0}, "CASH": 85}\n```'
    assert parse_final_allocation(text, require_side=True) is None


def test_parse_final_allocation_require_side_true_still_parses_when_present():
    text = (
        '```json\n{"GO10OZ": {"pct": 15, "price": 2000.0, "stop_loss": 1950.0, "side": "sell"}, '
        '"CASH": 85}\n```'
    )
    allocation = parse_final_allocation(text, require_side=True)
    assert allocation["GO10OZ"].side == "sell"


def test_parse_final_allocation_require_side_false_still_defaults_to_buy():
    # Default behavior (PSX, and the cold-start session-resume loader)
    # must stay exactly as before.
    text = '```json\n{"GO10OZ": {"pct": 15, "price": 2000.0}, "CASH": 85}\n```'
    allocation = parse_final_allocation(text, require_side=False)
    assert allocation["GO10OZ"].side == "buy"


def test_parse_final_allocation_require_side_true_bare_number_still_defaults_to_buy():
    # The old bare-number fallback shape (no object at all) is a
    # different tolerance than an object silently missing "side" — it
    # stays exempt from require_side since a bare number never carries
    # enough information to express a direction in the first place.
    text = '```json\n{"GO10OZ": 15, "CASH": 85}\n```'
    allocation = parse_final_allocation(text, require_side=True)
    assert allocation["GO10OZ"].side == "buy"


def test_parse_final_allocation_none_when_object_missing_pct():
    text = '```json\n{"GO10OZ": {"price": 2000.0}, "CASH": 85}\n```'
    assert parse_final_allocation(text) is None


def test_parse_final_allocation_none_when_price_not_numeric():
    text = '```json\n{"GO10OZ": {"pct": 15, "price": "high"}, "CASH": 85}\n```'
    assert parse_final_allocation(text) is None


def test_parse_final_allocation_none_when_block_missing():
    assert parse_final_allocation("Just prose, no code block.") is None


def test_parse_final_allocation_none_when_json_malformed():
    text = "```json\n{not valid json\n```"
    assert parse_final_allocation(text) is None


def test_parse_final_allocation_none_when_shape_is_wrong():
    text = '```json\n{"nested": {"a": 1}}\n```'
    assert parse_final_allocation(text) is None


def test_parse_final_allocation_uses_last_block_if_multiple():
    text = '```json\n{"WRONG": 100}\n```\nmore text\n```json\n{"RIGHT": 100}\n```'
    assert parse_final_allocation(text) == {"RIGHT": AllocationEntry(pct=100.0)}


def test_strip_allocation_block_removes_json_leaves_prose():
    text = 'Some reasoning here.\n\n```json\n{"CASH": 100}\n```'
    stripped = strip_allocation_block(text)
    assert "Some reasoning here." in stripped
    assert "```json" not in stripped
    assert "CASH" not in stripped


def test_strip_allocation_block_only_removes_the_last_block_not_earlier_ones():
    # Regression test: an earlier version used .sub() which deleted every
    # ```json block in the response, not just the final allocation one —
    # if the model used a code fence anywhere else in its reasoning, that
    # entire section vanished along with it. Only the final block (the one
    # parse_final_allocation actually uses) should ever be removed.
    text = (
        "Here is some data I found:\n\n"
        '```json\n{"note": "an example the model included mid-reasoning"}\n```\n\n'
        "More explanation follows this.\n\n"
        '```json\n{"GO10OZ": 20, "CASH": 80}\n```'
    )
    stripped = strip_allocation_block(text)
    assert "Here is some data I found:" in stripped
    assert "an example the model included mid-reasoning" in stripped
    assert "More explanation follows this." in stripped
    assert '{"GO10OZ": 20, "CASH": 80}' not in stripped
    # And parsing must still pick up the real (last) block correctly.
    assert parse_final_allocation(text) == {
        "GO10OZ": AllocationEntry(pct=20.0),
        "CASH": AllocationEntry(pct=80.0),
    }


def test_strip_leading_process_narration_removes_leaked_internal_section():
    # Regression test for a real live run (haiku): a leaked
    # "## Internal Review & Revision Process" section narrating which
    # audit findings it accepted, printed BEFORE the real report despite
    # the explicit instruction not to.
    text = (
        "I'll work through the audits and produce the final report.\n\n"
        "## Internal Review & Revision Process\n\n"
        "**Key audit findings I'm accepting:**\n1. Something internal.\n\n"
        "## Executive Summary\n\nThe real report starts here."
    )
    result = strip_leading_process_narration(text)
    assert result == "## Executive Summary\n\nThe real report starts here."
    assert "Internal Review" not in result


def test_strip_leading_process_narration_noop_when_already_clean():
    text = "## Executive Summary\n\nThe real report starts here."
    assert strip_leading_process_narration(text) == text


def test_strip_leading_process_narration_noop_when_marker_missing():
    # Can't safely guess what's leading content without the marker to
    # anchor on — leave the text untouched rather than guess wrong.
    text = "Some unusual response with no standard headers at all."
    assert strip_leading_process_narration(text) == text


def test_system_instruction_forbids_extra_code_fences():
    lowered = build_stage1_instruction().lower()
    assert "only fenced code block" in lowered


def _asset_analysis_with_spec(symbol, bid, ask, spec):
    return AssetAnalysis(symbol, symbol, bid, ask, None, contract_spec=spec)


def test_feasibility_line_flags_unaffordable_minimum_lot():
    # Reproduces the live finding: Palladium's minimum lot needs ~3.7x a
    # 1,000,000 PKR account.
    spec = ContractSpec(
        volume_min=1.0, volume_step=1.0, volume_max=1.0,
        trade_contract_size=100.0, currency_margin="PKR", margin_initial=3749300.0,
    )
    analysis = _asset_analysis_with_spec("PALDIUM100-SE26", 1350.0, 1353.0, spec)
    context = format_enriched_asset_context([analysis], account_equity=1000000.0)
    assert "feasibility:" in context
    assert "NOT AFFORDABLE" in context
    assert "374.9%" in context


def test_feasibility_line_shows_affordable_minimum_lot():
    # Reproduces the live finding: Corn's minimum lot is ~3% of the same account.
    spec = ContractSpec(
        volume_min=1.0, volume_step=1.0, volume_max=1.0,
        trade_contract_size=10000.0, currency_margin="PKR", margin_initial=30500.0,
    )
    analysis = _asset_analysis_with_spec("MAIZELD-AU26", 2430.0, 2448.0, spec)
    context = format_enriched_asset_context([analysis], account_equity=1000000.0)
    assert "feasibility:" in context
    assert "NOT AFFORDABLE" not in context
    assert "3.0%" in context


def test_feasibility_line_omitted_without_account_equity():
    spec = ContractSpec(
        volume_min=1.0, volume_step=1.0, volume_max=1.0,
        trade_contract_size=100.0, currency_margin="PKR", margin_initial=3749300.0,
    )
    analysis = _asset_analysis_with_spec("PALDIUM100-SE26", 1350.0, 1353.0, spec)
    context = format_enriched_asset_context([analysis])
    assert "feasibility:" not in context


def test_feasibility_line_discloses_when_contract_spec_missing():
    # A missing contract spec is a real data gap, not "nothing to say" —
    # confirmed live this let SP500-SE26 get a free pass on the
    # affordability check every other instrument got, silently, for a
    # full day of real runs. It must be disclosed, not omitted.
    analysis = _asset_analysis_with_spec("XYZ999", 1.0, 1.1, None)
    context = format_enriched_asset_context([analysis], account_equity=1000000.0)
    assert "feasibility:" in context
    assert "not available" in context.lower()
    assert "not been verified" in context.lower()


def test_build_enriched_asset_context_wrapper_has_no_feasibility_line():
    # The legacy no-account wrapper never had access to equity, so it
    # should keep behaving exactly as before this feature was added.
    assets = [MarketAsset("XYZ999", "Unmapped instrument", bid=1.0, ask=1.1)]
    assert "feasibility:" not in build_enriched_asset_context(assets)


def test_system_instruction_requires_feasibility_grounding():
    lowered = build_stage1_instruction().lower()
    assert "feasibility" in lowered
    assert "not affordable" in lowered
    assert "whole-lot" in lowered


def test_system_instruction_requires_trigger_time_fallback():
    assert "time-based fallback" in build_stage1_instruction().lower()


def test_system_instruction_requires_cash_opportunity_cost_research():
    lowered = build_stage1_instruction().lower()
    assert "opportunity cost" in lowered
    assert "risk-free" in lowered


def test_system_instruction_requires_volatility_aware_sizing_and_market_watch_only_hedges():
    lowered = build_stage1_instruction().lower()
    assert "volatility" in lowered and "hedge" in lowered
    assert "market watch instruments below" in lowered


@patch("ai.portfolio_suggest.run_openrouter")
def test_get_openrouter_audit_embeds_summary_and_draft_and_uses_given_model(mock_run):
    mock_run.return_value = "an audit report"
    result = get_openrouter_audit(
        "some summary", "Claude's draft suggestion text", model="openai/gpt-oss-20b:free"
    )
    assert result == "an audit report"
    prompt = mock_run.call_args.args[0]
    assert "some summary" in prompt
    assert "Claude's draft suggestion text" in prompt
    assert "do not have web search" in prompt.lower()
    assert mock_run.call_args.kwargs["model"] == "openai/gpt-oss-20b:free"


def test_audit_instruction_frames_critique_role_not_independent_opinion():
    lowered = AUDIT_INSTRUCTION.lower()
    assert "not to produce your own independent competing mix" in lowered
    assert "do not have web search" in lowered
    assert "fx" in lowered
    assert "roll yield" in lowered
    assert "correlation" in lowered
    assert "execution/liquidity" in lowered
    assert "idle-cash" in lowered
    assert "position-sizing" in lowered


@patch("ai.portfolio_suggest.get_openrouter_audit")
def test_build_audit_block_all_succeed(mock_audit):
    def audit_side_effect(summary, draft, model, **kwargs):
        return f"audit from {model}"

    mock_audit.side_effect = audit_side_effect
    result = build_audit_block("some summary", "Claude's draft mix.")
    assert result.audit_available is True
    for _, model, _ in AUDIT_MODELS:
        assert f"audit from {model}" in result.block


# These three tests exercise models that never succeed, which would
# otherwise trigger the real retry loop (config.AUDIT_RETRY_TIMEOUT_SECONDS
# // config.AUDIT_RETRY_INTERVAL_SECONDS attempts, sleeping for real
# between each). Pinning both to 1 forces max_attempts down to 1 so the
# aggregation logic under test still runs, without a multi-minute test.
@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 1)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 1)
@patch("ai.portfolio_suggest.get_openrouter_audit", return_value=OPENROUTER_FAILED_MESSAGE)
def test_build_audit_block_all_audits_fail(mock_audit):
    result = build_audit_block("some summary", "Claude's draft mix.")
    assert result.audit_available is False
    assert "not available this time" in result.block.lower()


@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 1)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 1)
@patch("ai.portfolio_suggest.get_openrouter_audit")
def test_build_audit_block_available_if_even_one_model_succeeds(mock_audit):
    # The whole point of a larger pool: audit_available stays True as
    # long as *any single* model in it responds, even if every other one
    # in the pool is down at the same time.
    working_model = AUDIT_MODELS[0][1]

    def audit_side_effect(summary, draft, model, **kwargs):
        return "a real audit" if model == working_model else OPENROUTER_FAILED_MESSAGE

    mock_audit.side_effect = audit_side_effect
    result = build_audit_block("some summary", "Claude's draft mix.")
    assert result.audit_available is True
    assert "a real audit" in result.block
    assert result.block.lower().count("not available this time") == len(AUDIT_MODELS) - 1


@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 1)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 1)
@patch("ai.portfolio_suggest.get_openrouter_audit")
def test_build_audit_block_treats_one_audit_exception_as_unavailable_without_losing_the_others(mock_audit):
    failing_model = AUDIT_MODELS[0][1]

    def audit_side_effect(summary, draft, model, **kwargs):
        if model == failing_model:
            raise RuntimeError("boom")
        return "surviving audit text"

    mock_audit.side_effect = audit_side_effect
    result = build_audit_block("some summary", "Claude's draft mix.")
    assert result.audit_available is True
    assert result.block.count("surviving audit text") == len(AUDIT_MODELS) - 1


@patch("ai.portfolio_suggest.get_openrouter_audit", return_value="an audit")
def test_build_audit_block_queries_every_audit_model(mock_audit):
    build_audit_block("some summary", "Claude's draft mix.")
    assert mock_audit.call_count == len(AUDIT_MODELS)


@patch("ai.portfolio_suggest.get_openrouter_audit", return_value="an audit")
def test_build_audit_block_labels_each_audit_with_its_source_model_profile(mock_audit):
    # Stage 2's weighting instruction only works if each model's own
    # capability/specialization profile actually reaches the prompt
    # alongside its audit text — not just the label.
    result = build_audit_block("some summary", "Claude's draft mix.")
    for label, _, profile in AUDIT_MODELS:
        assert f"{label} ({profile}) audit:" in result.block


@patch("ai.portfolio_suggest.run_copilot")
def test_build_copilot_verification_skips_subprocess_when_no_notes_section(mock_run_copilot):
    result = build_copilot_verification("Claude's draft mix with no notes section.")
    assert "nothing to independently verify" in result
    mock_run_copilot.assert_not_called()


@patch("ai.portfolio_suggest.run_copilot", return_value="SUPPORTED: found matching evidence.")
def test_build_copilot_verification_calls_run_copilot_when_notes_present(mock_run_copilot):
    draft = "Some reasoning.\n\n## External Research Notes\n- A claim — Source."
    result = build_copilot_verification(draft)
    assert result == "SUPPORTED: found matching evidence."
    prompt = mock_run_copilot.call_args.args[0]
    assert "External Research Notes" in prompt
    assert "- A claim — Source." in prompt


@patch("ai.portfolio_suggest.run_copilot", return_value="SUPPORTED: found matching evidence.")
def test_build_copilot_verification_sends_only_the_notes_section_not_the_whole_draft(mock_run_copilot):
    # Copilot's task is bounded to the listed claims — the surrounding
    # investment-thesis prose is both irrelevant to that task and the
    # larger part of the draft, so it must not be sent at all.
    draft = (
        "Some lengthy reasoning about sector allocation and position sizing "
        "that has nothing to do with fact-checking.\n\n"
        "## External Research Notes\n- A claim — Source."
    )
    build_copilot_verification(draft)
    prompt = mock_run_copilot.call_args.args[0]
    assert "sector allocation and position sizing" not in prompt
    assert "- A claim — Source." in prompt


@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 1)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 1)
@patch("ai.portfolio_suggest.get_openrouter_audit", return_value=OPENROUTER_FAILED_MESSAGE)
@patch("ai.portfolio_suggest.run_copilot", return_value="SUPPORTED: verified live.")
def test_build_audit_block_includes_successful_copilot_verification(mock_run_copilot, mock_audit):
    draft = "Draft.\n\n## External Research Notes\n- A claim — Source."
    result = build_audit_block("some summary", draft)
    assert "GitHub Copilot CLI" in result.block
    assert "SUPPORTED: verified live." in result.block
    assert result.audit_available is True  # Copilot alone can carry this, even if all 10 OpenRouter fail.


@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 1)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 1)
@patch("ai.portfolio_suggest.get_openrouter_audit", return_value=OPENROUTER_FAILED_MESSAGE)
@patch("ai.portfolio_suggest.run_copilot", return_value=COPILOT_MISSING_MESSAGE)
def test_build_audit_block_marks_copilot_unavailable_on_failure(mock_run_copilot, mock_audit):
    draft = "Draft.\n\n## External Research Notes\n- A claim — Source."
    result = build_audit_block("some summary", draft)
    assert "GitHub Copilot CLI" in result.block
    assert result.block.count("not available this time") == len(AUDIT_MODELS) + 1
    assert result.audit_available is False


@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 1)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 1)
@patch("ai.portfolio_suggest.get_openrouter_audit", return_value=OPENROUTER_FAILED_MESSAGE)
@patch("ai.portfolio_suggest.run_copilot")
def test_build_audit_block_surfaces_the_real_copilot_failure_reason(mock_run_copilot, mock_audit):
    # Regression: a Copilot failure used to collapse to a bare "not
    # available this time" everywhere (the live UI ticker AND the saved
    # session record), with the actual cause (timeout, auth failure, a
    # real exit code/stderr) visible nowhere — forcing a manual re-run
    # just to find out why.
    real_reason = f"{COPILOT_FAILED_PREFIX} (the `copilot` CLI did not respond within 240s and was terminated)."
    mock_run_copilot.return_value = real_reason
    draft = "Draft.\n\n## External Research Notes\n- A claim — Source."
    result = build_audit_block("some summary", draft)
    assert "did not respond within 240s" in result.block


@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 1)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 1)
@patch("ai.portfolio_suggest.get_openrouter_audit", return_value=OPENROUTER_FAILED_MESSAGE)
def test_build_audit_block_no_notes_case_does_not_flip_audit_available(mock_audit):
    # Regression guard: a draft with nothing for Copilot to check must NOT
    # make audit_available True on its own when every real audit failed —
    # "nothing to verify" is not the same as "a successful audit ran."
    result = build_audit_block("some summary", "Claude's draft mix with no notes section.")
    assert result.audit_available is False
    assert "nothing to independently verify" in result.block


@patch("ai.portfolio_suggest.get_openrouter_audit")
def test_build_audit_block_passes_past_lessons_to_every_model(mock_audit):
    # build_audit_block no longer computes past_lessons itself (that now
    # happens ONCE in suggest_portfolio, before stage 1, so stage 1's
    # draft can see it too — see build_past_lessons) — it just forwards
    # whatever the caller already computed to every audit model.
    def audit_side_effect(summary, draft, model, **kwargs):
        return kwargs.get("past_lessons", "")

    mock_audit.side_effect = audit_side_effect
    result = build_audit_block(
        "some summary", "Claude's draft mix.", past_lessons="margin math was wrong for SL10-SE26"
    )
    assert "margin math was wrong for SL10-SE26" in result.block


def test_build_past_audit_lessons_empty_when_records_dir_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert build_past_audit_lessons(records_dir=missing) == ""


def test_build_past_audit_lessons_empty_when_no_matching_files(tmp_path):
    (tmp_path / "not_a_record.txt").write_text("irrelevant", encoding="utf-8")
    assert build_past_audit_lessons(records_dir=tmp_path) == ""


def test_build_past_audit_lessons_extracts_stage_2_section(tmp_path):
    (tmp_path / "portfolio_suggestion_2026-08-09_120000.md").write_text(
        "# Portfolio Suggestion Session\n\n"
        "## Input data (account, market, macro summary)\n\nsome input\n\n"
        "## Stage 1 — Claude's initial draft\n\nsome draft\n\n"
        "## Stage 2 — Independent audits\n\n"
        "Nvidia audit: found a 3x margin error on SL10-SE26.\n\n"
        "## Stage 3 — Claude's final revised suggestion\n\nfinal text\n",
        encoding="utf-8",
    )

    lessons = build_past_audit_lessons(records_dir=tmp_path)
    assert "3x margin error on SL10-SE26" in lessons
    assert "some input" not in lessons  # only Stage 2 content, not the rest
    assert "final text" not in lessons


def test_build_past_audit_lessons_limits_to_max_sessions():
    # Uses a real tmp-style approach without the fixture so we can create
    # more files than max_sessions and confirm only the most recent ones
    # (by filename, which is timestamp-ordered) are included.
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = _Path(tmp)
        for i, ts in enumerate(["100000", "110000", "120000"]):
            (tmp_dir / f"portfolio_suggestion_2026-08-09_{ts}.md").write_text(
                "## Stage 2 — Independent audits\n\n"
                f"finding-{i}\n\n"
                "## Stage 3 — Claude's final revised suggestion\n\nx\n",
                encoding="utf-8",
            )
        lessons = build_past_audit_lessons(records_dir=tmp_dir, max_sessions=2)
        # Most recent two (110000, 120000) included, oldest (100000) not.
        assert "finding-1" in lessons
        assert "finding-2" in lessons
        assert "finding-0" not in lessons


def test_build_past_audit_lessons_truncates_long_sections(tmp_path):
    long_text = "x" * 5000
    (tmp_path / "portfolio_suggestion_2026-08-09_120000.md").write_text(
        f"## Stage 2 — Independent audits\n\n{long_text}\n\n"
        "## Stage 3 — Claude's final revised suggestion\n\nfinal\n",
        encoding="utf-8",
    )
    lessons = build_past_audit_lessons(records_dir=tmp_path, max_chars_per_session=100)
    assert len(lessons) < 5000


def test_build_past_audit_lessons_skips_file_missing_stage_2_section(tmp_path):
    (tmp_path / "portfolio_suggestion_2026-08-09_120000.md").write_text(
        "# No stage sections at all\n", encoding="utf-8"
    )
    assert build_past_audit_lessons(records_dir=tmp_path) == ""


_STAGE3_MARKER = "## Stage 3 — Claude's final revised suggestion\n\n"


def _write_stage3_record(tmp_path, allocation_json: str, filename="portfolio_suggestion_2026-08-09_120000.md"):
    (tmp_path / filename).write_text(f"{_STAGE3_MARKER}Some prose.\n\n```json\n{allocation_json}\n```\n", encoding="utf-8")


def test_build_past_outcome_lessons_empty_when_records_dir_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert build_past_outcome_lessons(fetch_current_price=lambda s: 100.0, records_dir=missing) == ""


def test_build_past_outcome_lessons_empty_when_no_matching_files(tmp_path):
    (tmp_path / "not_a_record.txt").write_text("irrelevant", encoding="utf-8")
    assert build_past_outcome_lessons(fetch_current_price=lambda s: 100.0, records_dir=tmp_path) == ""


def test_build_past_outcome_lessons_empty_when_no_json_allocation(tmp_path):
    (tmp_path / "portfolio_suggestion_2026-08-09_120000.md").write_text(
        f"{_STAGE3_MARKER}No JSON block here.\n", encoding="utf-8"
    )
    assert build_past_outcome_lessons(fetch_current_price=lambda s: 100.0, records_dir=tmp_path) == ""


def test_build_past_outcome_lessons_computes_real_pct_change_and_stop_status(tmp_path):
    _write_stage3_record(
        tmp_path, '{"GOLD-DE26": {"pct": 10, "price": 3400.00, "stop_loss": 3300.00}, "CASH": 90}'
    )
    result = build_past_outcome_lessons(fetch_current_price=lambda s: 3434.0, records_dir=tmp_path)
    assert "GOLD-DE26" in result
    assert "+1.0%" in result
    assert "stop not hit" in result


def test_build_past_outcome_lessons_flags_a_breached_stop(tmp_path):
    _write_stage3_record(
        tmp_path, '{"GOLD-DE26": {"pct": 10, "price": 3400.00, "stop_loss": 3300.00}, "CASH": 90}'
    )
    result = build_past_outcome_lessons(fetch_current_price=lambda s: 3250.0, records_dir=tmp_path)
    assert "STOP WOULD HAVE BEEN HIT" in result


def test_build_past_outcome_lessons_skips_cash_and_symbols_missing_price_or_stop(tmp_path):
    _write_stage3_record(
        tmp_path,
        '{"NOPRICE": {"pct": 5}, "GOLD-DE26": {"pct": 10, "price": 3400.00, "stop_loss": 3300.00}, "CASH": 85}',
    )
    calls = []

    def fetch(symbol):
        calls.append(symbol)
        return 3400.0

    result = build_past_outcome_lessons(fetch_current_price=fetch, records_dir=tmp_path)
    assert calls == ["GOLD-DE26"]  # CASH and the price-less symbol never even queried
    assert "NOPRICE" not in result


def test_build_past_outcome_lessons_skips_symbol_price_cannot_be_fetched(tmp_path):
    _write_stage3_record(
        tmp_path, '{"GOLD-DE26": {"pct": 10, "price": 3400.00, "stop_loss": 3300.00}, "CASH": 90}'
    )
    result = build_past_outcome_lessons(fetch_current_price=lambda s: None, records_dir=tmp_path)
    assert result == ""  # nothing could be re-priced — not fabricated


def test_build_past_outcome_lessons_respects_max_symbols_per_session(tmp_path):
    import json as _json

    symbols = {f"SYM{i}": {"pct": 1, "price": 100.0, "stop_loss": 90.0} for i in range(5)}
    symbols["CASH"] = 95
    _write_stage3_record(tmp_path, _json.dumps(symbols))
    result = build_past_outcome_lessons(
        fetch_current_price=lambda s: 105.0, records_dir=tmp_path, max_symbols_per_session=2
    )
    assert result.count("SYM") == 2


@patch("ai.portfolio_suggest.build_past_outcome_lessons", return_value="outcome text")
@patch("ai.portfolio_suggest.build_past_audit_lessons", return_value="audit text")
def test_build_past_lessons_combines_both_when_both_present(mock_audit_lessons, mock_outcome_lessons):
    result = build_past_lessons(fetch_current_price=lambda s: 1.0)
    assert "audit text" in result
    assert "outcome text" in result


@patch("ai.portfolio_suggest.build_past_outcome_lessons", return_value="")
@patch("ai.portfolio_suggest.build_past_audit_lessons", return_value="audit text only")
def test_build_past_lessons_omits_the_empty_half(mock_audit_lessons, mock_outcome_lessons):
    assert build_past_lessons(fetch_current_price=lambda s: 1.0) == "audit text only"


@patch("ai.portfolio_suggest.build_past_outcome_lessons", return_value="")
@patch("ai.portfolio_suggest.build_past_audit_lessons", return_value="")
def test_build_past_lessons_empty_when_both_empty(mock_audit_lessons, mock_outcome_lessons):
    assert build_past_lessons(fetch_current_price=lambda s: 1.0) == ""


@patch("ai.portfolio_suggest.fetch_price_history_ohlcv")
@patch("ai.portfolio_suggest.resolve_yahoo_ticker")
def test_fetch_current_pmex_price_returns_latest_close_over_a_short_fetch(mock_resolve, mock_history):
    mock_resolve.return_value = ("Gold", "GC=F")
    mock_history.return_value = _ohlc([100.0, 101.0, 102.5])
    assert _fetch_current_pmex_price("GO10OZ") == 102.5
    # Only the latest close is needed here — a much smaller/faster
    # request than the 5y default used for backtests.
    mock_history.assert_called_once_with("GC=F", period="5d")


@patch("ai.portfolio_suggest.resolve_yahoo_ticker", return_value=None)
def test_fetch_current_pmex_price_none_when_symbol_unresolvable(mock_resolve):
    assert _fetch_current_pmex_price("UNKNOWN123") is None


@patch("ai.portfolio_suggest.fetch_price_history_ohlcv")
@patch("ai.portfolio_suggest.resolve_yahoo_ticker")
def test_fetch_current_pmex_price_none_on_empty_history(mock_resolve, mock_history):
    mock_resolve.return_value = ("Gold", "GC=F")
    mock_history.return_value = _ohlc([])
    assert _fetch_current_pmex_price("GO10OZ") is None


@patch("ai.portfolio_suggest.build_past_lessons", return_value="PAST LESSONS TEXT")
@patch("ai.portfolio_suggest.build_audit_block", return_value=_NO_AUDIT)
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_includes_past_lessons_in_stage1_draft_prompt(mock_run_claude, mock_audit, mock_lessons):
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_portfolio("some summary")
    draft_prompt = mock_run_claude.call_args_list[0].args[0]
    assert "PAST LESSONS TEXT" in draft_prompt


@patch("ai.portfolio_suggest.build_past_lessons", return_value="PAST LESSONS TEXT")
@patch("ai.portfolio_suggest.build_audit_block")
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_passes_past_lessons_into_build_audit_block(mock_run_claude, mock_audit, mock_lessons):
    mock_run_claude.side_effect = ["draft", "final"]
    mock_audit.return_value = _NO_AUDIT
    suggest_portfolio("some summary")
    _, kwargs = mock_audit.call_args
    assert kwargs["past_lessons"] == "PAST LESSONS TEXT"


@patch("ai.portfolio_suggest.build_past_lessons", return_value="")
@patch("ai.portfolio_suggest.build_audit_block", return_value=_NO_AUDIT)
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_draft_prompt_unchanged_when_no_past_lessons(mock_run_claude, mock_audit, mock_lessons):
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_portfolio("some summary")
    draft_prompt = mock_run_claude.call_args_list[0].args[0]
    assert draft_prompt == f"{build_stage1_instruction()}\n\nsome summary"


@patch("ai.portfolio_suggest.build_past_lessons", return_value="PAST LESSONS TEXT")
@patch("ai.portfolio_suggest.build_audit_block", return_value=_NO_AUDIT)
@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_includes_past_lessons_in_fallback_retry_prompt(mock_run_claude, mock_audit, mock_lessons):
    fallback_failure = f"{CLI_FAILED_PREFIX} (no conversation found)."
    mock_run_claude.side_effect = ["draft", fallback_failure, "final after retry"]
    suggest_portfolio("some summary")
    retry_prompt = mock_run_claude.call_args_list[2].args[0]
    assert "PAST LESSONS TEXT" in retry_prompt


@patch("ai.portfolio_suggest.get_openrouter_audit", return_value="a fine audit")
def test_run_audit_with_retry_returns_immediately_on_success(mock_audit):
    label, result = _run_audit_with_retry("Label", "some/model:free", "summary", "draft")
    assert label == "Label"
    assert result == "a fine audit"
    assert mock_audit.call_count == 1


@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 600)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 60)
@patch("ai.portfolio_suggest.time.sleep")
@patch("ai.portfolio_suggest.get_openrouter_audit")
def test_run_audit_with_retry_retries_until_success(mock_audit, mock_sleep):
    mock_audit.side_effect = [OPENROUTER_FAILED_MESSAGE, OPENROUTER_FAILED_MESSAGE, "recovered audit"]
    label, result = _run_audit_with_retry("Label", "some/model:free", "summary", "draft")
    assert result == "recovered audit"
    assert mock_audit.call_count == 3
    # Slept between the two failed attempts and the eventual success, but
    # not a third time after success — no wasted wait once a model answers.
    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(60)


@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 180)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 60)
@patch("ai.portfolio_suggest.time.sleep")
@patch("ai.portfolio_suggest.get_openrouter_audit", return_value=OPENROUTER_FAILED_MESSAGE)
def test_run_audit_with_retry_gives_up_after_max_attempts(mock_audit, mock_sleep):
    # 180 // 60 = 3 attempts total, so a model that's down the entire time
    # is still eventually written off rather than retried forever.
    label, result = _run_audit_with_retry("Label", "some/model:free", "summary", "draft")
    assert result == OPENROUTER_FAILED_MESSAGE
    assert mock_audit.call_count == 3
    assert mock_sleep.call_count == 2


@patch("ai.portfolio_suggest.time.sleep")
@patch("ai.portfolio_suggest.get_openrouter_audit", return_value=OPENROUTER_MISSING_KEY_MESSAGE)
def test_run_audit_with_retry_does_not_retry_missing_key(mock_audit, mock_sleep):
    # A missing API key is a config problem, not a transient outage —
    # retrying it for 10 minutes would never help.
    label, result = _run_audit_with_retry("Label", "some/model:free", "summary", "draft")
    assert result == OPENROUTER_MISSING_KEY_MESSAGE
    assert mock_audit.call_count == 1
    mock_sleep.assert_not_called()


@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 120)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 60)
@patch("ai.portfolio_suggest.time.sleep")
@patch("ai.portfolio_suggest.get_openrouter_audit")
def test_run_audit_with_retry_treats_exception_as_failure_and_retries(mock_audit, mock_sleep):
    mock_audit.side_effect = [RuntimeError("boom"), "recovered after exception"]
    label, result = _run_audit_with_retry("Label", "some/model:free", "summary", "draft")
    assert result == "recovered after exception"
    assert mock_audit.call_count == 2
    assert mock_sleep.call_count == 1


@patch("ai.portfolio_suggest.get_openrouter_audit", return_value="a fine audit")
def test_run_audit_with_retry_updates_status_map_on_success(mock_audit):
    status_map = {}
    _run_audit_with_retry("Label", "some/model:free", "summary", "draft", status_map)
    assert status_map["Label"].state == "succeeded"
    assert status_map["Label"].attempt == 1


@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 600)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 60)
@patch("ai.portfolio_suggest.get_openrouter_audit")
def test_run_audit_with_retry_sets_retrying_status_before_sleeping(mock_audit):
    mock_audit.side_effect = [OPENROUTER_FAILED_MESSAGE, "recovered"]
    status_map = {}
    captured = {}

    def fake_sleep(seconds):
        captured["state"] = status_map["Label"].state
        captured["retry_at_is_set"] = status_map["Label"].retry_at is not None

    with patch("ai.portfolio_suggest.time.sleep", side_effect=fake_sleep):
        _run_audit_with_retry("Label", "some/model:free", "summary", "draft", status_map)

    assert captured["state"] == "retrying"
    assert captured["retry_at_is_set"] is True
    # Final status reflects the eventual outcome, not the transient retry.
    assert status_map["Label"].state == "succeeded"


@patch("config.AUDIT_RETRY_TIMEOUT_SECONDS", 120)
@patch("config.AUDIT_RETRY_INTERVAL_SECONDS", 60)
@patch("ai.portfolio_suggest.time.sleep")
@patch("ai.portfolio_suggest.get_openrouter_audit", return_value=OPENROUTER_FAILED_MESSAGE)
def test_run_audit_with_retry_sets_gave_up_status(mock_audit, mock_sleep):
    status_map = {}
    _run_audit_with_retry("Label", "some/model:free", "summary", "draft", status_map)
    assert status_map["Label"].state == "gave_up"
    assert status_map["Label"].attempt == 2  # 120 // 60 = 2 max attempts


@patch("ai.portfolio_suggest.run_copilot", return_value="SUPPORTED: verified live.")
def test_run_copilot_with_status_marks_succeeded(mock_run_copilot):
    status_map = {}
    draft = "Draft.\n\n## External Research Notes\n- A claim — Source."
    result = _run_copilot_with_status(draft, status_map)
    assert result == "SUPPORTED: verified live."
    assert status_map["Copilot CLI"].state == "succeeded"
    assert status_map["Copilot CLI"].detail is None


@patch("ai.portfolio_suggest.run_copilot", return_value=COPILOT_MISSING_MESSAGE)
def test_run_copilot_with_status_marks_gave_up_on_failure(mock_run_copilot):
    status_map = {}
    draft = "Draft.\n\n## External Research Notes\n- A claim — Source."
    _run_copilot_with_status(draft, status_map)
    assert status_map["Copilot CLI"].state == "gave_up"


def test_run_copilot_with_status_records_the_real_failure_reason():
    real_reason = f"{COPILOT_FAILED_PREFIX} (the `copilot` CLI exited with code 1: rate limited)."
    status_map = {}
    draft = "Draft.\n\n## External Research Notes\n- A claim — Source."
    with patch("ai.portfolio_suggest.run_copilot", return_value=real_reason):
        _run_copilot_with_status(draft, status_map)
    assert status_map["Copilot CLI"].detail == real_reason


def test_format_audit_progress_includes_the_gave_up_detail_when_present():
    status_map = {
        "Copilot CLI": ModelAuditStatus(
            state="gave_up", attempt=1, max_attempts=1,
            detail=f"{COPILOT_FAILED_PREFIX} (timed out after 240s).",
        ),
    }
    text = _format_audit_progress(status_map)
    assert "timed out after 240s" in text


@patch("ai.portfolio_suggest.run_copilot")
def test_run_copilot_with_status_succeeds_without_calling_copilot_when_no_notes(mock_run_copilot):
    # The zero-subprocess guard still applies — no notes section means no
    # real CLI call, but the status still resolves to a real terminal state.
    status_map = {}
    result = _run_copilot_with_status("Draft with no notes section.", status_map)
    mock_run_copilot.assert_not_called()
    assert status_map["Copilot CLI"].state == "succeeded"
    assert "nothing to independently verify" in result


def test_format_audit_progress_counts_and_lists_each_state():
    status_map = {
        "A": ModelAuditStatus(state="succeeded", attempt=1, max_attempts=10),
        "B": ModelAuditStatus(state="retrying", attempt=2, max_attempts=10, retry_at=time.monotonic() + 30),
        "C": ModelAuditStatus(state="gave_up", attempt=10, max_attempts=10),
        "D": ModelAuditStatus(state="in_progress", attempt=1, max_attempts=10),
        "E": ModelAuditStatus(state="waiting", attempt=0, max_attempts=0),
    }
    text = _format_audit_progress(status_map)
    assert "1/5 models completed" in text
    assert "A" in text and "audit received" in text
    assert "B" in text and "retrying in" in text
    assert "C" in text and "giving up" in text
    assert "D" in text and "waiting for response" in text
    assert "E" in text and "waiting to start" in text


@patch("ai.portfolio_suggest.time.sleep")
@patch("ai.portfolio_suggest.get_openrouter_audit", return_value="an audit")
def test_build_audit_block_calls_on_progress_and_reports_final_completion(mock_audit, mock_sleep):
    # +1 for Copilot CLI, which gets its own tracked status_map entry
    # alongside the OpenRouter pool — "draft" has no External Research
    # Notes section, so Copilot's own zero-subprocess guard fires and it
    # completes instantly, without a real CLI call.
    progress_calls = []
    build_audit_block("summary", "draft", on_progress=progress_calls.append)
    assert progress_calls
    total = len(AUDIT_MODELS) + 1
    assert f"{total}/{total} models completed" in progress_calls[-1]
    assert "Copilot CLI" in progress_calls[-1]


def test_build_audit_block_keeps_polling_while_copilot_is_still_running():
    # The actual bug being fixed: previously the polling loop's exit
    # condition only watched the 10 OpenRouter futures — if Copilot was
    # still running after all of them finished, on_progress simply
    # stopped being called, and the UI froze on a stale "10/10 complete"
    # render while Copilot kept working invisibly underneath it. Here
    # Copilot genuinely takes longer (a real short sleep) than the
    # instant-mocked OpenRouter calls, so a correct fix must show at
    # least one progress render with Copilot still "in progress".
    import time as real_time

    def _slow_copilot(prompt, timeout=None):
        real_time.sleep(0.3)
        return "SUPPORTED: verified live."

    progress_calls = []
    draft = "Draft.\n\n## External Research Notes\n- A claim — Source."
    with patch("ai.portfolio_suggest.get_openrouter_audit", return_value="an audit"), \
         patch("ai.portfolio_suggest.run_copilot", side_effect=_slow_copilot), \
         patch("ai.portfolio_suggest.time.sleep"):  # only the polling/retry sleep, not the real one above
        build_audit_block("summary", draft, on_progress=progress_calls.append)

    # At least one render must show every OpenRouter model already
    # succeeded while Copilot has NOT (whichever of "waiting to start" or
    # "waiting for response" its worker thread had reached by then — a
    # real, harmless scheduling race, not something worth pinning down
    # more precisely than "visibly not finished yet").
    assert any(
        f"{len(AUDIT_MODELS)}/{len(AUDIT_MODELS) + 1} models completed" in call
        and "Copilot CLI — audit received" not in call
        for call in progress_calls
    )
    # And it does eventually finish and get reported.
    assert "Copilot CLI — audit received" in progress_calls[-1]


def test_build_audit_block_without_on_progress_still_works():
    # on_progress defaults to None — the polling loop must not require it.
    with patch("ai.portfolio_suggest.get_openrouter_audit", return_value="an audit"), \
         patch("ai.portfolio_suggest.time.sleep"):
        result = build_audit_block("summary", "draft")
    assert result.audit_available is True


def test_build_stage2_instruction_audit_available_frames_synthesis_without_disclosure():
    lowered = build_stage2_instruction(True).lower()
    assert "the independent audit of my draft was not available for this run" not in lowered
    assert "revision" in lowered


def test_build_stage2_instruction_audit_unavailable_requires_disclosure_and_self_review():
    lowered = build_stage2_instruction(False).lower()
    assert "independent audit" in lowered and "not available this run" in lowered
    assert "executive summary" in lowered  # disclosed within the report's own section now


def test_build_stage1_and_stage2_share_failure_mode_checklist():
    for instruction in [build_stage1_instruction(), build_stage2_instruction(True), build_stage2_instruction(False)]:
        lowered = instruction.lower()
        assert "roll yield" in lowered
        assert "idle cash" in lowered


def test_system_instruction_directs_social_media_research():
    lowered = build_stage1_instruction().lower()
    assert "x/twitter" in lowered
    assert "reddit" in lowered
    # Must be framed as lower-confidence than official/news sources, not
    # treated as equally authoritative.
    assert "lower-confidence" in lowered
