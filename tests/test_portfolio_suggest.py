from unittest.mock import patch

import pandas as pd

from ai.portfolio_suggest import (
    SYSTEM_INSTRUCTION,
    AssetAnalysis,
    _detect_present_commodities,
    analyze_assets,
    build_crop_supply_demand_context,
    build_enriched_asset_context,
    build_fx_context,
    build_macro_snapshot,
    build_portfolio_summary,
    build_research_directives,
    format_enriched_asset_context,
    parse_final_allocation,
    strip_allocation_block,
    suggest_portfolio,
)
from data.macro_source import CountryIndicators, MarketIndicators
from data.mt5_source import ContractSpec
from data.mt5_source import AccountSummary, MarketAsset

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
    assert "technical:" not in context


@patch("ai.portfolio_suggest.fetch_recent_headlines")
@patch("ai.portfolio_suggest.fetch_price_history")
@patch("ai.portfolio_suggest.resolve_yahoo_ticker")
def test_build_enriched_asset_context_adds_technical_and_news_for_mapped_symbol(
    mock_resolve, mock_history, mock_headlines
):
    mock_resolve.return_value = ("Gold", "GC=F")
    mock_history.return_value = pd.Series(
        [100.0 + i for i in range(30)], index=pd.date_range("2026-01-01", periods=30)
    )
    mock_headlines.return_value = ["Gold rallies on rate cut bets"]

    assets = [MarketAsset("GO10OZ", "Gold 10oz", bid=2000.0, ask=2000.5)]
    context = build_enriched_asset_context(assets)

    assert "GO10OZ" in context
    assert "technical:" in context
    assert "uptrend" in context
    assert "Gold rallies on rate cut bets" in context


@patch("ai.portfolio_suggest.fetch_recent_headlines", return_value=[])
@patch("ai.portfolio_suggest.fetch_price_history")
@patch("ai.portfolio_suggest.resolve_yahoo_ticker")
def test_build_enriched_asset_context_adds_pattern_with_enough_history(
    mock_resolve, mock_history, mock_headlines
):
    mock_resolve.return_value = ("Gold", "GC=F")
    mock_history.return_value = pd.Series(
        [100.0 + i for i in range(60)], index=pd.date_range("2026-01-01", periods=60)
    )

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
        with patch("ai.portfolio_suggest.fetch_price_history", return_value=pd.Series(dtype=float)), \
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


@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_delegates_to_claude_cli_with_summary_embedded(mock_run_claude):
    mock_run_claude.return_value = "Consider a mix of gold and cash."
    result = suggest_portfolio("some summary")
    assert result == "Consider a mix of gold and cash."
    prompt = mock_run_claude.call_args.args[0]
    assert "some summary" in prompt


@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_grants_web_search_tools(mock_run_claude):
    mock_run_claude.return_value = "..."
    suggest_portfolio("some summary")
    assert mock_run_claude.call_args.kwargs["allowed_tools"] == ["WebSearch", "WebFetch"]


@patch("ai.portfolio_suggest.run_claude")
def test_suggest_portfolio_uses_configured_model(mock_run_claude):
    import config

    mock_run_claude.return_value = "..."
    suggest_portfolio("some summary")
    assert mock_run_claude.call_args.kwargs["model"] == config.PORTFOLIO_SUGGESTION_MODEL


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
@patch("ai.portfolio_suggest.fetch_price_history", return_value=pd.Series(dtype=float))
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
    lowered = SYSTEM_INSTRUCTION.lower()
    assert "fx" in lowered
    assert "roll yield" in lowered
    assert "contango" in lowered and "backwardation" in lowered
    assert "correlation" in lowered
    assert "execution" in lowered or "liquidity" in lowered
    assert "idle cash" in lowered


def test_system_instruction_hides_draft_stress_test_labels_from_output():
    # The reasoning steps must still happen, but not as visible headers —
    # and no hard word cap should remain (thoroughness over brevity).
    assert "DRAFT MIX" not in SYSTEM_INSTRUCTION
    assert "STRESS-TEST" not in SYSTEM_INSTRUCTION
    assert "REVISED MIX" not in SYSTEM_INSTRUCTION
    assert "do not print it as separate labeled" in SYSTEM_INSTRUCTION.lower()
    assert "no strict length limit" in SYSTEM_INSTRUCTION.lower()
    assert "under 1000 words" not in SYSTEM_INSTRUCTION.lower()


def test_system_instruction_requires_trailing_json_allocation_block():
    assert "```json" in SYSTEM_INSTRUCTION
    assert "CASH" in SYSTEM_INSTRUCTION


@patch("ai.portfolio_suggest.fetch_recent_headlines", return_value=["Gold rallies on rate cut bets"])
@patch("ai.portfolio_suggest.fetch_price_history")
@patch("ai.portfolio_suggest.resolve_yahoo_ticker")
def test_analyze_assets_and_format_match_build_enriched_asset_context(
    mock_resolve, mock_history, mock_headlines
):
    mock_resolve.return_value = ("Gold", "GC=F")
    mock_history.return_value = pd.Series(
        [100.0 + i for i in range(60)], index=pd.date_range("2026-01-01", periods=60)
    )
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
    text = (
        "Some prose explaining the reasoning.\n\n"
        '```json\n{"GO10OZ": 15.0, "CL100BBL": 10, "CASH": 75}\n```'
    )
    allocation = parse_final_allocation(text)
    assert allocation == {"GO10OZ": 15.0, "CL100BBL": 10.0, "CASH": 75.0}


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
    assert parse_final_allocation(text) == {"RIGHT": 100.0}


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
    assert parse_final_allocation(text) == {"GO10OZ": 20.0, "CASH": 80.0}


def test_system_instruction_forbids_extra_code_fences():
    lowered = SYSTEM_INSTRUCTION.lower()
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


def test_feasibility_line_omitted_when_contract_spec_missing():
    analysis = _asset_analysis_with_spec("XYZ999", 1.0, 1.1, None)
    context = format_enriched_asset_context([analysis], account_equity=1000000.0)
    assert "feasibility:" not in context


def test_build_enriched_asset_context_wrapper_has_no_feasibility_line():
    # The legacy no-account wrapper never had access to equity, so it
    # should keep behaving exactly as before this feature was added.
    assets = [MarketAsset("XYZ999", "Unmapped instrument", bid=1.0, ask=1.1)]
    assert "feasibility:" not in build_enriched_asset_context(assets)


def test_system_instruction_requires_feasibility_grounding():
    lowered = SYSTEM_INSTRUCTION.lower()
    assert "feasibility" in lowered
    assert "not affordable" in lowered
    assert "whole-lot" in lowered


def test_system_instruction_requires_trigger_time_fallback():
    assert "time-based fallback" in SYSTEM_INSTRUCTION.lower()


def test_system_instruction_requires_cash_opportunity_cost_research():
    lowered = SYSTEM_INSTRUCTION.lower()
    assert "opportunity cost" in lowered
    assert "risk-free" in lowered


def test_system_instruction_requires_volatility_aware_sizing_and_market_watch_only_hedges():
    lowered = SYSTEM_INSTRUCTION.lower()
    assert "volatility" in lowered and "hedge" in lowered
    assert "market watch instruments below" in lowered
