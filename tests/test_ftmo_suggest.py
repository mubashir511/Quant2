from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE
from ai.ftmo_suggest import (
    AUDIT_INSTRUCTION,
    FtmoAssetAnalysis,
    _fetch_current_ftmo_price,
    _ftmo_commission_pct_round_turn,
    analyze_ftmo_assets,
    build_ftmo_stage1_instruction,
    build_ftmo_stage2_instruction,
    build_ftmo_summary,
    format_ftmo_asset_context,
    format_ftmo_status_context,
    format_ftmo_trade_cost,
    suggest_ftmo_portfolio,
)
from ai.portfolio_suggest import AssetAnalysis, AuditResult
from analysis.chart_structure import ChartStructureSnapshot, SRLevel, SRLevelsResult
from analysis.technical import compute_technical_stats
from data.mt5_source import AccountSummary, ContractSpec, MarketAsset, TradeCost


def _empty_chart_structure() -> ChartStructureSnapshot:
    return ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])
from risk.ftmo_rules import FtmoStatus


@pytest.fixture(autouse=True)
def _no_real_past_lessons():
    """Same rationale as ai/portfolio_suggest.py's and ai/psx_suggest.py's
    own copies of this fixture: build_past_lessons does real file I/O
    against this actual project's real records/ftmo/ directory and real
    network calls to re-price past symbols — patch it to inert by default
    so every suggest_ftmo_portfolio()-calling test stays fast and
    network-free."""
    with patch("ai.ftmo_suggest.build_past_lessons", return_value=""):
        yield


def _make_intraday_history(n=30, start=1.1000):
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    # Plain lists, not a pd.Series — a Series carries its own default
    # RangeIndex, which pd.DataFrame(..., index=dates) would align
    # against (not assign positionally), silently turning every value to
    # NaN (the exact footgun data/mt5_source.py's own
    # fetch_mt5_price_history had to work around).
    prices = [start + i * 0.0001 for i in range(n)]
    return pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [100.0] * n},
        index=dates,
    )


def _empty_history():
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"], dtype=float)


def _make_base_analysis(symbol="EURUSD", description="Euro vs US Dollar"):
    return AssetAnalysis(
        symbol=symbol, description=description, bid=1.1000, ask=1.1005, display_name=None
    )


def _make_status(**overrides):
    defaults = dict(
        daily_loss_limit_pct=3.0,
        today_realized_pl=0.0,
        today_floating_pl=0.0,
        today_total_pl=0.0,
        daily_loss_headroom_pct=3.0,
        trailing_max_loss_floor=90_000.0,
        max_loss_headroom_pct=10.0,
        best_day_pl=None,
        total_positive_days_pl=None,
        best_day_rule_pct=None,
    )
    return FtmoStatus(**{**defaults, **overrides})


# --- analyze_ftmo_assets / format_ftmo_asset_context ---


def _make_trade_cost(**overrides):
    defaults = dict(
        category="Forex",
        spread_pct_of_price=0.005,
        swap_long_pct_per_day=-0.0075,
        swap_short_pct_per_day=0.0003,
    )
    return TradeCost(**{**defaults, **overrides})


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
@patch("ai.ftmo_suggest.analyze_assets")
def test_analyze_ftmo_assets_adds_real_h4_h1_stats(
    mock_analyze_assets, mock_fetch_history, mock_trade_economics
):
    base = _make_base_analysis()
    mock_analyze_assets.return_value = [base]
    mock_fetch_history.return_value = _make_intraday_history()
    mock_trade_economics.return_value = _make_trade_cost()

    results = analyze_ftmo_assets([MarketAsset("EURUSD", "Euro vs US Dollar", 1.1000, 1.1005)])

    assert len(results) == 1
    assert results[0].base is base
    assert results[0].h4_stats.last_price is not None
    assert results[0].h1_stats.last_price is not None
    assert results[0].trade_cost is not None
    assert results[0].h4_structure is not None
    assert results[0].h1_structure is not None
    # H4 fetched before H1, both for this exact symbol.
    assert mock_fetch_history.call_args_list[0].args == ("EURUSD", "H4")
    assert mock_fetch_history.call_args_list[1].args == ("EURUSD", "H1")
    mock_trade_economics.assert_called_once_with("EURUSD")


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
@patch("ai.ftmo_suggest.analyze_assets")
@patch("ai.ftmo_suggest.compute_technical_stats")
def test_analyze_ftmo_assets_scales_volatility_correctly_per_timeframe(
    mock_compute_stats, mock_analyze_assets, mock_fetch_history, mock_trade_economics
):
    # Regression: H4/H1 stats used to be computed with compute_technical_stats'
    # own default (daily-bar) scaling, understating real annualized
    # volatility by 2.5x-6x for intraday data (see analysis/technical.py's
    # own fix). H4 and H1 need genuinely DIFFERENT periods_per_year, not
    # the same value for both.
    base = _make_base_analysis()
    mock_analyze_assets.return_value = [base]
    mock_fetch_history.return_value = _make_intraday_history()
    mock_trade_economics.return_value = _make_trade_cost()
    mock_compute_stats.return_value = compute_technical_stats(pd.Series(dtype=float))

    analyze_ftmo_assets([MarketAsset("EURUSD", "Euro vs US Dollar", 1.1000, 1.1005)])

    h4_call, h1_call = mock_compute_stats.call_args_list
    h4_periods = h4_call.kwargs["periods_per_year"]
    h1_periods = h1_call.kwargs["periods_per_year"]
    assert h4_periods > 252  # scaled for intraday, not left at the daily default
    assert h1_periods > h4_periods  # H1 bars are more frequent than H4 bars


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
@patch("ai.ftmo_suggest.analyze_assets")
def test_analyze_ftmo_assets_reports_progress_for_both_passes(
    mock_analyze_assets, mock_fetch_history, mock_trade_economics
):
    base = _make_base_analysis()
    mock_analyze_assets.side_effect = lambda assets, on_progress=None: (
        on_progress("Analyzing instruments: 1/1 — EURUSD") if on_progress else None
    ) or [base]
    mock_fetch_history.return_value = _make_intraday_history()
    mock_trade_economics.return_value = _make_trade_cost()

    calls = []
    analyze_ftmo_assets([MarketAsset("EURUSD", "Euro vs US Dollar", 1.1000, 1.1005)], on_progress=calls.append)

    # The base pass's own progress (relayed through analyze_assets'
    # on_progress param) plus this function's own extension-loop message.
    assert any("Analyzing instruments" in c for c in calls)
    assert any("Multi-timeframe technical" in c and "EURUSD" in c for c in calls)


@patch("ai.ftmo_suggest.get_trade_economics", return_value=None)
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
@patch("ai.ftmo_suggest.analyze_assets")
def test_analyze_ftmo_assets_degrades_cleanly_on_empty_intraday_history(
    mock_analyze_assets, mock_fetch_history, mock_trade_economics
):
    base = _make_base_analysis(symbol="XAUUSD", description="Gold")
    mock_analyze_assets.return_value = [base]
    mock_fetch_history.return_value = _empty_history()

    results = analyze_ftmo_assets([MarketAsset("XAUUSD", "Gold", 2000.0, 2000.5)])

    assert results[0].h4_stats.last_price is None
    assert results[0].h1_stats.last_price is None
    assert results[0].trade_cost is None
    # Empty history degrades cleanly here too — no crash, just empty structure.
    assert results[0].h4_structure.fibonacci is None
    assert results[0].h4_structure.patterns == []


def test_format_ftmo_asset_context_includes_both_intraday_timeframes():
    base = _make_base_analysis()
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=compute_technical_stats(pd.Series(dtype=float)),
        h1_stats=compute_technical_stats(pd.Series(dtype=float)),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    assert "EURUSD" in text
    assert "H4 technical" in text
    assert "H1 technical" in text
    assert "entry timing" in text  # the "use for near-term entry timing/stop placement" framing
    assert "chart structure" in text


def test_format_ftmo_asset_context_shows_real_computed_intraday_numbers():
    base = _make_base_analysis()
    history = _make_intraday_history(n=30)
    stats = compute_technical_stats(history["Close"], history=history)
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=stats,
        h1_stats=stats,
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    assert f"last {stats.last_price:.4f}" in text


def _make_trending_history(n=30, start=1.0, drift=0.01):
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    prices = [start + i * drift for i in range(n)]
    return pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [100.0] * n}, index=dates
    )


def test_format_ftmo_asset_context_shows_real_chart_structure_when_present():
    from analysis.chart_structure import ChartPattern

    base = _make_base_analysis()
    structure = ChartStructureSnapshot(
        fibonacci=None, sr_levels=None, trendlines=None,
        patterns=[ChartPattern(name="double_top", detail="two peaks at 1.2000 and 1.2010")],
    )
    # classify_setups needs a real last_price to run at all — a trending
    # H4 series gives it real stats to synthesize alongside the pattern.
    h4_history = _make_trending_history()
    h4_stats = compute_technical_stats(h4_history["Close"], history=h4_history)
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=h4_stats,
        h1_stats=compute_technical_stats(pd.Series(dtype=float)),
        h4_structure=structure,
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    assert "double_top" in text
    assert "two peaks at 1.2000 and 1.2010" in text
    # The reversal pattern should also surface through the H4 setup read.
    assert "reversal_candidate" in text


def test_format_ftmo_asset_context_reports_conflicting_mtf_trend():
    base = _make_base_analysis()
    history_up = _make_trending_history(start=1.0, drift=0.01)
    history_down = _make_trending_history(start=2.0, drift=-0.01)
    h4_stats = compute_technical_stats(history_up["Close"], history=history_up)
    h1_stats = compute_technical_stats(history_down["Close"], history=history_down)
    assert h4_stats.trend == "uptrend"
    assert h1_stats.trend == "downtrend"

    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=h4_stats,
        h1_stats=h1_stats,
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    assert "CONFLICTING" in text
    assert "H4 reads uptrend" in text
    assert "H1 reads downtrend" in text


def test_format_ftmo_asset_context_finds_real_mtf_confluence():
    base = _make_base_analysis()
    h4_sr = SRLevelsResult(resistance_levels=[SRLevel(price=1.2000, touches=3, distance_pct=1.0)], support_levels=[])
    h1_sr = SRLevelsResult(resistance_levels=[SRLevel(price=1.2003, touches=2, distance_pct=1.0)], support_levels=[])
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=compute_technical_stats(pd.Series(dtype=float)),
        h1_stats=compute_technical_stats(pd.Series(dtype=float)),
        h4_structure=ChartStructureSnapshot(fibonacci=None, sr_levels=h4_sr, trendlines=None, patterns=[]),
        h1_structure=ChartStructureSnapshot(fibonacci=None, sr_levels=h1_sr, trendlines=None, patterns=[]),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    assert "confluence levels:" in text
    assert "none found" not in text.split("confluence levels:")[1][:50]


def test_format_ftmo_asset_context_shows_confluence_zones_nearest_current_price_when_truncated():
    # Regression: find_mtf_confluence's own output is sorted by raw price
    # (for its de-duplication pass), not relevance — truncating that
    # order directly used to show the lowest-priced zones rather than
    # the ones nearest current price, when there were more than 3.
    base = _make_base_analysis()  # ask=1.1005
    # 5 confluence-worthy level pairs, spread across a wide price range —
    # the ones genuinely NEAREST 1.1005 must survive a [:3] truncation.
    h4_prices = [0.9000, 0.9500, 1.0500, 1.1000, 1.2000]
    h1_prices = [0.9001, 0.9501, 1.0501, 1.1001, 1.2001]
    h4_sr = SRLevelsResult(
        resistance_levels=[SRLevel(price=p, touches=2, distance_pct=0.0) for p in h4_prices if p > 1.1005],
        support_levels=[SRLevel(price=p, touches=2, distance_pct=0.0) for p in h4_prices if p < 1.1005],
    )
    h1_sr = SRLevelsResult(
        resistance_levels=[SRLevel(price=p, touches=2, distance_pct=0.0) for p in h1_prices if p > 1.1005],
        support_levels=[SRLevel(price=p, touches=2, distance_pct=0.0) for p in h1_prices if p < 1.1005],
    )
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=compute_technical_stats(pd.Series(dtype=float)),
        h1_stats=compute_technical_stats(pd.Series(dtype=float)),
        h4_structure=ChartStructureSnapshot(fibonacci=None, sr_levels=h4_sr, trendlines=None, patterns=[]),
        h1_structure=ChartStructureSnapshot(fibonacci=None, sr_levels=h1_sr, trendlines=None, patterns=[]),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    confluence_text = text.split("confluence levels:")[1][:200]
    # 1.1000/1.1001 (~0.04% away) and 1.0500/1.0501 (~4.6% away) are the
    # two nearest current price (1.1005) and must both survive the [:3]
    # cutoff; 0.9000/0.9001 (the lowest-priced, farthest-away pair) must
    # NOT be one of the 3 shown.
    assert "1.1000" in confluence_text
    assert "1.0501" in confluence_text  # avg of the 1.0500/1.0501 pair
    assert "0.9000" not in confluence_text
    assert "0.9500" not in confluence_text


def test_format_ftmo_asset_context_reports_partial_when_only_one_timeframe_trends():
    base = _make_base_analysis()
    h4_history = _make_trending_history(start=1.0, drift=0.01)
    h4_stats = compute_technical_stats(h4_history["Close"], history=h4_history)
    assert h4_stats.trend == "uptrend"
    h1_stats = compute_technical_stats(pd.Series([1.0] * 30))  # flat

    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=h4_stats,
        h1_stats=h1_stats,
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    assert "PARTIAL" in text
    assert "CONFLICTING" not in text


def test_format_ftmo_asset_context_reports_neither_when_both_flat():
    base = _make_base_analysis()
    flat_stats = compute_technical_stats(pd.Series([1.0] * 30))
    assert flat_stats.trend == "flat"

    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=flat_stats,
        h1_stats=flat_stats,
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    assert "NEITHER TIMEFRAME SHOWS A CLEAR TREND" in text
    assert "ALIGNED" not in text


# --- _ftmo_commission_pct_round_turn / format_ftmo_trade_cost ---


def test_commission_forex_converts_usd_per_lot_to_pct_of_notional():
    # $5.00 round-turn / (100000 * 1.1000 notional) * 100 — matches
    # FTMO's own confirmed $2.50/lot PER SIDE rate, doubled here.
    pct, note = _ftmo_commission_pct_round_turn("Forex", contract_size=100_000.0, price=1.1000)
    assert pct == pytest.approx(config.FTMO_COMMISSION_FX_USD_PER_LOT_ROUND_TURN / 110_000.0 * 100)
    assert "FTMO-confirmed" in note


def test_commission_exotics_uses_same_fx_rate():
    pct, _ = _ftmo_commission_pct_round_turn("Exotics", contract_size=100_000.0, price=17.0)
    assert pct is not None and pct > 0


def test_commission_metals_commodities_and_dxy_share_one_confirmed_rate():
    for category in ("Metals CFD", "Commodities", "Cash III CFD"):
        pct, note = _ftmo_commission_pct_round_turn(category, contract_size=100.0, price=2000.0)
        assert pct == pytest.approx(config.FTMO_COMMISSION_METALS_COMMODITIES_PCT_ROUND_TURN)
        assert "FTMO-confirmed" in note


def test_commission_crypto_flagged_as_unconfirmed():
    pct, note = _ftmo_commission_pct_round_turn("Crypto I CFD", contract_size=1.0, price=63000.0)
    assert pct == pytest.approx(config.FTMO_COMMISSION_CRYPTO_PCT_ROUND_TURN)
    assert "NOT independently confirmed" in note


def test_commission_agriculture_is_zero():
    pct, note = _ftmo_commission_pct_round_turn("Agriculture", contract_size=1.0, price=300.0)
    assert pct == 0.0
    assert "confirmed" in note


def test_commission_indices_are_zero_per_ftmo_policy():
    for category in ("Cash CFD", "Cash II CFD"):
        pct, note = _ftmo_commission_pct_round_turn(category, contract_size=1.0, price=5000.0)
        assert pct == 0.0
        assert "Zero" not in note  # just checking it's the indices branch, not literal wording
        assert "confirmed" in note.lower()


def test_commission_unknown_category_returns_none_not_a_fabricated_zero():
    pct, note = _ftmo_commission_pct_round_turn("Equities I CFD", contract_size=1.0, price=200.0)
    assert pct is None
    assert "unknown" in note.lower()


def test_format_ftmo_trade_cost_none_when_no_live_data():
    base = _make_base_analysis()
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=compute_technical_stats(pd.Series(dtype=float)),
        h1_stats=compute_technical_stats(pd.Series(dtype=float)),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    text = format_ftmo_trade_cost(analysis)
    assert "not available" in text


def test_format_ftmo_trade_cost_combines_spread_and_commission():
    spec = ContractSpec(
        volume_min=0.01, volume_step=0.01, volume_max=500.0,
        trade_contract_size=100_000.0, currency_margin="USD", margin_initial=1156.95,
    )
    base = AssetAnalysis(
        symbol="EURUSD", description="Euro vs US Dollar", bid=1.15689, ask=1.15695,
        display_name=None, contract_spec=spec,
    )
    cost = _make_trade_cost(category="Forex", spread_pct_of_price=0.0052)
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=compute_technical_stats(pd.Series(dtype=float)),
        h1_stats=compute_technical_stats(pd.Series(dtype=float)),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=cost,
    )
    text = format_ftmo_trade_cost(analysis)

    assert "REAL trading cost (Forex)" in text
    assert "0.0052%" in text
    commission_pct = config.FTMO_COMMISSION_FX_USD_PER_LOT_ROUND_TURN / (100_000.0 * 1.15695) * 100
    round_trip = 0.0052 + commission_pct
    assert f"{round_trip:.4f}%" in text
    assert "swap" in text.lower()


def test_format_ftmo_trade_cost_flags_unknown_commission_as_a_floor():
    spec = ContractSpec(
        volume_min=1.0, volume_step=1.0, volume_max=100.0,
        trade_contract_size=1.0, currency_margin="USD", margin_initial=200.0,
    )
    base = AssetAnalysis(
        symbol="AAPL", description="Apple Inc", bid=199.5, ask=200.0,
        display_name=None, contract_spec=spec,
    )
    cost = _make_trade_cost(category="Equities I CFD", spread_pct_of_price=0.25)
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=compute_technical_stats(pd.Series(dtype=float)),
        h1_stats=compute_technical_stats(pd.Series(dtype=float)),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=cost,
    )
    text = format_ftmo_trade_cost(analysis)
    assert "unknown" in text.lower()
    assert "floor" in text.lower()


# --- format_ftmo_status_context ---


def test_format_ftmo_status_context_shows_real_numbers_and_caveat():
    status = _make_status(
        today_realized_pl=-50.0,
        today_total_pl=-50.0,
        daily_loss_headroom_pct=2.5,
        trailing_max_loss_floor=94_500.0,
        max_loss_headroom_pct=9.0,
        best_day_pl=200.0,
        total_positive_days_pl=300.0,
        best_day_rule_pct=66.7,
    )
    text = format_ftmo_status_context(status)
    assert "2.50%" in text
    assert "9.00%" in text
    assert "66.7%" in text
    assert "94,500.00" in text
    assert "NOT a certified mirror" in text
    assert "cross-check against FTMO's own dashboard" in text


def test_format_ftmo_status_context_handles_no_best_day_yet():
    status = _make_status()
    text = format_ftmo_status_context(status)
    assert "not yet computable" in text


# --- build_ftmo_summary ---


@patch("ai.ftmo_suggest.build_macro_snapshot", return_value="MACRO SNAPSHOT TEXT")
@patch("ai.ftmo_suggest.format_book_wisdom", return_value="BOOK WISDOM TEXT")
def test_build_ftmo_summary_includes_status_macro_and_wisdom(mock_wisdom, mock_macro):
    account = AccountSummary(balance=100_000.0, equity=99_500.0, free_margin=90_000.0, currency="USD")
    status = _make_status(today_floating_pl=-500.0, today_total_pl=-500.0)
    summary = build_ftmo_summary(account, assets=[], ftmo_status=status, positions=[], analyses=[])
    assert "MACRO SNAPSHOT TEXT" in summary
    assert "BOOK WISDOM TEXT" in summary
    assert "FTMO Compliance Status" in summary
    assert "none visible in Market Watch" in summary


@patch("ai.ftmo_suggest.build_macro_snapshot", return_value="MACRO")
@patch("ai.ftmo_suggest.format_book_wisdom", return_value="WISDOM")
def test_build_ftmo_summary_uses_supplied_analyses_without_refetching(mock_wisdom, mock_macro):
    account = AccountSummary(balance=100_000.0, equity=100_000.0, free_margin=90_000.0, currency="USD")
    status = _make_status()
    base = _make_base_analysis()
    analyses = [
        FtmoAssetAnalysis(
            base=base,
            h4_stats=compute_technical_stats(pd.Series(dtype=float)),
            h1_stats=compute_technical_stats(pd.Series(dtype=float)),
            h4_structure=_empty_chart_structure(),
            h1_structure=_empty_chart_structure(),
            trade_cost=None,
        )
    ]
    with patch("ai.ftmo_suggest.analyze_ftmo_assets") as mock_analyze:
        summary = build_ftmo_summary(
            account,
            assets=[MarketAsset("EURUSD", "Euro", 1.1, 1.1005)],
            ftmo_status=status,
            analyses=analyses,
        )
    mock_analyze.assert_not_called()
    assert "EURUSD" in summary


# --- instruction text content ---


def test_ftmo_stage1_instruction_states_the_real_rule_numbers():
    text = build_ftmo_stage1_instruction()
    assert "3%" in text
    assert "10%" in text
    assert "50%" in text
    assert "TRAILING" in text
    assert "static" in text
    assert "ZERO grace period" in text
    assert "INSTANT termination" in text


def test_ftmo_stage1_instruction_explains_best_day_rule_as_consistency_constraint():
    text = build_ftmo_stage1_instruction()
    assert "Best Day Rule" in text
    assert "CONSISTENCY" in text


def test_ftmo_stage1_instruction_allows_weekend_overnight_and_ea_trading():
    text = build_ftmo_stage1_instruction()
    assert "ALLOWED" in text
    assert "EA/algorithmic trading is explicitly permitted" in text


def test_ftmo_stage1_instruction_mentions_multi_timeframe_framing():
    text = build_ftmo_stage1_instruction()
    assert "H4" in text
    assert "H1" in text
    assert "PRIMARY basis for the actual thesis" in text


def test_ftmo_stage1_instruction_states_short_holding_horizon():
    text = build_ftmo_stage1_instruction()
    assert "HOLDING HORIZON" in text
    assert "INTRADAY" in text
    assert "at most one trading day" in text.lower() or "at most one" in text.lower()
    # The daily read is demoted to regime context, not the thesis itself.
    assert "regime/bias context" in text


def test_ftmo_stage1_instruction_requires_per_instrument_holding_period_debate():
    text = build_ftmo_stage1_instruction()
    assert "DEBATE THE HOLDING PERIOD" in text
    # The two real inputs the debate must actually use.
    assert "swap" in text.lower()
    assert "round-trip cost" in text.lower()


def test_ftmo_stage1_instruction_requires_per_instrument_position_size_debate():
    text = build_ftmo_stage1_instruction()
    assert "DEBATE THE POSITION SIZE" in text
    # The five real inputs the sizing debate must actually weigh.
    assert "conviction" in text.lower()
    assert "feasibility ceiling" in text.lower()
    assert "compliance headroom" in text.lower()


def test_ftmo_stage1_instruction_requires_per_instrument_stop_target_debate():
    text = build_ftmo_stage1_instruction()
    assert "DEBATE THE STOP AND TARGET" in text
    assert "ATR" in text
    assert "reliability" in text.lower()


def test_ftmo_stage1_instruction_mentions_asset_category_diversification():
    text = build_ftmo_stage1_instruction()
    for category in ("forex", "metals", "indices", "crypto"):
        assert category in text.lower()
    assert "1-2 instruments" in text


def test_ftmo_stage1_instruction_states_best_effort_reconstruction_caveat():
    text = build_ftmo_stage1_instruction()
    assert "best-effort reconstruction" in text
    assert "NOT a certified mirror" in text
    assert "cross-check" in text


def test_ftmo_stage2_instruction_synth_variant_references_audits():
    text = build_ftmo_stage2_instruction(audit_available=True)
    assert "independent audit reports" in text.lower()


def test_ftmo_stage2_instruction_self_review_variant_discloses_unavailable_audit():
    text = build_ftmo_stage2_instruction(audit_available=False)
    assert "not available this run" in text.lower()


def test_ftmo_audit_instruction_covers_trailing_vs_static_and_zero_grace_period():
    assert "TRAILING" in AUDIT_INSTRUCTION
    assert "static" in AUDIT_INSTRUCTION
    assert "ZERO grace period" in AUDIT_INSTRUCTION
    assert "Best Day Rule" in AUDIT_INSTRUCTION
    assert "CONSISTENCY" in AUDIT_INSTRUCTION


def test_ftmo_audit_instruction_has_no_beta_backtest_carveout():
    # Same reasoning as PMEX — no single natural benchmark for a mixed
    # forex/metals/indices account, so a beta backtest is deliberately
    # out of scope, not just forgotten.
    assert "no beta-vs-benchmark backtest" in AUDIT_INSTRUCTION.lower()


# --- _fetch_current_ftmo_price ---


@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_fetch_current_ftmo_price_returns_latest_close(mock_fetch):
    mock_fetch.return_value = _make_intraday_history(n=5)
    price = _fetch_current_ftmo_price("EURUSD")
    assert price == pytest.approx(_make_intraday_history(n=5)["Close"].iloc[-1])
    mock_fetch.assert_called_once_with("EURUSD", "D1", count=1)


@patch("ai.ftmo_suggest.fetch_mt5_price_history", return_value=pd.DataFrame())
def test_fetch_current_ftmo_price_none_on_empty_history(mock_fetch):
    assert _fetch_current_ftmo_price("EURUSD") is None


# --- suggest_ftmo_portfolio ---


@patch("ai.ftmo_suggest.build_audit_block")
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_runs_draft_audit_revise(mock_run_claude, mock_audit):
    mock_run_claude.side_effect = ["draft text", "final text"]
    mock_audit.return_value = AuditResult(block="audit block", audit_available=True)

    result = suggest_ftmo_portfolio("some ftmo summary")

    assert result == "final text"
    assert mock_run_claude.call_count == 2
    mock_audit.assert_called_once()
    _, kwargs = mock_audit.call_args
    assert kwargs["audit_instruction"] == AUDIT_INSTRUCTION


@patch("ai.ftmo_suggest.build_past_lessons", return_value="PAST LESSONS TEXT")
@patch("ai.ftmo_suggest.build_audit_block", return_value=AuditResult(block="", audit_available=False))
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_includes_past_lessons_in_stage1_draft_prompt(
    mock_run_claude, mock_audit, mock_lessons
):
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_ftmo_portfolio("some ftmo summary")
    draft_prompt = mock_run_claude.call_args_list[0].args[0]
    assert "PAST LESSONS TEXT" in draft_prompt


@patch("ai.ftmo_suggest.build_past_lessons")
@patch("ai.ftmo_suggest.build_audit_block", return_value=AuditResult(block="", audit_available=False))
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_scopes_past_lessons_to_ftmo_records_dir(
    mock_run_claude, mock_audit, mock_lessons
):
    mock_run_claude.side_effect = ["draft", "final"]
    mock_lessons.return_value = ""
    suggest_ftmo_portfolio("summary")
    _, kwargs = mock_lessons.call_args
    assert kwargs["records_dir"] == Path(config.FTMO_RECORDS_DIR)


@patch("ai.ftmo_suggest.build_audit_block", return_value=AuditResult(block="", audit_available=False))
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_resumes_stage1_session_for_a_lean_stage2_call(mock_run_claude, mock_audit):
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_ftmo_portfolio("some ftmo summary")
    draft_call, final_call = mock_run_claude.call_args_list
    session_id = draft_call.kwargs.get("session_id")
    assert session_id
    assert final_call.kwargs.get("resume_session_id") == session_id
    final_prompt = final_call.args[0]
    assert "some ftmo summary" not in final_prompt


@patch("ai.ftmo_suggest.build_audit_block", return_value=AuditResult(block="", audit_available=False))
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_falls_back_to_full_context_when_resume_fails(mock_run_claude, mock_audit):
    fallback_failure = f"{CLI_FAILED_PREFIX} (no conversation found)."
    mock_run_claude.side_effect = ["draft text", fallback_failure, "final answer after retry"]
    messages = []
    result = suggest_ftmo_portfolio("some ftmo summary", on_stage=messages.append)
    assert result == "final answer after retry"
    assert mock_run_claude.call_count == 3
    retry_call = mock_run_claude.call_args_list[2]
    assert "resume_session_id" not in retry_call.kwargs
    retry_prompt = retry_call.args[0]
    assert "some ftmo summary" in retry_prompt
    assert "draft text" in retry_prompt
    # The retry itself is now visible to the user, not just a log line.
    assert any("retrying with a" in m.lower() for m in messages)


@patch("ai.ftmo_suggest.save_portfolio_session")
@patch("ai.ftmo_suggest.build_audit_block")
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_on_stage_message_sequence_is_detailed(mock_run_claude, mock_audit, mock_save):
    mock_run_claude.side_effect = ["draft", "final"]
    mock_audit.return_value = AuditResult(block="", audit_available=True)
    messages = []
    suggest_ftmo_portfolio("summary", on_stage=messages.append, save_record=True)
    assert any("past-session context" in m.lower() for m in messages)
    assert any("drafting" in m.lower() for m in messages)
    assert any("audit" in m.lower() for m in messages)
    assert any("revising" in m.lower() for m in messages)
    assert any("saving session record" in m.lower() for m in messages)


@patch("ai.ftmo_suggest.build_audit_block")
@patch("ai.ftmo_suggest.run_claude", return_value=CLI_MISSING_MESSAGE)
def test_suggest_ftmo_portfolio_returns_early_on_missing_cli(mock_run_claude, mock_audit):
    result = suggest_ftmo_portfolio("summary")
    assert result == CLI_MISSING_MESSAGE
    assert mock_audit.call_count == 0


@patch("ai.ftmo_suggest.build_audit_block")
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_returns_early_on_cli_failure(mock_run_claude, mock_audit):
    failure = f"{CLI_FAILED_PREFIX} (boom)."
    mock_run_claude.return_value = failure
    result = suggest_ftmo_portfolio("summary")
    assert result == failure
    assert mock_audit.call_count == 0


@patch("ai.ftmo_suggest.save_portfolio_session")
@patch("ai.ftmo_suggest.build_audit_block")
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_saves_record_with_ftmo_records_dir(mock_run_claude, mock_audit, mock_save):
    mock_run_claude.side_effect = ["draft", "final"]
    mock_audit.return_value = AuditResult(block="", audit_available=False)
    suggest_ftmo_portfolio("summary", save_record=True)
    assert mock_save.call_count == 1
    _, kwargs = mock_save.call_args
    assert kwargs["records_dir"] == Path(config.FTMO_RECORDS_DIR)
