from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE
from ai.ftmo_suggest import (
    AUDIT_INSTRUCTION,
    FtmoAssetAnalysis,
    _MN1_BACKTEST_BARS,
    _enrich_with_native_d1,
    _fetch_current_ftmo_price,
    _ftmo_commission_pct_round_turn,
    _real_backtest_execution_kwargs,
    analyze_ftmo_asset_live,
    analyze_ftmo_assets,
    build_ftmo_stage1_instruction,
    build_ftmo_stage2_instruction,
    build_ftmo_summary,
    classify_long_term_alignment,
    format_ftmo_asset_context,
    format_ftmo_status_context,
    format_ftmo_trade_cost,
    format_long_term_alignment,
    format_long_term_alignment_short,
    read_latest_suggestion,
    suggest_ftmo_portfolio,
)
from ai.ftmo_suggest import _write_latest_suggestion
from ai.portfolio_suggest import AssetAnalysis, AuditResult
from analysis.chart_structure import ChartStructureSnapshot, SRLevel, SRLevelsResult
from analysis.technical import TechnicalStats, compute_technical_stats
from data.mt5_source import AccountSummary, ContractSpec, MarketAsset, TradeCost


def _ts(market_regime=None) -> TechnicalStats:
    """Minimal TechnicalStats with only market_regime set — everything
    format_long_term_alignment/its tests need, without needing a real
    price series shaped to produce a specific regime. Built by keyword,
    not position, specifically to avoid a real off-by-N field-index bug
    (caught while writing this: market_regime is TechnicalStats' 12th
    field, not its 10th, by direct field-order count)."""
    return TechnicalStats(
        last_price=None, sma20=None, pct_vs_sma20=None, trend=None,
        change_1m_pct=None, change_3m_pct=None, change_6m_pct=None,
        volatility_annualized_pct=None, support=None, resistance=None,
        range_width_pct=None, market_regime=market_regime, atr=None,
        atr_pct=None, rsi=None, volume_trend_pct=None,
    )


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


def _make_base_analysis(symbol="EURUSD", description="Euro vs US Dollar", display_name=None):
    return AssetAnalysis(
        symbol=symbol, description=description, bid=1.1000, ask=1.1005, display_name=display_name
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


def test_enrich_with_native_d1_passes_through_already_enriched_bases():
    # Yahoo already covered this one (e.g. XAUUSD -> GC=F) — must not be
    # touched or re-fetched.
    base = _make_base_analysis(display_name="Gold")
    with patch("ai.ftmo_suggest.fetch_mt5_price_history") as mock_fetch:
        result = _enrich_with_native_d1(base)
    assert result is base
    mock_fetch.assert_not_called()


@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_enrich_with_native_d1_backfills_from_mt5_when_no_yahoo_mapping(mock_fetch):
    # The real gap this closes: 16 of 17 real FTMO Market Watch symbols
    # (confirmed live) have no Yahoo mapping at all — every forex pair,
    # every single-stock CFD, both crypto pairs — so they never got any
    # D1 technical/backtest coverage before this existed.
    base = _make_base_analysis(display_name=None)
    mock_fetch.return_value = _make_intraday_history(n=300)

    result = _enrich_with_native_d1(base)

    mock_fetch.assert_called_once_with("EURUSD", "D1", count=1500)
    assert result is not base
    assert result.display_name == "Euro vs US Dollar"
    assert result.stats.last_price is not None
    assert result.headlines == []
    assert result.contract_spec == base.contract_spec


@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_enrich_with_native_d1_leaves_bare_when_mt5_also_has_nothing(mock_fetch):
    base = _make_base_analysis(display_name=None)
    mock_fetch.return_value = _empty_history()

    result = _enrich_with_native_d1(base)

    assert result is base
    assert result.display_name is None


@patch("ai.ftmo_suggest.backtest_support_resistance_reaction")
@patch("ai.ftmo_suggest.backtest_rsi_reaction")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_enrich_with_native_d1_threads_real_trade_cost_into_backtests(
    mock_fetch, mock_rsi_bt, mock_sr_bt
):
    # Direct user request 2026-08-22: "review the allowed lot size, trade
    # cost, and other execution related features and broker's allowed
    # guard rails and then apply your backtest trade... such results are
    # more realistic and dependable" — this locks in that a real,
    # already-fetched TradeCost actually reaches the backtest engine's
    # cost/guard-rail parameters, not just that the pure helper computes
    # the right dict in isolation (see the _real_backtest_execution_
    # kwargs tests above).
    history = _make_intraday_history(n=300)
    mock_fetch.return_value = history
    mock_rsi_bt.return_value = (None, None)
    mock_sr_bt.return_value = None
    base = AssetAnalysis(
        symbol="EURUSD", description="Euro vs US Dollar", bid=1.1000, ask=1.1005, display_name=None,
        contract_spec=ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=500.0,
            trade_contract_size=100_000.0, currency_margin="USD", margin_initial=1156.95,
        ),
    )
    cost = _make_trade_cost(
        category="Forex", spread_pct_of_price=0.005,
        swap_long_pct_per_day=-0.0075, swap_short_pct_per_day=0.0003, min_stop_distance_pct=0.2,
    )

    _enrich_with_native_d1(base, trade_cost=cost)

    # base.ask (1.1005), NOT the D1 series' last close — must match
    # format_ftmo_trade_cost's own price source exactly (see
    # _enrich_with_native_d1's own comment on this real fix).
    expected_kwargs = _real_backtest_execution_kwargs(cost, base.contract_spec, base.ask)
    mock_rsi_bt.assert_called_once_with(history, **expected_kwargs)
    mock_sr_bt.assert_called_once_with(history, **expected_kwargs)


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_assets_adds_real_h4_h1_stats(
    mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    # FTMO never touches Yahoo/analyze_assets anymore — every base is
    # built bare (_build_bare_base_analysis) then natively backfilled
    # from MT5 D1 (_enrich_with_native_d1), so the same mocked history
    # feeds the D1 backfill as well as the H4/H1 extension this test is
    # really about.
    mock_contract_spec.return_value = None
    mock_fetch_history.return_value = _make_intraday_history()
    mock_trade_economics.return_value = _make_trade_cost()

    results = analyze_ftmo_assets([MarketAsset("EURUSD", "Euro vs US Dollar", 1.1000, 1.1005)])

    assert len(results) == 1
    assert results[0].base.symbol == "EURUSD"
    assert results[0].base.display_name == "Euro vs US Dollar"
    assert results[0].base.data_source == "mt5"
    assert results[0].h4_stats.last_price is not None
    assert results[0].h1_stats.last_price is not None
    assert results[0].trade_cost is not None
    assert results[0].h4_structure is not None
    assert results[0].h1_structure is not None
    # D1 (native backfill) fetched first, then H4, then H1, all for this
    # exact symbol.
    assert mock_fetch_history.call_args_list[0].args == ("EURUSD", "D1")
    assert mock_fetch_history.call_args_list[1].args == ("EURUSD", "H4")
    assert mock_fetch_history.call_args_list[2].args == ("EURUSD", "H1")
    mock_trade_economics.assert_called_once_with("EURUSD")


@patch("ai.ftmo_suggest.backtest_rsi_reaction")
@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_assets_threads_real_trade_cost_into_rsi_backtest(
    mock_fetch_history, mock_contract_spec, mock_trade_economics, mock_rsi_bt
):
    # Confirms the wiring at the analyze_ftmo_assets level specifically
    # (not just _enrich_with_native_d1's own dedicated test above): the
    # SAME TradeCost fetched for the "REAL trading cost" line also
    # reaches the RSI backtest's real cost/guard-rail simulation, fetched
    # only once (see the redundant-fetch-avoidance comment in the real
    # code) rather than a second time for the backtest specifically.
    mock_contract_spec.return_value = None
    mock_fetch_history.return_value = _make_intraday_history()
    real_cost = _make_trade_cost(
        category="Forex", spread_pct_of_price=0.006,
        swap_long_pct_per_day=-0.008, swap_short_pct_per_day=0.001, min_stop_distance_pct=0.1,
    )
    mock_trade_economics.return_value = real_cost
    mock_rsi_bt.return_value = (None, None)

    analyze_ftmo_assets([MarketAsset("EURUSD", "Euro vs US Dollar", 1.1000, 1.1005)])

    mock_trade_economics.assert_called_once_with("EURUSD")
    mock_rsi_bt.assert_called_once()
    kwargs = mock_rsi_bt.call_args.kwargs
    assert kwargs["round_trip_cost_pct"] == pytest.approx(0.006)  # no contract_spec -> spread only
    assert kwargs["long_swap_pct_per_day"] == pytest.approx(-0.008)
    assert kwargs["short_swap_pct_per_day"] == pytest.approx(0.001)
    assert kwargs["min_stop_distance_pct"] == pytest.approx(0.1)


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_assets_adds_d1_and_monthly_structure(
    mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    # Direct user request (2026-08-22): "redesign the charts and its
    # decision with all H1, H4, D1 and monthly" — the backend half of
    # that is analyze_ftmo_assets computing a real technical read and
    # real chart structure for D1/monthly too, not just H4/H1.
    mock_contract_spec.return_value = None
    mock_fetch_history.return_value = _make_intraday_history()
    mock_trade_economics.return_value = _make_trade_cost()

    results = analyze_ftmo_assets([MarketAsset("EURUSD", "Euro vs US Dollar", 1.1000, 1.1005)])

    assert results[0].mn1_stats.last_price is not None
    assert results[0].d1_structure is not None
    assert results[0].mn1_structure is not None
    # D1, H4, H1, then MN1 (with its own bar count), all for this symbol.
    assert mock_fetch_history.call_args_list[3].args == ("EURUSD", "MN1")
    assert mock_fetch_history.call_args_list[3].kwargs == {"count": _MN1_BACKTEST_BARS}


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_asset_live_builds_full_analysis_natively(
    mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    # A live single-symbol read never touches analyze_assets() (no Yahoo/
    # news lookups) — only real MT5-native fetches, confirmed by NOT
    # patching analyze_assets at all here (a stray call would error since
    # it isn't mocked, e.g. it would try a real network call).
    mock_fetch_history.return_value = _make_intraday_history(n=300)
    mock_contract_spec.return_value = ContractSpec(
        volume_min=0.01, volume_step=0.01, volume_max=100.0,
        trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
    )
    mock_trade_economics.return_value = _make_trade_cost()

    result = analyze_ftmo_asset_live("EURUSD", bid=1.1000, ask=1.1005, description="Euro vs US Dollar")

    assert isinstance(result, FtmoAssetAnalysis)
    assert result.base.symbol == "EURUSD"
    assert result.base.data_source == "mt5"
    assert result.base.display_name == "Euro vs US Dollar"
    assert result.base.stats.last_price is not None
    # A straight monotonic synthetic series genuinely never touches a
    # real support/resistance level to react to, so that one backtest
    # legitimately comes back None here — momentum persistence doesn't
    # need real S/R touches, so it's the one this fixture can actually
    # exercise as "a backtest ran".
    assert result.base.momentum_persistence_backtest is not None
    assert result.h4_stats.last_price is not None
    assert result.h1_stats.last_price is not None
    assert result.h4_structure is not None
    assert result.h1_structure is not None
    assert result.trade_cost is not None
    # D1 (base) fetched first, then H4, then H1, all for this exact symbol.
    assert mock_fetch_history.call_args_list[0].args == ("EURUSD", "D1")
    assert mock_fetch_history.call_args_list[1].args == ("EURUSD", "H4")
    assert mock_fetch_history.call_args_list[2].args == ("EURUSD", "H1")


@patch("ai.ftmo_suggest.backtest_support_resistance_reaction")
@patch("ai.ftmo_suggest.backtest_rsi_reaction")
@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_asset_live_threads_real_cost_and_commission_into_backtests(
    mock_fetch_history, mock_contract_spec, mock_trade_economics, mock_rsi_bt, mock_sr_bt
):
    # Live-popup sibling of test_analyze_ftmo_assets_threads_real_trade_
    # cost_into_rsi_backtest above — this is the exact path app.py's
    # Asset Health popup calls. Also confirms get_contract_spec/
    # get_trade_economics are each fetched exactly ONCE per call (reused
    # for both the backtest simulation AND the analysis' own base/
    # trade_cost fields), not fetched a second time for the backtest.
    history = _make_intraday_history(n=300)
    mock_fetch_history.return_value = history
    spec = ContractSpec(
        volume_min=0.01, volume_step=0.01, volume_max=500.0,
        trade_contract_size=100_000.0, currency_margin="USD", margin_initial=1156.95,
    )
    mock_contract_spec.return_value = spec
    real_cost = _make_trade_cost(category="Forex", spread_pct_of_price=0.005)
    mock_trade_economics.return_value = real_cost
    mock_rsi_bt.return_value = (None, None)
    mock_sr_bt.return_value = None

    analyze_ftmo_asset_live("EURUSD", bid=1.1000, ask=1.1005, description="Euro vs US Dollar")

    mock_contract_spec.assert_called_once_with("EURUSD")
    mock_trade_economics.assert_called_once_with("EURUSD")
    # ask (1.1005), NOT the D1 series' last close — must match
    # format_ftmo_trade_cost's own price source exactly (see
    # analyze_ftmo_asset_live's own comment on this real fix).
    expected_kwargs = _real_backtest_execution_kwargs(real_cost, spec, 1.1005)
    mock_rsi_bt.assert_called_once_with(history, **expected_kwargs)
    mock_sr_bt.assert_called_once_with(history, **expected_kwargs)
    # Real commission actually got folded in, not just the bare spread.
    assert expected_kwargs["round_trip_cost_pct"] > 0.005


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_asset_live_adds_d1_and_monthly_structure(
    mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    # Live-single-symbol sibling of test_analyze_ftmo_assets_adds_d1_and_
    # monthly_structure above — this is the exact path app.py's Asset
    # Health popup calls, so it's the one that actually feeds the
    # redesigned 4-timeframe UI.
    mock_fetch_history.return_value = _make_intraday_history(n=300)
    mock_contract_spec.return_value = None
    mock_trade_economics.return_value = _make_trade_cost()

    result = analyze_ftmo_asset_live("EURUSD", bid=1.1000, ask=1.1005, description="Euro vs US Dollar")

    assert result.mn1_stats.last_price is not None
    assert result.d1_structure is not None
    assert result.mn1_structure is not None
    assert mock_fetch_history.call_args_list[3].args == ("EURUSD", "MN1")
    assert mock_fetch_history.call_args_list[3].kwargs == {"count": _MN1_BACKTEST_BARS}


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec", return_value=None)
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_asset_live_no_backtests_when_d1_history_empty(
    mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    # D1 empty, H4/H1/Monthly still have real bars — mirrors a genuinely
    # new listing with no daily history yet but active intraday trading.
    mock_fetch_history.side_effect = [
        _empty_history(), _make_intraday_history(), _make_intraday_history(), _make_intraday_history(),
    ]
    mock_trade_economics.return_value = _make_trade_cost()

    result = analyze_ftmo_asset_live("NEWSYM", bid=10.0, ask=10.01, description="New Symbol")

    assert result.base.stats.last_price is None
    assert result.base.support_resistance_backtest is None
    assert result.base.rsi_overbought_backtest is None
    assert result.h4_stats.last_price is not None  # H4/H1 unaffected by empty D1


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
@patch("ai.ftmo_suggest.compute_technical_stats")
def test_analyze_ftmo_assets_scales_volatility_correctly_per_timeframe(
    mock_compute_stats, mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    # Regression: H4/H1 stats used to be computed with compute_technical_stats'
    # own default (daily-bar) scaling, understating real annualized
    # volatility by 2.5x-6x for intraday data (see analysis/technical.py's
    # own fix). H4 and H1 need genuinely DIFFERENT periods_per_year, not
    # the same value for both.
    mock_contract_spec.return_value = None
    mock_fetch_history.return_value = _make_intraday_history()
    mock_trade_economics.return_value = _make_trade_cost()
    mock_compute_stats.return_value = compute_technical_stats(pd.Series(dtype=float))

    analyze_ftmo_assets([MarketAsset("EURUSD", "Euro vs US Dollar", 1.1000, 1.1005)])

    # _enrich_with_native_d1 computes its own D1 stats first (no
    # periods_per_year override — the daily default is correct there),
    # then the H4/H1/monthly extension loop computes the other three.
    _d1_call, h4_call, h1_call, mn1_call = mock_compute_stats.call_args_list
    h4_periods = h4_call.kwargs["periods_per_year"]
    h1_periods = h1_call.kwargs["periods_per_year"]
    mn1_periods = mn1_call.kwargs["periods_per_year"]
    assert h4_periods > 252  # scaled for intraday, not left at the daily default
    assert h1_periods > h4_periods  # H1 bars are more frequent than H4 bars
    assert mn1_periods == 12  # 12 monthly bars/year, genuinely different from daily/intraday


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_assets_reports_progress(
    mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    mock_contract_spec.return_value = None
    mock_fetch_history.return_value = _make_intraday_history()
    mock_trade_economics.return_value = _make_trade_cost()

    calls = []
    analyze_ftmo_assets([MarketAsset("EURUSD", "Euro vs US Dollar", 1.1000, 1.1005)], on_progress=calls.append)

    # One consolidated progress message per symbol now (previously two
    # separate passes — a base Yahoo-resolution pass, then this
    # function's own extension pass — before FTMO stopped needing Yahoo
    # at all).
    assert len(calls) == 1
    assert "1/1" in calls[0] and "EURUSD" in calls[0]


@patch("ai.ftmo_suggest.get_trade_economics", return_value=None)
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_assets_degrades_cleanly_on_empty_intraday_history(
    mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    mock_contract_spec.return_value = None
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


def test_format_ftmo_asset_context_includes_d1_and_monthly_chart_structure_and_setup_read():
    from analysis.chart_structure import ChartPattern

    # Direct user request (2026-08-22): "redesign...with all H1, H4, D1
    # and monthly" — this locks in that D1/monthly get the same real
    # chart-structure + setup-read treatment H4/H1 already had, not just
    # a bare stats line.
    d1_history = _make_trending_history(start=1.0, drift=0.01)
    mn1_history = _make_trending_history(start=1.0, drift=0.02)
    d1_stats = compute_technical_stats(d1_history["Close"], history=d1_history)
    mn1_stats = compute_technical_stats(mn1_history["Close"], history=mn1_history)
    base = AssetAnalysis(
        symbol="EURUSD", description="Euro vs US Dollar", bid=1.1000, ask=1.1005,
        display_name="Euro vs US Dollar", stats=d1_stats,
    )
    d1_structure = ChartStructureSnapshot(
        fibonacci=None, sr_levels=None, trendlines=None,
        patterns=[ChartPattern(name="double_top", detail="D1 peaks at 1.2000 and 1.2010")],
    )
    mn1_structure = ChartStructureSnapshot(
        fibonacci=None, sr_levels=None, trendlines=None,
        patterns=[ChartPattern(name="double_top", detail="monthly peaks at 1.3000 and 1.3020")],
    )
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=compute_technical_stats(pd.Series(dtype=float)),
        h1_stats=compute_technical_stats(pd.Series(dtype=float)),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
        mn1_stats=mn1_stats,
        d1_structure=d1_structure,
        mn1_structure=mn1_structure,
    )
    text = format_ftmo_asset_context([analysis])
    assert "D1 chart structure" in text
    assert "Monthly chart structure" in text
    assert "D1 peaks at 1.2000 and 1.2010" in text
    assert "monthly peaks at 1.3000 and 1.3020" in text
    assert "D1 setup read" in text
    assert "Monthly setup read" in text
    # The reversal pattern on each timeframe should surface through that
    # timeframe's own setup read, same as the existing H4-only test above.
    assert text.count("reversal_candidate") == 2


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


# --- format_long_term_alignment ---
# Real feature added live 2026-08-22 (user's own words: this account's
# analysis "reads, displays and analyzes H1, H4 but not the daily or
# monthly charts... this way the analysis is missing the long/medium
# term trends and can throw us in a fake trend unguarded") — compares
# the short-term (H4) directional read against the real daily/monthly
# structural backdrop, distinct from format_mtf_confluence's H4-vs-H1
# comparison above.


def _analysis_with_regimes(h4_regime, d1_regime, mn1_regime) -> FtmoAssetAnalysis:
    base = AssetAnalysis(
        symbol="EURUSD", description="Euro vs US Dollar", bid=1.1000, ask=1.1005,
        display_name="Euro vs US Dollar", stats=_ts(market_regime=d1_regime),
    )
    return FtmoAssetAnalysis(
        base=base,
        h4_stats=_ts(market_regime=h4_regime),
        h1_stats=_ts(market_regime=None),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
        mn1_stats=_ts(market_regime=mn1_regime),
    )


def test_long_term_alignment_structurally_backed_when_both_backdrops_agree():
    analysis = _analysis_with_regimes("trending_up", "trending_up", "choppy_up")
    text = format_long_term_alignment(analysis)
    assert "STRUCTURALLY BACKED" in text
    assert "daily and monthly" in text


def test_long_term_alignment_counter_trend_spike_when_both_backdrops_oppose():
    analysis = _analysis_with_regimes("choppy_up", "trending_down", "choppy_down")
    text = format_long_term_alignment(analysis)
    assert "COUNTER-TREND SPIKE" in text
    # The user explicitly wants this framed as tradeable-but-managed, not
    # discarded outright — confirm the language reflects that, not just
    # the label.
    assert "genuinely tradeable" in text
    assert "tighter stop" in text


def test_long_term_alignment_mixed_when_backdrops_disagree_with_each_other():
    analysis = _analysis_with_regimes("trending_up", "trending_up", "choppy_down")
    text = format_long_term_alignment(analysis)
    assert "MIXED" in text


def test_long_term_alignment_none_when_no_real_backdrop_direction():
    # Both daily and monthly read sideways — no direction to compare
    # against at all, distinct from a genuine CONTRADICTING backdrop.
    analysis = _analysis_with_regimes("choppy_up", "sideways", "sideways")
    text = format_long_term_alignment(analysis)
    assert "no real daily or monthly directional backdrop" in text
    assert "STRUCTURALLY BACKED" not in text
    assert "COUNTER-TREND" not in text


def test_long_term_alignment_none_when_h4_itself_flat_or_missing():
    flat = _analysis_with_regimes("sideways", "trending_up", "trending_up")
    assert "no real short-term direction" in format_long_term_alignment(flat)

    missing = _analysis_with_regimes(None, "trending_up", "trending_up")
    assert "not available" in format_long_term_alignment(missing)


def test_format_ftmo_asset_context_includes_monthly_and_long_term_alignment():
    analysis = _analysis_with_regimes("trending_up", "trending_up", "trending_up")
    text = format_ftmo_asset_context([analysis])
    assert "Monthly technical" in text
    assert "Long-term alignment" in text
    assert "STRUCTURALLY BACKED" in text


# --- format_long_term_alignment_short / classify_long_term_alignment ---
# Real UI feedback, live 2026-08-22: the long, reasoning-heavy AI-prompt
# text (format_long_term_alignment above) was showing up verbatim in the
# Asset Health popup — correct for a model, unreadable as a dashboard
# line. These two share one classification step (classify_long_term_
# alignment) with the long formatter so the two texts can never disagree
# about WHICH state applies, only how verbosely they describe it.


def test_long_term_alignment_short_is_one_short_sentence_per_state():
    cases = [
        ("trending_up", "trending_up", "choppy_up", "structurally_backed"),
        ("choppy_up", "trending_down", "choppy_down", "counter_trend_spike"),
        ("trending_up", "trending_up", "choppy_down", "mixed"),
        ("choppy_up", "sideways", "sideways", "no_backdrop"),
        ("sideways", "trending_up", "trending_up", "flat"),
        (None, "trending_up", "trending_up", "not_available"),
    ]
    for h4, d1, mn1, expected_state in cases:
        analysis = _analysis_with_regimes(h4, d1, mn1)
        state, *_ = classify_long_term_alignment(analysis)
        assert state == expected_state
        short_text = format_long_term_alignment_short(analysis)
        # "Short" is the whole point being tested here — no multi-
        # sentence reasoning paragraph, just one real, decisive line.
        assert short_text.count(". ") <= 1
        assert len(short_text) < 160


def test_long_term_alignment_short_and_long_never_disagree_on_state():
    # Both formatters read from the same classify_long_term_alignment
    # call — this locks in that a STRUCTURALLY BACKED long text always
    # pairs with the structurally_backed short message, not a stale or
    # independently-drifted one.
    analysis = _analysis_with_regimes("trending_up", "trending_up", "trending_up")
    assert "STRUCTURALLY BACKED" in format_long_term_alignment(analysis)
    assert "likely real" in format_long_term_alignment_short(analysis)

    analysis = _analysis_with_regimes("choppy_up", "trending_down", "choppy_down")
    assert "COUNTER-TREND SPIKE" in format_long_term_alignment(analysis)
    assert "short spike" in format_long_term_alignment_short(analysis)


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


# --- _real_backtest_execution_kwargs (real broker cost/guard-rail realism) ---


def _make_spec(**overrides):
    defaults = dict(
        volume_min=0.01, volume_step=0.01, volume_max=500.0,
        trade_contract_size=100_000.0, currency_margin="USD", margin_initial=1156.95,
    )
    return ContractSpec(**{**defaults, **overrides})


def test_real_backtest_execution_kwargs_empty_when_no_trade_cost():
    # No live TradeCost at all (e.g. MT5 unreachable) — the engine's own
    # existing zero-cost/zero-restriction default, unchanged, not an
    # error or a fabricated zero.
    assert _real_backtest_execution_kwargs(None, _make_spec(), 1.1000) == {}


def test_real_backtest_execution_kwargs_spread_only_without_contract_spec_or_price():
    cost = _make_trade_cost(category="Forex", spread_pct_of_price=0.005)
    kwargs = _real_backtest_execution_kwargs(cost, None, None)
    assert kwargs["round_trip_cost_pct"] == pytest.approx(0.005)


def test_real_backtest_execution_kwargs_adds_real_commission_when_available():
    cost = _make_trade_cost(category="Forex", spread_pct_of_price=0.005)
    spec = _make_spec(trade_contract_size=100_000.0)
    kwargs = _real_backtest_execution_kwargs(cost, spec, 1.1000)
    commission_pct, _ = _ftmo_commission_pct_round_turn("Forex", 100_000.0, 1.1000)
    assert kwargs["round_trip_cost_pct"] == pytest.approx(0.005 + commission_pct)


def test_real_backtest_execution_kwargs_skips_commission_for_unknown_category():
    # A category _ftmo_commission_pct_round_turn genuinely can't price
    # (returns None) must NOT be silently treated as zero — only the
    # real spread should be netted in.
    cost = _make_trade_cost(category="Equities I CFD", spread_pct_of_price=0.01)
    kwargs = _real_backtest_execution_kwargs(cost, _make_spec(), 200.0)
    assert kwargs["round_trip_cost_pct"] == pytest.approx(0.01)


def test_real_backtest_execution_kwargs_defaults_missing_swap_to_zero():
    cost = _make_trade_cost(swap_long_pct_per_day=None, swap_short_pct_per_day=None)
    kwargs = _real_backtest_execution_kwargs(cost, None, None)
    assert kwargs["long_swap_pct_per_day"] == 0.0
    assert kwargs["short_swap_pct_per_day"] == 0.0


def test_real_backtest_execution_kwargs_passes_through_real_swap_and_min_stop_distance():
    cost = _make_trade_cost(
        swap_long_pct_per_day=-0.0075, swap_short_pct_per_day=0.0003, min_stop_distance_pct=0.35
    )
    kwargs = _real_backtest_execution_kwargs(cost, None, None)
    assert kwargs["long_swap_pct_per_day"] == pytest.approx(-0.0075)
    assert kwargs["short_swap_pct_per_day"] == pytest.approx(0.0003)
    assert kwargs["min_stop_distance_pct"] == pytest.approx(0.35)


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


def test_format_ftmo_trade_cost_reports_both_long_and_short_swap():
    # A short recommendation needs its own real overnight cost to reason
    # about — this used to only ever surface the long side.
    spec = ContractSpec(
        volume_min=0.01, volume_step=0.01, volume_max=500.0,
        trade_contract_size=100_000.0, currency_margin="USD", margin_initial=1156.95,
    )
    base = AssetAnalysis(
        symbol="EURUSD", description="Euro vs US Dollar", bid=1.15689, ask=1.15695,
        display_name=None, contract_spec=spec,
    )
    cost = _make_trade_cost(swap_long_pct_per_day=-0.0075, swap_short_pct_per_day=0.0003)
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=compute_technical_stats(pd.Series(dtype=float)),
        h1_stats=compute_technical_stats(pd.Series(dtype=float)),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=cost,
    )
    text = format_ftmo_trade_cost(analysis)
    assert "-0.0075%/day held long" in text
    assert "+0.0003%/day held short" in text


def test_format_ftmo_trade_cost_handles_one_sided_swap_none():
    spec = ContractSpec(
        volume_min=0.01, volume_step=0.01, volume_max=500.0,
        trade_contract_size=100_000.0, currency_margin="USD", margin_initial=1156.95,
    )
    base = AssetAnalysis(
        symbol="EURUSD", description="Euro vs US Dollar", bid=1.15689, ask=1.15695,
        display_name=None, contract_spec=spec,
    )
    cost = _make_trade_cost(swap_long_pct_per_day=-0.0075, swap_short_pct_per_day=None)
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=compute_technical_stats(pd.Series(dtype=float)),
        h1_stats=compute_technical_stats(pd.Series(dtype=float)),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=cost,
    )
    text = format_ftmo_trade_cost(analysis)
    assert "-0.0075%/day held long" in text
    assert "held short" not in text


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


def test_ftmo_stage1_instruction_mentions_short_side_support():
    text = build_ftmo_stage1_instruction()
    assert '"side"' in text
    assert '"buy"' in text
    assert '"sell"' in text


def test_ftmo_stage1_instruction_mentions_pending_setups_section():
    text = build_ftmo_stage1_instruction()
    assert "## Pending Setups" in text
    assert "trigger_condition" in text
    # The "only one fenced block, ever" absolute claim must be gone now
    # that a second (optional) Pending Setups block is allowed — a stale
    # copy of this sentence would tell the model two contradictory things.
    assert "ONLY fenced code block" not in text
    assert "Exactly two fenced code blocks" in text


def test_ftmo_stage1_instruction_forbids_symbol_overlap_between_sections():
    text = build_ftmo_stage1_instruction()
    assert "never both at once" in text


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


def test_ftmo_stage1_instruction_requires_showing_stop_target_arithmetic_once():
    # Real bug found in a live report (2026-08-22): the same instrument's
    # ATR-based stop was stated as 1.26% of price in one paragraph and
    # 0.019% in another, and a "20 pips" stop claim didn't match the
    # actual 30-pip distance between the stated entry and stop prices —
    # the model recomputed (or misremembered) the same number twice
    # instead of reusing its own first, correct calculation.
    text = build_ftmo_stage1_instruction()
    assert "SHOW THE STOP/TARGET ARITHMETIC ONCE" in text
    assert "pip size" in text.lower()
    assert "Pending Setups" in text


def test_ftmo_stage1_instruction_pending_setups_requires_reusing_stop_target_numbers():
    text = build_ftmo_stage1_instruction()
    assert "copied from there verbatim" in text


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


def test_ftmo_audit_instruction_covers_pending_setups_scrutiny():
    assert "Pending Setups" in AUDIT_INSTRUCTION
    assert "trigger_condition" in AUDIT_INSTRUCTION
    assert "mechanically checkable" in AUDIT_INSTRUCTION
    assert "CONSISTENCY" in AUDIT_INSTRUCTION


def test_ftmo_audit_instruction_requires_recomputing_pending_setups_arithmetic():
    # Real bug found in a live report (2026-08-22): prose claimed one
    # R:R/pip count while the entry's own price/stop_loss/take_profit
    # numbers implied a different one. Auditors must actually recompute
    # from the JSON's own numbers, not just eyeball plausibility.
    assert "RECOMPUTE the stop distance" in AUDIT_INSTRUCTION
    assert "gross R:R" in AUDIT_INSTRUCTION


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
    # Real bug found live: the manual button's default used to keep
    # Copilot in FTMO's audit pool even after the user asked for it to
    # be removed — that request applies to FTMO's audit pool as a
    # whole, not just the scheduled path. Default is now False.
    assert kwargs["include_copilot"] is False


@patch("ai.ftmo_suggest.build_audit_block")
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_can_exclude_copilot_from_the_audit_pool(mock_run_claude, mock_audit):
    # Direct user request 2026-08-22: ai.mega_analysis.run_mega_analysis
    # calls this with include_copilot=False (Copilot has a separate,
    # dedicated role elsewhere) — this must actually reach build_audit_block.
    mock_run_claude.side_effect = ["draft text", "final text"]
    mock_audit.return_value = AuditResult(block="audit block", audit_available=True)

    suggest_ftmo_portfolio("some ftmo summary", include_copilot=False)

    _, kwargs = mock_audit.call_args
    assert kwargs["include_copilot"] is False


# --- read_latest_suggestion / _write_latest_suggestion ---
# Moved here from ai.mega_analysis (2026-08-23): a real design bug found
# live had this write happening only inside ai.mega_analysis.run_mega_
# analysis, so the manual "Suggest Portfolio Mix" button's own
# successful runs never armed Copilot at all — the user's own explicit
# intent was that manual and scheduled runs differ only in what
# triggers them. Moved into suggest_ftmo_portfolio() itself (below) so
# BOTH callers get it unconditionally, with no per-caller opt-in.


@pytest.fixture(autouse=True)
def _isolated_latest_suggestion_file(tmp_path):
    with patch.object(config, "MEGA_ANALYSIS_LATEST_SUGGESTION_FILE", str(tmp_path / "latest_suggestion.json")):
        yield


_WELL_FORMED_FINAL_ANSWER = (
    "## Executive Summary\nSome prose.\n\n"
    '```json\n{"EURUSD": {"pct": 1.5, "price": 1.09, "stop_loss": 1.08, '
    '"take_profit": 1.11, "side": "buy"}, "CASH": 98.5}\n```\n\n'
    '```json\n[{"symbol": "XAUUSD", "side": "buy", "pct": 1.0, '
    '"trigger_condition": "H1 closes above 2000 with RSI turning up", '
    '"price": 2000.0, "stop_loss": 1980.0, "take_profit": 2050.0, '
    '"reason": "breakout watch"}]\n```'
)


def test_read_latest_suggestion_empty_dict_when_file_missing():
    assert read_latest_suggestion() == {}


def test_write_latest_suggestion_then_read_round_trips():
    _write_latest_suggestion(_WELL_FORMED_FINAL_ANSWER)
    saved = read_latest_suggestion()
    assert saved["immediate_allocation"]["EURUSD"]["side"] == "buy"
    assert saved["immediate_allocation"]["EURUSD"]["pct"] == 1.5
    assert saved["pending_setups"][0]["symbol"] == "XAUUSD"
    assert saved["pending_setups"][0]["trigger_condition"] == (
        "H1 closes above 2000 with RSI turning up"
    )
    assert saved["generated_utc"]


def test_write_latest_suggestion_does_not_overwrite_on_malformed_allocation():
    _write_latest_suggestion(_WELL_FORMED_FINAL_ANSWER)
    prior = read_latest_suggestion()

    _write_latest_suggestion("Just prose, no allocation block at all.")

    assert read_latest_suggestion() == prior


def test_write_latest_suggestion_does_not_overwrite_on_malformed_pending_setups():
    _write_latest_suggestion(_WELL_FORMED_FINAL_ANSWER)
    prior = read_latest_suggestion()

    # Valid allocation block, but a Pending Setups array with a duplicate
    # symbol — malformed, must fail the whole write, not just skip the
    # pending-setups half.
    bad_answer = (
        '```json\n{"EURUSD": {"pct": 1.5, "side": "buy"}, "CASH": 98.5}\n```\n\n'
        '```json\n[{"symbol": "XAUUSD", "side": "buy", "pct": 1.0, "trigger_condition": "a"}, '
        '{"symbol": "XAUUSD", "side": "sell", "pct": 1.0, "trigger_condition": "b"}]\n```'
    )
    _write_latest_suggestion(bad_answer)

    assert read_latest_suggestion() == prior


def test_read_latest_suggestion_empty_dict_on_corrupt_file(tmp_path):
    corrupt_path = tmp_path / "corrupt_suggestion.json"
    corrupt_path.write_text("{not valid json")
    with patch.object(config, "MEGA_ANALYSIS_LATEST_SUGGESTION_FILE", str(corrupt_path)):
        assert read_latest_suggestion() == {}


# --- the real fix: BOTH callers of suggest_ftmo_portfolio get this for free ---


@patch("ai.ftmo_suggest.build_audit_block", return_value=AuditResult(block="", audit_available=False))
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_writes_latest_suggestion_on_success(mock_run_claude, mock_audit):
    # This is THE regression test for the real bug the user caught live:
    # a manual "Suggest Portfolio Mix" click (which calls
    # suggest_ftmo_portfolio directly, exactly like this test does —
    # NOT via ai.mega_analysis.run_mega_analysis) must arm Copilot's
    # clerk too, not just a scheduled run. Only run_claude/build_audit_
    # block are mocked here (not suggest_ftmo_portfolio itself), so the
    # real internal _write_latest_suggestion call actually executes.
    mock_run_claude.side_effect = ["draft text", _WELL_FORMED_FINAL_ANSWER]

    suggest_ftmo_portfolio("some ftmo summary")

    saved = read_latest_suggestion()
    assert saved["immediate_allocation"]["EURUSD"]["pct"] == 1.5
    assert saved["pending_setups"][0]["symbol"] == "XAUUSD"


@patch("ai.ftmo_suggest.build_audit_block", return_value=AuditResult(block="", audit_available=False))
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_does_not_write_latest_suggestion_on_cli_failure(mock_run_claude, mock_audit):
    # The resumed stage-2 call failing falls back to a full-context
    # retry (suggest_ftmo_portfolio's own existing behavior) — make that
    # fail too, so the overall result is a genuine CLI failure.
    mock_run_claude.side_effect = ["draft text", CLI_MISSING_MESSAGE, CLI_MISSING_MESSAGE]

    suggest_ftmo_portfolio("some ftmo summary")

    assert read_latest_suggestion() == {}


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
