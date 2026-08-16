from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE
from ai.portfolio_suggest import AllocationEntry, AuditResult
from ai.psx_suggest import (
    EPSGrowthTrend,
    PSXAssetAnalysis,
    _fetch_current_psx_price,
    analyze_psx_assets,
    build_psx_macro_context,
    build_psx_summary,
    compute_eps_growth_trend,
    compute_sector_allocation,
    format_psx_asset_context,
    suggest_psx_portfolio,
)
from analysis.backtest import (
    BetaStabilityBacktest,
    MomentumPersistenceBacktest,
    RSIReactionBacktest,
    SupportResistanceBacktest,
    VolatilityRegimeBacktest,
)
from analysis.technical import compute_technical_stats
from data.macro_source import CountryIndicators, PakistanRates
from data.psx_source import (
    PSXAnnouncement,
    PSXAsset,
    PSXCompanyData,
    PSXCompanyProfile,
    PSXFinancials,
    PSXFundamentals,
)


@pytest.fixture(autouse=True)
def _no_real_past_lessons():
    """Same rationale as ai/portfolio_suggest.py's own copy of this
    fixture: build_past_lessons does real file I/O against this actual
    project's real records/psx/ directory and real network calls to
    re-price past symbols — patch it to inert by default so every
    suggest_psx_portfolio()-calling test stays fast and network-free."""
    with patch("ai.psx_suggest.build_past_lessons", return_value=""):
        yield


def _make_fundamentals(**overrides):
    defaults = dict(
        pe_ratio=5.0,
        market_cap_pkr_000s=1000.0,
        free_float_pct=20.0,
        shares_outstanding=500.0,
        circuit_breaker_low=9.0,
        circuit_breaker_high=11.0,
        week52_low=8.0,
        week52_high=12.0,
    )
    return PSXFundamentals(**{**defaults, **overrides})


def _make_financials(**overrides):
    defaults = dict(
        annual_periods=["2025"],
        annual={"Sales": [1000.0], "EPS": [0.5]},
        quarterly_periods=["Q1 2026"],
        quarterly={"Sales": [250.0]},
        ratio_periods=["2025"],
        ratios={"PEG": [0.2]},
    )
    return PSXFinancials(**{**defaults, **overrides})


def _make_profile(**overrides):
    defaults = dict(
        description="Makes widgets.",
        key_people=[("Jane Doe", "CEO")],
        auditor="Some Audit Firm",
        fiscal_year_end="June",
    )
    return PSXCompanyProfile(**{**defaults, **overrides})


def _make_company_data(**overrides):
    defaults = dict(
        fundamentals=_make_fundamentals(),
        financials=_make_financials(),
        announcements=[PSXAnnouncement(category="Board Meetings", date="Apr 20, 2026", title="Board Meeting")],
        profile=_make_profile(),
    )
    return PSXCompanyData(**{**defaults, **overrides})


def _make_asset(symbol, listed_in=("KSE100",), volume=1000, current=10.0, change_pct=1.0):
    return PSXAsset(
        symbol=symbol,
        company_name=f"{symbol} Corp",
        sector_code="100",
        listed_in=list(listed_in),
        ldcp=current - 0.1,
        open=current - 0.05,
        high=current + 0.5,
        low=current - 0.5,
        current=current,
        change=0.1,
        change_pct=change_pct,
        volume=volume,
    )


def _make_history(n=25, start=10.0):
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    # Plain lists, not a pd.Series — a Series carries its own default
    # RangeIndex, which pd.DataFrame(..., index=dates) would align against
    # (not assign positionally), silently turning every value to NaN since
    # none of the integer labels match the DatetimeIndex.
    prices = [float(i) + start for i in range(n)]
    return pd.DataFrame({"Open": prices, "Close": prices, "Volume": [1000] * n}, index=dates)


@patch("ai.psx_suggest.get_psx_company_data")
@patch("ai.psx_suggest.get_psx_history")
def test_analyze_psx_assets_only_considers_kse100_members(mock_history, mock_company_data):
    mock_history.return_value = _make_history()
    mock_company_data.return_value = _make_company_data()
    assets = [
        _make_asset("KSEONE", listed_in=("KSE100",)),
        _make_asset("NOTINDEX", listed_in=("ALLSHR",)),
    ]
    analyses = analyze_psx_assets(assets)
    assert [a.symbol for a in analyses] == ["KSEONE"]


@patch("ai.psx_suggest.get_psx_company_data")
@patch("ai.psx_suggest.get_psx_history")
def test_analyze_psx_assets_reports_per_symbol_progress(mock_history, mock_company_data):
    mock_history.return_value = _make_history()
    mock_company_data.return_value = _make_company_data()
    assets = [
        _make_asset("A", listed_in=("KSE100",)),
        _make_asset("B", listed_in=("KSE100",)),
    ]
    calls = []
    analyze_psx_assets(assets, on_progress=calls.append)
    assert len(calls) == 2
    assert "1/2" in calls[0]
    assert "2/2" in calls[1]


@patch("ai.psx_suggest.get_psx_company_data")
@patch("ai.psx_suggest.get_psx_history")
def test_analyze_psx_assets_honors_selected_index_tag(mock_history, mock_company_data):
    mock_history.return_value = _make_history()
    mock_company_data.return_value = _make_company_data()
    assets = [
        _make_asset("KSEONE", listed_in=("KSE100",)),
        _make_asset("KMIONE", listed_in=("KMI30", "KMIALLSHR")),
    ]
    analyses = analyze_psx_assets(assets, index_tag="KMI30")
    assert [a.symbol for a in analyses] == ["KMIONE"]
    a = analyses[0]
    assert a.benchmark_index == "KMI30"
    # relative strength is benchmarked against the SAME index selected,
    # so the index history fetch must have been for "KMI30", not the
    # default KSE100.
    mock_history.assert_any_call("KMI30")


@patch("ai.psx_suggest.get_psx_company_data")
@patch("ai.psx_suggest.get_psx_history")
def test_analyze_psx_assets_caps_at_max_enriched_ranked_by_volume(mock_history, mock_company_data):
    mock_history.return_value = _make_history()
    mock_company_data.return_value = _make_company_data()
    assets = [_make_asset(f"SYM{i}", volume=i) for i in range(5)]
    with patch.object(config, "MAX_PSX_ENRICHED_ASSETS", 2):
        analyses = analyze_psx_assets(assets)
    assert [a.symbol for a in analyses] == ["SYM4", "SYM3"]


@patch("ai.psx_suggest.get_psx_company_data")
@patch("ai.psx_suggest.get_psx_history")
def test_analyze_psx_assets_computes_technical_stats_and_bundles_company_data(
    mock_history, mock_company_data
):
    mock_history.return_value = _make_history(n=30)
    mock_company_data.return_value = _make_company_data(
        fundamentals=_make_fundamentals(pe_ratio=5.0)
    )
    analyses = analyze_psx_assets([_make_asset("ABC")])
    assert len(analyses) == 1
    a = analyses[0]
    assert a.stats.rsi is not None
    assert a.stats.atr is None  # no High/Low in PSX history — never fabricated
    assert a.fundamentals.pe_ratio == 5.0
    assert a.financials.annual["Sales"] == [1000.0]
    assert a.announcements[0].title == "Board Meeting"
    mock_company_data.assert_called_once_with("ABC")


@patch("ai.psx_suggest.compute_eps_growth_trend")
@patch("ai.psx_suggest.backtest_support_resistance_reaction")
@patch("ai.psx_suggest.backtest_volatility_regime")
@patch("ai.psx_suggest.backtest_momentum_persistence")
@patch("ai.psx_suggest.backtest_beta_stability")
@patch("ai.psx_suggest.backtest_rsi_reaction")
@patch("ai.psx_suggest.get_psx_company_data")
@patch("ai.psx_suggest.get_psx_history")
def test_analyze_psx_assets_wires_backtests_onto_the_analysis(
    mock_history,
    mock_company_data,
    mock_rsi_bt,
    mock_beta_bt,
    mock_momentum_bt,
    mock_vol_regime_bt,
    mock_sr_bt,
    mock_eps_growth,
):
    mock_history.return_value = _make_history()
    mock_company_data.return_value = _make_company_data()
    overbought = RSIReactionBacktest("overbought", 70.0, 6, -5.0, 100.0, 10)
    oversold = RSIReactionBacktest("oversold", 30.0, 5, 6.0, 80.0, 10)
    mock_rsi_bt.return_value = (overbought, oversold)
    beta_bt = BetaStabilityBacktest(0.5, 0.6, 0.55, 0.52, True)
    mock_beta_bt.return_value = beta_bt
    momentum_bt = MomentumPersistenceBacktest(0.4, 20, "persistent")
    mock_momentum_bt.return_value = momentum_bt
    vol_regime_bt = VolatilityRegimeBacktest(5.0, 15, 10.0, 20, 10)
    mock_vol_regime_bt.return_value = vol_regime_bt
    sr_bt = SupportResistanceBacktest(10, 70.0, 8, 60.0, 10)
    mock_sr_bt.return_value = sr_bt
    eps_growth = EPSGrowthTrend(1.2, 1.0, 20.0, 2, 0.4, 0.3, 33.3)
    mock_eps_growth.return_value = eps_growth

    a = analyze_psx_assets([_make_asset("ABC")])[0]
    assert a.rsi_overbought_backtest is overbought
    assert a.rsi_oversold_backtest is oversold
    assert a.beta_stability_backtest is beta_bt
    assert a.momentum_persistence_backtest is momentum_bt
    assert a.volatility_regime_backtest is vol_regime_bt
    assert a.support_resistance_backtest is sr_bt
    assert a.eps_growth_trend is eps_growth
    mock_rsi_bt.assert_called_once()
    mock_beta_bt.assert_called_once()
    mock_momentum_bt.assert_called_once()
    mock_vol_regime_bt.assert_called_once()
    mock_sr_bt.assert_called_once()
    mock_eps_growth.assert_called_once()


@patch("ai.psx_suggest.get_psx_company_data")
@patch("ai.psx_suggest.get_psx_history")
def test_analyze_psx_assets_resolves_sector_name_and_dividend20_flag(
    mock_history, mock_company_data
):
    mock_history.return_value = _make_history()
    mock_company_data.return_value = _make_company_data()
    asset = _make_asset("ABC", listed_in=("KSE100", "PSXDIV20"))
    asset.sector_code = "0807"
    a = analyze_psx_assets([asset])[0]
    assert a.sector_name == "Commercial Banks"
    assert a.is_dividend20_member is True


@patch("ai.psx_suggest.get_psx_company_data")
@patch("ai.psx_suggest.get_psx_history")
def test_analyze_psx_assets_dividend20_false_when_not_a_member(
    mock_history, mock_company_data
):
    mock_history.return_value = _make_history()
    mock_company_data.return_value = _make_company_data()
    asset = _make_asset("ABC", listed_in=("KSE100",))
    a = analyze_psx_assets([asset])[0]
    assert a.is_dividend20_member is False


def test_compute_sector_performance_averages_and_sorts_best_first():
    from ai.psx_suggest import compute_sector_performance

    assets = [
        _make_asset("A1", change_pct=5.0),
        _make_asset("A2", change_pct=3.0),
        _make_asset("A3", change_pct=1.0),
        _make_asset("B1", change_pct=-2.0),
        _make_asset("B2", change_pct=-4.0),
        _make_asset("B3", change_pct=-6.0),
    ]
    for a in assets[:3]:
        a.sector_code = "0807"  # Commercial Banks
    for a in assets[3:]:
        a.sector_code = "0804"  # Cement

    performance = compute_sector_performance(assets)
    assert performance[0] == ("Commercial Banks", pytest.approx(3.0), 3)
    assert performance[1] == ("Cement", pytest.approx(-4.0), 3)


def test_compute_sector_performance_excludes_thin_sectors():
    from ai.psx_suggest import compute_sector_performance

    assets = [_make_asset("A1", change_pct=5.0), _make_asset("A2", change_pct=3.0)]
    for a in assets:
        a.sector_code = "0807"
    assert compute_sector_performance(assets) == []  # only 2 members, below the min of 3


def test_build_psx_sector_context_includes_real_sector_names():
    from ai.psx_suggest import build_psx_sector_context

    assets = [_make_asset(f"A{i}", change_pct=float(i)) for i in range(3)]
    for a in assets:
        a.sector_code = "0836"  # Real Estate Investment Trust
    text = build_psx_sector_context(assets)
    assert "Real Estate Investment Trust" in text
    assert "3 symbols" in text


def test_compute_sector_allocation_groups_by_real_sector_and_cash():
    bank = _analysis_with_stats("BANK1")
    bank.sector_name = "Commercial Banks"
    cement = _analysis_with_stats("CEM1")
    cement.sector_name = "Cement"
    analyses = [bank, cement]

    allocation = {
        "BANK1": AllocationEntry(pct=30.0, price=10.0, stop_loss=9.0),
        "CEM1": AllocationEntry(pct=45.0, price=20.0, stop_loss=18.0),
        "CASH": AllocationEntry(pct=25.0),
    }
    result = compute_sector_allocation(allocation, analyses)
    assert result == {"Commercial Banks": 30.0, "Cement": 45.0, "Cash": 25.0}


def test_compute_sector_allocation_sums_multiple_symbols_in_same_sector():
    bank1 = _analysis_with_stats("BANK1")
    bank1.sector_name = "Commercial Banks"
    bank2 = _analysis_with_stats("BANK2")
    bank2.sector_name = "Commercial Banks"

    allocation = {
        "BANK1": AllocationEntry(pct=20.0),
        "BANK2": AllocationEntry(pct=15.0),
        "CASH": AllocationEntry(pct=65.0),
    }
    result = compute_sector_allocation(allocation, [bank1, bank2])
    assert result == {"Commercial Banks": 35.0, "Cash": 65.0}


def test_compute_sector_allocation_unclassified_for_symbol_outside_pool():
    allocation = {"MYSTERY": AllocationEntry(pct=40.0), "CASH": AllocationEntry(pct=60.0)}
    result = compute_sector_allocation(allocation, [])
    assert result == {"Unclassified": 40.0, "Cash": 60.0}


@patch("ai.psx_suggest.get_psx_company_data")
@patch("ai.psx_suggest.get_psx_history")
def test_analyze_psx_assets_computes_week52_position_from_fundamentals(
    mock_history, mock_company_data
):
    mock_history.return_value = _make_history()
    mock_company_data.return_value = _make_company_data(
        fundamentals=_make_fundamentals(week52_low=8.0, week52_high=12.0)
    )
    asset = _make_asset("ABC", current=10.0)
    a = analyze_psx_assets([asset])[0]
    assert a.week52_position_pct == pytest.approx(50.0)


def test_compute_week52_position_pct_edges():
    from ai.psx_suggest import _compute_week52_position_pct

    assert _compute_week52_position_pct(10.0, 8.0, 12.0) == pytest.approx(50.0)
    assert _compute_week52_position_pct(8.0, 8.0, 12.0) == pytest.approx(0.0)
    assert _compute_week52_position_pct(12.0, 8.0, 12.0) == pytest.approx(100.0)
    assert _compute_week52_position_pct(10.0, None, 12.0) is None
    assert _compute_week52_position_pct(10.0, 12.0, 12.0) is None  # zero-width range


# Beta computation itself (compute_beta) now lives in analysis/backtest.py
# and is tested directly there — ai/psx_suggest.py only calls it.


def test_compute_eps_growth_trend_none_without_financials():
    assert compute_eps_growth_trend(None) is None


def test_compute_eps_growth_trend_none_without_enough_annual_periods():
    financials = _make_financials(annual_periods=["2025"], annual={"EPS": [0.5]})
    assert compute_eps_growth_trend(financials) is None


def test_compute_eps_growth_trend_computes_growth_and_streak():
    # Newest-first, per PSXFinancials' own column order: 3 consecutive
    # periods of growth (1.2 > 1.0 > 0.8), so the streak counts 2 (the
    # number of period-over-period increases immediately before latest).
    financials = _make_financials(
        annual_periods=["2028", "2027", "2026"],
        annual={"EPS": [1.2, 1.0, 0.8]},
        quarterly_periods=["Q4", "Q3", "Q2", "Q1", "Q4-ago"],
        quarterly={"EPS": [0.4, 0.35, 0.3, 0.25, 0.3]},
    )
    trend = compute_eps_growth_trend(financials)
    assert trend.latest_annual_eps == 1.2
    assert trend.prior_annual_eps == 1.0
    assert trend.annual_yoy_growth_pct == pytest.approx(20.0)
    assert trend.consecutive_growth_years == 2
    assert trend.latest_quarterly_eps == 0.4
    assert trend.year_ago_quarterly_eps == pytest.approx(0.3)
    assert trend.quarterly_yoy_growth_pct == pytest.approx(33.333, abs=0.01)


def test_compute_eps_growth_trend_breaks_streak_on_decline():
    financials = _make_financials(
        annual_periods=["2028", "2027", "2026"], annual={"EPS": [0.9, 1.0, 0.8]}
    )
    trend = compute_eps_growth_trend(financials)
    assert trend.consecutive_growth_years == 0  # latest year DECLINED vs prior


def test_compute_eps_growth_trend_growth_pct_none_when_base_not_positive():
    financials = _make_financials(annual_periods=["2027", "2026"], annual={"EPS": [3000.0, -5000.0]})
    trend = compute_eps_growth_trend(financials)
    assert trend.annual_yoy_growth_pct is None  # prior-year EPS was a loss — % change is meaningless


def test_compute_eps_growth_trend_none_without_year_ago_quarter():
    financials = _make_financials(
        annual_periods=["2027", "2026"],
        annual={"EPS": [1.0, 0.8]},
        quarterly_periods=["Q1"],
        quarterly={"EPS": [0.4]},
    )
    trend = compute_eps_growth_trend(financials)
    assert trend.latest_quarterly_eps == 0.4
    assert trend.year_ago_quarterly_eps is None
    assert trend.quarterly_yoy_growth_pct is None


def _analysis_with_stats(symbol="ABC", **stat_overrides):
    prices = _make_history()["Close"]
    stats = compute_technical_stats(prices, history=_make_history())
    if stat_overrides:
        stats = stats.__class__(**{**stats.__dict__, **stat_overrides})
    return PSXAssetAnalysis(
        symbol=symbol,
        display_name=f"{symbol} Corp",
        company_name=f"{symbol} Corp",
        sector_code="100",
        sector_name="Test Sector",
        current=10.0,
        change_pct=1.5,
        volume=1000,
        prices=prices,
        stats=stats,
        fundamentals=_make_fundamentals(),
        financials=_make_financials(),
        announcements=[PSXAnnouncement(category="Others", date="Apr 7, 2026", title="A disclosure")],
        profile=_make_profile(),
        relative_strength_1m_pct=3.5,
        relative_strength_3m_pct=-2.0,
        is_dividend20_member=True,
    )


def _make_correlated_analysis(symbol, returns, sign=1.0):
    """Builds an analysis whose price series follows a specific return
    pattern (optionally sign-inverted), so correlation between two such
    analyses is exactly known/controllable rather than incidental."""
    dates = pd.date_range("2026-01-01", periods=len(returns) + 1, freq="D")
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + sign * r))
    a = _analysis_with_stats(symbol)
    a.prices = pd.Series(prices, index=dates)
    return a


_CORR_TEST_RETURNS = [((-1) ** i) * 0.01 * (1 + i % 3) for i in range(40)]


def test_compute_correlation_pairs_detects_perfectly_correlated_pair():
    from ai.psx_suggest import compute_correlation_pairs

    a1 = _make_correlated_analysis("AAA", _CORR_TEST_RETURNS, sign=1.0)
    a2 = _make_correlated_analysis("BBB", _CORR_TEST_RETURNS, sign=1.0)
    pairs = compute_correlation_pairs([a1, a2])
    assert len(pairs) == 1
    sym_a, sym_b, corr = pairs[0]
    assert {sym_a, sym_b} == {"AAA", "BBB"}
    assert corr == pytest.approx(1.0, abs=1e-6)


def test_compute_correlation_pairs_detects_negative_correlation():
    from ai.psx_suggest import compute_correlation_pairs

    a1 = _make_correlated_analysis("AAA", _CORR_TEST_RETURNS, sign=1.0)
    a2 = _make_correlated_analysis("BBB", _CORR_TEST_RETURNS, sign=-1.0)
    pairs = compute_correlation_pairs([a1, a2])
    assert len(pairs) == 1
    _, _, corr = pairs[0]
    assert corr == pytest.approx(-1.0, abs=1e-6)


def test_compute_correlation_pairs_excludes_series_below_min_observations():
    from ai.psx_suggest import compute_correlation_pairs

    short_returns = _CORR_TEST_RETURNS[:5]
    a1 = _make_correlated_analysis("AAA", short_returns, sign=1.0)
    a2 = _make_correlated_analysis("BBB", short_returns, sign=1.0)
    assert compute_correlation_pairs([a1, a2]) == []


def test_format_correlation_context_labels_direction_correctly():
    from ai.psx_suggest import format_correlation_context

    a1 = _make_correlated_analysis("AAA", _CORR_TEST_RETURNS, sign=1.0)
    a2 = _make_correlated_analysis("BBB", _CORR_TEST_RETURNS, sign=-1.0)
    text = format_correlation_context([a1, a2])
    assert "AAA & BBB" in text
    assert "move opposite each other" in text


def test_format_correlation_context_no_pairs_message_when_none_qualify():
    from ai.psx_suggest import format_correlation_context

    assert "no pair currently has" in format_correlation_context([_analysis_with_stats("SOLO")])


def test_format_psx_asset_context_includes_backtest_evidence():
    analysis = _analysis_with_stats()
    analysis.rsi_overbought_backtest = RSIReactionBacktest("overbought", 70.0, 6, -4.5, 83.0, 10)
    analysis.rsi_oversold_backtest = RSIReactionBacktest("oversold", 30.0, 5, 3.2, 60.0, 10)
    analysis.beta_stability_backtest = BetaStabilityBacktest(0.8, 0.9, 0.85, 0.82, True)
    analysis.momentum_persistence_backtest = MomentumPersistenceBacktest(0.42, 25, "persistent")
    analysis.volatility_regime_backtest = VolatilityRegimeBacktest(8.0, 15, 4.0, 20, 10)
    analysis.support_resistance_backtest = SupportResistanceBacktest(10, 70.0, 8, 60.0, 10)
    analysis.eps_growth_trend = EPSGrowthTrend(1.2, 1.0, 20.0, 2, 0.4, 0.3, 33.3)

    text = format_psx_asset_context([analysis])
    assert "6 distinct past episodes" in text
    assert "reversed as the textbook convention would predict 83%" in text
    assert "STABLE" in text
    assert "0.42" in text and "persistent" in text
    assert "supports the 'coiled spring' reading" in text
    assert "support held" in text and "70%" in text and "resistance rejected" in text and "60%" in text
    assert "2 consecutive reported annual period(s)" in text
    assert "+20.0%" in text


def test_format_psx_asset_context_flags_volatility_regime_contradiction():
    analysis = _analysis_with_stats()
    analysis.volatility_regime_backtest = VolatilityRegimeBacktest(4.0, 15, 8.0, 20, 10)
    text = format_psx_asset_context([analysis])
    assert "CONTRADICTS the 'coiled spring' reading" in text


def test_format_psx_asset_context_discloses_missing_backtest_evidence():
    analysis = _analysis_with_stats()  # backtest fields default to None
    text = format_psx_asset_context([analysis])
    assert "not enough real historical episodes" in text
    assert "not enough aligned history to compute" in text
    assert "not enough history to compute" in text
    assert "not enough real historical tests of these levels to compute" in text
    assert "EPS growth trend: not available" in text


def test_format_psx_asset_context_discloses_unstable_beta():
    analysis = _analysis_with_stats()
    analysis.beta_stability_backtest = BetaStabilityBacktest(0.2, 0.9, 1.4, 0.5, False)
    text = format_psx_asset_context([analysis])
    assert "UNSTABLE" in text


def test_format_psx_asset_context_discloses_atr_unavailable():
    text = format_psx_asset_context([_analysis_with_stats()])
    assert "ATR: not available" in text
    assert "ABC" in text
    assert "P/E=5.00" in text


def test_format_psx_asset_context_includes_sector_and_dividend20_flag():
    text = format_psx_asset_context([_analysis_with_stats()])
    assert "sector: Test Sector" in text
    assert "PSX Dividend 20 Index member: yes" in text

    analysis = _analysis_with_stats()
    analysis.is_dividend20_member = False
    text = format_psx_asset_context([analysis])
    assert "PSX Dividend 20 Index member: no" in text


def test_format_psx_asset_context_discloses_missing_fundamentals():
    analysis = _analysis_with_stats()
    analysis.fundamentals = None
    text = format_psx_asset_context([analysis])
    assert "fundamentals: not available" in text


def test_format_psx_asset_context_includes_relative_strength():
    text = format_psx_asset_context([_analysis_with_stats()])
    assert "1-month=3.50%" in text
    assert "3-month=-2.00%" in text


def test_format_psx_asset_context_includes_business_profile():
    text = format_psx_asset_context([_analysis_with_stats()])
    assert "business: Makes widgets." in text
    assert "Jane Doe (CEO)" in text
    assert "auditor: Some Audit Firm" in text


def test_format_psx_asset_context_discloses_missing_profile():
    analysis = _analysis_with_stats()
    analysis.profile = None
    text = format_psx_asset_context([analysis])
    assert "business profile: not available" in text


def test_format_psx_asset_context_includes_financials_and_announcements():
    text = format_psx_asset_context([_analysis_with_stats()])
    assert "Sales: 2025=1,000.00" in text
    assert "PEG: 2025=0.20" in text
    assert "[Others] Apr 7, 2026: A disclosure" in text


def test_format_psx_asset_context_discloses_missing_financials_and_announcements():
    analysis = _analysis_with_stats()
    analysis.financials = None
    analysis.announcements = []
    text = format_psx_asset_context([analysis])
    assert "financials: not available — verify via WebSearch" in text
    assert "recent official PSX disclosures: none found" in text


@patch("ai.psx_suggest.fetch_pakistan_rates")
@patch("ai.psx_suggest.fetch_fx_rate_to_usd")
@patch("ai.psx_suggest.fetch_country_indicators")
def test_build_psx_macro_context_uses_real_pakistan_indicators(mock_indicators, mock_fx, mock_rates):
    mock_indicators.return_value = [
        CountryIndicators(country="Pakistan", gdp_growth_pct=3.7, inflation_pct=3.5, unemployment_pct=5.4)
    ]
    mock_fx.return_value = 277.4
    mock_rates.return_value = PakistanRates(
        as_of="07- Aug - 26", kibor_pct={"3-M": 11.56}, mtb_yield_pct={"3-M": 11.52}, pib_yield_pct={"5-Y": 11.80}
    )
    text = build_psx_macro_context()
    mock_indicators.assert_called_once_with(countries=("PK",))
    assert "3.70%" in text
    assert "277.40" in text
    assert "KIBOR" in text
    assert "11.56%" in text
    assert "SBP policy rate" in text


@patch("ai.psx_suggest.fetch_pakistan_rates", return_value=None)
@patch("ai.psx_suggest.get_psx_history")
@patch("ai.psx_suggest.fetch_fx_rate_to_usd", return_value=None)
@patch("ai.psx_suggest.fetch_country_indicators")
def test_build_psx_summary_includes_capital_and_movers_and_book_wisdom(
    mock_indicators, mock_fx, mock_history, mock_rates
):
    mock_indicators.return_value = [
        CountryIndicators(country="Pakistan", gdp_growth_pct=None, inflation_pct=None, unemployment_pct=None)
    ]
    mock_history.return_value = _make_history()
    assets = [_make_asset("BIGMOVE", change_pct=9.5), _make_asset("SMALLMOVE", change_pct=0.1)]
    summary = build_psx_summary(500000.0, assets, [_analysis_with_stats()])
    assert "500,000.00 PKR" in summary
    assert "hypothetical" in summary.lower()
    assert "BIGMOVE" in summary
    assert "Bulkowski" in summary or "Schwager" in summary or "Murphy" in summary
    assert "Pakistan macro snapshot" in summary
    assert "KSE-100 (top 100 large-cap) index" in summary
    assert "Pairwise correlation" in summary


@patch("ai.psx_suggest.fetch_pakistan_rates", return_value=None)
@patch("ai.psx_suggest.get_psx_history")
@patch("ai.psx_suggest.fetch_fx_rate_to_usd", return_value=None)
@patch("ai.psx_suggest.fetch_country_indicators")
def test_build_psx_summary_scopes_movers_to_selected_index(
    mock_indicators, mock_fx, mock_history, mock_rates
):
    mock_indicators.return_value = [
        CountryIndicators(country="Pakistan", gdp_growth_pct=None, inflation_pct=None, unemployment_pct=None)
    ]
    mock_history.return_value = _make_history()
    assets = [
        _make_asset("INSIDE", listed_in=("KMI30",), change_pct=5.0),
        _make_asset("OUTSIDE", listed_in=("KSE100",), change_pct=9.9),
    ]
    summary = build_psx_summary(500000.0, assets, [_analysis_with_stats()], index_tag="KMI30")
    assert "INSIDE" in summary
    assert "OUTSIDE" not in summary
    assert "KMI-30" in summary


@patch("ai.psx_suggest.get_psx_history")
def test_build_psx_index_context_uses_kse100_history(mock_history):
    from ai.psx_suggest import build_psx_index_context

    mock_history.return_value = _make_history(n=30)
    text = build_psx_index_context()
    mock_history.assert_called_once_with("KSE100")
    assert "KSE-100 (top 100 large-cap) index" in text
    assert "trend=" in text


@patch("ai.psx_suggest.get_psx_company_data")
@patch("ai.psx_suggest.get_psx_history")
def test_analyze_psx_assets_computes_relative_strength_vs_index(mock_history, mock_company_data):
    # Index and stock get different histories via side_effect keyed on the
    # call order (index is fetched once before the per-symbol loop).
    index_history = _make_history(n=30, start=100.0)
    stock_history = _make_history(n=30, start=10.0)
    mock_history.side_effect = [index_history, stock_history]
    mock_company_data.return_value = _make_company_data()

    analyses = analyze_psx_assets([_make_asset("ABC")])
    a = analyses[0]

    index_stats = compute_technical_stats(index_history["Close"], history=index_history)
    stock_stats = compute_technical_stats(stock_history["Close"], history=stock_history)
    expected = stock_stats.change_1m_pct - index_stats.change_1m_pct
    assert a.relative_strength_1m_pct == pytest.approx(expected)


@patch("ai.psx_suggest.build_audit_block")
@patch("ai.psx_suggest.run_claude")
def test_suggest_psx_portfolio_runs_draft_audit_revise(mock_run_claude, mock_audit):
    mock_run_claude.side_effect = ["draft text", "final text"]
    mock_audit.return_value = AuditResult(block="audit block", audit_available=True)

    result = suggest_psx_portfolio("some psx summary")

    assert result == "final text"
    assert mock_run_claude.call_count == 2
    mock_audit.assert_called_once()
    _, kwargs = mock_audit.call_args
    assert "past_lessons" in kwargs
    assert "records_dir" not in kwargs  # build_audit_block no longer takes this param


@patch("ai.psx_suggest.get_psx_history")
def test_fetch_current_psx_price_returns_latest_close(mock_history):
    history = _make_history(n=5)
    mock_history.return_value = history
    assert _fetch_current_psx_price("ABC") == history["Close"].iloc[-1]


@patch("ai.psx_suggest.get_psx_history")
def test_fetch_current_psx_price_none_on_empty_history(mock_history):
    mock_history.return_value = pd.DataFrame()
    assert _fetch_current_psx_price("ABC") is None


@patch("ai.psx_suggest.build_past_lessons", return_value="PAST LESSONS TEXT")
@patch("ai.psx_suggest.build_audit_block", return_value=AuditResult(block="", audit_available=False))
@patch("ai.psx_suggest.run_claude")
def test_suggest_psx_portfolio_includes_past_lessons_in_stage1_draft_prompt(
    mock_run_claude, mock_audit, mock_lessons
):
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_psx_portfolio("some psx summary")
    draft_prompt = mock_run_claude.call_args_list[0].args[0]
    assert "PAST LESSONS TEXT" in draft_prompt


@patch("ai.psx_suggest.build_past_lessons", return_value="PAST LESSONS TEXT")
@patch("ai.psx_suggest.build_audit_block")
@patch("ai.psx_suggest.run_claude")
def test_suggest_psx_portfolio_passes_past_lessons_into_build_audit_block(mock_run_claude, mock_audit, mock_lessons):
    mock_run_claude.side_effect = ["draft", "final"]
    mock_audit.return_value = AuditResult(block="", audit_available=False)
    suggest_psx_portfolio("some psx summary")
    _, kwargs = mock_audit.call_args
    assert kwargs["past_lessons"] == "PAST LESSONS TEXT"


@patch("ai.psx_suggest.build_audit_block", return_value=AuditResult(block="", audit_available=False))
@patch("ai.psx_suggest.run_claude")
def test_suggest_psx_portfolio_resumes_stage1_session_for_a_lean_stage2_call(mock_run_claude, mock_audit):
    # Mirrors ai/portfolio_suggest.py's own version of this test — same
    # session-continuation mechanism, same rationale (see that file's
    # test for the full comment).
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_psx_portfolio("some psx summary")
    draft_call, final_call = mock_run_claude.call_args_list
    session_id = draft_call.kwargs.get("session_id")
    assert session_id
    assert final_call.kwargs.get("resume_session_id") == session_id
    final_prompt = final_call.args[0]
    assert "some psx summary" not in final_prompt


@patch("ai.psx_suggest.build_audit_block", return_value=AuditResult(block="", audit_available=False))
@patch("ai.psx_suggest.run_claude")
def test_suggest_psx_portfolio_falls_back_to_full_context_when_resume_fails(mock_run_claude, mock_audit):
    fallback_failure = f"{CLI_FAILED_PREFIX} (no conversation found)."
    mock_run_claude.side_effect = ["draft text", fallback_failure, "final answer after retry"]
    messages = []
    result = suggest_psx_portfolio("some psx summary", on_stage=messages.append)
    assert result == "final answer after retry"
    assert mock_run_claude.call_count == 3
    retry_call = mock_run_claude.call_args_list[2]
    assert "resume_session_id" not in retry_call.kwargs
    retry_prompt = retry_call.args[0]
    assert "some psx summary" in retry_prompt
    assert "draft text" in retry_prompt
    assert any("retrying with a" in m.lower() for m in messages)


@patch("ai.psx_suggest.save_portfolio_session")
@patch("ai.psx_suggest.build_audit_block")
@patch("ai.psx_suggest.run_claude")
def test_suggest_psx_portfolio_on_stage_message_sequence_is_detailed(mock_run_claude, mock_audit, mock_save):
    mock_run_claude.side_effect = ["draft", "final"]
    mock_audit.return_value = AuditResult(block="", audit_available=True)
    messages = []
    suggest_psx_portfolio("summary", on_stage=messages.append, save_record=True)
    assert any("past-session context" in m.lower() for m in messages)
    assert any("drafting" in m.lower() for m in messages)
    assert any("audit" in m.lower() for m in messages)
    assert any("revising" in m.lower() for m in messages)
    assert any("saving session record" in m.lower() for m in messages)


@patch("ai.psx_suggest.build_audit_block")
@patch("ai.psx_suggest.run_claude", return_value=CLI_MISSING_MESSAGE)
def test_suggest_psx_portfolio_returns_early_on_missing_cli(mock_run_claude, mock_audit):
    result = suggest_psx_portfolio("summary")
    assert result == CLI_MISSING_MESSAGE
    assert mock_audit.call_count == 0


@patch("ai.psx_suggest.build_audit_block")
@patch("ai.psx_suggest.run_claude")
def test_suggest_psx_portfolio_returns_early_on_cli_failure(mock_run_claude, mock_audit):
    failure = f"{CLI_FAILED_PREFIX} (boom)."
    mock_run_claude.return_value = failure
    result = suggest_psx_portfolio("summary")
    assert result == failure
    assert mock_audit.call_count == 0


@patch("ai.psx_suggest.save_portfolio_session")
@patch("ai.psx_suggest.build_audit_block")
@patch("ai.psx_suggest.run_claude")
def test_suggest_psx_portfolio_saves_record_with_psx_records_dir(mock_run_claude, mock_audit, mock_save):
    mock_run_claude.side_effect = ["draft", "final"]
    mock_audit.return_value = AuditResult(block="", audit_available=False)
    suggest_psx_portfolio("summary", save_record=True)
    assert mock_save.call_count == 1
    _, kwargs = mock_save.call_args
    assert kwargs["records_dir"] == Path(config.PSX_RECORDS_DIR)
