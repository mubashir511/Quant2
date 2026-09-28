import math
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import pandas as pd
import pytest

import config
from ai.claude_cli import CLI_FAILED_PREFIX, CLI_MISSING_MESSAGE
from ai.ftmo_suggest import (
    AUDIT_INSTRUCTION,
    MIN_VIABLE_SIZE_ATR_MULTIPLE,
    FtmoAssetAnalysis,
    IntradayBacktests,
    _backtest_all_sr_levels,
    _build_bare_base_analysis,
    _compute_intraday_backtests,
    _enrich_with_native_d1,
    _fetch_current_ftmo_price,
    _fetch_ftmo_headlines,
    _format_ftmo_macro_news_context,
    _format_intraday_backtests,
    _ftmo_commission_pct_round_turn,
    _real_backtest_execution_kwargs,
    _tactical_trend_vs_regime_conflict,
    aligned_h1_h4_trend_direction,
    analyze_ftmo_asset_live,
    analyze_ftmo_assets,
    build_ftmo_stage1_instruction,
    build_ftmo_stage2_instruction,
    build_ftmo_summary,
    build_trend_radar,
    classify_long_term_alignment,
    compute_ftmo_correlation_pairs,
    _compute_divergence_for_history,
    format_chart_structure,
    format_ftmo_asset_context,
    format_ftmo_correlation_context,
    format_setup_signals,
    format_ftmo_status_context,
    format_ftmo_held_position_sizing_rates,
    format_ftmo_min_viable_size,
    format_ftmo_trade_cost,
    format_long_term_alignment,
    format_long_term_alignment_short,
    format_trade_zone,
    read_latest_suggestion,
    suggest_ftmo_portfolio,
)
from ai.ftmo_suggest import _write_latest_suggestion
from ai.portfolio_suggest import AssetAnalysis, AuditResult
from analysis.backtest import FavorableExcursionStats, RSIReactionBacktest, SupportResistanceBacktest
from analysis.chart_structure import (
    FIB_RATIOS,
    BreakoutEvent,
    ChartStructureSnapshot,
    FibonacciLevels,
    LiquiditySweepEvent,
    SRLevel,
    SRLevelsResult,
)
from analysis.setup_classifier import SetupSignal
from analysis.technical import TechnicalStats, compute_technical_stats
from data.mt5_source import AccountSummary, ContractSpec, MarketAsset, PendingOrder, Position, TradeCost


def _ts(market_regime=None, last_price=None, trend=None, momentum_acceleration=None, atr=None) -> TechnicalStats:
    """Minimal TechnicalStats with only a few fields set — everything
    format_long_term_alignment/its tests need, without needing a real
    price series shaped to produce a specific regime. Built by keyword,
    not position, specifically to avoid a real off-by-N field-index bug
    (caught while writing this: market_regime is TechnicalStats' 12th
    field, not its 10th, by direct field-order count)."""
    return TechnicalStats(
        last_price=last_price, sma20=None, pct_vs_sma20=None, trend=trend,
        change_1m_pct=None, change_3m_pct=None, change_6m_pct=None,
        volatility_annualized_pct=None, support=None, resistance=None,
        range_width_pct=None, market_regime=market_regime, atr=atr,
        atr_pct=None, rsi=None, volume_trend_pct=None,
        momentum_acceleration=momentum_acceleration,
    )


def _empty_chart_structure() -> ChartStructureSnapshot:
    return ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])


def test_format_chart_structure_renders_a_real_band_not_a_single_point():
    # Real gap found 2026-09-20, direct user challenge: S/R was always
    # shown as one price point even though the clustering that produces
    # it already implies a real band width.
    structure = ChartStructureSnapshot(
        fibonacci=None,
        sr_levels=SRLevelsResult(
            resistance_levels=[
                SRLevel(price=1.2001, touches=3, distance_pct=0.5, low=1.1998, high=1.2004, weighted_score=2.7)
            ],
            support_levels=[],
        ),
        trendlines=None,
        patterns=[],
    )
    text = format_chart_structure("H1", structure)
    assert "1.1998-1.2004" in text
    assert "2.7 recency-weighted" in text
    assert "3x raw" in text


def test_format_chart_structure_renders_breakouts_and_sweeps_with_tick_volume_wording():
    # Real vs. fake — added 2026-09-20, direct user challenge. "tick-
    # volume" wording is deliberate (MT5's Volume for FX/CFDs isn't real
    # traded volume), not "volume" bare.
    structure = ChartStructureSnapshot(
        fibonacci=None,
        sr_levels=None,
        trendlines=None,
        patterns=[],
        breakouts=[
            BreakoutEvent(
                level_price=1.2000, direction="up", close_through_pct=0.5,
                volume_ratio=2.1, volume_confirmed=True, bars_ago=0,
            )
        ],
        liquidity_sweeps=[
            LiquiditySweepEvent(
                level_price=1.1950, direction="swept_above", wick_penetration_pct=0.2,
                bars_ago=1, volume_ratio=None,
            )
        ],
    )
    text = format_chart_structure("H1", structure)
    assert "tick-volume-confirmed" in text
    assert "stop-hunt/liquidity sweep" in text
    assert "1.2000" in text
    assert "1.1950" in text


def test_format_chart_structure_falls_back_to_a_single_point_when_band_is_none():
    # A pre-upgrade SRLevel (low/high never set, e.g. an older direct
    # construction) must still render, not crash or print "None-None".
    structure = ChartStructureSnapshot(
        fibonacci=None,
        sr_levels=SRLevelsResult(
            resistance_levels=[SRLevel(price=1.2001, touches=3, distance_pct=0.5)], support_levels=[]
        ),
        trendlines=None,
        patterns=[],
    )
    text = format_chart_structure("H1", structure)
    assert "None" not in text
    assert "1.2001" in text
    assert "n/a recency-weighted" in text


def test_format_setup_signals_renders_the_confidence_tier():
    signals = [SetupSignal(name="reversal_candidate", detail="two peaks", confidence="strong")]
    text = format_setup_signals("H1", signals)
    assert "reversal_candidate [strong]" in text
    assert "two peaks" in text


def test_format_setup_signals_none_computed_when_empty():
    assert format_setup_signals("H1", []) == "  H1 setup read: none computed"


def _fib_with_real_levels(**overrides) -> FibonacciLevels:
    defaults = dict(
        swing_high=110.0, swing_low=90.0, high_is_more_recent=True,
        levels={}, current_price=100.0, nearest_level_name="50.0%",
        nearest_level_price=100.0, distance_to_nearest_pct=0.0,
    )
    f = FibonacciLevels(**{**defaults, **overrides})
    if not f.levels:
        span = f.swing_high - f.swing_low
        f.levels = {
            f"{ratio * 100:.1f}%": (f.swing_high - ratio * span if f.high_is_more_recent else f.swing_low + ratio * span)
            for ratio in FIB_RATIOS
        }
    return f


def test_format_trade_zone_labels_it_as_a_candidate_not_a_command():
    # Per the user's own confirmed decision: Phase 5's trade-zone
    # construction is advisory-only — the prompt text itself must say so
    # explicitly, never read like an enforced instruction.
    support = SRLevel(price=98.0, touches=3, distance_pct=-2.0, low=97.5, high=98.5)
    resistance = SRLevel(price=108.0, touches=2, distance_pct=8.0, low=107.5, high=108.5)
    structure = ChartStructureSnapshot(
        fibonacci=_fib_with_real_levels(),
        sr_levels=SRLevelsResult(resistance_levels=[resistance], support_levels=[support]),
        trendlines=None, patterns=[],
    )
    base = _make_base_analysis()
    a = FtmoAssetAnalysis(
        base=base, h4_stats=_ts(), h1_stats=_ts(),
        h4_structure=_empty_chart_structure(), h1_structure=_empty_chart_structure(), trade_cost=None,
        m5_stats=_ts(last_price=100.0, atr=1.0), m5_structure=structure,
    )
    signals = [SetupSignal(name="pullback_continuation", detail="x")]
    text = format_trade_zone(a, signals)
    assert text.lstrip().startswith("M5 trade-zone candidate")
    assert "advisory" in text.lower()
    assert "never a hard rule" in text
    assert "buy [moderate, basis=pullback_continuation" in text
    assert "sell: no real structure/reward-risk currently clears the bar" in text


def test_format_trade_zone_context_only_when_no_real_structure():
    base = _make_base_analysis()
    a = FtmoAssetAnalysis(
        base=base, h4_stats=_ts(), h1_stats=_ts(),
        h4_structure=_empty_chart_structure(), h1_structure=_empty_chart_structure(), trade_cost=None,
        m5_stats=_ts(last_price=100.0, atr=2.0),
    )
    text = format_trade_zone(a, [])
    assert "context only" in text


def test_compute_divergence_for_history_none_on_empty_history():
    assert _compute_divergence_for_history(pd.DataFrame()) is None


@patch("ai.ftmo_suggest.detect_rsi_divergence")
@patch("ai.ftmo_suggest.find_swing_points")
def test_compute_divergence_for_history_wires_swing_points_into_detect_rsi_divergence(mock_find, mock_detect):
    # A real divergence fixture tuned to survive find_swing_points' own
    # fractal window is already covered thoroughly in test_technical.py's
    # own detect_rsi_divergence tests — this only proves the wiring:
    # real swing points get passed through to the real detection function,
    # not re-derived a second, different way.
    mock_find.return_value = (["high1", "high2"], ["low1", "low2"])
    mock_detect.return_value = "a real DivergenceSignal"
    history = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
    result = _compute_divergence_for_history(history)
    assert result == "a real DivergenceSignal"
    mock_detect.assert_called_once()
    call_args, call_kwargs = mock_detect.call_args
    assert call_args[0].equals(history["Close"])  # pd.Series can't compare via plain == in assert_called_with
    assert call_kwargs == {"swing_highs": ["high1", "high2"], "swing_lows": ["low1", "low2"]}


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


@pytest.fixture(autouse=True)
def _no_real_curiosity_report():
    """Same rationale as _no_real_past_lessons above: build_curiosity_report
    does real MT5 deal-history I/O, real file I/O against records/ftmo/
    and records/ftmo_curiosity/, and a real OpenRouter model call — patch
    it to inert by default so every suggest_ftmo_portfolio()-calling test
    stays fast and network-free."""
    with patch("ai.ftmo_suggest.build_curiosity_report", return_value=""):
        yield


@pytest.fixture(autouse=True)
def _no_real_trade_journal():
    """Same rationale as _no_real_past_lessons/_no_real_curiosity_report
    above — real leak caught live 2026-09-19: _write_latest_suggestion
    unconditionally calls ai.trade_journal.record_proposals (a pure
    addition, direct user request), which does real file I/O against
    this project's own real records/ftmo_trade_journal/ and
    obsidian_vault/ directories. Confirmed by running the full suite
    once without this fixture and finding real test-fixture symbols
    written into both. Patched to inert by default so every
    suggest_ftmo_portfolio()-calling test stays fast and leak-free."""
    with patch("ai.ftmo_suggest.trade_journal.record_proposals", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _isolated_symbol_news_paths(tmp_path):
    """Real leak caught live 2026-09-20: once data.symbol_news.get_
    symbol_news_block gained its "intelligent search" fallback (a
    genuine Google News search built from MT5's own real symbol
    description, for a category with no known ticker convention), a
    test like test_analyze_ftmo_asset_live_no_backtests_when_d1_history_
    empty — which passes description="New Symbol" for a fake "NEWSYM"
    symbol, and doesn't mock _fetch_ftmo_headlines/get_symbol_category —
    started making a REAL Google search and a REAL vault write into
    this project's own real records/ftmo_symbol_news/ and
    obsidian_vault/News/, since neither path was isolated here before.
    Same rationale as _no_real_trade_journal above: isolates the WRITE
    side unconditionally so any test with imperfect/absent news mocking
    can never leak into real production paths, even if it still makes a
    real (undesirable but harmless) network call. Confirmed by running
    the full suite once without this fixture and finding a real
    "NEWSYM.json" written into the real records/ftmo_symbol_news/."""
    with (
        patch.object(config, "SYMBOL_NEWS_DIR", str(tmp_path / "ftmo_symbol_news")),
        patch.object(config, "CATEGORY_NEWS_DIR", str(tmp_path / "ftmo_category_news")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "NEWS_FETCH_CACHE_FILE", str(tmp_path / "news_fetch_cache.json")),
    ):
        yield


@pytest.fixture(autouse=True)
def _no_real_economic_calendar():
    """build_ftmo_summary fetches the free economic-calendar feed — never
    let a test hit the real network or the real records/ cache."""
    with patch("ai.ftmo_suggest.economic_calendar.fetch_calendar_events", return_value=[]):
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


def _make_ftmo_correlated_analysis(symbol, returns, sign=1.0):
    """Builds an FtmoAssetAnalysis whose base.prices follows a specific
    return pattern (optionally sign-inverted), so correlation between two
    such analyses is exactly known/controllable rather than incidental —
    mirrors tests/test_psx_suggest.py::_make_correlated_analysis exactly,
    adapted for FtmoAssetAnalysis's nested base.prices/base.symbol shape."""
    dates = pd.date_range("2026-01-01", periods=len(returns) + 1, freq="D")
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + sign * r))
    base = _make_base_analysis(symbol=symbol)
    base.prices = pd.Series(prices, index=dates)
    return FtmoAssetAnalysis(
        base=base,
        h4_stats=_ts(),
        h1_stats=_ts(),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )


_CORR_TEST_RETURNS = [((-1) ** i) * 0.01 * (1 + i % 3) for i in range(40)]


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


@patch("ai.ftmo_suggest._fetch_ftmo_headlines", return_value=[])
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_enrich_with_native_d1_backfills_from_mt5_when_no_yahoo_mapping(mock_fetch, mock_headlines):
    # The real gap this closes: 16 of 17 real FTMO Market Watch symbols
    # (confirmed live) have no Yahoo mapping at all — every forex pair,
    # every single-stock CFD, both crypto pairs — so they never got any
    # D1 technical/backtest coverage before this existed. This test is
    # about THAT D1 backfill, not headlines — _fetch_ftmo_headlines is
    # mocked out entirely (rather than relying on category resolution
    # failing in a test environment with no real MT5 connection) so this
    # stays deterministic and headlines==[] is asserted for real, not as
    # an accidental side effect of get_symbol_category degrading to
    # "Uncategorized" here.
    base = _make_base_analysis(display_name=None)
    mock_fetch.return_value = _make_intraday_history(n=300)

    result = _enrich_with_native_d1(base)

    mock_fetch.assert_called_once_with("EURUSD", "D1", count=1500)
    assert result is not base
    assert result.display_name == "Euro vs US Dollar"
    assert result.stats.last_price is not None
    assert result.headlines == []
    assert result.contract_spec == base.contract_spec


# --- _fetch_ftmo_headlines: real news for Mega Session, added 2026-09-20
# (previously always headlines=[], a long-standing disclosed gap) -------


@patch("ai.ftmo_suggest.symbol_news.get_symbol_news_block")
@patch("ai.ftmo_suggest.get_symbol_category", return_value="Metals CFD")
@patch("ai.ftmo_suggest.symbol_news.get_category_news_items", return_value=[])
def test_fetch_ftmo_headlines_returns_real_titles(mock_category_items, mock_category, mock_block):
    mock_block.return_value = [
        {"title": "Gold hits record high", "summary": "", "source": "Reuters", "published": ""},
        {"title": "Fed holds rates steady", "summary": "", "source": "AP", "published": ""},
    ]
    result = _fetch_ftmo_headlines("XAUUSD", "Gold vs US Dollar")
    assert result == ["Gold hits record high", "Fed holds rates steady"]
    mock_block.assert_called_once_with(
        "XAUUSD", "Metals CFD", limit=config.RESEARCHER_HEADLINES_PER_SYMBOL, description="Gold vs US Dollar"
    )


@patch("ai.ftmo_suggest.symbol_news.get_symbol_news_block", return_value=[])
@patch("ai.ftmo_suggest.get_symbol_category", return_value="Forex")
@patch("ai.ftmo_suggest.symbol_news.get_category_news_items")
def test_fetch_ftmo_headlines_includes_tagged_category_items(mock_category_items, mock_category, mock_block):
    mock_category_items.return_value = [{"title": "FXStreet: EUR outlook", "summary": "", "source": "FXStreet", "published": ""}]
    result = _fetch_ftmo_headlines("EURUSD", "Euro vs US Dollar")
    assert result == ["[Forex] FXStreet: EUR outlook"]
    mock_category_items.assert_called_once_with("Forex", limit=config.RESEARCHER_HEADLINES_PER_SYMBOL)


def test_fetch_ftmo_headlines_never_includes_macro_news():
    # Real design decision, not an oversight: macro is account-wide and
    # this function runs once PER SYMBOL feeding into ONE combined mega-
    # session prompt -- including it here would duplicate the same ~15-20
    # macro lines once per symbol. It's added exactly once at the top
    # level instead (_format_ftmo_macro_news_context / build_ftmo_summary).
    import inspect

    source = inspect.getsource(_fetch_ftmo_headlines)
    assert "get_macro_news_items" not in source


@patch("ai.ftmo_suggest.symbol_news.get_macro_news_items")
def test_format_ftmo_macro_news_context_lists_real_items(mock_macro):
    mock_macro.return_value = [{"title": "[^GSPC] Stocks rally", "summary": "", "source": "", "published": ""}]
    block = _format_ftmo_macro_news_context()
    assert "Stocks rally" in block
    assert "shared context, not symbol-specific" in block


@patch("ai.ftmo_suggest.symbol_news.get_macro_news_items", return_value=[])
def test_format_ftmo_macro_news_context_honest_when_none_found(mock_macro):
    assert "none found" in _format_ftmo_macro_news_context()


@patch("ai.ftmo_suggest.symbol_news.get_macro_news_items", side_effect=RuntimeError("network down"))
def test_format_ftmo_macro_news_context_degrades_on_exception(mock_macro):
    assert "unavailable" in _format_ftmo_macro_news_context()


@patch("ai.ftmo_suggest.get_symbol_category", side_effect=RuntimeError("MT5 not connected"))
def test_fetch_ftmo_headlines_degrades_to_empty_on_any_exception(mock_category):
    # Must never let a news-fetch failure withhold the real technical/
    # backtest data the rest of the pipeline computes.
    assert _fetch_ftmo_headlines("XAUUSD", "Gold vs US Dollar") == []


@patch("ai.ftmo_suggest.fetch_mt5_price_history")
@patch("ai.ftmo_suggest._fetch_ftmo_headlines")
def test_enrich_with_native_d1_includes_real_headlines_when_available(mock_headlines, mock_fetch):
    mock_headlines.return_value = ["A real EURUSD headline"]
    base = _make_base_analysis(display_name=None)
    mock_fetch.return_value = _make_intraday_history(n=300)

    result = _enrich_with_native_d1(base)

    assert result.headlines == ["A real EURUSD headline"]
    mock_headlines.assert_called_once_with("EURUSD", base.description)


@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_enrich_with_native_d1_leaves_bare_when_mt5_also_has_nothing(mock_fetch):
    base = _make_base_analysis(display_name=None)
    mock_fetch.return_value = _empty_history()

    result = _enrich_with_native_d1(base)

    assert result is base
    assert result.display_name is None


@patch("ai.ftmo_suggest.is_symbol_tradable_now")
@patch("ai.ftmo_suggest.get_contract_spec", return_value=None)
def test_build_bare_base_analysis_sets_market_open(mock_contract_spec, mock_tradable):
    mock_tradable.return_value = False
    base = _build_bare_base_analysis(MarketAsset("EURUSD", "Euro vs US Dollar", 1.1000, 1.1005))
    assert base.market_open is False
    mock_tradable.assert_called_once()
    assert mock_tradable.call_args.args[0] == "EURUSD"


@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_enrich_with_native_d1_preserves_market_open_from_base(mock_fetch):
    # _enrich_with_native_d1 rebuilds a fresh AssetAnalysis from scratch
    # on the real-backfill path -- must carry market_open over from the
    # bare base rather than silently dropping back to its own default.
    base = _make_base_analysis(display_name=None)
    base.market_open = False
    mock_fetch.return_value = _make_intraday_history(n=300)

    result = _enrich_with_native_d1(base)

    assert result.market_open is False


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
    #
    # backtest_rsi_reaction is now called TWICE per symbol (D1, then the
    # M5 intraday variant — see _compute_intraday_backtests) since
    # both share this same imported name in ai.ftmo_suggest's namespace.
    # The first call is the D1 one this test is really about; the second
    # (intraday) call is covered by its own dedicated test below.
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
    assert mock_rsi_bt.call_count == 2
    kwargs = mock_rsi_bt.call_args_list[0].kwargs
    assert kwargs["round_trip_cost_pct"] == pytest.approx(0.006)  # no contract_spec -> spread only
    assert kwargs["long_swap_pct_per_day"] == pytest.approx(-0.008)
    assert kwargs["short_swap_pct_per_day"] == pytest.approx(0.001)
    assert kwargs["min_stop_distance_pct"] == pytest.approx(0.1)


@patch("ai.ftmo_suggest.backtest_support_resistance_reaction")
@patch("ai.ftmo_suggest.backtest_rsi_reaction")
@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_assets_intraday_backtest_call_has_zeroed_swap_and_intraday_bars(
    mock_fetch_history, mock_contract_spec, mock_trade_economics, mock_rsi_bt, mock_sr_bt
):
    # Both backtest_rsi_reaction and backtest_support_resistance_reaction
    # are called TWICE (D1 via _enrich_with_native_d1, then the M5
    # intraday variant — both share this same imported name). The SECOND
    # call to each is the intraday one this test is really about: it must
    # use zeroed swap (see config.py's own real swap-per-bar-bug comment)
    # and the intraday-scaled max_holding_bars/excursion_horizon_bars,
    # not the D1 ones, even though the SAME real, nonzero swap is fetched
    # via TradeCost.
    mock_contract_spec.return_value = None
    mock_fetch_history.return_value = _make_intraday_history()
    real_cost = _make_trade_cost(
        category="Forex", spread_pct_of_price=0.006,
        swap_long_pct_per_day=-0.008, swap_short_pct_per_day=0.001, min_stop_distance_pct=0.1,
    )
    mock_trade_economics.return_value = real_cost
    mock_rsi_bt.return_value = (None, None)
    mock_sr_bt.return_value = None

    analyze_ftmo_assets([MarketAsset("EURUSD", "Euro vs US Dollar", 1.1000, 1.1005)])

    assert mock_rsi_bt.call_count == 2
    intraday_kwargs = mock_rsi_bt.call_args_list[1].kwargs
    assert intraday_kwargs["long_swap_pct_per_day"] == 0.0
    assert intraday_kwargs["short_swap_pct_per_day"] == 0.0
    assert intraday_kwargs["max_holding_bars"] == config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS
    assert intraday_kwargs["excursion_horizon_bars"] == config.TRADE_SIM_INTRADAY_EXCURSION_HORIZON_BARS
    # The simulated stop is the same 2x-M5-ATR distance Clerk enforces, target 2:1.
    assert intraday_kwargs["stop_atr_multiple"] == config.M5_ATR_STOP_MULTIPLE
    assert intraday_kwargs["target_atr_multiple"] == 2 * config.M5_ATR_STOP_MULTIPLE
    # Real spread cost still applies — only swap is overridden.
    assert intraday_kwargs["round_trip_cost_pct"] == pytest.approx(0.006)

    assert mock_sr_bt.call_count == 2
    sr_kwargs = mock_sr_bt.call_args_list[1].kwargs
    # 30 bars is too short for a median ATR% (needs >=20 rolling values) -> the clamp's upper bound.
    assert sr_kwargs["proximity_pct"] == config.M5_SR_PROXIMITY_MAX_PCT
    assert sr_kwargs["long_swap_pct_per_day"] == 0.0
    # D1's own S/R call (first) never gets an M5-only proximity override.
    assert "proximity_pct" not in mock_sr_bt.call_args_list[0].kwargs


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_assets_adds_d1_structure_and_fetches_no_monthly_or_m15(
    mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    # 2026-09-24: Monthly and M15 were dropped (D1 covers Monthly's context, M5 + H1 split M15's duties).
    mock_contract_spec.return_value = None
    mock_fetch_history.return_value = _make_intraday_history()
    mock_trade_economics.return_value = _make_trade_cost()

    results = analyze_ftmo_assets([MarketAsset("EURUSD", "Euro vs US Dollar", 1.1000, 1.1005)])

    assert results[0].d1_structure is not None
    assert results[0].m5_stats.last_price is not None
    fetched = [c.args[1] for c in mock_fetch_history.call_args_list]
    assert fetched[:3] == ["D1", "H4", "H1"]
    assert "MN1" not in fetched and "M15" not in fetched
    assert fetched.count("M5") == 1  # ONE full-depth fetch feeds the M5 reads, backtests and level reliability


@patch("ai.ftmo_suggest.is_symbol_tradable_now")
@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_asset_live_sets_market_open(
    mock_fetch_history, mock_contract_spec, mock_trade_economics, mock_tradable
):
    mock_fetch_history.return_value = _make_intraday_history(n=300)
    mock_contract_spec.return_value = None
    mock_trade_economics.return_value = _make_trade_cost()
    mock_tradable.return_value = False

    result = analyze_ftmo_asset_live("EURUSD", bid=1.1000, ask=1.1005, description="Euro vs US Dollar")

    assert result.base.market_open is False


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
    # Both are called TWICE now (D1, then the M5 intraday variant —
    # see _compute_intraday_backtests, which reuses the same imported
    # names); the first call to each is the D1 one this test is really
    # about, using the real, unmodified cost kwargs.
    assert mock_rsi_bt.call_count == 2
    assert mock_rsi_bt.call_args_list[0] == call(history, **expected_kwargs)
    assert mock_sr_bt.call_count == 2
    assert mock_sr_bt.call_args_list[0] == call(history, **expected_kwargs)
    # Real commission actually got folded in, not just the bare spread.
    assert expected_kwargs["round_trip_cost_pct"] > 0.005


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_asset_live_adds_d1_structure_and_the_m5_reads(
    mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    # Live-single-symbol sibling of the test above — the exact path Clerk's technical context and the
    # app's Asset Health popup call.
    mock_fetch_history.return_value = _make_intraday_history(n=300)
    mock_contract_spec.return_value = None
    mock_trade_economics.return_value = _make_trade_cost()

    result = analyze_ftmo_asset_live("EURUSD", bid=1.1000, ask=1.1005, description="Euro vs US Dollar")

    assert result.d1_structure is not None
    assert result.m5_stats.last_price is not None
    assert result.m5_atr_pct_median is not None  # 300 M5 bars -> 286 rolling ATR values (>= the 20 needed)
    fetched = [c.args[1] for c in mock_fetch_history.call_args_list]
    assert "MN1" not in fetched and "M15" not in fetched


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec", return_value=None)
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_asset_live_no_backtests_when_d1_history_empty(
    mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    # D1 empty, H4/H1 still have real bars — mirrors a genuinely new listing with no daily history yet
    # but active intraday trading. 4th item is the M5 intraday-backtest fetch (see
    # _compute_intraday_backtests) — unaffected by D1 being empty; 5th is the decision-tier M5 fetch
    # (see _compute_intraday_reads).
    mock_fetch_history.side_effect = [
        _empty_history(), _make_intraday_history(), _make_intraday_history(),
        _make_intraday_history(), _make_intraday_history(),
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
    # then the H4/H1 extension loop and the M5 decision-tier read compute the rest.
    _d1_call, h4_call, h1_call, m5_call = mock_compute_stats.call_args_list
    h4_periods = h4_call.kwargs["periods_per_year"]
    h1_periods = h1_call.kwargs["periods_per_year"]
    assert h4_periods > 252  # scaled for intraday, not left at the daily default
    assert h1_periods > h4_periods  # H1 bars are more frequent than H4 bars
    # Decision-tier read: the faster timeframe scales strictly higher.
    assert m5_call.kwargs["periods_per_year"] > h1_periods


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


def test_format_ftmo_asset_context_includes_d1_chart_structure_and_setup_read_but_no_monthly():
    from analysis.chart_structure import ChartPattern

    d1_history = _make_trending_history(start=1.0, drift=0.01)
    d1_stats = compute_technical_stats(d1_history["Close"], history=d1_history)
    base = AssetAnalysis(
        symbol="EURUSD", description="Euro vs US Dollar", bid=1.1000, ask=1.1005,
        display_name="Euro vs US Dollar", stats=d1_stats,
    )
    d1_structure = ChartStructureSnapshot(
        fibonacci=None, sr_levels=None, trendlines=None,
        patterns=[ChartPattern(name="double_top", detail="D1 peaks at 1.2000 and 1.2010")],
    )
    analysis = FtmoAssetAnalysis(
        base=base,
        h4_stats=compute_technical_stats(pd.Series(dtype=float)),
        h1_stats=compute_technical_stats(pd.Series(dtype=float)),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
        d1_structure=d1_structure,
    )
    text = format_ftmo_asset_context([analysis])
    assert "D1 chart structure" in text
    assert "D1 peaks at 1.2000 and 1.2010" in text
    assert "D1 setup read" in text
    assert "Monthly" not in text and "monthly" not in text.replace("monthly bars", "")


# --- Candlestick / momentum-acceleration signals actually reach the prompt text ---


def test_format_ftmo_asset_context_shows_short_term_momentum_line():
    base = _make_base_analysis()
    h4_stats = _ts(last_price=1.1000, momentum_acceleration="accelerating_up")
    analysis = FtmoAssetAnalysis(
        base=base, h4_stats=h4_stats, h1_stats=_ts(),
        h4_structure=_empty_chart_structure(), h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    assert "short-term momentum (12-bar): accelerating_up" in text


def test_format_ftmo_asset_context_surfaces_in_progress_move_signal():
    base = _make_base_analysis()
    h4_stats = _ts(last_price=1.1000, market_regime="sideways", momentum_acceleration="accelerating_up")
    analysis = FtmoAssetAnalysis(
        base=base, h4_stats=h4_stats, h1_stats=_ts(),
        h4_structure=_empty_chart_structure(), h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    assert "in_progress_move" in text


def test_format_ftmo_asset_context_surfaces_busted_pattern_reversal_signal():
    from analysis.chart_structure import ChartPattern

    base = _make_base_analysis()
    h4_stats = _ts(last_price=1.1000, market_regime="trending_up")
    h4_structure = ChartStructureSnapshot(
        fibonacci=None, sr_levels=None, trendlines=None,
        patterns=[ChartPattern(name="double_top", detail="two peaks at 1.2000 and 1.2010")],
    )
    analysis = FtmoAssetAnalysis(
        base=base, h4_stats=h4_stats, h1_stats=_ts(),
        h4_structure=h4_structure, h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    assert "busted_pattern_reversal" in text


def test_format_ftmo_asset_context_surfaces_candlestick_reversal_confirmed_signal():
    from analysis.chart_structure import ChartPattern

    base = _make_base_analysis()
    h4_stats = _ts(last_price=1.1000, trend="downtrend")
    h4_structure = ChartStructureSnapshot(
        fibonacci=None, sr_levels=None, trendlines=None,
        patterns=[ChartPattern(name="hammer", detail="lower shadow 3x the real body")],
    )
    analysis = FtmoAssetAnalysis(
        base=base, h4_stats=h4_stats, h1_stats=_ts(),
        h4_structure=h4_structure, h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    text = format_ftmo_asset_context([analysis])
    assert "candlestick_reversal_confirmed" in text
    assert "hammer" in text


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


def _analysis_with_regimes(h4_regime, d1_regime) -> FtmoAssetAnalysis:
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
        m5_stats=_ts(market_regime=h4_regime),  # the "short" regime is the M5 (decision) one since 2026-09-24
    )


def test_long_term_alignment_structurally_backed_when_the_daily_backdrop_agrees():
    analysis = _analysis_with_regimes("trending_up", "trending_up")
    text = format_long_term_alignment(analysis)
    assert "STRUCTURALLY BACKED" in text
    assert "daily backdrop" in text and "monthly" not in text


def test_long_term_alignment_counter_trend_spike_when_the_daily_backdrop_opposes():
    analysis = _analysis_with_regimes("choppy_up", "trending_down")
    text = format_long_term_alignment(analysis)
    assert "COUNTER-TREND SPIKE" in text
    # The user explicitly wants this framed as tradeable-but-managed, not
    # discarded outright — confirm the language reflects that, not just
    # the label.
    assert "genuinely tradeable" in text
    assert "tighter stop" in text


def test_long_term_alignment_none_when_no_real_backdrop_direction():
    # The daily read is sideways — no direction to compare against at all, distinct from a genuine
    # CONTRADICTING backdrop.
    analysis = _analysis_with_regimes("choppy_up", "sideways")
    text = format_long_term_alignment(analysis)
    assert "no real daily directional backdrop" in text
    assert "STRUCTURALLY BACKED" not in text
    assert "COUNTER-TREND" not in text


def test_long_term_alignment_none_when_h4_itself_flat_or_missing():
    flat = _analysis_with_regimes("sideways", "trending_up")
    assert "no real net direction" in format_long_term_alignment(flat)

    missing = _analysis_with_regimes(None, "trending_up")
    assert "not available" in format_long_term_alignment(missing)


def test_format_ftmo_asset_context_includes_long_term_alignment_and_no_monthly_read():
    analysis = _analysis_with_regimes("trending_up", "trending_up")
    text = format_ftmo_asset_context([analysis])
    assert "Monthly technical" not in text
    assert "Long-term alignment" in text
    assert "STRUCTURALLY BACKED" in text


# --- _tactical_trend_vs_regime_conflict / conflict warning, added
# 2026-09-11 after a real incident: a BTCUSD buy cited STRUCTURALLY
# BACKED (market_regime agreed with D1/Monthly) while H1 and H4's own
# TREND fields (a genuinely different metric, format_mtf_confluence's
# own H4-vs-H1 line) both explicitly read downtrend underneath it,
# unaddressed in the thesis.


def _analysis_with_trends(h4_trend, h1_trend, h4_regime, d1_regime) -> FtmoAssetAnalysis:
    base = AssetAnalysis(
        symbol="BTCUSD", description="Bitcoin vs US Dollar", bid=78327.46, ask=78328.46,
        display_name="Bitcoin vs US Dollar", stats=_ts(market_regime=d1_regime),
    )
    return FtmoAssetAnalysis(
        base=base,
        h4_stats=_ts(market_regime=h4_regime, trend=h4_trend),
        h1_stats=_ts(market_regime=None, trend=h1_trend),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
        m5_stats=_ts(market_regime=h4_regime),  # the "short" regime is the M5 (decision) one since 2026-09-24
    )


def test_aligned_h1_h4_trend_direction_up_when_both_agree_uptrend():
    assert aligned_h1_h4_trend_direction("uptrend", "uptrend") == "up"


def test_aligned_h1_h4_trend_direction_down_when_both_agree_downtrend():
    assert aligned_h1_h4_trend_direction("downtrend", "downtrend") == "down"


def test_aligned_h1_h4_trend_direction_none_when_h1_h4_disagree():
    assert aligned_h1_h4_trend_direction("uptrend", "downtrend") is None


def test_aligned_h1_h4_trend_direction_none_when_flat():
    assert aligned_h1_h4_trend_direction("flat", "flat") is None


def test_aligned_h1_h4_trend_direction_none_when_either_side_missing():
    assert aligned_h1_h4_trend_direction(None, "uptrend") is None
    assert aligned_h1_h4_trend_direction("uptrend", None) is None


def test_tactical_regime_conflict_fires_on_the_real_btcusd_pattern():
    # Exact real incident reconstruction: H1 and H4 TREND both read
    # downtrend, but market_regime (choppy_up, backed by D1/Monthly
    # trending_up) reads STRUCTURALLY BACKED upward.
    analysis = _analysis_with_trends(
        h4_trend="downtrend", h1_trend="downtrend",
        h4_regime="choppy_up", d1_regime="trending_up",
    )
    warning = _tactical_trend_vs_regime_conflict(analysis, "up")
    assert warning is not None
    assert "CONTEXT/DECISION CONFLICT" in warning
    assert "DOWNward" in warning
    assert "BTCUSD" in warning  # cites the real incident by name

    text = format_long_term_alignment(analysis)
    assert "STRUCTURALLY BACKED" in text
    assert "CONTEXT/DECISION CONFLICT" in text


def test_tactical_regime_conflict_silent_when_trend_and_regime_agree():
    analysis = _analysis_with_trends(
        h4_trend="uptrend", h1_trend="uptrend",
        h4_regime="trending_up", d1_regime="trending_up",
    )
    assert _tactical_trend_vs_regime_conflict(analysis, "up") is None
    assert "CONTEXT/DECISION CONFLICT" not in format_long_term_alignment(analysis)


def test_tactical_regime_conflict_silent_when_h1_and_h4_trend_disagree_with_each_other():
    # No real, un-hedged tactical read to compare against the regime at
    # all -- H1 and H4 themselves don't even agree.
    analysis = _analysis_with_trends(
        h4_trend="uptrend", h1_trend="downtrend",
        h4_regime="choppy_up", d1_regime="trending_up",
    )
    assert _tactical_trend_vs_regime_conflict(analysis, "up") is None


def test_tactical_regime_conflict_silent_when_trend_is_flat():
    analysis = _analysis_with_trends(
        h4_trend="flat", h1_trend="flat",
        h4_regime="choppy_up", d1_regime="trending_up",
    )
    assert _tactical_trend_vs_regime_conflict(analysis, "up") is None


def test_tactical_regime_conflict_silent_when_regime_direction_is_flat():
    # Real bug caught on self-review before this ever shipped: wording a
    # "conflict" against a FLAT regime direction (H4's own regime shows
    # no real net direction) would be misleading -- there's nothing
    # genuinely opposite for the tactical read to conflict WITH.
    analysis = _analysis_with_trends(
        h4_trend="downtrend", h1_trend="downtrend",
        h4_regime="sideways", d1_regime="trending_up",
    )
    assert _tactical_trend_vs_regime_conflict(analysis, "flat") is None


def test_tactical_regime_conflict_silent_when_regime_direction_is_none():
    analysis = _analysis_with_trends(
        h4_trend="downtrend", h1_trend="downtrend",
        h4_regime=None, d1_regime="trending_up",
    )
    assert _tactical_trend_vs_regime_conflict(analysis, None) is None


def test_long_term_alignment_regime_wording_distinguishes_from_trend_line():
    # Real incident, 2026-09-11: the old "the H4 {dir}ward move" wording
    # read as flatly restating format_mtf_confluence's own H4-vs-H1
    # TREND line, when the two are genuinely different metrics.
    analysis = _analysis_with_regimes("trending_up", "trending_up")
    text = format_long_term_alignment(analysis)
    assert "M5 REGIME direction" in text


# --- format_long_term_alignment_short / classify_long_term_alignment ---
# Real UI feedback, live 2026-08-22: the long, reasoning-heavy AI-prompt
# text (format_long_term_alignment above) was showing up verbatim in the
# Asset Health popup — correct for a model, unreadable as a dashboard
# line. These two share one classification step (classify_long_term_
# alignment) with the long formatter so the two texts can never disagree
# about WHICH state applies, only how verbosely they describe it.


def test_long_term_alignment_short_is_one_short_sentence_per_state():
    cases = [
        ("trending_up", "trending_up", "structurally_backed"),
        ("choppy_up", "trending_down", "counter_trend_spike"),
        ("choppy_up", "sideways", "no_backdrop"),
        ("sideways", "trending_up", "flat"),
        (None, "trending_up", "not_available"),
    ]
    for h4, d1, expected_state in cases:
        analysis = _analysis_with_regimes(h4, d1)
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
    analysis = _analysis_with_regimes("trending_up", "trending_up")
    assert "STRUCTURALLY BACKED" in format_long_term_alignment(analysis)
    assert "likely real" in format_long_term_alignment_short(analysis)

    analysis = _analysis_with_regimes("choppy_up", "trending_down")
    assert "COUNTER-TREND SPIKE" in format_long_term_alignment(analysis)
    assert "short spike" in format_long_term_alignment_short(analysis)


# --- build_trend_radar ---
# Direct root-cause fix, 2026-09-05: a real mega session left EURUSD/
# GBPUSD excluded (99.1% cash overall) despite each reading a live H1
# trend_intact setup — real, computed evidence that was correctly there
# but never surfaced saliently enough among ~300 lines of per-instrument
# detail to be genuinely engaged with. These tests lock in that a
# genuine trend setup gets caught and labeled, and that a non-trending
# instrument is correctly left out rather than padding the list.


def _trend_radar_analysis(symbol, h1_regime, h1_trend, h4_regime=None, h4_trend=None) -> FtmoAssetAnalysis:
    """h4 defaults to mirroring h1 (the common case — a clean trend reads
    the same way on both timeframes); pass h4_regime/h4_trend explicitly
    to build a genuine H4-vs-H1 conflict/partial case instead. base.stats
    and mn1_stats are left at their real defaults (market_regime=None),
    so classify_long_term_alignment always resolves to "no_backdrop"
    here — this fixture is about the H1/H4 setup + alignment read, not
    the longer-term backdrop, which format_long_term_alignment's own
    tests above already cover directly. last_price=1.0 on both stats —
    classify_setups bails to [] immediately when last_price is None
    (_ts()'s own default), which would silently defeat every case here."""
    h4_regime = h1_regime if h4_regime is None else h4_regime
    h4_trend = h1_trend if h4_trend is None else h4_trend
    base = AssetAnalysis(symbol=symbol, description=symbol, bid=1.0, ask=1.0005, display_name=symbol)
    return FtmoAssetAnalysis(
        base=base,
        h4_stats=_ts(market_regime=h4_regime, trend=h4_trend, last_price=1.0),
        h1_stats=_ts(market_regime=h1_regime, trend=h1_trend, last_price=1.0),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
        # The radar reads the M5 (decision) setups; H4/H1 supply only the alignment label.
        m5_stats=_ts(market_regime=h1_regime, trend=h1_trend, last_price=1.0),
    )


def test_build_trend_radar_flags_a_clean_trend_with_alignment_labels():
    analysis = _trend_radar_analysis("EURUSD", "trending_up", "uptrend")
    radar = build_trend_radar([analysis])
    assert "EURUSD" in radar
    assert "trend_intact" in radar
    assert "H4/H1 ALIGNED" in radar
    assert "no long-term backdrop available" in radar


def test_build_trend_radar_flags_h4_h1_conflict():
    analysis = _trend_radar_analysis(
        "GBPUSD", h1_regime="trending_up", h1_trend="uptrend",
        h4_regime="trending_down", h4_trend="downtrend",
    )
    radar = build_trend_radar([analysis])
    assert "GBPUSD" in radar
    assert "H4/H1 CONFLICTING" in radar


def test_build_trend_radar_empty_string_when_nothing_qualifies():
    # sideways regime, no momentum-acceleration override -> no rule in
    # classify_setups fires (confirmed against analysis/setup_classifier.py
    # directly: in_progress_move ALSO needs sideways, but additionally
    # requires momentum_acceleration in ("accelerating_up",
    # "accelerating_down"), which _ts()'s own default (None) never is).
    analysis = _trend_radar_analysis("USDCHF", "sideways", "flat")
    assert build_trend_radar([analysis]) == ""


def test_build_trend_radar_omits_non_trending_instruments_and_asks_for_specific_reasons():
    trending = _trend_radar_analysis("EURUSD", "trending_up", "uptrend")
    flat = _trend_radar_analysis("USDCHF", "sideways", "flat")
    radar = build_trend_radar([trending, flat])
    assert "EURUSD" in radar
    assert "USDCHF" not in radar
    assert "SPECIFIC technical reason" in radar


@patch("ai.ftmo_suggest.build_macro_snapshot", return_value="MACRO")
@patch("ai.ftmo_suggest.format_book_wisdom", return_value="WISDOM")
def test_build_ftmo_summary_includes_trend_radar_when_the_position_hunt_is_off(mock_wisdom, mock_macro, monkeypatch):
    monkeypatch.setattr(config, "POSITION_HUNT_ENABLED", False)
    account = AccountSummary(balance=100_000.0, equity=100_000.0, free_margin=90_000.0, currency="USD")
    analyses = [_trend_radar_analysis("EURUSD", "trending_up", "uptrend")]
    summary = build_ftmo_summary(
        account,
        assets=[MarketAsset("EURUSD", "Euro", 1.1, 1.1005)],
        ftmo_status=_make_status(),
        analyses=analyses,
    )
    assert "Trend Radar" in summary
    assert "EURUSD: trend_intact" in summary


@patch("ai.ftmo_suggest.build_macro_snapshot", return_value="MACRO")
@patch("ai.ftmo_suggest.format_book_wisdom", return_value="WISDOM")
def test_build_ftmo_summary_omits_trend_radar_when_nothing_qualifies(mock_wisdom, mock_macro, monkeypatch):
    monkeypatch.setattr(config, "POSITION_HUNT_ENABLED", False)
    account = AccountSummary(balance=100_000.0, equity=100_000.0, free_margin=90_000.0, currency="USD")
    analyses = [_trend_radar_analysis("USDCHF", "sideways", "flat")]
    summary = build_ftmo_summary(
        account,
        assets=[MarketAsset("USDCHF", "Swissy", 0.81, 0.8101)],
        ftmo_status=_make_status(),
        analyses=analyses,
    )
    assert "Trend Radar" not in summary


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


def test_commission_equities_flagged_as_unconfirmed():
    # Direct user correction 2026-08-26: this account's equities
    # ("Equities I CFD" in its own Market Watch path) used to have no
    # rate at all here, which a real mega session then read as "genuine
    # cost-data gap" and used to exclude the entire category outright.
    pct, note = _ftmo_commission_pct_round_turn("Equities I CFD", contract_size=1.0, price=200.0)
    assert pct == pytest.approx(config.FTMO_COMMISSION_EQUITIES_PCT_ROUND_TURN)
    assert "NOT independently confirmed" in note


def test_commission_equities_startswith_covers_a_future_second_group():
    # Mirrors Crypto's own startswith behavior — this account already has
    # Crypto I/II and Cash II/III groups, so a future "Equities II CFD"
    # is plausible and should hit the same rate, not fall through to
    # "unknown".
    pct, _ = _ftmo_commission_pct_round_turn("Equities II CFD", contract_size=1.0, price=200.0)
    assert pct == pytest.approx(config.FTMO_COMMISSION_EQUITIES_PCT_ROUND_TURN)


def test_commission_unknown_category_returns_none_not_a_fabricated_zero():
    pct, note = _ftmo_commission_pct_round_turn("Bonds CFD", contract_size=1.0, price=200.0)
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
    cost = _make_trade_cost(category="Bonds CFD", spread_pct_of_price=0.01)
    kwargs = _real_backtest_execution_kwargs(cost, _make_spec(), 200.0)
    assert kwargs["round_trip_cost_pct"] == pytest.approx(0.01)


def test_real_backtest_execution_kwargs_adds_real_commission_for_equities():
    cost = _make_trade_cost(category="Equities I CFD", spread_pct_of_price=0.01)
    kwargs = _real_backtest_execution_kwargs(cost, _make_spec(), 200.0)
    commission_pct, _ = _ftmo_commission_pct_round_turn("Equities I CFD", 1.0, 200.0)
    assert kwargs["round_trip_cost_pct"] == pytest.approx(0.01 + commission_pct)


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


# --- _backtest_all_sr_levels (Phase 4 orchestrator) ---


def _sine_wave_ohlc(n_cycles: int = 40, period: int = 12) -> pd.DataFrame:
    """Same clean 95-105 oscillation as tests/test_backtest.py's own
    verified backtest_level_reliability fixture (kept as an independent
    copy, same convention _cyclical_m5_series above already follows for
    this file) — real support trough near 95.0, real resistance peak
    near 105.0, both empirically confirmed (this session's own
    verification script) to hold every single time within the default
    holding window."""
    prices = [100.0 + 5.0 * math.sin(2 * math.pi * k / period) for k in range(n_cycles * period)]
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    closes = pd.Series(prices, index=dates)
    pad = closes * 0.0005
    return pd.DataFrame(
        {
            "Open": closes.shift(1).fillna(closes.iloc[0]), "High": closes + pad, "Low": closes - pad,
            "Close": closes, "Volume": [100.0] * len(closes),
        },
        index=dates,
    )


def test_backtest_all_sr_levels_skips_thin_levels_and_keys_by_price():
    ohlc = _sine_wave_ohlc()
    sr = SRLevelsResult(
        resistance_levels=[
            SRLevel(price=105.0, touches=10, distance_pct=5.0, low=104.5, high=105.5, weighted_score=8.0)
        ],
        support_levels=[
            SRLevel(price=95.0, touches=10, distance_pct=-5.0, low=94.5, high=95.5, weighted_score=8.0),
            # Never touched by this fixture (the sine wave stays within
            # 95-105) -> backtest_level_reliability returns None for it,
            # and the orchestrator must skip it rather than crash or add
            # a fabricated entry.
            SRLevel(price=200.0, touches=1, distance_pct=95.0, low=199.5, high=200.5, weighted_score=0.1),
            # A pre-Phase-1-shaped SRLevel with no real band at all (low/
            # high still None) -> must also be skipped, not crash.
            SRLevel(price=90.0, touches=5, distance_pct=-10.0),
        ],
        tolerance_pct=0.5,
    )
    trade_cost = _make_trade_cost(swap_long_pct_per_day=-0.0075, swap_short_pct_per_day=0.0003)
    result = _backtest_all_sr_levels(ohlc, sr, trade_cost, _make_spec(), 1.10)
    assert set(result.keys()) == {95.0, 105.0}
    assert result[95.0].side == "support"
    assert result[95.0].swap_pct_per_day_used == pytest.approx(-0.0075)  # long swap, since support bets long
    assert result[105.0].side == "resistance"
    assert result[105.0].swap_pct_per_day_used == pytest.approx(0.0003)  # short swap, since resistance bets short


def test_backtest_all_sr_levels_empty_when_no_sr_levels_or_history():
    ohlc = _sine_wave_ohlc()
    assert _backtest_all_sr_levels(ohlc, None, None, None, None) == {}
    empty_sr = SRLevelsResult(resistance_levels=[], support_levels=[])
    assert _backtest_all_sr_levels(pd.DataFrame(), empty_sr, None, None, None) == {}


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
    cost = _make_trade_cost(category="Bonds CFD", spread_pct_of_price=0.25)
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


def test_format_ftmo_trade_cost_equities_no_longer_flagged_as_unknown():
    # Direct user correction 2026-08-26 — real bug this guards against:
    # a real mega session's own final answer excluded MSFT/NVDA/AMD/INTC
    # entirely, citing exactly this "unconfirmed commission... explicitly
    # a floor, not the real number" line as its stated reason. Confirms
    # the fix actually removes that reason from the line equities get.
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
    assert "unknown" not in text.lower()
    assert "floor" not in text.lower()
    assert f"{config.FTMO_COMMISSION_EQUITIES_PCT_ROUND_TURN:.4f}%" in text


# --- format_ftmo_min_viable_size, added 2026-09-11 after a real
# incident: several real positions were sized so small (against a small
# aggregate-heat budget split across too many names) that their real
# risk distance couldn't clear the broker's own minimum lot at all --
# "Risking 0.1% of equity against this stop distance can't afford even
# the minimum 0.01-lot for this instrument" -- discovered only AFTER the
# mega session had already committed to including them.


def _analysis_for_min_viable_size(volume_min=0.01, trade_contract_size=100.0, atr=None) -> FtmoAssetAnalysis:
    spec = ContractSpec(
        volume_min=volume_min, volume_step=0.01, volume_max=500.0,
        trade_contract_size=trade_contract_size, currency_margin="USD", margin_initial=1000.0,
    )
    base = AssetAnalysis(
        symbol="XAUUSD", description="Gold", bid=4370.0, ask=4370.5,
        display_name=None, contract_spec=spec,
    )
    return FtmoAssetAnalysis(
        base=base, h4_stats=_ts(), h1_stats=_ts(atr=atr),
        h4_structure=_empty_chart_structure(), h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )


def test_min_viable_size_computes_the_real_minimum_pct():
    # Hand-verified: stop_distance = 1.5 x 30.0 = 45.0; min_pct =
    # (0.01 lot x 45.0 x 100.0 contract_size) / 10,000 equity x 100 =
    # 45.0 / 10,000 x 100 = 0.45%.
    analysis = _analysis_for_min_viable_size(atr=30.0)
    text = format_ftmo_min_viable_size(analysis, account_equity=10_000.0)
    assert text is not None
    assert f"{MIN_VIABLE_SIZE_ATR_MULTIPLE:g}x H1 ATR" in text
    assert "45" in text  # the real stop distance in price units
    assert "0.45%" in text


def test_min_viable_size_scales_with_min_lot_and_contract_size():
    # Doubling volume_min must exactly double the required pct -- pure
    # linear relationship, a real regression guard on the formula itself.
    base_text = format_ftmo_min_viable_size(_analysis_for_min_viable_size(volume_min=0.01, atr=30.0), 10_000.0)
    doubled_text = format_ftmo_min_viable_size(_analysis_for_min_viable_size(volume_min=0.02, atr=30.0), 10_000.0)
    assert "0.45%" in base_text
    assert "0.90%" in doubled_text


def test_min_viable_size_none_without_contract_spec():
    analysis = _analysis_for_min_viable_size(atr=30.0)
    analysis.base.contract_spec = None
    assert format_ftmo_min_viable_size(analysis, account_equity=10_000.0) is None


def test_min_viable_size_none_without_atr():
    analysis = _analysis_for_min_viable_size(atr=None)
    assert format_ftmo_min_viable_size(analysis, account_equity=10_000.0) is None


def test_min_viable_size_none_without_account_equity():
    analysis = _analysis_for_min_viable_size(atr=30.0)
    assert format_ftmo_min_viable_size(analysis, account_equity=None) is None
    assert format_ftmo_min_viable_size(analysis, account_equity=0.0) is None


def test_min_viable_size_wired_into_asset_context():
    analysis = _analysis_for_min_viable_size(atr=30.0)
    text = format_ftmo_asset_context([analysis], account_equity=10_000.0)
    assert "minimum viable size" in text
    assert "0.45%" in text


def test_min_viable_size_absent_from_asset_context_when_unavailable():
    # Real bug class this guards against: a bare None accidentally joined
    # into the output as the literal string "None".
    analysis = _analysis_for_min_viable_size(atr=None)
    text = format_ftmo_asset_context([analysis], account_equity=10_000.0)
    assert "minimum viable size" not in text
    assert "\nNone\n" not in text
    assert not text.rstrip().endswith("None")


def test_format_ftmo_asset_context_can_exclude_favorable_excursion_for_clerk():
    # Real gap caught on a self-recheck (2026-09-12): ai/clerk_execution.py's
    # own _fetch_technical_context reuses this exact function's output as
    # `technical_context` for its tactical-verdict prompts, which never
    # include this file's own _INSTRUCTION_HEAD — the ONLY place the
    # favorable-excursion figure's critical misread warning and HOLDING
    # HORIZON scale-mismatch caveat actually live. Locks in that
    # include_favorable_excursion=False threads all the way from
    # format_ftmo_asset_context down through format_enriched_asset_context
    # to format_backtests, while leaving the underlying win-rate/avg-R
    # backtest line untouched.
    base = AssetAnalysis(
        symbol="XAUUSD", description="Gold", bid=4370.0, ask=4370.5,
        display_name="Gold", stats=_ts(last_price=4370.0),
        rsi_overbought_backtest=RSIReactionBacktest(
            "overbought", 70.0, 6, 5, 1, 0, 83.3, 1.5, 1.5, 3.0, 10,
            excursion=FavorableExcursionStats(sample_size=8, avg_r=4.53, median_r=3.81, horizon_bars=90),
        ),
    )
    analysis = FtmoAssetAnalysis(
        base=base, h4_stats=_ts(), h1_stats=_ts(),
        h4_structure=_empty_chart_structure(), h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )

    text = format_ftmo_asset_context([analysis], account_equity=10_000.0, include_favorable_excursion=False)
    assert "historical favorable-excursion magnitude" not in text
    assert "6 distinct past episodes" in text
    assert "win rate 83%" in text


def test_format_ftmo_asset_context_can_exclude_market_status_entirely():
    # Same real gap, same fix pattern as the favorable-excursion
    # exclusion test above: ai/clerk_execution.py's own tactical prompts
    # never include _INSTRUCTION_HEAD, the only place "market CLOSED" is
    # explained/actionable, and Clerk never proposes new trades at all --
    # confirms include_market_status=False threads all the way through.
    base = AssetAnalysis(
        symbol="EURUSD", description="Euro", bid=1.09, ask=1.0905,
        display_name="Euro", stats=_ts(last_price=1.09), market_open=False,
    )
    analysis = FtmoAssetAnalysis(
        base=base, h4_stats=_ts(), h1_stats=_ts(),
        h4_structure=_empty_chart_structure(), h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )

    text = format_ftmo_asset_context([analysis], account_equity=10_000.0, include_market_status=False)
    assert "market CLOSED" not in text


# --- format_ftmo_held_position_sizing_rates, added 2026-09-11 after a
# real incident: a held NVDA position's own revised stop was reported at
# pct=0.02%, a razor-thin 17% short of the 0.0242% that exact distance
# actually needed to reach even ONE whole share -- every clerk poll since
# then failed to "infeasible" for hours, leaving the real position stuck
# on its old, stretched stop/target the entire time.


def _position_and_analysis_for_sizing_rate(volume=1.0, trade_contract_size=1.0) -> tuple[Position, FtmoAssetAnalysis]:
    spec = ContractSpec(
        volume_min=1.0, volume_step=1.0, volume_max=1000.0,
        trade_contract_size=trade_contract_size, currency_margin="USD", margin_initial=21.79,
    )
    base = AssetAnalysis(
        symbol="NVDA", description="Nvidia", bid=218.30, ask=218.35,
        display_name=None, contract_spec=spec,
    )
    analysis = FtmoAssetAnalysis(
        base=base, h4_stats=_ts(), h1_stats=_ts(),
        h4_structure=_empty_chart_structure(), h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    position = Position(
        symbol="NVDA", volume=volume, side="buy", price_open=217.85, price_current=218.30,
        sl=216.92, profit=0.45, opened_at=None, ticket=99, tp=232.0,
    )
    return position, analysis


def test_held_position_sizing_rate_matches_the_real_nvda_incident():
    # Hand-verified against the real incident: rate = (1.0 lot x 1.0
    # contract_size) / 9933.43 equity x 100 = 0.010067...%. At the real
    # 2.40 stop distance Claude actually chose, that's 0.024161% -- the
    # true floor, versus the 0.02% Claude actually reported (the exact
    # razor-thin miss that blocked every poll for hours).
    position, analysis = _position_and_analysis_for_sizing_rate()
    text = format_ftmo_held_position_sizing_rates([position], [analysis], account_equity=9933.43)
    assert text != ""
    assert "NVDA sizing rate" in text
    assert "0.010067%" in text
    # A distance of 2.40 (the real one Claude actually used) x this rate
    # = 0.024161% -- the real minimum Claude's own reported 0.02% missed.
    # Not asserted against the text directly (the function only prints
    # 1.00/5.00 worked examples), verified here as a standalone sanity
    # check that the rate itself is right.
    rate = (1.0 * 1.0) / 9933.43 * 100
    assert rate * 2.40 == pytest.approx(0.024161, abs=0.000001)
    assert f"{rate * 5:.4f}%" in text  # the "distance of 5.00" worked example


def test_held_position_sizing_rate_scales_with_volume_and_contract_size():
    position, analysis = _position_and_analysis_for_sizing_rate(volume=2.0, trade_contract_size=100.0)
    text = format_ftmo_held_position_sizing_rates([position], [analysis], account_equity=10_000.0)
    # rate = (2.0 x 100.0) / 10,000 x 100 = 2.0% per unit of stop distance.
    assert "2.000000%" in text


def test_held_position_sizing_rate_empty_without_positions():
    _, analysis = _position_and_analysis_for_sizing_rate()
    assert format_ftmo_held_position_sizing_rates([], [analysis], account_equity=10_000.0) == ""


def test_held_position_sizing_rate_empty_without_equity():
    position, analysis = _position_and_analysis_for_sizing_rate()
    assert format_ftmo_held_position_sizing_rates([position], [analysis], account_equity=None) == ""
    assert format_ftmo_held_position_sizing_rates([position], [analysis], account_equity=0.0) == ""


def test_held_position_sizing_rate_skips_a_position_with_no_contract_spec():
    position, analysis = _position_and_analysis_for_sizing_rate()
    analysis.base.contract_spec = None
    assert format_ftmo_held_position_sizing_rates([position], [analysis], account_equity=10_000.0) == ""


def test_held_position_sizing_rate_skips_a_position_with_no_matching_analysis():
    position, _ = _position_and_analysis_for_sizing_rate()
    unrelated_analysis = _analysis_for_min_viable_size(atr=30.0)  # symbol XAUUSD, not NVDA
    assert format_ftmo_held_position_sizing_rates([position], [unrelated_analysis], account_equity=10_000.0) == ""


def test_held_position_sizing_rate_wired_into_build_ftmo_summary():
    position, analysis = _position_and_analysis_for_sizing_rate()
    asset = MarketAsset(symbol="NVDA", description="Nvidia", bid=218.30, ask=218.35)
    account = AccountSummary(balance=9933.0, equity=9933.43, free_margin=9900.0, currency="USD")
    status = _make_status()
    text = build_ftmo_summary(account, [asset], status, positions=[position], analyses=[analysis])
    assert "Held-position sizing rates" in text
    assert "NVDA sizing rate" in text


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


# --- compute_ftmo_correlation_pairs / format_ftmo_correlation_context ---


def test_compute_ftmo_correlation_pairs_detects_perfectly_correlated_pair():
    a1 = _make_ftmo_correlated_analysis("AAA", _CORR_TEST_RETURNS, sign=1.0)
    a2 = _make_ftmo_correlated_analysis("BBB", _CORR_TEST_RETURNS, sign=1.0)
    pairs = compute_ftmo_correlation_pairs([a1, a2])
    assert len(pairs) == 1
    sym_a, sym_b, corr = pairs[0]
    assert {sym_a, sym_b} == {"AAA", "BBB"}
    assert corr == pytest.approx(1.0, abs=1e-6)


def test_compute_ftmo_correlation_pairs_detects_negative_correlation():
    a1 = _make_ftmo_correlated_analysis("AAA", _CORR_TEST_RETURNS, sign=1.0)
    a2 = _make_ftmo_correlated_analysis("BBB", _CORR_TEST_RETURNS, sign=-1.0)
    pairs = compute_ftmo_correlation_pairs([a1, a2])
    assert len(pairs) == 1
    _, _, corr = pairs[0]
    assert corr == pytest.approx(-1.0, abs=1e-6)


def test_compute_ftmo_correlation_pairs_excludes_series_below_min_observations():
    short_returns = _CORR_TEST_RETURNS[:5]
    a1 = _make_ftmo_correlated_analysis("AAA", short_returns, sign=1.0)
    a2 = _make_ftmo_correlated_analysis("BBB", short_returns, sign=1.0)
    assert compute_ftmo_correlation_pairs([a1, a2]) == []


def test_compute_ftmo_correlation_pairs_sorts_by_magnitude_and_caps_at_15():
    # 6 symbols -> 15 possible pairs; make them all qualify (>=0.7) with
    # distinct magnitudes so sort-order and the 15-cap are both provable.
    analyses = []
    for i in range(6):
        # A tiny per-symbol tweak keeps magnitudes distinct without
        # dropping any pair below the 0.7 threshold.
        returns = [r * (1.0 - i * 0.02) for r in _CORR_TEST_RETURNS]
        analyses.append(_make_ftmo_correlated_analysis(f"SYM{i}", returns, sign=1.0))
    pairs = compute_ftmo_correlation_pairs(analyses)
    assert len(pairs) == 15  # all 6-choose-2 pairs qualify, capped at 15
    magnitudes = [abs(c) for _, _, c in pairs]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_format_ftmo_correlation_context_labels_direction_correctly():
    a1 = _make_ftmo_correlated_analysis("AAA", _CORR_TEST_RETURNS, sign=1.0)
    a2 = _make_ftmo_correlated_analysis("BBB", _CORR_TEST_RETURNS, sign=-1.0)
    text = format_ftmo_correlation_context([a1, a2])
    assert "AAA & BBB" in text
    assert "move opposite each other" in text


def test_format_ftmo_correlation_context_no_pairs_message_when_none_qualify():
    solo = _make_ftmo_correlated_analysis("SOLO", _CORR_TEST_RETURNS, sign=1.0)
    assert "no pair currently has" in format_ftmo_correlation_context([solo])


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


@patch("ai.ftmo_suggest.build_macro_snapshot", return_value="MACRO")
@patch("ai.ftmo_suggest.format_book_wisdom", return_value="WISDOM")
def test_build_ftmo_summary_includes_correlation_context_when_analyses_supplied(mock_wisdom, mock_macro):
    account = AccountSummary(balance=100_000.0, equity=100_000.0, free_margin=90_000.0, currency="USD")
    status = _make_status()
    # Real overlapping price series (not the empty pd.Series(dtype=float)
    # the other supplied-analyses test uses) so the "N pairs found" branch
    # is genuinely exercised, not just the "no pair currently has" one.
    analyses = [
        _make_ftmo_correlated_analysis("AAA", _CORR_TEST_RETURNS, sign=1.0),
        _make_ftmo_correlated_analysis("BBB", _CORR_TEST_RETURNS, sign=1.0),
    ]
    with patch("ai.ftmo_suggest.analyze_ftmo_assets") as mock_analyze:
        summary = build_ftmo_summary(
            account,
            assets=[MarketAsset("AAA", "A", 1.1, 1.1005), MarketAsset("BBB", "B", 1.1, 1.1005)],
            ftmo_status=status,
            analyses=analyses,
        )
    mock_analyze.assert_not_called()
    assert "Pairwise correlation among the Market Watch instruments analyzed above" in summary
    assert "AAA & BBB" in summary


@patch("ai.ftmo_suggest.build_macro_snapshot", return_value="MACRO")
@patch("ai.ftmo_suggest.format_book_wisdom", return_value="WISDOM")
def test_build_ftmo_summary_includes_pending_orders_section_when_given(mock_wisdom, mock_macro):
    account = AccountSummary(balance=100_000.0, equity=100_000.0, free_margin=90_000.0, currency="USD")
    status = _make_status()
    orders = [
        PendingOrder(
            symbol="EURUSD", volume=1.0, order_type="buy limit", price_open=1.09,
            sl=1.08, tp=1.11, ticket=777, time_setup=None,
        )
    ]
    summary = build_ftmo_summary(
        account, assets=[], ftmo_status=status, positions=[], analyses=[], pending_orders=orders,
    )
    assert "Outstanding Pending Orders" in summary
    assert "EURUSD" in summary


@patch("ai.ftmo_suggest.build_macro_snapshot", return_value="MACRO")
@patch("ai.ftmo_suggest.format_book_wisdom", return_value="WISDOM")
def test_build_ftmo_summary_omits_pending_orders_section_when_none(mock_wisdom, mock_macro):
    account = AccountSummary(balance=100_000.0, equity=100_000.0, free_margin=90_000.0, currency="USD")
    status = _make_status()
    summary = build_ftmo_summary(account, assets=[], ftmo_status=status, positions=[], analyses=[])
    assert "Outstanding Pending Orders" not in summary


# --- instruction text content ---


def test_instruction_head_mentions_outstanding_pending_orders():
    from ai.ftmo_suggest import _INSTRUCTION_HEAD

    assert "Outstanding Pending Orders" in _INSTRUCTION_HEAD


def test_instruction_tail_requires_reason_and_invalidation_condition_fields():
    from ai.ftmo_suggest import _INSTRUCTION_TAIL

    assert '"reason"' in _INSTRUCTION_TAIL
    assert '"invalidation_condition"' in _INSTRUCTION_TAIL


def test_ftmo_stage1_instruction_states_the_real_rule_numbers():
    text = build_ftmo_stage1_instruction()
    assert "3%" in text
    assert "10%" in text
    assert "50%" in text
    assert "TRAILING" in text
    assert "static" in text
    assert "ZERO grace period" in text
    assert "INSTANT termination" in text


def test_ftmo_stage1_instruction_invites_a_well_evidenced_small_win():
    # 2026-09-18, direct user request: the reward:risk paragraph already
    # told the model about the automatic-rejection side of the new R:R
    # floor guard — this confirms the acceptance side (a genuinely
    # strong, well-sampled M15 win rate can justify a modest target below
    # the usual 2:1 aim) is also stated, not just the rejection warning.
    text = build_ftmo_stage1_instruction()
    assert "the same backstop works in your favor too" in text
    assert "modest-target, high-probability trade" in text


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
    for tf in ("M5", "H1", "H4", "D1"):
        assert tf in text
    assert "M15" not in text  # dropped 2026-09-24 (M5 + H1 split its duties)
    # Intraday decision-tier upgrade (2026-09-24): M5 alone decides, D1/H4/H1 are context.
    assert "DECISION TIER" in text and "CONTEXT TIER" in text
    assert "PRIMARY (and only) basis for the actual" in " ".join(text.split())
    assert "NEVER as the source of an entry, stop, target or size" in " ".join(text.split())
    assert "H1 supplies the STRUCTURE" not in text


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
    # The higher timeframes are demoted to regime/permission context, never the source of prices.
    assert "regime and the permission" in text


def test_ftmo_stage1_instruction_requires_per_instrument_holding_period_debate():
    text = build_ftmo_stage1_instruction()
    assert "DEBATE THE HOLDING PERIOD" in text
    # The two real inputs the debate must actually use.
    assert "swap" in text.lower()
    assert "round-trip cost" in text.lower()


def test_ftmo_stage1_instruction_requires_per_instrument_position_size_debate():
    text = build_ftmo_stage1_instruction()
    assert "DEBATE THE POSITION SIZE" in text


def test_ftmo_stage1_instruction_checks_current_market_open_status():
    text = build_ftmo_stage1_instruction()
    assert "CHECK WHETHER THIS INSTRUMENT'S MARKET IS EVEN OPEN RIGHT NOW" in text
    assert "market CLOSED (weekend)" in text
    assert "reopen Sunday evening UTC" in text
    assert "Do NOT propose a NEW immediate allocation or a NEW pending setup on an instrument marked CLOSED" in text


def test_ftmo_stage1_instruction_warns_against_doomed_weekend_pending_setups():
    text = build_ftmo_stage1_instruction()
    assert "HEADING INTO A WEEKEND" in text
    # Crypto is explicitly exempted -- it genuinely trades through the
    # weekend, unlike every other instrument on this account.
    assert "BTCUSD/ETHUSD" in text
    assert "does NOT apply to crypto" in text
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
    # Direct user instruction 2026-09-05: don't artificially cap a
    # category's own instrument count -- a prior version of this prompt
    # explicitly told Claude to "aim for roughly 1-2 instruments per
    # represented category," which was concentrating the mix into 1-2
    # positions even when several genuinely trending, independently-
    # supported instruments sat excluded. Assert the new policy is
    # present and the old numeric cap is gone.
    assert "1-2 instruments" not in text
    assert "many small" in text.lower()
    assert "position-count" in text.lower()


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


def test_ftmo_stage2_synth_instruction_requires_reconciling_data_backed_objections():
    # A backtest win-rate or a live trend-classification disagreement is a
    # checkable fact, not a subjective judgment call an audit's model-size
    # weighting is allowed to discount into silence (see the XAUUSD
    # incident this instruction quotes: two audits caught a mismatched H1
    # trend read and a 14-17% historical win rate, and the final revision
    # kept the same thesis anyway).
    text = build_ftmo_stage2_instruction(audit_available=True)
    assert "NOT OPTIONAL TO WEIGH AWAY" in text
    assert "backtest" in text.lower()
    assert "is required, not merely internal process" in text


def test_ftmo_stage1_instruction_warns_against_chasing_ratio_over_missing_the_trend():
    # Real incident: an NVDA pending buy limit sat unfilled for days as
    # price ran 11%+ past it and past its own take-profit, each session
    # just re-asserting "not yet invalidated" instead of re-examining
    # anything -- direct user request not to hard-code a mechanical
    # override, but to require honest disclosure and let the model judge.
    text = build_ftmo_stage1_instruction()
    assert "DON'T LET CHASING A BETTER RATIO BECOME AN EXCUSE TO MISS THE" in text
    assert "how long it's been resting" in text
    assert "ORIGINAL TARGET" in text


def test_ftmo_stage1_instruction_requires_equal_research_depth_across_categories():
    text = build_ftmo_stage1_instruction()
    assert "EQUAL research depth across" in text
    assert "forex ending up" in text.lower() or "forex ending up" in text


def test_ftmo_audit_instruction_covers_stale_carried_forward_levels():
    assert "STALE CARRIED-FORWARD LEVELS" in AUDIT_INSTRUCTION
    assert "ORIGINAL TARGET" in AUDIT_INSTRUCTION


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


def test_ftmo_audit_instruction_covers_invalidation_condition_scrutiny():
    # Real gap found on self-review 2026-08-24: Pending Setups' own
    # trigger_condition already gets this exact scrutiny (see the test
    # above) but invalidation_condition -- the sibling field on the main
    # allocation block, central to Copilot's own watch mechanism -- had
    # no audit-pool check at all, meaning nothing would catch a lazy or
    # unfalsifiable one before it shipped.
    assert "invalidation_condition" in AUDIT_INSTRUCTION


def test_ftmo_audit_instruction_requires_recomputing_pending_setups_arithmetic():
    # Real bug found in a live report (2026-08-22): prose claimed one
    # R:R/pip count while the entry's own price/stop_loss/take_profit
    # numbers implied a different one. Auditors must actually recompute
    # from the JSON's own numbers, not just eyeball plausibility.
    assert "RECOMPUTE the stop distance" in AUDIT_INSTRUCTION
    assert "gross R:R" in AUDIT_INSTRUCTION


def test_ftmo_audit_instruction_flags_new_proposals_on_closed_markets():
    assert "market is CLOSED right now" in AUDIT_INSTRUCTION
    assert "it cannot fill" in AUDIT_INSTRUCTION
    # Must not penalize managing an existing position on a now-closed
    # instrument -- that's a separate, legitimate decision.
    assert "EXISTING position on a currently-closed instrument" in AUDIT_INSTRUCTION


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


def test_write_latest_suggestion_persists_reason_and_invalidation_condition():
    final_answer = (
        "## Executive Summary\nSome prose.\n\n"
        '```json\n{"EURUSD": {"pct": 1.5, "price": 1.09, "stop_loss": 1.08, '
        '"take_profit": 1.11, "side": "buy", "reason": "Pullback into support.", '
        '"invalidation_condition": "H4 closes below 1.07"}, "CASH": 98.5}\n```'
    )
    _write_latest_suggestion(final_answer)
    saved = read_latest_suggestion()
    assert saved["immediate_allocation"]["EURUSD"]["reason"] == "Pullback into support."
    assert saved["immediate_allocation"]["EURUSD"]["invalidation_condition"] == "H4 closes below 1.07"


def test_write_latest_suggestion_backward_compatible_when_new_fields_absent():
    # _WELL_FORMED_FINAL_ANSWER's own EURUSD entry has neither field —
    # must still parse and persist cleanly (as "" / None), not fail the
    # whole write, so an older-style response can never wipe out a prior
    # valid suggestion just because it predates this feature.
    _write_latest_suggestion(_WELL_FORMED_FINAL_ANSWER)
    saved = read_latest_suggestion()
    assert saved["immediate_allocation"]["EURUSD"]["reason"] == ""
    assert saved["immediate_allocation"]["EURUSD"]["invalidation_condition"] is None


def test_write_latest_suggestion_allows_a_pct_zero_symbol_to_also_be_a_pending_setup():
    # Real bug found live 2026-08-25: Claude cancelled a stale pending
    # order via "pct": 0 for a symbol AND separately re-listed that same
    # symbol in Pending Setups as a fresh watch idea ("cancel the old one,
    # replace with a cleaner trigger") — a real, unambiguous, valid
    # instruction. The pre-existing duplicate-symbol check treated ANY
    # symbol appearing in the allocation block (even at pct: 0) as
    # off-limits for Pending Setups, so the whole Pending Setups block
    # failed to parse and the ENTIRE day's suggestion — including every
    # other symbol's real reason/invalidation_condition — got silently
    # discarded, leaving Copilot working off a stale, days-old file.
    final_answer = (
        "## Executive Summary\nSome prose.\n\n"
        '```json\n{"BTCUSD": {"side": "buy", "pct": 0, "price": 78555.0, '
        '"reason": "Cancelling the stale pending order."}, "CASH": 100}\n```\n\n'
        '```json\n[{"symbol": "BTCUSD", "side": "buy", "pct": 0.5, '
        '"trigger_condition": "H1 closes back above 77466.31", '
        '"price": 77500.0, "stop_loss": 76419.0, "take_profit": 79410.0, '
        '"reason": "Fresh, cleaner watch setup replacing the cancelled order."}]\n```'
    )
    _write_latest_suggestion(final_answer)
    saved = read_latest_suggestion()
    assert saved["immediate_allocation"]["BTCUSD"]["pct"] == 0
    assert saved["pending_setups"][0]["symbol"] == "BTCUSD"


def test_write_latest_suggestion_still_rejects_a_nonzero_pct_symbol_duplicated_in_pending_setups():
    # The exclusion rule must still fire for a GENUINE conflict — a
    # symbol with REAL, nonzero exposure in the allocation block cannot
    # also appear in Pending Setups, since that's still a real ambiguous
    # double-instruction, not the pct: 0 cancel-and-replace case above.
    _write_latest_suggestion(_WELL_FORMED_FINAL_ANSWER)
    prior = read_latest_suggestion()

    final_answer = (
        "## Executive Summary\nSome prose.\n\n"
        '```json\n{"EURUSD": {"side": "buy", "pct": 1.5, "price": 1.09, '
        '"stop_loss": 1.08}, "CASH": 98.5}\n```\n\n'
        '```json\n[{"symbol": "EURUSD", "side": "buy", "pct": 0.5, '
        '"trigger_condition": "H1 closes above 1.10", "reason": "Duplicate, invalid."}]\n```'
    )
    _write_latest_suggestion(final_answer)

    assert read_latest_suggestion() == prior


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


@patch("ai.ftmo_suggest.build_curiosity_report", return_value="CURIOSITY REPORT TEXT")
@patch("ai.ftmo_suggest.build_audit_block", return_value=AuditResult(block="", audit_available=False))
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_includes_curiosity_report_in_stage1_draft_prompt(
    mock_run_claude, mock_audit, mock_curiosity
):
    # Direct user request 2026-09-05: the curiosity function's own report
    # card must be compulsory (always attempted, folded into the same
    # past_lessons text the draft prompt already includes), not an
    # opt-in extra.
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_ftmo_portfolio("some ftmo summary")
    draft_prompt = mock_run_claude.call_args_list[0].args[0]
    assert "CURIOSITY REPORT TEXT" in draft_prompt


@patch("ai.ftmo_suggest.build_curiosity_report", return_value="")
@patch("ai.ftmo_suggest.build_past_lessons", return_value="PAST LESSONS TEXT")
@patch("ai.ftmo_suggest.build_audit_block", return_value=AuditResult(block="", audit_available=False))
@patch("ai.ftmo_suggest.run_claude")
def test_suggest_ftmo_portfolio_empty_curiosity_report_leaves_past_lessons_untouched(
    mock_run_claude, mock_audit, mock_lessons, mock_curiosity
):
    # build_curiosity_report degrading to "" (no qualifying trades this
    # week, e.g.) must never blank out or otherwise disturb whatever
    # build_past_lessons already contributed.
    mock_run_claude.side_effect = ["draft", "final"]
    suggest_ftmo_portfolio("some ftmo summary")
    draft_prompt = mock_run_claude.call_args_list[0].args[0]
    assert "PAST LESSONS TEXT" in draft_prompt


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


# --- IntradayBacktests / _compute_intraday_backtests / _format_intraday_backtests ---


def _cyclical_m5_series(cycles: int = 20, up_bars: int = 20, down_bars: int = 10) -> pd.DataFrame:
    """M5-cadence OHLC fixture with repeating rally/pullback cycles, so
    RSI reliably crosses overbought/oversold — mirrors tests/test_backtest
    .py's own _cyclical_series/_make_ohlc helpers (kept as an independent
    copy, same convention _make_intraday_history above already follows
    for this file), just on a real 5-minute-spaced DatetimeIndex instead
    of daily, to genuinely represent what _compute_intraday_backtests
    actually feeds the backtest functions."""
    prices = [100.0]
    for _ in range(cycles):
        for _ in range(up_bars):
            prices.append(prices[-1] * 1.004)
        for _ in range(down_bars):
            prices.append(prices[-1] * 0.992)
    dates = pd.date_range("2026-01-01", periods=len(prices), freq="5min")
    closes = pd.Series(prices, index=dates)
    pad = closes * 0.0005
    return pd.DataFrame(
        {
            "Open": closes, "High": closes + pad, "Low": closes - pad,
            "Close": closes, "Volume": [100.0] * len(closes),
        },
        index=dates,
    )


@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_compute_intraday_backtests_disabled_returns_empty(mock_fetch, monkeypatch):
    monkeypatch.setattr(config, "INTRADAY_BACKTEST_ENABLED", False)
    result = _compute_intraday_backtests("EURUSD", None, 1.1005, _make_trade_cost())
    assert result == IntradayBacktests()
    mock_fetch.assert_not_called()


@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_compute_intraday_backtests_no_trade_cost_returns_empty(mock_fetch):
    result = _compute_intraday_backtests("EURUSD", None, 1.1005, None)
    assert result == IntradayBacktests()
    mock_fetch.assert_not_called()


@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_compute_intraday_backtests_empty_m5_history_returns_empty(mock_fetch):
    mock_fetch.return_value = _empty_history()
    result = _compute_intraday_backtests("EURUSD", None, 1.1005, _make_trade_cost())
    assert result == IntradayBacktests()


@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_compute_intraday_backtests_populates_from_real_cyclical_m5_data(mock_fetch):
    mock_fetch.return_value = _cyclical_m5_series()
    result = _compute_intraday_backtests("EURUSD", None, 1.1005, _make_trade_cost())
    assert result.rsi_overbought_backtest is not None
    assert result.rsi_oversold_backtest is not None
    assert result.rsi_overbought_backtest.max_holding_bars == config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS
    assert result.rsi_overbought_backtest.excursion.horizon_bars == config.TRADE_SIM_INTRADAY_EXCURSION_HORIZON_BARS
    # Swap zeroed even though _make_trade_cost() has real nonzero swap.
    assert result.rsi_overbought_backtest.swap_pct_per_day_used == 0.0
    mock_fetch.assert_called_once_with("EURUSD", "M5", count=config.M5_BACKTEST_BARS)


def _make_ftmo_analysis_with_intraday(intraday: IntradayBacktests) -> FtmoAssetAnalysis:
    return FtmoAssetAnalysis(
        base=_make_base_analysis(),
        h4_stats=_ts(),
        h1_stats=_ts(),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
        intraday_backtests=intraday,
    )


def test_format_intraday_backtests_states_real_hour_equivalent():
    analysis = _make_ftmo_analysis_with_intraday(IntradayBacktests())
    text = _format_intraday_backtests(analysis)
    expected_hours = config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS * 5 / 60
    assert f"~{expected_hours:.1f}h" in text
    assert "Same-session (M5)" in text


def test_format_intraday_backtests_all_none_states_not_enough_data():
    analysis = _make_ftmo_analysis_with_intraday(IntradayBacktests())
    text = _format_intraday_backtests(analysis)
    assert "not enough real M5 history/episodes" in text
    assert "not enough real M5 tests" in text


def test_format_intraday_backtests_invites_a_well_evidenced_small_win():
    # 2026-09-18, direct user request: right where the real win-rate
    # numbers appear (the point of decision), not just in the distant
    # general instruction paragraph.
    analysis = _make_ftmo_analysis_with_intraday(IntradayBacktests())
    text = _format_intraday_backtests(analysis)
    assert "well-sampled win rate below can justify a smaller" in text


def test_format_intraday_backtests_renders_real_rsi_and_sr_numbers():
    rsi_bt = RSIReactionBacktest(
        condition="oversold", threshold=30.0, trades=100, wins=40, losses=50, timeouts=10,
        win_rate_pct=44.4, avg_r_multiple=0.12, stop_atr_multiple=1.5, target_atr_multiple=3.0,
        max_holding_bars=16, min_stop_distance_pct=0.0, round_trip_cost_pct=0.005,
        swap_pct_per_day_used=0.0, excursion=None,
    )
    sr_bt = SupportResistanceBacktest(
        support_tests=9, support_wins=4, support_losses=4, support_timeouts=1, support_win_rate_pct=44.4,
        support_avg_r_multiple=0.1, resistance_tests=18, resistance_wins=2, resistance_losses=15,
        resistance_timeouts=1, resistance_win_rate_pct=11.1, resistance_avg_r_multiple=-0.2,
        stop_atr_multiple=1.5, target_atr_multiple=3.0, max_holding_bars=16, min_stop_distance_pct=0.0,
        round_trip_cost_pct=0.005, support_swap_pct_per_day_used=0.0, resistance_swap_pct_per_day_used=0.0,
        support_excursion=None, resistance_excursion=None,
    )
    analysis = _make_ftmo_analysis_with_intraday(
        IntradayBacktests(rsi_oversold_backtest=rsi_bt, support_resistance_backtest=sr_bt)
    )
    text = _format_intraday_backtests(analysis)
    assert "100 real M5 episodes" in text
    assert "9 real M5 tests" in text
    assert "18 real M5 tests" in text


@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec", return_value=None)
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_analyze_ftmo_asset_live_carries_the_m5_decision_tier_reads(
    mock_fetch_history, mock_contract_spec, mock_trade_economics
):
    import config as _config

    mock_fetch_history.return_value = _make_intraday_history()
    mock_trade_economics.return_value = _make_trade_cost()

    result = analyze_ftmo_asset_live("EURUSD", bid=1.10, ask=1.1005, description="Euro")

    assert result.m5_stats.last_price is not None
    assert result.m5_structure is not None
    assert not hasattr(result, "m15_stats") and not hasattr(result, "mn1_stats")
    timeframes_fetched = [c.args[1] for c in mock_fetch_history.call_args_list]
    assert timeframes_fetched.count("M5") == 1  # ONE full-depth fetch feeds the M5 reads, backtests and level reliability
    assert "M15" not in timeframes_fetched and "MN1" not in timeframes_fetched
    m5_call = [c for c in mock_fetch_history.call_args_list if c.args[1] == "M5"][0]
    assert m5_call.kwargs.get("count") == max(_config.INTRADAY_M5_BARS, _config.M5_BACKTEST_BARS)


def test_compute_intraday_backtests_proximity_is_atr_relative_and_clamped():
    from unittest.mock import patch as _patch

    history = _cyclical_m5_series(cycles=40)
    with (
        _patch("ai.ftmo_suggest.fetch_mt5_price_history", return_value=history),
        _patch("ai.ftmo_suggest.backtest_rsi_reaction", return_value=(None, None)),
        _patch("ai.ftmo_suggest.backtest_support_resistance_reaction", return_value=None) as sr,
    ):
        _compute_intraday_backtests("EURUSD", None, 1.1005, _make_trade_cost())
    from ai.ftmo_suggest import median_atr_pct

    expected = min(config.M5_SR_PROXIMITY_MAX_PCT, max(config.M5_SR_PROXIMITY_MIN_PCT, config.M5_SR_PROXIMITY_ATR_MULTIPLE * median_atr_pct(history)))
    assert sr.call_args.kwargs["proximity_pct"] == pytest.approx(expected)
    assert config.M5_SR_PROXIMITY_MIN_PCT <= sr.call_args.kwargs["proximity_pct"] <= config.M5_SR_PROXIMITY_MAX_PCT


# --- Decision-tier (H1/M5) formatters, intraday decision-tier upgrade 2026-09-24 ---


def _decision_tier_analysis(**overrides) -> FtmoAssetAnalysis:
    spec = ContractSpec(
        volume_min=0.01, volume_step=0.01, volume_max=500.0,
        trade_contract_size=100.0, currency_margin="USD", margin_initial=1000.0,
    )
    base = AssetAnalysis(symbol="XAUUSD", description="Gold", bid=4370.0, ask=4370.5, display_name=None, contract_spec=spec)
    fields = dict(
        base=base,
        h4_stats=_ts(trend="uptrend", last_price=4370.0, atr=30.0),
        h1_stats=_ts(trend="uptrend", last_price=4370.0, atr=10.0),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
        m5_stats=_ts(trend="uptrend", last_price=4370.0, atr=4.0),
    )
    fields.update(overrides)
    return FtmoAssetAnalysis(**fields)


def test_asset_context_labels_h1_m5_primary_and_d1_h4_context():
    from ai.ftmo_suggest import format_ftmo_asset_context

    analysis = _decision_tier_analysis()
    analysis.m5_stats.atr_pct = 0.1  # the ATR line only renders with a real atr_pct
    analysis.h1_stats.atr_pct = 0.3
    text = format_ftmo_asset_context([analysis], account_equity=10_000.0)
    assert "DECISION TIER for XAUUSD — M5" in text and "the ONLY basis for entry, stop, target" in text
    assert "CONTEXT TIER for XAUUSD — D1 / H4 / H1" in text and "never source an entry, stop, target or size" in text
    assert text.index("DECISION TIER") < text.index("M5 technical") < text.index("CONTEXT TIER")
    assert text.index("CONTEXT TIER") < text.index("H4 technical") < text.index("H1 technical")
    assert "M5 trade-zone candidate" in text and "H1 trade-zone candidate" not in text
    assert "M15" not in text and "Monthly" not in text
    # M5 is the decision read: its ATR sizes the stop and checks reachability; H1's is context only.
    assert "size the stop from this ATR" in text
    assert "H1 technical (regime/trend CONTEXT only" in text


def test_context_stats_lines_are_labeled_context_and_the_m5_line_is_the_decision_read():
    from ai.ftmo_suggest import format_timeframe_stats as fts

    stats = TechnicalStats(**{**{f: None for f in TechnicalStats.__dataclass_fields__}, "last_price": 100.0, "atr_pct": 0.3})
    line = fts("X", "H4", stats, context=True)
    assert "CONTEXT only" in line and "NOT the basis for entry/stop/target" in line
    assert "NOT a stop distance" in line and "size stops from M5 ATR" in line
    decision = fts("X", "M5", stats, decision=True)
    assert "DECISION tier" in decision and "size the stop from this ATR" in decision


def test_min_viable_size_uses_m5_atr_when_available_else_h1():
    text = format_ftmo_min_viable_size(_decision_tier_analysis(), account_equity=10_000.0)
    assert "2x M5 ATR" in text and "= 8 price units" in text  # 2.0 * 4.0
    fallback = format_ftmo_min_viable_size(_decision_tier_analysis(m5_stats=_ts()), account_equity=10_000.0)
    assert "1.5x H1 ATR" in fallback and "= 15 price units" in fallback  # 1.5 * 10.0


def test_intraday_sizing_sheet_reports_lots_margin_and_min_lot_shortfall():
    from ai.ftmo_suggest import format_intraday_sizing_sheet

    sheet = format_intraday_sizing_sheet(_decision_tier_analysis(), 10_000.0)
    # stop 8.0 units; 0.5% of 10k = $50 -> 50/(8*100)=0.0625 -> 0.06 lots, margin 0.06*1000/10000 = 0.6%
    assert "0.5% risk -> 0.06 lots (~0.6% of equity as margin)" in sheet
    assert "1% risk -> 0.12 lots" in sheet
    assert "2x M5 ATR" in sheet and "M5-based" in sheet
    assert format_intraday_sizing_sheet(_decision_tier_analysis(m5_stats=_ts()), 10_000.0) is None
    tiny = format_intraday_sizing_sheet(_decision_tier_analysis(), 100.0)
    assert "below the 0.01-lot minimum" in tiny


def test_format_intraday_levels_reports_prior_day_vwap_and_range_used():
    from analysis.intraday_context import IntradayLevels
    from ai.ftmo_suggest import format_intraday_levels

    levels = IntradayLevels(
        last_price=100.0, prev_day_high=102.0, prev_day_low=97.0, prev_day_close=99.0,
        day_open=99.5, day_high=101.0, day_low=98.5, vwap=99.8, adr=4.0, range_used_pct=62.5, session_bars=120,
    )
    text = format_intraday_levels("X", levels)
    assert "prev-day H 102.0000 (+2.00% from price)" in text
    assert "session VWAP 99.8000" in text
    assert "% of it already used in that session" in text
    assert "not available" in format_intraday_levels("X", None)


def test_format_intraday_alignment_flags_an_m5_vs_h1_conflict():
    from ai.ftmo_suggest import format_intraday_alignment

    conflict = _decision_tier_analysis(m5_stats=_ts(trend="downtrend", last_price=1.0, atr=1.0))
    assert "M5 vs H1 CONFLICT" in format_intraday_alignment(conflict)
    aligned = format_intraday_alignment(_decision_tier_analysis())
    assert "ALIGNED — M5, H1 and H4 all read uptrend" in aligned


def test_stage1_instruction_makes_m5_the_only_stop_target_and_reachability_basis():
    text = build_ftmo_stage1_instruction()
    flat = " ".join(text.split())
    assert "how many multiples of M5 ATR the stop sits away" in flat
    assert "derived from the real M5 ATR" in flat
    assert "reward distance = <price/pips> = <N>x the M5 ATR" in flat
    assert "6x the M5 ATR" in flat and "H1 ATR" not in flat.split("reward distance")[1][:600]
    assert "Session levels" in text and "Intraday sizing sheet" in text
    # Trigger/invalidation conditions read on M5 in the schema text and examples.
    assert '"trigger_condition": "M5 closes above 2000.00 with M5 RSI(14) below ' in text
    assert '"invalidation_condition": "M5 closes below 76.00"' in text
    # The old H1/H4-as-primary claims and the dropped tiers are gone.
    assert "The H4 and H1 reads are the PRIMARY basis" not in text
    assert "M15" not in text
    # The glued-word bug (a slice ending without its trailing space) must never come back.
    assert "onANY" not in flat and "statedholding" not in flat


def test_daily_atr_line_is_context_only_in_the_ftmo_context_but_unchanged_for_pmex():
    from ai.ftmo_suggest import format_ftmo_asset_context
    from ai.portfolio_suggest import format_enriched_asset_context

    analysis = _decision_tier_analysis()
    analysis.base.display_name = "Gold"
    analysis.base.stats = _ts(last_price=4370.0, trend="uptrend", atr=30.0)
    analysis.base.stats.atr_pct = 0.7
    analysis.base.stats.pct_vs_sma20 = 1.0
    ftmo_text = format_ftmo_asset_context([analysis], account_equity=10_000.0)
    assert "CONTEXT only — NOT a stop distance; size stops from the M5 ATR" in ftmo_text
    assert "use this, not a flat percentage" not in ftmo_text
    assert "use this, not a flat percentage" in format_enriched_asset_context([analysis.base])


# --- Audit fixes 2026-09-24 ---


def test_effective_stop_floor_is_one_definition_used_by_min_viable_size_and_the_sheet():
    from ai.ftmo_suggest import effective_stop_floor, format_intraday_sizing_sheet

    # EURUSD-shaped: M5 ATR 0.00046 -> 2.0x = 0.00092 (0.081%); the 0.1% minimum (x1.02) = 0.001162 binds.
    a = _decision_tier_analysis()
    a.base.ask, a.base.bid = 1.1385, 1.1384
    a.m5_stats.atr = 0.00046
    floor = effective_stop_floor(a)
    assert floor.binding == "0.1% minimum"
    assert floor.distance == pytest.approx(0.001 * 1.1385 * 1.02)
    sheet = format_intraday_sizing_sheet(a, 10_000.0)
    assert "the 0.1% minimum floor" in sheet and f"{floor.distance:.5g}" in sheet
    assert "wider than 2x M5 ATR" in format_ftmo_min_viable_size(a, 10_000.0)


def test_effective_stop_floor_spread_and_broker_components():
    from ai.ftmo_suggest import effective_stop_floor

    wide_spread = _decision_tier_analysis()
    wide_spread.base.ask, wide_spread.base.bid = 4371.0, 4370.0  # spread 1.0 x4 = 4.0
    wide_spread.m5_stats.atr = 0.5                                 # 2.0x = 1.0; 0.1% min = 4.46 binds here
    assert effective_stop_floor(wide_spread).binding == "0.1% minimum"
    wide_spread.base.ask, wide_spread.base.bid = 4371.0, 4360.0    # spread 11 x4 = 44 binds
    assert effective_stop_floor(wide_spread).binding == "spread"
    assert effective_stop_floor(wide_spread).distance == pytest.approx(44.0)
    assert effective_stop_floor(_decision_tier_analysis(m5_stats=_ts(), h1_stats=_ts())) is None


def test_sizing_sheet_lot_rounding_is_epsilon_safe():
    from ai.ftmo_suggest import format_intraday_sizing_sheet

    a = _decision_tier_analysis()
    a.base.ask = a.base.bid = 1000.0
    a.m5_stats.atr = 6.25 / 2.0       # floor 6.25 -> 1% of 10k / (6.25*100) = 0.16 lots
    text = format_intraday_sizing_sheet(a, 10_000.0)
    assert "1% risk -> 0.16 lots" in text  # a bare double-slash floor divide would have said 0.15


def test_confluence_tolerance_is_one_m5_atr_not_a_flat_half_percent():
    from analysis.chart_structure import SRLevel, SRLevelsResult
    from ai.ftmo_suggest import format_intraday_alignment

    def structure(price):
        return ChartStructureSnapshot(
            fibonacci=None, patterns=[], trendlines=None,
            sr_levels=SRLevelsResult(resistance_levels=[SRLevel(price=price, touches=2, distance_pct=0.3, low=price, high=price)], support_levels=[]),
        )

    a = _decision_tier_analysis(h1_structure=structure(1000.0), m5_structure=structure(1003.0))  # 0.3% apart
    a.base.ask = 1000.0
    a.m5_stats.atr_pct = 0.05           # tolerance 0.05% -> NOT confluent
    assert "M5 levels backed by H1/H4: none found" in format_intraday_alignment(a)
    a.m5_stats.atr_pct = 0.4            # tolerance 0.4% -> confluent
    assert "none found" not in format_intraday_alignment(a)


def test_m5_zone_stop_respects_the_effective_floor_so_claude_is_not_handed_a_stop_the_guard_widens():
    from analysis.chart_structure import SRLevel, SRLevelsResult
    from ai.ftmo_suggest import effective_stop_floor, format_ftmo_asset_context

    support = SRLevel(price=1.1300, touches=3, distance_pct=-0.7, low=1.1298, high=1.1302)
    resistance = SRLevel(price=1.1500, touches=3, distance_pct=1.0, low=1.1498, high=1.1502)
    structure = ChartStructureSnapshot(
        fibonacci=None, patterns=[], trendlines=None,
        sr_levels=SRLevelsResult(resistance_levels=[resistance], support_levels=[support]),
    )
    a = _decision_tier_analysis(m5_structure=structure)
    a.base.ask, a.base.bid = 1.1385, 1.1384
    a.m5_stats.last_price = 1.1385
    a.m5_stats.atr = 0.00046  # 2.0x = 0.00092 < the 0.1% minimum 0.001161
    floor = effective_stop_floor(a)
    text = format_ftmo_asset_context([a], account_equity=10_000.0)
    zone_line = next(l for l in text.split("\n") if l.strip().startswith("buy [") and "stop" in l and "entry " in l)
    stop = float(zone_line.split("stop ")[1].split(",")[0])
    entry_low = float(zone_line.split("entry ")[1].split("-")[0])
    assert entry_low - stop == pytest.approx(floor.distance, abs=1e-4)  # zone prices print to 4 decimals


def test_session_levels_name_the_session_date_not_just_today():
    from analysis.intraday_context import IntradayLevels
    from ai.ftmo_suggest import format_intraday_levels

    lv = IntradayLevels(100.0, 102.0, 97.0, 99.0, 99.5, 101.0, 98.5, 99.8, 4.0, 62.5, 120, session_date="2026-09-23")
    assert "session 2026-09-23 O 99.5000" in format_intraday_levels("MSFT", lv)


def test_compact_chart_structure_keeps_the_numbers_and_drops_the_legend():
    from analysis.chart_structure import SRLevel, SRLevelsResult
    from ai.ftmo_suggest import format_chart_structure

    lvl = SRLevel(price=1.2001, touches=3, distance_pct=0.5, low=1.1998, high=1.2004, weighted_score=2.7, is_liquidity_pool=True)
    snap = ChartStructureSnapshot(fibonacci=None, patterns=[], trendlines=None,
                                  sr_levels=SRLevelsResult(resistance_levels=[lvl], support_levels=[]))
    full = format_chart_structure("M5", snap)
    compact = format_chart_structure("M5", snap, compact=True)
    assert "1.1998-1.2004 (3x, +0.50%) [LIQUIDITY POOL]" in compact
    assert "recency-weighted" not in compact and "recency-weighted" in full
    assert len(compact) < len(full)


def test_zone_states_each_target_as_a_multiple_of_the_m5_atr_for_the_reachability_rule():
    from analysis.chart_structure import SRLevel, SRLevelsResult
    from ai.ftmo_suggest import format_trade_zone_for

    support = SRLevel(price=98.0, touches=3, distance_pct=-2.0, low=97.5, high=98.5)
    resistance = SRLevel(price=108.0, touches=2, distance_pct=8.0, low=107.5, high=108.5)
    structure = ChartStructureSnapshot(
        fibonacci=None, patterns=[], trendlines=None,
        sr_levels=SRLevelsResult(resistance_levels=[resistance], support_levels=[support]),
    )
    stats = _ts(last_price=100.0, atr=2.0)
    text = format_trade_zone_for("M5", stats, structure, [], reach_atr=2.0)
    # buy zone 97.5-98.5 -> entry_ref 98.5; target 108.0 -> 9.5 away = 4.75 -> "4.8x" (or 4.7x by rounding)
    assert "distance in M5 ATRs: 4." in text
    assert "distance in M5 ATRs" not in format_trade_zone_for("M5", stats, structure, [])


# --- M5-only decision tier (2026-09-24): level reliability, session targets, M5 trend alignment ---


def test_backtest_all_sr_levels_intraday_mode_uses_the_m5_simulation_parameters():
    import config as _config

    ohlc = _sine_wave_ohlc(n_cycles=60, period=12)
    sr = SRLevelsResult(
        resistance_levels=[SRLevel(price=105.0, touches=10, distance_pct=5.0, low=104.5, high=105.5)],
        support_levels=[SRLevel(price=95.0, touches=10, distance_pct=-5.0, low=94.5, high=95.5)],
        tolerance_pct=0.5,
    )
    trade_cost = _make_trade_cost(swap_long_pct_per_day=-0.5, swap_short_pct_per_day=0.5)
    daily = _backtest_all_sr_levels(ohlc, sr, trade_cost, _make_spec(), 1.10)
    intraday = _backtest_all_sr_levels(ohlc, sr, trade_cost, _make_spec(), 1.10, intraday=True)
    assert daily[95.0].max_holding_bars != intraday[95.0].max_holding_bars
    assert intraday[95.0].max_holding_bars == _config.TRADE_SIM_INTRADAY_MAX_HOLDING_BARS
    assert intraday[95.0].stop_atr_multiple == _config.M5_ATR_STOP_MULTIPLE
    assert intraday[95.0].target_atr_multiple == 2 * _config.M5_ATR_STOP_MULTIPLE
    assert intraday[95.0].swap_pct_per_day_used == 0.0 and intraday[105.0].swap_pct_per_day_used == 0.0


@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_compute_intraday_reads_level_reliability_is_opt_in_and_one_fetch_feeds_everything(mock_fetch_history):
    from ai.ftmo_suggest import _compute_intraday_reads

    mock_fetch_history.return_value = _make_intraday_history(n=900)
    trade_cost = _make_trade_cost()
    d1 = _make_intraday_history(n=40)
    off = _compute_intraday_reads("EURUSD", d1, trade_cost, None, 1.1005, compute_level_reliability=False)
    on = _compute_intraday_reads("EURUSD", d1, trade_cost, None, 1.1005, compute_level_reliability=True)
    assert off["m5_level_reliability"] == {}
    assert isinstance(on["m5_level_reliability"], dict)
    assert {"m5_stats", "m5_structure", "intraday_backtests", "intraday_levels", "m5_atr_pct_median"} <= set(on)
    assert [c.args[1] for c in mock_fetch_history.call_args_list] == ["M5", "M5"]  # one M5 fetch per call, nothing else


def test_session_target_prices_are_the_prev_day_and_today_extremes():
    from analysis.intraday_context import IntradayLevels
    from ai.ftmo_suggest import _session_target_prices

    levels = IntradayLevels(
        last_price=100.0, prev_day_high=102.0, prev_day_low=97.0, prev_day_close=99.0,
        day_open=99.5, day_high=101.0, day_low=98.5, vwap=99.8, adr=4.0, range_used_pct=62.5, session_bars=120,
    )
    assert _session_target_prices(levels) == [102.0, 97.0, 101.0, 98.5]
    assert _session_target_prices(None) == []


def test_m5_zone_offers_the_session_extremes_as_targets_when_m5_structure_has_no_far_band():
    from analysis.chart_structure import SRLevel, SRLevelsResult
    from analysis.intraday_context import IntradayLevels

    support = SRLevel(price=98.0, touches=3, distance_pct=-2.0, low=97.5, high=98.5)
    structure = ChartStructureSnapshot(
        fibonacci=None, patterns=[], trendlines=None,
        sr_levels=SRLevelsResult(resistance_levels=[], support_levels=[support]),
    )
    levels = IntradayLevels(
        last_price=100.0, prev_day_high=112.0, prev_day_low=90.0, prev_day_close=99.0,
        day_open=99.5, day_high=101.0, day_low=98.5, vwap=99.8, adr=4.0, range_used_pct=62.5, session_bars=120,
    )
    a = _decision_tier_analysis(
        m5_structure=structure, m5_stats=_ts(trend="uptrend", last_price=100.0, atr=1.0), intraday_levels=levels
    )
    a.base.ask, a.base.bid = 100.05, 100.0
    text = format_trade_zone(a, [])
    assert "buy [" in text and "112.0000" in text and "distance in M5 ATRs" in text


def test_aligned_m5_trend_direction_needs_the_trend_and_the_regime_to_agree():
    from ai.ftmo_suggest import aligned_m5_trend_direction

    assert aligned_m5_trend_direction(_ts(trend="uptrend", market_regime="choppy_up")) == "up"
    assert aligned_m5_trend_direction(_ts(trend="downtrend", market_regime="trending_down")) == "down"
    assert aligned_m5_trend_direction(_ts(trend="downtrend", market_regime="trending_up")) is None
    assert aligned_m5_trend_direction(_ts(trend="uptrend", market_regime="sideways")) is None
    assert aligned_m5_trend_direction(_ts(trend="flat", market_regime="trending_up")) is None
    assert aligned_m5_trend_direction(_ts(trend=None, market_regime="trending_up")) is None
    assert aligned_m5_trend_direction(None) is None


def test_trend_radar_is_driven_by_the_m5_setups_not_h1_h4():
    m5_trending = _trend_radar_analysis("EURUSD", "trending_up", "uptrend")
    # H1/H4 read a clean trend but M5 reads flat -> no radar row (H1/H4 are context only now).
    m5_flat = _trend_radar_analysis("GBPUSD", "trending_up", "uptrend")
    m5_flat.m5_stats = _ts(market_regime="sideways", trend="flat", last_price=1.0)
    radar = build_trend_radar([m5_trending, m5_flat])
    assert "EURUSD" in radar and "GBPUSD" not in radar
    assert "on M5 (the decision timeframe" in radar


def test_zone_states_how_many_atrs_the_entry_zone_sits_from_the_last_price():
    from analysis.chart_structure import SRLevel, SRLevelsResult
    from ai.ftmo_suggest import format_trade_zone_for

    support = SRLevel(price=98.0, touches=3, distance_pct=-2.0, low=97.5, high=98.5)
    resistance = SRLevel(price=108.0, touches=2, distance_pct=8.0, low=107.5, high=108.5)
    structure = ChartStructureSnapshot(
        fibonacci=None, patterns=[], trendlines=None,
        sr_levels=SRLevelsResult(resistance_levels=[resistance], support_levels=[support]),
    )
    stats = _ts(last_price=100.0, atr=0.5)
    text = format_trade_zone_for("M5", stats, structure, [], reach_atr=0.5, stop_atr=0.1)
    # buy zone 97.5-98.5 sits 1.5 below 100.0 -> 3.0 ATRs of 0.5
    assert "[entry zone 3.0x the M5 ATR from the last price]" in text
    # price INSIDE the zone -> 0.0x
    inside = format_trade_zone_for("M5", _ts(last_price=98.0, atr=0.5), structure, [], reach_atr=0.5, stop_atr=0.1)
    assert "[entry zone 0.0x the M5 ATR from the last price]" in inside
    # no reach_atr -> no gap text at all (H1-era callers unchanged)
    assert "entry zone 0" not in format_trade_zone_for("M5", stats, structure, [], stop_atr=0.1)
    assert "from the last price" not in format_trade_zone_for("M5", stats, structure, [], stop_atr=0.1)


def test_format_trade_zone_uses_the_m5_measured_hold_rate_cutoffs_not_the_h1_ones():
    import config as _config
    from analysis.backtest import LevelReliabilityBacktest
    from analysis.chart_structure import SRLevel, SRLevelsResult

    support = SRLevel(price=98.0, touches=3, distance_pct=-2.0, low=97.5, high=98.5)
    structure = ChartStructureSnapshot(
        fibonacci=None, patterns=[], trendlines=None,
        sr_levels=SRLevelsResult(resistance_levels=[SRLevel(price=112.0, touches=2, distance_pct=12.0, low=111.5, high=112.5)], support_levels=[support]),
    )
    rel = {98.0: LevelReliabilityBacktest(
        level_price=98.0, level_low=97.5, level_high=98.5, side="support", tests=20, holds=4, breaks=16, hold_rate_pct=20.0,
        trades=20, wins=8, losses=12, timeouts=0, win_rate_pct=40.0, avg_r_multiple=0.2, stop_atr_multiple=2.0,
        target_atr_multiple=4.0, max_holding_bars=48,
    )}
    a = _decision_tier_analysis(m5_structure=structure, m5_stats=_ts(trend="uptrend", last_price=100.0, atr=1.0), m5_level_reliability=rel)
    a.base.ask, a.base.bid = 100.05, 100.0
    text = format_trade_zone(a, [])
    # basis is the bare "nearest_sr_level" (base confidence moderate); a 20% hold rate is mid-pack on M5
    # (weak <= 6 < 20 < 25 <= strong), so it must NOT be downgraded the way the H1-era 40% cutoff would.
    assert _config.M5_ZONE_WEAK_HOLD_RATE_PCT < 20.0 < _config.M5_ZONE_STRONG_HOLD_RATE_PCT
    assert "buy [moderate, basis=nearest_sr_level" in text


def test_zone_stars_targets_beyond_the_reachability_limit_and_only_then_prints_the_legend():
    from analysis.chart_structure import SRLevel, SRLevelsResult
    from ai.ftmo_suggest import format_trade_zone_for

    support = SRLevel(price=98.0, touches=3, distance_pct=-2.0, low=97.5, high=98.5)
    near = SRLevel(price=108.0, touches=2, distance_pct=8.0, low=107.5, high=108.5)
    far = SRLevel(price=130.0, touches=2, distance_pct=30.0, low=129.5, high=130.5)
    structure = ChartStructureSnapshot(
        fibonacci=None, patterns=[], trendlines=None,
        sr_levels=SRLevelsResult(resistance_levels=[near, far], support_levels=[support]),
    )
    stats = _ts(last_price=100.0, atr=2.0)
    # entry_ref 98.5: near target 108.0 = 4.75x, far target 130.0 = 15.75x of a 2.0 ATR; limit 6x
    text = format_trade_zone_for("M5", stats, structure, [], reach_atr=2.0, stop_atr=0.5, reach_limit_atr=6.0)
    assert "4.8x, 15.8x*" in text or "4.7x, 15.8x*" in text
    assert "* = beyond the ~6x same-session reachability limit" in text
    within = format_trade_zone_for("M5", stats, _structure_without_far(structure), [], reach_atr=2.0, stop_atr=0.5, reach_limit_atr=6.0)
    assert "*" not in within.split("distance in M5 ATRs:")[1]
    # no limit given (H1-era callers) -> never a star
    assert "*" not in format_trade_zone_for("M5", stats, structure, [], reach_atr=2.0, stop_atr=0.5).split("distance in M5 ATRs:")[1]


def _structure_without_far(structure):
    from analysis.chart_structure import SRLevelsResult

    return ChartStructureSnapshot(
        fibonacci=None, patterns=[], trendlines=None,
        sr_levels=SRLevelsResult(
            resistance_levels=[structure.sr_levels.resistance_levels[0]], support_levels=structure.sr_levels.support_levels
        ),
    )


# --- Quote-currency-aware risk sizing (2026-09-25): USDJPY's risk was overstated ~159x ---


def _jpy_spec():
    from data.mt5_source import ContractSpec

    # Real FTMO USDJPY: contract 100000, tick 0.001, tick value 0.63125 USD -> 631.25 USD per 1.0 price unit per lot.
    return ContractSpec(
        volume_min=0.01, volume_step=0.01, volume_max=100.0, trade_contract_size=100000.0,
        currency_margin="USD", margin_initial=1000.0, money_per_price_unit=631.25,
    )


def test_risk_per_price_unit_uses_mt5_tick_money_and_falls_back_to_the_contract_size():
    from data.mt5_source import ContractSpec

    usd = ContractSpec(volume_min=0.01, volume_step=0.01, volume_max=1.0, trade_contract_size=100000.0, currency_margin="USD", margin_initial=1.0)
    assert usd.risk_per_price_unit == 100000.0            # no tick data -> the USD-quote behaviour, unchanged
    assert _jpy_spec().risk_per_price_unit == 631.25
    zero = ContractSpec(volume_min=0.01, volume_step=0.01, volume_max=1.0, trade_contract_size=10.0, currency_margin="USD", margin_initial=1.0, money_per_price_unit=0.0)
    assert zero.risk_per_price_unit == 10.0


def test_money_per_price_unit_snaps_usd_quoted_specs_to_the_exact_contract_size():
    from types import SimpleNamespace
    from data.mt5_source import _money_per_price_unit

    sol = SimpleNamespace(trade_tick_size=0.01, trade_tick_value=1.0, trade_tick_value_loss=1.0, trade_contract_size=100.0)
    assert _money_per_price_unit(sol) == 100.0 and isinstance(_money_per_price_unit(sol), float)
    jpy = SimpleNamespace(trade_tick_size=0.001, trade_tick_value=0.63125, trade_tick_value_loss=0.63126, trade_contract_size=100000.0)
    assert _money_per_price_unit(jpy) == pytest.approx(631.26)          # the LOSS-side value is what a stop risks
    eur = SimpleNamespace(trade_tick_size=0.01, trade_tick_value=0.011369, trade_tick_value_loss=0.011369, trade_contract_size=1.0)
    assert _money_per_price_unit(eur) == pytest.approx(1.1369)
    assert _money_per_price_unit(SimpleNamespace(trade_tick_size=0.0, trade_tick_value=1.0, trade_contract_size=1.0)) is None
    assert _money_per_price_unit(SimpleNamespace(trade_tick_size=0.01, trade_tick_value=0.0, trade_contract_size=1.0)) is None
    assert _money_per_price_unit(SimpleNamespace()) is None


def test_min_viable_size_for_usdjpy_is_a_hundredth_of_a_percent_not_1_63():
    # The real 2026-09-24 line: stop 0.16216 JPY, 0.01 lot, equity 9943.18. Old maths: 0.01*0.16216*100000/9943 = 1.63%.
    from ai.ftmo_suggest import effective_stop_floor

    a = _decision_tier_analysis(m5_stats=_ts(trend="uptrend", last_price=158.98, atr=0.05))
    a.base.contract_spec = _jpy_spec()
    a.base.ask, a.base.bid = 158.985, 158.982
    text = format_ftmo_min_viable_size(a, 9943.18)
    floor = effective_stop_floor(a).distance
    expected = 0.01 * floor * 631.25 / 9943.18 * 100
    assert f"AT LEAST {expected:.2f}% risk" in text
    assert expected < 0.05  # nowhere near the 1.63% that made Mega call it "mechanically untradeable"


def test_rebalance_plan_sizes_a_jpy_quoted_position_by_its_real_money_per_price_unit():
    from ai.portfolio_suggest import AllocationEntry
    from risk.apply_suggestion import compute_rebalance_plan
    from data.mt5_source import AccountSummary, MarketAsset

    account = AccountSummary(balance=9943.18, equity=9943.18, free_margin=9000.0, currency="USD")
    alloc = {"USDJPY": AllocationEntry(pct=0.5, price=158.985, stop_loss=158.785, take_profit=159.5, side="buy", reason="r")}
    plan = compute_rebalance_plan(
        [], account, alloc, lambda s: _jpy_spec(), {"USDJPY": MarketAsset("USDJPY", "US Dollar vs Yen", 158.982, 158.985)},
    )
    order = next(o for o in plan if o.symbol == "USDJPY")
    # risk 0.5% = $49.72; stop 0.2 JPY x $631.25 = $126.25 per lot -> 0.39 lots (the old maths: 0.0025 -> infeasible)
    assert order.action == "open" and order.volume == pytest.approx(0.39)


# --- Notional in ACCOUNT currency for commission and swap (2026-09-25): USDJPY was understated ~159x ---


def test_usdjpy_commission_pct_uses_the_usd_notional_not_contract_size_times_the_jpy_price():
    from ai.ftmo_suggest import _real_backtest_execution_kwargs

    trade_cost = _make_trade_cost(category="Forex", spread_pct_of_price=0.002)
    kwargs = _real_backtest_execution_kwargs(trade_cost, _jpy_spec(), 158.985)
    commission_pct = config.FTMO_COMMISSION_FX_USD_PER_LOT_ROUND_TURN / (631.25 * 158.985) * 100   # ~ $ / $100k
    assert kwargs["round_trip_cost_pct"] == pytest.approx(0.002 + commission_pct)
    old_wrong = config.FTMO_COMMISSION_FX_USD_PER_LOT_ROUND_TURN / (100000.0 * 158.985) * 100
    assert commission_pct == pytest.approx(old_wrong * 100000.0 / 631.25)          # ~158x larger than the old figure


def test_eurusd_commission_is_unchanged_because_its_money_per_unit_equals_the_contract_size():
    from ai.ftmo_suggest import _real_backtest_execution_kwargs

    eur = _make_spec()  # USD-quoted: no tick money -> falls back to the contract size
    trade_cost = _make_trade_cost(category="Forex", spread_pct_of_price=0.002)
    kwargs = _real_backtest_execution_kwargs(trade_cost, eur, 1.14)
    expected = 0.002 + config.FTMO_COMMISSION_FX_USD_PER_LOT_ROUND_TURN / (eur.trade_contract_size * 1.14) * 100
    assert kwargs["round_trip_cost_pct"] == pytest.approx(expected)


def test_swap_pct_of_notional_uses_the_account_currency_notional_for_a_jpy_quoted_symbol():
    from types import SimpleNamespace
    from data.mt5_source import _SWAP_MODE_POINTS, _compute_swap_pct_per_day

    # Real USDJPY numbers: contract 100000, tick 0.001, tick value 0.63125 USD, point == tick size.
    info = SimpleNamespace(
        swap_mode=_SWAP_MODE_POINTS, swap_long=-10.0, swap_short=4.0, point=0.001,
        trade_tick_size=0.001, trade_tick_value=0.63125, trade_tick_value_loss=0.63125, trade_contract_size=100000.0,
    )
    long_pct, short_pct = _compute_swap_pct_per_day(info, 158.985)
    # -10 points x $0.63125 per point = -$6.3125/day on a ~$100,364 notional (631.25 * 158.985) = -0.00629%/day
    assert long_pct == pytest.approx(-10 * 0.63125 / (631.25 * 158.985) * 100)
    assert short_pct == pytest.approx(4 * 0.63125 / (631.25 * 158.985) * 100)


def test_color_pnl_never_crashes_on_a_missing_pnl():
    import ast
    from pathlib import Path

    source = Path("app.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_color_pnl")
    namespace: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "app.py", "exec"), namespace)
    color = namespace["_color_pnl"]
    assert color(None) == "color: inherit" and color(float("nan")) == "color: inherit"
    assert color(5.0) == "color: green" and color(-1.0) == "color: red" and color(0.0) == "color: inherit"


# --- Cost drag as an explicit, objective gate (2026-09-25) --------------------------------------------

def test_cost_drag_r_is_round_trip_cost_over_the_stop_and_its_inverse_round_trips():
    from ai.ftmo_suggest import cost_drag_r, stop_distance_for_drag

    a = _decision_tier_analysis(trade_cost=_make_trade_cost(category="Metals", spread_pct_of_price=0.02))
    price = a.base.ask
    cost_price = 0.02 / 100 * price  # 0.8741 price units (metals: no FX commission)
    assert cost_drag_r(a, 8.0) == pytest.approx(cost_price / 8.0, rel=1e-3)
    assert stop_distance_for_drag(a, 0.10) == pytest.approx(cost_price / 0.10, rel=1e-3)
    assert cost_drag_r(a, stop_distance_for_drag(a, 0.10)) == pytest.approx(0.10, rel=1e-6)
    assert cost_drag_r(_decision_tier_analysis(trade_cost=None), 8.0) is None
    assert cost_drag_r(a, 0.0) is None


def test_sizing_sheet_shows_the_stop_that_caps_drag_and_flags_the_cost_veto():
    from ai.ftmo_suggest import format_intraday_sizing_sheet

    cheap = format_intraday_sizing_sheet(
        _decision_tier_analysis(trade_cost=_make_trade_cost(category="Metals", spread_pct_of_price=0.001)), 10_000.0
    )
    assert "COST VETO" not in cheap and "would cap that drag" not in cheap  # 0.0055R: nothing to say
    costly = format_intraday_sizing_sheet(
        _decision_tier_analysis(trade_cost=_make_trade_cost(category="Metals", spread_pct_of_price=0.05)), 10_000.0
    )
    # 0.05% of 4370.5 = 2.185 over an 8-unit stop = 0.27R: above the 0.10R target, below the 0.35R veto.
    assert "would cap that drag at 0.1R" in costly and "not a floor" in costly and "COST VETO" not in costly
    ruinous = format_intraday_sizing_sheet(
        _decision_tier_analysis(trade_cost=_make_trade_cost(category="Metals", spread_pct_of_price=0.10)), 10_000.0
    )
    assert "COST VETO" in ruinous  # 0.55R



# --- Position Hunt in the Mega prompt (2026-09-25) ---------------------------------------------------------

def _hunt_analysis(symbol="EURUSD", d1="uptrend", h4="uptrend", m5_trend="uptrend", m5_regime="trending_up", structure=True):
    from analysis.chart_structure import ChartStructureSnapshot, FibonacciLevels

    a = _trend_radar_analysis(symbol, m5_regime, m5_trend, h4_regime=m5_regime, h4_trend=h4)
    a.base.stats = _ts(trend=d1, market_regime=m5_regime, last_price=1.0)
    a.base.stats.pct_vs_sma20 = 1.0
    if structure:  # a real swing leg is enough structure for the V2 check (the bare fixture has none)
        a.m5_structure = ChartStructureSnapshot(
            fibonacci=FibonacciLevels(
                swing_high=1.1, swing_low=0.9, high_is_more_recent=True, levels={"38.2%": 1.02, "61.8%": 0.98},
                current_price=1.0, nearest_level_name="50.0%", nearest_level_price=1.0, distance_to_nearest_pct=0.0,
            ),
            sr_levels=None, trendlines=None, patterns=[],
        )
    return a


def _fixed_hunt(analyses, **kw):
    from ai.ftmo_suggest import build_position_hunt

    kwargs = dict(
        account_equity=100_000.0, ftmo_status=_make_status(), calendar_events=None,
        market_open_fn=lambda symbol, now: True, minutes_to_close_fn=lambda symbol, now: None,
    )
    kwargs.update(kw)
    return build_position_hunt(analyses, **kwargs)


def test_position_hunt_facts_read_the_real_analysis_fields():
    from ai.ftmo_suggest import build_hunt_facts

    a = _hunt_analysis()
    facts = build_hunt_facts(a, 100_000.0, None, None, market_open_fn=lambda s, n: True, minutes_to_close_fn=lambda s, n: 42.0)
    assert (facts.symbol, facts.d1_dir, facts.h4_dir, facts.m5_dir) == ("EURUSD", "up", "up", "up")
    assert facts.market_open is True and facts.minutes_to_close == 42.0
    assert facts.edge_verdicts == {"buy": {}, "sell": {}}  # no intraday backtests -> nothing claimed
    assert facts.blackout_reason is None and facts.cost_drag_r is None  # no trade cost read -> no drag, so no veto


def test_position_hunt_block_ranks_an_aligned_candidate_first_and_a_closed_market_is_a_v7_veto():
    text = _fixed_hunt([_hunt_analysis("EURUSD"), _hunt_analysis("GBPUSD", d1="downtrend", h4="downtrend",
                                                                  m5_trend="downtrend", m5_regime="trending_down")])
    assert "POSITION HUNT" in text and "1." in text and "tier A" in text
    closed = _fixed_hunt([_hunt_analysis("EURUSD")], market_open_fn=lambda symbol, now: False)
    assert "V7: market is closed right now" in closed and "shortlist: none" in closed


def test_a_symbol_with_no_structure_at_all_is_a_v2_veto():
    text = _fixed_hunt([_hunt_analysis(structure=False)])
    assert "V2: no real structure" in text and "shortlist: none" in text


def test_position_hunt_uses_the_ftmo_headroom_as_the_v4_size_budget():
    from analysis.position_hunter import HuntFacts
    from ai.ftmo_suggest import build_position_hunt

    # Headroom 0.2% x 50% = a 0.1% budget: any instrument whose minimum lot needs more is vetoed (V4).
    tight = _make_status(daily_loss_headroom_pct=0.2)
    with patch("ai.ftmo_suggest.min_viable_risk_pct", return_value=0.5):
        text = _fixed_hunt([_hunt_analysis()], ftmo_status=tight)
    assert "V4:" in text and "shortlist: none" in text
    with patch("ai.ftmo_suggest.min_viable_risk_pct", return_value=0.5):
        roomy = _fixed_hunt([_hunt_analysis()], ftmo_status=_make_status(daily_loss_headroom_pct=3.0))
    assert "1. EURUSD BUY" in roomy


def test_position_hunt_survives_an_unreadable_symbol_and_returns_empty_without_any():
    broken = _hunt_analysis("EURUSD")
    broken.base.stats = None  # unreadable
    good = _hunt_analysis("GBPUSD")
    assert "GBPUSD" in _fixed_hunt([broken, good])
    assert _fixed_hunt([]) == ""


@patch("ai.ftmo_suggest.build_macro_snapshot", return_value="MACRO")
@patch("ai.ftmo_suggest.format_book_wisdom", return_value="WISDOM")
def test_build_ftmo_summary_puts_the_position_hunt_where_the_trend_radar_was(mock_wisdom, mock_macro):
    account = AccountSummary(balance=100_000.0, equity=100_000.0, free_margin=90_000.0, currency="USD")
    with (
        patch("ai.ftmo_suggest.is_symbol_tradable_now", return_value=True),
        patch("ai.ftmo_suggest._default_minutes_to_close", return_value=None),
    ):
        summary = build_ftmo_summary(
            account, assets=[MarketAsset("EURUSD", "Euro", 1.1, 1.1005)], ftmo_status=_make_status(),
            analyses=[_hunt_analysis("EURUSD")],
        )
    assert "POSITION HUNT" in summary and "Trend Radar" not in summary
    assert summary.index("POSITION HUNT") < summary.index("Tradable instruments (Market Watch)") + 400
    assert "1. EURUSD BUY [tier A]" in summary


@patch("ai.ftmo_suggest.build_macro_snapshot", return_value="MACRO")
@patch("ai.ftmo_suggest.format_book_wisdom", return_value="WISDOM")
def test_build_ftmo_summary_falls_back_to_the_radar_if_the_hunt_raises(mock_wisdom, mock_macro):
    account = AccountSummary(balance=100_000.0, equity=100_000.0, free_margin=90_000.0, currency="USD")
    with patch("ai.ftmo_suggest.build_position_hunt", side_effect=RuntimeError("boom")):
        summary = build_ftmo_summary(
            account, assets=[MarketAsset("EURUSD", "Euro", 1.1, 1.1005)], ftmo_status=_make_status(),
            analyses=[_hunt_analysis("EURUSD")],
        )
    assert "Trend Radar" in summary and "POSITION HUNT" not in summary


def test_the_mega_prompt_carries_the_protocol_the_entry_mode_field_and_the_verdict_language():
    from ai.ftmo_suggest import build_ftmo_stage1_instruction

    text = build_ftmo_stage1_instruction()
    assert "POSITION-HUNTING PROTOCOL" in text and "veto id (V1-V4, V7)" in text
    assert "\"entry_mode\": \"stop\"" in text and "never use it for a Pending Setup" in text
    assert "NO-INFORMATION is never a reason to exclude" in text
    assert "not CONTRADICTED (SUPPORTED is better, NO-INFORMATION is neutral)" in text
    assert "real backtest support for that direction" not in text  # the old win-rate-as-veto wording is gone


def test_the_audit_prompt_scores_against_the_random_baseline_and_checks_the_shortlist():
    from ai.ftmo_suggest import AUDIT_INSTRUCTION

    assert "RANDOM-ENTRY BASELINE" in AUDIT_INSTRUCTION and "never CONTRADICTED" in AUDIT_INSTRUCTION
    assert "shortlisted candidate the draft omitted" in AUDIT_INSTRUCTION


def test_the_rehunt_block_reaches_the_mega_prompt_when_an_entry_died(monkeypatch):
    from datetime import datetime, timezone
    from ai import rehunt

    rehunt.record_dead_entry("SOLUSD", "buy", 82.5, 81.0, 86.0, "target reached without a fill", datetime.now(timezone.utc))
    account = AccountSummary(balance=100_000.0, equity=100_000.0, free_margin=90_000.0, currency="USD")
    monkeypatch.setattr(config, "POSITION_HUNT_ENABLED", False)
    with patch("ai.ftmo_suggest.build_macro_snapshot", return_value="M"), patch("ai.ftmo_suggest.format_book_wisdom", return_value="W"):
        summary = build_ftmo_summary(
            account, assets=[MarketAsset("EURUSD", "Euro", 1.1, 1.1005)], ftmo_status=_make_status(),
            analyses=[_hunt_analysis("EURUSD")],
        )
    assert "RE-HUNT CANDIDATES" in summary and "SOLUSD BUY" in summary


def test_m5_trade_zone_candidates_are_built_with_the_same_arguments_the_formatter_prints():
    from ai.ftmo_suggest import m5_trade_zone_candidates

    a = _hunt_analysis()
    zones = m5_trade_zone_candidates(a, [])
    assert set(zones) == {"buy", "sell"}  # no structure -> None each, but the same shape the recheck iterates


def test_the_hunter_sees_every_correlated_pair_not_just_the_15_shown_in_the_prompt_table():
    from ai.ftmo_suggest import compute_ftmo_correlation_pairs

    idx = pd.date_range("2026-01-01", periods=80, freq="D")
    base = np.cumsum(np.random.default_rng(1).normal(0, 1, 80)) + 100
    analyses = []
    for i in range(8):  # 8 near-identical series -> 28 pairs with |r| >= 0.7
        a = _trend_radar_analysis(f"S{i}", "trending_up", "uptrend")
        a.base.prices = pd.Series(base + np.random.default_rng(10 + i).normal(0, 0.05, 80), index=idx)
        analyses.append(a)
    assert len(compute_ftmo_correlation_pairs(analyses)) == 15  # the prompt table stays capped
    assert len(compute_ftmo_correlation_pairs(analyses, limit=None)) == 28  # the V5 veto sees them all


def test_the_prompts_carry_the_playbook_level_map_and_structured_trigger_rules():
    from ai.ftmo_suggest import AUDIT_INSTRUCTION, build_ftmo_stage1_instruction

    text = build_ftmo_stage1_instruction()
    assert "PLAYBOOK line printed under each shortlisted" in text and "WORST entry" in text
    assert "3b. LEVEL CHOICE" in text and "SOLUSD case" in text and "P90 2.7 ATR" in text
    assert "3c. STRUCTURED TRIGGERS" in text and "range_break / reclaim / close_beyond" in text
    assert "\"trigger\": a STRUCTURED trigger" in text
    assert "Do not add soft (close-based) stops, time stops or a TP1 partial" in text
    assert "ENTRY / LEVEL CHOICE" in AUDIT_INSTRUCTION and "'the strongest'" in AUDIT_INSTRUCTION
    from data.book_wisdom import format_book_wisdom

    wisdom = format_book_wisdom()
    assert "Use more than one unit" not in wisdom  # the multi-position advice was dropped on purpose
    assert "Use market orders rather than limit orders" in wisdom and "RANDOM price at the same distance" in wisdom


def test_hunt_cost_drag_is_charged_at_the_playbook_structure_stop_not_the_tightest_stop():
    from types import SimpleNamespace

    import config
    from ai.ftmo_suggest import _playbook_stop_distance

    a = SimpleNamespace(m5_stats=SimpleNamespace(atr=2.0), m5_range=SimpleNamespace(high=112.0, low=100.0))  # range 12 = 6 ATR
    floor = SimpleNamespace(distance=4.0)  # the tightest valid stop (2 ATR)
    assert _playbook_stop_distance(a, floor) == config.PLAYBOOK_STOP_MAX_ATR * 2.0  # clipped to 4 ATR
    a.m5_range = SimpleNamespace(high=102.0, low=100.0)  # a 1 ATR range clips UP to the 1.5 ATR minimum ... but never below the floor
    assert _playbook_stop_distance(a, floor) == 4.0
    a.m5_range = SimpleNamespace(high=107.0, low=100.0)  # 3.5 ATR range
    assert _playbook_stop_distance(a, floor) == 7.0
    a.m5_range = None
    assert _playbook_stop_distance(a, floor) == 4.0


def test_the_protocol_treats_trend_correlation_and_backtests_as_light_context_and_caps_size_by_measured_math():
    from ai.ftmo_suggest import AUDIT_INSTRUCTION, build_ftmo_stage1_instruction

    text = build_ftmo_stage1_instruction()
    assert "LIGHT preference for trades with the closed D1/H4 trend" in text and "needs no higher bar" in text
    assert "Backtest lines are CONTEXT for sizing only" in text and "Correlation between symbols is information, not a reason" in text
    assert "SIZE BY MEASURED QUALITY" in text and "base 0.4% risk per trade; up to 0.75%" in text and "Never above 0.75% per trade" in text
    assert "higher bar and a smaller size" not in text
    assert "Correlation under stress (LOW weight)" in text


def test_the_clerk_context_is_a_compact_m5_read_without_the_daily_backtests_and_the_context_tier():
    from ai.ftmo_suggest import format_clerk_context, format_ftmo_asset_context

    analysis = _decision_tier_analysis()
    compact = format_clerk_context(analysis)
    full = format_ftmo_asset_context([analysis])
    assert len(compact) < 0.6 * len(full)
    assert compact.startswith("- XAUUSD (Gold): bid 4370.0, ask 4370.5")
    assert "M5 DECISION READ for XAUUSD" in compact and "M5 technical" in compact and "Higher-timeframe context (regime only" in compact
    for dropped in ("historical overbought RSI reaction", "CONTEXT TIER", "Intraday sizing sheet", "minimum viable size", "M5 trade-zone candidate"):
        assert dropped not in compact


@patch("ai.ftmo_suggest._fetch_ftmo_headlines", return_value=["a headline"])
@patch("ai.ftmo_suggest.get_trade_economics")
@patch("ai.ftmo_suggest.get_contract_spec")
@patch("ai.ftmo_suggest.fetch_mt5_price_history")
def test_lean_live_analysis_skips_backtests_headlines_and_the_hunter_only_reads(
    mock_fetch_history, mock_contract_spec, mock_trade_economics, mock_headlines
):
    mock_fetch_history.return_value = _make_intraday_history(n=300)
    mock_contract_spec.return_value = None
    mock_trade_economics.return_value = _make_trade_cost()

    lean = analyze_ftmo_asset_live("EURUSD", bid=1.1000, ask=1.1005, description="Euro", lean=True)
    full = analyze_ftmo_asset_live("EURUSD", bid=1.1000, ask=1.1005, description="Euro")

    assert lean.base.momentum_persistence_backtest is None and lean.base.rsi_overbought_backtest is None
    assert full.base.momentum_persistence_backtest is not None
    assert lean.base.headlines == [] and full.base.headlines == ["a headline"]
    # The decision-tier M5 reads the Clerk defends with are still there, and identical to the full analysis.
    assert lean.m5_stats is not None and lean.m5_structure is not None and len(lean.m5_recent) == len(full.m5_recent) > 0
    assert lean.m5_stats.last_price == full.m5_stats.last_price
    assert lean.h1_stats.last_price == full.h1_stats.last_price
    assert lean.htf_closed == {}
    assert lean.intraday_backtests.support_resistance_backtest is None and lean.intraday_backtests.null_baseline_buy is None
