import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import config
from ai.openrouter_client import FAILED_MESSAGE as OPENROUTER_FAILED_MESSAGE, MISSING_KEY_MESSAGE as OPENROUTER_MISSING_KEY_MESSAGE
from ai.portfolio_suggest import AUDIT_MODELS
from ai.researcher import (
    _build_macro_snapshot_block,
    _build_technical_snapshot,
    _build_track_record_block,
    _category_rss_feed_url,
    _check_price_grounding,
    _export_research_note,
    _extract_price_anchors,
    _extract_section,
    _fetch_category_news_block,
    _fetch_news_with_fallback,
    _filter_relevant_news,
    _fmt,
    _format_equity_fundamentals,
    _format_fiscal_snapshot,
    _format_news_block,
    _google_news_query,
    _parse_published,
    _read_calibration_ledger,
    _record_calibration_entry,
    _relative_age,
    _resolve_ftmo_yahoo_ticker,
    _symbol_relevance_keywords,
    _strip_sentiment_line,
    _run_researcher_model,
    is_researcher_due,
    latest_research_report,
    next_researcher_check_utc,
    parse_researcher_sentiment,
    read_researcher_enabled,
    read_researcher_state,
    read_researcher_trigger,
    run_researcher_check,
    save_research_report,
    set_researcher_enabled,
    set_researcher_trigger,
)
from analysis.backtest import RSIReactionBacktest
from analysis.chart_structure import ChartStructureSnapshot, SRLevel, SRLevelsResult
from analysis.technical import TechnicalStats
from data.fundamentals_source import EquityFundamentals
from data.macro_source import CountryFiscalIndicators
from data.mt5_source import MarketAsset


# --- _resolve_ftmo_yahoo_ticker: real Yahoo conventions per real category ---


def test_resolve_ftmo_yahoo_ticker_forex():
    assert _resolve_ftmo_yahoo_ticker("EURUSD", "Forex") == "EURUSD=X"


def test_resolve_ftmo_yahoo_ticker_exotics():
    assert _resolve_ftmo_yahoo_ticker("USDCNH", "Exotics") == "USDCNH=X"


def test_resolve_ftmo_yahoo_ticker_metals():
    assert _resolve_ftmo_yahoo_ticker("XAUUSD", "Metals CFD") == "XAUUSD=X"


def test_resolve_ftmo_yahoo_ticker_crypto():
    assert _resolve_ftmo_yahoo_ticker("BTCUSD", "Crypto I CFD") == "BTC-USD"
    assert _resolve_ftmo_yahoo_ticker("ETHUSD", "Crypto I CFD") == "ETH-USD"


def test_resolve_ftmo_yahoo_ticker_crypto_without_usd_suffix_is_unmapped():
    # Deliberately conservative -- only the well-known "<BASE>USD" shape
    # is transformed; anything else is an honest skip, never a guess.
    assert _resolve_ftmo_yahoo_ticker("BTCEUR", "Crypto I CFD") is None


def test_resolve_ftmo_yahoo_ticker_equities():
    assert _resolve_ftmo_yahoo_ticker("NVDA", "Equities I CFD") == "NVDA"
    assert _resolve_ftmo_yahoo_ticker("INTC", "Equities I CFD") == "INTC"


def test_resolve_ftmo_yahoo_ticker_unmapped_categories():
    for category in ("Commodities", "Agriculture", "Cash CFD", "Uncategorized"):
        assert _resolve_ftmo_yahoo_ticker("SOMESYMBOL", category) is None


# --- _symbol_relevance_keywords / _filter_relevant_news: deterministic anti-noise filter ---


def test_symbol_relevance_keywords_forex_pair_includes_both_currencies():
    keywords = _symbol_relevance_keywords("GBPUSD", "Forex")
    assert "gbp" in keywords
    assert "pound" in keywords
    assert "usd" in keywords
    assert "dollar" in keywords


def test_symbol_relevance_keywords_metals_includes_gold_and_usd():
    keywords = _symbol_relevance_keywords("XAUUSD", "Metals CFD")
    assert "gold" in keywords
    assert "dollar" in keywords


def test_symbol_relevance_keywords_crypto_includes_coin_and_usd():
    keywords = _symbol_relevance_keywords("BTCUSD", "Crypto I CFD")
    assert "bitcoin" in keywords
    assert "dollar" in keywords


def test_symbol_relevance_keywords_equities_falls_back_to_bare_symbol():
    assert _symbol_relevance_keywords("NVDA", "Equities I CFD") == ["NVDA"]


def test_filter_relevant_news_drops_a_real_but_irrelevant_item():
    # Real, live-observed defect fixed 2026-09-15/16: Yahoo's own
    # ticker-news search for GBPUSD=X returned genuine cocoa/Ghana
    # commodity articles among its results -- not hallucinated by any
    # model, a real upstream data-relevance gap.
    news_items = [
        {"title": "Cocoa Prices Settle Higher on Ghana Supply Concerns", "summary": "", "source": "Barchart"},
        {"title": "US Dollar Price Forecast: Fed Hike Odds Lift DXY as GBP/USD Weakens", "summary": "", "source": "FX Empire"},
    ]
    filtered = _filter_relevant_news(news_items, _symbol_relevance_keywords("GBPUSD", "Forex"))
    assert len(filtered) == 1
    assert "Dollar" in filtered[0]["title"]


def test_filter_relevant_news_matches_dxy_as_a_real_dollar_keyword():
    news_items = [{"title": "DXY Stays Weak as ECB Hike Looms", "summary": "", "source": "FX Empire"}]
    filtered = _filter_relevant_news(news_items, _symbol_relevance_keywords("GBPUSD", "Forex"))
    assert len(filtered) == 1


def test_filter_relevant_news_falls_back_to_original_when_everything_would_be_dropped():
    # Safer to keep possibly-noisy real items than to report zero news
    # for a symbol that DID have real items fetched.
    news_items = [{"title": "Some unrelated headline", "summary": "", "source": "X"}]
    filtered = _filter_relevant_news(news_items, ["gbp", "pound"])
    assert filtered == news_items


def test_filter_relevant_news_empty_keywords_returns_original():
    news_items = [{"title": "Anything", "summary": "", "source": "X"}]
    assert _filter_relevant_news(news_items, []) == news_items


# --- parse_researcher_sentiment: defensive SENTIMENT: line extraction ---


def test_parse_researcher_sentiment_bullish():
    assert parse_researcher_sentiment("Some synthesis.\nSENTIMENT: BULLISH") == "BULLISH"


def test_parse_researcher_sentiment_bearish():
    assert parse_researcher_sentiment("Some synthesis.\nSENTIMENT: BEARISH") == "BEARISH"


def test_parse_researcher_sentiment_case_insensitive():
    assert parse_researcher_sentiment("sentiment: neutral") == "NEUTRAL"


def test_parse_researcher_sentiment_last_occurrence_wins():
    text = "SENTIMENT: BULLISH\nOn reflection, actually...\nSENTIMENT: BEARISH"
    assert parse_researcher_sentiment(text) == "BEARISH"


def test_parse_researcher_sentiment_none_when_absent():
    assert parse_researcher_sentiment("Just a synthesis with no verdict line.") is None


def test_parse_researcher_sentiment_matches_the_saved_files_own_bolded_field():
    # Real bug, caught live 2026-09-15: app.py's own UI badge calls this
    # on the SAVED report file (latest_research_report's own output),
    # which only ever contains the bolded "**Sentiment:** TAG" structured
    # field -- the raw "SENTIMENT: TAG" line is deliberately stripped out
    # of the saved copy by save_research_report to avoid showing it
    # twice. Every symbol's badge silently fell back to "unknown"
    # regardless of its real sentiment until this matched too.
    saved_file_text = (
        "# Research Report — INTC — 2026-09-15 01:45:00 UTC\n\n"
        "**Sentiment:** BEARISH\n\n"
        "## Research Note\n\nSome real synthesis text.\n"
    )
    assert parse_researcher_sentiment(saved_file_text) == "BEARISH"


# --- _fmt: None-safe numeric formatting ---


def test_fmt_none_is_na():
    assert _fmt(None) == "n/a"


def test_fmt_real_number():
    assert _fmt(12.345, digits=1) == "12.3"


# --- _format_news_block: real title/summary/source, never a fabricated item ---


def test_format_news_block_includes_title_summary_source():
    block = _format_news_block(
        [{"title": "Real headline", "summary": "Real summary text.", "source": "Reuters", "published": ""}]
    )
    assert "Real headline" in block
    assert "Real summary text." in block
    assert "Reuters" in block


def test_format_news_block_empty_is_none_placeholder():
    assert _format_news_block([]) == "(none)"


# --- _parse_published / _relative_age / news recency: real timestamp shapes ---


def test_parse_published_handles_yahoo_iso_with_z_suffix():
    dt = _parse_published("2026-09-14T00:00:00Z")
    assert dt == datetime(2026, 9, 14, 0, 0, 0, tzinfo=timezone.utc)


def test_parse_published_handles_rfc822_google_news_format():
    dt = _parse_published("Fri, 11 Sep 2026 07:00:00 GMT")
    assert dt == datetime(2026, 9, 11, 7, 0, 0, tzinfo=timezone.utc)


def test_parse_published_handles_rfc822_with_numeric_offset():
    dt = _parse_published("Mon, 14 Sep 2026 08:07:16 +0000")
    assert dt == datetime(2026, 9, 14, 8, 7, 16, tzinfo=timezone.utc)


def test_parse_published_handles_investing_com_shape():
    dt = _parse_published("Sep 11, 2026 19:01 GMT")
    assert dt == datetime(2026, 9, 11, 19, 1, 0, tzinfo=timezone.utc)


def test_parse_published_none_on_empty_or_unrecognized():
    assert _parse_published("") is None
    assert _parse_published("not a date at all") is None


def test_relative_age_minutes():
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert _relative_age("2026-09-15T11:55:00Z", now) == "5m ago"


def test_relative_age_hours():
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert _relative_age("2026-09-15T09:00:00Z", now) == "3h ago"


def test_relative_age_days():
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert _relative_age("2026-09-12T12:00:00Z", now) == "3d ago"


def test_relative_age_empty_when_unparseable():
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert _relative_age("garbage", now) == ""


def test_relative_age_empty_when_in_the_future():
    # Real, if rare, possibility (clock skew) -- never print a
    # nonsensical negative age.
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert _relative_age("2026-09-15T12:05:00Z", now) == ""


def test_format_news_block_includes_real_age():
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    block = _format_news_block(
        [{"title": "Real headline", "summary": "", "source": "Reuters", "published": "2026-09-15T10:00:00Z"}],
        now_utc=now,
    )
    assert "Reuters, 2h ago" in block


def test_format_news_block_omits_age_when_published_missing():
    block = _format_news_block([{"title": "Real headline", "summary": "", "source": "Reuters", "published": ""}])
    assert "- Real headline — Reuters" in block
    assert "ago" not in block


# --- _google_news_query / _fetch_news_with_fallback: Yahoo-then-Google ---


def test_google_news_query_forex():
    assert _google_news_query("USDCHF", "Forex") == "USDCHF forex"


def test_google_news_query_metals():
    assert _google_news_query("XAUUSD", "Metals CFD") == "XAUUSD price"


def test_google_news_query_crypto():
    assert _google_news_query("BTCUSD", "Crypto I CFD") == "BTC crypto"


def test_google_news_query_equities():
    assert _google_news_query("NVDA", "Equities I CFD") == "NVDA stock"


@patch("ai.researcher.fetch_google_news")
@patch("ai.researcher.fetch_recent_news")
def test_fetch_news_with_fallback_prefers_yahoo_when_available(mock_yahoo, mock_google):
    mock_yahoo.return_value = [{"title": "Real Yahoo item", "summary": "", "source": "", "published": ""}]
    result = _fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=5)
    assert result == [{"title": "Real Yahoo item", "summary": "", "source": "", "published": ""}]
    mock_google.assert_not_called()


@patch("ai.researcher.fetch_google_news")
@patch("ai.researcher.fetch_recent_news", return_value=[])
def test_fetch_news_with_fallback_uses_google_when_yahoo_empty(mock_yahoo, mock_google):
    mock_google.return_value = [{"title": "Real Google item", "summary": "", "source": "", "published": ""}]
    result = _fetch_news_with_fallback("USDCHF=X", "USDCHF forex", limit=5)
    assert result == [{"title": "Real Google item", "summary": "", "source": "", "published": ""}]
    mock_google.assert_called_once_with("USDCHF forex", limit=5)


# --- _category_rss_feed_url / _fetch_category_news_block: extra per-category coverage ---


def test_category_rss_feed_url_forex_and_exotics():
    assert _category_rss_feed_url("Forex") == "https://www.fxstreet.com/rss/news"
    assert _category_rss_feed_url("Exotics") == "https://www.fxstreet.com/rss/news"


def test_category_rss_feed_url_crypto_prefix_match():
    assert _category_rss_feed_url("Crypto I CFD") == "https://www.coindesk.com/arc/outboundfeeds/rss/"


def test_category_rss_feed_url_metals_prefix_match():
    assert _category_rss_feed_url("Metals CFD") == "https://www.investing.com/rss/commodities.rss"


def test_category_rss_feed_url_equities_has_no_specialty_feed():
    # A real, live-confirmed gap: no free, no-key, genuinely equities-only
    # feed was found working (MarketWatch's top-stories turned out to be
    # general/lifestyle content, not market news) -- honestly None, not a
    # guess.
    assert _category_rss_feed_url("Equities I CFD") is None


@patch("ai.researcher.fetch_rss_feed")
def test_fetch_category_news_block_uses_the_real_feed_for_the_category(mock_rss):
    mock_rss.return_value = [{"title": "Real FXStreet item", "summary": "", "source": "FXStreet", "published": ""}]
    block = _fetch_category_news_block("Forex", limit=5)
    assert "Real FXStreet item" in block
    mock_rss.assert_called_once_with("https://www.fxstreet.com/rss/news", limit=5)


def test_fetch_category_news_block_honest_note_when_no_feed_exists():
    block = _fetch_category_news_block("Equities I CFD", limit=5)
    assert "no category-specialty feed" in block


# --- _format_fiscal_snapshot / _build_macro_snapshot_block: real fiscal data layer ---


def test_format_fiscal_snapshot_includes_only_reported_fields():
    indicators = [
        CountryFiscalIndicators(
            country="United States", debt_to_gdp_pct=115.8, fiscal_balance_pct=None, current_account_pct=-3.6
        ),
        # World Bank genuinely has nothing at all for this one -- must be
        # skipped entirely, not printed as an empty "- Nowhere:" line.
        CountryFiscalIndicators(
            country="Nowhere", debt_to_gdp_pct=None, fiscal_balance_pct=None, current_account_pct=None
        ),
    ]
    block = _format_fiscal_snapshot(indicators)
    assert "United States: debt/GDP 115.8%, current account -3.6% of GDP" in block
    assert "Nowhere" not in block


def test_format_fiscal_snapshot_empty_list_is_empty_string():
    assert _format_fiscal_snapshot([]) == ""


@patch("ai.researcher.fetch_country_fiscal_indicators")
@patch("ai.researcher.build_macro_snapshot", return_value="Macro snapshot:\n- US Treasury yields: 3M 5.00%")
def test_build_macro_snapshot_block_appends_real_fiscal_data(mock_macro, mock_fiscal):
    mock_fiscal.return_value = [
        CountryFiscalIndicators(
            country="Japan", debt_to_gdp_pct=None, fiscal_balance_pct=None, current_account_pct=4.9
        )
    ]
    block = _build_macro_snapshot_block()
    assert "US Treasury yields: 3M 5.00%" in block
    assert "Japan: current account +4.9% of GDP" in block


@patch("ai.researcher.fetch_country_fiscal_indicators", return_value=[])
@patch("ai.researcher.build_macro_snapshot", return_value="Macro snapshot:\n- US Treasury yields: 3M 5.00%")
def test_build_macro_snapshot_block_no_fiscal_section_when_nothing_found(mock_macro, mock_fiscal):
    block = _build_macro_snapshot_block()
    assert "Fiscal/external-balance" not in block


@patch("ai.researcher.fetch_country_fiscal_indicators", side_effect=RuntimeError("World Bank API down"))
@patch("ai.researcher.build_macro_snapshot", return_value="Macro snapshot:\n- US Treasury yields: 3M 5.00%")
def test_build_macro_snapshot_block_degrades_honestly_when_fiscal_fetch_fails(mock_macro, mock_fiscal):
    # A fiscal-fetch failure must not take down the (already-working)
    # macro snapshot above it.
    block = _build_macro_snapshot_block()
    assert "US Treasury yields: 3M 5.00%" in block
    assert "Fiscal/external-balance" not in block


# --- _build_technical_snapshot: real fields, condensed, never fabricated ---


def _analysis_with_stats(h1_trend="uptrend") -> object:
    @dataclass
    class _FakeBase:
        stats: TechnicalStats
        rsi_oversold_backtest: object = None
        rsi_overbought_backtest: object = None
        support_resistance_backtest: object = None

    @dataclass
    class _FakeAnalysis:
        base: _FakeBase
        h4_stats: TechnicalStats
        h1_stats: TechnicalStats
        h1_structure: ChartStructureSnapshot

    flat = TechnicalStats(*([None] * 17))
    d1_stats = replace(flat, trend="uptrend", market_regime="trending_up", rsi=60.0, atr_pct=1.2, change_1m_pct=3.0)
    h1_stats = replace(flat, trend=h1_trend, market_regime="trending_up", rsi=55.0, atr_pct=0.4, change_1m_pct=1.0)
    structure = ChartStructureSnapshot(
        fibonacci=None,
        sr_levels=SRLevelsResult(
            resistance_levels=[SRLevel(price=1.10, touches=3, distance_pct=0.5)],
            support_levels=[SRLevel(price=1.08, touches=4, distance_pct=-0.5)],
        ),
        trendlines=None,
        patterns=[],
    )
    return _FakeAnalysis(base=_FakeBase(stats=d1_stats), h4_stats=d1_stats, h1_stats=h1_stats, h1_structure=structure)


def test_build_technical_snapshot_includes_real_trend_and_sr_levels():
    snapshot = _build_technical_snapshot(_analysis_with_stats())
    assert "uptrend" in snapshot
    assert "Nearest H1 support" in snapshot
    assert "Nearest H1 resistance" in snapshot


def test_build_technical_snapshot_includes_backtest_when_present():
    analysis = _analysis_with_stats()
    analysis.base.rsi_oversold_backtest = RSIReactionBacktest(
        condition="oversold", threshold=30.0, trades=10, wins=6, losses=4, timeouts=0,
        win_rate_pct=60.0, avg_r_multiple=0.8, stop_atr_multiple=1.0, target_atr_multiple=2.0, max_holding_bars=10,
    )
    snapshot = _build_technical_snapshot(analysis)
    assert "60% win rate" in snapshot


def test_build_technical_snapshot_degrades_honestly_when_no_history():
    flat = TechnicalStats(*([None] * 17))
    empty_structure = ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])

    @dataclass
    class _FakeBase:
        stats: TechnicalStats
        rsi_oversold_backtest: object = None
        rsi_overbought_backtest: object = None
        support_resistance_backtest: object = None

    @dataclass
    class _FakeAnalysis:
        base: _FakeBase
        h4_stats: TechnicalStats
        h1_stats: TechnicalStats
        h1_structure: ChartStructureSnapshot

    analysis = _FakeAnalysis(base=_FakeBase(stats=flat), h4_stats=flat, h1_stats=flat, h1_structure=empty_structure)
    assert "not enough real MT5 history" in _build_technical_snapshot(analysis)


# --- _format_equity_fundamentals: real analyst/earnings context, honest degrade ---


def test_format_equity_fundamentals_includes_real_fields():
    fundamentals = EquityFundamentals(
        analyst_target_mean_price=327.18,
        analyst_recommendation="strong_buy",
        analyst_count=58,
        forward_pe=13.5,
        next_earnings_date="2026-11-18",
        earnings_eps_estimate=2.47,
    )
    block = _format_equity_fundamentals(fundamentals)
    assert "strong buy (58 analysts)" in block
    assert "327.18" in block
    assert "13.5" in block
    assert "2026-11-18" in block
    assert "2.47" in block


def test_format_equity_fundamentals_omits_missing_fields_only():
    fundamentals = EquityFundamentals(
        analyst_target_mean_price=None,
        analyst_recommendation="buy",
        analyst_count=None,
        forward_pe=None,
        next_earnings_date=None,
        earnings_eps_estimate=None,
    )
    block = _format_equity_fundamentals(fundamentals)
    assert "buy" in block
    assert "price target" not in block.lower()
    assert "earnings date" not in block.lower()


def test_format_equity_fundamentals_none_when_unavailable():
    assert "no real equity fundamentals" in _format_equity_fundamentals(None)


def test_format_equity_fundamentals_none_when_everything_missing():
    fundamentals = EquityFundamentals(None, None, None, None, None, None)
    assert "no real equity fundamentals" in _format_equity_fundamentals(fundamentals)


# --- _extract_section / _extract_price_anchors / _check_price_grounding: deterministic anti-hallucination guard ---


def test_extract_section_finds_the_named_heading():
    text = "### Bull Case\nSome bull text.\n\n### Bear Case\nSome bear text.\n\nSENTIMENT: NEUTRAL"
    assert _extract_section(text, "Bull Case") == "Some bull text."
    assert _extract_section(text, "Bear Case") == "Some bear text."


def test_extract_section_empty_when_heading_absent():
    assert _extract_section("no headings here", "Bull Case") == ""


def test_extract_price_anchors_includes_current_price_support_and_resistance():
    structure = ChartStructureSnapshot(
        fibonacci=None,
        sr_levels=SRLevelsResult(
            resistance_levels=[SRLevel(price=1.3524, touches=12, distance_pct=0.38)],
            support_levels=[SRLevel(price=1.3400, touches=6, distance_pct=-0.5)],
        ),
        trendlines=None,
        patterns=[],
    )

    @dataclass
    class _FakeAnalysis:
        h1_structure: ChartStructureSnapshot

    anchors = _extract_price_anchors(_FakeAnalysis(h1_structure=structure), current_price=1.3450)
    assert anchors == [1.3450, 1.3400, 1.3524]


def test_extract_price_anchors_handles_missing_sr_levels():
    empty_structure = ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])

    @dataclass
    class _FakeAnalysis:
        h1_structure: ChartStructureSnapshot

    anchors = _extract_price_anchors(_FakeAnalysis(h1_structure=empty_structure), current_price=1.3450)
    assert anchors == [1.3450]


def test_check_price_grounding_none_when_a_real_anchor_is_cited():
    report = "### Price Action Hypothesis\nExpect a test of 1.3524 resistance.\n\nSENTIMENT: BEARISH"
    assert _check_price_grounding(report, [1.3450, 1.3400, 1.3524]) is None


def test_check_price_grounding_none_when_within_one_percent_tolerance():
    # 1.3520 vs the real 1.3524 anchor -- close enough to be a real
    # rounding difference, not a fabrication.
    report = "### Price Action Hypothesis\nExpect a test of 1.3520.\n\nSENTIMENT: BEARISH"
    assert _check_price_grounding(report, [1.3450, 1.3400, 1.3524]) is None


def test_check_price_grounding_flags_a_fabricated_level():
    # Real, live-observed case: qwen3:8b cited 1.3200 with no basis in
    # the real data during the audited GBPUSD comparison.
    report = "### Price Action Hypothesis\nExpect a move to 1.3200.\n\nSENTIMENT: BEARISH"
    note = _check_price_grounding(report, [1.3450, 1.3400, 1.3524])
    assert note is not None
    assert "could not be matched" in note


def test_check_price_grounding_none_when_no_anchors_available():
    report = "### Price Action Hypothesis\nExpect a move to 1.3200.\n\nSENTIMENT: BEARISH"
    assert _check_price_grounding(report, []) is None


def test_check_price_grounding_none_when_no_numbers_cited():
    report = "### Price Action Hypothesis\nExpect further weakness.\n\nSENTIMENT: BEARISH"
    assert _check_price_grounding(report, [1.3450]) is None


def test_check_price_grounding_none_when_section_missing():
    report = "### Bull Case\nSome text.\n\nSENTIMENT: NEUTRAL"
    assert _check_price_grounding(report, [1.3450]) is None


# --- calibration ledger / track record: real self-calibration ---


def test_record_and_build_track_record_scores_correct_bullish_call(tmp_path):
    ledger_path = tmp_path / "calibration.json"
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    with patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(ledger_path)):
        # A BULLISH call made 2 full intervals ago at 1.1000 -- price is
        # now higher, so it should score as correct.
        called_at = now - timedelta(hours=2 * config.RESEARCHER_CALIBRATION_MIN_AGE_HOURS)
        _record_calibration_entry("EURUSD", "BULLISH", 1.1000, called_at)
        block = _build_track_record_block("EURUSD", 1.1050, now)
    assert "1/1" in block


def test_record_and_build_track_record_scores_wrong_bearish_call(tmp_path):
    ledger_path = tmp_path / "calibration.json"
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    with patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(ledger_path)):
        called_at = now - timedelta(hours=2 * config.RESEARCHER_CALIBRATION_MIN_AGE_HOURS)
        _record_calibration_entry("EURUSD", "BEARISH", 1.1000, called_at)
        block = _build_track_record_block("EURUSD", 1.1050, now)
    assert "0/1" in block


def test_track_record_excludes_neutral_calls(tmp_path):
    ledger_path = tmp_path / "calibration.json"
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    with patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(ledger_path)):
        called_at = now - timedelta(hours=2 * config.RESEARCHER_CALIBRATION_MIN_AGE_HOURS)
        _record_calibration_entry("EURUSD", "NEUTRAL", 1.1000, called_at)
        block = _build_track_record_block("EURUSD", 1.1050, now)
    assert "old enough to judge" in block


def test_track_record_excludes_calls_younger_than_the_min_age(tmp_path):
    ledger_path = tmp_path / "calibration.json"
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    with patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(ledger_path)):
        # Called only 5 minutes ago -- not enough time has passed to
        # fairly judge it yet.
        called_at = now - timedelta(minutes=5)
        _record_calibration_entry("EURUSD", "BULLISH", 1.1000, called_at)
        block = _build_track_record_block("EURUSD", 1.2000, now)
    assert "old enough to judge" in block


def test_track_record_empty_ledger_says_no_calls_yet(tmp_path):
    ledger_path = tmp_path / "calibration.json"
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    with patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(ledger_path)):
        block = _build_track_record_block("EURUSD", 1.1050, now)
    assert "old enough to judge" in block


def test_calibration_ledger_never_mixes_up_symbols(tmp_path):
    ledger_path = tmp_path / "calibration.json"
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    called_at = now - timedelta(hours=2 * config.RESEARCHER_CALIBRATION_MIN_AGE_HOURS)
    with patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(ledger_path)):
        _record_calibration_entry("EURUSD", "BULLISH", 1.1000, called_at)
        _record_calibration_entry("GBPUSD", "BEARISH", 1.3000, called_at)
        eur_block = _build_track_record_block("EURUSD", 1.1050, now)
        gbp_block = _build_track_record_block("GBPUSD", 1.3050, now)
    assert "1/1" in eur_block  # BULLISH, price up -> correct
    assert "0/1" in gbp_block  # BEARISH, price up -> wrong


def test_calibration_ledger_trims_to_max_entries(tmp_path):
    ledger_path = tmp_path / "calibration.json"
    now = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    with (
        patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(ledger_path)),
        patch.object(config, "RESEARCHER_CALIBRATION_MAX_ENTRIES_PER_SYMBOL", 3),
    ):
        for i in range(5):
            _record_calibration_entry("EURUSD", "BULLISH", 1.1000 + i, now)
        ledger = _read_calibration_ledger()
    assert len(ledger["EURUSD"]) == 3


def test_calibration_ledger_read_returns_empty_dict_when_missing(tmp_path):
    with patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(tmp_path / "does_not_exist.json")):
        assert _read_calibration_ledger() == {}


def test_calibration_ledger_read_returns_empty_dict_on_corrupt_json(tmp_path):
    ledger_path = tmp_path / "calibration.json"
    ledger_path.write_text("not valid json", encoding="utf-8")
    with patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(ledger_path)):
        assert _read_calibration_ledger() == {}


# --- _run_researcher_model: primary/backup cascade ---


_NEWS_BLOCK = "- Some real headline\n  A real summary."
_CATEGORY_NEWS_BLOCK = "- Some real analyst-grade headline — FXStreet"
_MACRO_HEADLINES_BLOCK = "- [^GSPC] Some real macro headline"
_MACRO_SNAPSHOT_BLOCK = "Macro snapshot:\n- US Treasury yields: 3M 5.00%"
_TECHNICAL_BLOCK = "- H1: trend=uptrend, regime=trending_up, RSI=55.0, ATR%=0.4, 1-month change=1.0%"
_EQUITY_FUNDAMENTALS_BLOCK = "- Analyst consensus: strong buy (58 analysts)"
_TRACK_RECORD_BLOCK = "3/5 of this symbol's own past directional calls have been correct so far."


def _call_run_researcher_model(mock_run=None):
    return _run_researcher_model(
        "EURUSD", _NEWS_BLOCK, _CATEGORY_NEWS_BLOCK, _MACRO_HEADLINES_BLOCK, _MACRO_SNAPSHOT_BLOCK, _TECHNICAL_BLOCK,
        _EQUITY_FUNDAMENTALS_BLOCK, _TRACK_RECORD_BLOCK,
    )


@patch("ai.researcher.run_openrouter")
def test_run_researcher_model_uses_tier_one_when_available(mock_run):
    mock_run.return_value = "Real synthesis.\nSENTIMENT: BULLISH"
    result = _call_run_researcher_model()
    assert result == "Real synthesis.\nSENTIMENT: BULLISH"
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["model"] == AUDIT_MODELS[0][1]


@patch("ai.researcher.run_openrouter")
def test_run_researcher_model_falls_back_through_the_cascade_on_failure(mock_run):
    mock_run.side_effect = [OPENROUTER_FAILED_MESSAGE, "Tier-two synthesis.\nSENTIMENT: NEUTRAL"]
    result = _call_run_researcher_model()
    assert "Tier-two synthesis." in result
    assert AUDIT_MODELS[1][0] in result
    assert mock_run.call_count == 2


@patch("ai.researcher.run_openrouter", return_value=OPENROUTER_FAILED_MESSAGE)
def test_run_researcher_model_returns_failed_message_when_all_tiers_unavailable(mock_run):
    result = _call_run_researcher_model()
    assert result == OPENROUTER_FAILED_MESSAGE
    assert mock_run.call_count == len(AUDIT_MODELS)


@patch("ai.researcher.run_openrouter", return_value=OPENROUTER_MISSING_KEY_MESSAGE)
def test_run_researcher_model_returns_missing_key_message_when_every_tier_lacks_a_key(mock_run):
    result = _call_run_researcher_model()
    assert result == OPENROUTER_MISSING_KEY_MESSAGE
    assert mock_run.call_count == len(AUDIT_MODELS)


@patch("ai.researcher.run_openrouter")
def test_run_researcher_model_prompt_contains_only_the_real_blocks_given(mock_run):
    mock_run.return_value = "ok\nSENTIMENT: NEUTRAL"
    _call_run_researcher_model()
    prompt = mock_run.call_args.args[0]
    assert "Some real headline" in prompt
    assert "Some real analyst-grade headline" in prompt
    assert "Some real macro headline" in prompt
    assert "3M 5.00%" in prompt
    assert "trend=uptrend" in prompt
    assert "do not invent" in prompt.lower()
    assert "Price Action Hypothesis" in prompt
    assert "Bull Case" in prompt
    assert "Bear Case" in prompt


@patch("ai.researcher.run_openrouter")
def test_run_researcher_model_explicitly_instructs_weighing_fiscal_data(mock_run):
    mock_run.return_value = "ok\nSENTIMENT: NEUTRAL"
    _call_run_researcher_model()
    prompt = mock_run.call_args.args[0]
    assert "debt-to-GDP" in prompt
    assert "current-account" in prompt or "current account" in prompt


@patch("ai.researcher.run_openrouter")
def test_run_researcher_model_instructs_sentiment_must_match_its_own_reasoning(mock_run):
    mock_run.return_value = "ok\nSENTIMENT: NEUTRAL"
    _call_run_researcher_model()
    prompt = mock_run.call_args.args[0]
    assert "must match" in prompt.lower()
    assert "default hedge" in prompt.lower() or "not as a default" in prompt.lower()


@patch("ai.researcher.run_openrouter")
def test_run_researcher_model_instructs_flagging_conflicting_sources(mock_run):
    mock_run.return_value = "ok\nSENTIMENT: NEUTRAL"
    _call_run_researcher_model()
    prompt = mock_run.call_args.args[0]
    assert "conflict" in prompt.lower()


@patch("ai.researcher.run_openrouter")
def test_run_researcher_model_instructs_weighing_news_recency(mock_run):
    mock_run.return_value = "ok\nSENTIMENT: NEUTRAL"
    _call_run_researcher_model()
    prompt = mock_run.call_args.args[0]
    assert "stale" in prompt.lower() or "age" in prompt.lower()


@patch("ai.researcher.run_openrouter")
def test_run_researcher_model_includes_equity_fundamentals_and_track_record_blocks(mock_run):
    mock_run.return_value = "ok\nSENTIMENT: NEUTRAL"
    _call_run_researcher_model()
    prompt = mock_run.call_args.args[0]
    assert _EQUITY_FUNDAMENTALS_BLOCK in prompt
    assert _TRACK_RECORD_BLOCK in prompt


@patch("ai.researcher.run_openrouter")
def test_run_researcher_model_instructs_against_misattributing_macro_wide_items(mock_run):
    # Real, audited failure fixed 2026-09-15/16: qwen3:8b-with-thinking
    # misattributed broad macro-proxy headlines (Saudi pipeline, ECB, an
    # unrelated cocoa item) as if they were GBPUSD-specific drivers.
    mock_run.return_value = "ok\nSENTIMENT: NEUTRAL"
    _call_run_researcher_model()
    prompt = mock_run.call_args.args[0]
    assert "not specific to this symbol" in prompt.lower()


@patch("ai.researcher.run_openrouter")
def test_run_researcher_model_instructs_citing_only_real_price_numbers(mock_run):
    mock_run.return_value = "ok\nSENTIMENT: NEUTRAL"
    _call_run_researcher_model()
    prompt = mock_run.call_args.args[0]
    assert "exact real numbers" in prompt.lower()


# --- _strip_sentiment_line / _export_research_note: real Obsidian vault integration ---


def test_strip_sentiment_line_removes_only_the_sentiment_line():
    text = "Real synthesis line one.\nSENTIMENT: BULLISH\nReal synthesis line two."
    stripped = _strip_sentiment_line(text)
    assert "SENTIMENT" not in stripped
    assert "Real synthesis line one." in stripped
    assert "Real synthesis line two." in stripped


def test_export_research_note_writes_into_research_subfolder_with_wikilinks(tmp_path):
    _export_research_note("EURUSD", "Real synthesis.\nSENTIMENT: BULLISH", vault_path=tmp_path)
    note = tmp_path / "Research" / "EURUSD.md"
    assert note.exists()
    text = note.read_text(encoding="utf-8")
    assert "tags: [research, bullish]" in text
    assert "Real synthesis." in text
    assert "SENTIMENT" not in text.split("---\n\n", 1)[1]  # not duplicated in the body
    assert "[[EURUSD]]" in text
    assert "[[Research]]" in text


def test_export_research_note_overwrites_the_same_symbols_note_each_run(tmp_path):
    _export_research_note("EURUSD", "Older synthesis.\nSENTIMENT: NEUTRAL", vault_path=tmp_path)
    _export_research_note("EURUSD", "Newer synthesis.\nSENTIMENT: BEARISH", vault_path=tmp_path)
    text = (tmp_path / "Research" / "EURUSD.md").read_text(encoding="utf-8")
    assert "Newer synthesis." in text
    assert "Older synthesis." not in text


def test_export_research_note_never_raises_on_oserror(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    _export_research_note("EURUSD", "text\nSENTIMENT: NEUTRAL", vault_path=blocker)  # must not raise


# --- save_research_report / latest_research_report round trip ---


def test_save_research_report_writes_a_real_file(tmp_path):
    path = save_research_report(
        "EURUSD",
        "Synthesis.\nSENTIMENT: BULLISH",
        _NEWS_BLOCK,
        _CATEGORY_NEWS_BLOCK,
        _MACRO_HEADLINES_BLOCK,
        _MACRO_SNAPSHOT_BLOCK,
        _TECHNICAL_BLOCK,
        _EQUITY_FUNDAMENTALS_BLOCK,
        _TRACK_RECORD_BLOCK,
        tmp_path,
    )
    assert path is not None
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "EURUSD" in text
    assert "BULLISH" in text
    assert "Some real headline" in text
    assert "Some real analyst-grade headline" in text
    assert "Some real macro headline" in text
    assert "3M 5.00%" in text
    assert "trend=uptrend" in text
    assert "strong buy" in text
    assert "3/5 of this symbol's own" in text


def test_save_research_report_returns_none_on_oserror(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    result = save_research_report(
        "EURUSD", "text", "(none)", "(none)", "(none found this run)", "", "", "(not applicable)", "(no calls yet)",
        blocker,
    )
    assert result is None


def test_latest_research_report_empty_when_no_prior_reports(tmp_path):
    assert latest_research_report("EURUSD", tmp_path / "does_not_exist") is None


def test_latest_research_report_uses_only_the_newest_for_that_symbol(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "EURUSD_2026-09-01_010000.md").write_text("OLDER REPORT", encoding="utf-8")
    (tmp_path / "EURUSD_2026-09-03_010000.md").write_text("NEWEST REPORT", encoding="utf-8")
    text = latest_research_report("EURUSD", tmp_path)
    assert text == "NEWEST REPORT"


def test_latest_research_report_never_mixes_up_symbols(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "EURUSD_2026-09-03_010000.md").write_text("EURUSD REPORT", encoding="utf-8")
    (tmp_path / "USDCHF_2026-09-03_010000.md").write_text("USDCHF REPORT", encoding="utf-8")
    assert latest_research_report("EURUSD", tmp_path) == "EURUSD REPORT"
    assert latest_research_report("USDCHF", tmp_path) == "USDCHF REPORT"


# --- run_researcher_check: the real standalone pipeline, end to end ---


@patch("ai.researcher.fetch_country_fiscal_indicators", return_value=[])
@patch("ai.researcher.fetch_rss_feed", return_value=[])
@patch("ai.researcher.fetch_google_news", return_value=[])
@patch("ai.researcher.save_research_report")
@patch("ai.researcher._run_researcher_model", return_value="Synthesis.\nSENTIMENT: BULLISH")
@patch("ai.researcher._build_technical_snapshot", return_value=_TECHNICAL_BLOCK)
@patch("ai.researcher.analyze_ftmo_asset_live")
@patch("ai.researcher.build_macro_snapshot", return_value=_MACRO_SNAPSHOT_BLOCK)
@patch("ai.researcher.fetch_recent_news")
@patch("ai.researcher.get_symbol_category")
@patch("ai.researcher.get_market_watch")
@patch("ai.researcher.connect")
def test_run_researcher_check_saves_a_report_for_a_resolvable_symbol_with_news(
    mock_connect, mock_watch, mock_category, mock_news, mock_macro, mock_analyze, mock_tech, mock_model, mock_save,
    mock_google, mock_rss, mock_fiscal, tmp_path,
):
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.09, ask=1.0901)]
    mock_category.return_value = "Forex"

    def _news_side_effect(ticker, limit=8):
        if ticker == "EURUSD=X":
            return [{"title": "A real EURUSD headline", "summary": "", "source": "", "published": ""}]
        return []  # macro-proxy tickers: no macro headlines this run

    mock_news.side_effect = _news_side_effect

    with (
        patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(tmp_path / "researcher_calibration.json")),
    ):
        run_researcher_check()
        state = read_researcher_state()

    mock_news.assert_any_call("EURUSD=X", limit=config.RESEARCHER_HEADLINES_PER_SYMBOL)
    mock_save.assert_called_once()
    assert mock_save.call_args.args[0] == "EURUSD"
    assert "A real EURUSD headline" in mock_save.call_args.args[2]
    assert state["last_status"] == "success"
    # Non-equity symbols get an honest "not applicable" note, not a
    # real fetch attempt.
    assert "not applicable" in mock_save.call_args.args[7]


@patch("ai.researcher.fetch_country_fiscal_indicators", return_value=[])
@patch("ai.researcher.fetch_rss_feed", return_value=[])
@patch("ai.researcher.fetch_google_news", return_value=[])
@patch("ai.researcher.save_research_report")
@patch("ai.researcher._run_researcher_model", return_value="ok\nSENTIMENT: BULLISH")
@patch("ai.researcher._build_technical_snapshot", return_value=_TECHNICAL_BLOCK)
@patch("ai.researcher.analyze_ftmo_asset_live")
@patch("ai.researcher.fetch_equity_fundamentals")
@patch("ai.researcher.build_macro_snapshot", return_value=_MACRO_SNAPSHOT_BLOCK)
@patch("ai.researcher.fetch_recent_news")
@patch("ai.researcher.get_symbol_category", return_value="Equities I CFD")
@patch("ai.researcher.get_market_watch")
@patch("ai.researcher.connect")
def test_run_researcher_check_fetches_real_equity_fundamentals_for_equities(
    mock_connect, mock_watch, mock_category, mock_news, mock_macro, mock_fundamentals, mock_analyze, mock_tech,
    mock_model, mock_save, mock_google, mock_rss, mock_fiscal, tmp_path,
):
    mock_watch.return_value = [MarketAsset(symbol="NVDA", description="NVIDIA Corp", bid=180.0, ask=180.05)]
    mock_news.return_value = [{"title": "A real NVDA headline", "summary": "", "source": "", "published": ""}]
    mock_fundamentals.return_value = EquityFundamentals(327.18, "strong_buy", 58, 13.5, "2026-11-18", 2.47)

    with (
        patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(tmp_path / "researcher_calibration.json")),
    ):
        run_researcher_check()

    mock_fundamentals.assert_called_once_with("NVDA")
    mock_save.assert_called_once()
    assert "strong buy" in mock_save.call_args.args[7]
    assert "327.18" in mock_save.call_args.args[7]


@patch("ai.researcher.fetch_country_fiscal_indicators", return_value=[])
@patch("ai.researcher.fetch_rss_feed", return_value=[])
@patch("ai.researcher.fetch_google_news", return_value=[])
@patch("ai.researcher.save_research_report")
@patch("ai.researcher._build_technical_snapshot", return_value=_TECHNICAL_BLOCK)
@patch("ai.researcher.analyze_ftmo_asset_live")
@patch("ai.researcher.build_macro_snapshot", return_value=_MACRO_SNAPSHOT_BLOCK)
@patch("ai.researcher.fetch_recent_news")
@patch("ai.researcher.get_symbol_category", return_value="Forex")
@patch("ai.researcher.get_market_watch")
@patch("ai.researcher.connect")
def test_run_researcher_check_records_a_real_calibration_entry_on_success(
    mock_connect, mock_watch, mock_category, mock_news, mock_macro, mock_analyze, mock_tech, mock_save,
    mock_google, mock_rss, mock_fiscal, tmp_path,
):
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.09, ask=1.0901)]
    mock_news.return_value = [{"title": "A real headline", "summary": "", "source": "", "published": ""}]
    calibration_path = tmp_path / "researcher_calibration.json"

    with (
        patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(calibration_path)),
        patch("ai.researcher._run_researcher_model", return_value="ok\nSENTIMENT: BULLISH"),
    ):
        run_researcher_check()
        ledger = json.loads(calibration_path.read_text())

    assert ledger["EURUSD"][0]["sentiment"] == "BULLISH"
    assert ledger["EURUSD"][0]["price"] == pytest.approx(1.09005)


@patch("ai.researcher.fetch_country_fiscal_indicators", return_value=[])
@patch("ai.researcher.fetch_rss_feed", return_value=[])
@patch("ai.researcher.fetch_google_news", return_value=[])
@patch("ai.researcher.save_research_report")
@patch("ai.researcher._run_researcher_model")
@patch("ai.researcher.analyze_ftmo_asset_live")
@patch("ai.researcher.build_macro_snapshot", return_value=_MACRO_SNAPSHOT_BLOCK)
@patch("ai.researcher.fetch_recent_news", return_value=[])
@patch("ai.researcher.get_symbol_category")
@patch("ai.researcher.get_market_watch")
@patch("ai.researcher.connect")
def test_run_researcher_check_skips_a_symbol_with_no_known_yahoo_mapping(
    mock_connect, mock_watch, mock_category, mock_news, mock_macro, mock_analyze, mock_model, mock_save,
    mock_google, mock_rss, mock_fiscal, tmp_path,
):
    mock_watch.return_value = [MarketAsset(symbol="US500", description="Index", bid=6500.0, ask=6500.5)]
    mock_category.return_value = "Cash CFD"

    with (
        patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(tmp_path / "researcher_calibration.json")),
    ):
        run_researcher_check()

    mock_model.assert_not_called()
    mock_save.assert_not_called()
    mock_analyze.assert_not_called()


@patch("ai.researcher.fetch_country_fiscal_indicators", return_value=[])
@patch("ai.researcher.fetch_rss_feed", return_value=[])
@patch("ai.researcher.fetch_google_news", return_value=[])
@patch("ai.researcher.save_research_report")
@patch("ai.researcher._run_researcher_model")
@patch("ai.researcher.analyze_ftmo_asset_live")
@patch("ai.researcher.build_macro_snapshot", return_value=_MACRO_SNAPSHOT_BLOCK)
@patch("ai.researcher.fetch_recent_news", return_value=[])
@patch("ai.researcher.get_symbol_category", return_value="Forex")
@patch("ai.researcher.get_market_watch")
@patch("ai.researcher.connect")
def test_run_researcher_check_skips_a_symbol_with_no_real_news(
    mock_connect, mock_watch, mock_category, mock_news, mock_macro, mock_analyze, mock_model, mock_save,
    mock_google, mock_rss, mock_fiscal, tmp_path,
):
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.09, ask=1.0901)]

    with (
        patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(tmp_path / "researcher_calibration.json")),
    ):
        run_researcher_check()

    # Never fabricates a report when there's genuinely nothing real to report on
    # even after trying BOTH real news sources (Google is also tried for the
    # shared macro-proxy headlines above, hence assert_any_call rather than
    # assert_called_once).
    mock_google.assert_any_call("EURUSD forex", limit=config.RESEARCHER_HEADLINES_PER_SYMBOL)
    mock_model.assert_not_called()
    mock_save.assert_not_called()
    mock_analyze.assert_not_called()


@patch("ai.researcher.fetch_country_fiscal_indicators", return_value=[])
@patch("ai.researcher.fetch_rss_feed", return_value=[])
@patch("ai.researcher.fetch_google_news")
@patch("ai.researcher.save_research_report")
@patch("ai.researcher._run_researcher_model", return_value="ok\nSENTIMENT: NEUTRAL")
@patch("ai.researcher._build_technical_snapshot", return_value=_TECHNICAL_BLOCK)
@patch("ai.researcher.analyze_ftmo_asset_live")
@patch("ai.researcher.build_macro_snapshot", return_value=_MACRO_SNAPSHOT_BLOCK)
@patch("ai.researcher.fetch_recent_news")
@patch("ai.researcher.get_symbol_category")
@patch("ai.researcher.get_market_watch")
@patch("ai.researcher.connect")
def test_run_researcher_check_falls_back_to_google_news_when_yahoo_has_nothing(
    mock_connect, mock_watch, mock_category, mock_news, mock_macro, mock_analyze, mock_tech, mock_model, mock_save,
    mock_google, mock_rss, mock_fiscal, tmp_path,
):
    mock_watch.return_value = [MarketAsset(symbol="USDCHF", description="US Dollar vs Swiss Franc", bid=0.95, ask=0.9501)]
    mock_category.return_value = "Forex"
    mock_news.return_value = []  # Yahoo has nothing this run, same real gap observed live
    mock_google.return_value = [{"title": "A real Google News item", "summary": "", "source": "DailyForex", "published": ""}]

    with (
        patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(tmp_path / "researcher_calibration.json")),
    ):
        run_researcher_check()

    mock_google.assert_any_call("USDCHF forex", limit=config.RESEARCHER_HEADLINES_PER_SYMBOL)
    mock_save.assert_called_once()
    assert mock_save.call_args.args[0] == "USDCHF"
    assert "A real Google News item" in mock_save.call_args.args[2]


@patch("ai.researcher.fetch_country_fiscal_indicators", return_value=[])
@patch("ai.researcher.fetch_rss_feed", return_value=[])
@patch("ai.researcher.fetch_google_news", return_value=[])
@patch("ai.researcher.save_research_report")
@patch("ai.researcher._run_researcher_model", return_value="ok\nSENTIMENT: NEUTRAL")
@patch("ai.researcher._build_technical_snapshot", return_value=_TECHNICAL_BLOCK)
@patch("ai.researcher.analyze_ftmo_asset_live")
@patch("ai.researcher.build_macro_snapshot", return_value=_MACRO_SNAPSHOT_BLOCK)
@patch("ai.researcher.fetch_recent_news")
@patch("ai.researcher.get_symbol_category")
@patch("ai.researcher.get_market_watch")
@patch("ai.researcher.connect")
def test_run_researcher_check_one_symbols_failure_does_not_abort_the_run(
    mock_connect, mock_watch, mock_category, mock_news, mock_macro, mock_analyze, mock_tech, mock_model, mock_save,
    mock_google, mock_rss, mock_fiscal, tmp_path,
):
    # Real resilience requirement: a bad/unexpected failure on ONE symbol
    # (here, get_symbol_category raising) must never stop the rest of a
    # ~17-symbol run.
    mock_watch.return_value = [
        MarketAsset(symbol="BROKEN", description="", bid=1.0, ask=1.0),
        MarketAsset(symbol="EURUSD", description="Euro", bid=1.09, ask=1.0901),
    ]
    mock_category.side_effect = [RuntimeError("boom"), "Forex"]

    def _news_side_effect(ticker, limit=8):
        if ticker == "EURUSD=X":
            return [{"title": "A real headline", "summary": "", "source": "", "published": ""}]
        return []

    mock_news.side_effect = _news_side_effect

    with (
        patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(tmp_path / "researcher_calibration.json")),
    ):
        run_researcher_check()

    mock_save.assert_called_once()
    assert mock_save.call_args.args[0] == "EURUSD"


@patch("ai.researcher.fetch_country_fiscal_indicators", return_value=[])
@patch("ai.researcher.fetch_rss_feed", return_value=[])
@patch("ai.researcher.fetch_google_news", return_value=[])
@patch("ai.researcher.save_research_report")
@patch("ai.researcher._run_researcher_model", return_value="ok\nSENTIMENT: NEUTRAL")
@patch("ai.researcher.analyze_ftmo_asset_live", side_effect=RuntimeError("MT5 fetch failed"))
@patch("ai.researcher.build_macro_snapshot", return_value=_MACRO_SNAPSHOT_BLOCK)
@patch("ai.researcher.fetch_recent_news")
@patch("ai.researcher.get_symbol_category", return_value="Forex")
@patch("ai.researcher.get_market_watch")
@patch("ai.researcher.connect")
def test_run_researcher_check_still_saves_a_report_when_technical_snapshot_fetch_fails(
    mock_connect, mock_watch, mock_category, mock_news, mock_macro, mock_analyze, mock_model, mock_save,
    mock_google, mock_rss, mock_fiscal, tmp_path,
):
    # Real news was already found for this symbol -- a technical-snapshot
    # failure should note the gap plainly, not throw away the real news
    # data and skip the symbol entirely.
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.09, ask=1.0901)]

    def _news_side_effect(ticker, limit=8):
        if ticker == "EURUSD=X":
            return [{"title": "A real headline", "summary": "", "source": "", "published": ""}]
        return []

    mock_news.side_effect = _news_side_effect

    with (
        patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(tmp_path / "researcher_calibration.json")),
    ):
        run_researcher_check()

    mock_save.assert_called_once()
    technical_block_arg = mock_save.call_args.args[6]
    assert "unavailable" in technical_block_arg.lower()


@patch("ai.researcher.get_market_watch", return_value=[])
@patch("ai.researcher.connect")
def test_run_researcher_check_marks_no_assets_when_market_watch_empty(mock_connect, mock_watch, tmp_path):
    with (
        patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(tmp_path / "researcher_calibration.json")),
    ):
        run_researcher_check()
        state = read_researcher_state()
    assert state["last_status"] == "no_assets"


@patch("ai.researcher.get_market_watch", return_value=[])
@patch("ai.researcher.connect")
def test_run_researcher_check_total_runs_counter_increments_across_runs(mock_connect, mock_watch, tmp_path):
    with (
        patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "RESEARCHER_CALIBRATION_FILE", str(tmp_path / "researcher_calibration.json")),
    ):
        run_researcher_check()
        assert read_researcher_state()["total_runs"] == 1
        run_researcher_check()
        assert read_researcher_state()["total_runs"] == 2


# --- job-lifecycle helpers: enabled / trigger / due-check ---


def test_read_researcher_enabled_defaults_true_when_missing(tmp_path):
    with patch.object(config, "RESEARCHER_ENABLED_FILE", str(tmp_path / "researcher_enabled.json")):
        assert read_researcher_enabled() is True


def test_set_and_read_researcher_enabled_round_trip(tmp_path):
    with patch.object(config, "RESEARCHER_ENABLED_FILE", str(tmp_path / "researcher_enabled.json")):
        set_researcher_enabled(False)
        assert read_researcher_enabled() is False


@pytest.fixture
def _fixed_researcher_schedule(tmp_path):
    # Fixed, test-local trigger/grace/state-file values -- deliberately
    # different from production defaults so these tests stay isolated
    # from config.py ever changing the real once-daily schedule. Mirrors
    # tests/test_mega_analysis.py's own _fixed_schedule fixture.
    with (
        patch.object(config, "RESEARCHER_TRIGGER_HOUR_UTC", 12),
        patch.object(config, "RESEARCHER_TRIGGER_MINUTE_UTC", 0),
        patch.object(config, "RESEARCHER_GRACE_MINUTES", 60),
        patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")),
        patch.object(config, "RESEARCHER_TRIGGER_FILE", str(tmp_path / "researcher_trigger.json")),
    ):
        yield


def _utc(y, m, d, h, mi):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def test_read_researcher_trigger_falls_back_to_config_when_file_missing(_fixed_researcher_schedule):
    assert read_researcher_trigger() == (12, 0)


def test_set_researcher_trigger_then_read_round_trips(_fixed_researcher_schedule):
    set_researcher_trigger(15, 45)
    assert read_researcher_trigger() == (15, 45)


def test_read_researcher_trigger_falls_back_on_corrupt_file(_fixed_researcher_schedule, tmp_path):
    corrupt_path = tmp_path / "corrupt_trigger.json"
    corrupt_path.write_text("{not valid json")
    with patch.object(config, "RESEARCHER_TRIGGER_FILE", str(corrupt_path)):
        assert read_researcher_trigger() == (12, 0)


def test_read_researcher_trigger_falls_back_on_out_of_range_value(_fixed_researcher_schedule, tmp_path):
    bad_path = tmp_path / "bad_trigger.json"
    bad_path.write_text(json.dumps({"hour_utc": 25, "minute_utc": 0}))
    with patch.object(config, "RESEARCHER_TRIGGER_FILE", str(bad_path)):
        assert read_researcher_trigger() == (12, 0)


def test_is_researcher_due_false_before_trigger_time(_fixed_researcher_schedule):
    assert is_researcher_due(_utc(2026, 9, 14, 11, 59), state={}) is False


def test_is_researcher_due_true_at_trigger_instant(_fixed_researcher_schedule):
    assert is_researcher_due(_utc(2026, 9, 14, 12, 0), state={}) is True


def test_is_researcher_due_true_within_grace_window(_fixed_researcher_schedule):
    assert is_researcher_due(_utc(2026, 9, 14, 12, 45), state={}) is True


def test_is_researcher_due_false_outside_grace_window(_fixed_researcher_schedule):
    assert is_researcher_due(_utc(2026, 9, 14, 13, 1), state={}) is False


def test_is_researcher_due_false_once_already_run_today(_fixed_researcher_schedule):
    state = {"last_run_date_utc": "2026-09-14"}
    assert is_researcher_due(_utc(2026, 9, 14, 12, 5), state=state) is False


def test_is_researcher_due_true_again_next_day_after_yesterdays_run(_fixed_researcher_schedule):
    state = {"last_run_date_utc": "2026-09-14"}
    assert is_researcher_due(_utc(2026, 9, 15, 12, 5), state=state) is True


def test_next_researcher_check_utc_is_today_trigger_when_not_yet_due(_fixed_researcher_schedule):
    now = _utc(2026, 9, 14, 6, 0)
    assert next_researcher_check_utc(now, state={}) == _utc(2026, 9, 14, 12, 0)


def test_next_researcher_check_utc_is_tomorrow_when_already_ran_today(_fixed_researcher_schedule):
    now = _utc(2026, 9, 14, 13, 0)
    state = {"last_run_date_utc": "2026-09-14"}
    assert next_researcher_check_utc(now, state=state) == _utc(2026, 9, 15, 12, 0)


def test_next_researcher_check_utc_is_tomorrow_when_todays_window_already_passed(_fixed_researcher_schedule):
    now = _utc(2026, 9, 14, 20, 0)
    assert next_researcher_check_utc(now, state={}) == _utc(2026, 9, 15, 12, 0)


def test_write_researcher_state_marks_last_run_date_only_on_success(tmp_path):
    with patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")):
        from ai.researcher import _write_researcher_state

        _write_researcher_state("no_assets")
        assert read_researcher_state().get("last_run_date_utc") is None
        _write_researcher_state("success", "3/3 symbol(s) researched")
        assert read_researcher_state()["last_run_date_utc"] == datetime.now(timezone.utc).date().isoformat()


def test_write_researcher_state_keeps_prior_last_run_date_on_a_later_failure(tmp_path):
    with patch.object(config, "RESEARCHER_STATE_FILE", str(tmp_path / "researcher_state.json")):
        from ai.researcher import _write_researcher_state

        _write_researcher_state("success", "3/3 symbol(s) researched")
        today = read_researcher_state()["last_run_date_utc"]
        _write_researcher_state("no_assets")
        assert read_researcher_state()["last_run_date_utc"] == today
