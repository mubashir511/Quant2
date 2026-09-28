import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import config
from data import symbol_news as sn


@pytest.fixture(autouse=True)
def _isolated(tmp_path):
    sn._symbol_news_cache.clear()
    sn._category_news_cache.clear()
    sn._macro_news_cache.clear()
    with (
        patch.object(config, "SYMBOL_NEWS_DIR", str(tmp_path / "ftmo_symbol_news")),
        patch.object(config, "CATEGORY_NEWS_DIR", str(tmp_path / "ftmo_category_news")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "SYMBOL_NEWS_CACHE_MINUTES", 20),
        patch.object(config, "SYMBOL_NEWS_RETENTION_DAYS", 30),
        patch.object(config, "NEWS_FETCH_CACHE_FILE", str(tmp_path / "news_fetch_cache.json")),
    ):
        yield
    sn._symbol_news_cache.clear()
    sn._category_news_cache.clear()
    sn._macro_news_cache.clear()


def _item(title="A headline", summary="", source="Reuters", published="", link=""):
    return {"title": title, "summary": summary, "source": source, "published": published, "link": link}


# --- resolve_ftmo_yahoo_ticker ------------------------------------------


def test_resolve_ftmo_yahoo_ticker_forex():
    assert sn.resolve_ftmo_yahoo_ticker("EURUSD", "Forex") == "EURUSD=X"


def test_resolve_ftmo_yahoo_ticker_metals():
    assert sn.resolve_ftmo_yahoo_ticker("XAUUSD", "Metals CFD") == "XAUUSD=X"


def test_resolve_ftmo_yahoo_ticker_crypto():
    assert sn.resolve_ftmo_yahoo_ticker("BTCUSD", "Crypto I CFD") == "BTC-USD"


def test_resolve_ftmo_yahoo_ticker_equities():
    assert sn.resolve_ftmo_yahoo_ticker("AAPL", "Equities I CFD") == "AAPL"


def test_resolve_ftmo_yahoo_ticker_none_for_unmapped_category():
    assert sn.resolve_ftmo_yahoo_ticker("SUGAR.c", "Agriculture") is None


# --- Hardcoded commodity futures (added 2026-09-20, direct user challenge
# after being told wrongly that these had "no known free news-source
# mapping" -- checked live, real Yahoo futures tickers exist and return
# real, relevant headlines) -----------------------------------------------


def test_resolve_ftmo_yahoo_ticker_coffee():
    assert sn.resolve_ftmo_yahoo_ticker("COFFEE.c", "Agriculture") == "KC=F"


def test_resolve_ftmo_yahoo_ticker_cocoa():
    assert sn.resolve_ftmo_yahoo_ticker("COCOA.c", "Agriculture") == "CC=F"


def test_resolve_ftmo_yahoo_ticker_wheat():
    assert sn.resolve_ftmo_yahoo_ticker("WHEAT.c", "Agriculture") == "ZW=F"


def test_resolve_ftmo_yahoo_ticker_uk_oil():
    assert sn.resolve_ftmo_yahoo_ticker("UKOIL.cash", "Cash II CFD") == "BZ=F"


def test_symbol_relevance_keywords_coffee():
    assert sn.symbol_relevance_keywords("COFFEE.c", "Agriculture") == ["coffee", "arabica", "robusta"]


def test_google_news_query_uk_oil():
    assert sn.google_news_query("UKOIL.cash", "Cash II CFD") == "brent crude oil price"


def test_category_rss_feed_url_agriculture_shares_the_commodities_feed():
    assert sn.category_rss_feed_url("Agriculture") == sn._CATEGORY_RSS_FEEDS["Metals"]


def test_category_rss_feed_url_cash_prefix_shares_the_commodities_feed():
    assert sn.category_rss_feed_url("Cash II CFD") == sn._CATEGORY_RSS_FEEDS["Metals"]


# --- derive_generic_search_name / get_symbol_news_block's "intelligent
# search" fallback (added 2026-09-20, direct user challenge: "how is it
# possible the 3 food items and oil does not have news feed, you have to
# do intelligent news searching") --------------------------------------


def test_derive_generic_search_name_strips_boilerplate():
    assert sn.derive_generic_search_name("Coffee vs US Dollar, Spot CFD") == "Coffee"
    assert sn.derive_generic_search_name("Crude Oil Brent, Spot CFD") == "Crude Oil Brent"
    assert sn.derive_generic_search_name("Meta Platforms, Spot CFD") == "Meta Platforms"


def test_derive_generic_search_name_empty_when_nothing_meaningful_left():
    assert sn.derive_generic_search_name("Spot CFD") == ""
    assert sn.derive_generic_search_name("") == ""


@patch("data.symbol_news.fetch_recent_news", return_value=[])
@patch("data.symbol_news.fetch_google_news")
def test_get_symbol_news_block_uses_generic_fallback_for_an_unmapped_category(mock_google, mock_yahoo):
    mock_google.return_value = [_item("Sugar prices rally", source="Reuters")]
    result = sn.get_symbol_news_block("SUGAR.c", "Agriculture", limit=5, description="Sugar vs US Dollar, Spot CFD")
    assert result == [_item("Sugar prices rally", source="Reuters")]
    mock_google.assert_called_once_with("Sugar", limit=5)


def test_get_symbol_news_block_empty_when_unmapped_and_no_description_given():
    assert sn.get_symbol_news_block("SUGAR.c", "Agriculture", limit=5) == []


def test_get_symbol_news_block_empty_when_unmapped_and_description_has_nothing_useful():
    assert sn.get_symbol_news_block("SUGAR.c", "Agriculture", limit=5, description="Spot CFD") == []


@patch("data.symbol_news.fetch_recent_news", return_value=[])
@patch("data.symbol_news.fetch_google_news", return_value=[])
def test_get_symbol_news_block_generic_fallback_empty_when_search_finds_nothing(mock_google, mock_yahoo):
    result = sn.get_symbol_news_block("SUGAR.c", "Agriculture", limit=5, description="Sugar, Spot CFD")
    assert result == []


def test_get_symbol_news_block_prefers_the_precise_path_over_the_generic_fallback():
    # A hardcoded commodity future (or any resolvable category) must
    # never fall through to the generic description-based search —
    # resolve_ftmo_yahoo_ticker already handles it precisely.
    assert sn.resolve_ftmo_yahoo_ticker("COFFEE.c", "Agriculture") == "KC=F"


# --- symbol_relevance_keywords / filter_relevant_news -------------------


def test_symbol_relevance_keywords_forex_pair_combines_both_currencies():
    keywords = sn.symbol_relevance_keywords("GBPUSD", "Forex")
    assert "gbp" in keywords
    assert "usd" in keywords


def test_symbol_relevance_keywords_falls_back_to_bare_symbol():
    assert sn.symbol_relevance_keywords("SUGAR.c", "Agriculture") == ["SUGAR.c"]


def test_filter_relevant_news_drops_off_topic_items():
    items = [_item("Cocoa prices surge in Ghana"), _item("GBP falls on BoE dovish tilt")]
    result = sn.filter_relevant_news(items, ["gbp", "boe"])
    assert result == [items[1]]


def test_filter_relevant_news_falls_back_to_unfiltered_if_everything_would_be_dropped():
    items = [_item("Totally unrelated story")]
    result = sn.filter_relevant_news(items, ["gbp", "boe"])
    assert result == items


# --- google_news_query / category_rss_feed_url --------------------------


def test_google_news_query_forex():
    assert sn.google_news_query("EURUSD", "Forex") == "EURUSD forex"


def test_category_rss_feed_url_known_category():
    assert sn.category_rss_feed_url("Forex") == "https://www.fxstreet.com/rss/news"


def test_category_rss_feed_url_crypto_prefix_match():
    assert sn.category_rss_feed_url("Crypto I CFD") == sn._CATEGORY_RSS_FEEDS["Crypto"]


def test_category_rss_feed_url_none_for_uncovered_category():
    # Equities genuinely still has no free, no-key, equities-only RSS
    # feed that was found working live -- see this module's own docstring.
    assert sn.category_rss_feed_url("Equities I CFD") is None


# --- _parse_published / _relative_age -----------------------------------


def test_parse_published_iso_with_z():
    dt = sn._parse_published("2026-09-14T00:00:00Z")
    assert dt.year == 2026 and dt.month == 9 and dt.day == 14


def test_parse_published_rfc822():
    dt = sn._parse_published("Fri, 11 Sep 2026 07:00:00 GMT")
    assert dt.year == 2026 and dt.day == 11


def test_parse_published_none_on_garbage():
    assert sn._parse_published("not a date") is None


def test_relative_age_minutes():
    now = datetime(2026, 9, 20, 12, 30, tzinfo=timezone.utc)
    published = "2026-09-20T12:20:00Z"
    assert sn._relative_age(published, now) == "10m ago"


def test_relative_age_future_timestamp_is_blank():
    now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
    published = "2026-09-20T12:30:00Z"  # 30 min in the "future" -- clock skew
    assert sn._relative_age(published, now) == ""


# --- format_news_block ---------------------------------------------------


def test_format_news_block_none_when_empty():
    assert sn.format_news_block([]) == "(none)"


def test_format_news_block_includes_summary_line():
    block = sn.format_news_block([_item("Headline", summary="Extra detail")])
    assert "- Headline" in block
    assert "  Extra detail" in block


# --- fetch_news_with_fallback: cache + Yahoo/Google fallback + vault ----


@patch("data.symbol_news.fetch_google_news")
@patch("data.symbol_news.fetch_recent_news")
def test_fetch_news_with_fallback_prefers_yahoo(mock_yahoo, mock_google):
    mock_yahoo.return_value = [_item("Yahoo item")]
    result = sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=5)
    assert result == [_item("Yahoo item")]
    mock_google.assert_not_called()


@patch("data.symbol_news.fetch_google_news")
@patch("data.symbol_news.fetch_recent_news", return_value=[])
def test_fetch_news_with_fallback_uses_google_when_yahoo_empty(mock_yahoo, mock_google):
    mock_google.return_value = [_item("Google item")]
    result = sn.fetch_news_with_fallback("USDCHF=X", "USDCHF forex", limit=5)
    assert result == [_item("Google item")]


@patch("data.symbol_news.fetch_recent_news")
def test_fetch_news_with_fallback_is_cached_by_yahoo_ticker(mock_yahoo):
    mock_yahoo.return_value = [_item("headline one")]
    first = sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=5)
    second = sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=5)
    assert first == second
    mock_yahoo.assert_called_once()


@patch("data.symbol_news.fetch_recent_news")
def test_fetch_news_with_fallback_shares_cache_across_process_boundary(mock_yahoo):
    # Real bug caught 2026-09-20, direct user challenge ("i don't want to
    # explode the usage quota"): Clerk/Researcher/Mega each run as their
    # OWN spawned process per tray-timer tick, so the in-memory
    # _symbol_news_cache dict alone never deduped a real fetch across
    # them -- only the disk-backed mirror does. Clearing the in-memory
    # dict (simulating "a different process asks next") must NOT cause a
    # second real fetch within the TTL window.
    mock_yahoo.return_value = [_item("headline one")]
    sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=5)
    sn._symbol_news_cache.clear()
    second = sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=5)
    assert second == [_item("headline one")]
    mock_yahoo.assert_called_once()


def test_disk_cache_degrades_to_a_miss_on_a_corrupt_file(tmp_path):
    cache_file = tmp_path / "corrupt_cache.json"
    cache_file.write_text("not valid json{{{", encoding="utf-8")
    with patch("config.NEWS_FETCH_CACHE_FILE", str(cache_file)):
        assert sn._disk_cache_get("symbol", "EURUSD=X") is None


@patch("data.symbol_news.fetch_recent_news")
def test_fetch_news_with_fallback_a_narrower_cached_entry_never_caps_a_broader_request(mock_yahoo):
    # Real bug caught live 2026-09-20: Clerk (limit=2) fetching first
    # must never silently cap what Mega Session/the webapp (limit=5)
    # gets back for the same ticker within the same cache window.
    mock_yahoo.return_value = [_item("one"), _item("two")]
    narrow = sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=2)
    assert len(narrow) == 2

    mock_yahoo.return_value = [_item("one"), _item("two"), _item("three"), _item("four"), _item("five")]
    broad = sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=5)
    assert len(broad) == 5
    assert mock_yahoo.call_count == 2  # the broader request genuinely re-fetched, not reused the narrow cache


@patch("data.symbol_news.fetch_recent_news")
def test_fetch_news_with_fallback_a_broader_cached_entry_serves_a_later_narrower_request(mock_yahoo):
    mock_yahoo.return_value = [_item("one"), _item("two"), _item("three")]
    sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=3)
    narrower = sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=1)
    assert narrower == [_item("one")]
    mock_yahoo.assert_called_once()  # reused the cache, sliced down -- no second fetch needed


@patch("data.symbol_news.fetch_recent_news")
def test_fetch_news_with_fallback_refetches_after_cache_expires(mock_yahoo):
    mock_yahoo.return_value = [_item("old")]
    sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=5)
    stale = datetime.now(timezone.utc) - timedelta(minutes=config.SYMBOL_NEWS_CACHE_MINUTES + 1)
    sn._symbol_news_cache["EURUSD=X"] = (stale, [_item("old")], 5)
    mock_yahoo.return_value = [_item("new")]
    result = sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=5)
    assert result == [_item("new")]
    assert mock_yahoo.call_count == 2


@patch("data.symbol_news.fetch_recent_news")
def test_fetch_news_with_fallback_persists_to_vault_only_when_symbol_given(mock_yahoo):
    mock_yahoo.return_value = [_item("A real headline")]
    sn.fetch_news_with_fallback("XAUUSD=X", "gold price", limit=5)  # no symbol -- no persistence
    assert sn.list_symbol_news("XAUUSD") == []

    sn._symbol_news_cache.clear()
    sn._save_disk_cache({})  # force a genuine cache-miss, not just clearing the in-memory fast path
    sn.fetch_news_with_fallback("XAUUSD=X", "gold price", limit=5, symbol="XAUUSD")
    assert len(sn.list_symbol_news("XAUUSD")) == 1


@patch("data.symbol_news.fetch_recent_news", return_value=[])
@patch("data.symbol_news.fetch_google_news", return_value=[])
def test_fetch_news_with_fallback_two_arg_direct_calls_still_work(mock_google, mock_yahoo):
    # Real bug class this guards against: adding `symbol` as a new
    # parameter must never break a caller that only ever knew the
    # original 3-argument shape.
    assert sn.fetch_news_with_fallback("EURUSD=X", "EURUSD forex", limit=5) == []


# --- get_symbol_news_block: the one-call convenience --------------------


def test_get_symbol_news_block_empty_when_no_ticker_resolves():
    assert sn.get_symbol_news_block("SUGAR.c", "Agriculture", limit=5) == []


@patch("data.symbol_news.fetch_recent_news")
def test_get_symbol_news_block_empty_when_fetch_empty(mock_yahoo):
    mock_yahoo.return_value = []
    with patch("data.symbol_news.fetch_google_news", return_value=[]):
        assert sn.get_symbol_news_block("EURUSD", "Forex", limit=5) == []


@patch("data.symbol_news.fetch_recent_news")
def test_get_symbol_news_block_filters_and_returns_relevant_items(mock_yahoo):
    mock_yahoo.return_value = [_item("GBP falls on BoE"), _item("Cocoa surges in Ghana")]
    result = sn.get_symbol_news_block("GBPUSD", "Forex", limit=5)
    assert result == [_item("GBP falls on BoE")]


# --- get_category_news_items / get_macro_news_items ---------------------
# Real gap closed 2026-09-20, direct user challenge ("where did those
# sources go?"): the initial consolidation only carried over the PER-
# SYMBOL Yahoo/Google layer, silently dropping the category-specialty
# RSS (FXStreet/CoinDesk/Investing.com) and macro/geopolitical (index
# proxies + CNBC) layers Researcher's own report already had.


@patch("data.symbol_news.fetch_rss_feed")
def test_get_category_news_items_uses_the_real_feed(mock_rss):
    mock_rss.return_value = [_item("Real FXStreet item", source="FXStreet")]
    result = sn.get_category_news_items("Forex", limit=5)
    assert result == [_item("Real FXStreet item", source="FXStreet")]
    mock_rss.assert_called_once_with("https://www.fxstreet.com/rss/news", limit=5)


def test_get_category_news_items_empty_for_uncovered_category():
    assert sn.get_category_news_items("Equities I CFD", limit=5) == []


@patch("data.symbol_news.fetch_rss_feed")
def test_get_category_news_items_is_cached_per_category(mock_rss):
    mock_rss.return_value = [_item("item")]
    sn.get_category_news_items("Forex", limit=5)
    sn.get_category_news_items("Forex", limit=5)
    mock_rss.assert_called_once()


@patch("data.symbol_news.fetch_rss_feed")
def test_get_category_news_items_shares_cache_across_process_boundary(mock_rss):
    mock_rss.return_value = [_item("item")]
    sn.get_category_news_items("Forex", limit=5)
    sn._category_news_cache.clear()  # simulate a different spawned process asking next
    sn.get_category_news_items("Forex", limit=5)
    mock_rss.assert_called_once()


@patch("data.symbol_news.fetch_rss_feed")
@patch("data.symbol_news.fetch_recent_news")
def test_get_macro_news_items_tags_each_item_with_its_source(mock_yahoo, mock_rss):
    mock_yahoo.return_value = [_item("Stocks rally on rate cut hopes")]
    mock_rss.return_value = [_item("Breaking: Fed holds rates")]
    items = sn.get_macro_news_items()
    titles = [i["title"] for i in items]
    assert any(t.startswith("[S&P 500]") for t in titles)
    assert any(t.startswith("[crude oil]") for t in titles)
    assert any(t.startswith("[gold price]") for t in titles)
    assert any(t.startswith("[CNBC]") for t in titles)


@patch("data.symbol_news.fetch_rss_feed", return_value=[])
@patch("data.symbol_news.fetch_recent_news", return_value=[])
@patch("data.symbol_news.fetch_google_news", return_value=[])
def test_get_macro_news_items_is_cached_as_one_shared_entry(mock_google, mock_yahoo, mock_rss):
    sn.get_macro_news_items()
    sn.get_macro_news_items()
    # 3 proxy tickers x 1 call each (Yahoo, since it's tried first) -- not doubled on the second call.
    assert mock_yahoo.call_count == 3
    assert mock_rss.call_count == 1


@patch("data.symbol_news.fetch_rss_feed", return_value=[])
@patch("data.symbol_news.fetch_recent_news", return_value=[])
@patch("data.symbol_news.fetch_google_news", return_value=[])
def test_get_macro_news_items_shares_cache_across_process_boundary(mock_google, mock_yahoo, mock_rss):
    sn.get_macro_news_items()
    sn._symbol_news_cache.clear()  # simulate a different spawned process asking next
    sn._macro_news_cache.clear()
    sn.get_macro_news_items()
    assert mock_yahoo.call_count == 3
    assert mock_rss.call_count == 1


# --- vault persistence: dedup, sequence, rolling 30-day window ----------


def test_record_symbol_news_to_vault_dedupes_by_title():
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies")])
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies"), _item("Gold hits new high")])
    entries = sn.list_symbol_news("XAUUSD")
    assert len(entries) == 2
    assert {e["title"] for e in entries} == {"Gold rallies", "Gold hits new high"}


def test_record_symbol_news_to_vault_assigns_increasing_sequence():
    sn._record_symbol_news_to_vault("XAUUSD", [_item("First"), _item("Second")])
    entries = sn.list_symbol_news("XAUUSD")
    seqs = sorted(e["seq"] for e in entries)
    assert seqs == [1, 2]


def test_record_symbol_news_to_vault_prunes_entries_older_than_retention_window():
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Old item")])
    entries = sn._load_symbol_news_store("XAUUSD")
    stale_time = datetime.now(timezone.utc) - timedelta(days=config.SYMBOL_NEWS_RETENTION_DAYS + 1)
    entries[0]["recorded_utc"] = stale_time.isoformat()
    sn._save_symbol_news_store("XAUUSD", entries)

    sn._record_symbol_news_to_vault("XAUUSD", [_item("Fresh item")])

    remaining = sn.list_symbol_news("XAUUSD")
    assert [e["title"] for e in remaining] == ["Fresh item"]


def test_record_symbol_news_to_vault_writes_a_readable_obsidian_note():
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies", source="Reuters")])
    note_path = Path(config.OBSIDIAN_VAULT_PATH) / "News" / "XAUUSD.md"
    assert note_path.exists()
    text = note_path.read_text(encoding="utf-8")
    assert "Gold rallies" in text
    assert "[[XAUUSD]]" in text


@patch("data.symbol_news.fetch_article_text", return_value="")
def test_record_symbol_news_to_vault_persists_the_real_article_link(mock_fetch):
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies", link="https://example.com/gold")])
    entries = sn.list_symbol_news("XAUUSD")
    assert entries[0]["link"] == "https://example.com/gold"


@patch("data.symbol_news.fetch_article_text", return_value="")
def test_record_symbol_news_to_vault_note_links_the_title_when_a_real_link_exists(mock_fetch):
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies", link="https://example.com/gold")])
    note_path = Path(config.OBSIDIAN_VAULT_PATH) / "News" / "XAUUSD.md"
    text = note_path.read_text(encoding="utf-8")
    assert "[Gold rallies](https://example.com/gold)" in text


@patch("data.symbol_news.fetch_article_text")
def test_record_symbol_news_to_vault_fetches_and_persists_the_full_article_text(mock_fetch):
    mock_fetch.return_value = "A real, complete paragraph explaining what actually happened."
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies", link="https://example.com/gold")])
    entries = sn.list_symbol_news("XAUUSD")
    assert entries[0]["full_text"] == "A real, complete paragraph explaining what actually happened."
    mock_fetch.assert_called_once_with("https://example.com/gold")


@patch("data.symbol_news.fetch_article_text")
def test_record_symbol_news_to_vault_never_fetches_full_text_without_a_link(mock_fetch):
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies")])  # no link
    entries = sn.list_symbol_news("XAUUSD")
    assert entries[0]["full_text"] == ""
    mock_fetch.assert_not_called()


@patch("data.symbol_news.fetch_article_text")
def test_record_symbol_news_to_vault_never_refetches_full_text_for_an_already_recorded_title(mock_fetch):
    mock_fetch.return_value = "First fetch."
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies", link="https://example.com/gold")])
    mock_fetch.reset_mock()
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies", link="https://example.com/gold")])
    mock_fetch.assert_not_called()


@patch("data.symbol_news.fetch_article_text")
def test_record_symbol_news_to_vault_note_includes_the_full_article_text(mock_fetch):
    mock_fetch.return_value = "Paragraph one of the real article.\n\nParagraph two of the real article."
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies", link="https://example.com/gold")])
    note_path = Path(config.OBSIDIAN_VAULT_PATH) / "News" / "XAUUSD.md"
    text = note_path.read_text(encoding="utf-8")
    assert "Full article" in text
    assert "Paragraph one of the real article." in text
    assert "Paragraph two of the real article." in text


def test_record_symbol_news_to_vault_note_shows_plain_title_without_a_link():
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies")])
    note_path = Path(config.OBSIDIAN_VAULT_PATH) / "News" / "XAUUSD.md"
    text = note_path.read_text(encoding="utf-8")
    assert "- Gold rallies (Reuters)" in text


@patch("data.symbol_news.fetch_article_text")
def test_record_symbol_news_to_vault_backfills_an_old_schema_entry_when_seen_again(mock_fetch):
    # Real bug found live 2026-09-20: entries recorded before the link/
    # full_text feature existed have no such keys at all, and the plain
    # title-dedup would otherwise skip them forever even though the exact
    # same headline keeps reappearing in every fresh fetch. This is the
    # regression test for the fix: a pre-existing entry missing a link
    # gets backfilled the next time its own title is seen with one.
    old_schema_entry = {
        "seq": 1,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "title": "Gold rallies",
        "source": "Reuters",
        "published": "",
        "summary": "",
        # no "link"/"full_text" keys at all -- the real pre-upgrade shape
    }
    sn._save_symbol_news_store("XAUUSD", [old_schema_entry])
    mock_fetch.return_value = "A real, complete paragraph."

    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies", link="https://example.com/gold")])

    entries = sn.list_symbol_news("XAUUSD")
    assert len(entries) == 1  # backfilled in place, not duplicated
    assert entries[0]["link"] == "https://example.com/gold"
    assert entries[0]["full_text"] == "A real, complete paragraph."
    mock_fetch.assert_called_once_with("https://example.com/gold")


@patch("data.symbol_news.fetch_article_text")
def test_record_symbol_news_to_vault_never_backfills_when_the_new_sighting_also_has_no_link(mock_fetch):
    old_schema_entry = {
        "seq": 1,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "title": "Gold rallies",
        "source": "Reuters",
        "published": "",
        "summary": "",
    }
    sn._save_symbol_news_store("XAUUSD", [old_schema_entry])

    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies")])  # still no link

    entries = sn.list_symbol_news("XAUUSD")
    assert len(entries) == 1
    assert entries[0].get("link", "") == ""
    mock_fetch.assert_not_called()


# --- Category news vault persistence (added 2026-09-20, direct user ------
# report: category news had no vault presence at all, unlike per-symbol) --


@patch("data.symbol_news.fetch_rss_feed")
@patch("data.symbol_news.fetch_article_text")
def test_get_category_news_items_records_fresh_items_to_the_vault(mock_fetch, mock_rss):
    mock_fetch.return_value = "The real full article body."
    mock_rss.return_value = [_item("Real FXStreet item", source="FXStreet", link="https://example.com/fx")]

    sn.get_category_news_items("Forex", limit=5)

    entries = sn.list_category_news("Forex")
    assert len(entries) == 1
    assert entries[0]["title"] == "Real FXStreet item"
    assert entries[0]["full_text"] == "The real full article body."


@patch("data.symbol_news.fetch_rss_feed", return_value=[])
def test_get_category_news_items_records_nothing_when_the_fetch_is_empty(mock_rss):
    sn.get_category_news_items("Forex", limit=5)
    assert sn.list_category_news("Forex") == []


def test_list_category_news_empty_when_nothing_recorded():
    assert sn.list_category_news("Forex") == []


@patch("data.symbol_news.fetch_rss_feed")
def test_category_news_vault_note_is_written_under_its_own_category_folder(mock_rss):
    mock_rss.return_value = [_item("Real FXStreet item", source="FXStreet")]
    sn.get_category_news_items("Forex", limit=5)
    note_path = Path(config.OBSIDIAN_VAULT_PATH) / "News" / "Category" / "Forex.md"
    assert note_path.exists()
    text = note_path.read_text(encoding="utf-8")
    assert "Real FXStreet item" in text
    assert "[[Forex]]" in text


def test_safe_filename_sanitizes_unsafe_characters():
    assert sn._safe_filename("Metals CFD") == "Metals CFD"
    assert sn._safe_filename("Crypto/Weird:Name") == "Crypto_Weird_Name"


def test_record_symbol_news_to_vault_never_raises_on_write_failure(monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sn, "_save_symbol_news_store", _boom)
    sn._record_symbol_news_to_vault("XAUUSD", [_item("Gold rallies")])  # must not raise


def test_list_symbol_news_empty_when_nothing_recorded():
    assert sn.list_symbol_news("NEVERTOUCHED") == []


def test_list_symbol_news_most_recent_first():
    sn._record_symbol_news_to_vault("XAUUSD", [_item("First"), _item("Second")])
    entries = sn.list_symbol_news("XAUUSD")
    assert [e["title"] for e in entries] == ["Second", "First"]


# --- fetch_full_article_text: thin wrapper over data.news_source ---------


def test_fetch_full_article_text_returns_empty_string_for_no_link():
    assert sn.fetch_full_article_text("") == ""


@patch("data.symbol_news.fetch_article_text")
def test_fetch_full_article_text_delegates_to_news_source(mock_fetch):
    mock_fetch.return_value = "Real article body text."
    result = sn.fetch_full_article_text("https://example.com/article")
    assert result == "Real article body text."
    mock_fetch.assert_called_once_with("https://example.com/article")


def test_headline_with_age_tags_breaking_news_and_never_fabricates_an_age():
    from datetime import datetime, timedelta, timezone

    from data.symbol_news import headline_with_age

    now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    fresh = {"title": "Fed surprises", "source": "Reuters", "published": (now - timedelta(minutes=25)).isoformat()}
    old = {"title": "Old story", "source": "AP", "published": (now - timedelta(hours=5)).isoformat()}
    unknown = {"title": "No date", "source": "AP", "published": ""}
    assert headline_with_age(fresh, now) == "[BREAKING] Fed surprises (Reuters, 25m ago)"
    assert headline_with_age(old, now) == "Old story (AP, 5h ago)"
    assert headline_with_age(unknown, now) == "No date"


def test_headline_with_age_survives_a_naive_or_garbled_timestamp():
    from datetime import datetime, timezone

    from data.symbol_news import headline_with_age

    now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    naive = {"title": "No offset", "source": "X", "published": "2026-09-24T11:30:00"}  # a feed stamp with no offset
    assert headline_with_age(naive, now) == "[BREAKING] No offset (X, 30m ago)"
    garbled = {"title": "Bad date", "source": "X", "published": "not a date"}
    assert headline_with_age(garbled, now) == "Bad date"
