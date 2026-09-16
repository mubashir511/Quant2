from unittest.mock import MagicMock, patch

from data.news_source import fetch_google_news, fetch_recent_headlines, fetch_recent_news, fetch_rss_feed


def _rss_response(items_xml: str, channel_title: str = "") -> MagicMock:
    """A fake urllib.request.urlopen(...) context-manager result whose
    .read() returns a real-shaped RSS <channel> body."""
    title_xml = f"<title>{channel_title}</title>" if channel_title else ""
    xml = f"<?xml version='1.0'?><rss><channel>{title_xml}{items_xml}</channel></rss>".encode()
    response = MagicMock()
    response.__enter__.return_value.read.return_value = xml
    return response


@patch("yfinance.Ticker")
def test_fetch_recent_headlines_returns_titles_flat_format(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = [{"title": "Gold rallies on rate cut bets"}, {"title": "Second headline"}]
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_recent_headlines("GC=F", limit=2)
    assert result == ["Gold rallies on rate cut bets", "Second headline"]


@patch("yfinance.Ticker")
def test_fetch_recent_headlines_returns_titles_nested_content_format(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = [{"content": {"title": "Nested format headline"}}]
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_recent_headlines("GC=F", limit=2)
    assert result == ["Nested format headline"]


@patch("yfinance.Ticker")
def test_fetch_recent_headlines_respects_limit(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = [{"title": f"Headline {i}"} for i in range(5)]
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_recent_headlines("GC=F", limit=2)
    assert len(result) == 2


@patch("yfinance.Ticker")
def test_fetch_recent_headlines_returns_empty_list_on_no_news(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = []
    mock_ticker_cls.return_value = mock_ticker

    assert fetch_recent_headlines("GC=F") == []


@patch("yfinance.Ticker", side_effect=RuntimeError("network down"))
def test_fetch_recent_headlines_returns_empty_list_on_error(mock_ticker_cls):
    assert fetch_recent_headlines("GC=F") == []


@patch("yfinance.Ticker")
def test_fetch_recent_news_keeps_summary_source_and_published(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = [
        {
            "content": {
                "title": "Gold rallies on rate cut bets",
                "summary": "A real, genuine article summary.",
                "provider": {"displayName": "Reuters"},
                "pubDate": "2026-09-14T00:00:00Z",
            }
        }
    ]
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_recent_news("GC=F", limit=2)
    assert result == [
        {
            "title": "Gold rallies on rate cut bets",
            "summary": "A real, genuine article summary.",
            "source": "Reuters",
            "published": "2026-09-14T00:00:00Z",
        }
    ]


@patch("yfinance.Ticker")
def test_fetch_recent_news_falls_back_to_flat_title_and_empty_fields(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = [{"title": "Flat-format headline"}]
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_recent_news("GC=F", limit=2)
    assert result == [{"title": "Flat-format headline", "summary": "", "source": "", "published": ""}]


@patch("yfinance.Ticker")
def test_fetch_recent_news_skips_items_with_no_title(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = [{"content": {"summary": "No title here"}}, {"title": "Has a title"}]
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_recent_news("GC=F", limit=5)
    assert len(result) == 1
    assert result[0]["title"] == "Has a title"


@patch("yfinance.Ticker")
def test_fetch_recent_news_respects_limit(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = [{"title": f"Headline {i}"} for i in range(5)]
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_recent_news("GC=F", limit=2)
    assert len(result) == 2


@patch("yfinance.Ticker")
def test_fetch_recent_news_returns_empty_list_on_no_news(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.news = []
    mock_ticker_cls.return_value = mock_ticker

    assert fetch_recent_news("GC=F") == []


@patch("yfinance.Ticker", side_effect=RuntimeError("network down"))
def test_fetch_recent_news_returns_empty_list_on_error(mock_ticker_cls):
    assert fetch_recent_news("GC=F") == []


@patch("urllib.request.urlopen")
def test_fetch_google_news_uses_the_real_source_element_when_present(mock_urlopen):
    mock_urlopen.return_value = _rss_response(
        "<item><title>USD/CHF Signal: Bullish Outlook</title>"
        "<source url='https://dailyforex.com'>DailyForex</source>"
        "<pubDate>Fri, 11 Sep 2026 06:22:51 GMT</pubDate></item>"
    )
    result = fetch_google_news("USDCHF forex", limit=5)
    assert result == [
        {
            "title": "USD/CHF Signal: Bullish Outlook",
            "summary": "",
            "source": "DailyForex",
            "published": "Fri, 11 Sep 2026 06:22:51 GMT",
        }
    ]


@patch("urllib.request.urlopen")
def test_fetch_google_news_splits_title_dash_source_when_no_source_element(mock_urlopen):
    mock_urlopen.return_value = _rss_response("<item><title>Gold rallies on rate cut bets - Reuters</title></item>")
    result = fetch_google_news("gold price", limit=5)
    assert result == [{"title": "Gold rallies on rate cut bets", "summary": "", "source": "Reuters", "published": ""}]


@patch("urllib.request.urlopen")
def test_fetch_google_news_respects_limit(mock_urlopen):
    items_xml = "".join(f"<item><title>Headline {i}</title></item>" for i in range(5))
    mock_urlopen.return_value = _rss_response(items_xml)
    result = fetch_google_news("query", limit=2)
    assert len(result) == 2


@patch("urllib.request.urlopen")
def test_fetch_google_news_returns_empty_list_on_no_items(mock_urlopen):
    mock_urlopen.return_value = _rss_response("")
    assert fetch_google_news("query") == []


@patch("urllib.request.urlopen", side_effect=RuntimeError("network down"))
def test_fetch_google_news_returns_empty_list_on_error(mock_urlopen):
    assert fetch_google_news("query") == []


@patch("urllib.request.urlopen")
def test_fetch_google_news_returns_empty_list_on_malformed_xml(mock_urlopen):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"not xml at all"
    mock_urlopen.return_value = response
    assert fetch_google_news("query") == []


@patch("urllib.request.urlopen")
def test_fetch_rss_feed_keeps_the_real_description_as_summary(mock_urlopen):
    mock_urlopen.return_value = _rss_response(
        "<item><title>Bitcoin bucks tech selloff</title>"
        "<description>Bitcoin rose above $77,000 as AI safety concerns weigh on stocks.</description>"
        "<pubDate>Mon, 14 Sep 2026 08:07:16 +0000</pubDate></item>",
        channel_title="CoinDesk",
    )
    result = fetch_rss_feed("https://www.coindesk.com/arc/outboundfeeds/rss/", limit=5)
    assert result == [
        {
            "title": "Bitcoin bucks tech selloff",
            "summary": "Bitcoin rose above $77,000 as AI safety concerns weigh on stocks.",
            "source": "CoinDesk",
            "published": "Mon, 14 Sep 2026 08:07:16 +0000",
        }
    ]


@patch("urllib.request.urlopen")
def test_fetch_rss_feed_leaves_summary_empty_when_feed_has_no_description(mock_urlopen):
    mock_urlopen.return_value = _rss_response(
        "<item><title>Gold Selloff Shows Why Real Yields Still Dominate</title></item>",
        channel_title="Commodities Analysis &amp; Opinion",
    )
    result = fetch_rss_feed("https://www.investing.com/rss/commodities.rss", limit=5)
    assert result == [
        {
            "title": "Gold Selloff Shows Why Real Yields Still Dominate",
            "summary": "",
            "source": "Commodities Analysis & Opinion",
            "published": "",
        }
    ]


@patch("urllib.request.urlopen")
def test_fetch_rss_feed_skips_items_with_no_title(mock_urlopen):
    mock_urlopen.return_value = _rss_response(
        "<item><description>No title here</description></item><item><title>Has a title</title></item>",
        channel_title="Feed",
    )
    result = fetch_rss_feed("https://example.com/rss", limit=5)
    assert len(result) == 1
    assert result[0]["title"] == "Has a title"


@patch("urllib.request.urlopen")
def test_fetch_rss_feed_respects_limit(mock_urlopen):
    items_xml = "".join(f"<item><title>Headline {i}</title></item>" for i in range(5))
    mock_urlopen.return_value = _rss_response(items_xml, channel_title="Feed")
    result = fetch_rss_feed("https://example.com/rss", limit=2)
    assert len(result) == 2


@patch("urllib.request.urlopen")
def test_fetch_rss_feed_returns_empty_list_on_no_items(mock_urlopen):
    mock_urlopen.return_value = _rss_response("", channel_title="Feed")
    assert fetch_rss_feed("https://example.com/rss") == []


@patch("urllib.request.urlopen", side_effect=RuntimeError("network down"))
def test_fetch_rss_feed_returns_empty_list_on_error(mock_urlopen):
    assert fetch_rss_feed("https://example.com/rss") == []


@patch("urllib.request.urlopen")
def test_fetch_rss_feed_returns_empty_list_on_malformed_xml(mock_urlopen):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"not xml at all"
    mock_urlopen.return_value = response
    assert fetch_rss_feed("https://example.com/rss") == []
