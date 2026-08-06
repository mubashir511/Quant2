from unittest.mock import MagicMock, patch

from data.news_source import fetch_recent_headlines


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
