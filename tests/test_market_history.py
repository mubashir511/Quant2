from unittest.mock import MagicMock, patch

import pandas as pd

from data.market_history import fetch_price_history


@patch("yfinance.Ticker")
def test_fetch_price_history_returns_close_series(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2026-01-01", periods=3),
    )
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_price_history("GC=F")
    assert list(result) == [100.0, 101.0, 102.0]
    mock_ticker_cls.assert_called_once_with("GC=F")


@patch("yfinance.Ticker")
def test_fetch_price_history_returns_empty_series_on_empty_history(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_price_history("UNKNOWN=F")
    assert result.empty
