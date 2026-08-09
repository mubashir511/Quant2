from unittest.mock import MagicMock, patch

import pandas as pd

from data.market_history import fetch_price_history_ohlcv


@patch("yfinance.Ticker")
def test_fetch_price_history_ohlcv_returns_high_low_close_volume(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame(
        {
            "Open": [99.0, 100.0, 101.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [98.0, 99.0, 100.0],
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.date_range("2026-01-01", periods=3),
    )
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_price_history_ohlcv("GC=F")
    assert list(result["Close"]) == [100.0, 101.0, 102.0]
    assert list(result["High"]) == [101.0, 102.0, 103.0]
    assert list(result["Low"]) == [98.0, 99.0, 100.0]
    assert list(result["Volume"]) == [1000, 1100, 1200]
    assert "Open" not in result.columns
    mock_ticker_cls.assert_called_once_with("GC=F")


@patch("yfinance.Ticker")
def test_fetch_price_history_ohlcv_returns_empty_frame_on_empty_history(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_price_history_ohlcv("UNKNOWN=F")
    assert result.empty
    assert list(result.columns) == ["High", "Low", "Close", "Volume"]
    assert result["Close"].dtype == float
