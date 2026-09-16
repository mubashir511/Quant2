from unittest.mock import MagicMock, patch

from data.fundamentals_source import EquityFundamentals, fetch_equity_fundamentals


def _fake_ticker(info: dict, calendar: dict) -> MagicMock:
    mock = MagicMock()
    mock.info = info
    mock.calendar = calendar
    return mock


@patch("yfinance.Ticker")
def test_fetch_equity_fundamentals_parses_real_fields(mock_ticker_cls):
    mock_ticker_cls.return_value = _fake_ticker(
        info={
            "targetMeanPrice": 327.17758,
            "recommendationKey": "strong_buy",
            "numberOfAnalystOpinions": 58,
            "forwardPE": 13.510182,
        },
        calendar={"Earnings Date": ["2026-11-18"], "Earnings Average": 2.47269},
    )
    result = fetch_equity_fundamentals("NVDA")
    assert result == EquityFundamentals(
        analyst_target_mean_price=327.17758,
        analyst_recommendation="strong_buy",
        analyst_count=58,
        forward_pe=13.510182,
        next_earnings_date="2026-11-18",
        earnings_eps_estimate=2.47269,
    )


@patch("yfinance.Ticker")
def test_fetch_equity_fundamentals_leaves_missing_fields_as_none(mock_ticker_cls):
    # A real, honest degrade -- e.g. thin analyst coverage or no
    # scheduled earnings date yet -- never fabricated to fill the gap.
    mock_ticker_cls.return_value = _fake_ticker(info={}, calendar={})
    result = fetch_equity_fundamentals("SOMETICKER")
    assert result == EquityFundamentals(
        analyst_target_mean_price=None,
        analyst_recommendation=None,
        analyst_count=None,
        forward_pe=None,
        next_earnings_date=None,
        earnings_eps_estimate=None,
    )


@patch("yfinance.Ticker")
def test_fetch_equity_fundamentals_handles_none_info_and_calendar(mock_ticker_cls):
    # yfinance itself can return None (not an empty dict) for either --
    # confirmed real shape variance, not fabricated.
    mock = MagicMock()
    mock.info = None
    mock.calendar = None
    mock_ticker_cls.return_value = mock
    result = fetch_equity_fundamentals("SOMETICKER")
    assert result.analyst_target_mean_price is None
    assert result.next_earnings_date is None


@patch("yfinance.Ticker", side_effect=RuntimeError("network down"))
def test_fetch_equity_fundamentals_returns_none_on_error(mock_ticker_cls):
    assert fetch_equity_fundamentals("NVDA") is None
