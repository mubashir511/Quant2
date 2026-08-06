from unittest.mock import MagicMock, patch

import pandas as pd

from data.macro_source import fetch_country_indicators, fetch_fx_rate_to_usd, fetch_market_indicators


def _fake_history(close_price: float) -> pd.DataFrame:
    return pd.DataFrame({"Close": [close_price]}, index=pd.date_range("2026-01-01", periods=1))


@patch("yfinance.Ticker")
def test_fetch_market_indicators_computes_yield_curve_spread(mock_ticker_cls):
    prices = {"^IRX": 3.73, "^TNX": 4.63, "^TYX": 5.19, "^VIX": 16.37, "DX-Y.NYB": 99.91}

    def ticker_side_effect(symbol):
        mock = MagicMock()
        mock.history.return_value = _fake_history(prices[symbol])
        return mock

    mock_ticker_cls.side_effect = ticker_side_effect

    result = fetch_market_indicators()
    assert result.yield_3m_pct == 3.73
    assert result.yield_10y_pct == 4.63
    assert result.yield_30y_pct == 5.19
    assert result.dxy == 99.91
    assert result.vix == 16.37
    assert round(result.yield_curve_10y_3m_spread, 2) == 0.90


@patch("yfinance.Ticker")
def test_fetch_market_indicators_tolerates_a_missing_ticker(mock_ticker_cls):
    def ticker_side_effect(symbol):
        mock = MagicMock()
        if symbol == "^VIX":
            mock.history.return_value = pd.DataFrame()  # empty = fetch failed
        else:
            mock.history.return_value = _fake_history(1.0)
        return mock

    mock_ticker_cls.side_effect = ticker_side_effect

    result = fetch_market_indicators()
    assert result.vix is None
    assert result.yield_3m_pct == 1.0  # other fields still populated


def _fake_worldbank_response(value):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [
        {"page": 1},
        [{"value": value}, {"value": None}],
    ]
    return mock_response


@patch("requests.get")
def test_fetch_country_indicators_parses_world_bank_response(mock_get):
    mock_get.return_value = _fake_worldbank_response(2.5)

    results = fetch_country_indicators(countries=("US",))
    assert len(results) == 1
    assert results[0].country == "United States"
    assert results[0].gdp_growth_pct == 2.5
    assert results[0].inflation_pct == 2.5
    assert results[0].unemployment_pct == 2.5


@patch("requests.get", side_effect=RuntimeError("network down"))
def test_fetch_country_indicators_tolerates_request_failure(mock_get):
    results = fetch_country_indicators(countries=("US",))
    assert len(results) == 1
    assert results[0].gdp_growth_pct is None
    assert results[0].inflation_pct is None
    assert results[0].unemployment_pct is None


def test_fetch_country_indicators_uses_unmapped_code_as_name_fallback():
    with patch("requests.get", side_effect=RuntimeError("network down")):
        results = fetch_country_indicators(countries=("ZZ",))
    assert results[0].country == "ZZ"


@patch("requests.get")
def test_fetch_country_indicators_maps_parallel_results_to_correct_country(mock_get):
    # Each country/indicator pair gets a distinct value; since these run
    # concurrently, this confirms results are mapped back by (country,
    # indicator) rather than by completion order.
    gdp_by_country = {"US": 2.0, "GB": 1.0, "FR": 0.5, "DE": 0.2, "JP": 1.5}

    def get_side_effect(url, params=None, timeout=None):
        country_code = url.split("/country/")[1].split("/indicator/")[0]
        return _fake_worldbank_response(gdp_by_country[country_code])

    mock_get.side_effect = get_side_effect

    results = fetch_country_indicators(countries=tuple(gdp_by_country.keys()))
    result_by_country = {r.country: r for r in results}
    assert result_by_country["United States"].gdp_growth_pct == 2.0
    assert result_by_country["United Kingdom"].gdp_growth_pct == 1.0
    assert result_by_country["France"].gdp_growth_pct == 0.5
    assert result_by_country["Germany"].gdp_growth_pct == 0.2
    assert result_by_country["Japan"].gdp_growth_pct == 1.5


def test_default_country_coverage_includes_all_ten_major_economies():
    import inspect

    from data.macro_source import fetch_country_indicators as fn

    default_countries = inspect.signature(fn).parameters["countries"].default
    assert set(default_countries) == {"US", "GB", "FR", "DE", "JP", "CN", "IN", "KR", "SA", "AE"}


@patch("yfinance.Ticker")
def test_fetch_fx_rate_to_usd_uses_currency_equals_x_ticker(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _fake_history(277.48)
    mock_ticker_cls.return_value = mock_ticker

    rate = fetch_fx_rate_to_usd("PKR")
    assert rate == 277.48
    mock_ticker_cls.assert_called_once_with("PKR=X")


def test_fetch_fx_rate_to_usd_returns_none_for_usd_itself():
    assert fetch_fx_rate_to_usd("USD") is None
    assert fetch_fx_rate_to_usd("usd") is None


@patch("yfinance.Ticker")
def test_fetch_fx_rate_to_usd_tolerates_failure(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker_cls.return_value = mock_ticker

    assert fetch_fx_rate_to_usd("XYZ") is None
