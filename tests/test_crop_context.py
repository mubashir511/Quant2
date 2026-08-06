from unittest.mock import MagicMock, patch

from data.crop_context import fetch_crop_supply_demand_context, fetch_weather_snapshot


def _fake_open_meteo_response(precip, tmax, tmin):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "daily": {
            "precipitation_sum": precip,
            "temperature_2m_max": tmax,
            "temperature_2m_min": tmin,
        }
    }
    return mock_response


@patch("requests.get")
def test_fetch_weather_snapshot_splits_past_and_forecast(mock_get):
    # 10 "past" days of 1mm each, then 7 "forecast" days of 2mm each.
    precip = [1.0] * 10 + [2.0] * 7
    tmax = [30.0] * 10 + [31.0] * 7
    tmin = [20.0] * 10 + [21.0] * 7
    mock_get.return_value = _fake_open_meteo_response(precip, tmax, tmin)

    snapshot = fetch_weather_snapshot(38.5, -98.0)
    assert snapshot.past_30d_precip_mm == 10.0
    assert snapshot.forecast_7d_precip_mm == 14.0
    assert snapshot.temp_max_avg_c == 30.0
    assert snapshot.temp_min_avg_c == 20.0


@patch("requests.get", side_effect=RuntimeError("network down"))
def test_fetch_weather_snapshot_tolerates_failure(mock_get):
    snapshot = fetch_weather_snapshot(38.5, -98.0)
    assert snapshot.past_30d_precip_mm is None
    assert snapshot.forecast_7d_precip_mm is None
    assert snapshot.temp_max_avg_c is None
    assert snapshot.temp_min_avg_c is None


@patch("requests.get")
def test_fetch_crop_supply_demand_context_includes_exporters_and_importers(mock_get):
    precip = [1.0] * 30 + [1.0] * 7
    tmax = [30.0] * 30 + [30.0] * 7
    tmin = [20.0] * 30 + [20.0] * 7
    mock_get.return_value = _fake_open_meteo_response(precip, tmax, tmin)

    context = fetch_crop_supply_demand_context({"Wheat"})
    assert "Wheat supply/demand context:" in context
    assert "Exporter Russia" in context
    assert "Exporter USA" in context
    assert "Importer Egypt" in context


def test_fetch_crop_supply_demand_context_skips_unrecognized_crop():
    context = fetch_crop_supply_demand_context({"NotACrop"})
    assert context == ""


def test_fetch_crop_supply_demand_context_empty_for_no_crops():
    assert fetch_crop_supply_demand_context(set()) == ""
