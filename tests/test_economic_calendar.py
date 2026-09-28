import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import config
from data import economic_calendar as ec

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def _raw(title, currency, dt_utc, impact="High"):
    # The real feed publishes US-Eastern offsets; emit one so tz handling is exercised.
    eastern = dt_utc.astimezone(timezone(timedelta(hours=-4)))
    return {"title": title, "country": currency, "date": eastern.isoformat(), "impact": impact, "forecast": "1.0%", "previous": "0.9%"}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path):
    with patch.object(config, "ECONOMIC_CALENDAR_CACHE_FILE", str(tmp_path / "cal.json")):
        yield


def test_parse_converts_offsets_to_utc_and_skips_junk():
    raw = [_raw("CPI", "USD", NOW), {"title": "broken"}, {"date": "not-a-date", "country": "USD"}]
    with patch("data.economic_calendar._fetch_raw", return_value=raw):
        events = ec.fetch_calendar_events(NOW)
    assert len(events) == 1
    assert events[0].time_utc == NOW and events[0].currency == "USD" and events[0].impact == "High"


def test_cache_is_reused_within_ttl_and_stale_cache_bridges_an_outage():
    with patch("data.economic_calendar._fetch_raw", return_value=[_raw("CPI", "USD", NOW)]) as fetch:
        ec.fetch_calendar_events(NOW)
        ec.fetch_calendar_events(NOW + timedelta(minutes=10))
        assert fetch.call_count == 1
    with patch("data.economic_calendar._fetch_raw", return_value=[]) as fetch:
        later = NOW + timedelta(hours=3)  # cache expired, live feed down
        assert len(ec.fetch_calendar_events(later)) == 1
        assert fetch.call_count == 1


def test_total_failure_returns_empty_never_raises():
    with patch("data.economic_calendar._fetch_raw", side_effect=RuntimeError("boom")):
        assert ec.fetch_calendar_events(NOW) == []


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("EURUSD", {"EUR", "USD"}), ("USDJPY", {"USD", "JPY"}), ("GBPUSD", {"GBP", "USD"}),
        ("XAUUSD", {"USD"}), ("UKOIL.cash", {"USD"}), ("WHEAT.c", {"USD"}), ("BTCUSD", {"USD"}),
        ("MSFT", {"USD"}), ("LVMH", {"EUR"}), ("DE40", {"EUR"}), ("JP225", {"JPY"}),
    ],
)
def test_currencies_for_symbol(symbol, expected):
    assert ec.currencies_for_symbol(symbol) == expected


def test_blackout_window_is_30_before_10_after_for_matching_currency_only():
    events = ec._parse_events([_raw("FOMC Rate Decision", "USD", NOW + timedelta(minutes=20))])
    assert ec.blackout_status("MSFT", NOW, events).active is True          # 20 min before
    assert ec.blackout_status("MSFT", NOW - timedelta(minutes=15), events).active is False  # 35 min before
    assert ec.blackout_status("MSFT", NOW + timedelta(minutes=25), events).active is True   # 5 min after
    assert ec.blackout_status("MSFT", NOW + timedelta(minutes=40), events).active is False  # 20 min after
    assert ec.blackout_status("LVMH", NOW, events).active is False          # EUR name, USD event


def test_blackout_ignores_medium_and_low_impact_by_default():
    events = ec._parse_events([_raw("Speaks", "USD", NOW + timedelta(minutes=5), impact="Medium")])
    assert ec.blackout_status("MSFT", NOW, events).active is False


def test_format_calendar_block_never_presents_an_empty_feed_as_no_risk():
    assert "UNKNOWN" in ec.format_calendar_block(NOW, events=[])
    events = ec._parse_events([_raw("US CPI m/m", "USD", NOW + timedelta(hours=2))])
    block = ec.format_calendar_block(NOW, events=events)
    assert "US CPI m/m" in block and "forecast 1.0%" in block
    assert "no High-impact events" in ec.format_calendar_block(NOW, events=ec._parse_events([_raw("x", "USD", NOW + timedelta(days=3))]))


def test_format_symbol_events_lists_only_relevant_currency():
    events = ec._parse_events([_raw("ECB Rate", "EUR", NOW + timedelta(hours=1)), _raw("US CPI", "USD", NOW + timedelta(hours=2))])
    line = ec.format_symbol_events("EURUSD", NOW, events=events)
    assert "ECB Rate" in line and "US CPI" in line
    assert ec.format_symbol_events("XAUUSD", NOW, events=events).count("ECB Rate") == 0
    assert ec.format_symbol_events("BTCUSD", NOW, events=[]) is None
