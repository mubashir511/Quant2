import numpy as np
import pandas as pd
import pytest

import config
from data import price_cache


def _bars(start, n, freq="5min", base=100.0):
    idx = pd.date_range(start, periods=n, freq=freq)
    close = base + np.arange(n) * 0.01
    return pd.DataFrame({"Open": close, "High": close + 0.05, "Low": close - 0.05, "Close": close, "Volume": 10.0}, index=idx)


class _FakeFeed:
    """A fake MT5: serves the most recent `count` bars of a fixed series and records the calls."""

    def __init__(self, frame):
        self.frame, self.calls = frame, []

    def __call__(self, symbol, timeframe, count):
        self.calls.append((symbol, timeframe, count))
        return self.frame.iloc[-count:]


def test_first_update_fetches_the_full_depth_and_writes_a_parquet(monkeypatch):
    monkeypatch.setattr(config, "DEEP_HISTORY_BARS_M5", 500)
    feed = _FakeFeed(_bars("2026-01-01", 1000))
    out = price_cache.update_cache("BTCUSD", "M5", feed)
    assert len(out) == 500 and feed.calls == [("BTCUSD", "M5", 500)]
    assert price_cache.cache_path("BTCUSD", "M5").exists()
    assert price_cache.deep_history("BTCUSD", "M5").equals(out)  # cache-only read, no fetch


def test_later_updates_fetch_only_the_new_bars_and_dedupe_the_overlap(monkeypatch):
    monkeypatch.setattr(config, "DEEP_HISTORY_BARS_M5", 5000)
    full = _bars("2026-01-01", 3000)
    price_cache.update_cache("ETHUSD", "M5", _FakeFeed(full.iloc[:2000]), initial_count=2000)
    feed = _FakeFeed(full)
    now = full.index[2999] + pd.Timedelta("1min")
    out = price_cache.update_cache("ETHUSD", "M5", feed, now=now)
    (call,) = feed.calls
    assert call[2] < 1500  # roughly the 1000 missing bars + the safety margin, NOT the full depth
    assert len(out) == 3000 and out.index.is_unique and out.index.is_monotonic_increasing
    assert out.equals(full)


def test_fresh_bars_win_on_overlap_because_the_last_cached_bar_may_be_unfinished():
    cached = _bars("2026-01-01", 10)
    fresh = _bars("2026-01-01 00:45", 5)
    fresh.iloc[0, fresh.columns.get_loc("Close")] = 999.0
    merged = price_cache.merge_history(cached, fresh)
    assert merged.loc[pd.Timestamp("2026-01-01 00:45"), "Close"] == 999.0 and len(merged) == 14


def test_the_cache_is_trimmed_to_the_most_recent_max_bars():
    merged = price_cache.merge_history(_bars("2026-01-01", 100), _bars("2026-01-01 08:20", 100), max_bars=120)
    assert len(merged) == 120 and merged.index[-1] == pd.Timestamp("2026-01-01 08:20") + pd.Timedelta("5min") * 99


def test_an_empty_or_failing_fetch_keeps_the_cached_history(monkeypatch):
    monkeypatch.setattr(config, "DEEP_HISTORY_BARS_M5", 500)
    good = price_cache.update_cache("XAUUSD", "M5", _FakeFeed(_bars("2026-01-01", 600)))
    assert price_cache.update_cache("XAUUSD", "M5", lambda *a: pd.DataFrame()).equals(good)

    def boom(*a):
        raise RuntimeError("terminal down")

    assert price_cache.update_cache("XAUUSD", "M5", boom).equals(good)
    assert price_cache.load_cached("XAUUSD", "M5").equals(good)  # nothing was erased


def test_a_corrupt_cache_file_is_treated_as_empty_and_rebuilt(monkeypatch):
    monkeypatch.setattr(config, "DEEP_HISTORY_BARS_M5", 300)
    path = price_cache.cache_path("EURUSD", "M5")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a parquet file")
    assert price_cache.load_cached("EURUSD", "M5").empty
    out = price_cache.update_cache("EURUSD", "M5", _FakeFeed(_bars("2026-01-01", 400)))
    assert len(out) == 300 and len(price_cache.load_cached("EURUSD", "M5")) == 300


def test_bars_needed_counts_the_gap_in_bars_plus_a_margin():
    cached = _bars("2026-01-01", 10)
    now = cached.index[-1] + pd.Timedelta(minutes=600)
    assert price_cache.bars_needed(cached, "M5", now, safety=50) == 120 + 50
    assert price_cache.bars_needed(cached, "H1", now, safety=50) == 10 + 50
    assert price_cache.bars_needed(pd.DataFrame(columns=["Close"]), "M5") is None


def test_symbol_names_with_dots_and_slashes_make_safe_file_names():
    assert price_cache.cache_path("UKOIL.cash", "M5").name == "UKOIL.cash_M5.parquet"
    assert "/" not in price_cache.cache_path("A/B", "H1").name
