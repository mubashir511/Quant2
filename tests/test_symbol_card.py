from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import config
from analysis.symbol_card import build_cards, compute_symbol_metrics, format_card, load_cards, save_cards


def _frame(returns, start=100.0, spread=0.05, n=None):
    close = start * np.exp(np.cumsum(returns))
    high, low = close * (1 + spread / 100), close * (1 - spread / 100)
    index = pd.date_range("2025-01-01", periods=len(close), freq="5min")
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": 100.0}, index=index)


def _random_walk(seed, n=12000):
    return _frame(np.random.default_rng(seed).normal(0, 0.0005, n))


def _trending(seed, n=12000):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.0004, n)
    drift = np.repeat(rng.choice([-0.0006, 0.0006], size=n // 60 + 1), 60)[:n]  # persistent one-hour legs
    return _frame(noise + drift)


def test_too_little_history_gives_no_card():
    assert compute_symbol_metrics(_random_walk(0, n=800)) is None
    assert compute_symbol_metrics(None) is None
    assert compute_symbol_metrics(pd.DataFrame({"Close": [1.0] * 6000})) is None


def test_random_walk_has_no_reliable_tendency_and_a_persistent_series_reads_as_momentum():
    raw = {f"RW{i}": compute_symbol_metrics(_random_walk(i)) for i in range(1, 9)}
    cards = build_cards({**raw, "TREND": compute_symbol_metrics(_trending(30))})
    rw = cards["symbols"]["RW1"]
    assert rw["follow"] == "no reliable tendency" and rw["vr_read"] == "neutral"
    assert 0.9 < raw["RW1"]["whole"]["vr12"] < 1.1  # (the card value is pulled toward a pool that contains the trending series)
    trend = cards["symbols"]["TREND"]
    assert trend["vr12"] > 1.05 and trend["vr_read"].startswith("momentum")
    assert trend["bars"] > 11000 and trend["break_events"] >= 40


def test_a_symbol_is_shrunk_toward_the_pool_and_needs_both_halves_to_agree():
    good = lambda cont, dec: {"bars": 20000, "adx20": 0.6, "adx40": 0.1, "break_continued": cont, "break_decided": dec, "vr12": 1.0}
    strong = {"whole": good(660, 1000), "first": good(330, 500), "second": good(330, 500), "hours": None, "days": 70}
    flip = {"whole": good(660, 1000), "first": good(400, 500), "second": good(260, 500), "hours": None, "days": 70}
    pool = {f"P{i}": {"whole": good(500, 1000), "first": good(250, 500), "second": good(250, 500), "hours": None, "days": 70} for i in range(6)}
    cards = build_cards({"STRONG": strong, "FLIP": flip, **pool})["symbols"]
    assert cards["STRONG"]["break_rate_raw"] == pytest.approx(0.66)
    assert 0.5 < cards["STRONG"]["break_rate"] < 0.66  # pulled toward the pool
    assert cards["STRONG"]["follow"] == "range breaks tend to CONTINUE"
    assert cards["FLIP"]["follow"] == "range breaks tend to CONTINUE" or cards["FLIP"]["break_halves"][1] < 0.53
    fade = {"whole": good(340, 1000), "first": good(170, 500), "second": good(170, 500), "hours": None, "days": 70}
    assert build_cards({"FADE": fade, **pool})["symbols"]["FADE"]["follow"].startswith("range breaks tend to FAIL")


def test_round_trip_and_the_prompt_line(tmp_path):
    cards = build_cards({"RW1": compute_symbol_metrics(_random_walk(1)), "RW2": compute_symbol_metrics(_random_walk(2))})
    path = str(tmp_path / "cards.json")
    save_cards(cards, path)
    loaded = load_cards(path)
    line = format_card("RW1", loaded)
    assert line.startswith("  Symbol behaviour card (RW1") and "range-break follow-through" in line and "variance ratio" in line
    assert "coin flip" in line and "STALE" not in line
    assert format_card("NOPE", loaded) is None and format_card("RW1", {}) is None
    assert load_cards(str(tmp_path / "missing.json")) == {}


def test_an_old_card_is_labelled_stale():
    cards = build_cards({"RW1": compute_symbol_metrics(_random_walk(1))})
    cards["generated_utc"] = (datetime.now(timezone.utc) - timedelta(days=config.SYMBOL_CARD_STALE_DAYS + 5)).isoformat()
    assert "STALE" in format_card("RW1", cards)
