import numpy as np
import pandas as pd
import pytest

import config
from analysis.chart_structure import FibonacciLevels, SRLevel, SRLevelsResult
from analysis.intraday_context import IntradayLevels
from analysis.level_map import build_level_map
from analysis.playbook import RangeRead


def _bars(highs, lows):
    highs, lows = np.asarray(highs, float), np.asarray(lows, float)
    return pd.DataFrame({"High": highs, "Low": lows, "Close": (highs + lows) / 2})


def _sol_like_bars():
    """SOLUSD-shaped: a slide to a 116.30 low, a sharp bounce, a lower-high grind - the 116.30 low is a reaction point that
    price turned from and never undercut; the multi-touch zone the old engine picked sits lower at 115.85."""
    highs, lows = [], []
    for p in np.linspace(118.6, 116.9, 20):  # slide
        highs.append(p + 0.1)
        lows.append(p - 0.1)
    for p in (116.6, 116.4, 116.3, 116.5, 116.9, 117.3, 117.6, 117.7):  # the reaction low at 116.30 (bar 22 low = 116.30 - 0.1)
        highs.append(p + 0.15)
        lows.append(p - 0.1 if p != 116.3 else 116.20)
    for p in (117.6, 117.5, 117.7, 117.6, 117.55, 117.6, 117.65, 117.6):
        highs.append(p + 0.1)
        lows.append(p - 0.1)
    return _bars(highs, lows)


def _zone_result():
    return SRLevelsResult(
        resistance_levels=[],
        support_levels=[SRLevel(price=115.85, touches=4, distance_pct=-1.5, is_liquidity_pool=True, low=115.7, high=116.0)],
    )


def test_sol_replay_lists_the_reaction_low_ahead_of_the_deeper_zone():
    bars = _sol_like_bars()
    lm = build_level_map("buy", 117.60, 0.35, bars=bars, sr_levels=_zone_result())
    prices = [round(c.price, 2) for c in lm.candidates]
    assert 116.20 in prices and 115.85 in prices
    assert prices.index(116.20) < prices.index(115.85)  # nearest first: the reaction point precedes the deeper zone
    reaction = next(c for c in lm.candidates if round(c.price, 2) == 116.20)
    assert reaction.is_reaction and "reaction low" in reaction.kinds
    deeper = next(c for c in lm.candidates if round(c.price, 2) == 115.85)
    assert not deeper.is_reaction and "4 touches" in deeper.notes[0]
    assert reaction.fill_odds_pct > deeper.fill_odds_pct  # distance sets the odds
    text = lm.text()
    assert "REACTION POINT" in text and "say why" in text and "P75 1.3" in text


def test_a_reaction_low_that_was_undercut_later_is_not_a_candidate():
    bars = _sol_like_bars()
    lows = bars["Low"].to_numpy().copy()
    lows[-2] = 116.0  # traded through 116.20 afterwards
    bars["Low"] = lows
    lm = build_level_map("buy", 117.60, 0.35, bars=bars)
    assert all(not c.is_reaction or round(c.price, 2) != 116.20 for c in lm.candidates)


def test_sell_side_mirrors_with_reaction_highs_and_resistance_zones():
    bars = _sol_like_bars()
    flipped = _bars(200.0 - bars["Low"].to_numpy(), 200.0 - bars["High"].to_numpy())  # mirror the path
    zones = SRLevelsResult(resistance_levels=[SRLevel(price=84.15, touches=4, distance_pct=1.5, low=84.0, high=84.3)], support_levels=[])
    lm = build_level_map("sell", 82.40, 0.35, bars=flipped, sr_levels=zones)
    prices = [round(c.price, 2) for c in lm.candidates]
    assert 83.8 in prices and 84.15 in prices and prices.index(83.8) < prices.index(84.15)
    assert any("reaction high" in c.kinds for c in lm.candidates)


def test_range_session_and_fibonacci_candidates_merge_when_they_coincide():
    rng = RangeRead(high=119.0, low=116.90, bars=24, vol_ratio=1.0, last_close=117.6)
    intraday = IntradayLevels(
        last_price=117.6, prev_day_high=120.0, prev_day_low=116.93, prev_day_close=118.0, day_open=118.0, day_high=119.0,
        day_low=116.2, vwap=None, adr=3.0, range_used_pct=90.0, session_bars=100,
    )
    fib = FibonacciLevels(
        swing_high=120.0, swing_low=114.0, high_is_more_recent=True, levels={"38.2%": 117.7, "50.0%": 117.4, "61.8%": 116.29},
        current_price=117.6, nearest_level_name="38.2%", nearest_level_price=117.7, distance_to_nearest_pct=0.1,
    )
    lm = build_level_map("buy", 117.60, 0.35, sr_levels=None, fibonacci=fib, intraday_levels=intraday, range_read=rng)
    by_price = {round(c.price, 2): c for c in lm.candidates}
    assert 116.93 in by_price and {"24-bar range low", "previous-day low"} <= set(by_price[116.93].kinds)  # 116.90 & 116.93 are one level
    assert any("Fibonacci 61.8%" in c.kinds for c in lm.candidates) and any("session low" in c.kinds for c in lm.candidates)
    assert 117.7 not in by_price  # above the price: not a support for a buy


def test_too_close_far_and_missing_inputs():
    lm = build_level_map("buy", 100.0, 1.0, range_read=RangeRead(high=105.0, low=99.7, bars=24, vol_ratio=None, last_close=100.0))
    assert lm.candidates[0].too_close and "noise band" in lm.text()
    far = build_level_map("buy", 100.0, 1.0, range_read=RangeRead(high=105.0, low=90.0, bars=24, vol_ratio=None, last_close=100.0))
    assert far.candidates == [] and "no candidate" in far.text()
    assert build_level_map("buy", None, 1.0) is None and build_level_map("buy", 100.0, 0) is None and build_level_map("hold", 100.0, 1.0) is None


def test_candidate_count_is_capped(monkeypatch):
    monkeypatch.setattr(config, "LEVEL_MAP_MAX_CANDIDATES", 2)
    zones = SRLevelsResult(
        resistance_levels=[],
        support_levels=[SRLevel(price=p, touches=2, distance_pct=-1.0) for p in (99.0, 98.0, 97.0, 96.0)],
    )
    assert len(build_level_map("buy", 100.0, 1.0, sr_levels=zones).candidates) == 2


def test_ftmo_suggest_builds_both_sides_and_degrades_to_empty(monkeypatch):
    from types import SimpleNamespace

    from ai import ftmo_suggest

    bars = _sol_like_bars()
    stats = SimpleNamespace(atr=0.35)
    structure = SimpleNamespace(sr_levels=_zone_result(), fibonacci=None)
    maps = ftmo_suggest._build_level_maps(bars, stats, structure, None, None, 117.6)
    assert set(maps) == {"buy", "sell"} and maps["buy"].candidates and maps["sell"].side == "sell"
    assert ftmo_suggest._build_level_maps(bars, SimpleNamespace(atr=None), structure, None, None, 117.6) == {}
    monkeypatch.setattr(config, "LEVEL_MAP_ENABLED", False)
    assert ftmo_suggest._build_level_maps(bars, stats, structure, None, None, 117.6) == {}
