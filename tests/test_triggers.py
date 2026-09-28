import pandas as pd
import pytest

import config
from ai.portfolio_suggest import parse_pending_setups
from analysis.triggers import evaluate_trigger, normalise_trigger


def _bars(rows):
    return pd.DataFrame(rows, columns=["High", "Low", "Close"])


ATR = 1.0


def test_normalise_accepts_only_known_kinds_with_a_positive_level():
    assert normalise_trigger({"kind": "Range_Break", "level": 100}) == {"kind": "range_break", "level": 100.0, "within_bars": 6}
    assert normalise_trigger({"kind": "reclaim", "level": 5.5, "within_bars": 500})["within_bars"] == 24
    assert normalise_trigger({"kind": "close_beyond", "level": 5, "within_bars": 0})["within_bars"] == 1
    for bad in (None, "x", {}, {"kind": "hammer", "level": 1}, {"kind": "reclaim", "level": 0}, {"kind": "reclaim", "level": "1"}, {"kind": "reclaim", "level": True}):
        assert normalise_trigger(bad) is None


def test_range_break_buy_fires_on_a_recent_touch_and_not_when_extended_or_failed():
    bars = _bars([(99.0, 98.0, 98.5), (100.2, 99.0, 99.4), (99.6, 99.0, 99.5)])
    trig = {"kind": "range_break", "level": 100.0}
    ok = evaluate_trigger(trig, "buy", bars, 99.9, 100.1, ATR)
    assert ok.fired and "traded through" in ok.reason
    extended = evaluate_trigger(trig, "buy", bars, 100.7, 100.8, ATR)  # 0.8 ATR past > the 0.5 cap
    assert not extended.fired and "extended" in extended.reason
    failed = evaluate_trigger(trig, "buy", bars, 99.5, 99.6, ATR)  # broke, fell back 0.4 ATR
    assert not failed.fired and "failed break" in failed.reason


def test_range_break_needs_a_touch_and_sell_mirrors_buy():
    quiet = _bars([(99.0, 98.0, 98.5), (99.2, 98.4, 98.9)])
    assert not evaluate_trigger({"kind": "range_break", "level": 100.0}, "buy", quiet, 98.9, 99.0, ATR).fired
    down = _bars([(101.0, 100.2, 100.4), (100.5, 99.8, 100.1)])
    assert evaluate_trigger({"kind": "range_break", "level": 100.0}, "sell", down, 99.9, 100.0, ATR).fired
    assert not evaluate_trigger({"kind": "range_break", "level": 100.0}, "sell", down, 99.1, 99.2, ATR).fired  # 0.9 ATR extended


def test_only_recent_bars_count():
    bars = _bars([(100.5, 99.0, 99.5)] + [(99.5, 98.5, 99.0)] * 8)
    trig = {"kind": "range_break", "level": 100.0, "within_bars": 3}
    assert not evaluate_trigger(trig, "buy", bars, 99.4, 99.5, ATR).fired  # the touch is 9 bars ago
    assert evaluate_trigger({**trig, "within_bars": 9}, "buy", bars, 99.9, 100.0, ATR).fired


def test_reclaim_needs_a_pierce_then_a_later_close_back_inside():
    spring = _bars([(101.0, 100.4, 100.8), (100.9, 99.6, 99.8), (100.6, 99.9, 100.3)])
    trig = {"kind": "reclaim", "level": 100.0}
    ok = evaluate_trigger(trig, "buy", spring, 100.3, 100.4, ATR)
    assert ok.fired
    same_bar_only = _bars([(101.0, 100.4, 100.8), (100.9, 99.6, 99.8)])  # pierced, no later close
    assert not evaluate_trigger(trig, "buy", same_bar_only, 99.9, 100.0, ATR).fired
    no_pierce = _bars([(101.0, 100.4, 100.8), (100.9, 100.2, 100.5)])
    assert "no bar pierced" in evaluate_trigger(trig, "buy", no_pierce, 100.3, 100.4, ATR).reason
    back_under = evaluate_trigger(trig, "buy", spring, 99.7, 99.8, ATR)
    assert not back_under.fired and "back under" in back_under.reason
    upthrust = _bars([(99.6, 99.0, 99.2), (100.4, 99.3, 99.7), (99.8, 99.2, 99.5)])
    assert evaluate_trigger(trig, "sell", upthrust, 99.5, 99.6, ATR).fired


def test_close_beyond_uses_the_last_completed_close():
    up = _bars([(100.0, 99.0, 99.5), (100.6, 99.8, 100.3)])
    trig = {"kind": "close_beyond", "level": 100.0}
    assert evaluate_trigger(trig, "buy", up, 100.3, 100.4, ATR).fired
    assert not evaluate_trigger(trig, "sell", up, 100.3, 100.4, ATR).fired
    assert "extended" in evaluate_trigger(trig, "buy", up, 100.8, 100.9, ATR).reason
    assert not evaluate_trigger(trig, "buy", _bars([(100.6, 99.0, 99.9)]), 100.1, 100.2, ATR).fired


def test_missing_data_never_fires():
    trig = {"kind": "range_break", "level": 100.0}
    bars = _bars([(101.0, 99.0, 100.5)])
    assert not evaluate_trigger(trig, "buy", None, 100.0, 100.1, ATR).fired
    assert not evaluate_trigger(trig, "buy", bars, None, 100.1, ATR).fired
    assert not evaluate_trigger(trig, "buy", bars, 100.0, 100.1, None).fired
    assert not evaluate_trigger(trig, "buy", bars, 100.0, 100.1, 0).fired
    assert not evaluate_trigger({"kind": "nope", "level": 1}, "buy", bars, 100.0, 100.1, ATR).fired
    assert not evaluate_trigger(trig, "long", bars, 100.0, 100.1, ATR).fired


def test_the_extension_cap_follows_the_config(monkeypatch):
    bars = _bars([(100.5, 99.0, 100.2)])
    trig = {"kind": "range_break", "level": 100.0}
    monkeypatch.setattr(config, "STOP_ENTRY_MAX_EXTENSION_ATR", 1.5)
    assert not evaluate_trigger(trig, "buy", bars, 100.9, 101.0, ATR).fired  # still capped by the market-entry slippage guard
    monkeypatch.setattr(config, "MARKET_ENTRY_MAX_SLIPPAGE_ATR", 1.5)
    assert evaluate_trigger(trig, "buy", bars, 100.9, 101.0, ATR).fired


def test_parse_pending_setups_carries_a_valid_trigger_and_stays_backward_compatible():
    text = (
        "prose\n```json\n"
        '[{"symbol": "AAA", "side": "buy", "pct": 1.0, "trigger_condition": "M5 closes above 100", '
        '"price": 100, "stop_loss": 98, "take_profit": 104, "reason": "r", '
        '"trigger": {"kind": "range_break", "level": 100, "within_bars": 4}},'
        ' {"symbol": "BBB", "side": "sell", "pct": 1.0, "trigger_condition": "old style", "trigger": {"kind": "bogus", "level": 1}},'
        ' {"symbol": "CCC", "side": "buy", "pct": 1.0, "trigger_condition": "no trigger object"}]\n```'
    )
    setups = parse_pending_setups(text)
    assert [s.trigger for s in setups] == [{"kind": "range_break", "level": 100.0, "within_bars": 4}, None, None]
    assert setups[1].trigger_condition == "old style"
