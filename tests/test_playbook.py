import numpy as np
import pandas as pd
import pytest

import config
from analysis.playbook import NAME_BREAKOUT, RangeRead, advise, range_read
from analysis.position_hunter import HuntFacts, format_position_hunt, hunt


def _bars(n=150, base=100.0, volume=100.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = np.full(n, base)
    return pd.DataFrame({"Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": volume}, index=idx)


def test_range_read_uses_the_last_24_completed_bars_and_ignores_the_forming_one():
    bars = _bars()
    bars.iloc[-1, bars.columns.get_loc("High")] = 500.0  # the forming bar spikes: must not count
    bars.iloc[-6, bars.columns.get_loc("High")] = 103.0
    bars.iloc[-10, bars.columns.get_loc("Low")] = 97.0
    r = range_read(bars)
    assert (r.high, r.low, r.bars) == (103.0, 97.0, 24) and r.last_close == 100.0


def test_range_read_volume_ratio_compares_the_last_12_bars_with_the_100_before():
    bars = _bars()
    bars.iloc[-13:-1, bars.columns.get_loc("Volume")] = 300.0  # the 12 latest completed bars are 3x busier
    assert range_read(bars).vol_ratio == pytest.approx(3.0)
    assert range_read(_bars(n=60)).vol_ratio is None  # not enough history for the norm


def test_range_read_needs_bars_and_columns():
    assert range_read(None) is None and range_read(_bars(n=20)) is None
    assert range_read(pd.DataFrame({"Close": np.arange(100.0)})) is None


RNG = RangeRead(high=101.0, low=98.0, bars=24, vol_ratio=1.4, last_close=100.6)  # ATR 0.5: range = 6 ATR


def _adv(side="buy", bid=100.55, ask=100.57, atr=0.5, adx=32.0, chg=4.0, rng=RNG):
    return advise(side, bid, ask, atr, adx, chg, rng)


def test_buy_stop_above_the_range_high_with_the_structure_stop_at_the_range_low():
    a = _adv()
    assert (a.name, a.status, a.order) == (NAME_BREAKOUT, "ACTIVE", "stop")
    assert a.entry == 101.0 and a.trigger_ahead_atr == pytest.approx((101.0 - 100.57) / 0.5)
    # structure distance |101-98| = 3.0 = 6 ATR -> clipped to the 4 ATR cap = 2.0
    assert a.stop == pytest.approx(99.0) and a.stop_atr == pytest.approx(4.0)
    assert a.target == pytest.approx(101.0 + 2 * 2.0)


def test_the_structure_stop_is_never_tighter_than_the_floor():
    tight = RangeRead(high=101.0, low=100.6, bars=24, vol_ratio=1.2, last_close=100.8)  # 0.8 ATR range
    a = advise("buy", 100.7, 100.72, 0.5, 30.0, 1.0, tight)
    assert a.stop_atr == pytest.approx(config.PLAYBOOK_STOP_MIN_ATR) and a.stop == pytest.approx(101.0 - 0.75)


def test_sell_mirrors_the_buy():
    a = advise("sell", 98.4, 98.42, 0.5, 32.0, 4.0, RNG)
    assert (a.status, a.order, a.entry) == ("ACTIVE", "stop", 98.0)
    assert a.stop == pytest.approx(100.0) and a.target == pytest.approx(98.0 - 4.0)


def test_price_already_through_the_trigger_but_not_extended_becomes_a_market_entry():
    a = _adv(bid=101.1, ask=101.12)  # 0.24 ATR beyond the trigger (cap 0.6)
    assert (a.status, a.order) == ("ACTIVE", "market") and a.entry == pytest.approx(101.12) and a.trigger_ahead_atr == 0.0


def test_an_extended_breakout_is_not_chased_and_a_far_trigger_waits():
    extended = _adv(bid=101.5, ask=101.52)  # 1.04 ATR through
    assert extended.status == "EXTENDED" and "secondary consolidation" in extended.text()
    far = _adv(bid=99.0, ask=99.02)  # trigger 3.96 ATR away
    assert far.status == "WAIT" and "range_break" in far.text()


def test_no_trend_strength_or_missing_data_is_not_supported_never_guessed():
    for kwargs in (dict(atr=None), dict(rng=None), dict(adx=None), dict(bid=0.0)):
        assert _adv(**kwargs).status == "NOT_SUPPORTED"
    assert advise("buy", 100.6, 100.5, 0.5, 30, 1, RNG).status == "NOT_SUPPORTED"  # crossed quote


def test_adx_is_a_quality_dial_by_default_and_a_gate_only_when_configured(monkeypatch):
    weak = _adv(adx=14.0)
    assert weak.status == "ACTIVE" and dict((n, ok) for n, ok, _ in weak.filters)["trend strength"] is False
    assert "weakest bucket" in weak.text()
    monkeypatch.setattr(config, "PLAYBOOK_MIN_ADX", 20.0)
    gated = _adv(adx=14.0)
    assert gated.status == "NOT_SUPPORTED" and "ADX 14" in gated.text()


def test_filters_report_adx_slope_and_volume_and_flag_the_weak_bucket():
    good = _adv(adx=45.0, chg=3.0)
    texts = {name: (ok, text) for name, ok, text in good.filters}
    assert texts["trend strength"] == (True, "ADX 45 (>40: strongest bucket)")
    assert texts["ADX slope"][0] is True and texts["volume"][0] is True
    weak = advise("buy", 100.55, 100.57, 0.5, 25.0, -3.0, RangeRead(101.0, 98.0, 24, 0.5, 100.6))
    weak_texts = {name: ok for name, ok, _ in weak.filters}
    assert weak_texts["ADX slope"] is False and weak_texts["volume"] is False
    assert "WEAK" in weak.text() and "weakest measured bucket" in weak.text()


def test_the_prompt_line_carries_every_number_and_the_measured_result():
    text = _adv().text()
    for needle in ("buy-STOP at 101", "structure stop 99", "4.0 M5 ATR", "2R target 105", "ADX 32", "+0.086R gross", "+0.077R net", 'entry_mode "stop"'):
        assert needle in text, needle


# ---- integration with the hunter ----------------------------------------------------------------------------------

def _facts(symbol="EURUSD", **kw):
    defaults = dict(d1_dir="up", h4_dir="up", m5_dir="up", cost_drag_r=0.05, has_structure=True, zone_rr={"buy": 2.5}, market_open=True,
                    playbook={"buy": _adv(), "sell": advise("sell", 100.55, 100.57, 0.5, 32.0, 4.0, RNG)})
    defaults.update(kw)
    return HuntFacts(symbol, **defaults)


def test_the_hunt_prints_the_playbook_line_under_the_chosen_side():
    text = format_position_hunt(hunt([_facts()]))
    assert "1. EURUSD BUY [tier A]" in text and "TREND-BREAKOUT: buy-STOP at 101" in text
    assert "measured only for trades WITH" not in text


def test_a_non_aligned_candidate_gets_the_unmeasured_caveat():
    text = format_position_hunt(hunt([_facts(d1_dir="flat", h4_dir="flat")]))
    assert "TREND-BREAKOUT" in text and "against it or mixed: +0.017..+0.038R - smaller, still positive" in text


def test_no_playbook_means_no_line_and_the_old_output_is_unchanged():
    text = format_position_hunt(hunt([_facts(playbook={})]))
    assert "TREND-BREAKOUT" not in text and "1. EURUSD BUY" in text
