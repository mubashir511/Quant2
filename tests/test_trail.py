import pytest

import config
from analysis.trail import TrailDecision, original_risk, trail_decision


def test_original_risk_prefers_the_persisted_value_then_the_recorded_stop_then_the_live_stop():
    assert original_risk(2000.0, 1996.0, 2002.0, 4.0) == 4.0
    assert original_risk(2000.0, 1996.0, 2002.0, None) == 4.0  # recorded stop
    assert original_risk(2000.0, None, 1997.0, None) == 3.0  # live stop
    assert original_risk(2000.0, None, None, None) is None
    assert original_risk(2000.0, 1996.0, 2002.0, 0.0) == 4.0  # a zero/invalid persisted value is ignored


def test_no_trail_before_plus_one_r_and_exactly_at_it_for_buy_and_sell():
    below = trail_decision("buy", 2000.0, 2003.9, 1996.0, 4.0, 2.0)
    assert below == TrailDecision(4.0, pytest.approx(0.975), None)
    at = trail_decision("buy", 2000.0, 2004.0, 1996.0, 4.0, 2.0)
    assert at.progress_r == pytest.approx(1.0) and at.new_stop == pytest.approx(2004.0 - 1.5 * 2.0)
    sell = trail_decision("sell", 2000.0, 1995.0, 2004.0, 4.0, 2.0)
    assert sell.progress_r == pytest.approx(1.25) and sell.new_stop == pytest.approx(1995.0 + 3.0)


def test_only_ratchets_by_a_real_step_and_never_widens():
    assert trail_decision("buy", 2000.0, 2006.0, 2002.9, 4.0, 2.0).new_stop is None  # 0.1 < the 0.5 step
    assert trail_decision("buy", 2000.0, 2006.0, 2002.0, 4.0, 2.0).new_stop == pytest.approx(2003.0)
    assert trail_decision("buy", 2000.0, 2006.0, 2004.0, 4.0, 2.0).new_stop is None  # already tighter
    assert trail_decision("sell", 2000.0, 1994.0, 1996.5, 4.0, 2.0).new_stop is None  # the live stop is already tighter than the candidate 1997


def test_the_brokers_minimum_stop_distance_pushes_the_stop_out():
    d = trail_decision("buy", 2000.0, 2006.0, 1996.0, 4.0, 2.0, min_stop_distance_pct=0.3)
    assert d.new_stop == pytest.approx(2006.0 - 2006.0 * 0.003 * 1.02)
    s = trail_decision("sell", 2000.0, 1994.0, 2004.0, 4.0, 2.0, min_stop_distance_pct=0.3)
    assert s.new_stop == pytest.approx(1994.0 + 1994.0 * 0.003 * 1.02)


def test_missing_inputs_and_the_switch_never_produce_a_stop(monkeypatch):
    assert trail_decision("buy", 2000.0, 2006.0, None, 4.0, 2.0).new_stop is None  # no live stop to ratchet
    assert trail_decision("buy", 2000.0, 2006.0, 1996.0, None, 2.0) == TrailDecision(None, None, None)
    assert trail_decision("buy", 2000.0, 2006.0, 1996.0, 4.0, None).new_stop is None
    monkeypatch.setattr(config, "CLERK_PROFIT_TRAIL_ENABLED", False)
    assert trail_decision("buy", 2000.0, 2006.0, 1996.0, 4.0, 2.0).new_stop is None
