import pytest

import config
from analysis.entry_mode import normalise_entry_mode, resolve_entry_mode

# A buy on a ~100.00 instrument with M5 ATR 0.20, spread 0.02: stop 99.20 (0.8 = 4 ATR), target 101.60.
_BUY = dict(side="buy", stop_loss=99.20, take_profit=101.60, bid=100.00, ask=100.02, atr=0.20)
_SELL = dict(side="sell", stop_loss=100.80, take_profit=98.40, bid=100.00, ask=100.02, atr=0.20)


def _resolve(requested, planned, base=_BUY, **kw):
    return resolve_entry_mode(requested=requested, planned_price=planned, **{**base, **kw})


def test_normalise_defaults_to_limit_for_anything_unknown():
    assert [normalise_entry_mode(v) for v in ("stop", " MARKET ", "Limit", "breakout", None, 5, "")] == [
        "stop", "market", "limit", "limit", "limit", "limit", "limit",
    ]


def test_limit_is_returned_untouched():
    d = _resolve("limit", 99.50)
    assert (d.mode, d.price, d.downgraded, d.reason) == ("limit", 99.50, False, "")


def test_kill_switch_forces_limit(monkeypatch):
    monkeypatch.setattr(config, "NEW_ENTRY_KINDS_ENABLED", False)
    d = _resolve("market", 100.02)
    assert d.mode == "limit" and d.downgraded and "kill switch" in d.reason


def test_missing_quote_or_atr_falls_back_to_limit():
    assert _resolve("market", 100.0, atr=None).mode == "limit"
    assert _resolve("market", 100.0, bid=0.0).mode == "limit"
    assert _resolve("stop", 100.0, ask=99.0).mode == "limit"  # crossed quote


def test_market_within_the_slippage_cap_fills_at_the_live_price_with_a_send_deviation():
    d = _resolve("market", 100.00)  # live ask 100.02 = 0.10 ATR worse than the plan
    assert d.mode == "market" and d.price == 100.02 and not d.downgraded
    assert d.max_deviation_price == pytest.approx(config.MARKET_ENTRY_SEND_DEVIATION_ATR * 0.20)
    assert "0.10" in d.reason


def test_market_better_than_planned_is_fine_worse_than_the_cap_is_not_chased():
    assert _resolve("market", 100.30).mode == "market"  # the market is BETTER than the plan
    chased = _resolve("market", 99.80)  # ask 100.02 is 1.1 ATR worse than the plan
    assert chased.mode == "limit" and chased.price == 99.80 and chased.downgraded and "not chasing" in chased.reason


def test_market_sell_uses_the_bid_and_mirrors_the_cap():
    ok = _resolve("market", 100.05, base=_SELL)  # bid 100.00 is 0.25 ATR worse than the plan
    assert ok.mode == "market" and ok.price == 100.00
    assert _resolve("market", 100.30, base=_SELL).mode == "limit"  # 1.5 ATR worse


def test_wide_spread_downgrades_a_market_entry():
    d = _resolve("market", 100.02, bid=99.80, ask=100.02, stop_loss=99.20)  # spread 0.22 vs stop 0.82 = 27%
    assert d.mode == "limit" and "spread" in d.reason


def test_only_one_market_entry_per_symbol_per_day():
    d = _resolve("market", 100.02, market_entry_already_used=True)
    assert d.mode == "limit" and "already used" in d.reason


def test_a_dead_setup_is_rejected_for_any_non_limit_mode():
    stopped = _resolve("market", 100.02, stop_loss=100.10)
    assert stopped.mode == "reject" and "through the stop" in stopped.reason
    done = _resolve("stop", 100.10, take_profit=100.01)
    assert done.mode == "reject" and "target" in done.reason


def test_buy_stop_above_the_market_is_placed_at_the_trigger():
    d = _resolve("stop", 100.20)  # 0.18 above the ask = 0.9 ATR
    assert d.mode == "stop" and d.price == 100.20 and not d.downgraded


def test_stop_trigger_too_close_is_nudged_to_the_broker_minimum():
    d = _resolve("stop", 100.03, min_stop_distance_price=0.10)
    assert d.mode == "stop" and d.price == pytest.approx(100.02 + 0.10 * 1.02) and "nudged" in d.reason


def test_stop_trigger_too_far_is_rejected_not_placed():
    d = _resolve("stop", 100.60)  # 0.58 = 2.9 ATR
    assert d.mode == "reject" and "too far" in d.reason


def test_stop_trigger_already_passed_and_not_extended_becomes_a_market_entry():
    d = _resolve("stop", 99.95)  # ask 100.02 is 0.07 above the trigger = 0.35 ATR (cap 0.6)
    assert d.mode == "market" and d.price == 100.02 and "already passed" in d.reason


def test_stop_trigger_already_passed_and_extended_is_not_chased():
    d = _resolve("stop", 99.70)  # 0.32 beyond = 1.6 ATR
    assert d.mode == "reject" and "extended" in d.reason


def test_sell_stop_below_the_market_and_its_passed_mirror():
    ok = _resolve("stop", 99.80, base=_SELL)  # 0.20 below the bid = 1 ATR
    assert ok.mode == "stop" and ok.price == 99.80
    passed = _resolve("stop", 100.05, base=_SELL)  # bid 100.00 is 0.05 through the trigger = 0.25 ATR
    assert passed.mode == "market" and passed.price == 100.00
