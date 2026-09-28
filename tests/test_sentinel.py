from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import config
from ai import clerk_execution
from ai.portfolio_suggest import AllocationEntry
from ai.sentinel import (
    apply_sentinel_ratchet,
    ingest_sentinel_stops,
    plan_trail_actions,
    read_sentinel_state,
    run_sentinel_check,
)
from data.mt5_execution import OrderResult
from data.mt5_source import Position

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


def _position(side="buy", price_open=2000.0, price_current=2006.0, sl=1996.0, ticket=11, symbol="XAUUSD"):
    return Position(
        symbol=symbol, volume=1.0, side=side, price_open=price_open, price_current=price_current, sl=sl,
        profit=0.0, opened_at=NOW, ticket=ticket, tp=2020.0,
    )


def _settled(stop=1996.0, tactical=None, origin=None):
    rec = {"state": "filled", "entry": {"pct": 1.0, "price": 2000.0, "stop_loss": stop, "take_profit": 2020.0, "side": "buy", "reason": ""}}
    if tactical is not None:
        rec["tactical"] = tactical
    if origin:
        rec["origin"] = origin
    return {"XAUUSD": rec}


def _pct(symbol, lots, entry_price, stop, equity, get_spec):
    return abs(entry_price - stop) * lots / equity * 100  # any deterministic stand-in for the real inverse sizing


def test_plan_trails_a_position_past_plus_one_r_and_carries_the_sizing_pct():
    actions = plan_trail_actions([_position()], _settled(), {"XAUUSD": 2.0}, {"XAUUSD": 0.0}, {}, 100000.0, None, _pct)
    assert len(actions) == 1
    a = actions[0]
    assert a.new_stop == pytest.approx(2006.0 - 1.5 * 2.0) and a.old_stop == 1996.0
    assert a.r0 == pytest.approx(4.0) and a.progress_r == pytest.approx(1.5)
    assert a.sizing_pct == pytest.approx(abs(2000.0 - a.new_stop) / 100000.0 * 100)


def test_plan_skips_below_plus_one_r_unmanaged_positions_and_unsizable_stops():
    assert plan_trail_actions([_position(price_current=2003.0)], _settled(), {"XAUUSD": 2.0}, {}, {}, 1e5, None, _pct) == []
    assert plan_trail_actions([_position()], {}, {"XAUUSD": 2.0}, {}, {}, 1e5, None, _pct) == []  # no settlement record
    assert plan_trail_actions([_position()], _settled(), {"XAUUSD": 2.0}, {}, {}, 1e5, None, lambda *a: None) == []


def test_plan_leaves_an_unexplained_live_stop_to_the_clerk_but_accepts_its_own_uningested_stop():
    drifted = _position(sl=1990.0)  # recorded is 1996
    assert plan_trail_actions([drifted], _settled(), {"XAUUSD": 2.0}, {}, {}, 1e5, None, _pct) == []
    own = {"XAUUSD": {"ticket": 11, "stop": 1990.0, "r0": 4.0}}  # the Sentinel itself put it there last minute
    assert len(plan_trail_actions([drifted], _settled(), {"XAUUSD": 2.0}, {}, own, 1e5, None, _pct)) == 1


def test_r0_is_not_shrunk_by_a_tightened_recorded_stop():
    tactical = {"profit_trail_r0": 4.0, "persisted_stop_loss": 2001.0}
    pos = _position(price_current=2008.0, sl=2001.0)
    a = plan_trail_actions([pos], _settled(stop=1996.0, tactical=tactical), {"XAUUSD": 2.0}, {}, {}, 1e5, None, _pct)[0]
    assert a.r0 == 4.0 and a.progress_r == pytest.approx(2.0)


def test_ingest_records_a_stop_the_live_position_really_carries_and_only_if_more_protective():
    settled = _settled()
    state = {"XAUUSD": {"ticket": 11, "stop": 2003.0, "r0": 4.0, "utc": "t"}}
    out, notes = ingest_sentinel_stops(settled, {"XAUUSD": _position(sl=2003.0)}, state, 1e5, None, _pct)
    t = out["XAUUSD"]["tactical"]
    assert t["sentinel_stop"] == 2003.0 and t["profit_trail_r0"] == 4.0 and t["sentinel_pct"] == pytest.approx(_pct("", 1.0, 2000.0, 2003.0, 1e5, None))
    assert notes and "ingested" in notes[0]
    # idempotent: an already recorded stop is not re-ingested
    assert ingest_sentinel_stops(out, {"XAUUSD": _position(sl=2003.0)}, state, 1e5, None, _pct)[1] == []
    # another ticket, or a live stop that is not the Sentinel's, is ignored
    assert ingest_sentinel_stops(_settled(), {"XAUUSD": _position(sl=2003.0, ticket=99)}, state, 1e5, None, _pct)[1] == []
    assert ingest_sentinel_stops(_settled(), {"XAUUSD": _position(sl=1996.0)}, state, 1e5, None, _pct)[1] == []
    # a stop LOOSER than the recorded one is never ingested
    assert ingest_sentinel_stops(_settled(stop=2005.0), {"XAUUSD": _position(sl=2003.0)}, state, 1e5, None, _pct)[1] == []
    assert ingest_sentinel_stops(_settled(), {"XAUUSD": _position(sl=2003.0)}, state, 1e5, None, lambda *a: None)[1][0].endswith("left for the drift check")


def test_apply_ratchet_only_tightens_and_only_with_a_sizing_pct():
    entry = AllocationEntry(pct=1.0, price=2000.0, stop_loss=1996.0, take_profit=2020.0, side="buy")
    out = apply_sentinel_ratchet(entry, {"sentinel_stop": 2003.0, "sentinel_pct": 0.4})
    assert (out.stop_loss, out.pct, out.take_profit) == (2003.0, 0.4, 2020.0)
    assert apply_sentinel_ratchet(entry, {"sentinel_stop": 1990.0, "sentinel_pct": 0.4}) is entry  # looser
    assert apply_sentinel_ratchet(entry, {"sentinel_stop": 2003.0}) is entry
    assert apply_sentinel_ratchet(entry, None) is entry
    sell = AllocationEntry(pct=1.0, stop_loss=2004.0, side="sell")
    assert apply_sentinel_ratchet(sell, {"sentinel_stop": 1997.0, "sentinel_pct": 0.5}).stop_loss == 1997.0
    assert apply_sentinel_ratchet(sell, {"sentinel_stop": 2010.0, "sentinel_pct": 0.5}) is sell


def test_clerk_carried_forward_and_drift_use_the_ingested_sentinel_stop():
    tactical = {"sentinel_stop": 2003.0, "sentinel_pct": 0.4, "profit_trail_r0": 4.0}
    settled = _settled(tactical=tactical, origin="pending_setup")
    carried = clerk_execution._build_carried_forward_allocation({}, settled, held_symbols=frozenset({"XAUUSD"}))
    assert (carried["XAUUSD"].stop_loss, carried["XAUUSD"].pct) == (2003.0, 0.4)
    live = {"XAUUSD": _position(sl=2003.0)}
    assert clerk_execution._iter_stop_drift(live, settled, carried) == []  # the trailed stop IS the recorded one
    immediate = {"XAUUSD": {"pct": 1.0, "price": 2000.0, "stop_loss": 1996.0, "take_profit": 2020.0, "side": "buy", "reason": ""}}
    carried2 = clerk_execution._build_carried_forward_allocation(immediate, _settled(tactical=tactical), held_symbols=frozenset({"XAUUSD"}))
    assert (carried2["XAUUSD"].stop_loss, carried2["XAUUSD"].pct) == (2003.0, 0.4)


def _deps(positions, settled, modify=None, atr=2.0):
    return dict(
        positions_fn=lambda: positions, settlement_fn=lambda: {"settled": settled}, atr_fn=lambda s: atr,
        min_stop_pct_fn=lambda s: 0.0, equity_fn=lambda: 1e5, spec_fn=lambda s: SimpleNamespace(trade_contract_size=100.0),
        modify_fn=modify or (lambda *a: OrderResult(True, 10009, "ok", 11)),
    )


def test_run_check_log_only_never_modifies_anything(monkeypatch):
    calls = []
    monkeypatch.setattr("risk.apply_suggestion.pct_for_target_lots", _pct)
    deps = _deps([_position()], _settled(), modify=lambda *a: calls.append(a))
    summary = run_sentinel_check(NOW, log_only=True, **deps)
    assert calls == [] and len(summary["actions"]) == 1 and summary["actions"][0]["applied"] is False
    assert read_sentinel_state()["stops"] == {}


def test_run_check_live_modifies_and_records_the_stop_for_the_clerk(monkeypatch):
    calls = []
    monkeypatch.setattr("risk.apply_suggestion.pct_for_target_lots", _pct)

    def modify(position, stop, tp):
        calls.append((position.symbol, stop, tp))
        return OrderResult(True, 10009, "ok", position.ticket)

    summary = run_sentinel_check(NOW, log_only=False, **_deps([_position()], _settled(), modify=modify))
    assert calls == [("XAUUSD", 2003.0, 2020.0)] and summary["actions"][0]["applied"] is True
    stop = read_sentinel_state()["stops"]["XAUUSD"]
    assert stop["stop"] == 2003.0 and stop["ticket"] == 11 and stop["r0"] == pytest.approx(4.0)


def test_run_check_a_rejected_modify_is_not_recorded_and_closed_positions_are_forgotten(monkeypatch):
    monkeypatch.setattr("risk.apply_suggestion.pct_for_target_lots", _pct)
    summary = run_sentinel_check(
        NOW, log_only=False, **_deps([_position()], _settled(), modify=lambda *a: OrderResult(False, 10016, "invalid stops", None))
    )
    assert summary["actions"][0]["applied"] is False and read_sentinel_state()["stops"] == {}
    from ai.sentinel import write_sentinel_state

    write_sentinel_state({"stops": {"GONE": {"ticket": 1, "stop": 1.0, "r0": 1.0}}})
    run_sentinel_check(NOW, log_only=True, **_deps([], _settled()))
    assert read_sentinel_state()["stops"] == {}


def test_defaults_are_the_approved_log_only_phase():
    assert config.SENTINEL_LOG_ONLY is True and config.SENTINEL_ENABLED is True
