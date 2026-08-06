from datetime import datetime

from data.mt5_source import Position
from risk.rebalance import evaluate_positions


def make_position(symbol="XAU", volume=1.0, side="buy", price_open=100.0,
                   price_current=100.0, sl=None, profit=0.0):
    return Position(
        symbol=symbol,
        volume=volume,
        side=side,
        price_open=price_open,
        price_current=price_current,
        sl=sl,
        profit=profit,
        opened_at=datetime.now(),
    )


def test_cut_loss_triggers_on_buy_beyond_threshold():
    p = make_position(price_open=100.0, price_current=85.0, sl=90.0, profit=-15.0)
    suggestions = evaluate_positions([p], max_loss_pct=10, max_position_count=10,
                                      max_symbol_exposure_pct=100)
    actions = [s.action for s in suggestions]
    assert "cut_loss" in actions


def test_cut_loss_does_not_trigger_within_threshold():
    p = make_position(price_open=100.0, price_current=95.0, sl=90.0, profit=-5.0)
    suggestions = evaluate_positions([p], max_loss_pct=10, max_position_count=10,
                                      max_symbol_exposure_pct=100)
    actions = [s.action for s in suggestions]
    assert "cut_loss" not in actions


def test_cut_loss_triggers_on_sell_beyond_threshold():
    p = make_position(side="sell", price_open=100.0, price_current=115.0, sl=110.0,
                       profit=-15.0)
    suggestions = evaluate_positions([p], max_loss_pct=10, max_position_count=10,
                                      max_symbol_exposure_pct=100)
    actions = [s.action for s in suggestions]
    assert "cut_loss" in actions


def test_no_stop_flags_missing_sl():
    p = make_position(sl=None)
    suggestions = evaluate_positions([p], max_loss_pct=100, max_position_count=10,
                                      max_symbol_exposure_pct=100)
    actions = [s.action for s in suggestions]
    assert "no_stop" in actions


def test_no_stop_does_not_flag_when_sl_present():
    p = make_position(sl=90.0)
    suggestions = evaluate_positions([p], max_loss_pct=100, max_position_count=10,
                                      max_symbol_exposure_pct=100)
    actions = [s.action for s in suggestions]
    assert "no_stop" not in actions


def test_trim_excess_count_flags_worst_positions_first():
    positions = [
        make_position(symbol=f"SYM{i}", sl=1.0, profit=float(i))
        for i in range(5)
    ]
    suggestions = evaluate_positions(positions, max_loss_pct=100, max_position_count=3,
                                      max_symbol_exposure_pct=100)
    trimmed = [s for s in suggestions if s.action == "trim_excess_count"]
    assert len(trimmed) == 2
    assert {s.symbol for s in trimmed} == {"SYM0", "SYM1"}


def test_trim_excess_count_no_trigger_under_limit():
    positions = [make_position(symbol=f"SYM{i}", sl=1.0) for i in range(3)]
    suggestions = evaluate_positions(positions, max_loss_pct=100, max_position_count=5,
                                      max_symbol_exposure_pct=100)
    assert not [s for s in suggestions if s.action == "trim_excess_count"]


def test_reduce_concentration_flags_dominant_symbol():
    positions = [
        make_position(symbol="BIG", volume=9.0, price_current=100.0, sl=1.0),
        make_position(symbol="SMALL", volume=1.0, price_current=100.0, sl=1.0),
    ]
    suggestions = evaluate_positions(positions, max_loss_pct=100, max_position_count=10,
                                      max_symbol_exposure_pct=50)
    concentrated = [s for s in suggestions if s.action == "reduce_concentration"]
    assert len(concentrated) == 1
    assert concentrated[0].symbol == "BIG"


def test_reduce_concentration_no_trigger_when_balanced():
    positions = [
        make_position(symbol="A", volume=1.0, price_current=100.0, sl=1.0),
        make_position(symbol="B", volume=1.0, price_current=100.0, sl=1.0),
    ]
    suggestions = evaluate_positions(positions, max_loss_pct=100, max_position_count=10,
                                      max_symbol_exposure_pct=60)
    assert not [s for s in suggestions if s.action == "reduce_concentration"]


def test_no_suggestions_for_empty_positions():
    assert evaluate_positions([]) == []
