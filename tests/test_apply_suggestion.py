from datetime import datetime

import pytest

from ai.portfolio_suggest import AllocationEntry
from data.mt5_source import AccountSummary, ContractSpec, MarketAsset, Position
from risk.apply_suggestion import compute_rebalance_plan

ACCOUNT = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")


def _spec(margin_initial=10000.0, volume_min=1.0, volume_step=1.0):
    return ContractSpec(
        volume_min=volume_min, volume_step=volume_step, volume_max=100.0,
        trade_contract_size=100.0, currency_margin="USD", margin_initial=margin_initial,
    )


def _get_spec_for(specs: dict):
    return lambda symbol: specs.get(symbol)


def _position(symbol="GO10OZ", side="buy", volume=1.0, ticket=1):
    return Position(
        symbol=symbol, volume=volume, side=side, price_open=2000.0,
        price_current=2000.0, sl=None, profit=0.0, opened_at=datetime.now(), ticket=ticket,
    )


def _asset(symbol="GO10OZ", bid=1999.0, ask=2000.0):
    return MarketAsset(symbol=symbol, description=symbol, bid=bid, ask=ask)


def test_opens_a_new_position_with_clamped_price_and_stop():
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0, take_profit=2100.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert len(plan) == 1
    order = plan[0]
    assert order.action == "open"
    assert order.side == "buy"
    assert order.order_type == "limit"
    assert order.volume == 1.0  # 10% of 100k equity / 10k margin_initial = 1 lot
    assert order.price == 2000.0
    assert order.stop_loss == 1950.0
    assert order.take_profit == 2100.0


def test_drops_a_take_profit_on_the_wrong_side_of_entry():
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, take_profit=1900.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    order = plan[0]
    assert order.take_profit is None
    assert "wrong side" in order.reason.lower()
    assert "take-profit" in order.reason.lower()


def test_take_profit_survives_price_clamping_when_still_valid():
    # The take-profit check must compare against the CLAMPED entry price,
    # not the AI's original (possibly out-of-band) proposed one.
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=5000.0, take_profit=2200.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
        price_sanity_band_pct=5.0,
    )
    order = plan[0]
    assert order.price == pytest.approx(2000.0 * 1.05)  # clamped
    assert order.take_profit == 2200.0  # still above the clamped entry, so kept


def test_take_profit_absent_when_not_given():
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert plan[0].take_profit is None


def test_clamps_an_unreasonable_proposed_price_to_the_sanity_band():
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=5000.0, stop_loss=1950.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
        price_sanity_band_pct=5.0,
    )
    order = plan[0]
    assert order.price == 2000.0 * 1.05
    assert "clamped" in order.reason.lower()


def test_drops_a_stop_loss_on_the_wrong_side_of_entry():
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=2100.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    order = plan[0]
    assert order.stop_loss is None
    assert "wrong side" in order.reason.lower()


def test_increase_when_target_exceeds_current_lots():
    positions = [_position(volume=1.0, ticket=1)]
    allocation = {"GO10OZ": AllocationEntry(pct=20.0, price=2000.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    order = plan[0]
    assert order.action == "increase"
    assert order.volume == 1.0  # target 2 lots - current 1 lot


def test_reduce_when_target_is_below_current_lots():
    positions = [_position(volume=3.0, ticket=1)]
    allocation = {"GO10OZ": AllocationEntry(pct=10.0)}  # target 1 lot
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    order = plan[0]
    assert order.action == "reduce"
    assert order.side == "sell"
    assert order.order_type == "market"
    assert order.volume == 2.0
    assert order.tickets_to_close == [(1, 2.0)]


def test_close_when_target_is_zero():
    positions = [_position(volume=1.0, ticket=1)]
    allocation = {"GO10OZ": AllocationEntry(pct=0.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    order = plan[0]
    assert order.action == "close"
    assert order.tickets_to_close == [(1, 1.0)]


def test_held_symbol_missing_from_allocation_is_treated_as_close():
    positions = [_position(volume=1.0, ticket=1)]
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation={},
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    assert len(plan) == 1
    assert plan[0].action == "close"


def test_hold_when_already_at_target():
    positions = [_position(volume=1.0, ticket=1)]
    allocation = {"GO10OZ": AllocationEntry(pct=10.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    order = plan[0]
    assert order.action == "hold"


def test_infeasible_when_target_below_minimum_lot():
    allocation = {"GO10OZ": AllocationEntry(pct=1.0, price=2000.0)}  # 1% of 100k = 1000 margin
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0, volume_min=1.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    order = plan[0]
    assert order.action == "infeasible"


def test_infeasible_when_no_price_available_for_a_new_position():
    allocation = {"GO10OZ": AllocationEntry(pct=10.0)}  # no price given
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    order = plan[0]
    assert order.action == "infeasible"


def test_existing_short_position_is_closed_fully_regardless_of_target():
    positions = [_position(side="sell", volume=2.0, ticket=9)]
    allocation = {"GO10OZ": AllocationEntry(pct=15.0, price=2000.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    assert len(plan) == 1
    order = plan[0]
    assert order.action == "close"
    assert order.tickets_to_close == [(9, 2.0)]


def test_multi_ticket_reduction_consumes_tickets_until_covered():
    positions = [
        _position(volume=1.0, ticket=1),
        _position(volume=1.0, ticket=2),
        _position(volume=1.0, ticket=3),
    ]
    allocation = {"GO10OZ": AllocationEntry(pct=10.0)}  # target 1 lot, currently 3
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    order = plan[0]
    assert order.volume == 2.0
    assert order.tickets_to_close == [(1, 1.0), (2, 1.0)]


def test_cash_key_is_ignored_as_a_tradable_symbol():
    allocation = {"CASH": AllocationEntry(pct=100.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({}), market_prices={},
    )
    assert plan == []


def test_compute_rebalance_plan_is_exchange_agnostic_for_ftmo_fixtures():
    # Proves compute_rebalance_plan needs zero changes for FTMO reuse —
    # same function, same shapes, just FTMO-typical symbols/account
    # numbers (a different real MT5 account, same Position/AccountSummary/
    # ContractSpec/MarketAsset dataclasses data/mt5_source.py already
    # defines generically for any connected account) instead of PMEX's.
    ftmo_account = AccountSummary(balance=100_000.0, equity=99_500.0, free_margin=95_000.0, currency="USD")
    ftmo_positions = [_position(symbol="EURUSD", side="buy", volume=0.5, ticket=555001)]
    allocation = {
        "EURUSD": AllocationEntry(pct=20.0, price=1.1000, stop_loss=1.0950),
        "XAUUSD": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0),
    }
    specs = {
        "EURUSD": _spec(margin_initial=1000.0, volume_min=0.01, volume_step=0.01),
        "XAUUSD": _spec(margin_initial=20000.0, volume_min=0.01, volume_step=0.01),
    }
    market_prices = {
        "EURUSD": _asset(symbol="EURUSD", bid=1.0999, ask=1.1000),
        "XAUUSD": _asset(symbol="XAUUSD", bid=1999.5, ask=2000.0),
    }

    plan = compute_rebalance_plan(
        ftmo_positions, ftmo_account, allocation, _get_spec_for(specs), market_prices,
    )

    by_symbol = {o.symbol: o for o in plan}
    assert by_symbol["EURUSD"].action == "increase"  # target > currently-held 0.5 lots
    assert by_symbol["XAUUSD"].action == "open"
    assert by_symbol["XAUUSD"].price == pytest.approx(2000.0)
    assert by_symbol["XAUUSD"].stop_loss == pytest.approx(1950.0)
