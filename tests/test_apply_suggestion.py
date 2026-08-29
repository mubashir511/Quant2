from datetime import datetime

import pytest

from ai.portfolio_suggest import AllocationEntry
from data.mt5_source import AccountSummary, ContractSpec, MarketAsset, PendingOrder, Position
from risk.apply_suggestion import check_execution_safety_gates, compute_rebalance_plan, pct_for_target_lots

ACCOUNT = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")


def _spec(margin_initial=10000.0, volume_min=1.0, volume_step=1.0):
    return ContractSpec(
        volume_min=volume_min, volume_step=volume_step, volume_max=100.0,
        trade_contract_size=100.0, currency_margin="USD", margin_initial=margin_initial,
    )


def _get_spec_for(specs: dict):
    return lambda symbol: specs.get(symbol)


def _position(symbol="GO10OZ", side="buy", volume=1.0, ticket=1, sl=None, tp=None):
    return Position(
        symbol=symbol, volume=volume, side=side, price_open=2000.0,
        price_current=2000.0, sl=sl, profit=0.0, opened_at=datetime.now(), ticket=ticket, tp=tp,
    )


def _asset(symbol="GO10OZ", bid=1999.0, ask=2000.0):
    return MarketAsset(symbol=symbol, description=symbol, bid=bid, ask=ask)


def _pending_order(symbol="GO10OZ", order_type="buy limit", volume=1.0, price_open=2000.0, sl=None, tp=None, ticket=501):
    return PendingOrder(
        symbol=symbol, volume=volume, order_type=order_type, price_open=price_open,
        sl=sl, tp=tp, ticket=ticket,
    )


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
    # Risk-based sizing: 10% of 100k equity = $10,000 risked, stop is 50
    # points away, contract_size=100 -> $10,000 / (50 x 100) = 2 lots.
    assert order.volume == 2.0
    assert order.price == 2000.0
    assert order.stop_loss == 1950.0
    assert order.take_profit == 2100.0


def test_drops_a_take_profit_on_the_wrong_side_of_entry():
    # A stop_loss is required now to size (and even open) a position at
    # all — risk-based sizing has nothing to size against without one.
    allocation = {
        "GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0, take_profit=1900.0)
    }
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    order = plan[0]
    assert order.action == "open"
    assert order.take_profit is None
    assert "wrong side" in order.reason.lower()
    assert "take-profit" in order.reason.lower()


def test_take_profit_survives_price_clamping_when_still_valid():
    # The take-profit check must compare against the CLAMPED entry price,
    # not the AI's original (possibly out-of-band) proposed one.
    allocation = {
        "GO10OZ": AllocationEntry(pct=10.0, price=5000.0, stop_loss=2050.0, take_profit=2200.0)
    }
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
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert plan[0].take_profit is None


def test_clamps_an_unreasonable_proposed_price_to_the_sanity_band():
    # stop_loss picked so the CLAMPED price (2100) still leaves a stop
    # distance affordable at this pct/equity/contract_size (50 points ->
    # 2 lots) — a too-wide distance would make this genuinely infeasible
    # by real risk math, which isn't what this test is checking.
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=5000.0, stop_loss=2050.0)}
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
    # A wrong-side (or missing) stop now makes the WHOLE position
    # infeasible, not just an unstopped order sized by some other means —
    # risk-based sizing has nothing to size against without a real stop,
    # and sending an order with no stop at all is an unbounded-risk
    # order this pipeline now deliberately refuses to place.
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=2100.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    order = plan[0]
    assert order.action == "infeasible"
    assert order.stop_loss is None
    assert "wrong side" in order.reason.lower()


def test_increase_when_target_exceeds_current_lots():
    positions = [_position(volume=1.0, ticket=1)]
    # stop 100 points away: 20% of 100k = $20,000 risked / (100 x 100) = 2
    # lots target, vs. 1 lot currently held -> increase by 1.
    allocation = {"GO10OZ": AllocationEntry(pct=20.0, price=2000.0, stop_loss=1900.0)}
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
    # stop 100 points away: 10% of 100k = $10,000 risked / (100 x 100) = 1
    # lot target, vs. 3 lots currently held -> reduce by 2.
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1900.0)}
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
    # sl matches the fresh target's own stop_loss exactly — otherwise
    # this would now (correctly) resolve to "amend_position" instead,
    # since the position's real stop no longer matches what's suggested.
    positions = [_position(volume=1.0, ticket=1, sl=1900.0)]
    # stop 100 points away: 10% of 100k / (100 x 100) = 1 lot target,
    # matching the 1 lot already held.
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1900.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    order = plan[0]
    assert order.action == "hold"


def test_infeasible_when_target_below_minimum_lot():
    # 1% of 100k = $1,000 risked; stop 50 points away / contract_size=100
    # -> 0.2 lots target, below the 1.0 minimum lot for this instrument.
    allocation = {"GO10OZ": AllocationEntry(pct=1.0, price=2000.0, stop_loss=1950.0)}
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


def test_genuinely_mixed_both_sides_position_is_closed_fully():
    # A real hedge-mode anomaly (both a long AND a short open on the same
    # symbol at once) — still an unconditional full close, regardless of
    # target, same as before. NOT the same case as a clean single-
    # direction short matching its own target (see the next test).
    positions = [_position(side="buy", volume=1.0, ticket=1), _position(side="sell", volume=2.0, ticket=9)]
    allocation = {"GO10OZ": AllocationEntry(side="buy", pct=15.0, price=2000.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    assert len(plan) == 1
    order = plan[0]
    assert order.action == "close"
    assert order.volume == 3.0
    assert order.tickets_to_close == [(1, 1.0), (9, 2.0)]


def test_existing_short_with_matching_short_target_is_resized_not_closed():
    # The core behavior this whole change replaces the old blanket
    # "any short gets closed" rule with: a clean, single-direction short
    # that matches its own target gets resized like a long would, not
    # force-closed just for being a short.
    positions = [_position(side="sell", volume=2.0, ticket=9)]
    # short stop 50 pts above entry: 15% of 100k / (50 x 100) = 3 lots
    # target, vs. 2 lots held -> increase by 1, not close.
    allocation = {"GO10OZ": AllocationEntry(side="sell", pct=15.0, price=1999.0, stop_loss=2049.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    assert len(plan) == 1
    order = plan[0]
    assert order.action == "increase"
    assert order.side == "sell"
    assert order.volume == 1.0


def test_opens_a_new_short_position_with_clamped_price_and_stop():
    # Mirrors test_opens_a_new_position_with_clamped_price_and_stop, short
    # side: stop ABOVE entry, target BELOW entry, sized off the live BID.
    allocation = {
        "GO10OZ": AllocationEntry(side="sell", pct=10.0, price=1999.0, stop_loss=2049.0, take_profit=1899.0)
    }
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(bid=1999.0, ask=2000.0)},
    )
    assert len(plan) == 1
    order = plan[0]
    assert order.action == "open"
    assert order.side == "sell"
    assert order.order_type == "limit"
    # 10% of 100k = $10,000 risked, stop is 50 points away, contract_size=100
    # -> $10,000 / (50 x 100) = 2 lots.
    assert order.volume == 2.0
    assert order.price == 1999.0
    assert order.stop_loss == 2049.0
    assert order.take_profit == 1899.0


def test_short_price_clamping_uses_bid_not_ask():
    # A short's limit-entry reference price must be the live BID, not the
    # ASK — the exact defect class this test protects against would use
    # the wrong side of the spread when clamping a short's entry. Stop is
    # close enough to the expected clamped price (~2098.95) to stay
    # affordable at the minimum lot — this test is about WHICH reference
    # price gets used, not about affordability.
    allocation = {"GO10OZ": AllocationEntry(side="sell", pct=10.0, price=2200.0, stop_loss=2150.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(bid=1999.0, ask=2000.0)},
        price_sanity_band_pct=5.0,
    )
    order = plan[0]
    # Clamped to 5% above the BID (1999 * 1.05), NOT 5% above the ASK
    # (2000 * 1.05 = 2100) — those two numbers are deliberately different
    # so a bid/ask mixup would fail this assertion.
    assert order.price == pytest.approx(1999.0 * 1.05)
    assert "bid" in order.reason.lower()
    assert "1999" in order.reason


def test_drops_a_short_stop_loss_on_the_wrong_side_of_entry():
    # For a short, the stop must be ABOVE entry — a stop below entry
    # (the long-side rule) is wrong here and must be dropped, making the
    # whole position infeasible (nothing left to size risk against).
    allocation = {"GO10OZ": AllocationEntry(side="sell", pct=10.0, price=1999.0, stop_loss=1950.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(bid=1999.0, ask=2000.0)},
    )
    order = plan[0]
    assert order.action == "infeasible"
    assert order.stop_loss is None
    assert "wrong side" in order.reason.lower()


def test_drops_a_short_take_profit_on_the_wrong_side_of_entry():
    # For a short, the target must be BELOW entry — one above entry (the
    # long-side rule) is wrong here and must be dropped, but the position
    # itself stays feasible (the stop alone is enough to size on).
    allocation = {
        "GO10OZ": AllocationEntry(side="sell", pct=10.0, price=1999.0, stop_loss=2049.0, take_profit=2100.0)
    }
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(bid=1999.0, ask=2000.0)},
    )
    order = plan[0]
    assert order.action == "open"
    assert order.take_profit is None
    assert "wrong side" in order.reason.lower()
    assert "take-profit" in order.reason.lower()


def test_reduce_short_when_target_is_below_current_short_lots():
    positions = [_position(side="sell", volume=3.0, ticket=1)]
    # short stop 50 pts above entry: 10% of 100k / (50 x 100) = 2 lots
    # target, vs. 3 lots held -> reduce by 1, buying back to cover.
    allocation = {"GO10OZ": AllocationEntry(side="sell", pct=10.0, price=1999.0, stop_loss=2049.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(bid=1999.0, ask=2000.0)},
    )
    order = plan[0]
    assert order.action == "reduce"
    assert order.side == "buy"  # buying back covers a short
    assert order.order_type == "market"
    assert order.volume == 1.0
    assert order.tickets_to_close == [(1, 1.0)]


def test_hold_short_when_already_at_target():
    # sl matches the fresh target's own stop_loss — see
    # test_hold_when_already_at_target's own comment for why this matters now.
    positions = [_position(side="sell", volume=2.0, ticket=1, sl=2049.0)]
    allocation = {"GO10OZ": AllocationEntry(side="sell", pct=10.0, price=1999.0, stop_loss=2049.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(bid=1999.0, ask=2000.0)},
    )
    order = plan[0]
    assert order.action == "hold"
    assert order.side == "sell"


def test_close_short_when_target_is_zero():
    positions = [_position(side="sell", volume=2.0, ticket=1)]
    allocation = {"GO10OZ": AllocationEntry(pct=0.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset()},
    )
    order = plan[0]
    assert order.action == "close"
    assert order.side == "buy"  # buying back closes a short
    assert order.tickets_to_close == [(1, 2.0)]


def test_flip_long_to_short_produces_close_then_open_pair():
    positions = [_position(side="buy", volume=2.0, ticket=5)]
    # target risks 15% (vs whatever the held long was), deliberately a
    # DIFFERENT number of lots than what's held, so a netted-against-the-
    # close bug (e.g. "3 - 2 = 1" or "3 + 2 = 5") would fail this test.
    allocation = {"GO10OZ": AllocationEntry(side="sell", pct=15.0, price=1999.0, stop_loss=2049.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(bid=1999.0, ask=2000.0)},
    )
    assert len(plan) == 2
    close, open_ = plan[0], plan[1]
    assert close.action == "close"
    assert close.side == "sell"  # closes a long with a sell
    assert close.volume == 2.0
    assert close.tickets_to_close == [(5, 2.0)]
    assert open_.action == "open"
    assert open_.side == "sell"
    # 15% of 100k / (50 x 100) = 3 lots — fresh, independent of the 2 closed.
    assert open_.volume == 3.0


def test_flip_short_to_long_produces_close_then_open_pair():
    positions = [_position(side="sell", volume=2.0, ticket=7)]
    allocation = {"GO10OZ": AllocationEntry(side="buy", pct=15.0, price=2000.0, stop_loss=1950.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(bid=1999.0, ask=2000.0)},
    )
    assert len(plan) == 2
    close, open_ = plan[0], plan[1]
    assert close.action == "close"
    assert close.side == "buy"  # closes a short with a buy
    assert close.volume == 2.0
    assert close.tickets_to_close == [(7, 2.0)]
    assert open_.action == "open"
    assert open_.side == "buy"
    # 15% of 100k / (50 x 100) = 3 lots — fresh, independent of the 2 closed.
    assert open_.volume == 3.0


def test_multi_ticket_reduction_consumes_tickets_until_covered():
    positions = [
        _position(volume=1.0, ticket=1),
        _position(volume=1.0, ticket=2),
        _position(volume=1.0, ticket=3),
    ]
    # stop 100 points away: 10% of 100k / (100 x 100) = 1 lot target,
    # currently 3.
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1900.0)}
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


def test_missing_contract_spec_marks_a_flat_target_infeasible_without_crashing():
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({}),  # no spec available for GO10OZ
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert len(plan) == 1
    assert plan[0].action == "infeasible"
    assert plan[0].volume == 0.0


def test_missing_contract_spec_still_force_closes_a_held_position():
    # Old behavior unconditionally closed any held short regardless of
    # spec availability; the risk-based rewrite generalized this to both
    # directions but the spec-missing check briefly ran BEFORE the
    # close/flip logic, silently leaving a held position open forever
    # instead of closing it (a real regression, since sizing a fresh
    # order needs a spec but closing an existing one doesn't).
    positions = [_position(symbol="GO10OZ", side="sell", volume=2.0, ticket=42)]
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=2050.0, side="sell")}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({}),  # spec unavailable
        market_prices={"GO10OZ": _asset(bid=1999.0)},
    )
    assert len(plan) == 1
    order = plan[0]
    assert order.action == "close"
    assert order.side == "buy"  # closes a held short via a buy
    assert order.volume == 2.0
    assert order.tickets_to_close == [(42, 2.0)]


def test_zero_contract_size_spec_marks_infeasible_instead_of_crashing():
    # trade_contract_size is a divisor in the risk-based sizing formula
    # (risk_dollars / (stop_distance * trade_contract_size)) — a
    # malformed spec with margin_initial > 0 but trade_contract_size <= 0
    # would previously raise an uncaught ZeroDivisionError and crash the
    # whole Apply Suggestion flow instead of degrading gracefully like
    # every other malformed-spec case already does.
    bad_spec = ContractSpec(
        volume_min=0.01, volume_step=0.01, volume_max=100.0,
        trade_contract_size=0.0, currency_margin="USD", margin_initial=10000.0,
    )
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": bad_spec}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert len(plan) == 1
    assert plan[0].action == "infeasible"


# --- check_execution_safety_gates ---


def test_check_execution_safety_gates_passes_when_everything_clears():
    ok, reason = check_execution_safety_gates(
        is_demo=True, allow_live_execution=False, trading_permitted=True,
    )
    assert ok is True
    assert reason == ""


def test_check_execution_safety_gates_blocks_non_demo_without_override():
    ok, reason = check_execution_safety_gates(
        is_demo=False, allow_live_execution=False, trading_permitted=True,
    )
    assert ok is False
    assert "doesn't" in reason
    assert "ALLOW_LIVE_EXECUTION" in reason


def test_check_execution_safety_gates_allows_non_demo_with_explicit_override():
    ok, reason = check_execution_safety_gates(
        is_demo=False, allow_live_execution=True, trading_permitted=True,
    )
    assert ok is True
    assert reason == ""


def test_check_execution_safety_gates_blocks_on_ftmo_heat():
    ok, reason = check_execution_safety_gates(
        is_demo=True, allow_live_execution=False, trading_permitted=True,
        ftmo_heat_blocked=True, ftmo_heat_blocked_reason="Execution blocked: heat too high.",
    )
    assert ok is False
    assert reason == "Execution blocked: heat too high."


def test_check_execution_safety_gates_blocks_when_trading_not_permitted():
    ok, reason = check_execution_safety_gates(
        is_demo=True, allow_live_execution=False, trading_permitted=False,
        trading_blocked_reason="AutoTrading is turned OFF.",
    )
    assert ok is False
    assert reason == "Execution blocked: AutoTrading is turned OFF."


def test_check_execution_safety_gates_demo_check_takes_priority_over_others():
    # First-failure-wins ordering: even if heat is also blocked and
    # trading isn't permitted, the demo/ALLOW_LIVE_EXECUTION message is
    # the one surfaced -- matches the original inline dialog's own
    # if/elif/elif priority order exactly.
    ok, reason = check_execution_safety_gates(
        is_demo=False, allow_live_execution=False, trading_permitted=False,
        trading_blocked_reason="AutoTrading is turned OFF.",
        ftmo_heat_blocked=True, ftmo_heat_blocked_reason="Execution blocked: heat too high.",
    )
    assert ok is False
    assert "ALLOW_LIVE_EXECUTION" in reason


def test_check_execution_safety_gates_heat_check_takes_priority_over_trading_permission():
    ok, reason = check_execution_safety_gates(
        is_demo=True, allow_live_execution=False, trading_permitted=False,
        trading_blocked_reason="AutoTrading is turned OFF.",
        ftmo_heat_blocked=True, ftmo_heat_blocked_reason="Execution blocked: heat too high.",
    )
    assert ok is False
    assert reason == "Execution blocked: heat too high."


# --- compute_aggregate_heat_pct ---


def test_compute_aggregate_heat_pct_sums_priced_and_stopped_entries():
    from risk.apply_suggestion import compute_aggregate_heat_pct

    allocation = {
        "EURUSD": AllocationEntry(pct=1.5, price=1.09, stop_loss=1.08),
        "XAUUSD": AllocationEntry(pct=2.0, price=2000.0, stop_loss=1980.0),
        "CASH": AllocationEntry(pct=96.5),
    }
    assert compute_aggregate_heat_pct(allocation) == 3.5


def test_compute_aggregate_heat_pct_ignores_entries_missing_price_or_stop():
    from risk.apply_suggestion import compute_aggregate_heat_pct

    allocation = {
        "EURUSD": AllocationEntry(pct=1.5, price=None, stop_loss=1.08),
        "XAUUSD": AllocationEntry(pct=2.0, price=2000.0, stop_loss=None),
    }
    assert compute_aggregate_heat_pct(allocation) == 0.0


# --- pending-order awareness (added 2026-08-23) ---


def test_stale_pending_order_cancelled_when_target_pct_is_zero():
    order = _pending_order(price_open=2000.0, sl=1900.0, ticket=777)
    allocation = {"GO10OZ": AllocationEntry(pct=0.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
        pending_orders=[order],
    )
    assert len(plan) == 1
    assert plan[0].action == "cancel"
    assert plan[0].pending_tickets_to_cancel == [777]


def test_pending_order_held_when_target_matches_existing_terms_exactly():
    order = _pending_order(price_open=2000.0, sl=1900.0, tp=None, volume=1.0, ticket=777)
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1900.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
        pending_orders=[order],
    )
    assert len(plan) == 1
    assert plan[0].action == "hold"


def test_pending_order_amended_when_price_changed():
    order = _pending_order(price_open=1990.0, sl=1900.0, ticket=777)
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=1995.0, stop_loss=1895.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
        pending_orders=[order],
    )
    assert len(plan) == 1
    assert plan[0].action == "amend_pending"
    assert plan[0].pending_tickets_to_cancel == [777]
    assert plan[0].price == 1995.0


def test_pending_order_amended_when_stop_loss_changed_only():
    order = _pending_order(price_open=2000.0, sl=1900.0, ticket=777)
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1800.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0, volume_min=0.01, volume_step=0.01)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
        pending_orders=[order],
    )
    assert len(plan) == 1
    assert plan[0].action == "amend_pending"


def test_pending_order_amended_when_side_flips_while_still_unfilled():
    order = _pending_order(order_type="buy limit", price_open=2000.0, sl=1900.0, ticket=777)
    allocation = {"GO10OZ": AllocationEntry(side="sell", pct=10.0, price=1999.0, stop_loss=2099.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(bid=1999.0, ask=2000.0)},
        pending_orders=[order],
    )
    assert len(plan) == 1
    assert plan[0].action == "amend_pending"
    assert plan[0].side == "sell"
    assert plan[0].pending_tickets_to_cancel == [777]


def test_multiple_resting_pending_orders_on_same_symbol_never_resolve_to_hold():
    orders = [
        _pending_order(price_open=2000.0, sl=1900.0, ticket=777),
        _pending_order(price_open=2000.0, sl=1900.0, ticket=778),
    ]
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1900.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
        pending_orders=orders,
    )
    assert len(plan) == 1
    assert plan[0].action == "amend_pending"
    assert set(plan[0].pending_tickets_to_cancel) == {777, 778}


def test_amend_tolerance_pct_prevents_thrash_on_trivial_price_differences():
    # price and stop_loss both shift by the same tiny +0.20 amount, so
    # stop_distance (and therefore lot sizing) is completely unaffected --
    # isolating this test to the tolerance check itself, not lot-size
    # rounding drift from a changed stop distance.
    order = _pending_order(price_open=2000.0, sl=1000.0, volume=1.0, ticket=777)
    # A 0.01-0.02% difference on each -- well inside the default 0.05%
    # amend_tolerance_pct.
    allocation = {"GO10OZ": AllocationEntry(pct=100.0, price=2000.20, stop_loss=1000.20)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0, volume_min=0.01, volume_step=0.01)}),
        market_prices={"GO10OZ": _asset(ask=2000.20)},
        pending_orders=[order],
    )
    assert len(plan) == 1
    assert plan[0].action == "hold"


def test_position_amend_when_stop_loss_changed_but_size_unchanged():
    positions = [_position(volume=1.0, ticket=1, sl=1900.0, tp=None)]
    # pct=5.0 (not 10.0) -- stop distance is now 50 (2000-1950), not the
    # original 100, so pct must halve too to keep target_lots at the
    # same 1.0 lot already held; otherwise this resolves to "increase"
    # instead of testing the amend-in-place path.
    allocation = {"GO10OZ": AllocationEntry(pct=5.0, price=2000.0, stop_loss=1950.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert len(plan) == 1
    assert plan[0].action == "amend_position"
    assert plan[0].stop_loss == 1950.0
    assert plan[0].take_profit is None
    assert plan[0].position_tickets_to_amend == [1]


def test_position_amend_when_take_profit_changed_but_size_unchanged():
    positions = [_position(volume=1.0, ticket=1, sl=1900.0, tp=2100.0)]
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1900.0, take_profit=2200.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert len(plan) == 1
    assert plan[0].action == "amend_position"
    assert plan[0].stop_loss == 1900.0
    assert plan[0].take_profit == 2200.0


def test_position_amend_carries_forward_live_take_profit_when_entry_omits_it():
    positions = [_position(volume=1.0, ticket=1, sl=1900.0, tp=2100.0)]
    # take_profit omitted from the fresh entry entirely -- must not wipe
    # the position's own already-live target to None. pct=5.0 keeps
    # target_lots at the already-held 1.0 lot given the new stop distance
    # (see test_position_amend_when_stop_loss_changed_but_size_unchanged).
    allocation = {"GO10OZ": AllocationEntry(pct=5.0, price=2000.0, stop_loss=1950.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert len(plan) == 1
    assert plan[0].action == "amend_position"
    assert plan[0].take_profit == 2100.0


def test_position_amend_never_changes_volume_or_side():
    positions = [_position(side="buy", volume=1.0, ticket=1, sl=1900.0)]
    # pct=5.0 keeps target_lots at the already-held 1.0 lot -- see
    # test_position_amend_when_stop_loss_changed_but_size_unchanged.
    allocation = {"GO10OZ": AllocationEntry(side="buy", pct=5.0, price=2000.0, stop_loss=1950.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert len(plan) == 1
    assert plan[0].action == "amend_position"
    assert plan[0].volume == 1.0
    assert plan[0].side == "buy"


def test_hold_still_fires_when_stop_and_target_genuinely_unchanged():
    positions = [_position(volume=1.0, ticket=1, sl=1900.0, tp=2100.0)]
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1900.0, take_profit=2100.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert len(plan) == 1
    assert plan[0].action == "hold"


def test_stray_pending_order_cancelled_when_position_already_at_target_size():
    positions = [_position(volume=1.0, ticket=1, sl=1900.0, tp=2100.0)]
    stray = _pending_order(price_open=2000.0, sl=1900.0, ticket=999)
    allocation = {"GO10OZ": AllocationEntry(pct=10.0, price=2000.0, stop_loss=1900.0, take_profit=2100.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec(margin_initial=10000.0)}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
        pending_orders=[stray],
    )
    assert len(plan) == 2
    actions = {o.action for o in plan}
    assert actions == {"hold", "cancel"}
    cancel_order = next(o for o in plan if o.action == "cancel")
    assert cancel_order.pending_tickets_to_cancel == [999]


def test_pending_orders_symbol_without_allocation_entry_produces_no_planned_order():
    # The critical safety-property regression: a symbol whose ONLY
    # footprint is a resting pending order (no filled position, no
    # allocation-block key -- exactly what a Pending-Setup-fired order
    # looks like from compute_rebalance_plan's point of view) must
    # produce ZERO PlannedOrders. Widening this would self-cancel every
    # Pending Setup order the instant Copilot places it.
    order = _pending_order(symbol="EURUSD", price_open=1.09, sl=1.08, ticket=555)
    allocation = {"CASH": AllocationEntry(pct=100.0)}
    plan = compute_rebalance_plan(
        positions=[], account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({}),
        market_prices={},
        pending_orders=[order],
    )
    assert all(o.symbol != "EURUSD" for o in plan)


def test_pct_for_target_lots_basic_correctness():
    # 100 lots x 50-point stop x contract_size 100 = $500,000 risk, which
    # is 5% of 10,000,000 equity.
    pct = pct_for_target_lots(
        symbol="GO10OZ", target_lots=100.0, entry_price=2000.0, stop_loss=1950.0,
        account_equity=10_000_000.0, get_spec=_get_spec_for({"GO10OZ": _spec()}),
    )
    assert pct == pytest.approx(5.0)


def test_pct_for_target_lots_matches_the_hand_derived_value_in_amend_test():
    # Cross-check against test_position_amend_when_stop_loss_changed_but_
    # size_unchanged's own hand-derived comment: keeping 1.0 lot held at a
    # new 50-point stop distance (2000-1950) on a 100k account requires
    # pct=5.0, not the original pct=10.0 (100-point stop).
    pct = pct_for_target_lots(
        symbol="GO10OZ", target_lots=1.0, entry_price=2000.0, stop_loss=1950.0,
        account_equity=100_000.0, get_spec=_get_spec_for({"GO10OZ": _spec()}),
    )
    assert pct == pytest.approx(5.0)


def test_pct_for_target_lots_returns_none_for_zero_stop_distance():
    pct = pct_for_target_lots(
        symbol="GO10OZ", target_lots=1.0, entry_price=2000.0, stop_loss=2000.0,
        account_equity=100_000.0, get_spec=_get_spec_for({"GO10OZ": _spec()}),
    )
    assert pct is None


def test_pct_for_target_lots_returns_none_for_missing_spec():
    pct = pct_for_target_lots(
        symbol="UNKNOWN", target_lots=1.0, entry_price=2000.0, stop_loss=1950.0,
        account_equity=100_000.0, get_spec=_get_spec_for({}),
    )
    assert pct is None


def test_pct_for_target_lots_returns_zero_for_zero_target_lots():
    pct = pct_for_target_lots(
        symbol="GO10OZ", target_lots=0.0, entry_price=2000.0, stop_loss=1950.0,
        account_equity=100_000.0, get_spec=_get_spec_for({"GO10OZ": _spec()}),
    )
    assert pct == 0.0


def test_pct_for_target_lots_round_trip_tighter_stop_keeps_same_size():
    # The real correctness trap this helper exists to close: a DEFEND
    # action tightening 1900 -> 1950 while holding 1.0 lot must resolve
    # to "amend_position" at the SAME 1.0 lot, never "increase" (which is
    # what naively reusing the original pct=10.0 against the now-smaller
    # 50-point stop distance would silently produce -- 2.0 lots instead
    # of 1.0).
    positions = [_position(volume=1.0, ticket=1, sl=1900.0)]
    new_pct = pct_for_target_lots(
        symbol="GO10OZ", target_lots=1.0, entry_price=2000.0, stop_loss=1950.0,
        account_equity=100_000.0, get_spec=_get_spec_for({"GO10OZ": _spec()}),
    )
    allocation = {"GO10OZ": AllocationEntry(pct=new_pct, price=2000.0, stop_loss=1950.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": _spec()}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert len(plan) == 1
    assert plan[0].action == "amend_position"
    assert plan[0].volume == 1.0
    assert plan[0].stop_loss == 1950.0


def test_pct_for_target_lots_round_trip_partial_close():
    # A 40% partial-close target (0.6 of the 1.0 held lot) must resolve
    # to "reduce" at exactly the intended remaining size, not a fresh
    # "increase"/"hold" from an un-rescaled pct. Fractional volume_step
    # so 0.6 lots is a legal size for this instrument.
    spec = _spec(volume_min=0.01, volume_step=0.01)
    positions = [_position(volume=1.0, ticket=1, sl=1900.0)]
    new_pct = pct_for_target_lots(
        symbol="GO10OZ", target_lots=0.6, entry_price=2000.0, stop_loss=1900.0,
        account_equity=100_000.0, get_spec=_get_spec_for({"GO10OZ": spec}),
    )
    allocation = {"GO10OZ": AllocationEntry(pct=new_pct, price=2000.0, stop_loss=1900.0)}
    plan = compute_rebalance_plan(
        positions=positions, account=ACCOUNT, allocation=allocation,
        get_spec=_get_spec_for({"GO10OZ": spec}),
        market_prices={"GO10OZ": _asset(ask=2000.0)},
    )
    assert len(plan) == 1
    assert plan[0].action == "reduce"
    assert plan[0].volume == pytest.approx(0.4)
