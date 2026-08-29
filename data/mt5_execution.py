from dataclasses import dataclass

from data.mt5_source import MT5ConnectionError, Position


@dataclass
class OrderResult:
    success: bool
    retcode: int | None
    comment: str
    ticket: int | None


def open_position(
    symbol: str,
    side: str,
    volume: float,
    price: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> OrderResult:
    """Places a pending LIMIT order to open/increase exposure — never a
    market order for entries, so an unclamped or stale price never fills
    at an unexpectedly worse level (the price sanity-clamping itself
    happens in risk/apply_suggestion.py, before this is ever called).
    Order-level outcomes (rejected/requoted/etc.) come back as a non-
    raising OrderResult, matching this file's sibling: a rejected order
    isn't a connection failure. MT5ConnectionError is still raised if
    order_send itself returns nothing (the terminal isn't reachable)."""
    import MetaTrader5 as mt5

    order_type = mt5.ORDER_TYPE_BUY_LIMIT if side == "buy" else mt5.ORDER_TYPE_SELL_LIMIT
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    if stop_loss is not None:
        request["sl"] = stop_loss
    if take_profit is not None:
        request["tp"] = take_profit

    result = mt5.order_send(request)
    if result is None:
        error = mt5.last_error()
        raise MT5ConnectionError(f"order_send returned nothing for {symbol} ({error}).")

    success = result.retcode == mt5.TRADE_RETCODE_DONE
    return OrderResult(
        success=success,
        retcode=result.retcode,
        comment=result.comment,
        ticket=result.order if success else None,
    )


def cancel_pending_order(ticket: int) -> OrderResult:
    """Cancels a still-unfilled pending order (the GTC limit order
    open_position places — see its own docstring). Nothing in this file
    could do this before: open_position/close_position cover placing an
    entry and exiting a filled position, but a pending order that never
    fills sits on the account indefinitely (MT5 tracks pending orders and
    filled positions as separate concepts — get_pending_orders() vs
    get_open_positions()), and this project's own automated setups need
    a way to withdraw one that's been superseded rather than leaving it
    to orphan. Same non-raising-on-rejection / raising-on-no-connection
    contract as open_position/close_position."""
    import MetaTrader5 as mt5

    request = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}

    result = mt5.order_send(request)
    if result is None:
        error = mt5.last_error()
        raise MT5ConnectionError(f"order_send returned nothing for order {ticket} ({error}).")

    success = result.retcode == mt5.TRADE_RETCODE_DONE
    return OrderResult(
        success=success,
        retcode=result.retcode,
        comment=result.comment,
        ticket=ticket if success else None,
    )


def close_position(position: Position, volume: float | None = None) -> OrderResult:
    """Closes (fully, or partially if `volume` is given) an existing
    position via an immediate market deal referencing its ticket. Exits
    always use market orders, never limit orders — reducing risk
    shouldn't wait on a hoped-for price the way opening new exposure
    can."""
    import MetaTrader5 as mt5

    close_volume = position.volume if volume is None else volume
    order_type = mt5.ORDER_TYPE_SELL if position.side == "buy" else mt5.ORDER_TYPE_BUY

    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        raise MT5ConnectionError(f"No live quote available to close {position.symbol}.")
    price = tick.bid if position.side == "buy" else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": close_volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        error = mt5.last_error()
        raise MT5ConnectionError(f"order_send returned nothing for {position.symbol} ({error}).")

    success = result.retcode == mt5.TRADE_RETCODE_DONE
    return OrderResult(
        success=success,
        retcode=result.retcode,
        comment=result.comment,
        ticket=result.order if success else None,
    )


def modify_position_sltp(position: Position, stop_loss: float | None, take_profit: float | None) -> OrderResult:
    """Modifies the stop-loss and/or take-profit on an ALREADY-FILLED
    position IN PLACE, via TRADE_ACTION_SLTP — deliberately never a
    close-then-reopen (added 2026-08-23, direct user request to let both
    Claude's daily mega session and the Copilot execution clerk revise an
    already-suggested position's risk management, not just open fresh
    ones). FTMO's own daily-loss/max-loss/Best-Day-Rule tracking
    (risk/ftmo_rules.py) keys off REALIZED P&L bucketed by calendar day:
    closing a position purely to re-stamp its stop or target would force
    today's still-floating P&L into today's REALIZED bucket, a real, not
    cosmetic, side-effect purely from adjusting a stop.

    ALWAYS sends both `sl` and `tp` in the request, even when only one is
    actually changing — MT5's TRADE_ACTION_SLTP request is NOT additive;
    a key omitted from the request is treated as clearing that value to
    0, not "leave it unchanged". Callers (risk/apply_suggestion.py's
    compute_rebalance_plan) are responsible for resolving both fields to
    concrete values (falling back to the position's own current value for
    whichever one isn't actually changing) before calling this — this
    function itself sends exactly what it's given, no resolution here."""
    import MetaTrader5 as mt5

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": position.ticket,
        "sl": stop_loss if stop_loss is not None else 0.0,
        "tp": take_profit if take_profit is not None else 0.0,
    }

    result = mt5.order_send(request)
    if result is None:
        error = mt5.last_error()
        raise MT5ConnectionError(f"order_send returned nothing for {position.symbol} ({error}).")

    success = result.retcode == mt5.TRADE_RETCODE_DONE
    return OrderResult(
        success=success,
        retcode=result.retcode,
        comment=result.comment,
        ticket=position.ticket if success else None,
    )
