from dataclasses import dataclass

from data.mt5_source import MT5ConnectionError, Position


@dataclass
class OrderResult:
    success: bool
    retcode: int | None
    comment: str
    ticket: int | None


def open_position(
    symbol: str, side: str, volume: float, price: float, stop_loss: float | None = None
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
