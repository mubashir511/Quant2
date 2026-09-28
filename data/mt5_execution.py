import math
from dataclasses import dataclass

# _serialize_mt5_access wraps every function below in the SAME lock
# mt5_source.py's own MT5-touching functions use — see that module's
# own comment for why (a real, live full-page-hang incident once
# in-app background threads existed). Importing the shared decorator
# rather than defining a second, independent lock here is deliberate:
# two different Lock objects would each correctly serialize calls
# WITHIN their own module but do nothing to stop a mt5_source.py read
# and a mt5_execution.py order call from racing each other, which is
# exactly the scenario that needs preventing.
from data.mt5_source import MT5ConnectionError, Position, _serialize_mt5_access


@dataclass
class OrderResult:
    success: bool
    retcode: int | None
    comment: str
    ticket: int | None


@_serialize_mt5_access
def open_position(
    symbol: str,
    side: str,
    volume: float,
    price: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    kind: str = "limit",
    max_deviation_price: float | None = None,
    expiration_hours: float | None = None,
) -> OrderResult:
    """Places an entry order: by default a pending LIMIT order (the price sanity-clamping happens in
    risk/apply_suggestion.py, before this is ever called), and — since 2026-09-25, see
    analysis/entry_mode.py for the full reasoning and the caller-side verification — optionally:

    - kind="stop": a pending BUY STOP / SELL STOP at `price` (a breakout trigger beyond the market).
    - kind="market": an immediate deal at the live ask (buy) / bid (sell). `price` is then only the
      REFERENCE the caller verified, and `max_deviation_price` (price units, REQUIRED) is the most the
      live price may have moved against the buyer/seller since — measured here, on a fresh tick, right
      before sending, and also passed to MT5 as the deal's own `deviation` in points. A missing cap, no
      quote, or a larger move sends NOTHING and returns a failed OrderResult.

    `expiration_hours` (pending kinds only): the broker cancels the resting order itself after that many hours
    (MT5 ORDER_TIME_SPECIFIED, measured on the SERVER clock via the symbol's own tick time) - used so an order
    for a session-limited instrument cannot sit overnight when nobody can manage it (the NVDA case). Skipped when the
    symbol does not support specified expiration, and if the broker rejects the expiration (retcode 10022) the order is
    re-sent once as plain GTC - a missing expiry never blocks an entry.

    Every kind carries its stop-loss/take-profit in the SAME request (no unprotected window). Order-level
    outcomes (rejected/requoted/etc.) come back as a non-raising OrderResult, matching this file's
    sibling: a rejected order isn't a connection failure. MT5ConnectionError is still raised if
    order_send itself returns nothing (the terminal isn't reachable)."""
    import MetaTrader5 as mt5

    if kind not in ("limit", "stop", "market"):
        return OrderResult(False, None, f"unknown entry order kind {kind!r} - nothing sent", None)

    if kind == "market":
        if max_deviation_price is None or max_deviation_price < 0:
            return OrderResult(False, None, "market entry needs a max_deviation_price cap - nothing sent", None)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MT5ConnectionError(f"No live quote available to open {symbol} at the market.")
        live = tick.ask if side == "buy" else tick.bid
        if not live:
            return OrderResult(False, None, f"no live {'ask' if side == 'buy' else 'bid'} for {symbol} - nothing sent", None)
        adverse = (live - price) if side == "buy" else (price - live)
        if adverse > max_deviation_price:
            return OrderResult(
                False, None,
                f"market entry skipped: price moved {adverse:.6g} against the verified {price:.6g} "
                f"(cap {max_deviation_price:.6g}) - nothing sent",
                None,
            )
        info = mt5.symbol_info(symbol)
        point = getattr(info, "point", 0) or 0
        deviation = int(math.ceil(max_deviation_price / point)) if point > 0 else 0
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
            "price": live,
            "deviation": deviation,
            "type_filling": _market_filling_mode(mt5, info),
        }
    else:
        if kind == "stop":
            order_type = mt5.ORDER_TYPE_BUY_STOP if side == "buy" else mt5.ORDER_TYPE_SELL_STOP
        else:
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
    gtc_request = None
    if kind != "market" and expiration_hours and expiration_hours > 0:
        expiry = _server_expiry(mt5, symbol, expiration_hours)
        if expiry is not None:
            gtc_request = dict(request)
            request["type_time"] = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = expiry

    result = mt5.order_send(request)
    if result is None:
        error = mt5.last_error()
        raise MT5ConnectionError(f"order_send returned nothing for {symbol} ({error}).")
    if gtc_request is not None and result.retcode == _INVALID_EXPIRATION_RETCODE:
        result = mt5.order_send(gtc_request)  # the broker refused the expiry: place it plain GTC rather than not at all
        if result is None:
            error = mt5.last_error()
            raise MT5ConnectionError(f"order_send returned nothing for {symbol} ({error}).")

    # A market deal can come back PARTIALLY filled (IOC/FOK-less venues, thin stock CFDs): the position exists,
    # so it is a success — treating it as a failure would make the Clerk retry and double the exposure.
    success_codes = {mt5.TRADE_RETCODE_DONE}
    if kind == "market":
        success_codes.add(mt5.TRADE_RETCODE_DONE_PARTIAL)
    success = result.retcode in success_codes
    return OrderResult(
        success=success,
        retcode=result.retcode,
        comment=result.comment,
        ticket=result.order if success else None,
    )


_INVALID_EXPIRATION_RETCODE = 10022  # TRADE_RETCODE_INVALID_EXPIRATION
# The Python package exposes no SYMBOL_EXPIRATION_* constants; these are the MQL5 values of SYMBOL_EXPIRATION_MODE bits.
_SYMBOL_EXPIRATION_GTC, _SYMBOL_EXPIRATION_SPECIFIED = 1, 4


def _server_expiry(mt5, symbol: str, hours: float) -> int | None:
    """Expiration timestamp (server-clock epoch seconds) `hours` from now, or None when the symbol does not allow a
    specified expiration or no live tick exists. The server clock comes from the symbol's own last tick so the broker's
    UTC offset never enters the arithmetic."""
    info = mt5.symbol_info(symbol)
    mode = getattr(info, "expiration_mode", None)
    if isinstance(mode, int) and not (mode & _SYMBOL_EXPIRATION_SPECIFIED):
        return None
    tick = mt5.symbol_info_tick(symbol)
    server_now = getattr(tick, "time", 0) if tick is not None else 0
    if not server_now:
        return None
    return int(server_now + hours * 3600)


def _market_filling_mode(mt5, info) -> int:
    """The deal filling mode this symbol actually supports (SYMBOL_FILLING_FOK = 1, IOC = 2 bits of
    symbol_info.filling_mode). Prefers IOC (what close_position uses), then FOK; RETURN only when the symbol
    advertises neither. An unsupported mode is rejected by the broker (retcode 10030) — safe, but the market
    entry would then never work on that instrument."""
    mask = getattr(info, "filling_mode", None)
    if isinstance(mask, int):
        if mask & 2:
            return mt5.ORDER_FILLING_IOC
        if mask & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN
    return mt5.ORDER_FILLING_IOC


@_serialize_mt5_access
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


@_serialize_mt5_access
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


@_serialize_mt5_access
def modify_position_sltp(position: Position, stop_loss: float | None, take_profit: float | None) -> OrderResult:
    """Modifies the stop-loss and/or take-profit on an ALREADY-FILLED
    position IN PLACE, via TRADE_ACTION_SLTP — deliberately never a
    close-then-reopen (added 2026-08-23, direct user request to let both
    Claude's daily mega session and the Execution Clerk revise an
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
