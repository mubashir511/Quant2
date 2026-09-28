import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

import MetaTrader5 as mt5

from data.mt5_execution import cancel_pending_order, close_position, modify_position_sltp, open_position
from data.mt5_source import MT5ConnectionError, Position


def _make_position(symbol="GO10OZ", side="buy", volume=2.0, ticket=777):
    return Position(
        symbol=symbol, volume=volume, side=side, price_open=2000.0,
        price_current=2010.0, sl=None, profit=20.0, opened_at=datetime.now(), ticket=ticket,
    )


@patch("MetaTrader5.order_send")
def test_open_position_sends_pending_limit_order(mock_order_send):
    mock_result = MagicMock()
    mock_result.retcode = 10009  # TRADE_RETCODE_DONE, using MT5's real constant value
    mock_result.comment = "Request executed"
    mock_result.order = 12345
    mock_order_send.return_value = mock_result

    result = open_position("GO10OZ", "buy", 1.0, 2005.5, stop_loss=1950.0)

    assert result.success is True
    assert result.ticket == 12345
    request = mock_order_send.call_args.args[0]
    assert request["symbol"] == "GO10OZ"
    assert request["volume"] == 1.0
    assert request["price"] == 2005.5
    assert request["sl"] == 1950.0


@patch("MetaTrader5.order_send")
def test_open_position_omits_sl_when_not_given(mock_order_send):
    mock_result = MagicMock()
    mock_result.retcode = 10009
    mock_result.comment = "Request executed"
    mock_result.order = 12346
    mock_order_send.return_value = mock_result

    open_position("GO10OZ", "sell", 1.0, 2005.5)

    request = mock_order_send.call_args.args[0]
    assert "sl" not in request


@patch("MetaTrader5.order_send")
def test_open_position_sends_take_profit_when_given(mock_order_send):
    mock_result = MagicMock()
    mock_result.retcode = 10009
    mock_result.comment = "Request executed"
    mock_result.order = 12347
    mock_order_send.return_value = mock_result

    open_position("GO10OZ", "buy", 1.0, 2005.5, stop_loss=1950.0, take_profit=2100.0)

    request = mock_order_send.call_args.args[0]
    assert request["sl"] == 1950.0
    assert request["tp"] == 2100.0


@patch("MetaTrader5.order_send")
def test_open_position_omits_tp_when_not_given(mock_order_send):
    mock_result = MagicMock()
    mock_result.retcode = 10009
    mock_result.comment = "Request executed"
    mock_result.order = 12348
    mock_order_send.return_value = mock_result

    open_position("GO10OZ", "buy", 1.0, 2005.5)

    request = mock_order_send.call_args.args[0]
    assert "tp" not in request


@patch("MetaTrader5.order_send")
def test_open_position_reports_rejection_without_raising(mock_order_send):
    mock_result = MagicMock()
    mock_result.retcode = 10006  # TRADE_RETCODE_REJECT
    mock_result.comment = "Rejected"
    mock_result.order = 0
    mock_order_send.return_value = mock_result

    result = open_position("GO10OZ", "buy", 1.0, 2005.5)

    assert result.success is False
    assert result.ticket is None
    assert result.comment == "Rejected"


@patch("MetaTrader5.order_send", return_value=None)
def test_open_position_raises_connection_error_when_order_send_returns_none(mock_order_send):
    try:
        open_position("GO10OZ", "buy", 1.0, 2005.5)
        assert False, "expected MT5ConnectionError"
    except MT5ConnectionError:
        pass


@patch("MetaTrader5.order_send")
@patch("MetaTrader5.symbol_info_tick")
def test_close_position_sends_market_deal_referencing_ticket(mock_tick, mock_order_send):
    mock_tick.return_value = MagicMock(bid=2010.0, ask=2011.0)
    mock_result = MagicMock()
    mock_result.retcode = 10009
    mock_result.comment = "Request executed"
    mock_result.order = 999
    mock_order_send.return_value = mock_result

    position = _make_position(side="buy", volume=2.0, ticket=777)
    result = close_position(position)

    assert result.success is True
    request = mock_order_send.call_args.args[0]
    assert request["position"] == 777
    assert request["volume"] == 2.0
    assert request["price"] == 2010.0  # closing a buy uses the bid


@patch("MetaTrader5.order_send")
@patch("MetaTrader5.symbol_info_tick")
def test_close_position_partial_volume_override(mock_tick, mock_order_send):
    mock_tick.return_value = MagicMock(bid=2010.0, ask=2011.0)
    mock_result = MagicMock()
    mock_result.retcode = 10009
    mock_result.comment = "Request executed"
    mock_result.order = 1000
    mock_order_send.return_value = mock_result

    position = _make_position(side="sell", volume=3.0, ticket=778)
    close_position(position, volume=1.0)

    request = mock_order_send.call_args.args[0]
    assert request["volume"] == 1.0
    assert request["price"] == 2011.0  # closing a sell uses the ask


@patch("MetaTrader5.symbol_info_tick", return_value=None)
def test_close_position_raises_when_no_live_quote(mock_tick):
    position = _make_position()
    try:
        close_position(position)
        assert False, "expected MT5ConnectionError"
    except MT5ConnectionError:
        pass


@patch("MetaTrader5.order_send")
def test_cancel_pending_order_sends_remove_request(mock_order_send):
    mock_result = MagicMock()
    mock_result.retcode = 10009
    mock_result.comment = "Request executed"
    mock_result.order = 555
    mock_order_send.return_value = mock_result

    result = cancel_pending_order(555)

    assert result.success is True
    assert result.ticket == 555
    request = mock_order_send.call_args.args[0]
    assert request["action"] == mt5.TRADE_ACTION_REMOVE
    assert request["order"] == 555


@patch("MetaTrader5.order_send")
def test_cancel_pending_order_reports_rejection_without_raising(mock_order_send):
    mock_result = MagicMock()
    mock_result.retcode = 10006  # TRADE_RETCODE_REJECT
    mock_result.comment = "Rejected"
    mock_result.order = 0
    mock_order_send.return_value = mock_result

    result = cancel_pending_order(555)

    assert result.success is False
    assert result.ticket is None
    assert result.comment == "Rejected"


@patch("MetaTrader5.order_send", return_value=None)
def test_cancel_pending_order_raises_connection_error_when_order_send_returns_none(mock_order_send):
    try:
        cancel_pending_order(555)
        assert False, "expected MT5ConnectionError"
    except MT5ConnectionError:
        pass


@patch("MetaTrader5.order_send")
def test_modify_position_sltp_sends_correct_sltp_request(mock_order_send):
    mock_result = MagicMock()
    mock_result.retcode = 10009
    mock_result.comment = "Request executed"
    mock_order_send.return_value = mock_result

    position = _make_position(symbol="GO10OZ", ticket=888)
    result = modify_position_sltp(position, stop_loss=1950.0, take_profit=2100.0)

    assert result.success is True
    assert result.ticket == 888
    request = mock_order_send.call_args.args[0]
    assert request["action"] == mt5.TRADE_ACTION_SLTP
    assert request["symbol"] == "GO10OZ"
    assert request["position"] == 888
    assert request["sl"] == 1950.0
    assert request["tp"] == 2100.0


@patch("MetaTrader5.order_send")
def test_modify_position_sltp_always_sends_both_sl_and_tp_even_when_only_one_given(mock_order_send):
    # TRADE_ACTION_SLTP is not additive -- an omitted key clears that
    # value to 0 rather than leaving it unchanged, so this function must
    # never send a request missing either key.
    mock_result = MagicMock()
    mock_result.retcode = 10009
    mock_result.comment = "Request executed"
    mock_order_send.return_value = mock_result

    position = _make_position(ticket=889)
    modify_position_sltp(position, stop_loss=1950.0, take_profit=None)

    request = mock_order_send.call_args.args[0]
    assert request["sl"] == 1950.0
    assert request["tp"] == 0.0
    assert "tp" in request


@patch("MetaTrader5.order_send")
def test_modify_position_sltp_reports_rejection_without_raising(mock_order_send):
    mock_result = MagicMock()
    mock_result.retcode = 10006  # TRADE_RETCODE_REJECT
    mock_result.comment = "Rejected"
    mock_order_send.return_value = mock_result

    position = _make_position(ticket=890)
    result = modify_position_sltp(position, stop_loss=1950.0, take_profit=2100.0)

    assert result.success is False
    assert result.ticket is None
    assert result.comment == "Rejected"


@patch("MetaTrader5.order_send", return_value=None)
def test_modify_position_sltp_raises_connection_error_when_order_send_returns_none(mock_order_send):
    position = _make_position(ticket=891)
    try:
        modify_position_sltp(position, stop_loss=1950.0, take_profit=2100.0)
        assert False, "expected MT5ConnectionError"
    except MT5ConnectionError:
        pass


# --- New entry kinds (2026-09-25): stop orders and capped market entries --------------------------------

def _done(order=555):
    result = MagicMock()
    result.retcode = 10009
    result.comment = "Request executed"
    result.order = order
    return result


@patch("MetaTrader5.order_send")
def test_open_position_kind_stop_sends_buy_stop_and_sell_stop_with_sl_tp(mock_order_send):
    mock_order_send.return_value = _done()
    open_position("XAUUSD", "buy", 0.1, 4400.0, stop_loss=4380.0, take_profit=4450.0, kind="stop")
    request = mock_order_send.call_args.args[0]
    assert request["action"] == mt5.TRADE_ACTION_PENDING and request["type"] == mt5.ORDER_TYPE_BUY_STOP
    assert request["price"] == 4400.0 and request["sl"] == 4380.0 and request["tp"] == 4450.0
    open_position("XAUUSD", "sell", 0.1, 4300.0, stop_loss=4320.0, kind="stop")
    assert mock_order_send.call_args.args[0]["type"] == mt5.ORDER_TYPE_SELL_STOP


@patch("MetaTrader5.order_send")
def test_open_position_default_kind_is_still_a_limit_order(mock_order_send):
    mock_order_send.return_value = _done()
    open_position("XAUUSD", "buy", 0.1, 4400.0)
    assert mock_order_send.call_args.args[0]["type"] == mt5.ORDER_TYPE_BUY_LIMIT
    open_position("XAUUSD", "sell", 0.1, 4400.0, kind="limit")
    assert mock_order_send.call_args.args[0]["type"] == mt5.ORDER_TYPE_SELL_LIMIT


@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_market_entry_deals_at_the_live_ask_with_sl_tp_and_a_deviation_in_points(mock_send, mock_tick, mock_info):
    mock_send.return_value = _done(order=901)
    mock_tick.return_value = MagicMock(ask=100.02, bid=100.00)
    mock_info.return_value = MagicMock(point=0.01)
    result = open_position("EURX", "buy", 2.0, 100.00, stop_loss=99.2, take_profit=101.6, kind="market", max_deviation_price=0.05)
    assert result.success and result.ticket == 901
    request = mock_send.call_args.args[0]
    assert request["action"] == mt5.TRADE_ACTION_DEAL and request["type"] == mt5.ORDER_TYPE_BUY
    assert request["price"] == 100.02 and request["deviation"] == 5  # 0.05 / 0.01 point
    assert request["sl"] == 99.2 and request["tp"] == 101.6 and request["type_filling"] == mt5.ORDER_FILLING_IOC
    assert "position" not in request  # a NEW position, never a close


@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_market_sell_uses_the_bid(mock_send, mock_tick, mock_info):
    mock_send.return_value = _done()
    mock_tick.return_value = MagicMock(ask=100.02, bid=100.00)
    mock_info.return_value = MagicMock(point=0.01)
    open_position("EURX", "sell", 1.0, 100.02, stop_loss=100.8, kind="market", max_deviation_price=0.05)
    request = mock_send.call_args.args[0]
    assert request["type"] == mt5.ORDER_TYPE_SELL and request["price"] == 100.00


@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_market_entry_is_not_sent_when_the_price_moved_past_the_cap(mock_send, mock_tick, mock_info):
    mock_tick.return_value = MagicMock(ask=100.20, bid=100.18)
    mock_info.return_value = MagicMock(point=0.01)
    result = open_position("EURX", "buy", 1.0, 100.02, stop_loss=99.2, kind="market", max_deviation_price=0.05)
    assert result.success is False and "nothing sent" in result.comment and "moved" in result.comment
    mock_send.assert_not_called()


@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_market_entry_without_a_cap_or_quote_never_reaches_the_broker(mock_send, mock_tick):
    result = open_position("EURX", "buy", 1.0, 100.0, kind="market")
    assert result.success is False and "max_deviation_price" in result.comment
    mock_tick.return_value = None
    with pytest.raises(MT5ConnectionError):
        open_position("EURX", "buy", 1.0, 100.0, kind="market", max_deviation_price=0.05)
    mock_tick.return_value = MagicMock(ask=0.0, bid=0.0)
    assert open_position("EURX", "buy", 1.0, 100.0, kind="market", max_deviation_price=0.05).success is False
    mock_send.assert_not_called()


@patch("MetaTrader5.order_send")
def test_unknown_kind_sends_nothing(mock_send):
    result = open_position("EURX", "buy", 1.0, 100.0, kind="iceberg")
    assert result.success is False and "unknown entry order kind" in result.comment
    mock_send.assert_not_called()


@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_market_entry_reports_a_broker_rejection_without_raising(mock_send, mock_tick, mock_info):
    rejected = MagicMock(retcode=10018, comment="Market closed", order=0)
    mock_send.return_value = rejected
    mock_tick.return_value = MagicMock(ask=100.02, bid=100.00)
    mock_info.return_value = MagicMock(point=0.01)
    result = open_position("EURX", "buy", 1.0, 100.0, kind="market", max_deviation_price=0.05)
    assert result.success is False and result.comment == "Market closed" and result.ticket is None


# --- Review fixes (2026-09-25 second pass) ------------------------------------------------------------------

@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_a_partially_filled_market_entry_is_a_success_so_the_clerk_never_retries_and_doubles_it(mock_send, mock_tick, mock_info):
    mock_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE_PARTIAL, comment="Partial", order=321)
    mock_tick.return_value = MagicMock(ask=100.02, bid=100.00)
    mock_info.return_value = MagicMock(point=0.01, filling_mode=2)
    result = open_position("EURX", "buy", 5.0, 100.0, stop_loss=99.2, kind="market", max_deviation_price=0.05)
    assert result.success is True and result.ticket == 321


@patch("MetaTrader5.order_send")
def test_a_partial_retcode_on_a_pending_order_is_still_not_a_success(mock_send):
    mock_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE_PARTIAL, comment="x", order=1)
    assert open_position("EURX", "buy", 1.0, 100.0, kind="limit").success is False


@pytest.mark.parametrize(
    "mask,expected",
    [(2, mt5.ORDER_FILLING_IOC), (3, mt5.ORDER_FILLING_IOC), (1, mt5.ORDER_FILLING_FOK), (0, mt5.ORDER_FILLING_RETURN)],
)
@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_market_entry_uses_a_filling_mode_the_symbol_supports(mock_send, mock_tick, mock_info, mask, expected):
    mock_send.return_value = _done()
    mock_tick.return_value = MagicMock(ask=100.02, bid=100.00)
    mock_info.return_value = MagicMock(point=0.01, filling_mode=mask)
    open_position("EURX", "buy", 1.0, 100.0, kind="market", max_deviation_price=0.05)
    assert mock_send.call_args.args[0]["type_filling"] == expected


# --- broker-side expiry of resting orders (2026-09-25) ---------------------------------------------------------

@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_a_resting_order_gets_a_specified_expiry_on_the_server_clock(mock_send, mock_tick, mock_info):
    mock_send.return_value = _done()
    mock_tick.return_value = MagicMock(time=1_800_000_000)
    mock_info.return_value = MagicMock(expiration_mode=1 | 4)
    open_position("NVDA", "buy", 10.0, 220.0, stop_loss=215.0, expiration_hours=2.5)
    request = mock_send.call_args.args[0]
    assert request["type_time"] == mt5.ORDER_TIME_SPECIFIED and request["expiration"] == 1_800_000_000 + 9000


@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_no_expiry_when_the_symbol_does_not_allow_it_or_there_is_no_tick_or_no_hours(mock_send, mock_tick, mock_info):
    mock_send.return_value = _done()
    mock_tick.return_value = MagicMock(time=1_800_000_000)
    mock_info.return_value = MagicMock(expiration_mode=1)  # SPECIFIED bit absent
    open_position("NVDA", "buy", 10.0, 220.0, expiration_hours=2.0)
    assert mock_send.call_args.args[0]["type_time"] == mt5.ORDER_TIME_GTC and "expiration" not in mock_send.call_args.args[0]
    mock_info.return_value = MagicMock(expiration_mode=4)
    mock_tick.return_value = MagicMock(time=0)
    open_position("NVDA", "buy", 10.0, 220.0, expiration_hours=2.0)
    assert "expiration" not in mock_send.call_args.args[0]
    mock_tick.return_value = MagicMock(time=1_800_000_000)
    open_position("NVDA", "buy", 10.0, 220.0, expiration_hours=None)
    assert "expiration" not in mock_send.call_args.args[0]


@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_a_broker_that_rejects_the_expiry_gets_the_order_once_more_as_plain_gtc(mock_send, mock_tick, mock_info):
    mock_tick.return_value = MagicMock(time=1_800_000_000)
    mock_info.return_value = MagicMock(expiration_mode=4)
    mock_send.side_effect = [MagicMock(retcode=10022, comment="Invalid expiration", order=0), _done(order=808)]
    result = open_position("NVDA", "buy", 10.0, 220.0, stop_loss=215.0, take_profit=230.0, expiration_hours=2.0)
    assert result.success and result.ticket == 808 and mock_send.call_count == 2
    first, second = (c.args[0] for c in mock_send.call_args_list)
    assert first["type_time"] == mt5.ORDER_TIME_SPECIFIED and second["type_time"] == mt5.ORDER_TIME_GTC and "expiration" not in second
    assert second["sl"] == 215.0 and second["tp"] == 230.0  # the retry keeps its protection


@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_market_entries_never_get_an_expiry(mock_send, mock_tick):
    mock_send.return_value = _done()
    mock_tick.return_value = MagicMock(ask=100.02, bid=100.0, time=1_800_000_000)
    with patch("MetaTrader5.symbol_info", return_value=MagicMock(point=0.01, filling_mode=2, expiration_mode=4)):
        open_position("EURX", "buy", 1.0, 100.0, kind="market", max_deviation_price=0.05, expiration_hours=2.0)
    assert "expiration" not in mock_send.call_args.args[0]
