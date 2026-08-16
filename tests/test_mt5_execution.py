from datetime import datetime
from unittest.mock import MagicMock, patch

from data.mt5_execution import close_position, open_position
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
