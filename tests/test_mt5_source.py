import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import config
from data.mt5_source import (
    MT5ConnectionError,
    HistoricalDeal,
    _ensure_symbol_selected,
    connect,
    fetch_mt5_price_history,
    get_contract_spec,
    get_current_bid_ask,
    get_history_deals,
    get_market_watch,
    get_symbol_category,
    get_trade_economics,
    group_closed_trades,
    is_trading_permitted,
)


@pytest.fixture(autouse=True)
def _no_csv_fallback_by_default():
    # Real .env may set MT5_SYMBOL_SPECS_CSV_PATH to an actual file on this
    # machine (data/mt5_source.py's fallback for get_contract_spec) — tests
    # must not depend on that being absent by accident of environment.
    # Tests that specifically exercise the CSV fallback override this via
    # their own nested patch.object(config, "MT5_SYMBOL_SPECS_CSV_PATH", ...).
    with patch.object(config, "MT5_SYMBOL_SPECS_CSV_PATH", ""):
        yield


@patch("data.mt5_source.time.sleep")
def test_ensure_symbol_selected_returns_true_immediately_on_first_success(mock_sleep):
    mt5 = MagicMock()
    mt5.symbol_select.return_value = True

    assert _ensure_symbol_selected(mt5, "SP500-SE26") is True
    mt5.symbol_select.assert_called_once_with("SP500-SE26", True)
    mock_sleep.assert_not_called()


@patch("data.mt5_source.time.sleep")
def test_ensure_symbol_selected_retries_and_recovers(mock_sleep):
    mt5 = MagicMock()
    mt5.symbol_select.side_effect = [False, False, True]

    assert _ensure_symbol_selected(mt5, "SP500-SE26") is True
    assert mt5.symbol_select.call_count == 3
    assert mock_sleep.call_count == 2  # slept between attempts 1-2 and 2-3, not after success


@patch("data.mt5_source.time.sleep")
def test_ensure_symbol_selected_gives_up_and_logs_after_exhausting_attempts(mock_sleep, caplog):
    mt5 = MagicMock()
    mt5.symbol_select.return_value = False
    mt5.last_error.return_value = (-2, "still not selected")

    with caplog.at_level(logging.WARNING):
        result = _ensure_symbol_selected(mt5, "SP500-SE26")

    assert result is False
    assert mt5.symbol_select.call_count == 3
    assert mock_sleep.call_count == 2
    assert "SP500-SE26" in caplog.text
    assert "still not selected" in caplog.text


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info")
def test_get_contract_spec_parses_symbol_info(mock_symbol_info, mock_select):
    mock_info = MagicMock()
    mock_info.volume_min = 1.0
    mock_info.volume_step = 1.0
    mock_info.volume_max = 1.0
    mock_info.trade_contract_size = 100.0
    mock_info.currency_margin = "PKR"
    mock_info.margin_initial = 3749300.0
    mock_symbol_info.return_value = mock_info

    spec = get_contract_spec("PALDIUM100-SE26")
    assert spec.volume_min == 1.0
    assert spec.currency_margin == "PKR"
    assert spec.margin_initial == 3749300.0
    # select must happen before info is read, and for the same symbol —
    # this is the actual fix for SP500-SE26 returning None despite the
    # broker having complete real spec data (confirmed via the MT5
    # terminal's own Symbols window).
    mock_select.assert_called_once_with("PALDIUM100-SE26", True)


@patch("data.mt5_source.time.sleep")
@patch("MetaTrader5.last_error", return_value=(-2, "symbol not selected"))
@patch("MetaTrader5.symbol_select", return_value=False)
@patch("MetaTrader5.symbol_info")
def test_get_contract_spec_still_tries_symbol_info_when_select_fails(
    mock_symbol_info, mock_select, mock_last_error, mock_sleep, caplog
):
    # symbol_select failing (even after retries) shouldn't short-circuit
    # — symbol_info might still succeed (or fail for its own, separately-
    # logged reason) — but the select failure itself must be visible,
    # not silently absorbed.
    mock_info = MagicMock()
    mock_info.volume_min = 1.0
    mock_info.volume_step = 1.0
    mock_info.volume_max = 1.0
    mock_info.trade_contract_size = 100.0
    mock_info.currency_margin = "PKR"
    mock_info.margin_initial = 140500.0
    mock_symbol_info.return_value = mock_info

    with caplog.at_level(logging.WARNING):
        spec = get_contract_spec("SP500-SE26")

    assert spec.margin_initial == 140500.0
    mock_symbol_info.assert_called_once_with("SP500-SE26")
    assert mock_select.call_count == 3  # exhausted every retry attempt
    assert "SP500-SE26" in caplog.text
    assert "symbol not selected" in caplog.text


@patch("MetaTrader5.symbol_info", return_value=None)
def test_get_contract_spec_none_when_symbol_not_found(mock_symbol_info):
    assert get_contract_spec("UNKNOWN") is None


@patch("MetaTrader5.symbol_info", side_effect=RuntimeError("not connected"))
def test_get_contract_spec_none_on_failure(mock_symbol_info):
    assert get_contract_spec("PALDIUM100-SE26") is None


@patch("MetaTrader5.last_error", return_value=(-2, "Terminal: no data for symbol"))
@patch("MetaTrader5.symbol_info", return_value=None)
def test_get_contract_spec_logs_last_error_when_symbol_info_returns_none(
    mock_symbol_info, mock_last_error, caplog
):
    # Confirmed live: this was previously a silent None for SP500-SE26 in
    # every real run, with no way to tell "genuine broker data gap" apart
    # from "transient failure" apart from "a bug" — this is the fix.
    with caplog.at_level(logging.WARNING):
        result = get_contract_spec("SP500-SE26")
    assert result is None
    assert "SP500-SE26" in caplog.text
    assert "no data for symbol" in caplog.text


@patch("MetaTrader5.symbol_info", side_effect=ConnectionAbortedError("terminal closed"))
def test_get_contract_spec_logs_exception_type_and_message(mock_symbol_info, caplog):
    with caplog.at_level(logging.WARNING):
        result = get_contract_spec("PALDIUM100-SE26")
    assert result is None
    assert "PALDIUM100-SE26" in caplog.text
    assert "ConnectionAbortedError" in caplog.text
    assert "terminal closed" in caplog.text


@patch("MetaTrader5.symbol_info")
def test_get_contract_spec_logs_and_returns_none_on_incomplete_info(mock_symbol_info, caplog):
    # An object that isn't None but is missing an expected field (rather
    # than raising during the mt5.symbol_info call itself) must still
    # degrade to None with a real reason logged, not an uncaught
    # AttributeError bubbling out of analyze_assets.
    incomplete_info = MagicMock(spec=["volume_min"])
    incomplete_info.volume_min = 1.0
    mock_symbol_info.return_value = incomplete_info

    with caplog.at_level(logging.WARNING):
        result = get_contract_spec("SP500-SE26")
    assert result is None
    assert "SP500-SE26" in caplog.text


@patch("MetaTrader5.symbol_info", return_value=None)
def test_get_contract_spec_uses_csv_fallback_when_live_api_has_nothing(mock_symbol_info, tmp_path):
    csv_path = tmp_path / "symbol_specs.csv"
    csv_path.write_text(
        "symbol,volume_min,volume_step,volume_max,trade_contract_size,currency_margin,margin_initial\n"
        "OTHER,1.0,1.0,1.0,100.0,PKR,1000.0\n"
        "SP500-SE26,1.0,1.0,10.0,50.0,PKR,140500.0\n",
        encoding="utf-8",
    )
    with patch.object(config, "MT5_SYMBOL_SPECS_CSV_PATH", str(csv_path)):
        spec = get_contract_spec("SP500-SE26")

    assert spec is not None
    assert spec.margin_initial == 140500.0
    assert spec.currency_margin == "PKR"


@patch("MetaTrader5.symbol_info", return_value=None)
def test_get_contract_spec_csv_fallback_disabled_when_path_unset(mock_symbol_info):
    # An empty MT5_SYMBOL_SPECS_CSV_PATH (the shipped default) falls
    # straight through to None, same as before this fallback existed.
    assert get_contract_spec("SP500-SE26") is None


@patch("MetaTrader5.symbol_info", return_value=None)
def test_get_contract_spec_csv_fallback_none_when_symbol_not_in_csv(mock_symbol_info, tmp_path):
    csv_path = tmp_path / "symbol_specs.csv"
    csv_path.write_text(
        "symbol,volume_min,volume_step,volume_max,trade_contract_size,currency_margin,margin_initial\n"
        "OTHER,1.0,1.0,1.0,100.0,PKR,1000.0\n",
        encoding="utf-8",
    )
    with patch.object(config, "MT5_SYMBOL_SPECS_CSV_PATH", str(csv_path)):
        assert get_contract_spec("SP500-SE26") is None


@patch("MetaTrader5.symbol_info", return_value=None)
def test_get_contract_spec_csv_fallback_logs_and_returns_none_when_file_missing(mock_symbol_info, caplog):
    with patch.object(config, "MT5_SYMBOL_SPECS_CSV_PATH", "does_not_exist.csv"):
        with caplog.at_level(logging.WARNING):
            result = get_contract_spec("SP500-SE26")
    assert result is None
    assert "does_not_exist.csv" in caplog.text


@patch("MetaTrader5.order_calc_margin", return_value=1156.95)
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info")
def test_get_contract_spec_falls_back_to_order_calc_margin_when_margin_initial_is_zero(
    mock_symbol_info, mock_select, mock_tick, mock_calc_margin
):
    # Confirmed live on a real FTMO account: FOREX-calc-mode symbols
    # (e.g. EURUSD) always report margin_initial=0 via symbol_info() —
    # that field only carries a real number for futures-style calc
    # modes. Without this fallback, compute_rebalance_plan treats
    # margin_initial<=0 as "no spec available" and marks the symbol
    # infeasible, silently breaking Apply Suggestion for every forex/
    # CFD instrument.
    mock_info = MagicMock()
    mock_info.volume_min = 0.01
    mock_info.volume_step = 0.01
    mock_info.volume_max = 500.0
    mock_info.trade_contract_size = 100000.0
    mock_info.currency_margin = "USD"
    mock_info.margin_initial = 0.0
    mock_symbol_info.return_value = mock_info
    mock_tick.return_value = _make_tick(1.15689, 1.15695)

    spec = get_contract_spec("EURUSD")

    assert spec is not None
    assert spec.margin_initial == pytest.approx(1156.95)
    mock_calc_margin.assert_called_once()
    # Buy-side margin, 1.0 lot, at the live ask (not bid) — matches
    # "margin required for 1.0 lot" ContractSpec's own docstring meaning.
    args, kwargs = mock_calc_margin.call_args
    assert args[1] == "EURUSD"
    assert args[2] == 1.0
    assert args[3] == pytest.approx(1.15695)


@patch("MetaTrader5.symbol_info_tick", return_value=None)
@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info")
def test_get_contract_spec_margin_zero_when_order_calc_margin_has_no_tick(
    mock_symbol_info, mock_select, mock_tick
):
    mock_info = MagicMock()
    mock_info.volume_min = 0.01
    mock_info.volume_step = 0.01
    mock_info.volume_max = 500.0
    mock_info.trade_contract_size = 100000.0
    mock_info.currency_margin = "USD"
    mock_info.margin_initial = 0.0
    mock_symbol_info.return_value = mock_info

    spec = get_contract_spec("EURUSD")

    assert spec is not None
    assert spec.margin_initial == 0.0


@patch("MetaTrader5.order_calc_margin", return_value=None)
@patch("MetaTrader5.last_error", return_value=(-2, "calc failed"))
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info")
def test_get_contract_spec_margin_zero_when_order_calc_margin_returns_none(
    mock_symbol_info, mock_select, mock_tick, mock_last_error, mock_calc_margin, caplog
):
    mock_info = MagicMock()
    mock_info.volume_min = 0.01
    mock_info.volume_step = 0.01
    mock_info.volume_max = 500.0
    mock_info.trade_contract_size = 100000.0
    mock_info.currency_margin = "USD"
    mock_info.margin_initial = 0.0
    mock_symbol_info.return_value = mock_info
    mock_tick.return_value = _make_tick(1.15689, 1.15695)

    with caplog.at_level(logging.WARNING):
        spec = get_contract_spec("EURUSD")

    assert spec is not None
    assert spec.margin_initial == 0.0
    assert "calc failed" in caplog.text


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info")
def test_get_contract_spec_does_not_call_order_calc_margin_when_margin_initial_is_nonzero(
    mock_symbol_info, mock_select,
):
    # PMEX's own futures contracts already report a real margin_initial —
    # the fallback must not fire (and must not need a tick) when the
    # direct field is already usable.
    mock_info = MagicMock()
    mock_info.volume_min = 1.0
    mock_info.volume_step = 1.0
    mock_info.volume_max = 1.0
    mock_info.trade_contract_size = 100.0
    mock_info.currency_margin = "PKR"
    mock_info.margin_initial = 140500.0
    mock_symbol_info.return_value = mock_info

    with patch("MetaTrader5.order_calc_margin") as mock_calc_margin:
        spec = get_contract_spec("SP500-SE26")

    assert spec.margin_initial == 140500.0
    mock_calc_margin.assert_not_called()


def _make_forex_info(**overrides):
    # Real values confirmed live against FTMO's own EURUSD.
    info = MagicMock()
    info.point = 1e-05
    info.trade_tick_size = 1e-05
    info.trade_tick_value = 1.0
    info.trade_contract_size = 100000.0
    info.swap_long = -8.74
    info.swap_short = 0.37
    info.swap_mode = 1  # SYMBOL_SWAP_MODE_POINTS
    info.path = "Forex\\Majors\\EURUSD"
    info.trade_stops_level = 0  # no broker-imposed minimum stop distance by default
    for k, v in overrides.items():
        setattr(info, k, v)
    return info


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbol_info")
def test_get_trade_economics_points_swap_mode_matches_real_ftmo_eurusd(
    mock_symbol_info, mock_tick, mock_select
):
    mock_symbol_info.return_value = _make_forex_info()
    mock_tick.return_value = _make_tick(1.15689, 1.15695)

    cost = get_trade_economics("EURUSD")

    assert cost is not None
    assert cost.category == "Forex"
    assert cost.spread_pct_of_price == pytest.approx((1.15695 - 1.15689) / 1.15695 * 100)
    # -8.74 points * $1.00/point / (100000 * 1.15695 notional) * 100
    assert cost.swap_long_pct_per_day == pytest.approx(-8.74 / 115695.0 * 100, rel=1e-6)
    assert cost.swap_short_pct_per_day == pytest.approx(0.37 / 115695.0 * 100, rel=1e-6)
    assert cost.min_stop_distance_pct == 0.0


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbol_info")
def test_get_trade_economics_computes_real_min_stop_distance_when_broker_enforces_one(
    mock_symbol_info, mock_tick, mock_select
):
    # A real, if less common, broker constraint: MT5's own
    # SYMBOL_TRADE_STOPS_LEVEL, a minimum stop/target distance in points
    # from the current price — 100 points here (0.00100 at 5-digit
    # EURUSD pricing), confirmed convertible to a %-of-price figure.
    info = _make_forex_info(trade_stops_level=100)
    mock_symbol_info.return_value = info
    mock_tick.return_value = _make_tick(1.15689, 1.15695)

    cost = get_trade_economics("EURUSD")

    assert cost is not None
    assert cost.min_stop_distance_pct == pytest.approx(100 * 1e-05 / 1.15695 * 100, rel=1e-6)


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbol_info")
def test_get_trade_economics_interest_swap_mode_matches_real_ftmo_btcusd(
    mock_symbol_info, mock_tick, mock_select
):
    # Real values confirmed live against FTMO's own BTCUSD — a DIFFERENT
    # swap_mode than EURUSD's on the very same account, the whole reason
    # _compute_swap_pct_per_day can't apply one formula to everything.
    info = _make_forex_info(
        swap_long=-30.0, swap_short=-30.0, swap_mode=5, path="Crypto I CFD\\BTCUSD"  # INTEREST_CURRENT
    )
    mock_symbol_info.return_value = info
    mock_tick.return_value = _make_tick(63067.26, 63068.26)

    cost = get_trade_economics("BTCUSD")

    assert cost is not None
    assert cost.category == "Crypto I CFD"
    # Annual % rate accrued daily on the standard 360-day basis.
    assert cost.swap_long_pct_per_day == pytest.approx(-30.0 / 360, rel=1e-6)
    assert cost.swap_short_pct_per_day == pytest.approx(-30.0 / 360, rel=1e-6)


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbol_info")
def test_get_trade_economics_unsupported_swap_mode_returns_none_swap_not_fabricated(
    mock_symbol_info, mock_tick, mock_select
):
    info = _make_forex_info(swap_mode=2)  # SYMBOL_SWAP_MODE_CURRENCY_SYMBOL — not implemented
    mock_symbol_info.return_value = info
    mock_tick.return_value = _make_tick(1.15689, 1.15695)

    cost = get_trade_economics("EURUSD")

    assert cost is not None  # spread is still real and usable
    assert cost.swap_long_pct_per_day is None
    assert cost.swap_short_pct_per_day is None


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbol_info")
def test_get_trade_economics_uncategorized_when_path_missing(mock_symbol_info, mock_tick, mock_select):
    mock_symbol_info.return_value = _make_forex_info(path="")
    mock_tick.return_value = _make_tick(1.15689, 1.15695)

    cost = get_trade_economics("EURUSD")

    assert cost.category == "Uncategorized"


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info")
def test_get_symbol_category_reads_top_level_path_segment(mock_symbol_info, mock_select):
    mock_symbol_info.return_value = _make_forex_info(path="Metals CFD\\XAUUSD")
    assert get_symbol_category("XAUUSD") == "Metals CFD"


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info")
def test_get_symbol_category_uncategorized_when_path_missing(mock_symbol_info, mock_select):
    mock_symbol_info.return_value = _make_forex_info(path="")
    assert get_symbol_category("EURUSD") == "Uncategorized"


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info", return_value=None)
def test_get_symbol_category_uncategorized_when_no_symbol_info(mock_symbol_info, mock_select):
    assert get_symbol_category("UNKNOWN") == "Uncategorized"


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick")
def test_get_current_bid_ask_returns_both_sides(mock_tick, mock_select):
    mock_tick.return_value = _make_tick(1.1000, 1.1005)
    assert get_current_bid_ask("EURUSD") == (1.1000, 1.1005)


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick", return_value=None)
def test_get_current_bid_ask_none_when_no_tick(mock_tick, mock_select):
    assert get_current_bid_ask("EURUSD") is None


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick", return_value=None)
@patch("MetaTrader5.symbol_info")
def test_get_trade_economics_none_when_no_tick(mock_symbol_info, mock_tick, mock_select):
    mock_symbol_info.return_value = _make_forex_info()
    assert get_trade_economics("EURUSD") is None


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbol_info", return_value=None)
def test_get_trade_economics_none_when_no_symbol_info(mock_symbol_info, mock_tick, mock_select):
    mock_tick.return_value = _make_tick(1.0, 1.0)
    assert get_trade_economics("UNKNOWN") is None


def _make_symbol(name, description="desc", visible=True):
    s = MagicMock()
    s.name = name
    s.description = description
    s.visible = visible
    return s


def _make_tick(bid, ask):
    t = MagicMock()
    t.bid = bid
    t.ask = ask
    return t


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbols_get")
def test_get_market_watch_selects_each_visible_symbol_before_reading_its_tick(
    mock_symbols_get, mock_tick, mock_select
):
    mock_symbols_get.return_value = [_make_symbol("SP500-SE26"), _make_symbol("DJ-SE26")]
    mock_tick.return_value = _make_tick(100.0, 100.5)

    assets = get_market_watch()

    assert {a.symbol for a in assets} == {"SP500-SE26", "DJ-SE26"}
    mock_select.assert_any_call("SP500-SE26", True)
    mock_select.assert_any_call("DJ-SE26", True)


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbols_get")
def test_get_market_watch_skips_non_visible_symbols_without_selecting_them(
    mock_symbols_get, mock_tick, mock_select
):
    mock_symbols_get.return_value = [_make_symbol("HIDDEN", visible=False)]
    mock_tick.return_value = _make_tick(100.0, 100.5)

    assets = get_market_watch()

    assert assets == []
    mock_select.assert_not_called()


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick", return_value=None)
@patch("MetaTrader5.symbols_get")
def test_get_market_watch_logs_visible_symbols_dropped_for_missing_tick(
    mock_symbols_get, mock_tick, mock_select, caplog
):
    # Confirmed live risk: a visible symbol silently vanishing here means
    # the AI never sees it at all — previously a bare `continue` with no
    # trace this had happened, indistinguishable from "correctly excluded."
    mock_symbols_get.return_value = [_make_symbol("SP500-SE26")]

    with caplog.at_level(logging.WARNING):
        assets = get_market_watch()

    assert assets == []
    assert "SP500-SE26" in caplog.text
    assert "dropped" in caplog.text.lower()


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbols_get")
def test_get_market_watch_logs_visible_symbols_dropped_for_zero_quote(
    mock_symbols_get, mock_tick, mock_select, caplog
):
    mock_symbols_get.return_value = [_make_symbol("ILLIQUID")]
    mock_tick.return_value = _make_tick(0.0, 0.0)

    with caplog.at_level(logging.WARNING):
        assets = get_market_watch()

    assert assets == []
    assert "ILLIQUID" in caplog.text


@patch("data.mt5_source.time.sleep")
@patch("MetaTrader5.last_error", return_value=(-2, "select failed"))
@patch("MetaTrader5.symbol_select", return_value=False)
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.symbols_get")
def test_get_market_watch_logs_when_select_fails_but_still_tries_the_tick(
    mock_symbols_get, mock_tick, mock_select, mock_last_error, mock_sleep, caplog
):
    mock_symbols_get.return_value = [_make_symbol("SP500-SE26")]
    mock_tick.return_value = _make_tick(100.0, 100.5)

    with caplog.at_level(logging.WARNING):
        assets = get_market_watch()

    assert len(assets) == 1  # select failing (even after retries) doesn't block a working tick
    assert "SP500-SE26" in caplog.text
    assert "select failed" in caplog.text


def _make_account_info(login, server):
    info = MagicMock()
    info.login = login
    info.server = server
    return info


@patch("MetaTrader5.account_info")
@patch("MetaTrader5.initialize", return_value=True)
def test_connect_uses_config_defaults_when_no_args(mock_initialize, mock_account_info):
    mock_account_info.return_value = _make_account_info(12345, "Broker-Demo")
    with patch.object(config, "MT5_LOGIN", "12345"), patch.object(config, "MT5_SERVER", "Broker-Demo"):
        connect()
    assert mock_initialize.call_args.kwargs.get("login") == 12345
    assert mock_initialize.call_args.kwargs.get("server") == "Broker-Demo"


@patch("MetaTrader5.account_info")
@patch("MetaTrader5.initialize", return_value=True)
def test_connect_overrides_config_with_explicit_args(mock_initialize, mock_account_info):
    mock_account_info.return_value = _make_account_info(999, "FTMO-Demo")

    connect(login=999, password="pw", server="FTMO-Demo", path="C:/ftmo/terminal64.exe")

    assert mock_initialize.call_args.kwargs == {
        "path": "C:/ftmo/terminal64.exe",
        "login": 999,
        "password": "pw",
        "server": "FTMO-Demo",
    }


@patch("MetaTrader5.last_error", return_value=(-1, "terminal not found"))
@patch("MetaTrader5.initialize", return_value=False)
def test_connect_raises_when_initialize_fails(mock_initialize, mock_last_error):
    with pytest.raises(MT5ConnectionError, match="terminal not found"):
        connect(login=999, password="pw", server="FTMO-Demo")


@patch("MetaTrader5.account_info", return_value=None)
@patch("MetaTrader5.initialize", return_value=True)
def test_connect_raises_when_account_info_missing_after_login(mock_initialize, mock_account_info):
    with pytest.raises(MT5ConnectionError, match="could not verify"):
        connect(login=999, password="pw", server="FTMO-Demo")


@patch("MetaTrader5.account_info")
@patch("MetaTrader5.initialize", return_value=True)
def test_connect_raises_on_stale_login_mismatch(mock_initialize, mock_account_info):
    # This is the real failure mode the research flagged: mt5.initialize()
    # can "succeed" while still leaving the *previous* account connected.
    mock_account_info.return_value = _make_account_info(111, "PMEX-Live")

    with pytest.raises(MT5ConnectionError, match="DIFFERENT account"):
        connect(login=999, password="pw", server="FTMO-Demo")


@patch("MetaTrader5.account_info")
@patch("MetaTrader5.initialize", return_value=True)
def test_connect_raises_on_server_mismatch(mock_initialize, mock_account_info):
    mock_account_info.return_value = _make_account_info(999, "SomeOtherServer")

    with pytest.raises(MT5ConnectionError, match="mismatched account"):
        connect(login=999, password="pw", server="FTMO-Demo")


@patch("MetaTrader5.account_info")
@patch("MetaTrader5.initialize", return_value=True)
def test_connect_skips_verification_when_no_login_given(mock_initialize, mock_account_info):
    with patch.object(config, "MT5_LOGIN", None), patch.object(config, "MT5_SERVER", None):
        connect()
    mock_account_info.assert_not_called()


def _make_deal(ticket, time, symbol, profit, swap, commission, volume):
    d = MagicMock()
    d.ticket = ticket
    d.time = time
    d.symbol = symbol
    d.profit = profit
    d.swap = swap
    d.commission = commission
    d.volume = volume
    return d


@patch("MetaTrader5.history_deals_get")
def test_get_history_deals_sums_profit_swap_and_commission(mock_history):
    ts = int(datetime(2026, 8, 10, 12, 0, 0).timestamp())
    mock_history.return_value = [_make_deal(1, ts, "EURUSD", 50.0, -1.5, -2.0, 0.5)]

    deals = get_history_deals(datetime(2026, 8, 1), datetime(2026, 8, 15))

    assert len(deals) == 1
    assert deals[0].symbol == "EURUSD"
    assert deals[0].profit == pytest.approx(46.5)  # 50.0 - 1.5 - 2.0
    assert deals[0].volume == 0.5


@patch("MetaTrader5.last_error", return_value=(-2, "not connected"))
@patch("MetaTrader5.history_deals_get", return_value=None)
def test_get_history_deals_empty_on_failure(mock_history, mock_last_error, caplog):
    with caplog.at_level(logging.WARNING):
        deals = get_history_deals(datetime(2026, 8, 1))
    assert deals == []
    assert "not connected" in caplog.text


@patch("MetaTrader5.history_deals_get", return_value=[])
def test_get_history_deals_empty_for_fresh_account(mock_history):
    # A brand-new account (e.g. a fresh FTMO Challenge) with zero trade
    # history is a normal, expected case, not a failure.
    assert get_history_deals(datetime(2026, 8, 1)) == []


def test_group_closed_trades_sums_every_leg_onto_the_position():
    # Real shape confirmed live: opening leg carries only commission,
    # closing leg carries the real profit+swap+commission — the true net
    # result only matched the account's real balance change when both
    # legs were summed together.
    deals = [
        HistoricalDeal(
            1, datetime(2026, 8, 18, 9, 47, 28), "XAUUSD", -1.51, 0.49,
            position_id=100, price=4407.58, side="buy",
        ),
        HistoricalDeal(
            2, datetime(2026, 8, 18, 10, 2, 56), "XAUUSD", -333.24, 0.49,
            position_id=100, price=4400.81, side="sell",
        ),
    ]
    trades = group_closed_trades(deals)
    assert len(trades) == 1
    assert trades[0].symbol == "XAUUSD"
    assert trades[0].profit == pytest.approx(-334.75)
    assert trades[0].closed_at == datetime(2026, 8, 18, 10, 2, 56)
    # side/open_price come from the opening leg, close_price from the
    # closing leg — confirmed live these match the real account exactly.
    assert trades[0].side == "buy"
    assert trades[0].opened_at == datetime(2026, 8, 18, 9, 47, 28)
    assert trades[0].open_price == pytest.approx(4407.58)
    assert trades[0].close_price == pytest.approx(4400.81)
    assert trades[0].duration == timedelta(minutes=15, seconds=28)


def test_group_closed_trades_excludes_still_open_positions():
    # Only one deal on record for this position — it hasn't closed yet.
    deals = [HistoricalDeal(1, datetime(2026, 8, 18), "EURUSD", 0.0, 1.0, position_id=200)]
    assert group_closed_trades(deals) == []


def test_group_closed_trades_excludes_non_trade_balance_deals():
    # A deposit/withdrawal deal has no symbol and no position_id — must
    # never show up as a "closed trade."
    deals = [HistoricalDeal(1, datetime(2026, 8, 18), "", 10000.0, 0.0, position_id=0)]
    assert group_closed_trades(deals) == []


def test_group_closed_trades_sorts_newest_first():
    deals = [
        HistoricalDeal(1, datetime(2026, 8, 1), "EURUSD", -1.0, 1.0, position_id=1),
        HistoricalDeal(2, datetime(2026, 8, 2), "EURUSD", 5.0, 1.0, position_id=1),
        HistoricalDeal(3, datetime(2026, 8, 10), "GBPUSD", -1.0, 1.0, position_id=2),
        HistoricalDeal(4, datetime(2026, 8, 11), "GBPUSD", 20.0, 1.0, position_id=2),
    ]
    trades = group_closed_trades(deals)
    assert [t.position_id for t in trades] == [2, 1]


@patch("MetaTrader5.history_deals_get", return_value=[])
def test_get_history_deals_default_date_to_extends_a_day_into_the_future(mock_history):
    # Real bug, confirmed live: MT5 deal timestamps are in the broker's
    # SERVER clock, which can run meaningfully ahead of this machine's
    # local clock (~2 hours observed against the real FTMO account) — a
    # default date_to=datetime.now() (local) silently excluded a deal
    # that had already happened in server time. The default must clear
    # local "now" by a real margin, not sit exactly on it.
    before_call = datetime.now()
    get_history_deals(datetime(2026, 8, 1))
    used_date_to = mock_history.call_args.args[1]
    assert used_date_to > before_call + timedelta(hours=12)


def _make_rate(t, o, h, l, c, tick_volume):
    return (t, o, h, l, c, tick_volume, 0, 0)


@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.copy_rates_from_pos")
def test_fetch_mt5_price_history_shapes_dataframe(mock_copy_rates, mock_select):
    import numpy as np

    dtype = np.dtype(
        [
            ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
            ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8"),
        ]
    )
    ts = int(datetime(2026, 8, 10, 12, 0, 0).timestamp())
    rates = np.array([_make_rate(ts, 1.1, 1.2, 1.05, 1.15, 1000)], dtype=dtype)
    mock_copy_rates.return_value = rates

    df = fetch_mt5_price_history("EURUSD", "H1", count=1)

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.iloc[0]["Open"] == pytest.approx(1.1)
    assert df.iloc[0]["Close"] == pytest.approx(1.15)
    assert df.iloc[0]["Volume"] == pytest.approx(1000.0)
    assert isinstance(df.index, pd.DatetimeIndex)
    mock_select.assert_called_once_with("EURUSD", True)


@patch("MetaTrader5.last_error", return_value=(-2, "no history"))
@patch("MetaTrader5.symbol_select", return_value=True)
@patch("MetaTrader5.copy_rates_from_pos", return_value=None)
def test_fetch_mt5_price_history_empty_on_no_data(mock_copy_rates, mock_select, mock_last_error, caplog):
    with caplog.at_level(logging.WARNING):
        df = fetch_mt5_price_history("UNKNOWN", "D1")
    assert df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert "UNKNOWN" in caplog.text
    assert "no history" in caplog.text


def test_fetch_mt5_price_history_rejects_unsupported_timeframe():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        fetch_mt5_price_history("EURUSD", "M15")


def _make_terminal_info(trade_allowed=True):
    info = MagicMock()
    info.trade_allowed = trade_allowed
    return info


def _make_account_info_for_trading(trade_allowed=True):
    info = MagicMock()
    info.trade_allowed = trade_allowed
    return info


@patch("MetaTrader5.account_info")
@patch("MetaTrader5.terminal_info")
def test_is_trading_permitted_true_when_both_flags_allow_it(mock_terminal, mock_account):
    mock_terminal.return_value = _make_terminal_info(trade_allowed=True)
    mock_account.return_value = _make_account_info_for_trading(trade_allowed=True)
    permitted, reason = is_trading_permitted()
    assert permitted is True
    assert reason == ""


@patch("MetaTrader5.account_info")
@patch("MetaTrader5.terminal_info")
def test_is_trading_permitted_false_when_terminal_autotrading_is_off(mock_terminal, mock_account):
    # Confirmed live: this exact condition (terminal_info().trade_allowed
    # False) silently caused every real order_send() to fail — the
    # actual root cause behind an "Apply Suggestion doesn't work" report
    # that turned out to have nothing to do with the plan's own math.
    mock_terminal.return_value = _make_terminal_info(trade_allowed=False)
    mock_account.return_value = _make_account_info_for_trading(trade_allowed=True)
    permitted, reason = is_trading_permitted()
    assert permitted is False
    assert "AutoTrading" in reason


@patch("MetaTrader5.account_info")
@patch("MetaTrader5.terminal_info")
def test_is_trading_permitted_false_when_account_itself_not_permitted(mock_terminal, mock_account):
    mock_terminal.return_value = _make_terminal_info(trade_allowed=True)
    mock_account.return_value = _make_account_info_for_trading(trade_allowed=False)
    permitted, reason = is_trading_permitted()
    assert permitted is False
    assert "not currently permitted" in reason


@patch("MetaTrader5.account_info", return_value=None)
@patch("MetaTrader5.terminal_info")
def test_is_trading_permitted_false_when_info_unavailable(mock_terminal, mock_account):
    mock_terminal.return_value = _make_terminal_info(trade_allowed=True)
    permitted, reason = is_trading_permitted()
    assert permitted is False
    assert reason != ""
