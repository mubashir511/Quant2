import logging
from unittest.mock import MagicMock, patch

import pytest

import config
from data.mt5_source import _ensure_symbol_selected, get_contract_spec, get_market_watch


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
