from unittest.mock import MagicMock, patch

from data.mt5_source import get_contract_spec


@patch("MetaTrader5.symbol_info")
def test_get_contract_spec_parses_symbol_info(mock_symbol_info):
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


@patch("MetaTrader5.symbol_info", return_value=None)
def test_get_contract_spec_none_when_symbol_not_found(mock_symbol_info):
    assert get_contract_spec("UNKNOWN") is None


@patch("MetaTrader5.symbol_info", side_effect=RuntimeError("not connected"))
def test_get_contract_spec_none_on_failure(mock_symbol_info):
    assert get_contract_spec("PALDIUM100-SE26") is None
