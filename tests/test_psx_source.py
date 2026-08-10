from unittest.mock import MagicMock, patch

import pytest
import requests

from data.psx_source import (
    PSXConnectionError,
    get_psx_company_data,
    get_psx_history,
    get_psx_market_watch,
    get_sector_name,
)

_MARKET_WATCH_HTML = """
<table><tbody class="tbl__body">
<tr>
<td data-search="ABC"><a class="tbl__symbol" href="/company/ABC" data-title="ABC Corp"><strong>ABC</strong></a></td>
<td>123</td>
<td>ALLSHR,KSE100</td>
<td class="right" data-order="10.00">10.00</td>
<td class="right" data-order="10.10">10.10</td>
<td class="right" data-order="10.50">10.50</td>
<td class="right" data-order="9.90">9.90</td>
<td class="right" data-order="10.25">10.25</td>
<td class="right" data-order="0.25">0.25</td>
<td class="right" data-order="2.500">2.50%</td>
<td class="right" data-order="500000">500,000</td>
</tr>
<tr>
<td data-search="XYZ"><a class="tbl__symbol" href="/company/XYZ" data-title="XYZ Ltd"><strong>XYZ</strong></a></td>
<td>456</td>
<td>ALLSHR</td>
<td class="right" data-order="50.00">50.00</td>
<td class="right" data-order="49.00">49.00</td>
<td class="right" data-order="51.00">51.00</td>
<td class="right" data-order="48.50">48.50</td>
<td class="right" data-order="49.50">49.50</td>
<td class="right" data-order="-0.50">-0.50</td>
<td class="right" data-order="-1.000">-1.00%</td>
<td class="right" data-order="10000">10,000</td>
</tr>
</tbody></table>
"""

_COMPANY_HTML = """
<div class="tabs__panel" data-name="REG">
<div class="stats_item"><div class="stats_label">Open</div><div class="stats_value">10.10</div></div>
<div class="stats_item"><div class="stats_label">P/E Ratio (TTM) **</div><div class="stats_value">4.13</div></div>
<div class="stats_item"><div class="stats_label">CIRCUIT BREAKER</div><div class="stats_value">10.13 — 12.38</div></div>
<div class="stats_item"><div class="stats_label">52-WEEK RANGE ^</div><div class="stats_value">5.87 — 12.24</div></div>
</div>
<div class="tabs__panel" data-name="FUT">
<div class="stats_item"><div class="stats_label">Open</div><div class="stats_value">99.99</div></div>
</div>
<div class="stats stats--wrappable">
<div class="stats_item"><div class="stats_label">Market Cap (000'<span>s</span>)</div><div class="stats_value">65,591,764.00</div></div>
<div class="stats_item"><div class="stats_label">Shares</div><div class="stats_value">5,493,447,571</div></div>
<div class="stats_item"><div class="stats_label">Free Float</div><div class="stats_value">1,373,361,893</div></div>
<div class="stats_item"><div class="stats_label">Free Float</div><div class="stats_value">25.00%</div></div>
</div>
<div class="company__profile">
<div class="profile__items">
<div class="profile__item profile__item--decription"><div class="item__head">BUSINESS DESCRIPTION</div><p>Makes widgets.</p></div>
<div class="profile__item profile__item--people"><div class="item__head">KEY PEOPLE</div><table class="tbl"><tbody class="tbl__body">
<tr><td><strong>Jane Doe</strong></td><td>CEO</td></tr>
<tr><td><strong>John Roe</strong></td><td>Chairperson</td></tr>
</tbody></table></div>
</div>
<div class="profile__items">
<div class="profile__item"><div class="item__head">ADDRESS</div><p>123 Main St</p><div class="item__head">WEBSITE</div><p>www.example.com</p></div>
<div class="profile__item"><div class="item__head">AUDITOR</div><p>Some Audit Firm</p><div class="item__head">Fiscal Year End</div><p>June</p></div>
</div>
</div>
<div id="financialTab">
<div class="tabs__panels">
<div class="tabs__panel" data-name="Annual">
<div class="tbl__wrapper"><table>
<thead class="tbl__head"><tr><th></th><th class="right">2025</th><th class="right">2024</th></tr></thead>
<tbody class="tbl__body">
<tr><td>Sales</td><td class="right">100,000</td><td class="right">90,000</td></tr>
<tr><td>Profit after Taxation</td><td class="right">(5,000)</td><td class="right">3,000</td></tr>
<tr><td>EPS</td><td class="right">(0.50)</td><td class="right">0.30</td></tr>
</tbody>
</table></div>
</div>
<div class="tabs__panel" data-name="Quarterly">
<div class="tbl__wrapper"><table>
<thead class="tbl__head"><tr><th></th><th class="right">Q1 2026</th></tr></thead>
<tbody class="tbl__body"><tr><td>Sales</td><td class="right">25,000</td></tr></tbody>
</table></div>
</div>
</div>
</div>
<div id="ratios">
<div class="company__ratios">
<div class="tbl__wrapper"><table>
<thead class="tbl__head"><tr><th></th><th class="right">2025</th></tr></thead>
<tbody class="tbl__body"><tr><td>PEG</td><td class="right">0.10</td></tr></tbody>
</table></div>
</div>
</div>
<div class="tabs__panel" data-name="Financial Results">
<table><tbody class="tbl__body">
<tr><td>Apr 28, 2026</td><td>Financial Results for the Quarter</td><td>PDF</td></tr>
</tbody></table>
</div>
<div class="tabs__panel" data-name="Board Meetings">
<table><tbody class="tbl__body">
<tr><td>Apr 20, 2026</td><td>Board Meeting and Closed Period</td><td>PDF</td></tr>
</tbody></table>
</div>
<div class="tabs__panel" data-name="Others">
<table><tbody class="tbl__body">
<tr><td>Mar 9, 2026</td><td>Disclosure of Interest</td><td>PDF</td></tr>
</tbody></table>
</div>
"""

_EOD_JSON = {
    "status": 1,
    "data": [
        [1700000000, 10.5, 500000, 10.0],
        [1700086400, 10.6, 600000, 10.5],
    ],
}


def _mock_response(text=None, json_data=None, status_ok=True):
    resp = MagicMock()
    if text is not None:
        resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    if not status_ok:
        resp.raise_for_status.side_effect = requests.HTTPError("boom")
    else:
        resp.raise_for_status.return_value = None
    return resp


@patch("data.psx_source.requests.get")
def test_get_psx_market_watch_parses_all_rows(mock_get):
    mock_get.return_value = _mock_response(text=_MARKET_WATCH_HTML)
    assets = get_psx_market_watch()
    assert {a.symbol for a in assets} == {"ABC", "XYZ"}

    abc = next(a for a in assets if a.symbol == "ABC")
    assert abc.company_name == "ABC Corp"
    assert abc.sector_code == "123"
    assert abc.listed_in == ["ALLSHR", "KSE100"]
    assert abc.ldcp == 10.00
    assert abc.current == 10.25
    assert abc.change_pct == 2.5
    assert abc.volume == 500000

    xyz = next(a for a in assets if a.symbol == "XYZ")
    assert xyz.change == -0.50
    assert "KSE100" not in xyz.listed_in


@patch("data.psx_source.requests.get")
def test_get_psx_market_watch_raises_on_request_failure(mock_get):
    mock_get.return_value = _mock_response(text="", status_ok=False)
    with pytest.raises(PSXConnectionError):
        get_psx_market_watch()


@patch("data.psx_source.requests.get")
def test_get_psx_market_watch_raises_when_table_missing(mock_get):
    mock_get.return_value = _mock_response(text="<html><body>no table here</body></html>")
    with pytest.raises(PSXConnectionError):
        get_psx_market_watch()


@patch("data.psx_source.requests.get")
def test_get_psx_market_watch_skips_unparseable_row_without_failing_whole_fetch(mock_get):
    bad_html = """
    <table><tbody class="tbl__body">
    <tr><td>ONLY</td><td>ONE</td><td>CELL</td></tr>
    <tr>
    <td data-search="OK"><a data-title="OK Corp"><strong>OK</strong></a></td>
    <td>1</td><td>ALLSHR</td>
    <td class="right" data-order="1.0">1.0</td>
    <td class="right" data-order="1.0">1.0</td>
    <td class="right" data-order="1.0">1.0</td>
    <td class="right" data-order="1.0">1.0</td>
    <td class="right" data-order="1.0">1.0</td>
    <td class="right" data-order="0.0">0.0</td>
    <td class="right" data-order="0.0">0.0%</td>
    <td class="right" data-order="100">100</td>
    </tr>
    </tbody></table>
    """
    mock_get.return_value = _mock_response(text=bad_html)
    assets = get_psx_market_watch()
    assert {a.symbol for a in assets} == {"OK"}


@patch("data.psx_source.requests.get")
def test_get_psx_history_parses_eod_json_oldest_first(mock_get):
    mock_get.return_value = _mock_response(json_data=_EOD_JSON)
    history = get_psx_history("ABC")
    assert list(history.columns) == ["Open", "Close", "Volume"]
    assert len(history) == 2
    assert history["Close"].iloc[0] == 10.5
    assert history.index.is_monotonic_increasing


@patch("data.psx_source.requests.get")
def test_get_psx_history_deduplicates_exact_duplicate_timestamps(mock_get):
    # Confirmed live: the real KSE100 index feed contains one exact
    # duplicate timestamp — left unhandled, this breaks any caller that
    # aligns this series against another by date (e.g. beta computation).
    mock_get.return_value = _mock_response(
        json_data={
            "status": 1,
            "data": [
                [1700086400, 10.6, 600000, 10.5],
                [1700086400, 99.9, 999999, 99.9],  # exact duplicate timestamp, different values
                [1700000000, 10.5, 500000, 10.0],
            ],
        }
    )
    history = get_psx_history("KSE100")
    assert len(history) == 2
    assert not history.index.duplicated().any()


@patch("data.psx_source.requests.get")
def test_get_psx_history_empty_on_request_failure(mock_get):
    mock_get.return_value = _mock_response(text="", status_ok=False)
    history = get_psx_history("ABC")
    assert history.empty
    assert list(history.columns) == ["Open", "Close", "Volume"]


@patch("data.psx_source.requests.get")
def test_get_psx_history_empty_when_no_data_key(mock_get):
    mock_get.return_value = _mock_response(json_data={"status": 0})
    history = get_psx_history("ABC")
    assert history.empty


@patch("data.psx_source.requests.get")
def test_get_psx_company_data_parses_fundamentals(mock_get):
    mock_get.return_value = _mock_response(text=_COMPANY_HTML)
    data = get_psx_company_data("ABC")
    f = data.fundamentals
    assert f.pe_ratio == 4.13
    assert f.circuit_breaker_low == 10.13
    assert f.circuit_breaker_high == 12.38
    assert f.week52_low == 5.87
    assert f.week52_high == 12.24
    assert f.market_cap_pkr_000s == 65591764.00
    assert f.free_float_pct == 25.00
    assert f.shares_outstanding == 5493447571.0


@patch("data.psx_source.requests.get")
def test_get_psx_company_data_parses_profile(mock_get):
    mock_get.return_value = _mock_response(text=_COMPANY_HTML)
    profile = get_psx_company_data("ABC").profile
    assert profile.description == "Makes widgets."
    assert profile.key_people == [("Jane Doe", "CEO"), ("John Roe", "Chairperson")]
    assert profile.auditor == "Some Audit Firm"
    assert profile.fiscal_year_end == "June"


@patch("data.psx_source.requests.get")
def test_get_psx_company_data_profile_none_when_section_missing(mock_get):
    mock_get.return_value = _mock_response(text="<html><body>no profile</body></html>")
    assert get_psx_company_data("ABC").profile is None


@patch("data.psx_source.requests.get")
def test_get_psx_company_data_parses_financials_with_negative_parentheses(mock_get):
    mock_get.return_value = _mock_response(text=_COMPANY_HTML)
    fin = get_psx_company_data("ABC").financials
    assert fin.annual_periods == ["2025", "2024"]
    assert fin.annual["Sales"] == [100000.0, 90000.0]
    assert fin.annual["Profit after Taxation"] == [-5000.0, 3000.0]
    assert fin.annual["EPS"] == [-0.50, 0.30]
    assert fin.quarterly_periods == ["Q1 2026"]
    assert fin.quarterly["Sales"] == [25000.0]
    assert fin.ratio_periods == ["2025"]
    assert fin.ratios["PEG"] == [0.10]


@patch("data.psx_source.requests.get")
def test_get_psx_company_data_merges_and_sorts_announcements_by_date(mock_get):
    mock_get.return_value = _mock_response(text=_COMPANY_HTML)
    announcements = get_psx_company_data("ABC").announcements
    assert [a.date for a in announcements] == ["Apr 28, 2026", "Apr 20, 2026", "Mar 9, 2026"]
    assert announcements[0].category == "Financial Results"
    assert announcements[1].category == "Board Meetings"
    assert announcements[0].title == "Financial Results for the Quarter"


@patch("data.psx_source.requests.get")
def test_get_psx_company_data_all_none_when_reg_panel_missing(mock_get):
    mock_get.return_value = _mock_response(text="<html><body>no panel</body></html>")
    data = get_psx_company_data("ABC")
    assert data.fundamentals is None
    assert data.financials is None
    assert data.announcements == []
    assert data.profile is None


@patch("data.psx_source.requests.get")
def test_get_psx_company_data_all_empty_on_request_failure(mock_get):
    mock_get.return_value = _mock_response(text="", status_ok=False)
    data = get_psx_company_data("ABC")
    assert data.fundamentals is None
    assert data.financials is None
    assert data.announcements == []
    assert data.profile is None


@patch("data.psx_source.requests.get")
def test_get_psx_company_data_only_one_fetch_per_symbol(mock_get):
    mock_get.return_value = _mock_response(text=_COMPANY_HTML)
    get_psx_company_data("ABC")
    assert mock_get.call_count == 1


def test_get_sector_name_normalizes_inconsistent_zero_padding():
    # Confirmed live: the real market-watch table mixes "807" (3 digits)
    # and "0825" (4 digits) for different symbols in the same column.
    assert get_sector_name("0807") == "Commercial Banks"
    assert get_sector_name("807") == "Commercial Banks"
    assert get_sector_name("0825") == "Refinery"


def test_get_sector_name_reit_and_technology_present():
    # Directly confirms the two categories the user explicitly named.
    assert get_sector_name("0836") == "Real Estate Investment Trust"
    assert get_sector_name("0828") == "Technology & Communication"


def test_get_sector_name_unknown_code_returns_labeled_fallback():
    result = get_sector_name("9999")
    assert "Unclassified" in result
    assert "9999" in result
