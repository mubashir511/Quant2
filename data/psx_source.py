import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

# PSX's own public Data Portal — free, but officially licensed only for
# non-commercial personal use (a real-time/licensed feed is a separate
# paid commercial product, per psx.com.pk's own data-services page).
# Live-verified: /market-watch is a plain server-rendered HTML table (no
# JS rendering needed), /timeseries/eod/{symbol} is a clean JSON endpoint,
# and /company/{symbol} exposes real fundamentals (P/E, market cap, free
# float, circuit-breaker band, 52-week range) as labeled stat blocks.
_BASE_URL = "https://dps.psx.com.pk"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Curated from the real "listed in" tags found live across the full
# market-watch table (many more exist — asset-management-company-specific
# baskets like NBPPGI/JSMFI/OGTI with a handful of members each — but
# those aren't indices a retail user would recognize or want to narrow a
# suggestion pool to). Each of these is also confirmed live to work
# against /timeseries/eod/{tag} the same way an individual symbol does,
# so a selected index doubles as its own relative-strength benchmark.
PSX_INDICES: dict[str, str] = {
    "KSE100": "KSE-100 (top 100 large-cap)",
    "KSE30": "KSE-30 (top 30 by free-float market cap)",
    "KMI30": "KMI-30 (Shariah-compliant, top 30)",
    "KMIALLSHR": "KMI All Share (all Shariah-compliant listings)",
    "PSXDIV20": "PSX Dividend 20 (top 20 dividend payers)",
    "ALLSHR": "All Share (every listed company)",
}

# The real, official sector code -> name mapping, from the "sector"
# filter dropdown on PSX's own main site (psx.com.pk/psx/resources-and-
# tools/listings/listed-companies — a different domain than this file's
# usual dps.psx.com.pk, but still PSX's own site). market-watch's
# sector_code field is confirmed live to match these values, just
# inconsistently zero-padded (e.g. "807" for one symbol, "0825" for
# another in the same table) — always normalize via zfill(4) before
# looking up (see get_sector_name). Cross-checked against real data
# before trusting this mapping: CNERGY's code 0825 -> "Refinery" matches
# its own business description ("Oil Refinery Business"); BOP's code 807
# -> "Commercial banks" is correct for Bank of Punjab.
PSX_SECTOR_NAMES: dict[str, str] = {
    "0839": "Apparel",
    "0801": "Automobile Assembler",
    "0802": "Automobile Parts & Accessories",
    "0036": "Bonds",
    "0803": "Cable & Electrical Goods",
    "0804": "Cement",
    "0805": "Chemical",
    "0806": "Close-End Mutual Fund",
    "0807": "Commercial Banks",
    "0808": "Engineering",
    "0837": "Exchange Traded Funds",
    "0809": "Fertilizer",
    "0810": "Food & Personal Care Products",
    "0040": "Future Contracts",
    "0811": "Glass & Ceramics",
    "0812": "Insurance",
    "0813": "Inv. Banks / Inv. Cos. / Securities Cos.",
    "0814": "Jute",
    "0815": "Leasing Companies",
    "0816": "Leather & Tanneries",
    "0818": "Miscellaneous",
    "0819": "Modarabas",
    "0820": "Oil & Gas Exploration Companies",
    "0821": "Oil & Gas Marketing Companies",
    "0822": "Paper, Board & Packaging",
    "0823": "Pharmaceuticals",
    "0824": "Power Generation & Distribution",
    "0838": "Property",
    "0836": "Real Estate Investment Trust",
    "0825": "Refinery",
    "0041": "Stock Index Future Contracts",
    "0826": "Sugar & Allied Industries",
    "0827": "Synthetic & Rayon",
    "0828": "Technology & Communication",
    "0829": "Textile Composite",
    "0830": "Textile Spinning",
    "0831": "Textile Weaving",
    "0832": "Tobacco",
    "0833": "Transport",
    "0834": "Vanaspati & Allied Industries",
    "0835": "Woollen",
}


def get_sector_name(sector_code: str) -> str:
    """Normalizes market-watch's inconsistently zero-padded sector_code
    (observed live as both "807" and "0825" in the same table) to the
    4-digit form PSX's own official sector list uses, before lookup.
    Returns a clearly-marked fallback rather than raising or guessing —
    this mapping was hand-transcribed from a live page and could miss a
    code the site adds later."""
    normalized = sector_code.strip().zfill(4)
    return PSX_SECTOR_NAMES.get(normalized, f"Unclassified (code {sector_code})")


class PSXConnectionError(RuntimeError):
    pass


def _get(path: str) -> requests.Response:
    try:
        resp = requests.get(
            f"{_BASE_URL}{path}", headers=_HEADERS, timeout=config.PSX_REQUEST_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise PSXConnectionError(
            f"Failed to reach the PSX Data Portal ({type(e).__name__}: {e})."
        ) from e
    return resp


@dataclass
class PSXAsset:
    symbol: str
    company_name: str | None
    sector_code: str
    listed_in: list[str]  # index-membership tags, e.g. ["KSE100", "ALLSHR"]
    ldcp: float  # last day's closing price
    open: float
    high: float
    low: float
    current: float
    change: float
    change_pct: float
    volume: int


def _cell_number(cell) -> float:
    # Every numeric cell on the market-watch table carries the clean raw
    # value in data-order (no thousands separators, no up/down-arrow icon
    # text mixed in) — far more reliable than parsing the display text.
    raw = cell.get("data-order")
    if raw is None:
        raw = cell.get_text(strip=True)
    return float(str(raw).replace(",", ""))


def get_psx_market_watch() -> list[PSXAsset]:
    """Every symbol currently listed on PSX with its live quote (~490
    symbols, confirmed live), scraped from the exchange's own public
    market-watch page. Raises PSXConnectionError on a network failure or
    if the page's table structure isn't found at all — a layout change
    here would otherwise make every PSX symbol silently vanish at once,
    which should be loud, not silent (same reasoning as
    data/mt5_source.py's MT5ConnectionError for the PMEX side)."""
    resp = _get("/market-watch")
    soup = BeautifulSoup(resp.text, "lxml")
    body = soup.select_one("tbody.tbl__body")
    if body is None:
        raise PSXConnectionError(
            "PSX market-watch page didn't contain the expected table — its layout may have changed."
        )

    assets = []
    for row in body.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 11:
            continue
        symbol_link = cells[0].find("a")
        symbol = (symbol_link or cells[0]).get_text(strip=True)
        if not symbol:
            continue
        listed_in = [tag.strip() for tag in cells[2].get_text(strip=True).split(",") if tag.strip()]
        try:
            assets.append(
                PSXAsset(
                    symbol=symbol,
                    company_name=symbol_link.get("data-title") if symbol_link else None,
                    sector_code=cells[1].get_text(strip=True),
                    listed_in=listed_in,
                    ldcp=_cell_number(cells[3]),
                    open=_cell_number(cells[4]),
                    high=_cell_number(cells[5]),
                    low=_cell_number(cells[6]),
                    current=_cell_number(cells[7]),
                    change=_cell_number(cells[8]),
                    change_pct=_cell_number(cells[9]),
                    volume=int(_cell_number(cells[10])),
                )
            )
        except (ValueError, TypeError):
            # One row with an unparseable number shouldn't take down the
            # whole fetch — skip just that symbol.
            continue
    return assets


_HISTORY_COLUMNS = ["Open", "Close", "Volume"]


def get_psx_history(symbol: str) -> pd.DataFrame:
    """Daily Open/Close/Volume history for one symbol from PSX's own
    public EOD timeseries endpoint. No High/Low in this feed, unlike
    yfinance for the PMEX side — Average True Range stays unavailable for
    PSX instruments (analysis.technical.compute_technical_stats already
    handles a history frame with no High/Low by leaving ATR as None, the
    same "explicitly disclosed as unavailable" convention used everywhere
    else in this project, never silently omitted).

    Returns an empty (but correctly-typed) DataFrame on any failure —
    this is best-effort per-instrument enrichment, not core data, so it
    follows data/market_history.py's fetch_price_history_ohlcv contract
    rather than raising."""
    try:
        resp = _get(f"/timeseries/eod/{symbol}")
        payload = resp.json()
    except (PSXConnectionError, json.JSONDecodeError) as e:
        logger.warning("get_psx_history(%s): %s: %s", symbol, type(e).__name__, e)
        return pd.DataFrame(columns=_HISTORY_COLUMNS, dtype=float)

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not rows:
        return pd.DataFrame(columns=_HISTORY_COLUMNS, dtype=float)

    records = []
    for entry in rows:
        try:
            ts, close, volume, open_ = entry
            records.append(
                {
                    "date": pd.Timestamp.fromtimestamp(float(ts)),
                    "Open": float(open_),
                    "Close": float(close),
                    "Volume": float(volume),
                }
            )
        except (ValueError, TypeError, IndexError):
            continue

    if not records:
        return pd.DataFrame(columns=_HISTORY_COLUMNS, dtype=float)

    df = pd.DataFrame(records).set_index("date").sort_index()
    # The feed has been observed live to contain an occasional exact-
    # duplicate timestamp (confirmed for the KSE100 index specifically) —
    # a genuine source-data quirk, not a parsing bug. A daily series with
    # two rows for the same timestamp breaks any caller that aligns this
    # against another series by date (e.g. ai/psx_suggest.py's beta
    # calculation), so de-duplicate here once for every consumer rather
    # than making each caller defend against it independently.
    df = df[~df.index.duplicated(keep="last")]
    return df[_HISTORY_COLUMNS]


@dataclass
class PSXFundamentals:
    pe_ratio: float | None
    market_cap_pkr_000s: float | None
    free_float_pct: float | None
    shares_outstanding: float | None
    circuit_breaker_low: float | None
    circuit_breaker_high: float | None
    week52_low: float | None
    week52_high: float | None


@dataclass
class PSXCompanyProfile:
    """From the company page's own 'Company Profile' section — real
    qualitative context (what the business actually does, who runs it,
    who audits it) that no numeric stat captures. Only the fields with
    genuine decision relevance are kept — address/website/registrar are
    parsed by the page too but aren't useful for a portfolio decision, so
    they're deliberately not extracted here."""

    description: str | None
    key_people: list[tuple[str, str]]  # (name, role), e.g. ("Jane Doe", "CEO")
    auditor: str | None
    fiscal_year_end: str | None


@dataclass
class PSXFinancials:
    """Real reported figures straight from the company page's own
    financial-statements tables (annual + trailing quarters + ratios) —
    not derived/estimated. Each metrics dict's value-lists line up
    positionally with the matching *_periods list (newest period first,
    matching the page's own column order). A metric absent for this
    company simply doesn't appear as a key, rather than being padded with
    None entries."""

    annual_periods: list[str]
    annual: dict[str, list[float | None]]
    quarterly_periods: list[str]
    quarterly: dict[str, list[float | None]]
    ratio_periods: list[str]
    ratios: dict[str, list[float | None]]


@dataclass
class PSXAnnouncement:
    category: str  # "Financial Results" / "Board Meetings" / "Others"
    date: str
    title: str


@dataclass
class PSXCompanyData:
    fundamentals: PSXFundamentals | None
    financials: PSXFinancials | None
    announcements: list[PSXAnnouncement]
    profile: PSXCompanyProfile | None


_RANGE_NUMBER_PATTERN = re.compile(r"[\d,]+\.?\d*")


def _range_from_text(raw: str) -> tuple[float | None, float | None]:
    # The separator between the two numbers (an em dash in the source
    # page) has come back mangled under some encodings in testing —
    # extracting the numbers directly by regex sidesteps that entirely
    # rather than depending on a specific separator character.
    numbers = _RANGE_NUMBER_PATTERN.findall(raw)
    if len(numbers) != 2:
        return None, None
    try:
        return float(numbers[0].replace(",", "")), float(numbers[1].replace(",", ""))
    except ValueError:
        return None, None


def _stats_pairs(container) -> list[tuple[str, str]]:
    pairs = []
    for item in container.select("div.stats_item"):
        label = item.select_one("div.stats_label")
        value = item.select_one("div.stats_value")
        if label is not None and value is not None:
            pairs.append((label.get_text(strip=True), value.get_text(" ", strip=True)))
    return pairs


def _extract_fundamentals(soup, symbol: str) -> PSXFundamentals | None:
    """Scoped to the 'REG' (regular/READY market) tab specifically and
    the page's top-level 'Company Profile' stats block — the same page
    repeats OPEN/HIGH/LOW-shaped labels under FUT/CSF/ODL tabs for the
    same symbol's futures/leveraged variants, which would otherwise
    collide with a page-wide label scan."""
    reg_panel = soup.select_one('div.tabs__panel[data-name="REG"]')
    if reg_panel is None:
        logger.warning("get_psx_company_data(%s): REG panel not found on company page", symbol)
        return None
    reg_stats = dict(_stats_pairs(reg_panel))

    pe_ratio = None
    for label in ("P/E Ratio (TTM) **", "P/E Ratio (TTM)", "P/E Ratio"):
        if label in reg_stats:
            try:
                pe_ratio = float(reg_stats[label].replace(",", ""))
            except ValueError:
                pe_ratio = None
            break

    cb_low, cb_high = _range_from_text(reg_stats.get("CIRCUIT BREAKER", ""))
    week52_low, week52_high = None, None
    for label in ("52-WEEK RANGE ^", "52-WEEK RANGE"):
        if label in reg_stats:
            week52_low, week52_high = _range_from_text(reg_stats[label])
            break

    market_cap = None
    free_float_pct = None
    shares_outstanding = None
    profile_block = soup.select_one("div.stats.stats--wrappable")
    if profile_block is not None:
        for label, value in _stats_pairs(profile_block):
            cleaned = value.replace(",", "")
            if label.startswith("Market Cap") and market_cap is None:
                try:
                    market_cap = float(cleaned)
                except ValueError:
                    pass
            elif label == "Shares" and shares_outstanding is None:
                try:
                    shares_outstanding = float(cleaned)
                except ValueError:
                    pass
            elif label == "Free Float" and "%" in value:
                try:
                    free_float_pct = float(cleaned.replace("%", ""))
                except ValueError:
                    pass

    return PSXFundamentals(
        pe_ratio=pe_ratio,
        market_cap_pkr_000s=market_cap,
        free_float_pct=free_float_pct,
        shares_outstanding=shares_outstanding,
        circuit_breaker_low=cb_low,
        circuit_breaker_high=cb_high,
        week52_low=week52_low,
        week52_high=week52_high,
    )


def _extract_profile(soup, symbol: str) -> PSXCompanyProfile | None:
    """The 'Company Profile' section's label/value pairs aren't grouped
    consistently one-per-container (e.g. ADDRESS and WEBSITE share one
    wrapper) — walking every div.item__head and reading its very next
    sibling element (a <p> for plain text, a <table> for the key-people
    list) is robust to that regardless of how they're grouped."""
    section = soup.select_one("div.company__profile")
    if section is None:
        logger.info("get_psx_company_data(%s): no Company Profile section found", symbol)
        return None

    description = None
    auditor = None
    fiscal_year_end = None
    key_people: list[tuple[str, str]] = []

    for head in section.select("div.item__head"):
        label = head.get_text(strip=True)
        value_el = head.find_next_sibling()
        if value_el is None:
            continue
        if value_el.name == "table":
            for row in value_el.select("tbody tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    key_people.append((cells[0].get_text(strip=True), cells[1].get_text(strip=True)))
        elif value_el.name == "p":
            text = value_el.get_text(strip=True)
            if label == "BUSINESS DESCRIPTION":
                description = text or None
            elif label == "AUDITOR":
                auditor = text or None
            elif label.upper() == "FISCAL YEAR END":
                fiscal_year_end = text or None

    if description is None and not key_people and auditor is None:
        return None
    return PSXCompanyProfile(
        description=description,
        key_people=key_people,
        auditor=auditor,
        fiscal_year_end=fiscal_year_end,
    )


def _parse_financial_table(table) -> tuple[list[str], dict[str, list[float | None]]]:
    header_row = table.select_one("thead tr")
    if header_row is None:
        return [], {}
    periods = [th.get_text(strip=True) for th in header_row.find_all("th")[1:]]

    metrics: dict[str, list[float | None]] = {}
    for row in table.select("tbody tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        label = cells[0].get_text(strip=True)
        if not label:
            continue
        values = []
        for cell in cells[1:]:
            raw = cell.get_text(strip=True).replace(",", "")
            # Accounting convention: a negative figure is shown in
            # parentheses, e.g. "(2,895,421)" for a loss.
            negative = raw.startswith("(") and raw.endswith(")")
            if negative:
                raw = raw[1:-1]
            try:
                value = float(raw) if raw else None
            except ValueError:
                value = None
            values.append(-value if (value is not None and negative) else value)
        metrics[label] = values
    return periods, metrics


def _extract_financials(soup, symbol: str) -> PSXFinancials | None:
    """Real reported Sales/Profit-after-Tax/EPS (annual + trailing
    quarters) and Gross/Net Profit Margin/EPS Growth/PEG ratios, straight
    from the company page's own financial-statements tables — not
    estimated or derived. None only if the page has none of these tables
    at all (e.g. a symbol type that doesn't file PSX financial
    statements); a company missing just one of the three tables still
    gets the other two rather than an all-or-nothing None."""
    annual_table = soup.select_one('div#financialTab div.tabs__panel[data-name="Annual"] table')
    quarterly_table = soup.select_one(
        'div#financialTab div.tabs__panel[data-name="Quarterly"] table'
    )
    ratios_table = soup.select_one("div.company__ratios table")

    if annual_table is None and quarterly_table is None and ratios_table is None:
        logger.warning("get_psx_company_data(%s): no financial-statement tables found", symbol)
        return None

    annual_periods, annual = _parse_financial_table(annual_table) if annual_table else ([], {})
    quarterly_periods, quarterly = (
        _parse_financial_table(quarterly_table) if quarterly_table else ([], {})
    )
    ratio_periods, ratios = _parse_financial_table(ratios_table) if ratios_table else ([], {})

    return PSXFinancials(
        annual_periods=annual_periods,
        annual=annual,
        quarterly_periods=quarterly_periods,
        quarterly=quarterly,
        ratio_periods=ratio_periods,
        ratios=ratios,
    )


_ANNOUNCEMENT_CATEGORIES = ("Financial Results", "Board Meetings", "Others")
_ANNOUNCEMENTS_PER_CATEGORY = 5
_ANNOUNCEMENTS_TOTAL_LIMIT = 8


def _parse_announcement_date(date_text: str) -> datetime | None:
    try:
        return datetime.strptime(date_text, "%b %d, %Y")
    except ValueError:
        return None


def _extract_announcements(soup, symbol: str) -> list[PSXAnnouncement]:
    """The company page's own 'Financial Results' / 'Board Meetings' /
    'Others' tabs are PSX's own real, dated regulatory disclosures for
    this specific symbol — a genuinely official source for "company
    reports and news", not a generic web search. Merged across all three
    categories and sorted newest-first by real parsed date (falling back
    to the page's own order for any date that doesn't parse), then
    capped to _ANNOUNCEMENTS_TOTAL_LIMIT overall so a company with a long
    disclosure history doesn't dominate the prompt over its peers."""
    announcements = []
    for category in _ANNOUNCEMENT_CATEGORIES:
        panel = soup.select_one(f'div.tabs__panel[data-name="{category}"]')
        if panel is None:
            continue
        rows = panel.select("table tbody tr")[:_ANNOUNCEMENTS_PER_CATEGORY]
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            date_text = cells[0].get_text(strip=True)
            title = cells[1].get_text(strip=True)
            if date_text and title:
                announcements.append(PSXAnnouncement(category=category, date=date_text, title=title))

    if not announcements:
        logger.info("get_psx_company_data(%s): no announcements found", symbol)

    announcements.sort(
        key=lambda a: _parse_announcement_date(a.date) or datetime.min, reverse=True
    )
    return announcements[:_ANNOUNCEMENTS_TOTAL_LIMIT]


def get_psx_company_data(symbol: str) -> PSXCompanyData:
    """One fetch of the symbol's PSX Data Portal company page, bundling
    fundamentals, real reported financial-statement figures, recent
    official company disclosures, and the company profile (business
    description, key people, auditor) — deliberately a single HTTP
    request rather than four separate ones, since all of this lives on
    the same page. Never raises: each piece degrades to None/empty
    independently on its own parsing failure (logged), while a total
    fetch failure (network, non-2xx) degrades all of them at once — this
    is optional per-instrument enrichment, not core data, same
    convention as data/mt5_source.py::get_contract_spec."""
    try:
        resp = _get(f"/company/{symbol}")
    except PSXConnectionError as e:
        logger.warning("get_psx_company_data(%s): %s", symbol, e)
        return PSXCompanyData(fundamentals=None, financials=None, announcements=[], profile=None)

    soup = BeautifulSoup(resp.text, "lxml")
    return PSXCompanyData(
        fundamentals=_extract_fundamentals(soup, symbol),
        financials=_extract_financials(soup, symbol),
        announcements=_extract_announcements(soup, symbol),
        profile=_extract_profile(soup, symbol),
    )
