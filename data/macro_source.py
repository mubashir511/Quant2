import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2/country"

# State Bank of Pakistan's own public rates page — a real, official,
# free equivalent to this file's US Treasury yield curve (see
# fetch_market_indicators) for the PSX side. Live-verified: plain
# server-rendered <table>s (no JS needed), one for KIBOR (the interbank
# lending benchmark most Pakistani financial commentary actually cites,
# closer in practice to the policy rate than any other free structured
# figure found) and several for MTB/PIB government-bond auction cut-off
# yields — a genuine Pakistani yield curve.
SBP_RATES_URL = "https://www.sbp.org.pk/ecodata/kibor_index.asp"

# Taiwan is deliberately excluded — the World Bank API doesn't carry it as
# a reporting entity (political-recognition status), confirmed empty on a
# live check. Its government data is instead pointed at via a research
# directive in ai/portfolio_suggest.py (search its DGBAS statistics portal)
# rather than faked here as a structured fetch.
COUNTRY_NAMES = {
    "US": "United States",
    "GB": "United Kingdom",
    "FR": "France",
    "DE": "Germany",
    "JP": "Japan",
    "CN": "China",
    "IN": "India",
    "KR": "South Korea",
    "SA": "Saudi Arabia",
    "AE": "United Arab Emirates",
    "PK": "Pakistan",
}

INDICATOR_CODES = {
    "gdp_growth_pct": "NY.GDP.MKTP.KD.ZG",
    "inflation_pct": "FP.CPI.TOTL.ZG",
    "unemployment_pct": "SL.UEM.TOTL.ZS",
}


@dataclass
class MarketIndicators:
    yield_3m_pct: float | None
    yield_10y_pct: float | None
    yield_30y_pct: float | None
    yield_curve_10y_3m_spread: float | None
    dxy: float | None
    vix: float | None


@dataclass
class CountryIndicators:
    country: str
    gdp_growth_pct: float | None
    inflation_pct: float | None
    unemployment_pct: float | None


@dataclass
class PakistanRates:
    as_of: str | None
    kibor_pct: dict[str, float] = field(default_factory=dict)  # tenor -> mid of bid/offer
    mtb_yield_pct: dict[str, float] = field(default_factory=dict)  # T-Bill cut-off yields
    pib_yield_pct: dict[str, float] = field(default_factory=dict)  # Fixed-rate PIB cut-off yields


def _fetch_latest_price(ticker: str) -> float | None:
    import yfinance as yf

    try:
        history = yf.Ticker(ticker).history(period="5d")
    except Exception:
        return None
    if history.empty:
        return None
    return float(history["Close"].iloc[-1])


def fetch_fx_rate_to_usd(currency_code: str) -> float | None:
    """How many units of `currency_code` per 1 USD (e.g. "PKR" -> ~277).

    Uses Yahoo's f"{code}=X" convention (verified live). Returns None for
    "USD" itself or on any fetch failure — callers should treat that as
    "don't show an FX section" rather than an error."""
    if currency_code.upper() == "USD":
        return None
    return _fetch_latest_price(f"{currency_code.upper()}=X")


def fetch_market_indicators() -> MarketIndicators:
    """US Treasury yield curve + dollar index + VIX. Any ticker that fails
    to fetch is left as None rather than failing the whole snapshot. The 5
    tickers are independent HTTP calls, fetched in parallel for speed."""
    tickers = ["^IRX", "^TNX", "^TYX", "DX-Y.NYB", "^VIX"]
    with ThreadPoolExecutor(max_workers=len(tickers)) as pool:
        yield_3m, yield_10y, yield_30y, dxy, vix = pool.map(_fetch_latest_price, tickers)

    spread = None
    if yield_10y is not None and yield_3m is not None:
        spread = yield_10y - yield_3m

    return MarketIndicators(
        yield_3m_pct=yield_3m,
        yield_10y_pct=yield_10y,
        yield_30y_pct=yield_30y,
        yield_curve_10y_3m_spread=spread,
        dxy=dxy,
        vix=vix,
    )


def _fetch_latest_indicator_value(country_code: str, indicator_code: str) -> float | None:
    import time

    import requests

    url = f"{WORLD_BANK_BASE_URL}/{country_code}/indicator/{indicator_code}"
    payload = None
    # One retry with a short backoff: fetching many countries concurrently
    # occasionally trips a transient failure against this free public API,
    # confirmed live — a lone retry recovers those without falling back to
    # fully sequential fetching.
    for attempt in range(2):
        try:
            response = requests.get(url, params={"format": "json", "per_page": 10}, timeout=10)
            response.raise_for_status()
            payload = response.json()
            break
        except Exception:
            if attempt == 0:
                time.sleep(0.5)

    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        return None

    for entry in payload[1]:
        if entry.get("value") is not None:
            return float(entry["value"])
    return None


def fetch_country_indicators(
    countries: tuple[str, ...] = ("US", "GB", "FR", "DE", "JP", "CN", "IN", "KR", "SA", "AE")
) -> list[CountryIndicators]:
    """Latest available GDP growth / inflation / unemployment per country,
    via the World Bank's free, keyless public API. Any missing indicator
    for a country is left as None rather than dropping that country. Every
    (country, indicator) pair is an independent HTTP call, fetched with
    modest concurrency — sequential would mean 30 round-trips for the
    default 10 countries, but a live test showed 16-way concurrency against
    this API caused transient failures on some requests (confirmed to be a
    concurrency artifact, not a real data gap, by re-fetching sequentially);
    capping at 5 avoids that while still being much faster than sequential."""
    indicator_keys = list(INDICATOR_CODES.keys())
    jobs = [(code, key) for code in countries for key in indicator_keys]

    def _run(job: tuple[str, str]) -> float | None:
        code, key = job
        return _fetch_latest_indicator_value(code, INDICATOR_CODES[key])

    with ThreadPoolExecutor(max_workers=min(len(jobs), 5) or 1) as pool:
        values = list(pool.map(_run, jobs)) if jobs else []

    values_by_country = {code: {} for code in countries}
    for (code, key), value in zip(jobs, values):
        values_by_country[code][key] = value

    return [
        CountryIndicators(
            country=COUNTRY_NAMES.get(code, code),
            gdp_growth_pct=values_by_country[code].get("gdp_growth_pct"),
            inflation_pct=values_by_country[code].get("inflation_pct"),
            unemployment_pct=values_by_country[code].get("unemployment_pct"),
        )
        for code in countries
    ]


def _table_label(table) -> str:
    # Neither table on this page has a <caption> — the identifying text
    # sits in a preceding sibling (KIBOR) or the parent's preceding
    # sibling (MTB/PIB, each wrapped in their own container div),
    # confirmed by inspecting the real page rather than guessed.
    prev = table.find_previous_sibling()
    if prev is not None:
        text = prev.get_text(" ", strip=True)
        if text:
            return text
    parent_prev = table.parent.find_previous_sibling() if table.parent else None
    return parent_prev.get_text(" ", strip=True) if parent_prev else ""


def _parse_yield_table(table) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in table.select("tbody tr"):
        cells = row.find_all("td")
        if len(cells) != 2:
            continue
        tenor = cells[0].get_text(strip=True)
        raw = cells[1].get_text(strip=True).replace("%", "")
        try:
            result[tenor] = float(raw)
        except ValueError:
            # e.g. "Bids Rejected" for a tenor with no successful auction,
            # or a trailing "(as on ...)" date row — real absences, not
            # parse bugs, so skipped rather than raising.
            continue
    return result


def fetch_pakistan_rates() -> PakistanRates | None:
    """Real KIBOR and government-bond (MTB/PIB) cut-off yields, scraped
    from the State Bank of Pakistan's own public rates page. None on any
    failure (network, or the page having none of the expected tables at
    all) — this is best-effort enrichment, not core data, same
    convention as data/mt5_source.py::get_contract_spec."""
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(SBP_RATES_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table")
    if not tables:
        return None

    as_of = None
    kibor: dict[str, float] = {}
    mtb: dict[str, float] = {}
    pib: dict[str, float] = {}

    for table in tables:
        label = _table_label(table).upper()
        if "KIBOR" in label:
            match = re.search(r"AS ON\s*(.+)", label, re.IGNORECASE)
            if match:
                as_of = match.group(1).strip().title()
            for row in table.select("tbody tr"):
                cells = row.find_all("td")
                if len(cells) != 3:
                    continue
                tenor = cells[0].get_text(strip=True)
                try:
                    bid = float(cells[1].get_text(strip=True))
                    offer = float(cells[2].get_text(strip=True))
                except ValueError:
                    continue
                kibor[tenor] = round((bid + offer) / 2, 4)
        elif label == "MTBS":
            mtb.update(_parse_yield_table(table))
        elif "FIXED" in label and "PIB" in label:
            pib.update(_parse_yield_table(table))

    if not kibor and not mtb and not pib:
        return None
    return PakistanRates(as_of=as_of, kibor_pct=kibor, mtb_yield_pct=mtb, pib_yield_pct=pib)
