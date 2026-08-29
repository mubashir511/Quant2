import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Safe to top-import, unlike yfinance/requests/bs4 below (deliberately
# kept lazy, imported only inside the specific functions that need
# them) — pandas is already a hard, ubiquitous dependency used
# unconditionally elsewhere in this codebase (ai/portfolio_suggest.py,
# analysis/technical.py, ai/ftmo_suggest.py's own correlation code).
import pandas as pd

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


def fetch_market_indicators(precomputed_closes: dict[str, "pd.Series | None"] | None = None) -> MarketIndicators:
    """US Treasury yield curve + dollar index + VIX. Any ticker that fails
    to fetch is left as None rather than failing the whole snapshot. The 5
    tickers are independent HTTP calls, fetched in parallel for speed.

    `precomputed_closes` (optional, added 2026-08-27 as part of the
    Phase 3 macro-cycle diagnostic below): a {ticker: daily closes} map
    for any of this function's own 5 tickers ALREADY fetched elsewhere
    this same call — found on audit: build_macro_snapshot() calls both
    this function AND fetch_macro_cycle_diagnostic(), and 3 of the 5
    tickers here (^IRX, ^TNX, DX-Y.NYB) are the exact same ones that
    function fetches a full daily-close HISTORY for anyway (needed for
    its own 12-month-MA calc) — a redundant second network round-trip
    for the same 3 tickers otherwise. A ticker present here skips this
    function's own fetch and reads its latest value from the given
    series instead; any ticker NOT present is still fetched fresh,
    exactly as before — omitting this parameter entirely (the default)
    preserves the original fully-independent-fetch behavior for any
    other caller/test."""
    precomputed_closes = precomputed_closes or {}
    tickers = ["^IRX", "^TNX", "^TYX", "DX-Y.NYB", "^VIX"]
    to_fetch = [t for t in tickers if t not in precomputed_closes]

    fetched: dict[str, float | None] = {}
    if to_fetch:
        with ThreadPoolExecutor(max_workers=len(to_fetch)) as pool:
            fetched = dict(zip(to_fetch, pool.map(_fetch_latest_price, to_fetch)))

    def _latest(ticker: str) -> float | None:
        if ticker in precomputed_closes:
            closes = precomputed_closes[ticker]
            return float(closes.iloc[-1]) if closes is not None and not closes.empty else None
        return fetched.get(ticker)

    yield_3m = _latest("^IRX")
    yield_10y = _latest("^TNX")
    yield_30y = _latest("^TYX")
    dxy = _latest("DX-Y.NYB")
    vix = _latest("^VIX")

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


# --- Pring's Market Cycle Model diagnostic (added 2026-08-27, "Phase 3"
# of the book-wisdom initiative, direct user request: "search the books
# for the perspective of geopolitics shocks and macro patterns (within
# the broker limitations)"). Pure yfinance-fetch + pandas arithmetic —
# deliberately ZERO calls to Claude, Copilot, or any LLM of any kind
# (direct user constraint: "i am not in favor of stressing AI model
# because i have limited claude and copilot usage quota... donot stress
# metered AI models"). Same category of code as fetch_market_indicators
# above and ai.ftmo_suggest.compute_ftmo_correlation_pairs (Phase 2) —
# real data, real math, no AI reasoning involved at all. See the
# sharpened Pring business-cycle BookPrinciple in data/book_wisdom.py
# for the "how to read this" companion text. ---

# ^IRX/^TNX and DX-Y.NYB are the SAME tickers fetch_market_indicators()
# already uses for spot values — this fetches their HISTORY instead.
# ^GSPC and DBC are new. DBC (Invesco DB Commodity Index Tracking Fund)
# substitutes for Pring's own "CRB Spot Raw Industrials" index — the
# literal CRB ticker is confirmed delisted/unavailable via yfinance; GSG
# (iShares S&P GSCI Commodity-Indexed Trust) is a confirmed-working
# backup if DBC ever proves unreliable in practice.
_CYCLE_TICKERS = {
    "yield_3m": "^IRX",
    "yield_10y": "^TNX",
    "equity_index": "^GSPC",
    "commodity_index": "DBC",
    "dollar_index": "DX-Y.NYB",
}
_CYCLE_HISTORY_PERIOD = "450d"  # ~14-15 months of daily bars -- enough buffer for a
                                 # trailing-12-month-vs-current comparison even
                                 # accounting for occasional missing/holiday bars


@dataclass
class MacroCycleLeg:
    label: str
    ticker: str
    current: float | None
    moving_avg_12m: float | None
    above_ma: bool | None  # None whenever current or moving_avg_12m is unavailable --
                            # never a guessed True/False


@dataclass
class MacroCycleDiagnostic:
    yield_3m: MacroCycleLeg
    yield_10y: MacroCycleLeg
    equity_index: MacroCycleLeg
    commodity_index: MacroCycleLeg
    dollar_index: MacroCycleLeg  # bonus context only -- NOT part of stage matching
    stage_label: str | None  # e.g. "Stage III" -- None if no clean match
    stage_description: str | None  # one-line description paired with stage_label
    note: str  # always populated: ambiguity reason, missing-data reason,
               # or Murphy's own forecasting-value caveat


def _fetch_daily_closes(ticker: str) -> pd.Series | None:
    """~14-15 months of daily closes for `ticker`. None (not an empty
    Series) on total fetch failure, mirroring _fetch_latest_price's own
    bare-except-return-None convention."""
    import yfinance as yf

    try:
        history = yf.Ticker(ticker).history(period=_CYCLE_HISTORY_PERIOD)
    except Exception:
        return None
    if history.empty or "Close" not in history:
        return None
    closes = history["Close"].dropna()
    return closes if not closes.empty else None


def _leg_vs_12m_ma(label: str, ticker: str, daily_closes: pd.Series | None) -> MacroCycleLeg:
    """Pure computation, no network — the current (latest available)
    close vs. the mean of the 12 calendar months immediately preceding
    the current one (resampled to one value per month, last close of
    each month). Needs >=13 distinct months of data (12 prior + the
    current one) or above_ma stays None rather than computed from a
    shorter, misleading window — same rule TechnicalStats already
    follows for its own windowed stats."""
    if daily_closes is None:
        return MacroCycleLeg(label, ticker, None, None, None)

    current = float(daily_closes.iloc[-1])
    monthly = daily_closes.resample("ME").last().dropna()
    if len(monthly) < 13:
        return MacroCycleLeg(label, ticker, current, None, None)

    trailing_12 = monthly.iloc[-13:-1]  # the 12 months BEFORE the current one
    ma = float(trailing_12.mean())
    return MacroCycleLeg(label, ticker, current, ma, current > ma)


# (bonds_up, stocks_up, commodities_up) -> (stage label, one-line description).
# Only 6 of the 8 possible combinations are canonical Pring stages -- the other
# 2 (diagonal opposites) are real "mixed/ambiguous" outcomes, not a bug.
_PRING_STAGE_TABLE: dict[tuple[bool, bool, bool], tuple[str, str]] = {
    (True, False, False): (
        "Stage I", "Bond bull begins in recession; stocks and commodities still falling.",
    ),
    (True, True, False): (
        "Stage II",
        "Bonds and stocks rising, equities looking through still-falling profits; "
        "commodities still falling.",
    ),
    (True, True, True): (
        "Stage III",
        "Recovery underway -- bonds, stocks, and commodities all rising; "
        "commodity prices bottoming.",
    ),
    (False, True, True): (
        "Stage IV",
        "Rates rising/bond bear begins, but the equity uptrend continues on "
        "improving productivity; commodities rising.",
    ),
    (False, False, True): (
        "Stage V", "Economy overheats; equities top out as the profit outlook sours; commodities still rising.",
    ),
    (False, False, False): (
        "Stage VI", "Slide into recession -- bonds, stocks, and commodities all falling.",
    ),
}

_FORECASTING_VALUE_CAVEAT = (
    "Descriptive read of Pring's bond/stock/commodity market-cycle model, "
    "not a precise or reliably-timed predictive signal — \"the leads and "
    "lags vary from cycle to cycle and have little forecasting value\" "
    "(Murphy's own caveat)."
)


def _match_pring_stage(
    bonds_up: bool | None, stocks_up: bool | None, commodities_up: bool | None
) -> tuple[str | None, str | None, str]:
    """Pure, independently testable — returns (stage_label,
    stage_description, note). `note` is always populated: a specific
    reason when a leg is missing/ambiguous, else the forecasting-value
    caveat on a clean match, else the generic mixed/ambiguous line."""
    if bonds_up is None:
        return None, None, "3-month/10-year yield data incomplete — bond leg of the cycle model unavailable right now."
    if stocks_up is None or commodities_up is None:
        return None, None, "Equity or commodity index data incomplete — cycle-stage match unavailable right now."

    match = _PRING_STAGE_TABLE.get((bonds_up, stocks_up, commodities_up))
    if match is None:
        return None, None, (
            "Signals don't cleanly match one of Pring's 6 canonical stage "
            "patterns — mixed/ambiguous read, no stage label assigned "
            "rather than forcing a guess."
        )
    stage_label, stage_description = match
    return stage_label, stage_description, _FORECASTING_VALUE_CAVEAT


def fetch_macro_cycle_closes() -> dict[str, pd.Series | None]:
    """Fetches ~14-15 months of daily closes for the 5 cycle-model
    tickers in parallel, keyed by the same keys as _CYCLE_TICKERS
    (e.g. "yield_3m" -> "^IRX"'s closes). Split out from
    fetch_macro_cycle_diagnostic (added on audit, 2026-08-27) so a
    caller that also needs spot values for 3 of these same tickers
    (fetch_market_indicators's ^IRX/^TNX/DX-Y.NYB) can fetch each
    ticker exactly once and share the result, instead of that function
    independently re-fetching a fresh 5-day history for tickers this
    function already fetched 450 days of."""
    tickers = list(_CYCLE_TICKERS.values())
    with ThreadPoolExecutor(max_workers=len(tickers)) as pool:
        return dict(zip(_CYCLE_TICKERS.keys(), pool.map(_fetch_daily_closes, tickers)))


def fetch_macro_cycle_diagnostic(closes_by_key: dict[str, pd.Series | None] | None = None) -> MacroCycleDiagnostic:
    """Pure deterministic computation: reads each of 5 tickers' current
    value against its own trailing 12-month moving average — mirroring
    Pring's own quoted diagnostic method exactly. NEVER calls Claude,
    Copilot, OpenRouter, or any AI model — this is yfinance fetch +
    pandas arithmetic only, same category as fetch_market_indicators
    and ai.ftmo_suggest.compute_ftmo_correlation_pairs.

    `closes_by_key` (optional): pre-fetched closes from
    fetch_macro_cycle_closes(), keyed the same way. When omitted
    (the default — preserves every existing caller/test unchanged),
    fetches them fresh here.

    The 3M/10Y yield legs must AGREE (both above or both below their own
    12-month MA) before a single 'bonds' state is derived — a
    disagreement is a real yield-curve-twist scenario, reported as
    mixed/ambiguous rather than picking one arbitrarily. The dollar
    index leg is fetched and returned for bonus context only; it is NOT
    part of Pring's 3-market (bonds/stocks/commodities) stage model."""
    if closes_by_key is None:
        closes_by_key = fetch_macro_cycle_closes()

    yield_3m = _leg_vs_12m_ma("3-month T-bill yield", "^IRX", closes_by_key["yield_3m"])
    yield_10y = _leg_vs_12m_ma("10-year Treasury yield", "^TNX", closes_by_key["yield_10y"])
    equity_index = _leg_vs_12m_ma("S&P 500", "^GSPC", closes_by_key["equity_index"])
    commodity_index = _leg_vs_12m_ma("Commodity index (DBC proxy)", "DBC", closes_by_key["commodity_index"])
    dollar_index = _leg_vs_12m_ma("US Dollar Index (DXY)", "DX-Y.NYB", closes_by_key["dollar_index"])

    curve_twist = (
        yield_3m.above_ma is not None
        and yield_10y.above_ma is not None
        and yield_3m.above_ma != yield_10y.above_ma
    )
    if curve_twist:
        stage_label, stage_description = None, None
        note = (
            "3-month and 10-year yields disagree on direction relative to "
            "their own 12-month averages (a yield-curve twist) — no single "
            "clean 'bonds' reading, so no stage match attempted."
        )
    else:
        if yield_3m.above_ma is None or yield_10y.above_ma is None:
            bonds_up = None
        else:
            bonds_up = not yield_3m.above_ma  # yield ABOVE its own MA == rising yield == bond BEAR
        stage_label, stage_description, note = _match_pring_stage(
            bonds_up, equity_index.above_ma, commodity_index.above_ma
        )

    return MacroCycleDiagnostic(
        yield_3m=yield_3m,
        yield_10y=yield_10y,
        equity_index=equity_index,
        commodity_index=commodity_index,
        dollar_index=dollar_index,
        stage_label=stage_label,
        stage_description=stage_description,
        note=note,
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
