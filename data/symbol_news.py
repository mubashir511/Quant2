"""The ONE shared per-symbol news fetch, cache, and history — added
2026-09-20 direct user request, so Clerk, Researcher, Mega Session, and
the webapp's own News section all read the same real, live news instead
of each independently hitting Yahoo/Google/RSS for the same symbol
("save double calling of news").

Real bug found and fixed by building this: Clerk's own prior news fetch
used data.underlying.resolve_yahoo_ticker (a PMEX-symbol keyword map),
not ai.researcher's own already-correct, FTMO-native resolver — confirmed
live against this account's real symbol mix, 17 of 22 symbols (77%)
silently returned no Yahoo ticker at all under the old resolver. The
logic here (resolve_ftmo_yahoo_ticker, the relevance filter, the
category RSS feeds) is MOVED from ai/researcher.py, which built and
proved it first — ai/researcher.py's own private functions of the same
purpose are now thin delegating wrappers to here (same names/signatures,
zero behavior change, so none of that module's own ~85 existing tests
needed to change), so this file is the single canonical implementation,
not a second copy of it.

Caching is keyed by the real Yahoo ticker (not the FTMO symbol) at the
`fetch_news_with_fallback` layer — the layer that actually performs a
real network call — so Clerk resolving "XAUUSD" and Researcher resolving
the same "XAUUSD" both land on the exact same cache entry keyed by
"XAUUSD=X", regardless of which one asks first. TTL is config.
SYMBOL_NEWS_CACHE_MINUTES (20 by default, same value already accepted
for Clerk's own prior cache).

Real gap found 2026-09-20, direct user challenge ("i don't want to
explode the usage quota... neither do i want any analysis to go without
the news context"): Clerk/Researcher/Mega/Trade Audit each run as their
OWN separate spawned process per tray-timer tick (quant_app_tray.ps1's
fire-and-forget Process.Start() per job), so the in-memory cache dicts
below only ever deduped calls made within ONE process's lifetime, never
across those 4 real consumers -- each fresh process started with an
empty cache and would have refetched independently regardless of how
recently another process had just fetched the same thing. Fixed with a
small disk-backed mirror (config.NEWS_FETCH_CACHE_FILE, see
_disk_cache_get/_disk_cache_put) that every cache check also consults
and every fresh fetch also writes to, so a real fetch made by any one
process is visible to every other process for the rest of the same TTL
window. Same TTL, same "never serve stale-past-TTL data" contract --
this only closes the cross-process blind spot, it doesn't loosen it.

Vault persistence (config.SYMBOL_NEWS_DIR, one JSON file per FTMO
symbol — the source of truth — rendered into a matching obsidian_vault/
News/{symbol}.md note) happens on every genuinely FRESH fetch (a cache
miss that returned real items), deduplicated by title, with a rolling
config.SYMBOL_NEWS_RETENTION_DAYS-day window pruned on every write —
direct user request, "save the news with sequence and timestamps... for
one month... after one month the vault will be reset" (a continuously
self-pruning rolling window, not an abrupt calendar-date wipe — user's
own explicit choice when asked).

Direct user request 2026-09-20, after finding the news genuinely too
thin ("headlines only... i was expecting few paragraphs that completely
explain the event like the newspaper") then asking for the webapp to
stay simple regardless ("keep the ui simple like before but in the
obsidian vault add complete version of the news"): the webapp's own
table still shows only title/source, but every genuinely new headline
persisted to the vault also gets its real full article text fetched
(fetch_article_text — a real page fetch + paragraph extraction, never
model-generated) and written into that headline's own vault entry, so
the "complete version" lives in the vault note, not in a UI that has to
stay fast to open."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
from data.news_source import fetch_article_text, fetch_google_news, fetch_recent_news, fetch_rss_feed

logger = logging.getLogger(__name__)

# Real, well-known keyword sets per major currency/metal — for the
# per-symbol news relevance filter below. Moved verbatim from
# ai/researcher.py (added there 2026-09-15/16 after a real, root-caused
# defect: Yahoo's own ticker-news search for "GBPUSD=X" returned two
# genuinely irrelevant cocoa/Ghana commodity articles among its top 5
# results — not a model hallucination, a real upstream data-quality gap).
_CURRENCY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "USD": ("usd", "dollar", "fed", "u.s.", "greenback", "dxy"),
    "EUR": ("eur", "euro", "ecb", "eurozone"),
    "GBP": ("gbp", "pound", "sterling", "boe", "uk", "britain", "british"),
    "JPY": ("jpy", "yen", "boj", "japan"),
    "CHF": ("chf", "franc", "snb", "swiss", "switzerland"),
    "CAD": ("cad", "loonie", "canada", "boc"),
    "AUD": ("aud", "aussie", "australia", "rba"),
    "NZD": ("nzd", "kiwi", "zealand", "rbnz"),
    "CNH": ("cnh", "cny", "yuan", "renminbi", "china", "pboc"),
    "SEK": ("sek", "krona", "sweden", "riksbank"),
    "XAU": ("gold", "xau"),
    "XAG": ("silver", "xag"),
    "BTC": ("btc", "bitcoin"),
    "ETH": ("eth", "ethereum"),
}

# Real, free, no-key RSS feeds genuinely dedicated to one asset class —
# moved verbatim from ai/researcher.py. No equities-specific feed: no
# free, no-key, genuinely equities-only real feed was found working live.
_CATEGORY_RSS_FEEDS: dict[str, str] = {
    "Forex": "https://www.fxstreet.com/rss/news",
    "Exotics": "https://www.fxstreet.com/rss/news",
    "Crypto": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Metals": "https://www.investing.com/rss/commodities.rss",
}

# Real Yahoo Finance FUTURES ticker per FTMO symbol — added 2026-09-20,
# direct user challenge after being told (wrongly, without verifying
# live first) that these categories had "no known free news-source
# mapping." Checked live: they do. Unlike forex ("=X" suffix) or crypto
# ("-USD" suffix), commodity futures have no generic naming convention
# derivable from the FTMO symbol alone (there's no rule that gets you
# from "COFFEE.c" to "KC=F"), so this is an explicit lookup, same
# pattern as _MACRO_PROXY_TICKERS above. Confirmed live, each returns
# real, current, genuinely relevant headlines via fetch_recent_news.
_COMMODITY_FUTURES_TICKERS: dict[str, str] = {
    "COFFEE.c": "KC=F",
    "COCOA.c": "CC=F",
    "WHEAT.c": "ZW=F",
    "UKOIL.cash": "BZ=F",
}

# Keyed by real Yahoo ticker (not FTMO symbol) — see this module's own
# top-of-file docstring for why. Module-level and unbounded: the real
# key set is bounded by however many distinct tickers this account's
# Market Watch ever resolves, never unbounded growth.
# (fetched_at, items, limit_fetched_at) — the limit is remembered so a
# narrow request can never silently cap what a broader one gets back
# from the same cache entry (see fetch_news_with_fallback's own docstring).
#
# This in-memory dict is only a same-process fast path. Clerk/Researcher/
# Mega/Trade Audit each run as their own separate spawned process per
# tray-timer tick, so it resets empty on every invocation — the disk
# mirror below (_disk_cache_get/_disk_cache_put, config.NEWS_FETCH_
# CACHE_FILE) is what actually dedupes real fetches ACROSS those 4
# consumers, not this dict.
_symbol_news_cache: dict[str, tuple[datetime, list[dict], int]] = {}


def _load_disk_cache() -> dict:
    """Best-effort read of the shared cross-process cache file. Any
    problem (missing file, corrupt JSON, race with another process mid-
    write) degrades to "no disk cache" rather than raising — a cache is
    never allowed to block a real news fetch."""
    try:
        with open(config.NEWS_FETCH_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return {}


def _save_disk_cache(data: dict) -> None:
    """Best-effort write, via a temp-file-then-replace so a crash or a
    second writer mid-write can never leave a half-written, corrupt cache
    file behind. Never raises — a failed cache write should never take
    down a job that already has its real, freshly-fetched news items."""
    try:
        path = Path(config.NEWS_FETCH_CACHE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except OSError:
        logger.warning("Failed to write shared news fetch cache — continuing without it.", exc_info=True)


def _disk_cache_get(section: str, key: str) -> tuple[datetime, list[dict], int | None] | None:
    """Reads one (fetched_at, items, limit) entry from the shared disk
    cache — `limit` is None for the category/macro sections, which don't
    have the narrower/broader-request distinction the per-symbol layer
    does. Any malformed entry (partial write, manual edit) is treated as
    a miss rather than raised."""
    entry = _load_disk_cache().get(section, {}).get(key)
    if not isinstance(entry, dict):
        return None
    try:
        return datetime.fromisoformat(entry["fetched_at"]), entry["items"], entry.get("limit")
    except (KeyError, ValueError, TypeError):
        return None


def _disk_cache_put(section: str, key: str, now: datetime, items: list[dict], limit: int | None = None) -> None:
    data = _load_disk_cache()
    data.setdefault(section, {})[key] = {"fetched_at": now.isoformat(), "items": items, "limit": limit}
    _save_disk_cache(data)


def resolve_ftmo_yahoo_ticker(symbol: str, category: str) -> str | None:
    """Real, well-known Yahoo Finance ticker conventions for THIS
    account's own symbol naming — moved verbatim from ai.researcher.
    _resolve_ftmo_yahoo_ticker (see its own original docstring for the
    full "why not data.underlying.resolve_yahoo_ticker" reasoning):
      - A hardcoded commodity future (_COMMODITY_FUTURES_TICKERS) -> its
        real Yahoo futures ticker (checked first — no generic symbol-
        derivable convention exists for these, unlike the rules below).
      - Forex/Exotics/Metals CFD -> "<SYMBOL>=X".
      - Crypto (symbol ending "USD") -> "<BASE>-USD".
      - Equities -> the bare symbol.
    Everything else returns None rather than guessing."""
    if symbol in _COMMODITY_FUTURES_TICKERS:
        return _COMMODITY_FUTURES_TICKERS[symbol]
    if category in ("Forex", "Exotics") or category.startswith("Metals"):
        return f"{symbol}=X"
    if category.startswith("Crypto") and symbol.endswith("USD"):
        return f"{symbol[:-3]}-USD"
    if category.startswith("Equities"):
        return symbol
    return None


# Real keyword sets for the 4 hardcoded commodity futures above — same
# "checked live, not guessed" standard as _CURRENCY_KEYWORDS.
_COMMODITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "COFFEE.c": ("coffee", "arabica", "robusta"),
    "COCOA.c": ("cocoa",),
    "WHEAT.c": ("wheat", "grain"),
    "UKOIL.cash": ("brent", "oil", "crude"),
}


def symbol_relevance_keywords(symbol: str, category: str) -> list[str]:
    """Moved verbatim from ai.researcher._symbol_relevance_keywords,
    extended 2026-09-20 with real keywords for the 4 hardcoded commodity
    futures (_COMMODITY_KEYWORDS) — same fallback contract as everything
    else here."""
    if symbol in _COMMODITY_KEYWORDS:
        return list(_COMMODITY_KEYWORDS[symbol])
    if category in ("Forex", "Exotics") or category.startswith("Metals"):
        base, quote = symbol[:3], symbol[3:]
        keywords = list(_CURRENCY_KEYWORDS.get(base, ())) + list(_CURRENCY_KEYWORDS.get(quote, ()))
        return keywords or [symbol]
    if category.startswith("Crypto") and symbol.endswith("USD"):
        base = symbol[:-3]
        keywords = list(_CURRENCY_KEYWORDS.get(base, ())) + list(_CURRENCY_KEYWORDS.get("USD", ()))
        return keywords or [symbol]
    return [symbol]


def filter_relevant_news(news_items: list[dict], keywords: list[str]) -> list[dict]:
    """Moved verbatim from ai.researcher._filter_relevant_news. Falls
    back to the ORIGINAL, unfiltered list if filtering would remove
    every item — an empty result almost certainly means the keyword set
    is incomplete, not that every fetched item is genuinely irrelevant."""
    if not keywords:
        return news_items
    lowered_keywords = [k.lower() for k in keywords]
    filtered = [
        item
        for item in news_items
        if any(k in f"{item.get('title', '')} {item.get('summary', '')}".lower() for k in lowered_keywords)
    ]
    return filtered if filtered else news_items


def google_news_query(symbol: str, category: str) -> str:
    """Moved verbatim from ai.researcher._google_news_query, extended
    2026-09-20 with real human-readable queries for the 4 hardcoded
    commodity futures — the bare FTMO symbol (e.g. "COFFEE.c") is a poor
    Google News search term, unlike a real ticker or a plain word."""
    if symbol in _COMMODITY_FUTURES_TICKERS:
        return {
            "COFFEE.c": "coffee futures price",
            "COCOA.c": "cocoa futures price",
            "WHEAT.c": "wheat futures price",
            "UKOIL.cash": "brent crude oil price",
        }[symbol]
    if category in ("Forex", "Exotics"):
        return f"{symbol} forex"
    if category.startswith("Metals"):
        return f"{symbol} price"
    if category.startswith("Crypto") and symbol.endswith("USD"):
        return f"{symbol[:-3]} crypto"
    if category.startswith("Equities"):
        return f"{symbol} stock"
    return symbol


def category_rss_feed_url(category: str) -> str | None:
    """Moved verbatim from ai.researcher._category_rss_feed_url,
    extended 2026-09-20: Agriculture and Cash CFD (UKOIL.cash's own real
    category) both route to the same real, free Investing.com
    commodities feed Metals CFD already uses — confirmed live that feed
    genuinely covers commodities broadly, not just metals."""
    if category in _CATEGORY_RSS_FEEDS:
        return _CATEGORY_RSS_FEEDS[category]
    if category.startswith("Crypto"):
        return _CATEGORY_RSS_FEEDS["Crypto"]
    if category.startswith("Metals") or category == "Agriculture" or category.startswith("Cash"):
        return _CATEGORY_RSS_FEEDS["Metals"]
    return None


def _parse_published(published: str) -> datetime | None:
    """Moved verbatim from ai.researcher._parse_published — real
    published-date parsing across the three genuinely different formats
    this project's own news sources actually return: Yahoo's own ISO
    8601 with a trailing "Z", and RFC 822 from both Google News and
    every plain RSS feed. None (never guessed/fabricated) on anything
    neither parser recognizes."""
    if not published:
        return None
    try:
        return datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(published)
    except (ValueError, TypeError):
        return None


def _relative_age(published: str, now_utc: datetime) -> str:
    """Moved verbatim from ai.researcher._relative_age. "" (never a
    fabricated age) when the published string is missing/unparseable,
    or is somehow in the future (clock skew between this machine and a
    feed's own server) — safer to omit than print a negative age."""
    dt = _parse_published(published)
    if dt is None:
        return ""
    seconds = (now_utc - dt).total_seconds()
    if seconds < 0:
        return ""
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


BREAKING_NEWS_MAX_AGE_MINUTES = 90


def headline_with_age(item: dict, now_utc: datetime | None = None) -> str:
    """One headline as a single line for prompt use: the bare title when
    its publish time is unknown (never a fabricated age), otherwise
    "title (source, 25m ago)" — prefixed "[BREAKING] " when newer than
    BREAKING_NEWS_MAX_AGE_MINUTES, since on M5 decisions a headline
    from the last hour matters far more than one from yesterday."""
    now_utc = datetime.now(timezone.utc) if now_utc is None else now_utc
    title = item["title"]
    try:
        published = _parse_published(item.get("published", ""))
        if published is not None and published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)  # a feed stamp with no offset is UTC
        age = _relative_age(published.isoformat(), now_utc) if published is not None else ""
        if published is None or not age:
            return title
        seconds = (now_utc - published).total_seconds()
    except (TypeError, ValueError, AttributeError):
        # One malformed timestamp must never cost a symbol ALL of its headlines
        # (the caller's broad except would otherwise blank the whole list).
        return title
    bits = ", ".join(b for b in (item.get("source"), age) if b)
    prefix = "[BREAKING] " if seconds < BREAKING_NEWS_MAX_AGE_MINUTES * 60 else ""
    return f"{prefix}{title} ({bits})"


def format_news_block(news_items: list[dict], now_utc: datetime | None = None) -> str:
    """Moved verbatim from ai.researcher._format_news_block."""
    if not news_items:
        return "(none)"
    now_utc = datetime.now(timezone.utc) if now_utc is None else now_utc
    lines = []
    for item in news_items:
        age = _relative_age(item.get("published", ""), now_utc)
        bits = [b for b in (item.get("source"), age) if b]
        suffix = f" — {', '.join(bits)}" if bits else ""
        lines.append(f"- {item['title']}{suffix}")
        if item.get("summary"):
            lines.append(f"  {item['summary']}")
    return "\n".join(lines)


def fetch_news_with_fallback(
    yahoo_ticker: str, google_query: str, limit: int, symbol: str | None = None
) -> list[dict]:
    """Yahoo Finance first, Google News RSS as a fallback when Yahoo
    genuinely has nothing — moved from ai.researcher._fetch_news_with_
    fallback, now with a shared cache (config.SYMBOL_NEWS_CACHE_MINUTES,
    keyed by `yahoo_ticker` so every consumer resolving the same real
    ticker shares the same cache entry) and optional vault persistence.

    Real bug caught live 2026-09-20, fixed before it ever caused a
    silent under-serving: the cache entry also remembers the `limit` it
    was fetched at. A cache HIT only reuses the stored items when the
    stored limit is >= what's being asked for now (sliced down to
    exactly `limit`) — a narrower caller (Clerk, limit=2) fetching first
    can no longer silently cap what a broader caller (Mega Session/the
    webapp, limit=5) gets back within the same cache window; that
    broader request instead does its own real fetch and widens the
    shared cache entry for everyone after it.

    `symbol` is the ORIGINAL FTMO symbol (e.g. "XAUUSD"), OPTIONAL and
    keyword-only, purely so this function's own two pre-existing direct-
    call tests (which don't pass it) keep working unchanged — when given,
    a genuinely fresh (cache-miss) non-empty fetch also gets persisted to
    that symbol's own vault history. None (the default) simply skips
    persistence, since there's no symbol to file it under."""
    now = datetime.now(timezone.utc)
    cached = _symbol_news_cache.get(yahoo_ticker) or _disk_cache_get("symbol", yahoo_ticker)
    if cached is not None:
        cached_time, cached_items, cached_limit = cached
        if (now - cached_time) < timedelta(minutes=config.SYMBOL_NEWS_CACHE_MINUTES) and cached_limit >= limit:
            _symbol_news_cache[yahoo_ticker] = cached
            return cached_items[:limit]
    items = fetch_recent_news(yahoo_ticker, limit=limit)
    if not items:
        items = fetch_google_news(google_query, limit=limit)
    _symbol_news_cache[yahoo_ticker] = (now, items, limit)
    _disk_cache_put("symbol", yahoo_ticker, now, items, limit)
    if symbol and items:
        _record_symbol_news_to_vault(symbol, items)
    return items


_BORING_DESCRIPTION_WORDS = frozenset({"cfd", "spot", "cash", "vs", "us", "dollar", "index", "i", "ii", "iii"})


def derive_generic_search_name(description: str) -> str:
    """A clean, human-readable search phrase from MT5's own real symbol
    description — e.g. "Coffee vs US Dollar, Spot CFD" -> "Coffee",
    "Crude Oil Brent, Spot CFD" -> "Crude Oil Brent". Added 2026-09-20,
    direct user challenge ("how is it possible the 3 food items and oil
    does not have news feed, you have to do intelligent news
    searching"): rather than only ever recognizing a symbol via a
    hardcoded per-symbol ticker table or a known category convention —
    both of which silently stop working for the NEXT symbol added to the
    mix in a category nobody has mapped yet — MT5 already hands every
    symbol a real, human-readable description regardless of category.
    This is what lets get_symbol_news_block fall back to a genuine
    Google News search instead of simply giving up. Drops the generic
    "CFD"/"Spot"/"vs US Dollar" trading-type boilerplate every FTMO
    description carries, which would otherwise pollute the search query
    with words no real news article uses. "" (never a fabricated guess)
    when nothing meaningful is left after stripping that boilerplate."""
    if not description:
        return ""
    name = description.split(",")[0]
    words = [w for w in name.split() if w.strip(".").lower() not in _BORING_DESCRIPTION_WORDS]
    return " ".join(words).strip()


def get_symbol_news_block(symbol: str, category: str, limit: int, description: str = "") -> list[dict]:
    """One-call convenience for a NEW consumer (Clerk, Mega Session, the
    webapp's News section) that doesn't need Researcher's own more
    granular step-by-step control: resolve -> cached fetch-with-fallback
    (with vault persistence) -> relevance filter. [] (never fabricated)
    when no ticker resolves for this symbol/category or the real fetch
    comes back empty on both sources.

    `description` (MT5's own real per-symbol description, optional —
    every existing caller keeps working unchanged without it) is the
    "intelligent search" fallback added 2026-09-20: when this symbol's
    category matches no known ticker convention AND isn't one of the
    hardcoded commodity futures, this derives a genuine, human-readable
    search phrase from it (derive_generic_search_name) and searches
    Google News directly with THAT, instead of giving up. Reuses the
    exact same cache/backfill/vault-persistence pipeline as the precise
    path (fetch_news_with_fallback) via a synthetic, unambiguous cache
    key — fetch_recent_news degrades safely (empty, not an error) for
    a key that isn't a real Yahoo ticker, so this always falls straight
    through to the real Google search on the derived phrase."""
    yahoo_ticker = resolve_ftmo_yahoo_ticker(symbol, category)
    if yahoo_ticker is None:
        query = derive_generic_search_name(description)
        if not query:
            return []
        items = fetch_news_with_fallback(f"__generic__:{symbol}", query, limit, symbol=symbol)
        if not items:
            return []
        keywords = [w.lower() for w in query.split()]
        return filter_relevant_news(items, keywords)
    google_query = google_news_query(symbol, category)
    items = fetch_news_with_fallback(yahoo_ticker, google_query, limit, symbol=symbol)
    if not items:
        return []
    return filter_relevant_news(items, symbol_relevance_keywords(symbol, category))


# --- Category-specialty + macro/geopolitical news — MOVED here 2026-09-20
# alongside everything above. Real gap this closes, direct user
# challenge: "news quantity and sources are very less... where those all
# sources gone?" — the initial consolidation only moved the PER-SYMBOL
# Yahoo/Google layer; Researcher's own richer category-RSS (FXStreet/
# CoinDesk/Investing.com) and macro/geopolitical (S&P 500 + crude oil +
# gold proxies, plus CNBC's own top-news feed) layers stayed private to
# ai/researcher.py and were never available to Clerk, Mega Session, or
# the webapp's News section — a real completeness regression versus what
# Researcher's own daily report already had, not a deliberate scope
# decision. Both now cached here (same TTL convention as the per-symbol
# layer) so Mega Session/the webapp reading these after Researcher's own
# daily run already populated them costs nothing extra either. ---------

_MACRO_PROXY_TICKERS = {"^GSPC": "S&P 500", "CL=F": "crude oil", "GC=F": "gold price"}
_CNBC_TOP_NEWS_URL = "https://www.cnbc.com/id/100003114/device/rss/rss.html"

# Keyed by category name; macro uses the fixed key "__macro__" (there's
# only ever one, account-wide, shared across every symbol).
_category_news_cache: dict[str, tuple[datetime, list[dict]]] = {}
_macro_news_cache: dict[str, tuple[datetime, list[dict]]] = {}
_MACRO_CACHE_KEY = "__macro__"


def get_category_news_items(category: str, limit: int) -> list[dict]:
    """Real, additional analyst-grade coverage from a feed genuinely
    dedicated to `category` (FXStreet/CoinDesk/Investing.com) — cached
    per category name for config.SYMBOL_NEWS_CACHE_MINUTES. [] (not a
    fabricated placeholder) when no such feed exists for this category
    (equities, or anything else category_rss_feed_url doesn't cover) —
    callers wanting the old human-readable "(no category-specialty feed
    for ...)" text should call format_news_block themselves, or use the
    dedicated string-returning helper in ai.researcher for that exact
    legacy wording.

    A genuinely fresh (cache-miss, non-empty) fetch also gets persisted
    to this category's own vault history (_record_category_news_to_vault)
    — added 2026-09-20, direct user report that category news had no
    vault presence at all before this, same rolling-window/full-article
    treatment the per-symbol layer already gets."""
    url = category_rss_feed_url(category)
    if url is None:
        return []
    now = datetime.now(timezone.utc)
    cached = _category_news_cache.get(category) or _disk_cache_get("category", category)
    if cached is not None and (now - cached[0]) < timedelta(minutes=config.SYMBOL_NEWS_CACHE_MINUTES):
        _category_news_cache[category] = (cached[0], cached[1])
        return cached[1][:limit]
    items = fetch_rss_feed(url, limit=limit)
    _category_news_cache[category] = (now, items)
    _disk_cache_put("category", category, now, items)
    if items:
        _record_category_news_to_vault(category, items)
    return items


def get_macro_news_items() -> list[dict]:
    """Real, market-wide/geopolitical-proxy headlines (S&P 500/crude
    oil/gold proxies) plus CNBC's own real top-news RSS feed — genuinely
    account-wide, not per-symbol or per-category, so this has exactly
    ONE shared cache entry regardless of who asks or for which symbol.
    Each item's `title` is prefixed with a readable source label (e.g.
    "[S&P 500] ...", "[CNBC] ...") so a caller flattening this into a
    plain list can still tell a macro item apart from per-symbol/
    category news without needing a separate field for it."""
    now = datetime.now(timezone.utc)
    cached = _macro_news_cache.get(_MACRO_CACHE_KEY) or _disk_cache_get("macro", _MACRO_CACHE_KEY)
    if cached is not None and (now - cached[0]) < timedelta(minutes=config.SYMBOL_NEWS_CACHE_MINUTES):
        _macro_news_cache[_MACRO_CACHE_KEY] = (cached[0], cached[1])
        return cached[1]
    items: list[dict] = []
    for ticker, label in _MACRO_PROXY_TICKERS.items():
        for item in fetch_news_with_fallback(ticker, label, limit=3):
            items.append({**item, "title": f"[{label}] {item['title']}"})
    for item in fetch_rss_feed(_CNBC_TOP_NEWS_URL, limit=5):
        items.append({**item, "title": f"[CNBC] {item['title']}"})
    _macro_news_cache[_MACRO_CACHE_KEY] = (now, items)
    _disk_cache_put("macro", _MACRO_CACHE_KEY, now, items)
    return items


# --- Obsidian vault history (JSON = source of truth; the .md note is a
# deterministic render of it, regenerated on every write) -------------


def _safe_filename(raw: str) -> str:
    """A category name (e.g. "Metals CFD") turned into a safe filename —
    only word characters/spaces/hyphens are known-safe across Windows
    and POSIX; anything else (a real category name has never contained
    one, but this is cheap insurance) becomes an underscore."""
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in raw).strip() or "unknown"


def _news_store_path(symbol: str) -> Path:
    return Path(config.SYMBOL_NEWS_DIR) / f"{symbol}.json"


def _category_news_store_path(category: str) -> Path:
    return Path(config.CATEGORY_NEWS_DIR) / f"{_safe_filename(category)}.json"


def _load_news_store(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _save_news_store(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _load_symbol_news_store(symbol: str) -> list[dict]:
    return _load_news_store(_news_store_path(symbol))


def _save_symbol_news_store(symbol: str, entries: list[dict]) -> None:
    _save_news_store(_news_store_path(symbol), entries)


def _render_news_note_lines(header: str, entries: list[dict], link_targets: list[str]) -> list[str]:
    """Shared markdown body for both the per-symbol and category news
    vault notes. Includes the REAL full article text (fetch_full_
    article_text — a real page fetch + paragraph extraction, never
    model-generated) whenever an entry has one, since the webapp itself
    deliberately only shows title/source (direct user request 2026-09-20:
    "keep the ui simple like before but in the obsidian vault add
    complete version of the news")."""
    lines = ["---", "tags: [news]", "---", "", header, ""]
    for entry in sorted(entries, key=lambda e: e["seq"], reverse=True):
        suffix = f" ({entry['source']})" if entry.get("source") else ""
        lines.append(f"**#{entry['seq']}** — {entry['recorded_utc']}")
        title_line = f"[{entry['title']}]({entry['link']}){suffix}" if entry.get("link") else f"{entry['title']}{suffix}"
        lines.append(f"- {title_line}")
        if entry.get("summary"):
            lines.append(f"  {entry['summary']}")
        if entry.get("full_text"):
            lines.append("")
            lines.append("  **Full article:**")
            for paragraph in entry["full_text"].split("\n\n"):
                if paragraph.strip():
                    lines.append(f"  {paragraph.strip()}")
                    lines.append("")
        lines.append("")
    lines += [f"[[{target}]]" for target in link_targets]
    return lines


def _write_symbol_news_note(symbol: str, entries: list[dict]) -> None:
    try:
        vault_dir = Path(config.OBSIDIAN_VAULT_PATH) / "News"
        vault_dir.mkdir(parents=True, exist_ok=True)
        header = f"# {symbol} — News (rolling {config.SYMBOL_NEWS_RETENTION_DAYS}-day window)"
        lines = _render_news_note_lines(header, entries, [symbol, "News"])
        path = vault_dir / f"{symbol}.md"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text("\n".join(lines), encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError:
        logger.warning("symbol_news: could not write vault note for %s (cosmetic only).", symbol, exc_info=True)


def _write_category_news_note(category: str, entries: list[dict]) -> None:
    try:
        vault_dir = Path(config.OBSIDIAN_VAULT_PATH) / "News" / "Category"
        vault_dir.mkdir(parents=True, exist_ok=True)
        header = f"# {category} — Category News (rolling {config.SYMBOL_NEWS_RETENTION_DAYS}-day window)"
        lines = _render_news_note_lines(header, entries, [category, "News"])
        path = vault_dir / f"{_safe_filename(category)}.md"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text("\n".join(lines), encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError:
        logger.warning("symbol_news: could not write category vault note for %s (cosmetic only).", category, exc_info=True)


def _record_news_entries(entries: list[dict], items: list[dict], now: datetime) -> bool:
    """Shared append/backfill logic for both the per-symbol and category
    news stores. Mutates `entries` in place; returns whether anything
    changed (a new item appended, or an old-schema entry backfilled).

    Direct user request 2026-09-20 ("keep the ui simple like before but
    in the obsidian vault add complete version of the news"): the webapp
    only ever shows title/source, so the REAL full article text (see
    fetch_full_article_text) is fetched here, once, for each genuinely
    NEW headline. Deliberately only for new titles on first sight — never
    re-fetched for an already-recorded one — so this stays a rare,
    bounded cost, not a full-text fetch on every single poll cycle.

    Real gap found live 2026-09-20: an entry recorded BEFORE this full-
    text feature existed has no "link"/"full_text" key at all, and the
    plain title-dedup below would otherwise silently skip it forever —
    the exact same headline could keep reappearing in every fresh fetch
    without ever being upgraded. Fixed with an opportunistic backfill:
    an existing entry missing a link gets one (and its full text) filled
    in the next time that same headline is seen in a real fetch, without
    needing a forced, eager re-fetch of every already-recorded entry on
    every write (which would defeat the whole "bounded cost" point)."""
    existing_by_title = {e["title"]: e for e in entries}
    next_seq = max((e["seq"] for e in entries), default=0) + 1
    changed = False
    for item in items:
        title = item.get("title")
        if not title:
            continue
        if title in existing_by_title:
            existing_entry = existing_by_title[title]
            link = item.get("link", "")
            if not existing_entry.get("link") and link:
                existing_entry["link"] = link
                existing_entry["full_text"] = fetch_full_article_text(link)
                changed = True
            continue
        link = item.get("link", "")
        new_entry = {
            "seq": next_seq,
            "recorded_utc": now.isoformat(),
            "title": title,
            "source": item.get("source", ""),
            "published": item.get("published", ""),
            "summary": item.get("summary", ""),
            "link": link,
            "full_text": fetch_full_article_text(link),
        }
        entries.append(new_entry)
        existing_by_title[title] = new_entry
        next_seq += 1
        changed = True
    return changed


def _record_symbol_news_to_vault(symbol: str, items: list[dict]) -> None:
    """Best-effort side effect, same "cosmetic, must never affect real
    logic" contract as ai.trade_journal's own vault writers — appends
    genuinely new items (deduplicated by title, see _record_news_entries
    for the full contract including backfill) with a running sequence
    number and this recording's own real timestamp, then prunes anything
    older than config.SYMBOL_NEWS_RETENTION_DAYS on every call, so the
    store is always a rolling window, never grows unbounded and never
    needs a separate scheduled reset job."""
    try:
        now = datetime.now(timezone.utc)
        entries = _load_symbol_news_store(symbol)
        changed = _record_news_entries(entries, items, now)

        cutoff = now - timedelta(days=config.SYMBOL_NEWS_RETENTION_DAYS)
        pruned = [e for e in entries if datetime.fromisoformat(e["recorded_utc"]) >= cutoff]
        if changed or len(pruned) != len(entries):
            _save_symbol_news_store(symbol, pruned)
            _write_symbol_news_note(symbol, pruned)
    except Exception:
        logger.warning("symbol_news: failed to record vault news for %s (cosmetic only).", symbol, exc_info=True)


def _record_category_news_to_vault(category: str, items: list[dict]) -> None:
    """Same contract as _record_symbol_news_to_vault, for the category-
    specialty layer (FXStreet/CoinDesk/Investing.com) — added 2026-09-20,
    direct user report after checking the vault: category news had no
    vault presence at all before this, live-only in the webapp, so its
    real full article text had nowhere to be saved. Keyed by the raw
    category string (e.g. "Metals CFD"), same key the live cache already
    uses, so every symbol sharing a category accumulates into one shared
    note rather than duplicating it per symbol."""
    try:
        now = datetime.now(timezone.utc)
        path = _category_news_store_path(category)
        entries = _load_news_store(path)
        changed = _record_news_entries(entries, items, now)

        cutoff = now - timedelta(days=config.SYMBOL_NEWS_RETENTION_DAYS)
        pruned = [e for e in entries if datetime.fromisoformat(e["recorded_utc"]) >= cutoff]
        if changed or len(pruned) != len(entries):
            _save_news_store(path, pruned)
            _write_category_news_note(category, pruned)
    except Exception:
        logger.warning("symbol_news: failed to record category vault news for %s (cosmetic only).", category, exc_info=True)


def list_symbol_news(symbol: str) -> list[dict]:
    """Public read for the webapp's News section — the real, persisted
    history for one symbol, most recent first. [] if nothing recorded
    yet, never raises."""
    return sorted(_load_symbol_news_store(symbol), key=lambda e: e["seq"], reverse=True)


def list_category_news(category: str) -> list[dict]:
    """Public read, same contract as list_symbol_news, for the category-
    specialty layer's own persisted history."""
    return sorted(_load_news_store(_category_news_store_path(category)), key=lambda e: e["seq"], reverse=True)


def fetch_full_article_text(link: str) -> str:
    """Thin delegating wrapper over data.news_source.fetch_article_text
    (real page fetch + paragraph extraction, never model-generated),
    kept here so callers only ever need `from data import symbol_news`
    for everything news-related (same convention as every other function
    in this module). Called by _record_symbol_news_to_vault for each
    genuinely new headline, so the vault's own copy is the complete
    article, not just the title/summary the webapp shows. "" for a
    missing/empty link or any real fetch failure — never fabricated,
    never raises."""
    if not link:
        return ""
    return fetch_article_text(link)
