"""Researcher — a third, independent agent, standing alongside the Mega
Session (ai/mega_analysis.py, Claude-based) and Clerk (ai/clerk_
execution.py, local-model execution layer).

**Phase 1 build**: this module is deliberately standalone — nothing in
`ai/ftmo_suggest.py`/`ai/clerk_execution.py` is modified, and nothing
there imports FROM this module yet. This module DOES import a few
already-built, read-only computation functions FROM `ai/ftmo_suggest.py`
and `ai/portfolio_suggest.py` (real technical/backtest analysis, real
macro data) — a one-way, read-only dependency, not a behavior change to
either file; see the "Researcher" plan for the full two-phase design on
the OTHER direction (Researcher's own output feeding those agents).

Real data this pulls in per run, all free (no API key, no Claude
tokens) — general "search the web" APIs were evaluated and rejected:
DuckDuckGo's HTML search now hard-blocks non-browser requests (a real
bot-detection challenge page, not results), and its official Instant
Answer API returned genuinely empty results for real finance queries
live-tested during Phase 1 (it isn't a search engine). Real coverage
instead comes from several free, no-key sources combined:
  - Real news + summaries per symbol — Yahoo Finance first, Google
    News' own public RSS search feed as a second, free source when
    Yahoo genuinely has nothing for that symbol (a real, observed gap:
    USDCHF/USDCAD/USDSEK/USDCNH all came back with zero Yahoo items on
    the same live run).
  - Real category-specialty analyst news — FXStreet (forex/exotics),
    CoinDesk (crypto), Investing.com's commodities feed (metals) — each
    a real, free, no-key RSS feed genuinely dedicated to that asset
    class, added ON TOP of the per-symbol news above for deeper,
    analyst-grade coverage no single per-symbol headline search
    surfaces (direct user request 2026-09-14, after confirming no free
    general web-search API exists).
  - Real market-wide macro/geopolitical headlines (a few macro-proxy
    tickers — S&P 500, crude oil, gold — fetched once, shared across
    every symbol's report; same Yahoo-then-Google fallback) plus CNBC's
    own real top-news RSS feed for broader geopolitical/macro color
    (e.g. a real Iran/Strait-of-Hormuz story surfaced live during
    verification — exactly the kind of catalyst a single instrument's
    own feed would miss).
  - A real macro/economic snapshot (yield curve, DXY, VIX, GDP/
    inflation/unemployment, business-cycle stage) — `ai.portfolio_
    suggest.build_macro_snapshot`, the exact same function the Mega
    Session's own prompt uses, already free (no Claude call inside it)
    — PLUS a real fiscal-position layer this module adds on top
    (government debt-to-GDP, fiscal balance, current account, via the
    same free World Bank API `data.macro_source.fetch_country_
    indicators` already uses) for deeper fundamental analysis (direct
    user request 2026-09-14). The prompt below explicitly asks the model
    to weigh this fiscal data, not just leave it available and unused —
    confirmed live that a model otherwise tends to lean on the more
    news-driven narrative and skip it.
  - Real technical + backtest evidence per symbol (trend, RSI, ATR,
    chart structure, S/R levels, RSI-reaction backtest win rates) —
    `ai.ftmo_suggest.analyze_ftmo_asset_live`, the exact same live,
    native-MT5 analysis pipeline the Mega Session's own Watchlist detail
    popup already uses.
A 3-tier free OpenRouter model cascade (see _run_researcher_model's own
docstring for the real, audited evidence behind this — the local
qwen3:8b that originally filled this role, with and without thinking
enabled, scored 20-24/50 against Claude's 45-49/50 on identical real
data in a blind, third-model-judged audit) then synthesizes ALL of the
above into a structured note — news & catalysts, macro/geopolitical
context, technical positioning, a bull case, a bear case, and a
price-action hypothesis — never inventing a fact not present in the
real data it was handed.

Runs independently of the Mega Session (real Claude token cost, once a
day) and Clerk (technical-only, no research at all today) — on its own
once-daily schedule (see is_researcher_due/next_researcher_check_utc),
timed ~1hr before the US cash-equity-index open, at zero marginal
Claude-token cost per run (OpenRouter's free tier).

Output reaches the user two ways: records/researcher/ (the full
timestamped history, every input block included — see save_research_
report) and, since 2026-09-14 (direct user request — "integrate the
researcher bot with the graph chart we already have"), a living note
per symbol in the SAME Obsidian vault ai.clerk_execution's own closed-
trade journal already writes into (config.OBSIDIAN_VAULT_PATH,
Research/{symbol}.md — see _export_research_note), wikilinked to that
same symbol so a symbol's own note shows its trade history AND its
latest research connected in one graph, not two disconnected systems.
The user's own phrase, "the graph chart we already have," turned out to
mean Obsidian's own Graph View over this vault — NOT the live MT5
terminal's chart-drawing overlay (ai/chart_overlay.py), which was
investigated and deliberately left alone: that module writes ONE
shared file with a full-replace-on-every-write contract, and Clerk
already owns it every poll — a second, independently-scheduled writer
would race those writes and could intermittently wipe the live chart,
exactly the kind of harm to existing, working functionality this
module's whole Phase-1 discipline exists to avoid.

Degrades honestly everywhere: no Yahoo ticker convention for a symbol's
own category, no real news found, or a local-model failure for one
symbol all mean "skip this symbol this cycle, produce no report" —
never a fabricated report. A technical-snapshot fetch failure, or a
failure of the once-per-run shared macro snapshot/macro headlines, is
instead handled as "note the gap plainly and keep going" — real news
was already found for that symbol, so the report is still worth
producing, just honestly missing that one section, rather than throwing
away real data over an unrelated failure. Nothing here is ever allowed
to abort the rest of the run. Mirrors ai.curiosity's own "produce a
timestamped .md report, read the latest one back later" shape, and ai.
clerk_execution's own primary/backup local-model fallback pattern."""

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import config
from ai.ftmo_suggest import FtmoAssetAnalysis, analyze_ftmo_asset_live
from ai.openrouter_client import (
    FAILED_MESSAGE as OPENROUTER_FAILED_MESSAGE,
    MISSING_KEY_MESSAGE as OPENROUTER_MISSING_KEY_MESSAGE,
    run_openrouter,
)
from ai.portfolio_suggest import AUDIT_MODELS, build_macro_snapshot
from data.fundamentals_source import EquityFundamentals, fetch_equity_fundamentals
from data.macro_source import fetch_country_fiscal_indicators
from data.mt5_source import connect, get_market_watch, get_symbol_category
from data.news_source import fetch_google_news, fetch_recent_news, fetch_rss_feed

logger = logging.getLogger(__name__)

# Broad, real, well-known tickers used purely to catch market-wide/
# geopolitical headlines (Fed policy, energy shocks, broad risk-on/
# risk-off sentiment) that a single instrument's own headline feed
# wouldn't surface — S&P 500 (broad risk sentiment), crude oil (energy/
# geopolitical proxy), gold (safe-haven/macro proxy). Fetched ONCE per
# run, shared across every symbol's report, not per-symbol. Each ticker
# is paired with a plain, human-readable Google News search query (its
# own Yahoo ticker spelling, e.g. "^GSPC"/"CL=F", searches poorly there)
# used as the fallback source when Yahoo returns nothing for it.
_MACRO_PROXY_TICKERS = {"^GSPC": "S&P 500", "CL=F": "crude oil", "GC=F": "gold price"}

# CNBC's own real, free, no-key top-news RSS feed — added as a 4th
# shared macro/geopolitical source alongside the proxy tickers above
# (fetched once per run, same as them). Confirmed live to carry genuine
# geopolitical/macro stories a single instrument's own per-symbol news
# search doesn't surface (a real Iran/Strait-of-Hormuz story was live on
# this feed during Phase 1 verification).
_CNBC_TOP_NEWS_URL = "https://www.cnbc.com/id/100003114/device/rss/rss.html"

# Real, free, no-key RSS feeds genuinely dedicated to one asset class —
# added as EXTRA per-symbol grounding alongside the per-symbol Yahoo/
# Google News search above, for deeper analyst-grade coverage a generic
# per-symbol headline search doesn't surface (e.g. FXStreet's own
# cross-pair analyst commentary, which mentions a given pair by name far
# less often than it discusses the broader driver behind it). Live-
# verified content quality: FXStreet/CoinDesk carry a real, distinct
# <description> summary per item; Investing.com's commodities feed only
# a title (left "" honestly rather than duplicated — see fetch_rss_
# feed's own docstring). No equities-specific feed here: no free,
# no-key, genuinely equities-only real feed was found working live (the
# obvious candidate, MarketWatch's top-stories feed, turned out to be
# general/lifestyle content on inspection, not market news) — Yahoo/
# Google News' own per-symbol search already covers equities directly.
_CATEGORY_RSS_FEEDS = {
    "Forex": "https://www.fxstreet.com/rss/news",
    "Exotics": "https://www.fxstreet.com/rss/news",
    "Crypto": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Metals": "https://www.investing.com/rss/commodities.rss",
}

# Defined here (not duplicated as a private constant in researcher_job.py)
# so both callers acquire the literal same lock file, and can never drift
# apart on path or staleness window — same reasoning as ai.clerk_
# execution.EXECUTION_LOCK_PATH's own comment.
RESEARCHER_LOCK_PATH = Path(__file__).resolve().parent.parent / "researcher.lock"
# Comfortably above RESEARCHER_RUN_TIMEOUT_SECONDS, so a lock still
# fresher than this genuinely could be a real, still-running attempt.
RESEARCHER_LOCK_STALE_AFTER_SECONDS = 40 * 60

_SENTIMENT_TAGS = ("BULLISH", "BEARISH", "NEUTRAL")


def _resolve_ftmo_yahoo_ticker(symbol: str, category: str) -> str | None:
    """Real, well-known Yahoo Finance ticker conventions for THIS
    account's own symbol naming — deliberately NOT data.underlying's
    resolve_yahoo_ticker, whose own docstring says it matches "a PMEX
    symbol/description" via a commodity-keyword map (GOLD/SILVER/CRUDE/
    WHEAT/...). Checked directly against this account's real symbol
    descriptions during planning: that map only actually matches XAUUSD/
    XAGUSD (their own descriptions literally say "Gold"/"Silver") — every
    forex pair, all four equities (NVDA/INTC/AMD/MSFT), and both crypto
    symbols would silently return None from it. This function covers the
    real, documented ~17-symbol FTMO universe instead, keyed off the one
    existing deterministic classifier, data.mt5_source.get_symbol_
    category, using Yahoo's own real, publicly documented ticker
    conventions (never invented):
      - Forex/Exotics/Metals CFD -> "<SYMBOL>=X" (Yahoo's real FX/spot-
        metal suffix, e.g. "EURUSD=X", "XAUUSD=X").
      - Crypto (symbol ending "USD") -> "<BASE>-USD" (e.g. "BTC-USD").
      - Equities -> the bare symbol (NVDA/INTC/AMD/MSFT already ARE
        their own real Yahoo ticker).
    Everything else (Commodities, Agriculture, Cash CFD/indices,
    Uncategorized) has no reliable convention known — returns None
    rather than guessing."""
    if category in ("Forex", "Exotics") or category.startswith("Metals"):
        return f"{symbol}=X"
    if category.startswith("Crypto") and symbol.endswith("USD"):
        return f"{symbol[:-3]}-USD"
    if category.startswith("Equities"):
        return symbol
    return None


# Real, well-known keyword sets per major currency/metal — added for the
# per-symbol news relevance filter below, direct user request 2026-09-
# 15/16 ("strategic, targeted... make the scenario more deterministic")
# after a real, root-caused defect: Yahoo's own ticker-news search for
# "GBPUSD=X" returned two genuinely irrelevant cocoa/Ghana commodity
# articles among its top 5 results (confirmed live 2026-09-16 by
# inspecting the raw fetched payload) — NOT a model hallucination, the
# model correctly cited real data it was handed and told was "real news
# for GBPUSD"; the actual defect was upstream, in what got fetched. A
# prompt instruction can't fix a data-quality problem — only filtering
# the bad data out before it ever reaches the prompt can, which is what
# this does. Deliberately conservative and literal (currency
# name/abbreviation/central-bank keywords only) rather than a fuzzy
# relevance model of its own — the goal is catching an obviously
# unrelated item like "cocoa", not making a subjective editorial call.
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


def _symbol_relevance_keywords(symbol: str, category: str) -> list[str]:
    """Real, literal keywords a genuinely relevant news item about
    `symbol` should contain — used by _filter_relevant_news below to
    deterministically drop an obviously off-topic item (see that
    function's own docstring for the real defect this fixes). Forex/
    Exotics/Metals: both halves of the pair (e.g. "GBPUSD" -> GBP + USD
    keywords). Crypto ending "USD": the coin's own keywords + USD.
    Equities: just the bare symbol/ticker itself — a Yahoo equity-ticker
    search is already precise in practice (this defect was only ever
    observed on a currency-pair-style ticker), so no currency-style
    keyword set applies. Anything not covered falls back to [symbol]
    alone, never an empty list (an empty keyword list would make the
    filter below vacuously drop everything)."""
    if category in ("Forex", "Exotics") or category.startswith("Metals"):
        base, quote = symbol[:3], symbol[3:]
        keywords = list(_CURRENCY_KEYWORDS.get(base, ())) + list(_CURRENCY_KEYWORDS.get(quote, ()))
        return keywords or [symbol]
    if category.startswith("Crypto") and symbol.endswith("USD"):
        base = symbol[:-3]
        keywords = list(_CURRENCY_KEYWORDS.get(base, ())) + list(_CURRENCY_KEYWORDS.get("USD", ()))
        return keywords or [symbol]
    return [symbol]


def _filter_relevant_news(news_items: list[dict], keywords: list[str]) -> list[dict]:
    """Deterministically drops a fetched news item whose title+summary
    contains NONE of the real, literal `keywords` for this symbol — see
    _symbol_relevance_keywords's own docstring for the real defect this
    fixes (a genuinely irrelevant cocoa/Ghana item that Yahoo's own
    ticker-news search returned for GBPUSD). Deliberately falls back to
    the ORIGINAL, unfiltered list if filtering would remove every single
    item — an empty result here almost certainly means the keyword set
    is incomplete for this symbol, not that every real fetched item is
    genuinely irrelevant; degrading to "some possibly-noisy items" is
    safer than degrading to "no news at all" for a symbol that DID have
    real news fetched."""
    if not keywords:
        return news_items
    lowered_keywords = [k.lower() for k in keywords]
    filtered = [
        item
        for item in news_items
        if any(k in f"{item.get('title', '')} {item.get('summary', '')}".lower() for k in lowered_keywords)
    ]
    return filtered if filtered else news_items


def parse_researcher_sentiment(text: str) -> str | None:
    """Defensive extraction of a sentiment tag — same "last occurrence
    wins, case-insensitive" convention already proven for Clerk's own
    FINAL_VERDICT: parsing, since a model can restate/second-guess itself
    mid-response. None (never a guessed default) if no recognizable tag
    is present at all.

    Matches BOTH real shapes this is actually called on: the model's own
    raw `SENTIMENT: TAG` line (used by save_research_report/_export_
    research_note, called on report_text before that line is stripped
    out of the saved copy), AND the saved report file's own bolded
    `**Sentiment:** TAG` structured field (used by app.py's own UI badge,
    called on the file latest_research_report reads back — the ONLY
    sentiment line left in that text, since the raw one is deliberately
    stripped to avoid showing it twice). Real bug, caught live 2026-09-
    15: asterisks were never stripped before the prefix check, so
    `"**SENTIMENT:** BULLISH"` never matched `startswith("SENTIMENT:")` —
    every symbol's UI badge silently fell back to the "unknown" emoji
    regardless of its real, correctly-saved sentiment, since app.py only
    ever calls this on the saved file, never on raw model output."""
    found = None
    for line in text.splitlines():
        stripped = line.strip().replace("*", "").upper()
        if stripped.startswith("SENTIMENT:"):
            candidate = stripped.split(":", 1)[1].strip()
            for tag in _SENTIMENT_TAGS:
                if candidate.startswith(tag):
                    found = tag
                    break
    return found


def _fmt(value: float | None, digits: int = 1) -> str:
    """None-safe numeric formatting for the technical snapshot below —
    "n/a" (never a fabricated 0 or blank) when a real stat genuinely
    isn't available (e.g. not enough MT5 history yet for this symbol)."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _parse_published(published: str) -> datetime | None:
    """Real published-date parsing across the three genuinely different
    formats this project's own news sources actually return (confirmed
    live 2026-09-15): Yahoo's own ISO 8601 with a trailing "Z"
    ("2026-09-14T00:00:00Z"), and RFC 822 from both Google News and every
    plain RSS feed ("Fri, 11 Sep 2026 07:00:00 GMT", "Mon, 14 Sep 2026
    08:07:16 +0000", and Investing.com's own slightly different but
    still RFC-822-parseable "Sep 11, 2026 19:01 GMT"). None (never a
    guessed/fabricated date) on an empty string or a shape neither parser
    recognizes."""
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
    """A short, human-readable age ("5m ago", "3h ago", "2d ago") for one
    real news item — added so the model can actually tell a just-
    published headline apart from a stale one instead of treating every
    item in the prompt as equally current (direct user request 2026-09-
    15/16, "look for more dimensions which can further improve quality").
    "" (never a fabricated age) when the published string is missing or
    doesn't parse, or is somehow in the future (a real, if rare,
    possibility — clock skew between this machine and a feed's own
    server — safer to omit than print a nonsensical negative age)."""
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


def _format_news_block(news_items: list[dict], now_utc: datetime | None = None) -> str:
    """Real title + summary + source + age, one item per 1-2 lines — the
    richer shape data.news_source.fetch_recent_news returns, in place of
    the bare titles the original (pre-enrichment) version of this module
    handed the model. "(none)" when there's genuinely nothing, never a
    fabricated placeholder headline. `now_utc` is injectable (same
    reasoning as save_research_report's own records_dir parameter) so
    tests get a fixed, reproducible "now" instead of depending on the
    real clock; defaults to the real current time for every actual
    caller."""
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


def _google_news_query(symbol: str, category: str) -> str:
    """A plain, human-readable Google News search query for `symbol` —
    deliberately NOT its own Yahoo ticker spelling (e.g. "EURUSD=X"),
    which searches poorly there; a real instrument name/symbol phrase
    returns real, relevant coverage instead."""
    if category in ("Forex", "Exotics"):
        return f"{symbol} forex"
    if category.startswith("Metals"):
        return f"{symbol} price"
    if category.startswith("Crypto") and symbol.endswith("USD"):
        return f"{symbol[:-3]} crypto"
    if category.startswith("Equities"):
        return f"{symbol} stock"
    return symbol


def _fetch_news_with_fallback(yahoo_ticker: str, google_query: str, limit: int) -> list[dict]:
    """Yahoo Finance first (real title+summary+source+date); Google
    News' own public RSS search feed as a second, free, no-API-key
    source when Yahoo genuinely has nothing — a real, observed gap this
    closes: USDCHF/USDCAD/USDSEK/USDCNH all came back with zero Yahoo
    items on the same live run, even though real forex commentary for
    every one of them genuinely exists elsewhere. Both fetch functions
    already degrade to [] on any real failure (never raise — see their
    own docstrings), so nothing here needs its own try/except."""
    items = fetch_recent_news(yahoo_ticker, limit=limit)
    if items:
        return items
    return fetch_google_news(google_query, limit=limit)


def _category_rss_feed_url(category: str) -> str | None:
    """The real, free, no-key specialty RSS feed for `category` (see
    _CATEGORY_RSS_FEEDS's own comment) — None (an honest "no extra
    per-category feed exists for this one") for anything not covered,
    same "never guess" posture as _resolve_ftmo_yahoo_ticker."""
    if category in _CATEGORY_RSS_FEEDS:
        return _CATEGORY_RSS_FEEDS[category]
    if category.startswith("Crypto"):
        return _CATEGORY_RSS_FEEDS["Crypto"]
    if category.startswith("Metals"):
        return _CATEGORY_RSS_FEEDS["Metals"]
    return None


def _fetch_category_news_block(category: str, limit: int) -> str:
    """Real, additional analyst-grade coverage from a feed genuinely
    dedicated to `category` (FXStreet/CoinDesk/Investing.com — see
    _CATEGORY_RSS_FEEDS), on top of the per-symbol news search above.
    "(no category-specialty feed for ...)" — an honest, visible note,
    not silence — when no such feed exists for this category (equities,
    or anything else _category_rss_feed_url doesn't cover)."""
    url = _category_rss_feed_url(category)
    if url is None:
        return f"(no category-specialty feed for {category!r})"
    return _format_news_block(fetch_rss_feed(url, limit=limit))


def _fetch_macro_headlines_block() -> str:
    """Real, market-wide/geopolitical-proxy headlines (see
    _MACRO_PROXY_TICKERS's own comment for why these three) plus CNBC's
    own real top-news RSS feed (_CNBC_TOP_NEWS_URL), fetched ONCE per run
    and shared across every symbol's report — this is genuinely the same
    real news feed every symbol's own report would otherwise miss
    entirely, since a single instrument's own headline feed rarely
    surfaces e.g. a Fed rate decision or an OPEC+ supply cut by itself.
    Yahoo first, Google News RSS as a fallback per proxy ticker — same
    _fetch_news_with_fallback both per-symbol fetches use below; CNBC's
    own feed needs no such fallback, it's already a direct, reliable
    RSS pull with no ticker/query to resolve."""
    lines = []
    for ticker, google_query in _MACRO_PROXY_TICKERS.items():
        for item in _fetch_news_with_fallback(ticker, google_query, limit=3):
            suffix = f" — {item['source']}" if item.get("source") else ""
            lines.append(f"- [{ticker}] {item['title']}{suffix}")
    for item in fetch_rss_feed(_CNBC_TOP_NEWS_URL, limit=5):
        lines.append(f"- [CNBC] {item['title']}")
    return "\n".join(lines) if lines else "(none found this run)"


def _format_fiscal_snapshot(indicators: list) -> str:
    """Real per-country debt-to-GDP / fiscal balance / current account
    lines — only the fields a country actually reports (World Bank's
    own fiscal series have real, disclosed gaps for some countries,
    e.g. the US doesn't report cash-basis fiscal balance to it at all,
    confirmed live) are shown; a country with nothing at all reported is
    skipped entirely rather than printed as an empty line."""
    lines = []
    for c in indicators:
        bits = []
        if c.debt_to_gdp_pct is not None:
            bits.append(f"debt/GDP {c.debt_to_gdp_pct:.1f}%")
        if c.fiscal_balance_pct is not None:
            bits.append(f"fiscal balance {c.fiscal_balance_pct:+.1f}% of GDP")
        if c.current_account_pct is not None:
            bits.append(f"current account {c.current_account_pct:+.1f}% of GDP")
        if bits:
            lines.append(f"- {c.country}: {', '.join(bits)}")
    return "\n".join(lines)


def _build_macro_snapshot_block() -> str:
    """The real macro/economic snapshot — the exact same ai.portfolio_
    suggest.build_macro_snapshot() the Mega Session's own Claude prompt
    already uses (yield curve, DXY, VIX, GDP/inflation/unemployment) —
    PLUS a real fiscal-position layer (government debt-to-GDP, fiscal
    balance, current account, per country) this function adds on top,
    via the same free World Bank API, for deeper fundamental analysis
    (direct user request 2026-09-14). Deliberately appended here in ai.
    researcher's OWN function rather than added onto ai.portfolio_
    suggest.build_macro_snapshot() itself — that function is Mega
    Session's own shared dependency, and appending to it would change
    Mega Session's prompt too, which is out of scope for a Researcher-
    only enhancement (see this module's own docstring on the read-only,
    one-way dependency direction). Fetched ONCE per run and shared
    across every symbol's report. Wrapped in its own try/except (unlike
    the per-symbol fetches below, which already sit inside run_
    researcher_check's own per-symbol try/except) because this runs
    BEFORE the per-symbol loop even starts — an unhandled failure here
    would otherwise abort the entire run before a single symbol got
    researched, which is exactly the kind of "one thing failed, don't
    let it take down everything else" failure mode this whole module is
    built to avoid."""
    try:
        snapshot = build_macro_snapshot()
    except Exception:
        logger.exception("Researcher: could not build the macro/economic snapshot this run.")
        snapshot = "(macro/economic snapshot unavailable this run)"

    try:
        fiscal_lines = _format_fiscal_snapshot(fetch_country_fiscal_indicators())
    except Exception:
        logger.exception("Researcher: could not fetch fiscal indicators this run.")
        fiscal_lines = ""

    if fiscal_lines:
        snapshot += "\n\nFiscal/external-balance snapshot (World Bank, latest available year per country):\n" + fiscal_lines
    return snapshot


def _build_technical_snapshot(analysis: FtmoAssetAnalysis) -> str:
    """A condensed, deterministic technical/backtest read for one
    symbol — real fields pulled directly from analyze_ftmo_asset_live's
    own output (the exact same live, native-MT5 4-timeframe pipeline
    ai.clerk_execution's own tactical checks and the Mega Session's own
    Watchlist detail popup already use). Deliberately NOT the full prose
    ai.ftmo_suggest.format_ftmo_asset_context produces for Claude's own
    much larger context budget — kept to a short, bulleted summary
    because this feeds a small local model (qwen3:8b/phi4-mini) that has
    already shown real read-timeout risk on far shorter prompts (a real
    120s primary-model timeout was observed live during Phase 1
    verification on a plain bare-headline prompt)."""
    lines = []
    for label, stats in (("D1", analysis.base.stats), ("H4", analysis.h4_stats), ("H1", analysis.h1_stats)):
        if stats.trend is None:
            continue
        lines.append(
            f"- {label}: trend={stats.trend}, regime={stats.market_regime}, RSI={_fmt(stats.rsi)}, "
            f"ATR%={_fmt(stats.atr_pct)}, 1-month change={_fmt(stats.change_1m_pct)}%"
        )

    sr = analysis.h1_structure.sr_levels
    if sr is not None:
        if sr.support_levels:
            lvl = sr.support_levels[0]
            lines.append(f"- Nearest H1 support: {lvl.price:.5g} ({lvl.distance_pct:+.2f}%, {lvl.touches} touches)")
        if sr.resistance_levels:
            lvl = sr.resistance_levels[0]
            lines.append(
                f"- Nearest H1 resistance: {lvl.price:.5g} ({lvl.distance_pct:+.2f}%, {lvl.touches} touches)"
            )

    for label, bt in (
        ("RSI-oversold reaction", analysis.base.rsi_oversold_backtest),
        ("RSI-overbought reaction", analysis.base.rsi_overbought_backtest),
    ):
        if bt is not None and bt.win_rate_pct is not None:
            lines.append(
                f"- Backtest ({label}): {bt.trades} trades, {bt.win_rate_pct:.0f}% win rate, "
                f"avg {bt.avg_r_multiple:+.2f}R"
            )
    sr_bt = analysis.base.support_resistance_backtest
    if sr_bt is not None:
        if sr_bt.support_win_rate_pct is not None:
            lines.append(
                f"- Backtest (support holds): {sr_bt.support_tests} trades, "
                f"{sr_bt.support_win_rate_pct:.0f}% win rate, avg {sr_bt.support_avg_r_multiple:+.2f}R"
            )
        if sr_bt.resistance_win_rate_pct is not None:
            lines.append(
                f"- Backtest (resistance rejects): {sr_bt.resistance_tests} trades, "
                f"{sr_bt.resistance_win_rate_pct:.0f}% win rate, avg {sr_bt.resistance_avg_r_multiple:+.2f}R"
            )

    return "\n".join(lines) if lines else "(not enough real MT5 history yet for a technical read)"


def _format_equity_fundamentals(fundamentals: EquityFundamentals | None) -> str:
    """Real analyst consensus/price target/forward P/E/earnings-date
    read for one equity symbol — see data.fundamentals_source's own
    docstring for why this exists (equities previously had zero
    valuation/analyst/earnings context). Only the fields yfinance
    actually returned are shown; a symbol with thin/no analyst coverage
    degrades to the honest "(no real equity fundamentals available)"
    note rather than a half-filled, confusing block."""
    if fundamentals is None:
        return "(no real equity fundamentals available for this symbol)"
    lines = []
    if fundamentals.analyst_recommendation is not None:
        count = f" ({fundamentals.analyst_count} analysts)" if fundamentals.analyst_count else ""
        lines.append(f"- Analyst consensus: {fundamentals.analyst_recommendation.replace('_', ' ')}{count}")
    if fundamentals.analyst_target_mean_price is not None:
        lines.append(f"- Analyst mean price target: {fundamentals.analyst_target_mean_price:.2f}")
    if fundamentals.forward_pe is not None:
        lines.append(f"- Forward P/E: {fundamentals.forward_pe:.1f}")
    if fundamentals.next_earnings_date is not None:
        eps = f" (EPS estimate {fundamentals.earnings_eps_estimate:.2f})" if fundamentals.earnings_eps_estimate else ""
        lines.append(f"- Next earnings date: {fundamentals.next_earnings_date}{eps}")
    return "\n".join(lines) if lines else "(no real equity fundamentals available for this symbol)"


def _read_calibration_ledger() -> dict:
    """Best-effort read of the real per-symbol calibration ledger — `{}`
    on anything missing/unreadable, same safe-default convention as
    every other state file this module reads."""
    path = Path(config.RESEARCHER_CALIBRATION_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_calibration_ledger(ledger: dict) -> None:
    try:
        Path(config.RESEARCHER_CALIBRATION_FILE).write_text(json.dumps(ledger, indent=2))
    except OSError as e:
        logger.warning("Could not write researcher calibration ledger %s: %s", config.RESEARCHER_CALIBRATION_FILE, e)


def _record_calibration_entry(symbol: str, sentiment: str, price: float, now_utc: datetime) -> None:
    """Appends this run's own real {timestamp, sentiment, price} entry
    for `symbol`, trimmed to the most recent RESEARCHER_CALIBRATION_MAX_
    ENTRIES_PER_SYMBOL so the ledger file can't grow unbounded. Called
    only when a real sentiment tag was actually parsed (never records a
    fabricated/guessed entry for a totally failed model call)."""
    ledger = _read_calibration_ledger()
    entries = ledger.setdefault(symbol, [])
    entries.append({"timestamp": now_utc.isoformat(), "sentiment": sentiment, "price": price})
    ledger[symbol] = entries[-config.RESEARCHER_CALIBRATION_MAX_ENTRIES_PER_SYMBOL :]
    _write_calibration_ledger(ledger)


def _build_track_record_block(symbol: str, current_price: float, now_utc: datetime) -> str:
    """Real self-calibration (direct user request 2026-09-15/16, "look
    for more dimensions which can further improve quality"): compares
    THIS symbol's own past real sentiment calls (from the calibration
    ledger) against what price actually did since, so a new report can
    honestly weigh its own recent reliability on this specific symbol
    instead of every fresh call reading as equally trustworthy. A
    simple directional check — current price higher than at call time
    counts BULLISH correct, lower counts BEARISH correct — not a formal
    backtest with cost/slippage/R-multiples; it exists to give the model
    (and a human reader) an honest confidence signal, not a trading
    edge measurement. NEUTRAL calls are excluded from scoring (no
    directional claim to grade). A call younger than RESEARCHER_
    CALIBRATION_MIN_AGE_HOURS is excluded too — not enough time has
    genuinely passed to fairly judge it yet, so it would otherwise just
    add noise."""
    entries = _read_calibration_ledger().get(symbol, [])
    min_age = timedelta(hours=config.RESEARCHER_CALIBRATION_MIN_AGE_HOURS)
    correct = wrong = 0
    for entry in entries:
        try:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            sentiment = entry["sentiment"]
            price = float(entry["price"])
        except (KeyError, ValueError, TypeError):
            continue
        if sentiment not in ("BULLISH", "BEARISH"):
            continue
        if now_utc - entry_time < min_age:
            continue
        went_up = current_price > price
        if (sentiment == "BULLISH" and went_up) or (sentiment == "BEARISH" and not went_up):
            correct += 1
        else:
            wrong += 1
    total = correct + wrong
    if total == 0:
        return "(no past directional calls for this symbol are old enough to judge yet)"
    return (
        f"{correct}/{total} of this symbol's own past directional calls have been correct so far "
        "(simple price-direction check since the call was made, not a formal backtest)."
    )


def _extract_price_anchors(analysis: FtmoAssetAnalysis, current_price: float) -> list[float]:
    """Real, deterministically-computed price levels this symbol's own
    report should be checked against — the current price plus the
    nearest real H1 support/resistance from analyze_ftmo_asset_live's
    own output. NEVER anything the model itself proposes — this is the
    real "ideal reference state" a fabricated price level gets checked
    against by _check_price_grounding below."""
    anchors = [current_price]
    sr = analysis.h1_structure.sr_levels
    if sr is not None:
        if sr.support_levels:
            anchors.append(sr.support_levels[0].price)
        if sr.resistance_levels:
            anchors.append(sr.resistance_levels[0].price)
    return anchors


def _extract_section(text: str, heading: str) -> str:
    """Best-effort extraction of one "### {heading}" section's own body
    text, up to the next "###" heading or the closing SENTIMENT: line —
    "" (never a guess) if the heading isn't present at all, e.g. a
    genuinely malformed/failed model response."""
    match = re.search(
        rf"###\s*{re.escape(heading)}\s*\n(.*?)(?=\n###|\nSENTIMENT:|\Z)", text, re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else ""


def _check_price_grounding(report_text: str, price_anchors: list[float]) -> str | None:
    """Deterministic anti-hallucination guard — direct user request
    2026-09-15/16, after a real, independently-audited failure: an
    independent 550B-parameter audit model (Nvidia Nemotron-Ultra-550B,
    via a real blind scorecard comparison against Claude on identical
    GBPUSD data) scored qwen3:8b's own Price Action Hypothesis at 5/10
    for actionability specifically because it "targets 1.3200 with no
    source in the given data." Mirrors the same "verify deterministically,
    never just trust the model's own claim" philosophy as ai.clerk_
    execution's own never-widen-stop/sane-side-of-price checks — but
    since Researcher is read-only research, never a live execution gate,
    this never blocks or rewrites anything, it only appends an honest,
    visible disclosure line when the model's own cited price level
    doesn't match reality, so a human reading the report knows to treat
    it with extra caution rather than trusting it silently. None (no
    disclosure needed) when at least one real number in the Price Action
    Hypothesis is within 1% of a real anchor (current price, or the
    nearest real H1 support/resistance) — a tolerance generous enough
    for genuine rounding, tight enough to still catch a fabricated
    level."""
    if not price_anchors:
        return None
    section = _extract_section(report_text, "Price Action Hypothesis")
    if not section:
        return None
    numbers = [float(n) for n in re.findall(r"\d+\.\d+", section)]
    if not numbers:
        return None
    tolerance = max(price_anchors) * 0.01
    for number in numbers:
        if any(abs(number - anchor) <= tolerance for anchor in price_anchors):
            return None
    return (
        "⚠ The price level(s) cited in the Price Action Hypothesis above could not be matched "
        "to the real current price or the real computed support/resistance levels — treat the "
        "specific number with extra caution; the directional call itself may still be reasonable."
    )


def _run_researcher_model(
    symbol: str,
    news_block: str,
    category_news_block: str,
    macro_headlines_block: str,
    macro_snapshot_block: str,
    technical_block: str,
    equity_fundamentals_block: str,
    track_record_block: str,
) -> str:
    """A 3-tier free cloud-model cascade — direct user decision 2026-09-
    15/16, after a real, independently-audited head-to-head: the local
    qwen3:8b (both with and without thinking enabled) scored 20-24/50
    against Claude's 45-49/50 on identical real data, and one of its
    genuine failures traced back to citing a price level "with no
    source in the given data" — not something a prompt fix alone could
    fully close. The SAME real cascade this project already trusts for
    Mega Session's own audit pass (ai.portfolio_suggest.AUDIT_MODELS —
    Nvidia Nemotron-Ultra-550B, Dots Studio Dots3-Note-280B, Nvidia
    Nemotron-Super-120B, in that order) is reused here rather than
    inventing a second roster — when live-tested head-to-head against
    Claude on the SAME real GBPUSD data qwen3:8b was tested on, the
    280B tier alone scored 40/50, closing most of the gap in a single
    real comparison (33s, free, zero Claude-token cost). Tries each
    tier via run_openrouter in order, falling to the next ONLY on that
    exact tier's own real failure (a timeout, a rate limit, an error) —
    never a partial/low-quality response, since run_openrouter itself
    already degrades any such failure to one of its own two constant
    failure strings. No local Ollama fallback: this project's own real
    audit evidence didn't support keeping qwen3:8b in the loop at all,
    even as a last resort — a report skipped this cycle (see run_
    researcher_check's own honest-skip conventions) is preferred over
    one from a demonstrably weaker model. Hands the model ONLY real,
    already-fetched/already-computed data — nothing invented — and asks
    for a structured note covering
    news/catalysts, macro/geopolitical context, technical positioning, a
    bull case, a bear case, and a price-action hypothesis tying it all
    together, ending with one machine-parseable SENTIMENT: line. The
    Macro & Geopolitical Context heading carries an explicit instruction
    to weigh the fiscal snapshot (debt-to-GDP/fiscal balance/current
    account) — direct user request 2026-09-14, after live-observing the
    model's own synthesis lean on news/rate narrative and skip the
    fiscal data even though it was present in the prompt; the data was
    always available to it, this just makes using it an explicit ask
    rather than an optional one. The closing SENTIMENT: line also carries
    an explicit consistency instruction — direct user question 2026-09-14
    ("why all researches pushing sideways, is the research weak"), traced
    to a real, live-observed defect: EURUSD's own real technical read
    showed a downtrend on all three timeframes with oversold RSI, and its
    own Technical Positioning prose said so plainly, yet the model still
    tagged SENTIMENT: NEUTRAL — a real self-contradiction, not a data-
    pipeline bug (9 of 17 reports came back NEUTRAL that run). This is a
    genuine small-local-model weakness (qwen3:8b/phi4-mini hedging to
    NEUTRAL rather than committing, even against its own written
    reasoning) that the Mega Session's much larger Claude model wouldn't
    be expected to exhibit — the fix applied is a prompt instruction, not
    a deterministic override of the model's own read, since the model's
    own nuanced textual reasoning could legitimately outweigh a purely
    mechanical trend-agreement check in a real mixed-evidence case.

    `equity_fundamentals_block`/`track_record_block` (direct user request
    2026-09-15/16, "look for more dimensions which can further improve
    quality"): real analyst/earnings context for equity symbols (empty-
    category instruments get an honest "(not applicable...)" note rather
    than an empty section), and this symbol's own real recent-calibration
    track record (see _build_track_record_block's own docstring) — both
    ALWAYS passed (never optional/omitted) so a caller can never forget
    to wire one in, matching every other block here.

    The market-wide-vs-symbol-specific attribution instruction and the
    "cite only real numbers" instruction (direct user request 2026-09-
    15/16, "strategic, targeted prompts... make the scenario more
    deterministic") were both added after a real, independently-audited
    failure: a blind scorecard from Nvidia Nemotron-Ultra-550B (a 550B-
    parameter model, given the identical real GBPUSD data this function
    was fed) scored qwen3:8b-with-thinking at 24/50 against Claude's
    48/50 on the exact same inputs, explicitly citing two concrete,
    fixable defects — misattributing broad macro-proxy headlines
    (Saudi pipeline, ECB, an unrelated cocoa item) as if they were
    GBPUSD-specific drivers, and citing a price target "with no source
    in the given data." run_researcher_check's own _check_price_
    grounding is the deterministic half of this same fix — a check that
    verifies rather than just asks, mirroring ai.clerk_execution's own
    "never trust the model's own claim, verify it" philosophy."""
    prompt = (
        f"You are a markets researcher producing a detailed research note for {symbol}, "
        "in the style of a fund manager's daily instrument note. You are given ONLY real "
        "data below: real recently published news for this instrument, real analyst-grade "
        "news for its asset class, real market-wide macro/geopolitical headlines, a real "
        "macro/economic snapshot, a real technical/backtest read, real equity fundamentals "
        "(if applicable), and your own real recent track record on this symbol. Do not "
        "invent any fact, figure, event, or date that isn't present in this data — where "
        "the data doesn't support a claim, say so plainly instead of filling the gap with "
        "plausible-sounding text. Each news item is labeled with its real age (e.g. "
        "\"2h ago\", \"3d ago\") — weigh a several-day-old item as stale background, not a "
        "fresh catalyst, and prefer the freshest items when sources disagree. If two real "
        "sources genuinely conflict (e.g. one bullish narrative, one bearish, on the same "
        "driver), say so explicitly in your synthesis rather than silently picking one "
        "side as if there were no disagreement. The \"market-wide macro/geopolitical "
        "headlines\" section below is NOT specific to this symbol — treat it as broad "
        "background only, and never claim one of those items is a direct driver of THIS "
        "symbol unless the item's own text actually names this symbol, its underlying "
        "currency, or its issuing company. Any specific price level you cite anywhere in "
        "your answer (a target, a support, a resistance) MUST be one of the exact real "
        "numbers given in the technical/backtest read below — never a number you calculate "
        "or guess yourself.\n\n"
        f"## Real news for {symbol}\n{news_block}\n\n"
        f"## Real analyst-grade news for {symbol}'s asset class\n{category_news_block}\n\n"
        "## Real market-wide macro/geopolitical headlines\n"
        f"{macro_headlines_block}\n\n"
        f"## Real macro/economic snapshot\n{macro_snapshot_block}\n\n"
        f"## Real technical/backtest read for {symbol}\n{technical_block}\n\n"
        f"## Real equity fundamentals for {symbol}\n{equity_fundamentals_block}\n\n"
        f"## Your own real recent track record on {symbol}\n{track_record_block}\n\n"
        "Write the note with exactly these section headings, each 1-3 sentences (if "
        "truly nothing above supports a section, keep the heading and say so plainly "
        "rather than inventing content for it):\n"
        "### News & Catalysts\n"
        "### Macro & Geopolitical Context\n"
        "(explicitly weigh the relevant country/countries' fiscal position from the "
        "Fiscal/external-balance snapshot above — debt-to-GDP, fiscal balance, current "
        "account — whenever it's relevant to this instrument, e.g. a high debt-to-GDP or "
        "wide current-account deficit is a real currency/rate-risk driver worth naming, "
        "not just the growth/inflation/unemployment cycle data)\n"
        "### Technical Positioning\n"
        "(for an equity, also weigh the real analyst consensus/price target/forward P/E "
        "and any upcoming earnings date above when relevant — an earnings date within the "
        "next 1-2 weeks is a real, near-term volatility catalyst worth naming explicitly)\n"
        "### Bull Case\n"
        "### Bear Case\n"
        "### Price Action Hypothesis\n"
        "(tie the sections above together into one concrete, falsifiable expectation for "
        "where price likely goes next and what would invalidate it — also briefly note "
        "whether your own recent track record on this symbol above should make you more "
        "or less confident this time, rather than ignoring it)\n\n"
        "End your response with exactly one line in this exact format:\n"
        "SENTIMENT: BULLISH | BEARISH | NEUTRAL\n"
        "This tag MUST match the direction your own Technical Positioning, Bull Case, and "
        "Bear Case sections actually argue for — if you wrote that the trend is down across "
        "timeframes with confirming momentum, the tag must be BEARISH, not NEUTRAL; if it's "
        "up, BULLISH. Reserve NEUTRAL for when the real evidence above is genuinely mixed or "
        "balanced, not as a default hedge when you're simply unsure — a directional read "
        "backed by the data is more useful than a safe non-answer."
    )
    raw = OPENROUTER_FAILED_MESSAGE
    for i, (display_name, model_id, _description) in enumerate(AUDIT_MODELS):
        raw = run_openrouter(prompt, model=model_id, timeout=config.RESEARCHER_MODEL_TIMEOUT_SECONDS)
        if raw not in (OPENROUTER_FAILED_MESSAGE, OPENROUTER_MISSING_KEY_MESSAGE):
            if i == 0:
                return raw
            # A lower tier answered — disclosed the same way the old
            # primary/backup cascade disclosed a backup response, so a
            # reader always knows which real model actually produced
            # this specific report, never silently swapped.
            return f"[{AUDIT_MODELS[0][0]} unavailable — {display_name} responded]\n\n{raw}"
    return raw


def _strip_sentiment_line(report_text: str) -> str:
    """The model's own trailing "SENTIMENT: ..." line, removed — used
    everywhere the sentiment tag is ALSO shown as its own structured
    field (save_research_report's **Sentiment:** line, _export_research_
    note's frontmatter tag), so it doesn't appear a second time,
    verbatim, at the end of the prose too."""
    return "\n".join(
        line for line in report_text.splitlines() if not line.strip().upper().startswith("SENTIMENT:")
    ).strip()


def _export_research_note(symbol: str, report_text: str, vault_path: Path | None = None) -> None:
    """Best-effort side effect: writes this run's research synthesis into
    the SAME Obsidian vault ai.clerk_execution's own _export_closed_
    trade_notes already writes closed-trade journal entries into
    (config.OBSIDIAN_VAULT_PATH) — direct user request 2026-09-14, after
    confirming Researcher's own output wasn't connected to it at all.
    Written under Research/{symbol}.md as ONE living note per symbol
    (overwritten each successful run), unlike Trades/'s one-file-per-
    closed-trade history — Obsidian's own Graph View cares about the
    CURRENT connections between notes, not old versions, and records/
    researcher/ already keeps the full timestamped history for that.
    Cross-linked via `[[{symbol}]]` — the exact same wikilink _format_
    closed_trade_note already emits — so a symbol's own note in Obsidian
    shows both its trade history AND its latest research connected in
    one graph, plus `[[Research]]` (a bare, intentionally-unresolved
    index link, mirroring `[[Trade Journal]]`'s own role for Trades/ —
    Obsidian resolves/creates neither eagerly, both work fine as graph
    nodes regardless). Never raises: a locked file or missing/unmounted
    vault path is logged and swallowed, same "cosmetic side effect must
    never break the real feature" contract as every other best-effort
    write in this module and in ai.clerk_execution's own equivalent.
    `vault_path` is injectable (same reasoning as save_research_report's
    own records_dir parameter) so tests never write into the user's own
    real, live Obsidian vault — a real leak caught live during this
    feature's own first test run, cleaned up immediately, fixed here."""
    try:
        vault_root = Path(config.OBSIDIAN_VAULT_PATH) if vault_path is None else vault_path
        vault_dir = vault_root / "Research"
        vault_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc)
        sentiment = parse_researcher_sentiment(report_text) or "UNCLEAR"
        synthesis_only = _strip_sentiment_line(report_text)
        content = (
            "---\n"
            f"tags: [research, {sentiment.lower()}]\n"
            "---\n\n"
            f"# {symbol} — Research ({sentiment})\n\n"
            f"*Last updated {timestamp:%Y-%m-%d %H:%M} UTC by Researcher.*\n\n"
            f"{synthesis_only}\n\n"
            f"[[{symbol}]]\n"
            "[[Research]]\n"
        )
        (vault_dir / f"{symbol}.md").write_text(content, encoding="utf-8")
    except OSError as e:
        logger.warning("Could not export Obsidian research note for %s: %s", symbol, e)


def save_research_report(
    symbol: str,
    report_text: str,
    news_block: str,
    category_news_block: str,
    macro_headlines_block: str,
    macro_snapshot_block: str,
    technical_block: str,
    equity_fundamentals_block: str,
    track_record_block: str,
    records_dir: Path | None = None,
) -> Path | None:
    """Mirrors ai.curiosity.save_curiosity_report's own pattern — a
    timestamped, human-readable .md file, one per symbol per run. Saves
    ALL the real inputs the model was actually given (not just its own
    prose synthesis) so a human reading the file can see exactly what
    real data grounded it — same "show your work" reasoning as ai.
    ftmo_suggest's own prompt-transparency conventions elsewhere in this
    codebase. Never raises: an OSError (disk full, permissions) is
    swallowed and None returned, same "cosmetic side effect, must never
    break the real feature" contract as its model. `records_dir` is
    injectable (same reasoning as save_curiosity_report's own parameter)
    so tests never need to patch config.RESEARCHER_RECORDS_DIR globally."""
    records_dir = Path(config.RESEARCHER_RECORDS_DIR) if records_dir is None else records_dir
    timestamp = datetime.now(timezone.utc)
    sentiment = parse_researcher_sentiment(report_text) or "UNCLEAR"
    synthesis_only = _strip_sentiment_line(report_text)
    content = (
        f"# Research Report — {symbol} — {timestamp:%Y-%m-%d %H:%M:%S} UTC\n\n"
        f"**Sentiment:** {sentiment}\n\n"
        "## Research Note\n\n"
        f"{synthesis_only}\n\n"
        "## Real inputs used\n\n"
        f"### News — {symbol}\n\n{news_block}\n\n"
        f"### Analyst-grade asset-class news — {symbol}\n\n{category_news_block}\n\n"
        f"### Market-wide macro/geopolitical headlines\n\n{macro_headlines_block}\n\n"
        f"### Macro/economic snapshot\n\n{macro_snapshot_block}\n\n"
        f"### Technical/backtest read — {symbol}\n\n{technical_block}\n\n"
        f"### Equity fundamentals — {symbol}\n\n{equity_fundamentals_block}\n\n"
        f"### Self-calibration track record — {symbol}\n\n{track_record_block}\n"
    )
    try:
        records_dir.mkdir(parents=True, exist_ok=True)
        path = records_dir / f"{symbol}_{timestamp:%Y-%m-%d_%H%M%S}.md"
        path.write_text(content, encoding="utf-8")
        return path
    except OSError as e:
        logger.warning("Could not save research report for %s: %s", symbol, e)
        return None


def latest_research_report(symbol: str, records_dir: Path | None = None) -> str | None:
    """The most recent saved report's own full text for `symbol`, or
    None if none exists yet — same glob-newest-and-read idiom as ai.
    curiosity.build_recursion_context. Not called by anything yet
    (Phase 1) — built now so a future Phase 2 consumer has a single,
    already-tested read path rather than reinventing this glob."""
    records_dir = Path(config.RESEARCHER_RECORDS_DIR) if records_dir is None else records_dir
    if not records_dir.exists():
        return None
    files = sorted(records_dir.glob(f"{symbol}_*.md"), reverse=True)
    if not files:
        return None
    try:
        return files[0].read_text(encoding="utf-8")
    except OSError:
        return None


def run_researcher_check(on_stage: Callable[[str], None] | None = None) -> None:
    """The real, standalone pipeline: connect, list this account's live
    Market Watch, and for each symbol whose category has a real, known
    Yahoo ticker convention, fetch real headlines and save a synthesized
    report. A per-symbol failure (bad ticker, no headlines, model
    unavailable, unexpected exception) is logged and skipped — never
    allowed to abort the whole run.

    A real MT5 connection failure is deliberately NOT caught here — it
    propagates to the caller (researcher_job.py, matching ai.mega_
    analysis.run_scheduled_mega_analysis's own "never silently swallow a
    real failure" convention), so the run-frequency interval is only
    marked as run once a genuine attempt actually completed."""

    def _notify(message: str) -> None:
        logger.info(message)
        _write_researcher_progress(message)
        if on_stage:
            on_stage(message)

    _notify("Connecting to the FTMO MT5 account...")
    connect(login=config.FTMO_MT5_LOGIN, password=config.FTMO_MT5_PASSWORD, server=config.FTMO_MT5_SERVER)

    _notify("Fetching the live Market Watch symbol list...")
    assets = get_market_watch()
    if not assets:
        _notify("No instruments visible in Market Watch — nothing to research.")
        _write_researcher_state("no_assets")
        return

    _notify("Building the shared macro/economic snapshot and macro headlines for this run...")
    macro_snapshot_block = _build_macro_snapshot_block()
    macro_headlines_block = _fetch_macro_headlines_block()

    researched = 0
    for i, asset in enumerate(assets, start=1):
        _notify(f"Researching {i}/{len(assets)} — {asset.symbol}...")
        try:
            category = get_symbol_category(asset.symbol)
            yahoo_ticker = _resolve_ftmo_yahoo_ticker(asset.symbol, category)
            if yahoo_ticker is None:
                logger.info(
                    "%s: no known Yahoo ticker convention for category %r — skipped.", asset.symbol, category,
                )
                continue
            google_query = _google_news_query(asset.symbol, category)
            news_items = _fetch_news_with_fallback(
                yahoo_ticker, google_query, limit=config.RESEARCHER_HEADLINES_PER_SYMBOL
            )
            if not news_items:
                logger.info("%s: no real news found on Yahoo or Google News — skipped.", asset.symbol)
                continue
            news_items = _filter_relevant_news(news_items, _symbol_relevance_keywords(asset.symbol, category))
            news_block = _format_news_block(news_items)
            category_news_block = _fetch_category_news_block(category, limit=config.RESEARCHER_HEADLINES_PER_SYMBOL)

            analysis = None
            try:
                analysis = analyze_ftmo_asset_live(asset.symbol, asset.bid, asset.ask, asset.description)
                technical_block = _build_technical_snapshot(analysis)
            except Exception:
                logger.exception(
                    "%s: could not fetch a live technical/backtest snapshot — continuing without it.", asset.symbol
                )
                technical_block = "(technical snapshot unavailable this cycle — live MT5 fetch failed)"

            # fetch_equity_fundamentals already degrades to None on any
            # real failure (never raises — see its own docstring), so
            # this needs no try/except of its own, unlike the technical
            # snapshot above (analyze_ftmo_asset_live can genuinely
            # raise on a real MT5 hiccup).
            if category.startswith("Equities"):
                equity_fundamentals_block = _format_equity_fundamentals(fetch_equity_fundamentals(yahoo_ticker))
            else:
                equity_fundamentals_block = "(not applicable — not an equity)"

            current_price = (asset.bid + asset.ask) / 2
            now_utc = datetime.now(timezone.utc)
            track_record_block = _build_track_record_block(asset.symbol, current_price, now_utc)

            report_text = _run_researcher_model(
                asset.symbol, news_block, category_news_block, macro_headlines_block, macro_snapshot_block,
                technical_block, equity_fundamentals_block, track_record_block,
            )

            if analysis is not None:
                price_anchors = _extract_price_anchors(analysis, current_price)
                grounding_note = _check_price_grounding(report_text, price_anchors)
                if grounding_note:
                    report_text = f"{report_text}\n\n{grounding_note}"

            save_research_report(
                asset.symbol, report_text, news_block, category_news_block, macro_headlines_block,
                macro_snapshot_block, technical_block, equity_fundamentals_block, track_record_block,
            )
            _export_research_note(asset.symbol, report_text)

            sentiment_tag = parse_researcher_sentiment(report_text)
            if sentiment_tag is not None:
                _record_calibration_entry(asset.symbol, sentiment_tag, current_price, now_utc)

            researched += 1
        except Exception:
            logger.exception("Researcher: unhandled failure for %s — skipping it this cycle.", asset.symbol)

    _notify(f"Researcher run complete — {researched}/{len(assets)} symbol(s) got a fresh report.")
    _write_researcher_state("success", f"{researched}/{len(assets)} symbol(s) researched")


# --- job-lifecycle plumbing: state/progress/enabled/interval/due-check ---
# Identical shape to ai.clerk_execution's own equivalents (read_execution_
# state/_write_execution_state/read_execution_progress/read_clerk_
# execution_enabled/read_clerk_execution_interval_minutes/is_execution_
# due/_mark_interval_ran) — deliberately copied rather than shared, since
# each agent's own state file must stay fully independent (a corrupt/
# missing Researcher state file must never affect Clerk's or the Mega
# Session's own due-checks, and vice versa).


def read_researcher_state() -> dict:
    """Best-effort: the outcome of the last Researcher run. `{}` on
    anything missing/unreadable, same safe-default convention as ai.
    mega_analysis.read_state / ai.clerk_execution.read_execution_state."""
    path = Path(config.RESEARCHER_STATE_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_researcher_state(status: str, detail: str = "") -> None:
    """`last_run_date_utc` is set to today ONLY when `status == "success"`
    — mirrors ai.mega_analysis._write_state exactly, replacing the old
    always-mark-the-interval-ran behavior (direct user decision 2026-09-
    15/16, moving to a once-daily schedule): a failed/no_assets attempt
    now genuinely allows a retry later the SAME day, within the grace
    window, instead of permanently burning that day's one real chance —
    real reliability that matters far more once a missed slot means
    missing the whole day, not just one of 24 hourly attempts."""
    prior = read_researcher_state()
    now = datetime.now(timezone.utc)
    payload = {
        "last_attempt_utc": now.isoformat(),
        "last_status": status,
        "last_detail": detail,
        "last_run_date_utc": now.date().isoformat() if status == "success" else prior.get("last_run_date_utc"),
        # A real, persistent "how many times has Researcher actually run"
        # counter — direct user request 2026-09-14. Incremented on every
        # completed attempt regardless of outcome; a genuine MT5
        # connection failure never reaches this function at all (see
        # run_researcher_check's own docstring on why that's deliberate),
        # so it correctly does NOT count a failed-before-it-started attempt.
        "total_runs": prior.get("total_runs", 0) + 1,
    }
    try:
        Path(config.RESEARCHER_STATE_FILE).write_text(json.dumps(payload, indent=2))
    except OSError as e:
        logger.warning("Could not write researcher state file %s: %s", config.RESEARCHER_STATE_FILE, e)


def read_researcher_progress() -> dict:
    """Best-effort read of this job's own LIVE, in-progress status —
    mirrors ai.mega_analysis.read_progress / ai.clerk_execution.read_
    execution_progress exactly, same standing "automated runs must be as
    visible as a manual button click" principle applied to this third
    unattended job."""
    path = Path(config.RESEARCHER_PROGRESS_FILE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_researcher_progress(message: str) -> None:
    payload = {"message": message, "updated_utc": datetime.now(timezone.utc).isoformat()}
    try:
        Path(config.RESEARCHER_PROGRESS_FILE).write_text(json.dumps(payload))
    except OSError as e:
        logger.warning("Could not write researcher progress file %s: %s", config.RESEARCHER_PROGRESS_FILE, e)


def researcher_is_live(progress: dict, state: dict) -> bool:
    """True iff `progress` reflects a Researcher run that's still
    genuinely in flight — same shape of check as ai.mega_analysis.mega_
    session_is_live / ai.clerk_execution.execution_check_is_live."""
    progress_ts = progress.get("updated_utc")
    if not progress_ts:
        return False
    try:
        progress_dt = datetime.fromisoformat(progress_ts)
    except ValueError:
        return False
    age_seconds = (datetime.now(timezone.utc) - progress_dt).total_seconds()
    last_attempt = state.get("last_attempt_utc")
    newer_than_last_attempt = last_attempt is None or progress_ts > last_attempt
    return 0 <= age_seconds < config.RESEARCHER_RUN_TIMEOUT_SECONDS and newer_than_last_attempt


def read_researcher_enabled() -> bool:
    """Whether Researcher is currently enabled — defaults to True on a
    missing/corrupt file, same opt-out-not-opt-in posture as ai.mega_
    analysis.read_mega_analysis_enabled / ai.clerk_execution.read_clerk_
    execution_enabled."""
    path = Path(config.RESEARCHER_ENABLED_FILE)
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    return bool(data.get("enabled", True))


def set_researcher_enabled(enabled: bool) -> None:
    try:
        Path(config.RESEARCHER_ENABLED_FILE).write_text(json.dumps({"enabled": enabled}))
    except OSError as e:
        logger.warning("Could not write researcher enabled-flag file %s: %s", config.RESEARCHER_ENABLED_FILE, e)


def read_researcher_trigger() -> tuple[int, int]:
    """(hour_utc, minute_utc) of Researcher's once-daily run — falls back
    to config.RESEARCHER_TRIGGER_HOUR_UTC/MINUTE_UTC on a missing/corrupt/
    out-of-range value. Mirrors ai.mega_analysis.read_mega_analysis_
    trigger exactly (direct user decision 2026-09-15/16: Researcher moves
    from a repeating-interval schedule to a once-a-day schedule, timed
    ~1hr before the US cash-equity-index open — same real anchor already
    used by MEGA_ANALYSIS_TRIGGER_HOUR_UTC's own comment, 13:30 UTC)."""
    path = Path(config.RESEARCHER_TRIGGER_FILE)
    default = (config.RESEARCHER_TRIGGER_HOUR_UTC, config.RESEARCHER_TRIGGER_MINUTE_UTC)
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text())
        hour = int(data["hour_utc"])
        minute = int(data["minute_utc"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return default
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return default
    return hour, minute


def set_researcher_trigger(hour_utc: int, minute_utc: int) -> None:
    try:
        Path(config.RESEARCHER_TRIGGER_FILE).write_text(json.dumps({"hour_utc": hour_utc, "minute_utc": minute_utc}))
    except OSError as e:
        logger.warning("Could not write researcher trigger-time file %s: %s", config.RESEARCHER_TRIGGER_FILE, e)


def _researcher_trigger_time_utc(for_date: date) -> datetime:
    hour, minute = read_researcher_trigger()
    return datetime(for_date.year, for_date.month, for_date.day, hour, minute, tzinfo=timezone.utc)


def is_researcher_due(now_utc: datetime, state: dict | None = None) -> bool:
    """True only within the RESEARCHER_GRACE_MINUTES window after today's
    trigger instant, and only if Researcher hasn't already succeeded
    today — same once-daily trigger/grace-window/dedup shape as ai.mega_
    analysis.is_due, just for this third job's own state file. A prior
    "no_assets"/failed attempt today does NOT count as done (see _write_
    researcher_state's own docstring), so a same-day retry within the
    grace window is still possible."""
    state = state if state is not None else read_researcher_state()
    today = now_utc.date()
    if state.get("last_run_date_utc") == today.isoformat():
        return False
    today_trigger = _researcher_trigger_time_utc(today)
    grace_end = today_trigger + timedelta(minutes=config.RESEARCHER_GRACE_MINUTES)
    return today_trigger <= now_utc <= grace_end


def next_researcher_check_utc(now_utc: datetime, state: dict | None = None) -> datetime:
    """Best-effort next Researcher-run instant, purely for display (the
    real trigger is researcher_job.py's own OS-level poll)."""
    state = state if state is not None else read_researcher_state()
    today = now_utc.date()
    ran_today = state.get("last_run_date_utc") == today.isoformat()
    today_trigger = _researcher_trigger_time_utc(today)
    if not ran_today and now_utc < today_trigger:
        return today_trigger
    return _researcher_trigger_time_utc(today + timedelta(days=1))
