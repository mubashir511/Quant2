from utils import run_with_timeout

# Confirmed live: yf.Ticker(...).news's own internal cookie/crumb
# negotiation can hang well past its own per-request 10-30s timeouts
# under some process contexts (a non-interactive Windows Scheduled Task,
# specifically) — this hard wall-clock ceiling is what turns that into a
# real, visible "no headlines this time" instead of hanging the entire
# caller (which, for the unattended daily mega-analysis job, meant
# silently killing the whole run with no error ever logged). See
# utils.run_with_timeout's own docstring for the full incident.
_NEWS_TIMEOUT_SECONDS = 15.0


def fetch_recent_headlines(yahoo_ticker: str, limit: int = 2) -> list[str]:
    """Up to `limit` recent headline titles for a Yahoo ticker. Empty on failure."""

    def _fetch() -> list:
        import yfinance as yf

        return yf.Ticker(yahoo_ticker).news

    news = run_with_timeout(_fetch, _NEWS_TIMEOUT_SECONDS, default=[])
    if not news:
        return []

    titles = []
    for item in news[:limit]:
        title = item.get("content", {}).get("title") or item.get("title")
        if title:
            titles.append(title)
    return titles


def fetch_recent_news(yahoo_ticker: str, limit: int = 8) -> list[dict]:
    """Like fetch_recent_headlines, but keeps the real summary/source/
    published-date fields yfinance already returns alongside each title
    instead of discarding them — added for ai.researcher, whose reports
    need more than a bare headline to synthesize real news/catalyst
    analysis from. Deliberately a NEW, separate function rather than a
    changed return shape on fetch_recent_headlines: that function's
    `list[str]` contract is depended on elsewhere (ai.portfolio_suggest.
    analyze_assets, PMEX/PSX) and changing it would be a breaking change
    to an already-working caller for no benefit to it.

    Each dict: {"title": str, "summary": str, "source": str,
    "published": str}. `summary`/`source`/`published` fall back to ""
    when yfinance's own payload doesn't carry them for a given item
    (never fabricated) — `title` is the one field an item is skipped
    for entirely if missing, same as fetch_recent_headlines. Empty list
    on any fetch failure/timeout, same as fetch_recent_headlines."""

    def _fetch() -> list:
        import yfinance as yf

        return yf.Ticker(yahoo_ticker).news

    news = run_with_timeout(_fetch, _NEWS_TIMEOUT_SECONDS, default=[])
    if not news:
        return []

    items = []
    for item in news[:limit]:
        content = item.get("content", {})
        title = content.get("title") or item.get("title")
        if not title:
            continue
        summary = content.get("summary") or ""
        source = (content.get("provider") or {}).get("displayName") or ""
        published = content.get("pubDate") or ""
        items.append({"title": title, "summary": summary, "source": source, "published": published})
    return items


def fetch_google_news(query: str, limit: int = 8) -> list[dict]:
    """A real, free, no-API-key second news source — Google News' own
    public RSS search feed. Added for ai.researcher, whose per-symbol
    news relied on Yahoo Finance alone; a real gap this closes, observed
    live: USDCHF/USDCAD/USDSEK/USDCNH all came back with zero Yahoo
    items on the same run, even though real forex commentary for all
    four genuinely exists elsewhere. Same dict shape as fetch_recent_
    news ({"title", "summary", "source", "published"}), so a caller can
    treat either source interchangeably. `summary` is always "" here —
    the RSS feed's own <description> just restates the title/source, not
    a genuine distinct summary, so it's honestly left empty rather than
    duplicated into a fake one. `query` should be a plain, human-
    readable search term (e.g. "EURUSD forex", "gold price") — a
    Yahoo-style suffixed ticker like "EURUSD=X" searches poorly here.
    Empty list on any fetch failure/timeout/malformed feed, same
    best-effort contract as fetch_recent_news — never raises."""

    def _fetch() -> list[dict]:
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET

        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=_NEWS_TIMEOUT_SECONDS) as response:
            root = ET.fromstring(response.read())

        parsed = []
        for item in root.findall("./channel/item"):
            title_el = item.find("title")
            if title_el is None or not title_el.text:
                continue
            title = title_el.text
            source_el = item.find("source")
            if source_el is not None and source_el.text:
                source = source_el.text
            elif " - " in title:
                # Google News' own de-facto title convention when no
                # separate <source> element is present: "Headline - Outlet".
                title, _, source = title.rpartition(" - ")
            else:
                source = ""
            pub_el = item.find("pubDate")
            published = pub_el.text if pub_el is not None and pub_el.text else ""
            parsed.append({"title": title, "summary": "", "source": source, "published": published})
        return parsed

    items = run_with_timeout(_fetch, _NEWS_TIMEOUT_SECONDS, default=[])
    return items[:limit]


def fetch_rss_feed(url: str, limit: int = 8) -> list[dict]:
    """A generic, real RSS 2.0 feed reader — added for ai.researcher's
    real, free finance-specialty sources (FXStreet forex news, CoinDesk
    crypto news, Investing.com commodities analysis), pulled in as
    additional per-category grounding alongside Yahoo/Google News' own
    per-symbol results. Same dict shape as fetch_recent_news/
    fetch_google_news ({"title", "summary", "source", "published"}), so
    any of the three can be treated interchangeably. `summary` is the
    feed's own real <description> when the feed genuinely provides one
    (confirmed live: FXStreet and CoinDesk do; Investing.com's
    commodities feed doesn't) — left "" rather than fabricated when
    absent. `source` is the feed's own <channel><title> (e.g.
    "CoinDesk"), the real, same attribution for every item pulled from
    one feed. Empty list on any fetch failure/timeout/malformed feed —
    never raises, same best-effort contract as the other two fetch
    functions here."""

    def _fetch() -> list[dict]:
        import urllib.request
        import xml.etree.ElementTree as ET

        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=_NEWS_TIMEOUT_SECONDS) as response:
            root = ET.fromstring(response.read())

        source = (root.findtext("./channel/title") or "").strip()
        parsed = []
        for item in root.findall("./channel/item"):
            title = item.findtext("title")
            if not title or not title.strip():
                continue
            summary = (item.findtext("description") or "").strip()
            published = (item.findtext("pubDate") or "").strip()
            parsed.append({"title": title.strip(), "summary": summary, "source": source, "published": published})
        return parsed

    items = run_with_timeout(_fetch, _NEWS_TIMEOUT_SECONDS, default=[])
    return items[:limit]
