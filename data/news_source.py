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

# Full article page fetches are heavier than a metadata call (a real page
# load, not a small JSON/RSS payload) and are only ever triggered by a
# genuine on-demand user click (see data.symbol_news.fetch_article_text's
# own callers) rather than in bulk for every headline, so a slightly
# longer ceiling than _NEWS_TIMEOUT_SECONDS is fine here.
_ARTICLE_TIMEOUT_SECONDS = 20.0


def _strip_html(raw: str) -> str:
    """Plain, readable text from a value that may or may not contain HTML
    markup (some RSS feeds' <description> embed real HTML tags — FXStreet
    and CoinDesk both do). Falls back to the original stripped string on
    any parse failure rather than losing real content."""
    raw = raw.strip()
    if not raw or "<" not in raw:
        return raw
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(raw, "lxml").get_text(separator=" ", strip=True)
    except Exception:
        return raw


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
    "published": str, "link": str}. `summary`/`source`/`published`/`link`
    fall back to "" when yfinance's own payload doesn't carry them for a
    given item (never fabricated) — `title` is the one field an item is
    skipped for entirely if missing, same as fetch_recent_headlines.
    `link` (added 2026-09-20, direct user request for real "few
    paragraphs, like a newspaper" context instead of just a headline) is
    the real article URL, for a caller wanting the full body text via
    fetch_article_text — yfinance's own summary field is often just a
    couple of sentences, not a full article. Empty list on any fetch
    failure/timeout, same as fetch_recent_headlines."""

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
        link = (content.get("canonicalUrl") or {}).get("url") or (content.get("clickThroughUrl") or {}).get("url") or ""
        items.append({"title": title, "summary": summary, "source": source, "published": published, "link": link})
    return items


def fetch_google_news(query: str, limit: int = 8) -> list[dict]:
    """A real, free, no-API-key second news source — Google News' own
    public RSS search feed. Added for ai.researcher, whose per-symbol
    news relied on Yahoo Finance alone; a real gap this closes, observed
    live: USDCHF/USDCAD/USDSEK/USDCNH all came back with zero Yahoo
    items on the same run, even though real forex commentary for all
    four genuinely exists elsewhere. Same dict shape as fetch_recent_
    news ({"title", "summary", "source", "published", "link"}), so a
    caller can treat either source interchangeably. `summary` is always
    "" here — the RSS feed's own <description> just restates the
    title/source, not a genuine distinct summary, so it's honestly left
    empty rather than duplicated into a fake one; `link` (added
    2026-09-20) is the real article URL, so a caller wanting genuine
    "few paragraphs" context can still fetch_article_text(link) even
    when this function's own summary is empty. `query` should be a
    plain, human-readable search term (e.g. "EURUSD forex", "gold
    price") — a Yahoo-style suffixed ticker like "EURUSD=X" searches
    poorly here. Empty list on any fetch failure/timeout/malformed feed,
    same best-effort contract as fetch_recent_news — never raises."""

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
            link_el = item.find("link")
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            parsed.append({"title": title, "summary": "", "source": source, "published": published, "link": link})
        return parsed

    items = run_with_timeout(_fetch, _NEWS_TIMEOUT_SECONDS, default=[])
    return items[:limit]


def fetch_rss_feed(url: str, limit: int = 8) -> list[dict]:
    """A generic, real RSS 2.0 feed reader — added for ai.researcher's
    real, free finance-specialty sources (FXStreet forex news, CoinDesk
    crypto news, Investing.com commodities analysis), pulled in as
    additional per-category grounding alongside Yahoo/Google News' own
    per-symbol results. Same dict shape as fetch_recent_news/
    fetch_google_news ({"title", "summary", "source", "published",
    "link"}), so any of the three can be treated interchangeably.
    `summary` is the feed's own real <description> when the feed
    genuinely provides one (confirmed live: FXStreet and CoinDesk do;
    Investing.com's commodities feed doesn't) — left "" rather than
    fabricated when absent, with any HTML markup the feed embeds in it
    (FXStreet/CoinDesk both do) stripped down to plain readable text
    rather than shown as raw tags. `source` is the feed's own
    <channel><title> (e.g. "CoinDesk"), the real, same attribution for
    every item pulled from one feed. `link` (added 2026-09-20) is the
    item's real article URL, for fetch_article_text. Empty list on any
    fetch failure/timeout/malformed feed — never raises, same best-effort
    contract as the other two fetch functions here."""

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
            summary = _strip_html(item.findtext("description") or "")
            published = (item.findtext("pubDate") or "").strip()
            link = (item.findtext("link") or "").strip()
            parsed.append(
                {"title": title.strip(), "summary": summary, "source": source, "published": published, "link": link}
            )
        return parsed

    items = run_with_timeout(_fetch, _NEWS_TIMEOUT_SECONDS, default=[])
    return items[:limit]


def fetch_article_text(url: str, max_chars: int = 4000) -> str:
    """The REAL full body text of one news article — added 2026-09-20,
    direct user complaint after manually reading the News section: "i
    found them like headlines only... i was expecting few paragraphs
    that completely explain the event like the newspaper." Every source
    this module already reads (Yahoo, Google News, the category RSS
    feeds) gives at most a headline plus, sometimes, a one/two-sentence
    summary — genuinely not enough to explain what happened. This
    fetches the real linked article page and extracts its actual
    paragraph text, rather than asking a model to invent or pad out
    "a few paragraphs" from a bare headline, which would risk presenting
    fabricated detail as real reporting.

    Deliberately NOT called for every headline up front — a full page
    load + parse per item would reintroduce exactly the "loads slowly"
    problem already fixed once for the metadata-only list. This is only
    ever meant to be triggered on a genuine, individual, user-initiated
    "read this one" action.

    Heuristic extraction (no reliable universal "find the article body"
    signal exists across arbitrary publisher HTML): every <p> tag's text
    with more than 40 characters (filters out nav/ad/caption boilerplate,
    which is almost always short) is kept, in document order, joined into
    paragraphs, and capped at `max_chars`. "" on any failure (network,
    timeout, non-HTML response, no real paragraphs found) — never a
    fabricated placeholder, and never raises."""

    def _fetch() -> str:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=_ARTICLE_TIMEOUT_SECONDS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        real_paragraphs = [p for p in paragraphs if len(p) > 40]
        text = "\n\n".join(real_paragraphs)
        return text[:max_chars]

    return run_with_timeout(_fetch, _ARTICLE_TIMEOUT_SECONDS, default="")
