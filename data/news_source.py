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
