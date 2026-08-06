def fetch_recent_headlines(yahoo_ticker: str, limit: int = 2) -> list[str]:
    """Up to `limit` recent headline titles for a Yahoo ticker. Empty on failure."""
    import yfinance as yf

    try:
        news = yf.Ticker(yahoo_ticker).news
    except Exception:
        return []

    if not news:
        return []

    titles = []
    for item in news[:limit]:
        title = item.get("content", {}).get("title") or item.get("title")
        if title:
            titles.append(title)
    return titles
