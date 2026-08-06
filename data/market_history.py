import pandas as pd


def fetch_price_history(yahoo_ticker: str, period: str = "1y") -> pd.Series:
    """Daily close prices for a Yahoo ticker, oldest first. Empty on failure.

    Default is 1 year, not 6 months: a "6mo" fetch reliably returns only
    ~125 daily rows (confirmed live), one short of the 126-day threshold
    analysis/technical.py uses for its 6-month change figure — so that
    stat was effectively never populating. 1y also gives enough daily
    history for the medium-term support/resistance and market-regime
    calculations, which need a genuinely multi-month window."""
    import yfinance as yf

    history = yf.Ticker(yahoo_ticker).history(period=period)
    if history.empty:
        return pd.Series(dtype=float)
    return history["Close"]
