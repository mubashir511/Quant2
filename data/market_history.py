import pandas as pd


def fetch_price_history_ohlcv(yahoo_ticker: str, period: str = "1y") -> pd.DataFrame:
    """High/Low/Close/Volume daily history for a Yahoo ticker, oldest
    first. Empty (but correctly float-typed) DataFrame on failure, so a
    caller's .empty check and column access both behave the same way
    they would on a real empty result. High/Low feed Average True Range;
    Volume feeds the volume-trend figure; Close alone still covers every
    other stat in analysis/technical.py, so callers that only need a
    price series can use history["Close"].

    Default is 1 year, not 6 months: a "6mo" fetch reliably returns only
    ~125 daily rows (confirmed live), one short of the 126-day threshold
    analysis/technical.py uses for its 6-month change figure — so that
    stat was effectively never populating. 1y also gives enough daily
    history for the medium-term support/resistance, market-regime, and
    ATR calculations, which all need a genuinely multi-month window."""
    import yfinance as yf

    history = yf.Ticker(yahoo_ticker).history(period=period)
    if history.empty:
        return pd.DataFrame({col: pd.Series(dtype=float) for col in ["High", "Low", "Close", "Volume"]})
    return history[["High", "Low", "Close", "Volume"]]
