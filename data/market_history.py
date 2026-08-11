import pandas as pd


def fetch_price_history_ohlcv(yahoo_ticker: str, period: str = "5y") -> pd.DataFrame:
    """High/Low/Close/Volume daily history for a Yahoo ticker, oldest
    first. Empty (but correctly float-typed) DataFrame on failure, so a
    caller's .empty check and column access both behave the same way
    they would on a real empty result. High/Low feed Average True Range;
    Volume feeds the volume-trend figure; Close alone still covers every
    other stat in analysis/technical.py, so callers that only need a
    price series can use history["Close"].

    Default was 1 year (not 6 months, since a "6mo" fetch reliably
    returns only ~125 daily rows, one short of the 126-day threshold
    analysis/technical.py uses for its 6-month change figure) until it
    was confirmed live that every representative PMEX-mapped Yahoo
    ticker (CL=F, GC=F, ZC=F, ^GSPC, etc.) actually returns ~5 years of
    real daily history (~1,250+ rows) for free, the same depth PSX's own
    EOD feed already gives analysis/backtest.py's functions on the PSX
    side — 1y was this project's own conservative default, not a real
    ceiling on what's available. Every stat/window this file's callers
    compute is a TRAILING window measured from the end of the series
    (SMA20, RSI14, 20-day volatility, 60-day support/resistance, 1M/3M/6M
    change), so the extra years of history at the front don't change any
    of those existing values — they only make analysis/backtest.py's
    multi-year backtests (which need real historical episodes, not just
    a recent window) usable against PMEX instruments too."""
    import yfinance as yf

    history = yf.Ticker(yahoo_ticker).history(period=period)
    if history.empty:
        return pd.DataFrame({col: pd.Series(dtype=float) for col in ["High", "Low", "Close", "Volume"]})
    return history[["High", "Low", "Close", "Volume"]]
