import pandas as pd

from utils import run_with_timeout

# Hard wall-clock ceiling for the whole call, on top of (not instead of)
# yfinance's own internal per-request timeouts — see utils.run_with_
# timeout's own docstring for why this is needed: yfinance's internal
# cookie/crumb negotiation has been confirmed live to hang well past its
# own 10-30s per-request caps under some process contexts, and this
# function otherwise has no ceiling on that at all.
_YAHOO_HISTORY_TIMEOUT_SECONDS = 20.0


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
    empty = pd.DataFrame({col: pd.Series(dtype=float) for col in ["High", "Low", "Close", "Volume"]})

    def _fetch() -> pd.DataFrame:
        import yfinance as yf

        return yf.Ticker(yahoo_ticker).history(period=period)

    history = run_with_timeout(_fetch, _YAHOO_HISTORY_TIMEOUT_SECONDS, default=empty)
    if history.empty:
        return empty
    return history[["High", "Low", "Close", "Volume"]]
