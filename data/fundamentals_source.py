from dataclasses import dataclass

from utils import run_with_timeout

# Same reasoning as data/news_source.py's own _NEWS_TIMEOUT_SECONDS: a
# yfinance call (here, .info and .calendar) can hang well past its own
# internal timeouts under some process contexts — this hard wall-clock
# ceiling turns that into an honest "unavailable" instead of hanging the
# caller. See utils.run_with_timeout's own docstring for the full
# incident this pattern already guards against elsewhere in this project.
_FUNDAMENTALS_TIMEOUT_SECONDS = 15.0


@dataclass
class EquityFundamentals:
    """Real, free equity fundamentals — added for ai.researcher's
    equity-symbol reports (NVDA/INTC/AMD/MSFT), which previously had zero
    valuation/analyst/earnings context despite it being genuinely free
    via yfinance (confirmed live 2026-09-15: NVDA returned a real 58-
    analyst "strong_buy" consensus, a real $327 price target, and a real
    upcoming earnings date with EPS estimates). Every field is
    independently optional — yfinance's own coverage varies by ticker
    (thin analyst coverage, no scheduled earnings date yet, etc.) — never
    fabricated to fill a gap; a missing field is just left off."""

    analyst_target_mean_price: float | None
    analyst_recommendation: str | None  # e.g. "strong_buy", "buy", "hold", "sell", "strong_sell"
    analyst_count: int | None
    forward_pe: float | None
    next_earnings_date: str | None  # ISO date string, e.g. "2026-11-18"
    earnings_eps_estimate: float | None


def fetch_equity_fundamentals(yahoo_ticker: str) -> EquityFundamentals | None:
    """Real analyst consensus/price target/forward P/E plus the next real
    scheduled earnings date + EPS estimate, via yfinance's own `.info`/
    `.calendar`. None on any fetch failure/timeout — never raises, same
    best-effort contract as data.news_source's own fetch functions."""

    def _fetch() -> EquityFundamentals:
        import yfinance as yf

        ticker = yf.Ticker(yahoo_ticker)
        info = ticker.info or {}
        calendar = ticker.calendar or {}

        earnings_dates = calendar.get("Earnings Date") or []
        next_earnings_date = str(earnings_dates[0]) if earnings_dates else None

        return EquityFundamentals(
            analyst_target_mean_price=info.get("targetMeanPrice"),
            analyst_recommendation=info.get("recommendationKey"),
            analyst_count=info.get("numberOfAnalystOpinions"),
            forward_pe=info.get("forwardPE"),
            next_earnings_date=next_earnings_date,
            earnings_eps_estimate=calendar.get("Earnings Average"),
        )

    return run_with_timeout(_fetch, _FUNDAMENTALS_TIMEOUT_SECONDS, default=None)
