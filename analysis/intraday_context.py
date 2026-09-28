"""Session/intraday context levels — previous-day high/low/close, today's
open/high/low, session VWAP, average daily range (ADR) and how much of it
today has already used (2026-09-24, intraday decision-tier upgrade).

None of this existed before: the M5/M15 decision tier needs to know where
the market has already been today (prior-day extremes are the levels an
intraday stop/target most often sits at or just beyond) and how much
movement is realistically left before a target stops being reachable in
the same session.

The session boundary is taken from the LATEST D1 BAR'S OWN TIMESTAMP on the
same broker clock as the M5 bars — never a hard-coded timezone or hour — so
it is correct whatever the broker's server offset or DST state is (this
codebase has contradictory notes about that offset, so no assumption is
made). VWAP is weighted by MT5 tick volume (the only volume MT5 exposes
here), which is a proxy, not exchange volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ADR_LOOKBACK_DAYS = 10


@dataclass
class IntradayLevels:
    last_price: float
    prev_day_high: float | None
    prev_day_low: float | None
    prev_day_close: float | None
    day_open: float | None
    day_high: float | None
    day_low: float | None
    vwap: float | None
    adr: float | None  # mean daily High-Low over the last ADR_LOOKBACK_DAYS completed days, in price
    range_used_pct: float | None  # today's High-Low as a % of ADR
    session_bars: int  # M5 bars in the session used for VWAP
    session_date: str = ""  # the D1 bar's own date — "today" only while that market is actually trading today


def compute_intraday_levels(m5_history: pd.DataFrame, d1_history: pd.DataFrame) -> IntradayLevels | None:
    """None whenever there isn't enough real data (no M5 bars, or fewer
    than two D1 bars) — never fabricated."""
    needed_d1 = {"Open", "High", "Low", "Close"}
    if (
        m5_history is None
        or m5_history.empty
        or "Close" not in m5_history.columns
        or d1_history is None
        or len(d1_history) < 2
        or not needed_d1.issubset(d1_history.columns)
    ):
        return None

    last_price = float(m5_history["Close"].iloc[-1])
    today = d1_history.iloc[-1]
    prev = d1_history.iloc[-2]

    completed = d1_history.iloc[:-1].tail(ADR_LOOKBACK_DAYS)
    ranges = (completed["High"] - completed["Low"]).dropna()
    adr = float(ranges.mean()) if len(ranges) else None
    day_high, day_low = float(today["High"]), float(today["Low"])
    range_used_pct = ((day_high - day_low) / adr * 100) if adr else None

    session_start = d1_history.index[-1]
    session = m5_history[m5_history.index >= session_start]
    vwap = None
    if not session.empty and {"High", "Low", "Close", "Volume"}.issubset(session.columns):
        volume = session["Volume"].astype(float)
        if volume.sum() > 0:
            typical = (session["High"] + session["Low"] + session["Close"]) / 3
            vwap = float((typical * volume).sum() / volume.sum())

    return IntradayLevels(
        last_price=last_price,
        prev_day_high=float(prev["High"]),
        prev_day_low=float(prev["Low"]),
        prev_day_close=float(prev["Close"]),
        day_open=float(today["Open"]),
        day_high=day_high,
        day_low=day_low,
        vwap=vwap,
        adr=adr,
        range_used_pct=range_used_pct,
        session_bars=len(session),
        session_date=str(d1_history.index[-1].date()),
    )
