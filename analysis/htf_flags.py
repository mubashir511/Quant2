"""Higher-timeframe trend flags built ONLY from higher-timeframe bars that have already CLOSED.

The one rule these helpers exist to enforce (position-hunting review, 2026-09-25): a higher-timeframe flag may be used at a
fine-timeframe bar only once the higher-timeframe bar has closed. The first alignment study broke it - `resample().last()`
labels a bucket by its START, so reindexing that series onto the fine bars handed every bar inside a bucket the bucket's own
FUTURE close - and produced a fake +0.19R vs -0.14R "edge" that dropped to the real +0.05R vs 0 once fixed.
`closed_bucket_flag` is the ONE way the studies AND the live Position Hunter build such a flag (tests/test_studies.py fails
if it ever leaks), so the tier the hunter reads is the tier the alignment numbers were measured on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def closed_bucket_flag(close: pd.Series, bucket: str, sma_window: int = 20) -> pd.Series:
    """Boolean trend flag (bucket close > SMA(sma_window) of bucket closes) for every bar of `close`, using ONLY
    buckets that have fully closed by that bar's own timestamp. `close` must have a DatetimeIndex marking each
    fine bar's OPEN time. NaN-safe: bars before the first computable flag are NaN (as object/float), never a
    guess.

    A bucket labelled T (start) covers [T, T + bucket) and its close is knowable at T + bucket, so its flag is
    re-indexed to T + bucket and forward-filled: a fine bar opening at time t sees the last bucket whose end is
    <= t, never the bucket t itself sits in."""
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("closed_bucket_flag needs a DatetimeIndex")
    buckets = close.resample(bucket).last().dropna()
    flag = (buckets > buckets.rolling(sma_window).mean()).astype(float)
    flag[buckets.rolling(sma_window).mean().isna()] = np.nan
    known_at = flag.copy()
    known_at.index = known_at.index + pd.tseries.frequencies.to_offset(bucket)
    return known_at.reindex(close.index, method="ffill")


def closed_htf_directions(m5_close: pd.Series | None) -> dict[str, str | None]:
    """{"h4": "up"/"down"/None, "d1": ...} at the LAST M5 bar, from closed H4 and D1 buckets only (close vs the SMA20 of
    bucket closes - exactly the flag the alignment study measured). None when the history is too short or has no
    DatetimeIndex."""
    out: dict[str, str | None] = {"h4": None, "d1": None}
    if m5_close is None or len(m5_close) < 2 or not isinstance(m5_close.index, pd.DatetimeIndex):
        return out
    for key, bucket in (("h4", "4h"), ("d1", "1D")):
        value = closed_bucket_flag(m5_close, bucket).iloc[-1]
        out[key] = None if pd.isna(value) else ("up" if value >= 0.5 else "down")
    return out
