"""Shared helpers for the re-runnable validation studies (position-hunting review, 2026-09-25).

The one rule these helpers exist to enforce: a higher-timeframe flag may be used at a fine-timeframe bar ONLY
once the higher-timeframe bar has CLOSED. My first alignment study broke this — `resample().last()` labels a
bucket by its START, so reindexing that series onto the fine bars handed every bar inside a bucket the bucket's
own FUTURE close — and produced a fake +0.19R vs -0.14R "edge" that dropped to the real +0.05R vs 0 once
fixed. `closed_bucket_flag` is the ONE way the studies build such a flag, and tests/test_studies.py fails if it
ever leaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.htf_flags import closed_bucket_flag  # noqa: F401 - re-exported: the studies' one way to build a flag


def true_range_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """Rolling-mean ATR over `window` bars (same definition analysis.backtest._rolling_atr uses)."""
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).rolling(window).mean().to_numpy()


def simulate_trade(
    hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, i: int, side: int, stop_dist: float, rr: float, hold: int,
    exit_rule: str = "base", trail_atr: float | None = None,
) -> float:
    """One bar-by-bar trade entered at close[i]; stop-first when a bar touches both; a trade unresolved after
    `hold` bars is marked to market. R is measured in units of `stop_dist`. `exit_rule`:
      base          fixed stop and target
      be@1R         stop moves to entry once +1R has been reached
      partial@1R    half is banked at +1R and the stop of the rest moves to entry
      trail@1R      once +1R has been reached the stop trails `trail_atr` (price units) behind the best price
      partial+trail@1R  half is banked at +1R, the stop of the rest moves to entry and then trails like trail@1R"""
    entry = cl[i]
    stop = entry - side * stop_dist
    target = entry + side * rr * stop_dist
    best = entry
    banked, remaining, armed = 0.0, 1.0, False
    for j in range(i + 1, min(i + hold, len(cl) - 1) + 1):
        h, l = hi[j], lo[j]
        best = max(best, h) if side == 1 else min(best, l)
        if (l <= stop) if side == 1 else (h >= stop):
            return banked + remaining * side * (stop - entry) / stop_dist
        if (h >= target) if side == 1 else (l <= target):
            return banked + remaining * rr
        progress = side * (best - entry) / stop_dist
        if progress >= 1.0:
            if exit_rule == "be@1R" and not armed:
                stop, armed = entry, True
            elif exit_rule == "partial@1R" and not armed:
                banked, remaining, stop, armed = banked + 0.5, 0.5, entry, True
            elif exit_rule == "trail@1R" and trail_atr:
                trailed = best - side * trail_atr
                stop = max(stop, trailed) if side == 1 else min(stop, trailed)
            elif exit_rule == "partial+trail@1R" and trail_atr:
                if not armed:
                    banked, remaining, stop, armed = banked + 0.5, 0.5, entry, True
                trailed = best - side * trail_atr
                stop = max(stop, trailed) if side == 1 else min(stop, trailed)
    end = min(i + hold, len(cl) - 1)
    return banked + remaining * side * (cl[end] - entry) / stop_dist
