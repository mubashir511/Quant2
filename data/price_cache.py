"""Deep price-history cache (position-hunting plan W2, 2026-09-25).

Why: the live pipeline fetches 6,000 M5 bars (~3 weeks) per symbol, but the broker serves far more - measured
2026-09-25: up to 200,000 M5 bars for FX/crypto (~2 years) and ~100,000 for stocks (~5 years of sessions). On 3 weeks
a per-symbol statistic is mostly noise (split-half correlation of a symbol's level hold-rate: 0.36; of its fill rate:
0.0), and every study needs out-of-sample halves. This module keeps a per-symbol/timeframe parquet cache under
config.PRICE_CACHE_DIR, extended incrementally (only the bars since the last cached one are fetched), so a symbol's
deep history is one disk read instead of a big MT5 pull.

Pure with respect to MT5: the fetch function is injectable (default data.mt5_source.fetch_mt5_price_history), and
`deep_history(..., refresh=False)` never touches the terminal. Every write is atomic (temp file + replace); a fetch
that returns nothing keeps what is cached (a broker/terminal hiccup never erases history).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Callable

import pandas as pd

import config

logger = logging.getLogger(__name__)

_BAR_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}
_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _max_bars(timeframe: str) -> int:
    return {"M5": config.DEEP_HISTORY_BARS_M5, "H1": config.DEEP_HISTORY_BARS_H1, "D1": config.DEEP_HISTORY_BARS_D1}.get(
        timeframe, config.DEEP_HISTORY_BARS_H1
    )


def cache_path(symbol: str, timeframe: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol)
    return Path(config.PRICE_CACHE_DIR) / f"{safe}_{timeframe}.parquet"


def load_cached(symbol: str, timeframe: str) -> pd.DataFrame:
    """The cached bars (oldest first, DatetimeIndex) or an empty typed frame; a corrupt file counts as empty."""
    path = cache_path(symbol, timeframe)
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS, dtype=float)
    try:
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(frame.index)
        return frame[_COLUMNS]
    except Exception:  # noqa: BLE001 - a corrupt cache must never break a run
        logger.warning("price cache %s is unreadable; treating it as empty", path, exc_info=True)
        return pd.DataFrame(columns=_COLUMNS, dtype=float)


def merge_history(cached: pd.DataFrame, fresh: pd.DataFrame, max_bars: int | None = None) -> pd.DataFrame:
    """Union of the two frames by timestamp, `fresh` winning on overlap (the last cached bar may have been
    unfinished), sorted oldest first, trimmed to the most recent `max_bars`."""
    if cached is None or cached.empty:
        merged = fresh.copy()
    elif fresh is None or fresh.empty:
        merged = cached.copy()
    else:
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    if max_bars and len(merged) > max_bars:
        merged = merged.iloc[-max_bars:]
    return merged


def _save(symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
    path = cache_path(symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp)
    os.replace(tmp, path)


def bars_needed(cached: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None, safety: int = 200) -> int | None:
    """How many recent bars to fetch to extend `cached` to now (with a safety margin for weekends/closed
    sessions and the unfinished last bar). None when nothing is cached (fetch the full initial depth)."""
    if cached.empty:
        return None
    minutes = _BAR_MINUTES.get(timeframe, 60)
    now = now or pd.Timestamp.now("UTC").tz_localize(None)  # broker clock runs ~3h ahead of UTC: the safety margin covers it
    gap_minutes = max((now - cached.index[-1]).total_seconds() / 60, 0)
    return int(gap_minutes // minutes) + safety


def update_cache(
    symbol: str,
    timeframe: str,
    fetch: Callable[[str, str, int], pd.DataFrame] | None = None,
    initial_count: int | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Extend (or create) the cache for `symbol`/`timeframe` and return the full deep history. Fetches the whole
    `initial_count` (default: the timeframe's DEEP_HISTORY_BARS_*) only when nothing is cached, otherwise just the
    bars since the last cached one. A failed/empty fetch returns whatever is cached, unchanged."""
    if fetch is None:
        from data.mt5_source import fetch_mt5_price_history as fetch  # noqa: PLW0127 - lazy: keeps this module MT5-free

    cached = load_cached(symbol, timeframe)
    limit = _max_bars(timeframe)
    need = bars_needed(cached, timeframe, now)
    count = min(initial_count or limit, limit) if need is None else min(need, limit)
    try:
        fresh = fetch(symbol, timeframe, count)
    except Exception:  # noqa: BLE001
        logger.warning("price cache: fetch failed for %s %s; keeping the cached history", symbol, timeframe, exc_info=True)
        return cached
    if fresh is None or fresh.empty:
        return cached
    fresh = fresh[_COLUMNS]
    merged = merge_history(cached, fresh, limit)
    try:
        _save(symbol, timeframe, merged)
    except Exception:  # noqa: BLE001 - e.g. disk full: still hand back the merged frame
        logger.warning("price cache: could not write %s %s", symbol, timeframe, exc_info=True)
    return merged


def deep_history(
    symbol: str, timeframe: str, refresh: bool = False, fetch: Callable[[str, str, int], pd.DataFrame] | None = None
) -> pd.DataFrame:
    """The deep history for a symbol. `refresh=False` reads the cache only (no MT5 call); `refresh=True` extends it
    first via `update_cache`."""
    return update_cache(symbol, timeframe, fetch) if refresh else load_cached(symbol, timeframe)
