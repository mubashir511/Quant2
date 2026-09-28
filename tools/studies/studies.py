"""The three studies behind the position-hunting rules, as functions over price frames (no MT5 inside, so they
run on real broker data through `run_all.py` and on synthetic data in the tests).

Every study takes {symbol: DataFrame(High, Low, Close; DatetimeIndex)} and returns plain dicts, so a report
(or a test) can read them without parsing prose. Random entries throughout: the point is to measure a RULE
(alignment, an exit, a cost) against a null, never to backtest a strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.studies.common import closed_bucket_flag, simulate_trade, true_range_atr


def htf_alignment_study(
    frames: dict[str, pd.DataFrame], buckets: tuple[str, str] = ("4h", "1D"), stop_atr: float = 2.0, rr: float = 2.2,
    hold: int = 24, step: int = 6, warmup: int = 600,
) -> dict:
    """Random entries on the frame's own bars, both sides, grouped by whether the CLOSED higher-timeframe trends
    (close vs SMA20 of each bucket) agree with the side: 'aligned' = both, 'counter' = neither, 'mixed' = one.
    Returns pooled gross mean R (+ standard error) per group, the per-year aligned/counter means, and how many
    symbols show aligned > counter. Gross of spread by design (the claim under test is directional)."""
    groups: dict[str, list[float]] = {"aligned": [], "mixed": [], "counter": []}
    by_year: dict[tuple[int, str], list[float]] = {}
    symbol_edge: dict[str, float] = {}
    for symbol, frame in frames.items():
        hi, lo, cl = frame["High"].to_numpy(), frame["Low"].to_numpy(), frame["Close"].to_numpy()
        close = frame["Close"]
        flags = [closed_bucket_flag(close, b).to_numpy() for b in buckets]
        atr = true_range_atr(hi, lo, cl)
        years = pd.DatetimeIndex(frame.index).year.to_numpy()
        local: dict[str, list[float]] = {"aligned": [], "counter": []}
        for i in range(warmup, len(cl) - hold - 1, step):
            x = atr[i]
            if not x or np.isnan(x) or x <= 0 or any(np.isnan(f[i]) for f in flags):
                continue
            for side in (1, -1):
                agree = [bool(f[i]) == (side == 1) for f in flags]
                group = "aligned" if all(agree) else "counter" if not any(agree) else "mixed"
                r = simulate_trade(hi, lo, cl, i, side, stop_atr * x, rr, hold)
                groups[group].append(r)
                by_year.setdefault((int(years[i]), group), []).append(r)
                if group in local:
                    local[group].append(r)
        if local["aligned"] and local["counter"]:
            symbol_edge[symbol] = float(np.mean(local["aligned"]) - np.mean(local["counter"]))
    pooled = {
        g: {"n": len(v), "mean_r": float(np.mean(v)) if v else None,
            "se": float(np.std(v) / np.sqrt(len(v))) if len(v) > 1 else None}
        for g, v in groups.items()
    }
    years_out = {}
    for year in sorted({k[0] for k in by_year}):
        aligned, counter = by_year.get((year, "aligned"), []), by_year.get((year, "counter"), [])
        if len(aligned) >= 50 and len(counter) >= 50:
            years_out[year] = {"aligned": float(np.mean(aligned)), "counter": float(np.mean(counter)), "n_aligned": len(aligned)}
    return {
        "pooled": pooled, "by_year": years_out,
        "symbols_aligned_beats_counter": sum(1 for v in symbol_edge.values() if v > 0),
        "symbols_tested": len(symbol_edge),
    }


def exit_rule_study(
    frames: dict[str, pd.DataFrame], stop_atr: float = 2.0, rr: float = 2.0, hold: int = 96, step: int = 6,
    trail_atr: float = 1.5, cost_by_symbol: dict[str, float] | None = None,
) -> dict:
    """Paired comparison of exit rules on the SAME random entries (both sides). `cost_by_symbol` = round-trip
    cost as a % of price (spread+commission), netted per trade in R. Returns mean net R, standard error and the
    number of symbols where each rule beats the fixed stop/target."""
    rules = ["base", "be@1R", "partial@1R", "trail@1R", "partial+trail@1R"]
    pooled: dict[str, list[float]] = {r: [] for r in rules}
    per_symbol: dict[str, dict[str, float]] = {}
    for symbol, frame in frames.items():
        hi, lo, cl = frame["High"].to_numpy(), frame["Low"].to_numpy(), frame["Close"].to_numpy()
        atr = true_range_atr(hi, lo, cl)
        cost_pct = (cost_by_symbol or {}).get(symbol, 0.0)
        local: dict[str, list[float]] = {r: [] for r in rules}
        for i in range(60, len(cl) - hold - 2, step):
            x = atr[i]
            if not x or np.isnan(x) or x <= 0:
                continue
            cost_r = cost_pct / 100 * cl[i] / (stop_atr * x)
            for side in (1, -1):
                for rule in rules:
                    local[rule].append(
                        simulate_trade(hi, lo, cl, i, side, stop_atr * x, rr, hold, rule, trail_atr * x) - cost_r
                    )
        for rule in rules:
            pooled[rule].extend(local[rule])
        per_symbol[symbol] = {rule: float(np.mean(v)) for rule, v in local.items() if v}
    out = {
        rule: {"n": len(v), "mean_net_r": float(np.mean(v)) if v else None,
               "se": float(np.std(v) / np.sqrt(len(v))) if len(v) > 1 else None}
        for rule, v in pooled.items()
    }
    beats = {
        rule: sum(1 for s in per_symbol.values() if rule in s and "base" in s and s[rule] > s["base"])
        for rule in rules[1:]
    }
    return {"rules": out, "beats_base_on_symbols": beats, "symbols": len(per_symbol)}


def cost_drag_by_stop_study(
    frames: dict[str, pd.DataFrame], cost_by_symbol: dict[str, float], multiples: tuple[float, ...] = (2.0, 4.0, 6.0),
) -> dict:
    """Mean round-trip cost in R for stops of k x the median M5 ATR, per symbol and pooled: cost_R =
    cost% x price / (k x ATR). Pure arithmetic on real ATR and real cost, no simulation."""
    rows: dict[str, dict[float, float]] = {}
    for symbol, frame in frames.items():
        hi, lo, cl = frame["High"].to_numpy(), frame["Low"].to_numpy(), frame["Close"].to_numpy()
        atr = true_range_atr(hi, lo, cl)
        valid = ~np.isnan(atr) & (atr > 0)
        if not valid.any():
            continue
        cost_pct = cost_by_symbol.get(symbol, 0.0)
        rows[symbol] = {k: float(np.mean(cost_pct / 100 * cl[valid] / (k * atr[valid]))) for k in multiples}
    pooled = {k: float(np.mean([r[k] for r in rows.values()])) if rows else None for k in multiples}
    return {"per_symbol": rows, "pooled": pooled}
