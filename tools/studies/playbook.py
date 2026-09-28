"""Playbook studies (plan W2/W3): which ENTRY ORDER should a with-the-trend trade use - a resting limit at a reaction
level, a market order, or a breakout stop - and what makes the breakout entry better or worse?

Measured 2026-09-25 (20 symbols, up to 120k M5 bars each, entries only WITH the closed H4+D1 trend, stop 2 ATR, target
2R, 96-bar hold, spread charged): gross R per fill limit -0.03..-0.05, market +0.02..+0.03, breakout stop +0.04..+0.07,
best at ADX >= 30; a structure stop (opposite side of the 24-bar range, clipped 1.5-4 ATR) beat the fixed 2 ATR stop
(+0.086 vs +0.056 gross; cheaper-half net +0.077 vs +0.004). This module makes those measurements re-runnable so they
can be repeated on the deep cache each month and gated out of sample (tools.studies.oos).

`collect_rows` returns one row per (opportunity, entry type); `summarise` and `breakout_table` turn rows into tables.
Trades never look ahead: the higher-timeframe flags come from `closed_bucket_flag`, the range from `shift(1)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.technical import compute_adx_series
from tools.studies.common import closed_bucket_flag

HOLD, VALID, RR = 96, 96, 2.0
STOP_ATR = 2.0
RANGE_BARS = 24


def _walk(hi, lo, cl, j, entry, stop, sign, hold=HOLD):
    risk = abs(entry - stop)
    target = entry + sign * RR * risk
    end = min(j + hold, len(cl) - 1)
    for m in range(j, end + 1):
        if (lo[m] <= stop) if sign > 0 else (hi[m] >= stop):
            return -1.0
        if (hi[m] >= target) if sign > 0 else (lo[m] <= target):
            return RR
    return sign * (cl[end] - entry) / risk


def _regime(adx: float) -> str:
    return "ADX<20" if adx < 20 else "ADX20-30" if adx < 30 else "ADX>=30"


def collect_rows(
    frames: dict[str, pd.DataFrame],
    cost_by_symbol: dict[str, float] | None = None,
    step: int = 12,
    warmup: int = 600,
    aligned_only: bool = True,
) -> pd.DataFrame:
    """One row per (opportunity, entry type) with columns: sym, time, regime, adx, adx_up, typ ('market' | 'limit' |
    'stop' | 'stop_struct'), filled, gross (R per fill), net (R after spread, 0 when unfilled), tight, ahead, vr.
    `frames` = {symbol: DataFrame(Open, High, Low, Close[, Volume]) with a DatetimeIndex}; `cost_by_symbol` = round-trip
    spread as a % of price (0 when missing)."""
    rows: list[dict] = []
    for symbol, frame in frames.items():
        frame = frame.dropna(subset=["High", "Low", "Close"])
        if len(frame) < warmup + HOLD + VALID + 50:
            continue
        cost_pct = (cost_by_symbol or {}).get(symbol, 0.0)
        hi, lo, cl = (frame[c].to_numpy(dtype=float) for c in ("High", "Low", "Close"))
        op = frame["Open"].to_numpy(dtype=float) if "Open" in frame.columns else cl
        vol = frame["Volume"].to_numpy(dtype=float) if "Volume" in frame.columns else np.ones(len(cl))
        n = len(cl)
        prev = np.roll(cl, 1)
        prev[0] = cl[0]
        tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev), np.abs(lo - prev)))
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        adx = compute_adx_series(hi, lo, cl)
        f4 = closed_bucket_flag(frame["Close"], "4h").to_numpy()
        f1 = closed_bucket_flag(frame["Close"], "1D").to_numpy()
        hi_roll = pd.Series(hi).rolling(RANGE_BARS).max().shift(1).to_numpy()
        lo_roll = pd.Series(lo).rolling(RANGE_BARS).min().shift(1).to_numpy()
        v12 = pd.Series(vol).rolling(12).mean().to_numpy()
        v100 = pd.Series(vol).rolling(100).mean().shift(12).to_numpy()
        pivl = lo == pd.Series(lo).rolling(11, center=True).min().to_numpy()
        pivh = hi == pd.Series(hi).rolling(11, center=True).max().to_numpy()
        times = frame.index
        for t in range(warmup, n - VALID - HOLD - 3, step):
            x = atr[t]
            if not x or np.isnan(x) or x <= 0 or np.isnan(adx[t]) or np.isnan(f4[t]) or np.isnan(f1[t]):
                continue
            if aligned_only and f4[t] != f1[t]:
                continue
            sign = 1 if f4[t] == 1 else -1
            price = cl[t]
            base = dict(sym=symbol, time=times[t], regime=_regime(adx[t]), adx=float(adx[t]),
                        adx_up=bool(adx[t] > adx[max(t - 12, 0)]), aligned=bool(f4[t] == f1[t]), sign=sign)

            def cost_r(ref, risk):
                return cost_pct / 100 * ref / risk

            risk = STOP_ATR * x
            r = _walk(hi, lo, cl, t + 1, price, price - sign * risk, sign)
            rows.append({**base, "typ": "market", "filled": True, "gross": r, "net": r - cost_r(price, risk)})
            # MARKET with the SAME structure stop the breakout uses (opposite side of the range, clipped 1.5-4 ATR): separates
            # "which ORDER" from "which STOP" when comparing against the breakout stop.
            opp_now = lo_roll[t] if sign > 0 else hi_roll[t]
            if not np.isnan(opp_now):
                dist_m = min(max(abs(price - opp_now), 1.5 * x), 4.0 * x)
                rm = _walk(hi, lo, cl, t + 1, price, price - sign * dist_m, sign)
                rows.append({**base, "typ": "market_struct", "filled": True, "gross": rm, "net": rm - cost_r(price, dist_m)})

            # LIMIT at the nearest unbroken reaction extreme 1-2.5 ATR against the trade (5-bar pivots, >= 5 bars old)
            level = None
            lo_w = max(0, t - 300)
            idx = np.flatnonzero((pivl if sign > 0 else pivh)[lo_w:t - 5]) + lo_w
            for i in idx[::-1]:
                lv = lo[i] if sign > 0 else hi[i]
                dist = (price - lv) / x if sign > 0 else (lv - price) / x
                if dist < 1.0 or dist > 2.5:
                    continue
                if (lo[i + 1:t + 1].min() < lv) if sign > 0 else (hi[i + 1:t + 1].max() > lv):
                    continue
                level = lv
                break
            if level is not None:
                fill = next((j for j in range(t + 1, min(t + VALID, n - 1) + 1) if ((lo[j] <= level) if sign > 0 else (hi[j] >= level))), None)
                if fill is None:
                    rows.append({**base, "typ": "limit", "filled": False, "gross": 0.0, "net": 0.0})
                else:
                    r = _walk(hi, lo, cl, fill, level, level - sign * risk, sign)
                    rows.append({**base, "typ": "limit", "filled": True, "gross": r, "net": r - cost_r(level, risk)})

            # BREAKOUT STOP at the RANGE_BARS extreme in the trade direction, 0-1.5 ATR ahead
            trig = hi_roll[t] if sign > 0 else lo_roll[t]
            opp = lo_roll[t] if sign > 0 else hi_roll[t]
            ahead = (trig - price) / x if sign > 0 else (price - trig) / x
            if np.isnan(trig) or not (0.0 < ahead <= 1.5):
                continue
            rng = (hi_roll[t] - lo_roll[t]) / x
            vr = float(v12[t] / v100[t]) if v100[t] and not np.isnan(v100[t]) else float("nan")
            fill = next((j for j in range(t + 1, min(t + VALID, n - 1) + 1) if ((hi[j] >= trig) if sign > 0 else (lo[j] <= trig))), None)
            extra = dict(tight=float(rng), ahead=float(ahead), vr=vr)
            if fill is None:
                rows.append({**base, **extra, "typ": "stop", "filled": False, "gross": 0.0, "net": 0.0})
                rows.append({**base, **extra, "typ": "stop_struct", "filled": False, "gross": 0.0, "net": 0.0})
                continue
            r = _walk(hi, lo, cl, fill, trig, trig - sign * risk, sign)
            rows.append({**base, **extra, "typ": "stop", "filled": True, "gross": r, "net": r - cost_r(trig, risk)})
            dist = min(max(abs(trig - opp), 1.5 * x), 4.0 * x)
            r2 = _walk(hi, lo, cl, fill, trig, trig - sign * dist, sign)
            rows.append({**base, **extra, "typ": "stop_struct", "filled": True, "gross": r2, "net": r2 - cost_r(trig, dist)})
    return pd.DataFrame(rows)


def summarise(rows: pd.DataFrame, cheap_symbols: set[str] | None = None) -> pd.DataFrame:
    """Per (regime, typ): opportunities, fill rate, mean gross R per fill, mean net R per fill and per opportunity (+ se),
    optionally restricted to `cheap_symbols`."""
    if rows.empty:
        return pd.DataFrame()
    data = rows if cheap_symbols is None else rows[rows.sym.isin(cheap_symbols)]
    out = []
    for (regime, typ), g in data.groupby(["regime", "typ"]):
        f = g[g.filled]
        out.append(dict(
            regime=regime, typ=typ, opps=len(g), fill_rate=float(g.filled.mean()),
            gross_per_fill=float(f.gross.mean()) if len(f) else float("nan"),
            net_per_fill=float(f.net.mean()) if len(f) else float("nan"),
            net_per_opp=float(g.net.mean()), se=float(g.net.std() / np.sqrt(len(g))) if len(g) > 1 else float("nan"),
        ))
    return pd.DataFrame(out).sort_values(["regime", "typ"]).reset_index(drop=True)


def cheap_symbols(rows: pd.DataFrame) -> set[str]:
    """The cheaper half of the symbols by mean round-trip cost in R (gross - net over market fills)."""
    m = rows[(rows.typ == "market") & rows.filled]
    if m.empty:
        return set()
    cost = (m.gross - m.net).groupby(m.sym).mean()
    return set(cost[cost <= cost.median()].index)


def breakout_table(rows: pd.DataFrame, feature: str, bins: list[float], typ: str = "stop_struct", cheap: set[str] | None = None) -> pd.DataFrame:
    """Mean gross / net R of filled breakout trades by bins of a feature (tight, ahead, vr, adx) - the refinement table."""
    d = rows[(rows.typ == typ) & rows.filled].copy()
    if d.empty or feature not in d.columns:
        return pd.DataFrame()
    d["bin"] = pd.cut(d[feature], bins)
    out = []
    for b, g in d.groupby("bin", observed=True):
        c = g if cheap is None else g[g.sym.isin(cheap)]
        out.append(dict(bin=str(b), n=len(g), gross=float(g.gross.mean()), cheap_n=len(c),
                        cheap_net=float(c.net.mean()) if len(c) else float("nan")))
    return pd.DataFrame(out)


def edge_by_symbol_and_half(rows: pd.DataFrame, better: str = "stop_struct", worse: str = "limit", column: str = "net") -> dict:
    """Inputs for `oos.oos_gate`: the mean (better - worse) `column` per symbol, and the same difference in the earlier
    and later half of the sample (split at the median opportunity time)."""
    per_symbol: dict[str, float] = {}
    for sym, g in rows.groupby("sym"):
        a, b = g[g.typ == better][column], g[g.typ == worse][column]
        if len(a) >= 20 and len(b) >= 20:
            per_symbol[sym] = float(a.mean() - b.mean())
    cut = rows.time.median() if not rows.empty else None
    halves = []
    for part in (rows[rows.time <= cut], rows[rows.time > cut]) if cut is not None else ():
        a, b = part[part.typ == better][column], part[part.typ == worse][column]
        halves.append(float(a.mean() - b.mean()) if len(a) >= 30 and len(b) >= 30 else None)
    first, second = (halves + [None, None])[:2]
    return {"per_symbol": per_symbol, "first_half": first, "second_half": second}
