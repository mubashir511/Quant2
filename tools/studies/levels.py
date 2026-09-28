"""Level-selection study (plan W2/W4): which properties of a support/resistance candidate predict that price TURNS there?

The question behind "which is the strongest reversal level": the books rank levels by touches, recency, volume, polarity,
retracement, round numbers and higher-timeframe confluence; our 3-week test found hold-after-touch of 54-60% for every
type, i.e. no obvious winner, and level identity barely distinguishable from random levels in R. On deep history this
study measures each feature against RANDOM levels drawn at the same distances, in two time halves.

For sampled moments t (structure computed from bars up to t only) every candidate is listed with its features and its
FORWARD outcome over 96 bars:
  touched  price came within 0.1 ATR of the level (a limit there would have filled)
  held     after the first touch price travelled >= 1 ATR the other way before trading 0.75 ATR THROUGH the level
Candidate kinds: zone (clustered S/R band edge), reaction (unbroken 5-bar swing extreme - the SOLUSD case: the levels price
actually turned from), fib (38.2/50/61.8 of the last leg when it ran with the side), range (24-bar extreme), random.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis.chart_structure import compute_chart_structure, find_swing_points
from analysis.timeframe_profiles import M5_PROFILE

FWD, TOUCH_TOL, BOUNCE, BREAK = 96, 0.10, 1.0, 0.75


def outcome(hi, lo, i, level, side, atr, fwd=FWD):
    """(touched, held, bounce_atr) for a `level` on `side` ('sup' below the price / 'res' above) after bar `i`."""
    end = min(i + fwd, len(hi) - 1)
    for j in range(i + 1, end + 1):
        touch = (lo[j] <= level + TOUCH_TOL * atr) if side == "sup" else (hi[j] >= level - TOUCH_TOL * atr)
        if not touch:
            continue
        best = 0.0
        for k in range(j, min(j + 48, len(hi) - 1) + 1):
            if side == "sup":
                if lo[k] <= level - BREAK * atr:
                    return True, best >= BOUNCE, best
                best = max(best, (hi[k] - level) / atr)
            else:
                if hi[k] >= level + BREAK * atr:
                    return True, best >= BOUNCE, best
                best = max(best, (level - lo[k]) / atr)
            if best >= BOUNCE:
                return True, True, best
        return True, best >= BOUNCE, best
    return False, False, 0.0


def round_step(price: float) -> float:
    return 10 ** (math.floor(math.log10(abs(price))) - 1) if price else 1.0


def is_round(level: float, atr: float, price: float) -> bool:
    step = round_step(price)
    return abs(level - round(level / step) * step) <= 0.15 * atr


def _h1_swings(window: pd.DataFrame) -> list[float]:
    """Price of every H1 fractal swing (3-bar) in the M5 window - the higher-timeframe levels an M5 level may coincide with."""
    if len(window) < 24 * 12:
        return []
    h1 = window.resample("1h").agg({"High": "max", "Low": "min"}).dropna()
    if len(h1) < 8:
        return []
    highs, lows = find_swing_points(h1.reset_index(drop=True), window=3)
    return [p.price for p in highs] + [p.price for p in lows]


def collect_level_rows(
    frames: dict[str, pd.DataFrame], step: int = 1500, warmup: int = 1300, seed: int = 7, shifts: tuple[float, ...] = (0.0,)
) -> pd.DataFrame:
    """One row per candidate (and per shift): sym, time, kind, side, dist (ATR, of the FINAL level), shift, touches, weighted,
    pool, round, h1_conf, bars_ago, vol_at (tick volume at a reaction bar vs its 50-bar mean), touched, held, bounce.
    `shifts` (ATRs) also evaluate each REAL level moved further from the price - an order placed just BEYOND the visible
    level (Schwager: stops cluster beyond ranges, so the poke-through is where the turn often happens)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for symbol, frame in frames.items():
        frame = frame.dropna(subset=["High", "Low", "Close"])
        n = len(frame)
        if n < warmup + FWD + 100:
            continue
        hi, lo, cl = (frame[c].to_numpy(dtype=float) for c in ("High", "Low", "Close"))
        vol = frame["Volume"].to_numpy(dtype=float) if "Volume" in frame.columns else np.ones(n)
        prev = np.roll(cl, 1)
        prev[0] = cl[0]
        atr_s = pd.Series(np.maximum(hi - lo, np.maximum(np.abs(hi - prev), np.abs(lo - prev)))).rolling(14).mean().to_numpy()
        times = frame.index
        for t in range(warmup, n - FWD - 60, step):
            atr = atr_s[t]
            if not atr or np.isnan(atr):
                continue
            window = frame.iloc[t - 1200:t + 1]
            try:
                structure = compute_chart_structure(window.reset_index(drop=True), profile=M5_PROFILE)
            except Exception:  # noqa: BLE001 - a degenerate window is skipped, never guessed
                continue
            price = cl[t]
            h1 = _h1_swings(window)
            cands: list[tuple] = []  # (kind, side, level, meta)
            if structure.sr_levels is not None:
                for lv in structure.sr_levels.support_levels[:4]:
                    if lv.high is not None and lv.high < price:
                        cands.append(("zone", "sup", lv.high, dict(touches=lv.touches, weighted=lv.weighted_score, pool=lv.is_liquidity_pool)))
                for lv in structure.sr_levels.resistance_levels[:4]:
                    if lv.low is not None and lv.low > price:
                        cands.append(("zone", "res", lv.low, dict(touches=lv.touches, weighted=lv.weighted_score, pool=lv.is_liquidity_pool)))
            tail = window.tail(300).reset_index(drop=True)
            swing_highs, swing_lows = find_swing_points(tail)
            last = len(tail) - 1
            for p in sorted([p for p in swing_lows if p.price < price and tail["Low"].iloc[p.index:].min() >= p.price - 1e-12], key=lambda q: -q.price)[:2]:
                base = float(vol[t - last + p.index - 50:t - last + p.index].mean()) if t - last + p.index - 50 > 0 else 0.0
                cands.append(("reaction", "sup", p.price, dict(bars_ago=last - p.index, vol_at=(vol[t - last + p.index] / base) if base > 0 else float("nan"))))
            for p in sorted([p for p in swing_highs if p.price > price and tail["High"].iloc[p.index:].max() <= p.price + 1e-12], key=lambda q: q.price)[:2]:
                base = float(vol[t - last + p.index - 50:t - last + p.index].mean()) if t - last + p.index - 50 > 0 else 0.0
                cands.append(("reaction", "res", p.price, dict(bars_ago=last - p.index, vol_at=(vol[t - last + p.index] / base) if base > 0 else float("nan"))))
            fib = structure.fibonacci
            if fib is not None and fib.swing_high > fib.swing_low:
                span = fib.swing_high - fib.swing_low
                for name, ratio in (("fib382", 0.382), ("fib50", 0.5), ("fib618", 0.618)):
                    if fib.high_is_more_recent and fib.swing_high - ratio * span < price:
                        cands.append(("fib", "sup", fib.swing_high - ratio * span, dict(fib=name)))
                    elif not fib.high_is_more_recent and fib.swing_low + ratio * span > price:
                        cands.append(("fib", "res", fib.swing_low + ratio * span, dict(fib=name)))
            r_hi, r_lo = float(hi[t - 24:t].max()), float(lo[t - 24:t].min())
            if r_lo < price:
                cands.append(("range", "sup", r_lo, {}))
            if r_hi > price:
                cands.append(("range", "res", r_hi, {}))
            for side in ("sup", "res"):
                for _ in range(2):
                    d = float(rng.uniform(0.5, 4.0))
                    cands.append(("random", side, price - d * atr if side == "sup" else price + d * atr, {}))
            for kind, side, level0, meta in cands:
                for shift in (shifts if kind != "random" else (0.0,)):
                    level = level0 - shift * atr if side == "sup" else level0 + shift * atr
                    dist = abs(price - level) / atr
                    if dist < 0.3 or dist > 5.0:
                        continue
                    touched, held, bounce = outcome(hi, lo, t, level, side, atr)
                    rows.append(dict(
                        sym=symbol, time=times[t], kind=kind, side=side, dist=float(dist), shift=float(shift), touches=meta.get("touches"), weighted=meta.get("weighted"),
                        pool=meta.get("pool"), bars_ago=meta.get("bars_ago"), vol_at=meta.get("vol_at"), fib=meta.get("fib"),
                        round=bool(is_round(level, atr, price)), h1_conf=bool(any(abs(level - x) <= 0.5 * atr for x in h1)),
                        touched=touched, held=held, bounce=float(bounce),
                    ))
    return pd.DataFrame(rows)


def hold_table(rows: pd.DataFrame, by: str, bins: list[float] | None = None, kinds: tuple[str, ...] | None = None, min_n: int = 40) -> pd.DataFrame:
    """Touch rate, hold-after-touch and fill-and-hold for real candidates grouped by feature `by` (optionally binned),
    next to the RANDOM candidates' rates at the same distances (weighted by the real rows' distance mix)."""
    real = rows[rows.kind != "random"]
    if kinds:
        real = real[real.kind.isin(kinds)]
    rand = rows[rows.kind == "random"]
    real = real.copy()
    real["g"] = pd.cut(real[by], bins) if bins is not None else real[by]
    out = []
    for g, part in real.groupby("g", observed=True):
        if len(part) < min_n:
            continue
        touched = part[part.touched]
        # random baseline restricted to the same distance range as this group's rows
        lo_d, hi_d = part.dist.quantile(0.1), part.dist.quantile(0.9)
        r = rand[(rand.dist >= lo_d) & (rand.dist <= hi_d)]
        rt = r[r.touched]
        out.append(dict(
            group=str(g), n=len(part), touch=float(part.touched.mean()),
            hold=float(touched.held.mean()) if len(touched) else float("nan"),
            fill_hold=float((part.touched & part.held).mean()),
            rand_hold=float(rt.held.mean()) if len(rt) else float("nan"),
            rand_fill_hold=float((r.touched & r.held).mean()) if len(r) else float("nan"),
        ))
    return pd.DataFrame(out)


def kind_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Per candidate kind: n, touch rate, hold-after-touch, fill-and-hold and mean distance."""
    out = []
    for kind, part in rows.groupby("kind"):
        touched = part[part.touched]
        out.append(dict(kind=kind, n=len(part), touch=float(part.touched.mean()), hold=float(touched.held.mean()) if len(touched) else float("nan"),
                        fill_hold=float((part.touched & part.held).mean()), mean_dist=float(part.dist.mean())))
    return pd.DataFrame(out).sort_values("kind").reset_index(drop=True)


def real_minus_random_by_half(rows: pd.DataFrame, kind: str, metric: str = "fill_hold") -> dict:
    """Inputs for tools.studies.oos.oos_gate: (real `kind` - random) in fill-and-hold, per symbol and per time half,
    with the random baseline matched by distance bucket (0.3-1.5, 1.5-3, 3-5 ATR)."""
    bins = [0.3, 1.5, 3.0, 5.0]
    d = rows.copy()
    d["bucket"] = pd.cut(d.dist, bins)
    d["fh"] = (d.touched & d.held).astype(float)

    def diff(part: pd.DataFrame) -> float | None:
        real, rand = part[part.kind == kind], part[part.kind == "random"]
        if len(real) < 20 or len(rand) < 20:
            return None
        rand_by_bucket = rand.groupby("bucket", observed=True).fh.mean()
        expected = real.bucket.map(rand_by_bucket).astype(float)
        if expected.isna().all():
            return None
        return float((real.fh - expected.fillna(expected.mean())).mean())

    per_symbol = {s: v for s, g in d.groupby("sym") if (v := diff(g)) is not None}
    cut = d.time.median()
    return {"per_symbol": per_symbol, "first_half": diff(d[d.time <= cut]), "second_half": diff(d[d.time > cut])}
