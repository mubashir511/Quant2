"""Symbol Behaviour Card: how a symbol has behaved over YEARS of M5 bars, shrunk toward the pool and checked for stability.

Why (2026-09-25 plan W6): "how do we understand a symbol's historical behaviour?" The books answer in one voice - Bulkowski wants
hundreds of samples per statistic, Schwager and Lo want out-of-sample stability because rules decay - and our own check agreed:
on the ~3 weeks of M5 the app used to profile a symbol on, the per-symbol hold rate correlated only 0.36 between the two halves
and the fill rate 0.0 (mostly noise). So a card is built from the deep price cache (up to ~200k M5 bars, data/price_cache.py)
and every number carries its sample size, its two half-period values and a SHRUNK value pulled toward the pooled value of all
symbols (empirical-Bayes-style: small samples move little). A tendency is only claimed when the shrunk value AND both halves agree.

What it measures (cheap, vectorised, all from completed bars, no look-ahead):
  * trending share: the share of bars with ADX >= 20 and >= 40 (how often the playbook's ADX filter would even be open);
  * range-break follow-through: after a close beyond the prior 24-bar high/low (first bar only), how often price goes +1.5 ATR
    before -1.5 ATR within 24 bars. A random moment scores about 50% by symmetry, so 50% (and the pooled value) is the comparison baseline;
  * variance ratio VR(12) of M5 returns: > 1 momentum, < 1 mean reversion (Lo & MacKinlay);
  * activity by broker hour: the three busiest hours and their share of total movement (the session profile).

It deliberately makes no claim about "strongest levels" (measured: none, Books/rule_ledger.md #29) and no cost claim (that is the
live trade-cost block). The cards are recomputed by `python -m tools.studies.symbol_cards` (monthly with the studies runner).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from analysis.technical import compute_adx_series

BREAK_BARS = 24
BREAK_TARGET_ATR = 1.5
BREAK_HORIZON = 24
VR_LAG = 12
MIN_EVENTS = 40  # fewer break events than this and the follow-through rate is not reported
SHRINK_EVENTS = 100.0  # pseudo-events of pooled evidence the shrinkage adds
SHRINK_BARS = 20000.0  # pseudo-bars for the bar-based metrics
FOLLOW_EDGE = 0.03  # a tendency needs the shrunk rate at least this far from 50% (and both halves on that side)
VR_EDGE = 0.03


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).rolling(window).mean().to_numpy()


def _break_events(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray) -> tuple[int, int]:
    """(continued, decided): first close beyond the prior BREAK_BARS extreme; +/-BREAK_TARGET_ATR reached first within the
    horizon. Stop-first when a bar reaches both. Events whose barrier is not reached at all are not counted as decided."""
    roll_high = pd.Series(high).rolling(BREAK_BARS).max().shift(1).to_numpy()
    roll_low = pd.Series(low).rolling(BREAK_BARS).min().shift(1).to_numpy()
    n = len(close)
    continued = decided = 0
    up_prev = dn_prev = False
    for i in range(BREAK_BARS + 1, n - BREAK_HORIZON):
        a = atr[i]
        up = bool(close[i] > roll_high[i])
        dn = bool(close[i] < roll_low[i])
        fire_up, fire_dn = up and not up_prev, dn and not dn_prev
        up_prev, dn_prev = up, dn
        if not (fire_up or fire_dn) or not a or np.isnan(a) or a <= 0:
            continue
        sign = 1.0 if fire_up else -1.0
        entry, barrier = close[i], BREAK_TARGET_ATR * a
        for j in range(i + 1, i + 1 + BREAK_HORIZON):
            adverse = sign * (entry - (low[j] if sign > 0 else high[j]))
            favourable = sign * ((high[j] if sign > 0 else low[j]) - entry)
            if adverse >= barrier:
                decided += 1
                break
            if favourable >= barrier:
                decided += 1
                continued += 1
                break
    return continued, decided


def _variance_ratio(close: np.ndarray, lag: int = VR_LAG) -> float | None:
    logc = np.log(close[close > 0])
    if len(logc) < lag * 50:
        return None
    r1 = np.diff(logc)
    rk = logc[lag:] - logc[:-lag]
    v1 = r1.var()
    return float(rk.var() / (lag * v1)) if v1 > 0 else None


def _half_metrics(frame: pd.DataFrame) -> dict:
    high, low, close = (frame[c].to_numpy(dtype=float) for c in ("High", "Low", "Close"))
    adx = compute_adx_series(high, low, close)
    valid = adx[~np.isnan(adx)][60:]
    atr = _atr(high, low, close)
    cont, decided = _break_events(high, low, close, atr)
    return {
        "bars": int(len(frame)),
        "adx20": float((valid >= 20).mean()) if len(valid) else None,
        "adx40": float((valid >= 40).mean()) if len(valid) else None,
        "break_continued": cont,
        "break_decided": decided,
        "vr12": _variance_ratio(close),
    }


def _hour_profile(frame: pd.DataFrame) -> dict | None:
    if not isinstance(frame.index, pd.DatetimeIndex) or len(frame) < 2000:
        return None
    span = (frame["High"] - frame["Low"]).to_numpy(dtype=float)
    per_hour = pd.Series(span, index=frame.index.hour).groupby(level=0).sum()
    total = float(per_hour.sum())
    if total <= 0:
        return None
    top = per_hour.sort_values(ascending=False).head(3)
    return {"hours": [int(h) for h in top.index], "share": float(top.sum() / total)}


def compute_symbol_metrics(m5: pd.DataFrame | None) -> dict | None:
    """Raw per-symbol metrics (whole sample + two halves + hour profile); None without enough completed bars."""
    if m5 is None or not {"High", "Low", "Close"}.issubset(m5.columns):
        return None
    frame = m5.dropna(subset=["High", "Low", "Close"])
    if len(frame) < config.SYMBOL_CARD_MIN_BARS:
        return None
    frame = frame.iloc[:-1]  # the forming bar is not evidence
    mid = len(frame) // 2
    days = None
    if isinstance(frame.index, pd.DatetimeIndex):
        days = int((frame.index[-1] - frame.index[0]).days)
    return {
        "whole": _half_metrics(frame),
        "first": _half_metrics(frame.iloc[:mid]),
        "second": _half_metrics(frame.iloc[mid:]),
        "hours": _hour_profile(frame),
        "days": days,
    }


def _shrink(value: float | None, weight: float, pooled: float | None, k: float) -> float | None:
    if value is None:
        return pooled
    if pooled is None:
        return value
    return (value * weight + pooled * k) / (weight + k)


def _rate(cont: int, decided: int) -> float | None:
    return cont / decided if decided >= MIN_EVENTS else None


def build_cards(raw: dict[str, dict]) -> dict:
    """Cards for every symbol in `raw` ({symbol: compute_symbol_metrics(...)}), shrunk toward the pooled values of all of them."""
    raw = {s: m for s, m in raw.items() if m}
    pooled_cont = sum(m["whole"]["break_continued"] for m in raw.values())
    pooled_dec = sum(m["whole"]["break_decided"] for m in raw.values())
    pooled_rate = pooled_cont / pooled_dec if pooled_dec else None

    def pooled_mean(key: str) -> float | None:
        vals = [m["whole"][key] for m in raw.values() if m["whole"].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    pooled = {"break_rate": pooled_rate, "adx20": pooled_mean("adx20"), "adx40": pooled_mean("adx40"), "vr12": pooled_mean("vr12")}
    cards: dict = {}
    for symbol, m in raw.items():
        whole, a, b = m["whole"], m["first"], m["second"]
        rate = _rate(whole["break_continued"], whole["break_decided"])
        ra, rb = _rate(a["break_continued"], a["break_decided"]), _rate(b["break_continued"], b["break_decided"])
        s_rate = _shrink(rate, whole["break_decided"], pooled_rate, SHRINK_EVENTS)
        follow = "no reliable tendency"
        if s_rate is not None and ra is not None and rb is not None:
            if s_rate >= 0.5 + FOLLOW_EDGE and ra > 0.5 and rb > 0.5:
                follow = "range breaks tend to CONTINUE"
            elif s_rate <= 0.5 - FOLLOW_EDGE and ra < 0.5 and rb < 0.5:
                follow = "range breaks tend to FAIL (fade-prone)"
        s_vr = _shrink(whole["vr12"], whole["bars"], pooled["vr12"], SHRINK_BARS)
        va, vb = a["vr12"], b["vr12"]
        vr_read = "neutral"
        if s_vr is not None and va is not None and vb is not None:
            if s_vr >= 1 + VR_EDGE and va > 1 and vb > 1:
                vr_read = "momentum (returns trend)"
            elif s_vr <= 1 - VR_EDGE and va < 1 and vb < 1:
                vr_read = "mean-reverting (returns snap back)"
        cards[symbol] = {
            "bars": whole["bars"], "days": m.get("days"),
            "adx20": _shrink(whole["adx20"], whole["bars"], pooled["adx20"], SHRINK_BARS),
            "adx40": _shrink(whole["adx40"], whole["bars"], pooled["adx40"], SHRINK_BARS),
            "break_rate": s_rate, "break_rate_raw": rate, "break_events": whole["break_decided"],
            "break_halves": [ra, rb], "follow": follow,
            "vr12": s_vr, "vr_halves": [va, vb], "vr_read": vr_read,
            "hours": m.get("hours"),
        }
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "pooled": pooled, "symbols": cards}


def save_cards(cards: dict, path: str | None = None) -> None:
    target = Path(path or config.SYMBOL_CARD_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(cards, indent=2))
    os.replace(tmp, target)


def load_cards(path: str | None = None) -> dict:
    target = Path(path or config.SYMBOL_CARD_FILE)
    try:
        data = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def format_card(symbol: str, cards: dict | None = None) -> str | None:
    """One prompt line, or None when there is no card for the symbol (never a made-up one)."""
    cards = load_cards() if cards is None else cards
    card = (cards.get("symbols") or {}).get(symbol)
    if not card:
        return None
    pooled = cards.get("pooled") or {}

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.0f}%"

    parts = [
        f"{card['bars']:,} M5 bars" + (f" = {card['days']} days" if card.get("days") else ""),
        f"trending share ADX>=20 {pct(card.get('adx20'))} / ADX>=40 {pct(card.get('adx40'))} (pool {pct(pooled.get('adx20'))} / {pct(pooled.get('adx40'))})",
    ]
    if card.get("break_rate") is not None:
        halves = "/".join("n/a" if h is None else f"{h * 100:.0f}%" for h in card.get("break_halves", [None, None]))
        parts.append(
            f"24-bar range-break follow-through {pct(card['break_rate'])} shrunk toward the pool {pct(pooled.get('break_rate'))} "
            f"(n={card['break_events']:,} decided breaks, halves {halves}; 50% = a coin flip) -> {card['follow']}"
        )
    if card.get("vr12") is not None:
        parts.append(f"12-bar variance ratio {card['vr12']:.2f} -> {card['vr_read']}")
    hours = card.get("hours")
    if hours:
        parts.append(f"busiest broker hours {', '.join(f'{h:02d}' for h in hours['hours'])} ({hours['share'] * 100:.0f}% of all movement)")
    stamp = (cards.get("generated_utc") or "")[:10]
    stale = ""
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(cards["generated_utc"])).days
        if age > config.SYMBOL_CARD_STALE_DAYS:
            stale = f" STALE ({age} days old - regimes drift, treat as history not prediction)"
    except (KeyError, TypeError, ValueError):
        pass
    return f"  Symbol behaviour card ({symbol}, computed {stamp}{stale}; shrunk + split-half checked): " + "; ".join(parts)
