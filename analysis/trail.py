"""Profit trail: the ONE definition shared by the Clerk's tactical signals and the fast sentinel job.

Measured 2026-09-25 (deep M5 history, 99,480 paired random trades on 20 symbols, stop 2 ATR / 2R target / 96-bar hold,
spread charged): trailing the stop 1.5 M5 ATR behind the best price once a trade is +1R gave net -0.154R vs -0.176R for
the fixed exit (better on 18 of 20 symbols) - beating breakeven-only (-0.169R) and partial+breakeven (-0.162R) - and
adding a 50% partial on top of the trail did NOT help (partial+trail -0.155R), so there is no TP1 partial. Soft
(close-based) stops and time stops were worse or neutral and are not used.

Pure: no I/O. The caller supplies the position numbers and the current M5 ATR; None means "no trail right now".
"""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass
class TrailDecision:
    r0: float | None  # the trade's ORIGINAL risk distance in price units (persisted by the caller)
    progress_r: float | None  # (price - entry) x side in units of r0
    new_stop: float | None  # the stop to ratchet to, or None


def original_risk(price_open: float, recorded_stop: float | None, live_stop: float | None, persisted_r0: float | None = None) -> float | None:
    """The trade's original risk distance: the persisted value when known (a later, tightened stop must never shrink the
    yardstick), else |entry - recorded stop| (the app's own stop for the trade), else |entry - live stop|."""
    if persisted_r0 and persisted_r0 > 0:
        return float(persisted_r0)
    basis = recorded_stop or live_stop
    return abs(price_open - basis) if basis else None


def trail_decision(
    side: str,
    price_open: float,
    price_current: float,
    live_stop: float | None,
    r0: float | None,
    atr: float | None,
    min_stop_distance_pct: float = 0.0,
) -> TrailDecision:
    """Ratchet rule: once progress >= config.CLERK_PROFIT_TRAIL_START_R, the candidate stop sits
    config.CLERK_PROFIT_TRAIL_ATR M5 ATR behind the price (pushed out to the broker's minimum stop distance when that is
    larger); it is returned only when it is tighter than the live stop by at least config.CLERK_PROFIT_TRAIL_MIN_STEP_ATR
    ATR (no micro-amends) - never a widening. Disabled by config.CLERK_PROFIT_TRAIL_ENABLED."""
    if not config.CLERK_PROFIT_TRAIL_ENABLED or not atr or atr <= 0 or not r0 or r0 <= 0:
        return TrailDecision(float(r0) if r0 and r0 > 0 else None, None, None)
    sign = 1.0 if side == "buy" else -1.0
    progress = sign * (price_current - price_open) / r0
    if progress < config.CLERK_PROFIT_TRAIL_START_R or live_stop is None:
        return TrailDecision(float(r0), progress, None)
    candidate = price_current - sign * config.CLERK_PROFIT_TRAIL_ATR * atr
    if min_stop_distance_pct > 0:
        minimum = min_stop_distance_pct / 100 * price_current * 1.02
        candidate = min(candidate, price_current - minimum) if sign > 0 else max(candidate, price_current + minimum)
    step = config.CLERK_PROFIT_TRAIL_MIN_STEP_ATR * atr
    if sign * (candidate - live_stop) >= step:
        return TrailDecision(float(r0), progress, candidate)
    return TrailDecision(float(r0), progress, None)
