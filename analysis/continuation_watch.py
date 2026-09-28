"""Continuation Watch: PHASE 1 ONLY — the deterministic filter itself, calibrated and tested against every
real closed trade this account has ever had, before it is wired into anything live.

Why (2026-09-28, direct user request): two winning trades (SOLUSD, closed 2026-09-25; XAGUSD, closed
2026-09-28) both kept running hard in the same direction after their own exit — the trade's own discipline
worked exactly as designed, but the system had no mechanism to even LOOK at a symbol again once its trade
closed. The user's own plan, in order: (1) build a strong deterministic checklist and test it against every
winning trade this account has ever had — it must pick out ONLY those two, nothing else; (2) only once that
test passes, wire it into the 7-point live design (2-3 M5 candles to decide, escalate to a model when the
local one is weak, log every event to the trade journal, reuse the EXISTING sizing/veto/kill-switch
pipeline — never a second order path, LOG-ONLY trial before it is ever allowed to act). This module is (1)
only — a pure, no-I/O checklist plus the calibration this file's own docstring promised.

Calibration (this account's ENTIRE closed-trade history — 6 positions, ever, as of 2026-09-28; see
tests/fixtures/continuation_watch_real_trades.json for the real M5 bars used):

  symbol    result              bars after close checked?
  SOLUSD    WIN, +47.15         3 real M5 bars arrived within 15 min -> price ran +4.5x ATR further -> PASS
  XAGUSD    WIN, +49.70         3 real M5 bars arrived within 15 min -> price ran +21x ATR further -> PASS
  WHEAT.c   win, +6.00 (tiny)   market didn't trade again for ~4.5h (an exchange session gap) -> the
                                15-minute window closes with 0 real bars -> REJECTED on timeliness alone,
                                never even reaches the price check (when the market DID reopen, price had
                                reversed -5x ATR against the trade — a real save, not a miss)
  EURUSD    win, +0.64 (tiny)   market didn't trade again for ~45h (the weekend) -> REJECTED on
                                timeliness. Real finding from calibrating this: without a timeliness gate,
                                the eventual post-weekend price (a stale, unrelated Monday session) actually
                                LOOKS like a +11x ATR continuation in isolation — a genuine false-positive
                                the first version of this filter would have produced. The timeliness gate
                                exists specifically because of this discovery, not as a theoretical caution.
  MSFT      loss, -62.82        never evaluated at all — this filter only ever runs after a WIN (the
                                caller's job, not this module's; see the module docstring's own "why")
  SOLUSD    loss, -16.42        same as above — losers are never fed to this filter

Two objective, deterministic gates, both must pass (same "hard veto only, nothing fuzzy" style as
analysis.position_hunter — no score, no "probably"):
  1. TIMELY: at least `min_bars` real M5 bars must have actually traded within `max_wait_minutes` of the
     close. A closed/thin market that hasn't produced them yet is not a "no" — the caller re-checks on its
     own next pass — but once the deadline passes without enough bars, it IS a permanent no for this trade
     (see `deadline_passed`, which the caller uses to stop watching this symbol rather than re-check
     forever on a market that already reopened onto an unrelated new session).
  2. REAL MOVE: measured against the bars available at (or up to) that deadline, price has continued in the
     trade's OWN direction by at least `min_atr_move` times the ATR the trade closed against — the same
     yardstick (ATR, not raw points) the rest of this project already uses for "a real step" (the profit
     trail, the tactical stop candidates), so this filter's language matches everything else it sits beside.

Deliberately NOT in this module yet (phase 2, only after this phase is approved): the cost-drag/correlation
vetoes, the local-model-then-OpenRouter escalation, the trade-journal write, or anything that touches an
order. This file only answers one question: "did this specific win keep running, fast, right after it
closed?" — a fact, not a decision to trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

import config


@dataclass
class ContinuationVerdict:
    passed: bool
    reason: str
    bars_seen: int  # real M5 bars observed within the window, whatever the outcome
    minutes_since_close: float | None  # how long after the close the LAST bar used actually traded
    move_atr: float | None  # the continuation move, in ATR — None when there weren't enough bars to measure it
    deadline_passed: bool  # True once max_wait_minutes has elapsed — the caller's cue to stop re-checking this trade


def evaluate_continuation(
    side: str,
    exit_price: float,
    exit_time: datetime,
    atr_at_close: float | None,
    bars_since_close: pd.DataFrame,
    now: datetime,
    min_bars: int = config.CONTINUATION_MIN_BARS,
    max_wait_minutes: float = config.CONTINUATION_MAX_WAIT_MINUTES,
    min_atr_move: float = config.CONTINUATION_MIN_ATR_MOVE,
) -> ContinuationVerdict:
    """`bars_since_close` — real M5 OHLC bars (datetime index) strictly AFTER `exit_time`, however many the
    caller has fetched (this function does its own windowing; passing more than `max_wait_minutes` worth is
    fine and expected). `now` is wall-clock "as of when we're checking" — separate from the bars' own last
    timestamp so a market that has gone quiet (no new bar, but the deadline hasn't passed yet) is correctly
    read as "still waiting", not "rejected"."""
    deadline = exit_time + timedelta(minutes=max_wait_minutes)
    deadline_passed = now >= deadline
    within = bars_since_close[bars_since_close.index <= deadline] if not bars_since_close.empty else bars_since_close
    bars_seen = len(within)

    if atr_at_close is None or atr_at_close <= 0:
        return ContinuationVerdict(False, "no usable ATR at the close to measure the move against", bars_seen, None, None, deadline_passed)

    if bars_seen < min_bars:
        reason = (
            f"only {bars_seen} of the required {min_bars} real M5 bars have traded within {max_wait_minutes:g} "
            "minutes of the close"
            + (" - window has closed, this trade is done" if deadline_passed else " - still waiting")
        )
        return ContinuationVerdict(False, reason, bars_seen, None, None, deadline_passed)

    last_bar_time = within.index[-1]
    minutes_since_close = (last_bar_time - exit_time).total_seconds() / 60.0
    sign = 1.0 if side == "buy" else -1.0
    move = sign * (float(within["Close"].iloc[-1]) - exit_price)
    move_atr = move / atr_at_close

    if move_atr < min_atr_move:
        return ContinuationVerdict(
            False, f"price only moved {move_atr:+.2f}x ATR since the close (needs >= {min_atr_move:g}x)",
            bars_seen, minutes_since_close, move_atr, deadline_passed,
        )

    return ContinuationVerdict(
        True, f"price ran {move_atr:+.2f}x ATR further in the trade's own direction within {minutes_since_close:.1f} minutes of the close",
        bars_seen, minutes_since_close, move_atr, deadline_passed,
    )
