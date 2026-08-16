"""Pure computation of this FTMO 1-Stage Challenge account's real standing
against its three real compliance rules — no MT5 calls, no LLM calls; the
caller (app.py) fetches `get_history_deals()`/`get_account_summary()` and
passes the real results in.

IMPORTANT CAVEAT, stated plainly here and repeated in both the UI
(app.py) and the AI prompt (ai/ftmo_suggest.py) per this project's
disclosure convention: this is a real, best-effort RECONSTRUCTION from
this account's own MT5 trade history, NOT a certified mirror of FTMO's
internal ledger. Two real sources of drift from FTMO's own numbers:
1. Day boundaries here are bucketed by each deal's own server-time
   timestamp's calendar date, not FTMO's real CE(S)T midnight reset —
   a trade near a day boundary can bucket into the "wrong" day relative
   to FTMO's own accounting if the MT5 server's timezone doesn't line up
   exactly with CE(S)T (and doesn't account for the CET/CEST DST switch
   at all). This only matters for a trade genuinely straddling midnight.
2. FTMO's own internal ledger may include adjustments (e.g. a manual
   correction) this reconstruction has no way to see.
Cross-check against FTMO's own dashboard before trusting this near a
hard limit — a breach of the daily-loss or max-loss rule is instant
account termination with zero grace period.

Real FTMO 1-Stage Challenge rules encoded here (confirmed against FTMO's
own published rules, not invented):
- 3% Max Daily Loss, on EQUITY, resetting at midnight CE(S)T.
- 10% Max Loss as a TRAILING end-of-day floor (90% of the account's
  highest-ever end-of-day balance) — specific to the 1-Stage Challenge;
  the 2-Step Challenge instead uses a STATIC floor off the initial
  balance, which this deliberately does NOT implement.
- Best Day Rule: the single best day's profit must not exceed 50% of
  the total profit summed across all positive days.
"""

from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime

from data.mt5_source import HistoricalDeal

DAILY_LOSS_LIMIT_PCT = 3.0
MAX_LOSS_LIMIT_PCT = 10.0
# The trailing floor is 90% of the account's peak end-of-day balance —
# derived directly from MAX_LOSS_LIMIT_PCT so the two can never drift
# apart if this ever needs tuning.
_TRAILING_FLOOR_MULTIPLIER = 1 - MAX_LOSS_LIMIT_PCT / 100
_BEST_DAY_RULE_LIMIT_PCT = 50.0


@dataclass
class FtmoStatus:
    daily_loss_limit_pct: float
    today_realized_pl: float
    today_floating_pl: float
    today_total_pl: float
    daily_loss_headroom_pct: float
    trailing_max_loss_floor: float
    max_loss_headroom_pct: float
    best_day_pl: float | None
    total_positive_days_pl: float | None
    best_day_rule_pct: float | None


def compute_ftmo_status(
    deals: list[HistoricalDeal],
    initial_balance: float,
    current_equity: float,
    current_balance: float,
    now: datetime | None = None,
) -> FtmoStatus:
    """Pure, deterministic reconstruction of this account's real standing
    against all three 1-Stage Challenge rules (see module docstring for
    the exact rules and their caveats). `deals` should already be scoped
    to this account's own real MT5 trade history (typically from Challenge
    start through now) — this function does no fetching or filtering by
    time range itself beyond bucketing what it's given into calendar days.

    Day bucketing uses each deal's own `HistoricalDeal.time` (already a
    naive local/server-time datetime from data/mt5_source.py) — see the
    module docstring's caveat #1 for why this is a best-effort
    approximation of the true CE(S)T boundary, not an exact one.

    Zero-history (a fresh account with no trades yet) degrades cleanly:
    daily-loss/max-loss headroom read as the full 3%/10% (nothing has
    happened today or ever to erode them), and all three Best-Day-Rule
    fields come back None (nothing to compute a rule violation from) —
    never a crash or a 0/0.
    """
    now = now if now is not None else datetime.now()
    today = now.date()

    # A deposit/withdrawal/credit adjustment shows up in MT5's own deal
    # history as a real HistoricalDeal too (MetaTrader5's DEAL_TYPE_BALANCE
    # etc.), but data/mt5_source.py's HistoricalDeal doesn't carry the raw
    # deal type — only `symbol`, which is always empty for a non-trade
    # deal and always real for an actual trade. Filtering on that is the
    # only way to keep e.g. the Challenge's own initial-funding deposit
    # from being counted as "today's trading P&L" or a day's EOD-balance
    # move, which would badly distort every rule computed below.
    trade_deals = [d for d in deals if d.symbol]

    daily_realized_pl: dict[date_cls, float] = {}
    for d in trade_deals:
        day = d.time.date()
        daily_realized_pl[day] = daily_realized_pl.get(day, 0.0) + d.profit

    # --- 3% Max Daily Loss (equity-based) ---
    today_realized_pl = daily_realized_pl.get(today, 0.0)
    today_floating_pl = current_equity - current_balance
    today_total_pl = today_realized_pl + today_floating_pl

    daily_loss_limit_amount = initial_balance * DAILY_LOSS_LIMIT_PCT / 100
    loss_so_far_today = max(0.0, -today_total_pl)  # a profitable day has zero "loss so far"
    daily_loss_headroom_pct = (
        (daily_loss_limit_amount - loss_so_far_today) / initial_balance * 100
        if initial_balance
        else 0.0
    )

    # --- 10% Max Loss (TRAILING end-of-day floor, only ever moves up) ---
    # Reconstructs each bucketed day's EOD balance by cumulatively summing
    # that day's own realized P&L onto initial_balance, running forward in
    # calendar order — the running MAX of that series is what makes the
    # floor monotonically non-decreasing even after a later losing day
    # drags the running balance back down from its peak (a losing day
    # can never un-happen a peak that already occurred).
    running_balance = initial_balance
    peak_eod_balance = initial_balance
    for day in sorted(daily_realized_pl):
        running_balance += daily_realized_pl[day]
        peak_eod_balance = max(peak_eod_balance, running_balance)

    trailing_max_loss_floor = peak_eod_balance * _TRAILING_FLOOR_MULTIPLIER
    max_loss_headroom_pct = (
        (current_equity - trailing_max_loss_floor) / initial_balance * 100 if initial_balance else 0.0
    )

    # --- Best Day Rule ---
    if not daily_realized_pl:
        best_day_pl = None
        total_positive_days_pl = None
        best_day_rule_pct = None
    else:
        positive_days_pl = [v for v in daily_realized_pl.values() if v > 0]
        best_day_pl = max(daily_realized_pl.values())
        total_positive_days_pl = sum(positive_days_pl)  # 0.0, not None, when no day was ever positive
        best_day_rule_pct = (
            best_day_pl / total_positive_days_pl * 100 if total_positive_days_pl > 0 else None
        )

    return FtmoStatus(
        daily_loss_limit_pct=DAILY_LOSS_LIMIT_PCT,
        today_realized_pl=today_realized_pl,
        today_floating_pl=today_floating_pl,
        today_total_pl=today_total_pl,
        daily_loss_headroom_pct=daily_loss_headroom_pct,
        trailing_max_loss_floor=trailing_max_loss_floor,
        max_loss_headroom_pct=max_loss_headroom_pct,
        best_day_pl=best_day_pl,
        total_positive_days_pl=total_positive_days_pl,
        best_day_rule_pct=best_day_rule_pct,
    )


def would_breach_daily_loss_headroom(
    status: FtmoStatus, planned_heat_pct: float, headroom_fraction: float = 0.5
) -> bool:
    """True if a planned trade's aggregate heat (% of equity at risk if
    every stop is hit — same figure risk/apply_suggestion.py-adjacent
    code already computes elsewhere) would exceed `headroom_fraction` of
    the account's REAL remaining daily-loss headroom right now. Used by
    app.py as a hard pre-execution gate specific to FTMO (see the plan's
    explicit 50%-of-headroom pre-execution check) — PMEX has no such
    check since it has no daily-loss rule to check against.

    A non-positive headroom (already at/over the daily-loss limit) means
    ANY new risk breaches this gate, not just a large one — there's no
    fraction of zero (or negative) room that's still safe to use."""
    if status.daily_loss_headroom_pct <= 0:
        return True
    return planned_heat_pct > status.daily_loss_headroom_pct * headroom_fraction
