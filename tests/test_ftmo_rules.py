from datetime import datetime

import pytest

from data.mt5_source import HistoricalDeal
from risk.ftmo_rules import compute_ftmo_status, would_breach_daily_loss_headroom


def _deal(profit: float, day: datetime, ticket: int = 1, symbol: str = "EURUSD") -> HistoricalDeal:
    return HistoricalDeal(ticket=ticket, time=day, symbol=symbol, profit=profit, volume=1.0)


def test_fresh_account_zero_history_has_full_headroom_and_no_best_day_fields():
    now = datetime(2026, 8, 15, 10, 0)
    status = compute_ftmo_status(
        deals=[], initial_balance=100_000.0, current_equity=100_000.0, current_balance=100_000.0, now=now
    )

    assert status.daily_loss_limit_pct == 3.0
    assert status.today_realized_pl == 0.0
    assert status.today_floating_pl == 0.0
    assert status.today_total_pl == 0.0
    # Nothing has happened yet, so the full 3% is still available.
    assert status.daily_loss_headroom_pct == pytest.approx(3.0)
    # Trailing floor starts at 90% of the initial balance itself.
    assert status.trailing_max_loss_floor == pytest.approx(90_000.0)
    assert status.max_loss_headroom_pct == pytest.approx(10.0)
    assert status.best_day_pl is None
    assert status.total_positive_days_pl is None
    assert status.best_day_rule_pct is None


def test_trailing_floor_only_moves_up_after_a_losing_day_drags_balance_down():
    # Day 1: +5000 (EOD 105000, a new peak) -> Day 2: -3000 (EOD 102000,
    # BELOW the peak) -> Day 3: +2000 (EOD 104000, still below the peak).
    # The floor must stay pinned to the day-1 peak (105000 * 0.9 = 94500)
    # even though the account's own balance dropped and only partially
    # recovered afterward — this is the whole point of a TRAILING floor
    # that only ever moves up, never back down.
    deals = [
        _deal(5000.0, datetime(2026, 8, 10, 12, 0), ticket=1),
        _deal(-3000.0, datetime(2026, 8, 11, 12, 0), ticket=2),
        _deal(2000.0, datetime(2026, 8, 12, 12, 0), ticket=3),
    ]
    now = datetime(2026, 8, 12, 18, 0)
    status = compute_ftmo_status(
        deals, initial_balance=100_000.0, current_equity=104_000.0, current_balance=104_000.0, now=now
    )

    assert status.trailing_max_loss_floor == pytest.approx(94_500.0)  # 105000 * 0.90
    assert status.max_loss_headroom_pct == pytest.approx((104_000.0 - 94_500.0) / 100_000.0 * 100)


def test_trailing_floor_never_decreases_even_on_a_new_low_day():
    # Same peak-then-loss shape as above, but check the floor explicitly
    # against a NAIVE (wrong) static-floor implementation's answer: a
    # static floor would still read 90000 (10% of the ORIGINAL balance),
    # but the real trailing floor must be higher once a peak has occurred.
    deals = [
        _deal(5000.0, datetime(2026, 8, 10, 12, 0), ticket=1),
        _deal(-8000.0, datetime(2026, 8, 11, 12, 0), ticket=2),  # big losing day
    ]
    now = datetime(2026, 8, 11, 18, 0)
    status = compute_ftmo_status(
        deals, initial_balance=100_000.0, current_equity=97_000.0, current_balance=97_000.0, now=now
    )

    static_floor_a_bug_would_produce = 90_000.0
    assert status.trailing_max_loss_floor > static_floor_a_bug_would_produce
    assert status.trailing_max_loss_floor == pytest.approx(94_500.0)  # still 105000 * 0.90


def test_best_day_rule_flags_one_dominant_winning_day():
    deals = [
        _deal(100.0, datetime(2026, 8, 10, 12, 0), ticket=1),
        _deal(500.0, datetime(2026, 8, 11, 12, 0), ticket=2),  # the dominant day
        _deal(150.0, datetime(2026, 8, 12, 12, 0), ticket=3),
    ]
    now = datetime(2026, 8, 12, 18, 0)
    status = compute_ftmo_status(
        deals, initial_balance=100_000.0, current_equity=100_750.0, current_balance=100_750.0, now=now
    )

    assert status.best_day_pl == pytest.approx(500.0)
    assert status.total_positive_days_pl == pytest.approx(750.0)
    assert status.best_day_rule_pct == pytest.approx(500.0 / 750.0 * 100)
    assert status.best_day_rule_pct > 50.0  # this account would be in breach territory


def test_best_day_rule_none_when_no_day_has_ever_been_positive():
    deals = [
        _deal(-100.0, datetime(2026, 8, 10, 12, 0), ticket=1),
        _deal(-50.0, datetime(2026, 8, 11, 12, 0), ticket=2),
    ]
    now = datetime(2026, 8, 11, 18, 0)
    status = compute_ftmo_status(
        deals, initial_balance=100_000.0, current_equity=99_850.0, current_balance=99_850.0, now=now
    )

    assert status.best_day_pl == pytest.approx(-50.0)  # the least-bad day, still reported
    assert status.total_positive_days_pl == pytest.approx(0.0)
    assert status.best_day_rule_pct is None  # nothing to divide by


def test_daily_loss_headroom_breached_by_a_realized_loss_today():
    now = datetime(2026, 8, 15, 14, 0)
    deals = [_deal(-500.0, datetime(2026, 8, 15, 10, 0), ticket=1)]
    status = compute_ftmo_status(
        deals, initial_balance=10_000.0, current_equity=9_500.0, current_balance=9_500.0, now=now
    )

    # 3% of 10000 = 300 limit; a 500 loss today already exceeds it.
    assert status.today_realized_pl == pytest.approx(-500.0)
    assert status.today_total_pl == pytest.approx(-500.0)
    assert status.daily_loss_headroom_pct == pytest.approx(-2.0)  # (300 - 500) / 10000 * 100
    assert status.daily_loss_headroom_pct <= 0  # already at/over the limit


def test_daily_loss_headroom_positive_and_partially_consumed():
    now = datetime(2026, 8, 15, 14, 0)
    deals = [_deal(-100.0, datetime(2026, 8, 15, 10, 0), ticket=1)]
    status = compute_ftmo_status(
        deals, initial_balance=10_000.0, current_equity=9_900.0, current_balance=9_900.0, now=now
    )

    assert status.daily_loss_headroom_pct == pytest.approx(2.0)  # (300 - 100) / 10000 * 100


def test_daily_loss_headroom_accounts_for_todays_floating_pl_too():
    # No realized trades today, but an open position is currently down —
    # the daily-loss rule is EQUITY-based, so unrealized P&L counts too.
    now = datetime(2026, 8, 15, 14, 0)
    status = compute_ftmo_status(
        deals=[], initial_balance=10_000.0, current_equity=9_700.0, current_balance=10_000.0, now=now
    )

    assert status.today_realized_pl == 0.0
    assert status.today_floating_pl == pytest.approx(-300.0)
    assert status.today_total_pl == pytest.approx(-300.0)
    assert status.daily_loss_headroom_pct == pytest.approx(0.0)  # (300 - 300) / 10000 * 100


def test_only_todays_deals_count_toward_todays_realized_pl():
    now = datetime(2026, 8, 15, 14, 0)
    deals = [
        _deal(-1000.0, datetime(2026, 8, 14, 10, 0), ticket=1),  # yesterday, shouldn't count today
        _deal(-50.0, datetime(2026, 8, 15, 10, 0), ticket=2),  # today
    ]
    status = compute_ftmo_status(
        deals, initial_balance=10_000.0, current_equity=8_950.0, current_balance=8_950.0, now=now
    )

    assert status.today_realized_pl == pytest.approx(-50.0)


def test_would_breach_daily_loss_headroom_within_half_of_headroom_is_allowed():
    now = datetime(2026, 8, 15, 14, 0)
    status = compute_ftmo_status(
        deals=[], initial_balance=10_000.0, current_equity=10_000.0, current_balance=10_000.0, now=now
    )
    # Full 3% headroom -> half of it is 1.5%; a 1.0% planned heat is fine.
    assert would_breach_daily_loss_headroom(status, planned_heat_pct=1.0) is False


def test_would_breach_daily_loss_headroom_over_half_is_blocked():
    now = datetime(2026, 8, 15, 14, 0)
    status = compute_ftmo_status(
        deals=[], initial_balance=10_000.0, current_equity=10_000.0, current_balance=10_000.0, now=now
    )
    assert would_breach_daily_loss_headroom(status, planned_heat_pct=2.0) is True  # > 1.5% (half of 3%)


def test_balance_deposit_deal_is_excluded_from_daily_pl_and_trailing_floor():
    # MT5 records an account's own funding as a real deal too (symbol=""
    # — e.g. the Challenge's initial deposit) — it must never be counted
    # as "today's trading P&L" or a day's EOD-balance move, since that
    # would badly distort both the daily-loss and trailing-floor numbers
    # this function computes.
    now = datetime(2026, 8, 15, 14, 0)
    deals = [
        _deal(100_000.0, datetime(2026, 8, 15, 9, 0), ticket=1, symbol=""),  # initial deposit
        _deal(-50.0, datetime(2026, 8, 15, 10, 0), ticket=2),  # a real trade, today
    ]
    status = compute_ftmo_status(
        deals, initial_balance=100_000.0, current_equity=99_950.0, current_balance=99_950.0, now=now
    )

    assert status.today_realized_pl == pytest.approx(-50.0)
    assert status.trailing_max_loss_floor == pytest.approx(90_000.0)  # 100000 * 0.90, deposit excluded


def test_would_breach_daily_loss_headroom_any_new_risk_blocked_once_at_or_over_limit():
    now = datetime(2026, 8, 15, 14, 0)
    deals = [_deal(-500.0, datetime(2026, 8, 15, 10, 0), ticket=1)]
    status = compute_ftmo_status(
        deals, initial_balance=10_000.0, current_equity=9_500.0, current_balance=9_500.0, now=now
    )
    assert status.daily_loss_headroom_pct <= 0
    assert would_breach_daily_loss_headroom(status, planned_heat_pct=0.01) is True
