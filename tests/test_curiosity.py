from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from ai.curiosity import (
    CuriosityCandidate,
    _candidate_symbol_and_reference_time,
    _run_curiosity_model_with_retry,
    _velocity_tier_for,
    build_candidate_dossier,
    build_curiosity_report,
    build_recursion_context,
    find_thesis_for_symbol,
    save_curiosity_report,
    score_abnormal_moves,
    score_missed_opportunities,
    select_curiosity_candidates,
    select_worst_losers,
)
from ai.openrouter_client import FAILED_MESSAGE, MISSING_KEY_MESSAGE
from data.mt5_source import CancelledPendingOrder, ClosedTrade


def _trade(position_id, symbol="EURUSD", profit=0.0, opened_at=None, closed_at=None, open_price=1.1, close_price=1.1):
    opened_at = opened_at or datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    closed_at = closed_at or (opened_at + timedelta(hours=2))
    return ClosedTrade(
        position_id=position_id, symbol=symbol, side="buy", volume=1.0,
        opened_at=opened_at, closed_at=closed_at,
        open_price=open_price, close_price=close_price, profit=profit,
    )


def _cancelled_order(
    ticket, symbol="INTC", side="buy", order_type="buy limit", price_open=95.82,
    sl=93.4, tp=97.7, volume=8.0, time_setup=None, time_done=None,
):
    time_setup = time_setup or datetime(2026, 9, 8, 16, 50, 16, tzinfo=timezone.utc)
    time_done = time_done or (time_setup + timedelta(hours=15))
    return CancelledPendingOrder(
        ticket=ticket, symbol=symbol, side=side, order_type=order_type,
        volume=volume, price_open=price_open, sl=sl, tp=tp,
        time_setup=time_setup, time_done=time_done,
    )


def _bars(closes, start=None, freq="h"):
    start = start or datetime(2026, 9, 1, 0, 0)
    index = pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame(
        {"Open": closes, "High": [c + 0.5 for c in closes], "Low": [c - 0.5 for c in closes], "Close": closes, "Volume": [100.0] * len(closes)},
        index=index,
    )


# --------------------------------------------------------------------------
# select_worst_losers
# --------------------------------------------------------------------------

def test_select_worst_losers_filters_and_sorts():
    trades = [
        _trade(1, profit=-500.0),
        _trade(2, profit=200.0),
        _trade(3, profit=-1500.0),
        _trade(4, profit=-2.0),  # below the noise floor
    ]
    result = select_worst_losers(trades, max_count=2, min_loss_usd=10.0)
    assert [t.position_id for t in result] == [3, 1]


def test_select_worst_losers_empty_when_all_profitable():
    trades = [_trade(1, profit=100.0), _trade(2, profit=50.0)]
    assert select_worst_losers(trades, max_count=1) == []


# --------------------------------------------------------------------------
# score_abnormal_moves
# --------------------------------------------------------------------------

def test_score_abnormal_moves_computes_atr_multiple():
    trade = _trade(1, symbol="EURUSD", close_price=100.0, closed_at=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc))
    # Pre-entry: steady +1.0/hour drift -> known ATR (True Range dominated
    # by the gap from prior close, same construction as test_technical.py's
    # own ATR fixtures) = 1.5.
    pre_entry_closes = [100.0 + i * 1.0 for i in range(20)]
    # Post-exit: jumps straight to 106.0 and stays there -> a 6.0 move,
    # i.e. 4.0x the 1.5 ATR baseline.
    post_exit_closes = [106.0] * 5

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        if date_to == trade.opened_at:
            return _bars(pre_entry_closes)
        return _bars(post_exit_closes)

    scored = score_abnormal_moves([trade], fake_fetch_range, pool_size=5, atr_context_hours=20, post_exit_hours=5)
    assert len(scored) == 1
    _, move_size, atr_multiple, pre_entry, post_exit, atr = scored[0]
    assert atr == pytest.approx(1.5)
    assert move_size == pytest.approx(6.0)
    assert atr_multiple == pytest.approx(4.0)


def test_score_abnormal_moves_none_multiple_when_post_exit_empty_and_sorts_last():
    trade_no_data = _trade(1, symbol="NODATA")
    trade_with_data = _trade(2, symbol="EURUSD", close_price=100.0)

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        if symbol == "NODATA":
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        if date_to == trade_with_data.opened_at:
            return _bars([100.0 + i for i in range(20)])
        return _bars([110.0] * 5)

    scored = score_abnormal_moves(
        [trade_no_data, trade_with_data], fake_fetch_range, pool_size=5, atr_context_hours=20, post_exit_hours=5
    )
    # The real, scoreable trade sorts FIRST (higher atr_multiple beats None).
    assert scored[0][0].symbol == "EURUSD"
    assert scored[0][2] is not None
    assert scored[-1][0].symbol == "NODATA"
    assert scored[-1][2] is None


def test_score_abnormal_moves_respects_pool_size():
    trades = [_trade(i) for i in range(5)]
    calls = []

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        calls.append(symbol)
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    score_abnormal_moves(trades, fake_fetch_range, pool_size=2, atr_context_hours=1, post_exit_hours=1)
    # 2 trades scored, 2 fetches each (pre-entry + post-exit) = 4 calls.
    assert len(calls) == 4


# --------------------------------------------------------------------------
# _velocity_tier_for
# --------------------------------------------------------------------------

def test_velocity_tier_for_none_without_atr_or_price():
    assert _velocity_tier_for(None, 100.0) is None
    assert _velocity_tier_for(1.0, None) is None
    assert _velocity_tier_for(1.0, 0.0) is None


def test_velocity_tier_for_classifies_from_atr_pct():
    # 4400 * 0.0025 = 11.0 -> atr_pct = 0.25% -> exactly the fast threshold
    assert _velocity_tier_for(11.0, 4400.0) == "fast"
    # A calm FX cross: atr_pct well under 0.25%
    assert _velocity_tier_for(0.001, 1.38) == "slow"


# --------------------------------------------------------------------------
# score_missed_opportunities (real INTC/AMD incident, 2026-09-09)
# --------------------------------------------------------------------------

def test_score_missed_opportunities_real_intc_numbers():
    # The exact real incident this feature was built for: entry 95.82,
    # the market's closest approach was 100.30 (never within 4.48 of the
    # entry), and price traded through the original 97.70 take-profit
    # before the order was ever filled.
    order = _cancelled_order(
        ticket=1, symbol="INTC", side="buy", price_open=95.82, tp=97.7,
        time_setup=datetime(2026, 9, 8, 16, 50, 16, tzinfo=timezone.utc),
        time_done=datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc),
    )
    pre_placement_closes = [95.0 + i * 0.05 for i in range(20)]  # a calm run-up -> a real, small ATR

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        if date_to == order.time_setup:
            return _bars(pre_placement_closes)
        # The resting window: low never drops below 100.30, high clears the 97.70 TP easily.
        return _bars([105.09, 104.36, 106.04, 100.30, 103.5])

    scored = score_missed_opportunities([order], fake_fetch_range, pool_size=5, atr_context_hours=20, min_atr_multiple=0.1)
    assert len(scored) == 1
    scored_order, atr_multiple, pre_placement, resting, atr, closest_approach, target_exceeded = scored[0]
    assert scored_order.symbol == "INTC"
    # _bars() synthesizes Low = close - 0.5, so the real closest LOW
    # reached is 100.30 - 0.5 = 99.80, still nowhere near the 95.82 entry.
    assert closest_approach == pytest.approx(99.80 - 95.82)  # never reached the entry -> positive
    assert target_exceeded is True
    assert atr_multiple is not None and atr_multiple > 0


def test_score_missed_opportunities_sell_side_uses_high_not_low():
    # A sell limit rests ABOVE the market, approached from below via the
    # window's own HIGH — using Low here (the buy-side logic) would give
    # a materially different (and wrong) answer, which this test's own
    # exact-value assertion below would catch.
    order = _cancelled_order(
        ticket=2, symbol="USDCNH", side="sell", price_open=100.0, tp=90.0,
        time_setup=datetime(2026, 9, 8, 4, 34, 17, tzinfo=timezone.utc),
        time_done=datetime(2026, 9, 9, 12, 35, 21, tzinfo=timezone.utc),
    )
    pre_placement_closes = [95.0 + i * 0.1 for i in range(20)]

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        if date_to == order.time_setup:
            return _bars(pre_placement_closes)
        # Closes stay well below the 100.0 entry; _bars() High = close+0.5,
        # so the real closest approach is 96.5 (from the 96.0 close), and
        # using Low instead (close-0.5 -> min 92.5) would give a different,
        # wrong 7.5 answer instead of the correct 3.5.
        return _bars([95.0, 94.0, 96.0, 93.0])

    scored = score_missed_opportunities([order], fake_fetch_range, pool_size=5, atr_context_hours=20, min_atr_multiple=0.0)
    assert len(scored) == 1
    _, _, _, _, _, closest_approach, target_exceeded = scored[0]
    assert closest_approach == pytest.approx(100.0 - 96.5)
    assert target_exceeded is False  # low never reached the 90.0 target


def test_score_missed_opportunities_excludes_orders_below_the_atr_threshold():
    # A cancellation for a genuinely benign reason (e.g. a fresh mega
    # session simply changed its mind) with the market never moving
    # meaningfully must NOT be flagged as a "missed opportunity" story.
    order = _cancelled_order(ticket=3, symbol="EURUSD", side="buy", price_open=1.1000)
    pre_placement_closes = [1.10 + i * 0.0001 for i in range(20)]

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        if date_to == order.time_setup:
            return _bars(pre_placement_closes)
        return _bars([1.1002, 1.1003])  # barely moved

    scored = score_missed_opportunities([order], fake_fetch_range, pool_size=5, atr_context_hours=20, min_atr_multiple=2.0)
    assert scored == []


def test_score_missed_opportunities_excludes_orders_with_no_real_atr_baseline():
    # No pre-placement history at all -- never fabricate a threshold
    # comparison from data that doesn't exist.
    order = _cancelled_order(ticket=4, symbol="NODATA")

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    assert score_missed_opportunities([order], fake_fetch_range, pool_size=5, min_atr_multiple=0.1) == []


def test_score_missed_opportunities_sorts_worst_miss_first():
    near_miss = _cancelled_order(ticket=5, symbol="NEAR", side="buy", price_open=100.0)
    far_miss = _cancelled_order(ticket=6, symbol="FAR", side="buy", price_open=100.0)
    pre_placement_closes = [100.0 + i * 0.1 for i in range(20)]  # same small ATR for both

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        if date_to == near_miss.time_setup:
            return _bars(pre_placement_closes)
        if symbol == "NEAR":
            return _bars([102.0])  # 2.0 away
        return _bars([110.0])  # 10.0 away

    scored = score_missed_opportunities(
        [near_miss, far_miss], fake_fetch_range, pool_size=5, atr_context_hours=20, min_atr_multiple=0.1,
    )
    assert [s[0].symbol for s in scored] == ["FAR", "NEAR"]


# --------------------------------------------------------------------------
# select_curiosity_candidates
# --------------------------------------------------------------------------

def test_select_curiosity_candidates_priority_and_dedup():
    # position_id=1 is both the worst loser AND would otherwise score
    # highest for "abnormal" -- must appear exactly once, as a worst_loss,
    # not duplicated as an abnormal_move too.
    worst = _trade(1, symbol="EURUSD", profit=-1000.0, close_price=100.0)
    other = _trade(2, symbol="XAUUSD", profit=50.0, close_price=2000.0)
    trades = [worst, other]

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        if date_to in (worst.opened_at, other.opened_at):
            return _bars([100.0 + i for i in range(20)]) if symbol == "EURUSD" else _bars([2000.0 + i for i in range(20)])
        # post-exit: huge move for BOTH, so both would qualify as abnormal
        return _bars([500.0] * 5) if symbol == "EURUSD" else _bars([2500.0] * 5)

    candidates = select_curiosity_candidates(
        trades, fake_fetch_range, max_losers=1, max_abnormal=1,
        min_loss_usd=10.0, pool_size=5, atr_context_hours=20, post_exit_hours=5,
        abnormal_atr_multiple=0.5,
    )
    assert len(candidates) == 2
    categories_by_symbol = {c.trade.symbol: c.reason_category for c in candidates}
    assert categories_by_symbol["EURUSD"] == "worst_loss"
    assert categories_by_symbol["XAUUSD"] == "abnormal_move"


def test_select_curiosity_candidates_caps_total_count():
    trades = [_trade(i, symbol=f"SYM{i}", profit=-100.0 - i) for i in range(5)]

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    candidates = select_curiosity_candidates(
        trades, fake_fetch_range, max_losers=1, max_abnormal=1, min_loss_usd=10.0,
    )
    assert len(candidates) <= 2


def test_select_curiosity_candidates_includes_missed_opportunities():
    order = _cancelled_order(ticket=1, symbol="INTC", side="buy", price_open=95.82, tp=97.7)
    pre_placement_closes = [95.0 + i * 0.05 for i in range(20)]

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        if date_to == order.time_setup:
            return _bars(pre_placement_closes)
        return _bars([105.09, 104.36, 106.04, 100.30, 103.5])

    candidates = select_curiosity_candidates(
        [], fake_fetch_range, cancelled_orders=[order],
        missed_pool_size=5, atr_context_hours=20, missed_min_atr_multiple=0.1,
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert c.reason_category == "missed_opportunity"
    assert c.trade is None
    assert c.cancelled_order is not None and c.cancelled_order.symbol == "INTC"
    assert c.target_exceeded_before_fill is True
    assert c.velocity_tier in ("fast", "slow")  # computed, not None, given a real ATR baseline was available


def test_select_curiosity_candidates_no_cancelled_orders_arg_is_backward_compatible():
    # cancelled_orders defaults to None -> zero missed-opportunity
    # candidates, never a crash — every caller written before this
    # category existed must keep working unchanged.
    trades = [_trade(1, symbol="EURUSD", profit=-100.0)]

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    candidates = select_curiosity_candidates(trades, fake_fetch_range, min_loss_usd=10.0)
    assert all(c.reason_category != "missed_opportunity" for c in candidates)


def test_select_curiosity_candidates_respects_max_missed_cap():
    orders = [
        _cancelled_order(ticket=i, symbol=f"SYM{i}", side="buy", price_open=100.0)
        for i in range(3)
    ]
    pre_placement_closes = [100.0 + i * 0.1 for i in range(20)]

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        if date_to == orders[0].time_setup:
            return _bars(pre_placement_closes)
        return _bars([110.0])  # a real, qualifying miss for every order

    candidates = select_curiosity_candidates(
        [], fake_fetch_range, cancelled_orders=orders, max_missed=1,
        missed_pool_size=5, atr_context_hours=20, missed_min_atr_multiple=0.1,
    )
    assert len(candidates) == 1


# --------------------------------------------------------------------------
# find_thesis_for_symbol
# --------------------------------------------------------------------------

_IMMEDIATE_ALLOCATION_RECORD = (
    "Some prose.\n\n"
    '```json\n{"NVDA": {"pct": 1.0, "price": 220.0, "stop_loss": 216.0, "side": "buy", '
    '"reason": "structurally backed pullback buy", '
    '"invalidation_condition": "H4 closes below 214.50"}, "CASH": 99}\n```'
)

_PENDING_SETUP_RECORD = (
    "Some prose.\n\n"
    '```json\n{"CASH": 100}\n```\n\n'
    "## Pending Setups\n"
    '```json\n[{"symbol": "BTCUSD", "side": "buy", "pct": 0.3, '
    '"trigger_condition": "H1 closes above 81400", '
    '"reason": "breakout watch"}]\n```'
)


def _write_record(tmp_path, filename_stamp, body):
    d = tmp_path / "records"
    d.mkdir(exist_ok=True)
    path = d / f"portfolio_suggestion_{filename_stamp}.md"
    path.write_text(
        "## Stage 3 — Claude's final revised suggestion\n\n" + body, encoding="utf-8"
    )
    return d


def test_find_thesis_for_symbol_recovers_immediate_allocation(tmp_path):
    records_dir = _write_record(tmp_path, "2026-09-01_010000", _IMMEDIATE_ALLOCATION_RECORD)
    reason, invalidation = find_thesis_for_symbol(
        "NVDA", datetime(2026, 9, 2, tzinfo=timezone.utc), records_dir
    )
    assert reason == "structurally backed pullback buy"
    assert invalidation == "H4 closes below 214.50"


def test_find_thesis_for_symbol_falls_back_to_pending_setups(tmp_path):
    records_dir = _write_record(tmp_path, "2026-09-01_010000", _PENDING_SETUP_RECORD)
    reason, invalidation = find_thesis_for_symbol(
        "BTCUSD", datetime(2026, 9, 2, tzinfo=timezone.utc), records_dir
    )
    assert reason == "breakout watch"
    assert invalidation == "H1 closes above 81400"


# Real bug found on self-review: passing frozenset(allocation.keys())
# (ALL symbols, including a pct=0 cancel) as parse_pending_setups' own
# immediate_symbols made it fail the WHOLE pending-setups array parse
# whenever a just-cancelled symbol was legitimately re-listed as a
# fresh Pending Setup the same session — a real, previously-fixed
# pattern in this account's own history (see _write_latest_suggestion's
# own comment for the real BTCUSD incident). That failure would then
# silently lose an UNRELATED symbol's own thesis too, since the whole
# array (not just the colliding entry) fails to parse. ETHUSD here is
# ONLY a pending setup (never in the allocation), so recovering its
# thesis REQUIRES parse_pending_setups to succeed — this only passes
# once the pct=0 BTCUSD cancel is correctly excluded from
# immediate_symbols.
_MIXED_CANCEL_AND_PENDING_RECORD = (
    "Some prose.\n\n"
    '```json\n{"BTCUSD": {"pct": 0, "side": "buy"}, "CASH": 99.7}\n```\n\n'
    "## Pending Setups\n"
    '```json\n[{"symbol": "BTCUSD", "side": "buy", "pct": 0.3, '
    '"trigger_condition": "H1 closes above 81400", '
    '"reason": "cancelled then re-watched for a cleaner trigger"}, '
    '{"symbol": "ETHUSD", "side": "buy", "pct": 0.3, '
    '"trigger_condition": "H1 closes above 2500", '
    '"reason": "fresh breakout watch"}]\n```'
)


def test_find_thesis_for_symbol_pct_zero_cancel_does_not_break_other_symbols_lookup(tmp_path):
    records_dir = _write_record(tmp_path, "2026-09-01_010000", _MIXED_CANCEL_AND_PENDING_RECORD)
    reason, invalidation = find_thesis_for_symbol(
        "ETHUSD", datetime(2026, 9, 2, tzinfo=timezone.utc), records_dir
    )
    assert reason == "fresh breakout watch"
    assert invalidation == "H1 closes above 2500"


# Real bug found on self-review: the allocation check above returned on
# ANY match for the symbol, even a pct=0 cancel with no real reason/
# invalidation_condition of its own -- shadowing the SAME symbol's own
# richer pending-setup entry in _MIXED_CANCEL_AND_PENDING_RECORD (a
# re-watched BTCUSD, re-listed as a fresh Pending Setup the same
# session, exactly the real incident this file's own comments already
# describe one function away). Before the fix this returned (None, None)
# for BTCUSD -- the pct=0 entry's own reason/invalidation_condition are
# both absent from the fixture's JSON -- instead of falling through to
# the pending setup that's the trade's real origin.
def test_find_thesis_for_symbol_pct_zero_cancel_does_not_shadow_its_own_pending_setup(tmp_path):
    records_dir = _write_record(tmp_path, "2026-09-01_010000", _MIXED_CANCEL_AND_PENDING_RECORD)
    reason, invalidation = find_thesis_for_symbol(
        "BTCUSD", datetime(2026, 9, 2, tzinfo=timezone.utc), records_dir
    )
    assert reason == "cancelled then re-watched for a cleaner trigger"
    assert invalidation == "H1 closes above 81400"


def test_find_thesis_for_symbol_skips_sessions_generated_after_the_trade(tmp_path):
    # This record's own filename timestamp is AFTER the trade's opened_at
    # -- it cannot be the trade's origin and must be ignored, not matched.
    records_dir = _write_record(tmp_path, "2026-09-05_010000", _IMMEDIATE_ALLOCATION_RECORD)
    reason, invalidation = find_thesis_for_symbol(
        "NVDA", datetime(2026, 9, 1, tzinfo=timezone.utc), records_dir
    )
    assert (reason, invalidation) == (None, None)


def test_find_thesis_for_symbol_none_when_nothing_matches(tmp_path):
    records_dir = _write_record(tmp_path, "2026-09-01_010000", _IMMEDIATE_ALLOCATION_RECORD)
    reason, invalidation = find_thesis_for_symbol(
        "GBPUSD", datetime(2026, 9, 2, tzinfo=timezone.utc), records_dir
    )
    assert (reason, invalidation) == (None, None)


def test_find_thesis_for_symbol_none_when_directory_missing(tmp_path):
    reason, invalidation = find_thesis_for_symbol(
        "NVDA", datetime(2026, 9, 2, tzinfo=timezone.utc), tmp_path / "does_not_exist"
    )
    assert (reason, invalidation) == (None, None)


# --------------------------------------------------------------------------
# build_candidate_dossier
# --------------------------------------------------------------------------

def test_build_candidate_dossier_includes_all_real_facts():
    trade = _trade(1, symbol="NVDA", profit=-123.45, open_price=220.0, close_price=216.0)
    candidate = CuriosityCandidate(
        trade=trade, reason_category="worst_loss",
        thesis="structurally backed pullback buy",
        invalidation_condition="H4 closes below 214.50",
        pre_entry_path=_bars([218.0, 219.0, 220.0]),
        post_exit_path=_bars([215.0, 214.0]),
        pre_entry_atr=1.5,
        post_exit_move_atr_multiple=3.0,
    )
    text = build_candidate_dossier(candidate)
    assert "NVDA" in text
    assert "structurally backed pullback buy" in text
    assert "H4 closes below 214.50" in text
    assert "-123.45" in text
    assert "3.00x" in text
    assert "1.5" in text


def test_build_candidate_dossier_handles_missing_thesis_gracefully():
    trade = _trade(1, symbol="NVDA")
    candidate = CuriosityCandidate(
        trade=trade, reason_category="abnormal_move", thesis=None, invalidation_condition=None,
        pre_entry_path=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
        post_exit_path=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
        pre_entry_atr=None,
    )
    text = build_candidate_dossier(candidate)
    assert "not recoverable from saved records" in text
    assert "thesis: None" not in text.lower()


def test_build_candidate_dossier_missed_opportunity_shows_target_exceeded():
    order = _cancelled_order(ticket=1, symbol="INTC", side="buy", price_open=95.82, sl=93.4, tp=97.7)
    candidate = CuriosityCandidate(
        trade=None, reason_category="missed_opportunity",
        thesis="Genuinely strong fundamental backdrop; wait for the pullback.",
        invalidation_condition="H1 RSI(14) cools back below 60",
        pre_entry_path=_bars([95.0, 95.5]),
        post_exit_path=pd.DataFrame(),
        pre_entry_atr=0.95,
        post_exit_move_atr_multiple=4.7,
        velocity_tier="fast",
        cancelled_order=order,
        resting_price_path=_bars([100.3, 104.36, 105.09]),
        closest_approach_distance=100.30 - 95.82,
        target_exceeded_before_fill=True,
    )
    text = build_candidate_dossier(candidate)
    assert "INTC" in text
    assert "missed opportunity" in text
    assert "95.82" in text
    assert "TARGET ALREADY EXCEEDED" in text
    assert "FAST" in text
    assert "never reached the entry level" in text
    assert "Genuinely strong fundamental backdrop" in text


def test_build_candidate_dossier_missed_opportunity_reports_negative_approach_as_anomaly():
    # closest_approach_distance <= 0 means price actually crossed the
    # entry level without the order filling -- a real anomaly, must be
    # reported as such, never silently treated as "never reached."
    order = _cancelled_order(ticket=2, symbol="AMD", side="buy", price_open=481.0, tp=497.25)
    candidate = CuriosityCandidate(
        trade=None, reason_category="missed_opportunity", thesis=None, invalidation_condition=None,
        pre_entry_path=pd.DataFrame(), post_exit_path=pd.DataFrame(), pre_entry_atr=1.0,
        cancelled_order=order, resting_price_path=pd.DataFrame(),
        closest_approach_distance=-2.5,
    )
    text = build_candidate_dossier(candidate)
    assert "anomaly" in text.lower()
    assert "PAST the entry level" in text


# --------------------------------------------------------------------------
# _candidate_symbol_and_reference_time
# --------------------------------------------------------------------------

def test_candidate_symbol_and_reference_time_from_trade():
    trade = _trade(1, symbol="NVDA", opened_at=datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc))
    candidate = CuriosityCandidate(
        trade=trade, reason_category="worst_loss", thesis=None, invalidation_condition=None,
        pre_entry_path=pd.DataFrame(), post_exit_path=pd.DataFrame(), pre_entry_atr=None,
    )
    symbol, before = _candidate_symbol_and_reference_time(candidate)
    assert symbol == "NVDA"
    assert before == datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


def test_candidate_symbol_and_reference_time_from_cancelled_order():
    order = _cancelled_order(ticket=1, symbol="INTC", time_setup=datetime(2026, 9, 8, 16, 50, 16, tzinfo=timezone.utc))
    candidate = CuriosityCandidate(
        trade=None, reason_category="missed_opportunity", thesis=None, invalidation_condition=None,
        pre_entry_path=pd.DataFrame(), post_exit_path=pd.DataFrame(), pre_entry_atr=None,
        cancelled_order=order,
    )
    symbol, before = _candidate_symbol_and_reference_time(candidate)
    assert symbol == "INTC"
    assert before == datetime(2026, 9, 8, 16, 50, 16, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# build_recursion_context
# --------------------------------------------------------------------------

def test_build_recursion_context_empty_when_no_prior_reports(tmp_path):
    assert build_recursion_context(tmp_path / "does_not_exist") == ""


def test_build_recursion_context_uses_only_the_latest_report(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "curiosity_report_2026-09-01_010000.md").write_text("OLDER REPORT", encoding="utf-8")
    (tmp_path / "curiosity_report_2026-09-03_010000.md").write_text("NEWEST REPORT", encoding="utf-8")
    text = build_recursion_context(tmp_path)
    assert "NEWEST REPORT" in text
    assert "OLDER REPORT" not in text


# --------------------------------------------------------------------------
# _run_curiosity_model_with_retry
# --------------------------------------------------------------------------

@patch("ai.curiosity.time.sleep")
@patch("ai.curiosity.run_openrouter")
def test_run_curiosity_model_with_retry_falls_through_to_next_model(mock_run, mock_sleep):
    mock_run.side_effect = [FAILED_MESSAGE, "a real report card"]
    result = _run_curiosity_model_with_retry("some prompt")
    assert result == "a real report card"
    assert mock_run.call_count == 2


@patch("ai.curiosity.time.sleep")
@patch("ai.curiosity.run_openrouter", return_value="a real report card")
def test_run_curiosity_model_with_retry_uses_its_own_short_timeout_not_the_audit_pools(mock_run, mock_sleep):
    # Real bug found on self-review: this used to reuse config.
    # OPENROUTER_TIMEOUT_SECONDS (90s, the audit pool's own per-call
    # budget) here — since the deadline check only ever stops a NEW
    # attempt from starting (never bounds one already in flight), that
    # could let a run with several genuinely-hanging fallbacks take
    # several minutes despite CURIOSITY_RETRY_TIMEOUT_SECONDS' own
    # "give up far sooner" design. Must use its own, shorter,
    # dedicated CURIOSITY_MODEL_TIMEOUT_SECONDS instead.
    import config

    _run_curiosity_model_with_retry("some prompt")
    _, kwargs = mock_run.call_args
    assert kwargs["timeout"] == config.CURIOSITY_MODEL_TIMEOUT_SECONDS
    assert kwargs["timeout"] < config.OPENROUTER_TIMEOUT_SECONDS


@patch("ai.curiosity.time.sleep")
@patch("ai.curiosity.run_openrouter", return_value=FAILED_MESSAGE)
def test_run_curiosity_model_with_retry_empty_on_total_exhaustion(mock_run, mock_sleep):
    result = _run_curiosity_model_with_retry("some prompt")
    assert result == ""


@patch("ai.curiosity.time.sleep")
@patch("ai.curiosity.run_openrouter", return_value=MISSING_KEY_MESSAGE)
def test_run_curiosity_model_with_retry_stops_immediately_on_missing_key(mock_run, mock_sleep):
    result = _run_curiosity_model_with_retry("some prompt")
    assert result == ""
    assert mock_run.call_count == 1


# --------------------------------------------------------------------------
# save_curiosity_report
# --------------------------------------------------------------------------

def test_save_curiosity_report_writes_a_real_file(tmp_path):
    trade = _trade(1, symbol="NVDA")
    candidate = CuriosityCandidate(
        trade=trade, reason_category="worst_loss", thesis=None, invalidation_condition=None,
        pre_entry_path=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
        post_exit_path=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
        pre_entry_atr=None,
    )
    path = save_curiosity_report("a report", [candidate], tmp_path / "curiosity")
    assert path is not None
    assert path.exists()
    assert "NVDA" in path.read_text(encoding="utf-8")
    assert "a report" in path.read_text(encoding="utf-8")


def test_save_curiosity_report_returns_none_on_oserror(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    result = save_curiosity_report("a report", [], blocker)
    assert result is None


def test_save_curiosity_report_handles_a_missed_opportunity_candidate(tmp_path):
    # Real bug caught while wiring this in: the old code read
    # c.trade.symbol unconditionally, which crashes with AttributeError
    # on a missed-opportunity candidate (trade is None there) — this
    # must work cleanly for a mix of both candidate kinds.
    trade_candidate = CuriosityCandidate(
        trade=_trade(1, symbol="NVDA"), reason_category="worst_loss", thesis=None, invalidation_condition=None,
        pre_entry_path=pd.DataFrame(), post_exit_path=pd.DataFrame(), pre_entry_atr=None,
    )
    missed_candidate = CuriosityCandidate(
        trade=None, reason_category="missed_opportunity", thesis=None, invalidation_condition=None,
        pre_entry_path=pd.DataFrame(), post_exit_path=pd.DataFrame(), pre_entry_atr=None,
        cancelled_order=_cancelled_order(ticket=1, symbol="INTC"),
    )
    path = save_curiosity_report("a report", [trade_candidate, missed_candidate], tmp_path / "curiosity")
    assert path is not None
    content = path.read_text(encoding="utf-8")
    assert "NVDA" in content
    assert "INTC" in content


# --------------------------------------------------------------------------
# build_curiosity_report (full orchestration, every I/O boundary faked)
# --------------------------------------------------------------------------

def test_build_curiosity_report_empty_when_no_closed_trades():
    result = build_curiosity_report(
        get_history_deals_fn=lambda date_from: [],
        fetch_range_fn=lambda *a: pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
    )
    assert result == ""


def test_build_curiosity_report_empty_when_no_candidates_qualify(tmp_path):
    # A real closed trade, but profitable and with no abnormal post-exit
    # move -- nothing clears either selection bar.
    trade = _trade(1, profit=50.0)
    from data.mt5_source import HistoricalDeal

    deals = [
        HistoricalDeal(ticket=1, time=trade.opened_at, symbol=trade.symbol, profit=0.0, volume=1.0, position_id=1, price=1.1, side="buy", entry=0),
        HistoricalDeal(ticket=2, time=trade.closed_at, symbol=trade.symbol, profit=50.0, volume=1.0, position_id=1, price=1.1, side="sell", entry=1),
    ]
    result = build_curiosity_report(
        records_dir=tmp_path, curiosity_records_dir=tmp_path / "curiosity",
        get_history_deals_fn=lambda date_from: deals,
        fetch_range_fn=lambda *a: pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
    )
    assert result == ""


@patch("ai.curiosity._run_curiosity_model_with_retry", return_value="a real critique")
def test_build_curiosity_report_full_success_path(mock_model, tmp_path):
    from data.mt5_source import HistoricalDeal

    opened_at = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    closed_at = opened_at + timedelta(hours=2)
    deals = [
        HistoricalDeal(ticket=1, time=opened_at, symbol="NVDA", profit=0.0, volume=1.0, position_id=1, price=220.0, side="buy", entry=0),
        HistoricalDeal(ticket=2, time=closed_at, symbol="NVDA", profit=-500.0, volume=1.0, position_id=1, price=216.0, side="sell", entry=1),
    ]
    result = build_curiosity_report(
        records_dir=tmp_path, curiosity_records_dir=tmp_path / "curiosity",
        get_history_deals_fn=lambda date_from: deals,
        fetch_range_fn=lambda *a: pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
    )
    assert "a real critique" in result
    assert "Compulsory retrospective self-critique" in result
    # The report was actually saved for future recursion.
    saved_files = list((tmp_path / "curiosity").glob("curiosity_report_*.md"))
    assert len(saved_files) == 1


def test_build_curiosity_report_never_raises_on_unexpected_exception():
    def _boom(date_from):
        raise RuntimeError("MT5 unreachable")

    result = build_curiosity_report(get_history_deals_fn=_boom)
    assert result == ""


@patch("ai.curiosity._run_curiosity_model_with_retry", return_value="a real missed-opportunity critique")
def test_build_curiosity_report_full_success_path_from_missed_opportunity_alone(mock_model, tmp_path):
    # Critical end-to-end regression test: ZERO closed trades this
    # lookback window, but a real cancelled order that's a genuine
    # missed opportunity — the real INTC incident this whole feature
    # exists for. Exercises the exact path that used to crash
    # (find_thesis_for_symbol called against candidate.trade.symbol
    # unconditionally) before the _candidate_symbol_and_reference_time
    # fix, silently swallowed by the outer try/except into an empty
    # report — this must now produce a real, non-empty report instead.
    order = _cancelled_order(
        ticket=1, symbol="INTC", side="buy", price_open=95.82, tp=97.7,
        time_setup=datetime(2026, 9, 8, 16, 50, 16, tzinfo=timezone.utc),
        time_done=datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc),
    )
    pre_placement_closes = [95.0 + i * 0.05 for i in range(20)]

    def fake_fetch_range(symbol, timeframe, date_from, date_to):
        if date_to == order.time_setup:
            return _bars(pre_placement_closes)
        return _bars([105.09, 104.36, 106.04, 100.30, 103.5])

    result = build_curiosity_report(
        records_dir=tmp_path, curiosity_records_dir=tmp_path / "curiosity",
        get_history_deals_fn=lambda date_from: [],
        fetch_range_fn=fake_fetch_range,
        get_cancelled_pending_orders_fn=lambda date_from: [order],
    )
    assert "a real missed-opportunity critique" in result
    assert "Compulsory retrospective self-critique" in result
    saved_files = list((tmp_path / "curiosity").glob("curiosity_report_*.md"))
    assert len(saved_files) == 1
    assert "INTC" in saved_files[0].read_text(encoding="utf-8")


def test_build_curiosity_report_empty_when_neither_trades_nor_cancelled_orders():
    result = build_curiosity_report(
        get_history_deals_fn=lambda date_from: [],
        fetch_range_fn=lambda *a: pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
        get_cancelled_pending_orders_fn=lambda date_from: [],
    )
    assert result == ""
