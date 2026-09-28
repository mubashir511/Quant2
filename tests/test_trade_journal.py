from unittest.mock import patch

import pytest

import config
from ai import trade_journal as tj


@pytest.fixture(autouse=True)
def _isolated_journal(tmp_path):
    with (
        patch.object(config, "TRADE_JOURNAL_DIR", str(tmp_path / "ftmo_trade_journal")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
    ):
        yield


def _payload(symbol="XAUUSD", pct=2.0, reason="aligned uptrend", pending=None):
    return {
        "immediate_allocation": {
            symbol: {
                "pct": pct, "price": 4360.0, "stop_loss": 4325.0, "take_profit": 4404.0,
                "side": "buy", "reason": reason, "invalidation_condition": "H4 closes below 4325.00",
            }
        } if pct else {},
        "pending_setups": pending or [],
    }


def test_record_proposals_creates_a_new_story():
    tj.record_proposals(_payload())
    story = tj.find_open_story("XAUUSD")
    assert story is not None
    assert story.status == "proposed"
    assert len(story.events) == 1
    assert story.events[0].type == "proposed"
    assert story.events[0].data["reason"] == "aligned uptrend"


def test_record_proposals_excludes_cash():
    payload = _payload()
    payload["immediate_allocation"]["CASH"] = {"pct": 50.0}
    tj.record_proposals(payload)
    assert tj.find_open_story("CASH") is None


def test_record_proposals_ignores_a_zero_pct_immediate_entry():
    tj.record_proposals(_payload(pct=0.0))
    assert tj.find_open_story("XAUUSD") is None


def test_record_proposals_appends_carried_forward_not_a_duplicate_story():
    tj.record_proposals(_payload(reason="first reason"))
    tj.record_proposals(_payload(reason="second reason, still holding"))
    story = tj.find_open_story("XAUUSD")
    assert len(story.events) == 2
    assert story.events[0].type == "proposed"
    assert story.events[0].data["reason"] == "first reason"
    assert story.events[1].type == "carried_forward"
    assert story.events[1].data["reason"] == "second reason, still holding"
    # Still exactly one story on disk for this symbol, not two.
    assert len([s for s in tj.list_all_stories() if s.symbol == "XAUUSD"]) == 1


def test_record_proposals_marks_a_dropped_proposed_story_expired():
    tj.record_proposals(_payload("XAUUSD"))
    tj.record_proposals(_payload("GBPUSD"))  # a fresh session that no longer mentions XAUUSD
    stories = {s.symbol: s for s in tj.list_all_stories()}
    assert stories["XAUUSD"].status == "expired_unfilled"
    assert stories["XAUUSD"].events[-1].type == "superseded"
    assert stories["GBPUSD"].status == "proposed"


def test_record_proposals_never_expires_a_real_open_position_left_unmentioned():
    # Real bug caught on direct user challenge: a genuinely open, live
    # position must NEVER be relabeled "expired_unfilled" just because a
    # later Mega Session silently reaffirms it without repeating its own
    # reasoning that cycle -- that's normal, expected behavior (mirrors
    # the real gap ai.clerk_execution._backfill_settlement_for_held_
    # positions exists to cover), not evidence the trade is done.
    tj.record_proposals(_payload("XAUUSD"))
    tj.record_order_result("XAUUSD", "open", True, "placed")
    tj.record_filled("XAUUSD", 1, 4360.0)
    assert tj.find_open_story("XAUUSD").status == "open"

    tj.record_proposals(_payload("GBPUSD"))  # fresh session, XAUUSD not mentioned at all
    xau = tj.find_open_story("XAUUSD")
    assert xau is not None
    assert xau.status == "open"  # unchanged -- NOT "expired_unfilled"
    assert xau.events[-1].type == "filled"  # no spurious "superseded" event appended either


def test_record_proposals_still_expires_an_unfilled_resting_order_left_unmentioned():
    tj.record_proposals(_payload("XAUUSD"))
    tj.record_order_result("XAUUSD", "open", True, "placed")
    assert tj.find_open_story("XAUUSD").status == "order_placed"

    tj.record_proposals(_payload("GBPUSD"))  # fresh session, XAUUSD's resting order not mentioned
    stories = {s.symbol: s for s in tj.list_all_stories()}
    assert stories["XAUUSD"].status == "expired_unfilled"


def test_record_proposals_never_reopens_an_already_closed_story():
    tj.record_proposals(_payload("XAUUSD"))
    tj.record_closed("XAUUSD", 10.0, "take_profit_hit")
    tj.record_proposals(_payload("GBPUSD"))  # XAUUSD not mentioned again
    xau = next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    assert xau.status == "closed_won"  # unchanged -- never flipped to expired_unfilled
    assert xau.events[-1].type == "closed"


def test_record_order_result_success_advances_to_order_placed():
    tj.record_proposals(_payload())
    tj.record_order_result("XAUUSD", "open", True, "placed")
    story = tj.find_open_story("XAUUSD")
    assert story.status == "order_placed"
    assert story.events[-1].data == {"action": "open", "success": True, "detail": "placed"}


def test_record_order_result_failure_on_first_attempt_marks_rejected():
    tj.record_proposals(_payload())
    tj.record_order_result("XAUUSD", "open", False, "no money")
    stories = [s for s in tj.list_all_stories() if s.symbol == "XAUUSD"]
    assert stories[0].status == "rejected"
    assert tj.find_open_story("XAUUSD") is None  # rejected is terminal


def test_record_order_result_is_a_noop_with_no_open_story():
    tj.record_order_result("NEVERPROPOSED", "open", True, "placed")
    assert tj.find_open_story("NEVERPROPOSED") is None
    assert tj.list_all_stories() == []


def test_record_filled_advances_to_open():
    tj.record_proposals(_payload())
    tj.record_order_result("XAUUSD", "open", True, "placed")
    tj.record_filled("XAUUSD", 12345, 4360.5)
    story = tj.find_open_story("XAUUSD")
    assert story.status == "open"
    assert story.events[-1].data == {"ticket": 12345, "fill_price": 4360.5}


def test_record_clerk_check_and_tactical_action_append_without_changing_status():
    tj.record_proposals(_payload())
    tj.record_order_result("XAUUSD", "open", True, "placed")
    tj.record_filled("XAUUSD", 1, 4360.0)
    tj.record_clerk_check("XAUUSD", "invalidation", False, "still holds")
    tj.record_tactical_action("XAUUSD", "defend", True, "", "tightening stop")
    story = tj.find_open_story("XAUUSD")
    assert story.status == "open"
    assert [e.type for e in story.events[-2:]] == ["clerk_check", "tactical_action"]


@pytest.mark.parametrize("pnl,expected_status", [(15.0, "closed_won"), (-15.0, "closed_lost"), (0.0, "closed_won")])
def test_record_closed_sets_status_from_pnl_sign(pnl, expected_status):
    tj.record_proposals(_payload())
    tj.record_order_result("XAUUSD", "open", True, "placed")
    tj.record_filled("XAUUSD", 1, 4360.0)
    tj.record_closed("XAUUSD", pnl, "take_profit_hit")
    stories = [s for s in tj.list_all_stories() if s.symbol == "XAUUSD"]
    assert stories[0].status == expected_status
    assert tj.find_open_story("XAUUSD") is None


def test_record_closed_with_none_pnl_is_unknown_never_a_win():
    # Real bug 2026-09-22: an unmatched deal used to be stamped "closed_won"
    # (MSFT, a real losing/unknown trade, showed as a win with no P&L).
    tj.record_proposals(_payload())
    tj.record_closed("XAUUSD", None, "manual_or_unknown")
    stories = [s for s in tj.list_all_stories() if s.symbol == "XAUUSD"]
    assert stories[0].status == "closed_unknown"
    assert stories[0].events[-1].data["net_pnl"] is None
    assert tj.find_open_story("XAUUSD") is None  # still terminal


class _FakeClosedTrade:
    def __init__(self, symbol, position_id, closed_at, profit, open_price=100.0, close_price=99.0, volume=1.0):
        self.symbol, self.position_id, self.closed_at, self.profit = symbol, position_id, closed_at, profit
        self.open_price, self.close_price, self.volume = open_price, close_price, volume


def _unknown_close_story(pnl_status="closed_unknown"):
    tj.record_proposals(_payload())
    tj.record_order_result("XAUUSD", "open", True, "placed")
    tj.record_filled("XAUUSD", 555, 4360.0)
    tj.record_closed("XAUUSD", None, "manual_or_unknown")
    story = next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    if pnl_status != "closed_unknown":  # legacy shape: stamped "closed_won" with a null P&L
        story.status = pnl_status
        tj._save_story(story)
    return story


def test_reconcile_fills_in_real_pnl_and_flips_status_to_lost():
    from datetime import datetime, timedelta, timezone

    story = _unknown_close_story()
    closed_at = datetime.fromisoformat(story.events[-1].timestamp_utc)
    trade = _FakeClosedTrade("XAUUSD", 555, closed_at + timedelta(hours=2), -123.45)
    assert tj.has_unknown_closures()
    assert len(tj.reconcile_unknown_closures([trade], lambda t, s: "stop_loss_hit")) == 1
    fixed = next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    assert fixed.status == "closed_lost"
    assert fixed.events[-1].type == "closed_reconciled"
    assert fixed.events[-1].data["net_pnl"] == -123.45
    assert fixed.events[-1].data["cause"] == "stop_loss_hit"
    assert not tj.has_unknown_closures()
    assert tj.reconcile_unknown_closures([trade]) == []  # idempotent


def test_reconcile_also_repairs_a_legacy_closed_won_with_null_pnl():
    from datetime import datetime, timedelta

    story = _unknown_close_story(pnl_status="closed_won")
    closed_at = datetime.fromisoformat(story.events[-1].timestamp_utc)
    trade = _FakeClosedTrade("XAUUSD", 555, closed_at, -50.0)
    assert len(tj.reconcile_unknown_closures([trade])) == 1
    assert next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD").status == "closed_lost"


def test_reconcile_ignores_other_symbols_and_far_away_trades():
    from datetime import datetime, timedelta

    story = _unknown_close_story()
    closed_at = datetime.fromisoformat(story.events[-1].timestamp_utc)
    wrong_symbol = _FakeClosedTrade("EURUSD", 555, closed_at, 10.0)
    too_old = _FakeClosedTrade("XAUUSD", 999, closed_at - timedelta(days=3), 10.0)
    assert tj.reconcile_unknown_closures([wrong_symbol, too_old]) == []
    assert next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD").status == "closed_unknown"


def test_record_order_result_records_the_terms_actually_sent():
    tj.record_proposals(_payload())
    terms = {"price": 4361.5, "stop_loss": 4330.0, "take_profit": 4404.0, "volume": 0.12}
    tj.record_order_result("XAUUSD", "open", True, "placed", terms=terms)
    story = tj.find_open_story("XAUUSD")
    assert story.events[-1].data["terms"] == terms
    assert "Actually sent: price 4361.5" in tj.render_story_note(story)


def _fake_story(symbol, timestamp, status="closed_won"):
    """Directly constructs and saves a story with a controlled story_id,
    bypassing the real record_* flow (which always uses datetime.now())
    -- lets ordering tests assert against known, deterministic timestamps
    instead of relying on real sleep() delays between symbols."""
    story = tj.TradeStory(
        symbol=symbol,
        story_id=f"{symbol}_{timestamp}",
        status=status,
        events=[tj.TradeStoryEvent(type="proposed", timestamp_utc=f"{timestamp}Z", data={"side": "buy"})],
    )
    if status in ("closed_won", "closed_lost"):
        story.events.append(tj.TradeStoryEvent(type="closed", timestamp_utc=f"{timestamp}Z", data={"net_pnl": 1.0, "cause": "take_profit_hit"}))
    tj._save_story(story)
    return story


def test_list_all_stories_orders_by_real_timestamp_not_symbol_name():
    # Real bug caught live 2026-09-20: sorting the raw filenames sorts by
    # SYMBOL NAME first (filenames are "{symbol}_{timestamp}.json"), not
    # chronologically -- ZEBRA closing before APPLE would still sort
    # ZEBRA last alphabetically. This locks in the fix: sort on the
    # timestamp segment alone, most-recent-first.
    _fake_story("ZEBRA", "2026-09-19_100000")
    _fake_story("APPLE", "2026-09-19_110000")
    _fake_story("MANGO", "2026-09-19_120000")
    assert [s.symbol for s in tj.list_all_stories()] == ["MANGO", "APPLE", "ZEBRA"]


def test_find_unaudited_closed_stories_returns_oldest_first_not_newest_first():
    # Real bug caught live 2026-09-20: find_unaudited_closed_stories used
    # to inherit list_all_stories()'s newest-first order, then
    # ai.trade_audit.run_trade_audit_check took the first N -- meaning
    # the NEWEST closed trades always won the daily audit slots and an
    # older backlogged trade could starve forever if trades kept closing
    # faster than the daily cap. Oldest-first guarantees fair draining.
    _fake_story("ZEBRA", "2026-09-19_100000")  # closed FIRST -- must be audited first
    _fake_story("APPLE", "2026-09-19_110000")
    _fake_story("MANGO", "2026-09-19_120000")  # closed LAST -- must be audited last
    assert [s.symbol for s in tj.find_unaudited_closed_stories()] == ["ZEBRA", "APPLE", "MANGO"]


def test_list_all_stories_empty_when_nothing_recorded():
    assert tj.list_all_stories() == []


def test_render_story_note_never_raises_and_includes_every_event():
    tj.record_proposals(_payload())
    tj.record_order_result("XAUUSD", "open", True, "placed")
    tj.record_filled("XAUUSD", 1, 4360.0)
    tj.record_closed("XAUUSD", 20.0, "take_profit_hit")
    story = tj.find_open_story("XAUUSD") or next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    note = tj.render_story_note(story)
    assert "XAUUSD" in note
    assert "Proposed" in note
    assert "Filled" in note
    assert "Closed" in note


def test_a_record_call_never_raises_even_if_saving_fails(monkeypatch):
    tj.record_proposals(_payload())

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tj, "_save_story", _boom)
    # None of these should raise -- the whole "never puncture existing
    # logic" contract depends on this.
    tj.record_order_result("XAUUSD", "open", True, "placed")
    tj.record_clerk_check("XAUUSD", "invalidation", False, "x")
    tj.record_tactical_action("XAUUSD", "defend", True, "", "x")
    tj.record_filled("XAUUSD", 1, 1.0)
    tj.record_closed("XAUUSD", 1.0, "take_profit_hit")


def _closed_story(symbol="XAUUSD", pnl=20.0, cause="take_profit_hit"):
    tj.record_proposals(_payload(symbol))
    tj.record_order_result(symbol, "open", True, "placed")
    tj.record_filled(symbol, 1, 4360.0)
    tj.record_closed(symbol, pnl, cause)
    return next(s for s in tj.list_all_stories() if s.symbol == symbol)


def test_find_unaudited_closed_stories_finds_a_real_closed_trade():
    story = _closed_story("XAUUSD", pnl=20.0)
    pending = tj.find_unaudited_closed_stories()
    assert [s.story_id for s in pending] == [story.story_id]


def test_find_unaudited_closed_stories_finds_both_wins_and_losses():
    _closed_story("XAUUSD", pnl=20.0)
    _closed_story("GBPUSD", pnl=-5.0)
    symbols = {s.symbol for s in tj.find_unaudited_closed_stories()}
    assert symbols == {"XAUUSD", "GBPUSD"}


def test_find_unaudited_closed_stories_excludes_an_already_audited_story():
    story = _closed_story("XAUUSD")
    tj.record_audit(story, "some review", True)
    assert tj.find_unaudited_closed_stories() == []


def test_find_unaudited_closed_stories_excludes_non_terminal_and_other_terminal_statuses():
    # proposed only -- not closed at all
    tj.record_proposals(_payload("EURUSD"))
    # order rejected -- terminal, but not a real trade outcome
    tj.record_proposals(_payload("USDCAD"))
    tj.record_order_result("USDCAD", "open", False, "no money")
    # a real open position -- not resolved yet
    tj.record_proposals(_payload("GBPUSD"))
    tj.record_order_result("GBPUSD", "open", True, "placed")
    tj.record_filled("GBPUSD", 1, 1.33)

    assert tj.find_unaudited_closed_stories() == []


def test_record_audit_appends_event_without_changing_status():
    story = _closed_story("XAUUSD", pnl=20.0)
    assert story.status == "closed_won"
    tj.record_audit(story, "a real 3-model review", True)
    updated = next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    assert updated.status == "closed_won"  # unchanged
    assert updated.events[-1].type == "audit"
    assert updated.events[-1].data == {"block": "a real 3-model review", "audit_available": True}


def test_record_audit_never_raises_even_if_saving_fails(monkeypatch):
    story = _closed_story("XAUUSD")

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tj, "_save_story", _boom)
    tj.record_audit(story, "block", True)  # must not raise


def test_render_story_note_includes_the_audit_section_when_available():
    story = _closed_story("XAUUSD")
    tj.record_audit(story, "Group A says do X, Group B says do Y.", True)
    updated = next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    note = tj.render_story_note(updated)
    assert "Retrospective audit" in note
    assert "Group A says do X" in note


def test_render_story_note_shows_unavailable_message_when_audit_not_available():
    story = _closed_story("XAUUSD")
    tj.record_audit(story, "", False)
    updated = next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    note = tj.render_story_note(updated)
    assert "no independent model was available" in note


def test_render_story_note_links_to_trade_audit_hub_only_once_audited():
    story = _closed_story("XAUUSD")
    note_before = tj.render_story_note(story)
    assert "[[Trade Audit]]" not in note_before
    assert "audited" not in note_before  # not yet in the tags line either

    tj.record_audit(story, "Group A says do X.", True)
    updated = next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    note_after = tj.render_story_note(updated)
    assert "[[Trade Audit]]" in note_after
    assert "audited" in note_after


def _closed_story_with_terms():
    tj.record_proposals(_payload(reason="thesis text"))
    for _ in range(3):
        tj.record_clerk_check("XAUUSD", "pending_setup_trigger", False, "not yet")
    tj.record_order_result(
        "XAUUSD", "open", True, "placed",
        terms={"price": 4361.5, "stop_loss": 4330.0, "take_profit": 4404.0, "volume": 0.12},
    )
    tj.record_filled("XAUUSD", 555, 4360.0)
    tj.record_tactical_action("XAUUSD", "defend", True, "", "tighten it")
    tj.record_closed("XAUUSD", None, "manual_or_unknown")
    return tj.find_open_story("XAUUSD") or next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")


def test_summarize_story_exposes_status_timestamps_terms_and_analysis():
    story = _closed_story_with_terms()
    m = tj.summarize_story(story)
    assert m["status_label"] == "Closed — P&L not yet matched"
    assert m["side"] == "buy" and m["thesis"] == "thesis text"
    assert m["proposed_at"] and m["order_placed_at"] and m["filled_at"] and m["closed_at"]
    assert (m["planned_price"], m["order_price"], m["order_stop"], m["order_volume"]) == (4360.0, 4361.5, 4330.0, 0.12)
    assert m["analysis"]["label"].startswith("Clerk tactical DEFEND") and m["analysis"]["text"] == "tighten it"
    assert m["net_pnl"] is None


def test_summarize_story_closed_at_is_the_poll_time_not_the_reconcile_time():
    from datetime import datetime, timedelta

    story = _closed_story_with_terms()
    poll_closed_at = story.events[-1].timestamp_utc
    trade = _FakeClosedTrade("XAUUSD", 555, datetime.fromisoformat(poll_closed_at), -5.0)
    tj.reconcile_unknown_closures([trade])
    m = tj.summarize_story(next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD"))
    assert m["closed_at"] == poll_closed_at
    assert m["net_pnl"] == -5.0 and m["status"] == "closed_lost"


def test_render_story_note_leads_with_a_summary_and_collapses_repeated_checks():
    story = _closed_story_with_terms()
    note = tj.render_story_note(story)
    assert note.index("## Summary") < note.index("## Event log")
    assert "**Thesis:** thesis text" in note
    assert "Actually sent" in note and "Order Clerk actually sent:** entry 4361.5" in note
    assert "clerk_check ×3" in note  # three identical not-confirmed polls -> one line
    assert note.count("**Clerk pending_setup_trigger**") == 1
    assert len(story.events) == 8  # the JSON itself still keeps every event


def test_order_terms_render_the_guards_that_changed_the_numbers():
    tj.record_proposals(_payload())
    terms = {
        "price": 95.0, "stop_loss": 91.0, "take_profit": 110.0, "volume": 0.5,
        "guards": ["[Stale-entry re-anchor: moved to M5 zone]", "[Intraday size scalar x0.50: vol]"],
    }
    tj.record_order_result("XAUUSD", "open", True, "placed", terms=terms)
    note = tj.render_story_note(tj.find_open_story("XAUUSD"))
    assert "  - [Stale-entry re-anchor: moved to M5 zone]" in note
    assert "  - [Intraday size scalar x0.50: vol]" in note


# --- Continuation Watch: find_story_by_id / record_continuation_watch ------------------------------------

def test_find_story_by_id_returns_the_exact_story():
    story = _closed_story("XAUUSD", pnl=47.15)
    found = tj.find_story_by_id(story.story_id)
    assert found is not None and found.story_id == story.story_id and found.status == "closed_won"


def test_find_story_by_id_returns_none_for_an_unknown_id():
    assert tj.find_story_by_id("NOPE_2026-01-01_000000") is None


def test_record_continuation_watch_appends_event_without_changing_status():
    story = _closed_story("XAUUSD", pnl=47.15)
    tj.record_continuation_watch(story.story_id, "proposed", {"move_atr": 4.5, "entry": 116.5})
    updated = tj.find_story_by_id(story.story_id)
    assert updated.status == "closed_won"  # unchanged -- this is retrospective evidence, not a status change
    assert updated.events[-1].type == "continuation_watch"
    assert updated.events[-1].data == {"outcome": "proposed", "move_atr": 4.5, "entry": 116.5}


def test_record_continuation_watch_is_a_no_op_for_an_unknown_story_id():
    tj.record_continuation_watch("NOPE_2026-01-01_000000", "no_continuation", {})  # must not raise
    assert tj.find_story_by_id("NOPE_2026-01-01_000000") is None


def test_record_continuation_watch_never_raises_even_if_saving_fails(monkeypatch):
    story = _closed_story("XAUUSD")

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tj, "_save_story", _boom)
    tj.record_continuation_watch(story.story_id, "skipped", {})  # must not raise
