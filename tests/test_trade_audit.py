from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import config
from ai import trade_audit as ta
from ai import trade_journal as tj
from ai.portfolio_suggest import AuditResult


@pytest.fixture(autouse=True)
def _isolated(tmp_path):
    with (
        patch.object(config, "TRADE_JOURNAL_DIR", str(tmp_path / "ftmo_trade_journal")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch.object(config, "TRADE_AUDIT_ENABLED_FILE", str(tmp_path / "trade_audit_enabled.json")),
        patch.object(config, "TRADE_AUDIT_STATE_FILE", str(tmp_path / "trade_audit_state.json")),
        patch.object(config, "TRADE_AUDIT_TRIGGER_HOUR_LOCAL", 17),
        patch.object(config, "TRADE_AUDIT_TRIGGER_MINUTE_LOCAL", 0),
        patch.object(config, "TRADE_AUDIT_TRIGGER_TZ", "America/New_York"),
        patch.object(config, "TRADE_AUDIT_SETTLEMENT_BUFFER_MINUTES", 45),
        patch.object(config, "TRADE_AUDIT_GRACE_MINUTES", 120),
        patch.object(config, "TRADE_AUDIT_MAX_STORIES_PER_RUN", 3),
    ):
        yield


def _payload(symbol="XAUUSD", reason="aligned uptrend"):
    return {
        "immediate_allocation": {
            symbol: {
                "pct": 2.0, "price": 4360.0, "stop_loss": 4325.0, "take_profit": 4404.0,
                "side": "buy", "reason": reason, "invalidation_condition": "H4 closes below 4325.00",
            }
        },
        "pending_setups": [],
    }


def _closed_story(symbol="XAUUSD", pnl=20.0, cause="take_profit_hit"):
    tj.record_proposals(_payload(symbol))
    tj.record_order_result(symbol, "open", True, "placed")
    tj.record_filled(symbol, 1, 4360.0)
    tj.record_clerk_check(symbol, "invalidation", False, "still holds, price above 4325")
    tj.record_tactical_action(symbol, "defend", True, "", "tightening stop on adverse move")
    tj.record_closed(symbol, pnl, cause)
    return next(s for s in tj.list_all_stories() if s.symbol == symbol)


# --- _ny_close_trigger_utc / is_trade_audit_due -----------------------


def test_ny_close_trigger_utc_is_21_utc_during_edt():
    assert ta._ny_close_trigger_utc(date(2026, 7, 15)) == datetime(2026, 7, 15, 21, 0, tzinfo=timezone.utc)


def test_ny_close_trigger_utc_is_22_utc_during_est():
    assert ta._ny_close_trigger_utc(date(2026, 1, 15)) == datetime(2026, 1, 15, 22, 0, tzinfo=timezone.utc)


def test_is_trade_audit_due_false_before_the_settlement_buffer_elapses():
    now = datetime(2026, 7, 15, 21, 10, tzinfo=timezone.utc)  # close was 21:00, buffer is 45 min
    assert ta.is_trade_audit_due(now, state={}) is False


def test_is_trade_audit_due_true_inside_the_grace_window():
    now = datetime(2026, 7, 15, 21, 50, tzinfo=timezone.utc)  # due_from = 21:45
    assert ta.is_trade_audit_due(now, state={}) is True


def test_is_trade_audit_due_false_once_the_grace_window_closes():
    now = datetime(2026, 7, 16, 0, 0, tzinfo=timezone.utc)  # due_until = 21:45 + 120min = 23:45
    assert ta.is_trade_audit_due(now, state={}) is False


def test_is_trade_audit_due_false_if_already_run_today():
    now = datetime(2026, 7, 15, 21, 50, tzinfo=timezone.utc)
    state = {"last_run_date_utc": "2026-07-15"}
    assert ta.is_trade_audit_due(now, state=state) is False


# --- build_case_file ---------------------------------------------------


def test_build_case_file_sections_in_order_with_full_untruncated_text():
    long_text = "x" * 500 + " CRITICAL WARNING TEXT"
    story = _closed_story("XAUUSD")
    # overwrite the clerk_check event's raw_text with something long, to prove no truncation
    for event in story.events:
        if event.type == "clerk_check":
            event.data["raw_text"] = long_text
    case_file = ta.build_case_file(story)
    assert "Mega Session's original reasoning" in case_file
    assert "Clerk's real management" in case_file
    assert "Real final outcome" in case_file
    assert long_text in case_file  # untruncated, unlike the 400-char Obsidian preview
    # section order: reasoning before management before outcome
    assert case_file.index("original reasoning") < case_file.index("real management")
    assert case_file.index("real management") < case_file.index("final outcome")


def test_build_case_file_excludes_superseded_events():
    tj.record_proposals(_payload("XAUUSD"))
    tj.record_proposals(_payload("GBPUSD"))  # XAUUSD superseded -> expired_unfilled
    story = next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    case_file = ta.build_case_file(story)
    assert "superseded" not in case_file.lower()


# --- run_trade_audit_check / _audit_one_story --------------------------


def test_run_trade_audit_check_makes_zero_audit_calls_when_nothing_pending():
    with patch("ai.trade_audit.build_audit_block") as mock_build:
        ta.run_trade_audit_check()
        mock_build.assert_not_called()
    state = ta.read_trade_audit_state()
    assert state["last_status"] == "success"
    assert state["last_audited_count"] == 0


def test_run_trade_audit_check_audits_each_pending_story():
    _closed_story("XAUUSD")
    _closed_story("GBPUSD")
    fake_result = AuditResult(block="Group A: do X. Group B: do Y. Group C: no warnings ignored.", audit_available=True)
    with patch("ai.trade_audit.build_audit_block", return_value=fake_result) as mock_build:
        ta.run_trade_audit_check()
        assert mock_build.call_count == 2
    audited = {s.symbol: s for s in tj.list_all_stories()}
    assert any(e.type == "audit" for e in audited["XAUUSD"].events)
    assert any(e.type == "audit" for e in audited["GBPUSD"].events)
    state = ta.read_trade_audit_state()
    assert state["last_audited_count"] == 2
    assert state["last_failed_count"] == 0


def test_run_trade_audit_check_respects_the_max_stories_per_run_cap():
    _closed_story("XAUUSD")
    _closed_story("GBPUSD")
    _closed_story("EURUSD")
    fake_result = AuditResult(block="review", audit_available=True)
    with patch.object(config, "TRADE_AUDIT_MAX_STORIES_PER_RUN", 2), patch(
        "ai.trade_audit.build_audit_block", return_value=fake_result
    ) as mock_build:
        ta.run_trade_audit_check()
        assert mock_build.call_count == 2


def test_run_trade_audit_check_skips_recording_when_audit_unavailable():
    _closed_story("XAUUSD")
    fake_result = AuditResult(block="", audit_available=False)
    with patch("ai.trade_audit.build_audit_block", return_value=fake_result):
        ta.run_trade_audit_check()
    story = next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    assert not any(e.type == "audit" for e in story.events)  # left pending, not falsely marked done
    state = ta.read_trade_audit_state()
    assert state["last_failed_count"] == 1


def test_run_trade_audit_check_continues_past_one_storys_exception():
    _closed_story("XAUUSD")
    _closed_story("GBPUSD")
    fake_result = AuditResult(block="review", audit_available=True)

    def _side_effect(*args, **kwargs):
        # fail deterministically for the first call, succeed for the rest
        if not _side_effect.called:
            _side_effect.called = True
            raise RuntimeError("boom")
        return fake_result

    _side_effect.called = False
    with patch("ai.trade_audit.build_audit_block", side_effect=_side_effect):
        ta.run_trade_audit_check()  # must not raise
    state = ta.read_trade_audit_state()
    assert state["last_audited_count"] == 1
    assert state["last_failed_count"] == 1


def test_trade_audit_focus_groups_has_exactly_one_entry_per_audit_model():
    from ai.portfolio_suggest import AUDIT_MODELS

    assert len(ta.TRADE_AUDIT_FOCUS_GROUPS) == len(AUDIT_MODELS)


# --- Audit fixes 2026-09-24: the models must be told the outcome and what "pct" means ---


def test_case_file_states_win_or_loss_and_the_real_pnl_at_the_top_of_the_outcome():
    win = ta.build_case_file(_closed_story("XAUUSD", pnl=20.0))
    assert "OUTCOME: WIN — real net P&L +20.00, cause of closure take_profit_hit." in win
    assert win.index("OUTCOME: WIN") < win.index("### ", win.index("## 3. Real final outcome"))
    loss = ta.build_case_file(_closed_story("GBPUSD", pnl=-62.82, cause="stop_loss_hit"))
    assert "OUTCOME: LOSS — real net P&L -62.82, cause of closure stop_loss_hit." in loss


def test_case_file_never_lets_a_model_guess_when_the_pnl_is_unknown():
    text = ta._outcome_headline({"net_pnl": None, "cause": "manual_or_unknown"})
    assert "NOT YET KNOWN" in text and "do not guess win or loss" in text
    assert "BREAKEVEN" in ta._outcome_headline({"net_pnl": 0.0, "cause": "x"})


def test_case_file_uses_the_reconciled_pnl_once_it_exists():
    story = _closed_story("XAUUSD", pnl=None, cause="manual_or_unknown")
    assert "NOT YET KNOWN" in ta.build_case_file(story)
    tj._append_event(story, "closed_reconciled", {"net_pnl": -62.82, "cause": "manual_or_unknown"})
    story = next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    case_file = ta.build_case_file(story)
    assert "OUTCOME: LOSS — real net P&L -62.82" in case_file and "NOT YET KNOWN" not in case_file


def test_audit_context_defines_pct_as_equity_risked_and_forbids_cross_timeframe_atr():
    flat = " ".join(ta._CASE_FILE_CONTEXT_STUB.split())
    assert "% of ACCOUNT EQUITY RISKED" in flat and "NOT the position's size" in flat
    assert "never attribute one timeframe's figure to another" in flat
    assert "pct" in " ".join(ta.TRADE_AUDIT_INSTRUCTION.split()) and "equity risked" in " ".join(ta.TRADE_AUDIT_INSTRUCTION.split())


def test_case_file_uses_the_original_close_time_not_the_later_reconciliation_time():
    story = _closed_story("XAUUSD", pnl=None, cause="manual_or_unknown")
    original_close = next(e for e in story.events if e.type == "closed").timestamp_utc
    tj._append_event(story, "closed_reconciled", {"net_pnl": -62.82, "cause": "manual_or_unknown"})
    story = next(s for s in tj.list_all_stories() if s.symbol == "XAUUSD")
    reconciled_at = next(e for e in story.events if e.type == "closed_reconciled").timestamp_utc
    case_file = ta.build_case_file(story)
    assert f"The position actually closed at {original_close}." in case_file
    assert f"### {original_close} — closed" in case_file and f"### {reconciled_at}" not in case_file
    assert "not the close time" in case_file


def test_audit_context_explains_stop_direction_and_the_never_widen_rejection():
    flat = " ".join(ta._CASE_FILE_CONTEXT_STUB.split())
    assert "for a LONG (buy) a MORE protective stop is a HIGHER price" in flat
    assert "not a comparison bug" in flat
