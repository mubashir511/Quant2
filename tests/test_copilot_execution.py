import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import config
from ai.copilot_cli import CLI_FAILED_PREFIX as COPILOT_FAILED_PREFIX
from ai.copilot_cli import CLI_MISSING_MESSAGE as COPILOT_MISSING_MESSAGE
from ai.copilot_execution import (
    COPILOT_BACKUP_MODELS,
    SymbolSettlement,
    _build_carried_forward_allocation,
    _is_copilot_unavailable,
    _is_openrouter_unavailable,
    _mega_session_is_live,
    _reconcile_settlement,
    _reset_settlement_for_new_session,
    _run_clerk_prompt,
    execution_check_is_live,
    is_execution_due,
    next_execution_check_utc,
    parse_copilot_verdict,
    read_copilot_execution_enabled,
    read_copilot_execution_interval_minutes,
    read_execution_progress,
    read_execution_state,
    read_settlement,
    run_copilot_execution_check,
    set_copilot_execution_enabled,
    set_copilot_execution_interval_minutes,
)
from ai.openrouter_client import FAILED_MESSAGE as OPENROUTER_FAILED_MESSAGE
from ai.openrouter_client import MISSING_KEY_MESSAGE as OPENROUTER_MISSING_KEY_MESSAGE
from ai.portfolio_suggest import AllocationEntry, PendingSetup
from data.mt5_execution import OrderResult
from data.mt5_source import PendingOrder, Position


@pytest.fixture(autouse=True)
def _fixed_files(tmp_path):
    with (
        patch.object(config, "COPILOT_EXECUTION_STATE_FILE", str(tmp_path / "copilot_execution_state.json")),
        patch.object(config, "COPILOT_EXECUTION_SETTLEMENT_FILE", str(tmp_path / "copilot_execution_settlement.json")),
        patch.object(config, "COPILOT_EXECUTION_PROGRESS_FILE", str(tmp_path / "copilot_execution_progress.json")),
        patch.object(config, "MEGA_ANALYSIS_LATEST_SUGGESTION_FILE", str(tmp_path / "mega_analysis_latest_suggestion.json")),
        patch.object(config, "MEGA_ANALYSIS_STATE_FILE", str(tmp_path / "mega_analysis_state.json")),
        patch.object(config, "MEGA_ANALYSIS_PROGRESS_FILE", str(tmp_path / "mega_analysis_progress.json")),
        patch.object(config, "COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES", 15),
        patch.object(config, "COPILOT_EXECUTION_GRACE_MINUTES", 10),
        patch.object(config, "COPILOT_EXECUTION_MAX_PENDING_SETUPS", 10),
        patch.object(config, "COPILOT_EXECUTION_MAX_WATCHED_POSITIONS", 10),
        patch.object(config, "AMEND_TOLERANCE_PCT", 0.05),
        # Isolated proactively (2026-08-23) — the mega-analysis enabled
        # toggle already caused a real collision the same day where the
        # live app's own genuine use of a brand-new toggle broke tests
        # relying on the real project-root default; done here up front
        # for the Copilot equivalent instead of waiting for a repeat.
        patch.object(config, "COPILOT_EXECUTION_ENABLED_FILE", str(tmp_path / "copilot_execution_enabled.json")),
        patch.object(config, "COPILOT_EXECUTION_INTERVAL_FILE", str(tmp_path / "copilot_execution_interval.json")),
    ):
        yield tmp_path


def _position(symbol="EURUSD", side="buy", volume=1.0, ticket=1):
    return Position(
        symbol=symbol, volume=volume, side=side, price_open=1.09, price_current=1.10,
        sl=1.08, profit=10.0, opened_at=datetime.now(), ticket=ticket,
    )


def _pending_order(symbol="EURUSD", ticket=100):
    return PendingOrder(symbol=symbol, volume=1.0, order_type="buy limit", price_open=1.09, sl=1.08, tp=1.11, ticket=ticket)


# --- parse_copilot_verdict: the full fail-safe branch table ---


def test_parse_copilot_verdict_true_on_clean_confirmed():
    assert parse_copilot_verdict("Some reasoning.\n\nFINAL_VERDICT: CONFIRMED") is True


def test_parse_copilot_verdict_false_on_clean_not_confirmed():
    assert parse_copilot_verdict("Some reasoning.\n\nFINAL_VERDICT: NOT_CONFIRMED") is False


def test_parse_copilot_verdict_false_on_empty_string():
    assert parse_copilot_verdict("") is False


def test_parse_copilot_verdict_false_on_whitespace_only():
    assert parse_copilot_verdict("   \n  ") is False


def test_parse_copilot_verdict_false_when_cli_missing():
    assert parse_copilot_verdict(COPILOT_MISSING_MESSAGE) is False


def test_parse_copilot_verdict_false_when_cli_failed():
    assert parse_copilot_verdict(f"{COPILOT_FAILED_PREFIX} (timed out).") is False


def test_parse_copilot_verdict_false_when_no_token_present():
    assert parse_copilot_verdict("I think this looks like a good trade.") is False


def test_parse_copilot_verdict_false_on_unrecognized_value():
    assert parse_copilot_verdict("FINAL_VERDICT: MAYBE") is False


def test_parse_copilot_verdict_last_occurrence_wins_when_model_restates():
    text = "Initially I thought FINAL_VERDICT: CONFIRMED but on reflection FINAL_VERDICT: NOT_CONFIRMED"
    assert parse_copilot_verdict(text) is False


def test_parse_copilot_verdict_last_occurrence_wins_the_other_direction():
    text = "FINAL_VERDICT: NOT_CONFIRMED -- wait, actually FINAL_VERDICT: CONFIRMED"
    assert parse_copilot_verdict(text) is True


def test_parse_copilot_verdict_case_insensitive():
    assert parse_copilot_verdict("final_verdict: confirmed") is True


# --- settlement reconciliation: the state machine transitions ---


def test_reconcile_order_placed_stays_order_placed_while_still_pending():
    settled = {"EURUSD": {"origin": "immediate", "state": "order_placed", "entry": {}, "order_ticket": 100}}
    updated = _reconcile_settlement(settled, positions=[], pending_orders=[_pending_order(ticket=100)])
    assert updated["EURUSD"]["state"] == "order_placed"


def test_reconcile_order_placed_becomes_filled_when_ticket_gone_and_symbol_held():
    settled = {"EURUSD": {"origin": "immediate", "state": "order_placed", "entry": {}, "order_ticket": 100}}
    updated = _reconcile_settlement(settled, positions=[_position(ticket=5)], pending_orders=[])
    assert updated["EURUSD"]["state"] == "filled"


def test_reconcile_order_placed_is_dropped_when_ticket_gone_and_not_held():
    # Rejected/cancelled, never filled -- eligible for a fresh check.
    settled = {"EURUSD": {"origin": "immediate", "state": "order_placed", "entry": {}, "order_ticket": 100}}
    updated = _reconcile_settlement(settled, positions=[], pending_orders=[])
    assert "EURUSD" not in updated


def test_reconcile_filled_stays_filled_while_still_held():
    settled = {"EURUSD": {"origin": "immediate", "state": "filled", "entry": {}, "order_ticket": 100}}
    updated = _reconcile_settlement(settled, positions=[_position(ticket=5)], pending_orders=[])
    assert updated["EURUSD"]["state"] == "filled"


def test_reconcile_filled_becomes_closed_after_fill_when_no_longer_held():
    settled = {"EURUSD": {"origin": "immediate", "state": "filled", "entry": {}, "order_ticket": 100}}
    updated = _reconcile_settlement(settled, positions=[], pending_orders=[])
    assert updated["EURUSD"]["state"] == "closed_after_fill"


def test_reconcile_closed_after_fill_is_permanent():
    settled = {"EURUSD": {"origin": "immediate", "state": "closed_after_fill", "entry": {}, "order_ticket": 100}}
    # Even if the symbol somehow reappears held, closed_after_fill never reverts.
    updated = _reconcile_settlement(settled, positions=[_position(ticket=5)], pending_orders=[])
    assert updated["EURUSD"]["state"] == "closed_after_fill"


def test_closed_after_fill_symbol_never_reappears_in_a_later_merge():
    settled = {"EURUSD": {"origin": "immediate", "state": "closed_after_fill", "entry": {"pct": 1.0, "side": "buy"}, "order_ticket": 100}}
    immediate_allocation_raw = {"EURUSD": {"pct": 1.0, "side": "buy"}}
    carried = _build_carried_forward_allocation(immediate_allocation_raw, settled)
    assert "EURUSD" not in carried


def test_immediate_allocation_order_placed_symbol_is_still_carried_forward():
    # Real bug found on audit: excluding an order_placed symbol here
    # entirely would make compute_rebalance_plan see an ALREADY-HELD
    # position (e.g. a filled base position plus a still-unfilled top-up
    # "increase" order) as "held but missing from the target," which
    # defaults to a 0% target and force-closes the whole real position —
    # just because an unrelated top-up order happened to still be
    # pending. It must stay in the merge; not resubmitting a duplicate
    # order is handled separately, at the execution-loop level.
    settled = {"EURUSD": {"origin": "immediate", "state": "order_placed", "entry": {"pct": 1.0, "side": "buy"}, "order_ticket": 100}}
    immediate_allocation_raw = {"EURUSD": {"pct": 1.0, "side": "buy", "price": 1.09, "stop_loss": 1.08}}
    carried = _build_carried_forward_allocation(immediate_allocation_raw, settled)
    assert "EURUSD" in carried
    assert carried["EURUSD"] == AllocationEntry(pct=1.0, price=1.09, stop_loss=1.08, side="buy")


def test_pending_setup_order_placed_symbol_stays_excluded_from_the_merge():
    # Unlike the immediate-allocation case above, a fired-but-unfilled
    # Pending Setup has ZERO held volume yet — nothing to protect from a
    # forced close — so including it here would just resubmit a
    # duplicate fresh-open order every poll instead. Correctly excluded
    # by the second loop, which only carries forward "filled" entries.
    settled = {"XAUUSD": {"origin": "pending_setup", "state": "order_placed", "entry": {"pct": 1.0, "side": "buy"}, "order_ticket": 200}}
    carried = _build_carried_forward_allocation({}, settled)
    assert "XAUUSD" not in carried


def test_filled_symbol_not_in_immediate_allocation_is_carried_forward():
    settled = {"XAUUSD": {"origin": "pending_setup", "state": "filled", "entry": {"pct": 1.0, "side": "buy", "price": 2000.0, "stop_loss": 1980.0}, "order_ticket": 100}}
    carried = _build_carried_forward_allocation({}, settled)
    assert carried["XAUUSD"] == AllocationEntry(pct=1.0, price=2000.0, stop_loss=1980.0, side="buy")


def test_symbol_with_no_settlement_record_is_carried_forward_normally():
    immediate_allocation_raw = {"EURUSD": {"pct": 1.5, "side": "buy", "price": 1.09, "stop_loss": 1.08}}
    carried = _build_carried_forward_allocation(immediate_allocation_raw, {})
    assert carried["EURUSD"] == AllocationEntry(pct=1.5, price=1.09, stop_loss=1.08, side="buy")


def test_carried_forward_allocation_excludes_cash():
    immediate_allocation_raw = {"CASH": {"pct": 98.5, "side": "buy"}}
    carried = _build_carried_forward_allocation(immediate_allocation_raw, {})
    assert carried == {}


def test_carried_forward_allocation_preserves_reason_and_invalidation_condition():
    # Real bug found on self-review 2026-08-24: _allocation_entry_from_dict
    # silently dropped reason/invalidation_condition, so a symbol's own
    # thesis/watch-condition disappeared the moment it was carried forward
    # across a poll — no safety impact (watched_positions reads straight
    # from immediate_allocation_raw, not through this path), but it did
    # mean the amend_position reason text always fell back to a generic
    # message instead of the real thesis.
    immediate_allocation_raw = {
        "EURUSD": {
            "pct": 1.5, "side": "buy", "price": 1.09, "stop_loss": 1.08,
            "reason": "Pullback into H4 support.", "invalidation_condition": "H4 closes below 1.07",
        }
    }
    carried = _build_carried_forward_allocation(immediate_allocation_raw, {})
    assert carried["EURUSD"].reason == "Pullback into H4 support."
    assert carried["EURUSD"].invalidation_condition == "H4 closes below 1.07"


def test_filled_symbol_carry_forward_preserves_reason_and_invalidation_condition():
    # The settlement-entry path (the second loop) must ALSO preserve
    # these fields, since a filled Pending-Setup-originated symbol is
    # carried forward from settlement["settled"][symbol]["entry"], not
    # from immediate_allocation_raw.
    settled = {
        "XAUUSD": {
            "origin": "pending_setup", "state": "filled",
            "entry": {
                "pct": 1.0, "side": "buy", "price": 2000.0, "stop_loss": 1980.0,
                "reason": "Trigger fired.", "invalidation_condition": "H1 RSI below 30",
            },
            "order_ticket": 100,
        }
    }
    carried = _build_carried_forward_allocation({}, settled)
    assert carried["XAUUSD"].reason == "Trigger fired."
    assert carried["XAUUSD"].invalidation_condition == "H1 RSI below 30"


# --- settlement reset: cancels unfilled orders, resets tracking ---


@patch("ai.copilot_execution.cancel_pending_order")
def test_reset_cancels_every_order_placed_symbol(mock_cancel):
    mock_cancel.return_value = OrderResult(success=True, retcode=10009, comment="", ticket=100)
    settled = {
        "EURUSD": {"origin": "immediate", "state": "order_placed", "entry": {}, "order_ticket": 100},
        "XAUUSD": {"origin": "pending_setup", "state": "order_placed", "entry": {}, "order_ticket": 200},
    }
    _reset_settlement_for_new_session(settled)
    assert mock_cancel.call_count == 2
    mock_cancel.assert_any_call(100)
    mock_cancel.assert_any_call(200)


@patch("ai.copilot_execution.cancel_pending_order")
def test_reset_does_not_cancel_filled_or_closed_after_fill_symbols(mock_cancel):
    settled = {
        "EURUSD": {"origin": "immediate", "state": "filled", "entry": {}, "order_ticket": 100},
        "XAUUSD": {"origin": "pending_setup", "state": "closed_after_fill", "entry": {}, "order_ticket": 200},
    }
    _reset_settlement_for_new_session(settled)
    mock_cancel.assert_not_called()


@patch("ai.copilot_execution.cancel_pending_order")
def test_reset_clears_tracking_entirely(mock_cancel):
    mock_cancel.return_value = OrderResult(success=True, retcode=10009, comment="", ticket=100)
    settled = {"EURUSD": {"origin": "immediate", "state": "order_placed", "entry": {}, "order_ticket": 100}}
    result = _reset_settlement_for_new_session(settled)
    assert result == {}


@patch("ai.copilot_execution.cancel_pending_order", side_effect=Exception("should not be reached via non-MT5ConnectionError"))
def test_reset_logs_and_continues_when_cancel_raises_mt5_connection_error(mock_cancel, caplog):
    from data.mt5_source import MT5ConnectionError

    mock_cancel.side_effect = MT5ConnectionError("terminal unreachable")
    settled = {"EURUSD": {"origin": "immediate", "state": "order_placed", "entry": {}, "order_ticket": 100}}
    result = _reset_settlement_for_new_session(settled)  # must not raise
    assert result == {}


# --- interval due-check ---


def _utc(y, m, d, h, mi):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def test_is_execution_due_true_at_start_of_interval_when_not_yet_run():
    assert is_execution_due(_utc(2026, 8, 23, 15, 0), state={}) is True


def test_is_execution_due_true_within_grace_window():
    assert is_execution_due(_utc(2026, 8, 23, 15, 8), state={}) is True


def test_is_execution_due_false_after_grace_window():
    # minute=12 is still inside the 15:00-15:14 interval, but past the
    # 10-minute grace window within it.
    assert is_execution_due(_utc(2026, 8, 23, 15, 12), state={}) is False


def test_is_execution_due_false_when_already_run_this_interval():
    from ai.copilot_execution import _interval_start

    state = {"last_run_interval_utc": _interval_start(_utc(2026, 8, 23, 15, 0)).isoformat()}
    assert is_execution_due(_utc(2026, 8, 23, 15, 5), state=state) is False


def test_is_execution_due_true_again_next_interval():
    from ai.copilot_execution import _interval_start

    state = {"last_run_interval_utc": _interval_start(_utc(2026, 8, 23, 15, 0)).isoformat()}
    # 15:20 falls in the NEXT 15-minute window (15:15-15:29), distinct
    # from the one already marked as run.
    assert is_execution_due(_utc(2026, 8, 23, 15, 20), state=state) is True


# --- enable/disable toggle + review-frequency override (2026-08-23) ---


def test_read_copilot_execution_enabled_true_when_file_missing(_fixed_files):
    assert read_copilot_execution_enabled() is True


def test_read_copilot_execution_enabled_true_on_corrupt_file(_fixed_files, tmp_path):
    corrupt_path = tmp_path / "corrupt_enabled.json"
    corrupt_path.write_text("{not valid json")
    with patch.object(config, "COPILOT_EXECUTION_ENABLED_FILE", str(corrupt_path)):
        assert read_copilot_execution_enabled() is True


def test_set_copilot_execution_enabled_false_then_read_round_trips(_fixed_files):
    set_copilot_execution_enabled(False)
    assert read_copilot_execution_enabled() is False


def test_read_copilot_execution_interval_minutes_falls_back_to_config_when_file_missing(_fixed_files):
    # _fixed_files patches COPILOT_EXECUTION_CHECK_INTERVAL_MINUTES to 15.
    assert read_copilot_execution_interval_minutes() == 15


def test_set_copilot_execution_interval_minutes_then_read_round_trips(_fixed_files):
    set_copilot_execution_interval_minutes(30)
    assert read_copilot_execution_interval_minutes() == 30


def test_read_copilot_execution_interval_minutes_falls_back_on_non_positive_value(_fixed_files, tmp_path):
    bad_path = tmp_path / "bad_interval.json"
    bad_path.write_text(json.dumps({"minutes": 0}))
    with patch.object(config, "COPILOT_EXECUTION_INTERVAL_FILE", str(bad_path)):
        assert read_copilot_execution_interval_minutes() == 15


def test_interval_start_uses_the_overridden_interval(_fixed_files):
    from ai.copilot_execution import _interval_start

    set_copilot_execution_interval_minutes(30)
    # Under the default 15-minute interval, minute=20 buckets to :15; the
    # override to 30 minutes should instead bucket it to :00 — proves
    # _interval_start (and therefore is_execution_due/next_execution_
    # check_utc, which both call it) reads the live override, not the
    # config constant captured at import time.
    assert _interval_start(_utc(2026, 8, 23, 15, 20)) == _utc(2026, 8, 23, 15, 0)


def test_next_execution_check_utc_is_now_when_current_interval_not_yet_run(_fixed_files):
    now = _utc(2026, 8, 23, 15, 5)
    assert next_execution_check_utc(now, state={}) == now


def test_next_execution_check_utc_is_next_window_when_already_run(_fixed_files):
    from ai.copilot_execution import _interval_start

    now = _utc(2026, 8, 23, 15, 5)
    state = {"last_run_interval_utc": _interval_start(now).isoformat()}
    assert next_execution_check_utc(now, state=state) == _utc(2026, 8, 23, 15, 15)


# --- read_execution_state / read_execution_progress: safe defaults ---


def test_read_execution_state_empty_dict_when_file_missing():
    assert read_execution_state() == {}


def test_read_execution_progress_empty_dict_when_file_missing():
    assert read_execution_progress() == {}


def test_read_execution_state_empty_dict_on_corrupt_file(tmp_path):
    corrupt_path = tmp_path / "corrupt_state.json"
    corrupt_path.write_text("{not valid json")
    with patch.object(config, "COPILOT_EXECUTION_STATE_FILE", str(corrupt_path)):
        assert read_execution_state() == {}


# --- _write_execution_state: preserves last_run_interval_utc, merges last_verdicts ---


def test_write_execution_state_preserves_last_run_interval_utc_across_calls(_fixed_files):
    from ai.copilot_execution import _interval_start, _mark_interval_ran, _write_execution_state

    marked_at = datetime(2026, 8, 23, 15, 5, tzinfo=timezone.utc)
    _mark_interval_ran(marked_at)
    # A later write for an unrelated outcome (e.g. a "no_suggestion" poll
    # in a different interval) must not erase the interval dedup marker.
    _write_execution_state("no_suggestion")
    assert read_execution_state()["last_run_interval_utc"] == _interval_start(marked_at).isoformat()


def test_write_execution_state_merges_last_verdicts_instead_of_replacing(_fixed_files):
    from ai.copilot_execution import _write_execution_state

    _write_execution_state("success", last_verdicts={"EURUSD": {"confirmed": False}})
    # A later poll that only checks a DIFFERENT symbol (e.g. because the
    # pending-setups cap truncated the list) must not discard EURUSD's
    # still-relevant verdict from the earlier poll.
    _write_execution_state("success", last_verdicts={"XAUUSD": {"confirmed": True}})
    verdicts = read_execution_state()["last_verdicts"]
    assert verdicts["EURUSD"]["confirmed"] is False
    assert verdicts["XAUUSD"]["confirmed"] is True


def test_write_execution_state_fresh_verdict_overwrites_the_same_symbols_own_prior_one(_fixed_files):
    from ai.copilot_execution import _write_execution_state

    _write_execution_state("success", last_verdicts={"EURUSD": {"confirmed": False}})
    _write_execution_state("success", last_verdicts={"EURUSD": {"confirmed": True}})
    assert read_execution_state()["last_verdicts"]["EURUSD"]["confirmed"] is True


def test_write_execution_state_merges_last_execution_results_instead_of_replacing(_fixed_files):
    from ai.copilot_execution import _write_execution_state

    _write_execution_state("success", last_execution_results={"NVDA": {"success": False, "detail": "Market closed"}})
    _write_execution_state("success", last_execution_results={"BTCUSD": {"success": True, "detail": "placed"}})
    results = read_execution_state()["last_execution_results"]
    assert results["NVDA"]["success"] is False
    assert results["BTCUSD"]["success"] is True


def test_write_execution_state_fresh_execution_result_overwrites_the_same_symbols_own_prior_one(_fixed_files):
    from ai.copilot_execution import _write_execution_state

    _write_execution_state("success", last_execution_results={"NVDA": {"success": False, "detail": "Market closed"}})
    _write_execution_state("success", last_execution_results={"NVDA": {"success": True, "detail": "placed"}})
    assert read_execution_state()["last_execution_results"]["NVDA"]["success"] is True


# --- _describe_elapsed ---


def test_describe_elapsed_unknown_when_generated_utc_missing():
    from ai.copilot_execution import _describe_elapsed

    assert _describe_elapsed(None, datetime.now(timezone.utc)) == "unknown"


def test_describe_elapsed_minutes_only_under_an_hour():
    from ai.copilot_execution import _describe_elapsed

    generated = "2026-08-23T10:00:00+00:00"
    now = _utc(2026, 8, 23, 10, 25)
    assert _describe_elapsed(generated, now) == "25 minute(s)"


def test_describe_elapsed_hours_and_minutes_over_an_hour():
    from ai.copilot_execution import _describe_elapsed

    generated = "2026-08-23T10:00:00+00:00"
    now = _utc(2026, 8, 23, 15, 30)
    assert _describe_elapsed(generated, now) == "5 hour(s) 30 minute(s)"


# --- _mega_session_is_live ---


def test_mega_session_is_live_true_when_progress_is_fresh_and_newer_than_last_attempt():
    now = datetime.now(timezone.utc).isoformat()
    progress = {"updated_utc": now}
    state = {"last_attempt_utc": "2020-01-01T00:00:00+00:00"}
    assert _mega_session_is_live(progress, state) is True


def test_mega_session_is_live_false_when_no_progress():
    assert _mega_session_is_live({}, {}) is False


def test_mega_session_is_live_false_when_progress_is_stale():
    old = "2020-01-01T00:00:00+00:00"
    progress = {"updated_utc": old}
    assert _mega_session_is_live(progress, {}) is False


def test_mega_session_is_live_false_when_progress_is_older_than_last_completed_attempt():
    now = datetime.now(timezone.utc).isoformat()
    progress = {"updated_utc": now}
    state = {"last_attempt_utc": now}  # a completed attempt at least as new as the progress marker
    assert _mega_session_is_live(progress, state) is False


# --- execution_check_is_live ---


def test_execution_check_is_live_true_when_progress_is_fresh_and_newer_than_last_attempt():
    now = datetime.now(timezone.utc).isoformat()
    progress = {"updated_utc": now}
    state = {"last_attempt_utc": "2020-01-01T00:00:00+00:00"}
    assert execution_check_is_live(progress, state) is True


def test_execution_check_is_live_false_when_no_progress():
    assert execution_check_is_live({}, {}) is False


def test_execution_check_is_live_false_when_progress_is_stale():
    old = "2020-01-01T00:00:00+00:00"
    progress = {"updated_utc": old}
    assert execution_check_is_live(progress, {}) is False


def test_execution_check_is_live_false_when_progress_is_older_than_last_completed_attempt():
    now = datetime.now(timezone.utc).isoformat()
    progress = {"updated_utc": now}
    state = {"last_attempt_utc": now}
    assert execution_check_is_live(progress, state) is False


def test_execution_check_is_live_true_within_the_real_run_ceiling_past_the_old_5_minute_cutoff():
    # Real bug found live 2026-08-27 (the same day, and the same root
    # cause, as ai.mega_analysis.mega_session_is_live's own equivalent
    # fix): a flat 5-minute cutoff wrongly hid this panel's live-status
    # banner mid-run whenever a verdict check ran long — e.g. via the
    # OpenRouter backup chain retrying, which was independently
    # confirmed live the same day to sometimes take several minutes
    # across all of its models at once. 8 minutes must still read as
    # live — past the old 300-second cutoff, comfortably under the real
    # config.COPILOT_EXECUTION_RUN_TIMEOUT_SECONDS ceiling (600s default).
    from datetime import timedelta

    old_but_within_run_budget = (datetime.now(timezone.utc) - timedelta(minutes=8)).isoformat()
    progress = {"updated_utc": old_but_within_run_budget}
    state = {"last_attempt_utc": "2020-01-01T00:00:00+00:00"}
    assert execution_check_is_live(progress, state) is True


# --- run_copilot_execution_check: end-to-end with everything mocked ---


def _write_suggestion(tmp_path, immediate_allocation=None, pending_setups=None, generated_utc=None):
    payload = {
        "generated_utc": generated_utc or datetime.now(timezone.utc).isoformat(),
        "immediate_allocation": immediate_allocation or {},
        "pending_setups": pending_setups or [],
    }
    Path(config.MEGA_ANALYSIS_LATEST_SUGGESTION_FILE).write_text(json.dumps(payload))


def test_run_copilot_execution_check_skips_when_disabled(_fixed_files):
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": {"pct": 1.0, "side": "buy"}})
    set_copilot_execution_enabled(False)
    with patch("ai.copilot_execution.connect") as mock_connect:
        run_copilot_execution_check()
        mock_connect.assert_not_called()  # disabled must short-circuit before any MT5 connection
    assert read_execution_state()["last_status"] == "disabled"


def test_run_copilot_execution_check_noop_when_no_suggestion(_fixed_files):
    run_copilot_execution_check()
    assert read_execution_state()["last_status"] == "no_suggestion"


@patch("ai.copilot_execution._mega_session_is_live", return_value=True)
def test_run_copilot_execution_check_skips_when_mega_session_live(mock_live, _fixed_files):
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": {"pct": 1.0, "side": "buy"}})
    run_copilot_execution_check()
    assert read_execution_state()["last_status"] == "skipped_mega_live"


@patch("ai.copilot_execution.open_position")
@patch("ai.copilot_execution.close_position")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions", return_value=[])
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_run_copilot_execution_check_executes_immediate_allocation(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_close, mock_open, _fixed_files,
):
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_open.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=555)

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            immediate_allocation={"EURUSD": {"pct": 1.0, "side": "buy", "price": 1.0900, "stop_loss": 1.0850}},
        )
        run_copilot_execution_check()

    mock_open.assert_called_once()
    state = read_execution_state()
    assert state["last_status"] == "success"


@patch("ai.copilot_execution.open_position")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions", return_value=[])
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_run_copilot_execution_check_records_a_failed_immediate_allocation_attempt(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, _fixed_files,
):
    # Real bug found live 2026-08-23: a FAILED open (e.g. "Market closed"
    # on a Sunday) never got a settlement record — the ONLY place it was
    # ever visible was the log file, since app.py's panel had nothing
    # else to read. last_execution_results is what closes that gap.
    # compute_rebalance_plan itself is mocked here so this test targets
    # the execution-loop's own result-recording, not the real sizing
    # logic (a different, already-covered concern).
    from data.mt5_source import AccountSummary, MarketAsset
    from risk.apply_suggestion import PlannedOrder
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="NVDA", description="Nvidia", bid=214.0, ask=214.4)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_open.return_value = OrderResult(success=False, retcode=10018, comment="Market closed", ticket=None)

    with patch(
        "ai.copilot_execution.compute_rebalance_plan",
        return_value=[
            PlannedOrder(
                symbol="NVDA", action="open", side="buy", volume=0.1,
                order_type="limit", price=214.4, stop_loss=211.8,
            )
        ],
    ):
        _write_suggestion(
            _fixed_files,
            immediate_allocation={"NVDA": {"pct": 0.3, "side": "buy", "price": 214.4, "stop_loss": 211.8}},
        )
        run_copilot_execution_check()

    mock_open.assert_called_once()
    results = read_execution_state()["last_execution_results"]
    assert results["NVDA"]["success"] is False
    assert results["NVDA"]["detail"] == "Market closed"


@patch("ai.copilot_execution.check_execution_safety_gates", return_value=(False, "Execution blocked: test reason."))
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions", return_value=[])
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_run_copilot_execution_check_never_executes_when_safety_gate_blocks(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_gates, _fixed_files,
):
    from data.mt5_source import AccountSummary, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )

    with patch("ai.copilot_execution.open_position") as mock_open:
        _write_suggestion(
            _fixed_files,
            immediate_allocation={"EURUSD": {"pct": 1.0, "side": "buy", "price": 1.0900, "stop_loss": 1.0850}},
        )
        run_copilot_execution_check()
        mock_open.assert_not_called()

    state = read_execution_state()
    assert state["last_status"] == "blocked"


# --- Copilot backup chain (_run_clerk_prompt) ---


def test_copilot_backup_models_has_five_entries_in_priority_order():
    assert COPILOT_BACKUP_MODELS == [
        ("Nvidia Nemotron-Ultra-550B", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        ("Nvidia Nemotron-Super-120B", "nvidia/nemotron-3-super-120b-a12b:free"),
        ("Dots Studio Dots3-Note Preview", "dots-studio/dots-3-note-preview:free"),
        ("Nvidia Nemotron-Nano-Omni-30B-Reasoning", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
        ("Poolside Laguna S 2.1", "poolside/laguna-s-2.1:free"),
    ]


def test_is_copilot_unavailable_true_for_missing_and_failed_sentinels():
    assert _is_copilot_unavailable(COPILOT_MISSING_MESSAGE) is True
    assert _is_copilot_unavailable(f"{COPILOT_FAILED_PREFIX} (quota exceeded).") is True


def test_is_copilot_unavailable_false_for_a_real_response():
    assert _is_copilot_unavailable("Some real reasoning.\nFINAL_VERDICT: NOT_CONFIRMED") is False


def test_is_openrouter_unavailable_true_for_missing_and_failed_sentinels():
    assert _is_openrouter_unavailable(OPENROUTER_MISSING_KEY_MESSAGE) is True
    assert _is_openrouter_unavailable(OPENROUTER_FAILED_MESSAGE) is True


def test_is_openrouter_unavailable_false_for_a_real_response():
    assert _is_openrouter_unavailable("Some real reasoning.\nFINAL_VERDICT: CONFIRMED") is False


@patch("ai.copilot_execution.run_openrouter")
@patch("ai.copilot_execution.run_copilot", return_value="Real reasoning.\nFINAL_VERDICT: NOT_CONFIRMED")
def test_run_clerk_prompt_uses_copilot_when_available(mock_copilot, mock_openrouter):
    result = _run_clerk_prompt("some prompt", timeout=240)
    assert result == "Real reasoning.\nFINAL_VERDICT: NOT_CONFIRMED"
    mock_openrouter.assert_not_called()


@patch("ai.copilot_execution.run_openrouter", return_value="Backup reasoning.\nFINAL_VERDICT: CONFIRMED")
@patch("ai.copilot_execution.run_copilot", return_value=f"{COPILOT_FAILED_PREFIX} (quota exceeded).")
def test_run_clerk_prompt_falls_back_to_first_backup_on_copilot_failure(mock_copilot, mock_openrouter):
    result = _run_clerk_prompt("some prompt", timeout=240)
    assert "Nvidia Nemotron-Ultra-550B" in result
    assert "Backup reasoning.\nFINAL_VERDICT: CONFIRMED" in result
    mock_openrouter.assert_called_once()
    _, kwargs = mock_openrouter.call_args
    assert kwargs["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"


@patch("ai.copilot_execution.run_copilot", return_value=COPILOT_MISSING_MESSAGE)
def test_run_clerk_prompt_treats_cli_missing_message_as_unavailable_too(mock_copilot):
    with patch(
        "ai.copilot_execution.run_openrouter", return_value="Backup reasoning.\nFINAL_VERDICT: CONFIRMED"
    ) as mock_openrouter:
        result = _run_clerk_prompt("some prompt", timeout=240)
    assert "FINAL_VERDICT: CONFIRMED" in result
    mock_openrouter.assert_called_once()


@patch("ai.copilot_execution.run_copilot", return_value=f"{COPILOT_FAILED_PREFIX} (quota exceeded).")
def test_run_clerk_prompt_tries_backups_in_order_until_one_succeeds(mock_copilot):
    with patch(
        "ai.copilot_execution.run_openrouter",
        side_effect=[OPENROUTER_FAILED_MESSAGE, OPENROUTER_FAILED_MESSAGE, "Third backup answered.\nFINAL_VERDICT: CONFIRMED"],
    ) as mock_openrouter:
        result = _run_clerk_prompt("some prompt", timeout=240)
    assert "Dots Studio Dots3-Note Preview" in result
    assert "Third backup answered." in result
    assert mock_openrouter.call_count == 3
    models_tried = [call.kwargs["model"] for call in mock_openrouter.call_args_list]
    assert models_tried == [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "dots-studio/dots-3-note-preview:free",
    ]


@patch("ai.copilot_execution.run_openrouter", return_value=OPENROUTER_FAILED_MESSAGE)
@patch("ai.copilot_execution.run_copilot", return_value=f"{COPILOT_FAILED_PREFIX} (quota exceeded).")
def test_run_clerk_prompt_returns_original_copilot_failure_when_all_backups_fail(mock_copilot, mock_openrouter):
    result = _run_clerk_prompt("some prompt", timeout=240)
    assert result == f"{COPILOT_FAILED_PREFIX} (quota exceeded)."
    assert mock_openrouter.call_count == len(COPILOT_BACKUP_MODELS)
    assert parse_copilot_verdict(result) is False


def test_parse_copilot_verdict_still_works_on_a_backup_attributed_response():
    text = (
        "[Copilot unavailable — backup reviewer Nvidia Nemotron-Ultra-550B responded]\n\n"
        "Some reasoning about the condition.\nFINAL_VERDICT: CONFIRMED"
    )
    assert parse_copilot_verdict(text) is True


@patch("ai.copilot_execution._run_copilot_verdict")
@patch("ai.copilot_execution._fetch_technical_context", return_value="fake technical context")
@patch("ai.copilot_execution.open_position")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions", return_value=[])
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_run_copilot_execution_check_executes_a_confirmed_pending_setup(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_verdict, _fixed_files,
):
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=1999.0, ask=2000.0)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_open.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=777)
    setup = PendingSetup(
        symbol="XAUUSD", side="buy", pct=1.0, trigger_condition="cond",
        price=2000.0, stop_loss=1980.0, take_profit=2050.0,
    )
    mock_verdict.return_value = (setup, True, "FINAL_VERDICT: CONFIRMED")

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            pending_setups=[
                {"symbol": "XAUUSD", "side": "buy", "pct": 1.0, "trigger_condition": "cond",
                 "price": 2000.0, "stop_loss": 1980.0, "take_profit": 2050.0, "reason": "r"}
            ],
        )
        run_copilot_execution_check()

    mock_open.assert_called_once()
    state = read_execution_state()
    assert state["last_verdicts"]["XAUUSD"]["confirmed"] is True


@patch("ai.copilot_execution._run_copilot_verdict")
@patch("ai.copilot_execution._fetch_technical_context", return_value="fake technical context")
@patch("ai.copilot_execution.open_position")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions", return_value=[])
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_run_copilot_execution_check_does_not_execute_an_unconfirmed_pending_setup(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_verdict, _fixed_files,
):
    from data.mt5_source import AccountSummary, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=1999.0, ask=2000.0)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    setup = PendingSetup(symbol="XAUUSD", side="buy", pct=1.0, trigger_condition="cond")
    mock_verdict.return_value = (setup, False, "FINAL_VERDICT: NOT_CONFIRMED")

    _write_suggestion(
        _fixed_files,
        pending_setups=[
            {"symbol": "XAUUSD", "side": "buy", "pct": 1.0, "trigger_condition": "cond", "reason": ""}
        ],
    )
    run_copilot_execution_check()

    mock_open.assert_not_called()
    state = read_execution_state()
    assert state["last_verdicts"]["XAUUSD"]["confirmed"] is False


# --- watched positions: Copilot's own invalidation-condition check (2026-08-23) ---


def _seed_settled(fixed_files, symbol, state, generated_utc, entry=None):
    Path(config.COPILOT_EXECUTION_SETTLEMENT_FILE).write_text(json.dumps({
        "generated_utc": generated_utc,
        "settled": {
            symbol: {
                "origin": "immediate", "state": state,
                "entry": entry or {"pct": 1.0, "price": 1.09, "stop_loss": 1.08, "take_profit": None, "side": "buy"},
                "order_ticket": 100,
            }
        },
    }))


@patch("ai.copilot_execution._run_copilot_invalidation_check")
@patch("ai.copilot_execution._fetch_technical_context", return_value="fake technical context")
@patch("ai.copilot_execution.close_position")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_confirmed_invalidation_forces_pct_zero_and_closes_the_position(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_close, mock_fetch_ctx, mock_invalidation, _fixed_files,
):
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_positions.return_value = [_position(symbol="EURUSD", side="buy", volume=1.0, ticket=200)]
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_close.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=200)
    mock_invalidation.return_value = ("EURUSD", True, "FINAL_VERDICT: CONFIRMED")

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {
        "pct": 1.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None,
        "side": "buy", "reason": "r", "invalidation_condition": "H1 closes below 1.0800",
    }
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "EURUSD", "filled", generated_utc, entry=entry)

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_copilot_execution_check()

    mock_invalidation.assert_called_once()
    mock_close.assert_called_once()
    state = read_execution_state()
    assert state["last_verdicts"]["EURUSD"]["confirmed"] is True


@patch("ai.copilot_execution._run_copilot_invalidation_check")
@patch("ai.copilot_execution._fetch_technical_context", return_value="fake technical context")
@patch("ai.copilot_execution.close_position")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_not_confirmed_invalidation_leaves_merged_allocation_unchanged(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_close, mock_fetch_ctx, mock_invalidation, _fixed_files,
):
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_positions.return_value = [_position(symbol="EURUSD", side="buy", volume=1.0, ticket=200)]
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_invalidation.return_value = ("EURUSD", False, "FINAL_VERDICT: NOT_CONFIRMED")

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {
        "pct": 1.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None,
        "side": "buy", "reason": "r", "invalidation_condition": "H1 closes below 1.0800",
    }
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "EURUSD", "filled", generated_utc, entry=entry)

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_copilot_execution_check()

    mock_invalidation.assert_called_once()
    mock_close.assert_not_called()
    state = read_execution_state()
    assert state["last_verdicts"]["EURUSD"]["confirmed"] is False


@patch("ai.copilot_execution._run_copilot_invalidation_check")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_watched_positions_excludes_symbols_without_invalidation_condition(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_invalidation, _fixed_files,
):
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_positions.return_value = [_position(symbol="EURUSD", side="buy", volume=1.0, ticket=200)]
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )

    generated_utc = datetime.now(timezone.utc).isoformat()
    # No invalidation_condition key at all -- must never be watched.
    entry = {"pct": 1.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None, "side": "buy", "reason": "r"}
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "EURUSD", "filled", generated_utc, entry=entry)

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec, patch("ai.copilot_execution.close_position"):
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_copilot_execution_check()

    mock_invalidation.assert_not_called()


@patch("ai.copilot_execution._run_copilot_invalidation_check")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions", return_value=[])
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_watched_positions_excludes_closed_after_fill_symbols(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_invalidation, _fixed_files,
):
    from data.mt5_source import AccountSummary, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {
        "pct": 1.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None,
        "side": "buy", "reason": "r", "invalidation_condition": "H1 closes below 1.0800",
    }
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "EURUSD", "closed_after_fill", generated_utc, entry=entry)

    run_copilot_execution_check()

    mock_invalidation.assert_not_called()


@patch("ai.copilot_execution._run_copilot_invalidation_check")
@patch("ai.copilot_execution._fetch_technical_context", return_value="fake technical context")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_watched_positions_capped_at_max_watched_positions_config_value(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_fetch_ctx, mock_invalidation, _fixed_files,
):
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_positions.return_value = [
        _position(symbol="EURUSD", side="buy", volume=1.0, ticket=200),
        _position(symbol="GBPUSD", side="buy", volume=1.0, ticket=201),
    ]
    mock_watch.return_value = [
        MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900),
        MarketAsset(symbol="GBPUSD", description="Pound", bid=1.2699, ask=1.2700),
    ]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_invalidation.return_value = ("EURUSD", False, "FINAL_VERDICT: NOT_CONFIRMED")

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry_eur = {
        "pct": 1.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None,
        "side": "buy", "reason": "r", "invalidation_condition": "H1 closes below 1.0800",
    }
    entry_gbp = {
        "pct": 1.0, "price": 1.2700, "stop_loss": 1.2650, "take_profit": None,
        "side": "buy", "reason": "r", "invalidation_condition": "H1 closes below 1.2600",
    }
    _write_suggestion(
        _fixed_files, immediate_allocation={"EURUSD": entry_eur, "GBPUSD": entry_gbp}, generated_utc=generated_utc,
    )
    Path(config.COPILOT_EXECUTION_SETTLEMENT_FILE).write_text(json.dumps({
        "generated_utc": generated_utc,
        "settled": {
            "EURUSD": {"origin": "immediate", "state": "filled", "entry": entry_eur, "order_ticket": 200},
            "GBPUSD": {"origin": "immediate", "state": "filled", "entry": entry_gbp, "order_ticket": 201},
        },
    }))

    with (
        patch.object(config, "COPILOT_EXECUTION_MAX_WATCHED_POSITIONS", 1),
        patch("ai.copilot_execution.get_contract_spec") as mock_spec,
        patch("ai.copilot_execution.close_position"),
    ):
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_copilot_execution_check()

    mock_invalidation.assert_called_once()


@patch("ai.copilot_execution._run_copilot_invalidation_check")
@patch("ai.copilot_execution._run_copilot_verdict")
@patch("ai.copilot_execution._fetch_technical_context", return_value="fake technical context")
@patch("ai.copilot_execution.open_position")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_pending_setup_verdicts_and_invalidation_verdicts_both_processed_in_one_poll(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_verdict, mock_invalidation, _fixed_files,
):
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_positions.return_value = [_position(symbol="EURUSD", side="buy", volume=1.0, ticket=200)]
    mock_watch.return_value = [
        MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900),
        MarketAsset(symbol="XAUUSD", description="Gold", bid=1999.0, ask=2000.0),
    ]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_open.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=777)
    setup = PendingSetup(symbol="XAUUSD", side="buy", pct=1.0, trigger_condition="cond")
    mock_verdict.return_value = (setup, True, "FINAL_VERDICT: CONFIRMED")
    mock_invalidation.return_value = ("EURUSD", False, "FINAL_VERDICT: NOT_CONFIRMED")

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {
        "pct": 1.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None,
        "side": "buy", "reason": "r", "invalidation_condition": "H1 closes below 1.0800",
    }
    _write_suggestion(
        _fixed_files,
        immediate_allocation={"EURUSD": entry},
        pending_setups=[
            {"symbol": "XAUUSD", "side": "buy", "pct": 1.0, "trigger_condition": "cond",
             "price": 2000.0, "stop_loss": 1980.0, "take_profit": 2050.0, "reason": "r"}
        ],
        generated_utc=generated_utc,
    )
    _seed_settled(_fixed_files, "EURUSD", "filled", generated_utc, entry=entry)

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_copilot_execution_check()

    mock_verdict.assert_called_once()
    mock_invalidation.assert_called_once()
    state = read_execution_state()
    assert state["last_verdicts"]["XAUUSD"]["confirmed"] is True
    assert state["last_verdicts"]["EURUSD"]["confirmed"] is False


# --- regression: cancelling a STRAY pending order must never wipe the ---
# settlement record for an unrelated, still-held "filled" position on the
# same symbol (real bug found on self-review 2026-08-24)


@patch("ai.copilot_execution.cancel_pending_order")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders")
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_cancelling_a_stray_pending_order_never_wipes_the_filled_settlement_record(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_cancel, _fixed_files,
):
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    # A real, already-held position that already exactly matches today's
    # target (same side/size/sl/tp) -- resolves to "hold", NOT
    # amend_position -- with a separate, stray pending order also
    # resting on the same symbol (e.g. a leftover from an earlier poll).
    mock_positions.return_value = [_position(symbol="EURUSD", side="buy", volume=1.0, ticket=200)]
    mock_pending.return_value = [_pending_order(symbol="EURUSD", ticket=999)]
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0799, ask=1.0800)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_cancel.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=999)

    generated_utc = datetime.now(timezone.utc).isoformat()
    # _position's own defaults are side="buy", price_open=1.09, sl=1.08 —
    # match the target exactly so this resolves to "hold", isolating the
    # test to the stray-cancel behavior specifically.
    entry = {"pct": 1.0, "price": 1.09, "stop_loss": 1.08, "take_profit": None, "side": "buy"}
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "EURUSD", "filled", generated_utc, entry=entry)

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_copilot_execution_check()

    mock_cancel.assert_called_once_with(999)
    settled = read_settlement()["settled"]
    assert "EURUSD" in settled
    assert settled["EURUSD"]["state"] == "filled"


# --- regression: an outstanding "increase" order must never force-close ---
# the already-held base position it was topping up (real bug found on audit)


@patch("ai.copilot_execution.close_position")
@patch("ai.copilot_execution.open_position")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders")
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_held_position_with_outstanding_increase_order_is_never_force_closed(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_close, _fixed_files,
):
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    # The symbol is genuinely held (1 lot already filled)...
    mock_positions.return_value = [_position(symbol="EURUSD", side="buy", volume=1.0, ticket=1)]
    # ...AND has a real outstanding pending order (the earlier top-up
    # "increase" that hasn't filled yet).
    mock_pending.return_value = [_pending_order(symbol="EURUSD", ticket=100)]
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )

    generated_utc = datetime.now(timezone.utc).isoformat()
    _write_suggestion(
        _fixed_files,
        immediate_allocation={"EURUSD": {"pct": 2.0, "side": "buy", "price": 1.0900, "stop_loss": 1.0850}},
        generated_utc=generated_utc,
    )
    # Pre-seed settlement: the SAME mega-session cycle already placed an
    # "increase" order for EURUSD (ticket 100), still unfilled.
    Path(config.COPILOT_EXECUTION_SETTLEMENT_FILE).write_text(json.dumps({
        "generated_utc": generated_utc,
        "settled": {
            "EURUSD": {
                "origin": "immediate", "state": "order_placed",
                "entry": {"pct": 2.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None, "side": "buy"},
                "order_ticket": 100,
            }
        },
    }))

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_copilot_execution_check()

    # The real, already-filled position must never be force-closed just
    # because an unrelated top-up order is still pending.
    mock_close.assert_not_called()
    # And no DUPLICATE order gets stacked on top of the already-
    # outstanding one for the same symbol.
    mock_open.assert_not_called()
