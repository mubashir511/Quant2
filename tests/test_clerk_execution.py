import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import config
import data.symbol_news as symbol_news
from ai.ollama_client import FAILED_MESSAGE as OLLAMA_FAILED_MESSAGE
from ai.clerk_execution import (
    SymbolSettlement,
    TacticalVerdict,
    _apply_atr_stop_floor_guard,
    _apply_correlation_guard,
    _append_stale_pending_setup_cancels,
    _apply_event_blackout_guard,
    _apply_intraday_size_scalar,
    _apply_stale_entry_reanchor,
    _stabilize_resting_orders,
    _tactical_event_note,
    _apply_reward_risk_floor_guard,
    _build_carried_forward_allocation,
    _build_verdict_prompt,
    _compute_realized_r,
    _detect_external_stop_drift,
    _detect_newly_closed_symbols,
    _detect_newly_filled_symbols,
    _detect_ollama_outage,
    _determine_close_cause,
    _export_closed_trade_notes,
    _fetch_clerk_news_block,
    _fetch_correlation_closes,
    _fetch_technical_context,
    _format_closed_trade_note,
    _iter_stop_drift,
    _mega_session_is_live,
    _reconcile_settlement,
    _reset_settlement_for_new_session,
    _restore_external_stop_drift,
    _run_clerk_prompt,
    _write_execution_state,
    execution_check_is_live,
    is_execution_due,
    is_pre_weekend_cleanup_due,
    next_execution_check_utc,
    parse_clerk_verdict,
    read_clerk_execution_enabled,
    read_clerk_execution_interval_minutes,
    read_execution_progress,
    read_execution_state,
    read_settlement,
    run_clerk_execution_check,
    set_clerk_execution_enabled,
    set_clerk_execution_interval_minutes,
)
from ai.ftmo_suggest import FtmoAssetAnalysis, IntradayBacktests
from analysis.backtest import RSIReactionBacktest, SupportResistanceBacktest
from ai.portfolio_suggest import AllocationEntry, AssetAnalysis, PendingSetup
from analysis.chart_structure import ChartStructureSnapshot
from analysis.technical import TechnicalStats
from data.mt5_execution import OrderResult
from data.mt5_source import ClosedTrade, MarketAsset, PendingOrder, Position, TradeCost


@pytest.fixture(autouse=True)
def _no_real_economic_calendar():
    """Clerk now reads the economic calendar each poll — never hit the network in tests."""
    with (
        patch("ai.clerk_execution.economic_calendar.fetch_calendar_events", return_value=[]),
        patch("ai.clerk_execution.get_server_time_offset", return_value=None),
    ):
        yield


def _fake_technical_stats(atr=None, **overrides) -> TechnicalStats:
    defaults = dict(
        last_price=None, sma20=None, pct_vs_sma20=None, trend=None,
        change_1m_pct=None, change_3m_pct=None, change_6m_pct=None,
        volatility_annualized_pct=None, support=None, resistance=None,
        range_width_pct=None, market_regime=None, atr=atr,
        atr_pct=None, rsi=None, volume_trend_pct=None,
        momentum_acceleration=None,
    )
    return TechnicalStats(**{**defaults, **overrides})


def _empty_chart_structure() -> ChartStructureSnapshot:
    return ChartStructureSnapshot(fibonacci=None, sr_levels=None, trendlines=None, patterns=[])


def _fake_ftmo_analysis(symbol="EURUSD", h1_atr=None, **overrides) -> FtmoAssetAnalysis:
    # _fetch_technical_context now returns (FtmoAssetAnalysis, str) instead
    # of a bare str (2026-08-30, so the tactical pre-screen can read
    # h1_stats.atr without a second MT5 fetch) — this is the fake half of
    # that tuple. h1_atr defaults to None (matching real life whenever
    # there's insufficient H1 history) rather than a made-up number.
    defaults = dict(
        base=AssetAnalysis(symbol=symbol, description=symbol, bid=1.0, ask=1.0, display_name=None),
        h4_stats=_fake_technical_stats(),
        h1_stats=_fake_technical_stats(atr=h1_atr),
        h4_structure=_empty_chart_structure(),
        h1_structure=_empty_chart_structure(),
        trade_cost=None,
    )
    return FtmoAssetAnalysis(**{**defaults, **overrides})


def test_fetch_technical_context_excludes_favorable_excursion():
    # Real gap caught on a self-recheck (2026-09-12): this function's
    # formatted output becomes `technical_context` for the tactical-
    # verdict prompts, which never include ai/ftmo_suggest.py's own
    # _INSTRUCTION_HEAD — the ONLY place the favorable-excursion figure's
    # critical misread warning and HOLDING HORIZON scale-mismatch caveat
    # actually live. Without explicitly excluding it here, the local
    # tactical model would see a bare, uncaveated positive-looking
    # magnitude figure with a dangling "see the TP-sizing instruction
    # above" reference that doesn't exist in its own prompt at all — this
    # locks in that format_ftmo_asset_context is called with
    # include_favorable_excursion=False from this exact call site.
    asset = MarketAsset(symbol="EURUSD", description="Euro vs US Dollar", bid=1.1, ask=1.1005)
    with patch("ai.clerk_execution.analyze_ftmo_asset_live", return_value=_fake_ftmo_analysis()):
        with patch(
            "ai.clerk_execution.format_ftmo_asset_context", return_value="  historical overbought RSI reaction: ..."
        ) as mock_format:
            result = _fetch_technical_context("EURUSD", {"EURUSD": asset}, 10_000.0)
    assert result is not None
    mock_format.assert_called_once()
    assert mock_format.call_args.kwargs.get("include_favorable_excursion") is False
    # Same real-gap pattern, caught on this recheck (2026-09-13): the
    # "market CLOSED" tag exists to support _INSTRUCTION_HEAD's "don't
    # propose a NEW trade on a closed market" guidance, which this prompt
    # also never includes -- and Clerk never proposes new trades at all,
    # so the tag would be pure noise here.
    assert mock_format.call_args.kwargs.get("include_market_status") is False


@pytest.fixture(autouse=True)
def _no_real_tactical_calls():
    # Real hang found live 2026-08-30, right after Copilot CLI/OpenRouter
    # were replaced with a local Ollama model: the tactical-defense check
    # fires every poll on every FILLED position REGARDLESS of whether
    # tactical_defense is enabled (shadow mode still runs the LLM call,
    # see run_clerk_execution_check's own docstring) — so any test here
    # that sets up a filled position without mocking
    # _run_clerk_tactical_check now reaches a REAL local Ollama call
    # instead of the old Copilot CLI, which used to fail in under a
    # second on a machine with no `copilot` on PATH (harmless by
    # accident). A real model genuinely running takes 30-90+ seconds per
    # call, so an unmocked test here silently went from "instant" to
    # "hangs for real minutes" the moment this machine got a working
    # local model — this file isn't about tactical-defense behavior at
    # all (see tests/test_clerk_tactical.py for that), so a safe,
    # inert default HOLD here for every test is correct, not a
    # workaround. Autouse + module-scoped-by-file (NOT in a shared
    # conftest), so it never affects test_clerk_tactical.py's own
    # per-test tactical mocks.
    with patch(
        "ai.clerk_execution._run_clerk_tactical_check",
        return_value=("_", TacticalVerdict(tier="hold"), "FINAL_VERDICT: HOLD"),
    ):
        yield


@pytest.fixture(autouse=True)
def _fixed_files(tmp_path):
    with (
        patch.object(config, "CLERK_EXECUTION_STATE_FILE", str(tmp_path / "clerk_execution_state.json")),
        patch.object(config, "CLERK_EXECUTION_SETTLEMENT_FILE", str(tmp_path / "clerk_execution_settlement.json")),
        patch.object(config, "CLERK_EXECUTION_PROGRESS_FILE", str(tmp_path / "clerk_execution_progress.json")),
        patch.object(config, "MEGA_ANALYSIS_LATEST_SUGGESTION_FILE", str(tmp_path / "mega_analysis_latest_suggestion.json")),
        patch.object(config, "MEGA_ANALYSIS_STATE_FILE", str(tmp_path / "mega_analysis_state.json")),
        patch.object(config, "MEGA_ANALYSIS_PROGRESS_FILE", str(tmp_path / "mega_analysis_progress.json")),
        patch.object(config, "CLERK_EXECUTION_CHECK_INTERVAL_MINUTES", 15),
        patch.object(config, "CLERK_EXECUTION_MAX_PENDING_SETUPS", 10),
        patch.object(config, "CLERK_EXECUTION_MAX_WATCHED_POSITIONS", 10),
        patch.object(config, "AMEND_TOLERANCE_PCT", 0.05),
        # Isolated proactively (2026-08-23) — the mega-analysis enabled
        # toggle already caused a real collision the same day where the
        # live app's own genuine use of a brand-new toggle broke tests
        # relying on the real project-root default; done here up front
        # for the Clerk equivalent instead of waiting for a repeat.
        patch.object(config, "CLERK_EXECUTION_ENABLED_FILE", str(tmp_path / "clerk_execution_enabled.json")),
        patch.object(config, "CLERK_EXECUTION_INTERVAL_FILE", str(tmp_path / "clerk_execution_interval.json")),
        # Isolated 2026-09-19 — real leak caught live: ai.trade_journal's
        # new record_* calls (wired into run_clerk_execution_check as a
        # pure addition) and the pre-existing vault trade-note export
        # both write into these real, unmocked paths by default, and
        # this file's own broader integration-style tests exercise both
        # without otherwise touching either — confirmed by running the
        # full suite once without this line and finding real test-
        # fixture symbols (BTCUSD, CASH, EURUSD, XAUUSD) written into
        # this project's own real records/ftmo_trade_journal/ and
        # obsidian_vault/Trades/ directories afterward.
        patch.object(config, "TRADE_JOURNAL_DIR", str(tmp_path / "ftmo_trade_journal")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
    ):
        yield tmp_path


def _position(symbol="EURUSD", side="buy", volume=1.0, ticket=1):
    return Position(
        symbol=symbol, volume=volume, side=side, price_open=1.09, price_current=1.10,
        sl=1.08, profit=10.0, opened_at=datetime.now(), ticket=ticket,
    )


def _pending_order(symbol="EURUSD", ticket=100):
    return PendingOrder(symbol=symbol, volume=1.0, order_type="buy limit", price_open=1.09, sl=1.08, tp=1.11, ticket=ticket)


# --- parse_clerk_verdict: the full fail-safe branch table ---


def test_parse_clerk_verdict_true_on_clean_confirmed():
    assert parse_clerk_verdict("Some reasoning.\n\nFINAL_VERDICT: CONFIRMED") is True


def test_parse_clerk_verdict_false_on_clean_not_confirmed():
    assert parse_clerk_verdict("Some reasoning.\n\nFINAL_VERDICT: NOT_CONFIRMED") is False


def test_parse_clerk_verdict_false_on_empty_string():
    assert parse_clerk_verdict("") is False


def test_parse_clerk_verdict_false_on_whitespace_only():
    assert parse_clerk_verdict("   \n  ") is False


def test_parse_clerk_verdict_false_when_ollama_unavailable():
    assert parse_clerk_verdict(OLLAMA_FAILED_MESSAGE) is False


def test_parse_clerk_verdict_false_when_no_token_present():
    assert parse_clerk_verdict("I think this looks like a good trade.") is False


def test_parse_clerk_verdict_false_on_unrecognized_value():
    assert parse_clerk_verdict("FINAL_VERDICT: MAYBE") is False


def test_parse_clerk_verdict_last_occurrence_wins_when_model_restates():
    text = "Initially I thought FINAL_VERDICT: CONFIRMED but on reflection FINAL_VERDICT: NOT_CONFIRMED"
    assert parse_clerk_verdict(text) is False


def test_parse_clerk_verdict_last_occurrence_wins_the_other_direction():
    text = "FINAL_VERDICT: NOT_CONFIRMED -- wait, actually FINAL_VERDICT: CONFIRMED"
    assert parse_clerk_verdict(text) is True


def test_parse_clerk_verdict_case_insensitive():
    assert parse_clerk_verdict("final_verdict: confirmed") is True


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


# --- vault trade journal: newly-closed detection, R math, note formatting ---


def test_detect_newly_closed_symbols_counts_filled_to_closed_transition():
    old = {"EURUSD": {"state": "filled"}}
    new = {"EURUSD": {"state": "closed_after_fill"}}
    assert _detect_newly_closed_symbols(old, new) == ["EURUSD"]


def test_detect_newly_closed_symbols_ignores_filled_stays_filled():
    old = {"EURUSD": {"state": "filled"}}
    new = {"EURUSD": {"state": "filled"}}
    assert _detect_newly_closed_symbols(old, new) == []


def test_detect_newly_closed_symbols_ignores_order_placed_to_filled():
    old = {"EURUSD": {"state": "order_placed"}}
    new = {"EURUSD": {"state": "filled"}}
    assert _detect_newly_closed_symbols(old, new) == []


def test_detect_newly_closed_symbols_ignores_already_closed_in_both():
    # No duplicate journaling of a close already reported on a prior poll.
    old = {"EURUSD": {"state": "closed_after_fill"}}
    new = {"EURUSD": {"state": "closed_after_fill"}}
    assert _detect_newly_closed_symbols(old, new) == []


def test_detect_newly_closed_symbols_ignores_symbol_with_no_old_record():
    old = {}
    new = {"EURUSD": {"state": "closed_after_fill"}}
    assert _detect_newly_closed_symbols(old, new) == []


def test_compute_realized_r_buy_win():
    entry = {"side": "buy", "stop_loss": 1.08}
    assert _compute_realized_r(entry, open_price=1.10, close_price=1.13) == pytest.approx(1.5)


def test_compute_realized_r_buy_loss():
    entry = {"side": "buy", "stop_loss": 1.08}
    assert _compute_realized_r(entry, open_price=1.10, close_price=1.09) == pytest.approx(-0.5)


def test_compute_realized_r_sell_win():
    entry = {"side": "sell", "stop_loss": 1.12}
    assert _compute_realized_r(entry, open_price=1.10, close_price=1.07) == pytest.approx(1.5)


def test_compute_realized_r_sell_loss():
    entry = {"side": "sell", "stop_loss": 1.12}
    assert _compute_realized_r(entry, open_price=1.10, close_price=1.11) == pytest.approx(-0.5)


def test_compute_realized_r_none_when_stop_loss_missing():
    entry = {"side": "buy", "stop_loss": None}
    assert _compute_realized_r(entry, open_price=1.10, close_price=1.13) is None


def test_compute_realized_r_none_when_risk_non_positive():
    # Stop on the wrong side of entry for a buy -- bad/stale data, never fabricated.
    entry = {"side": "buy", "stop_loss": 1.12}
    assert _compute_realized_r(entry, open_price=1.10, close_price=1.13) is None


def _closed_trade(**overrides) -> ClosedTrade:
    defaults = dict(
        position_id=1, symbol="EURUSD", side="buy", volume=1.0,
        opened_at=datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc),
        closed_at=datetime(2026, 9, 10, 14, 0, tzinfo=timezone.utc),
        open_price=1.10, close_price=1.13, profit=300.0, gross_profit=305.0,
    )
    return ClosedTrade(**{**defaults, **overrides})


def _story_with_events(*event_specs):
    from ai.trade_journal import TradeStory, TradeStoryEvent

    events = [TradeStoryEvent(type=t, timestamp_utc="2026-09-10T00:00:00+00:00", data=d) for t, d in event_specs]
    return TradeStory(symbol="EURUSD", story_id="EURUSD_2026-09-10_080000", status="open", events=events)


def test_detect_newly_filled_symbols_catches_order_placed_to_filled():
    old_settled = {"EURUSD": {"state": "order_placed"}}
    new_settled = {"EURUSD": {"state": "filled", "order_ticket": 42, "entry": {"price": 1.10}}}
    result = _detect_newly_filled_symbols(old_settled, new_settled)
    assert result == [("EURUSD", new_settled["EURUSD"])]


def test_detect_newly_filled_symbols_ignores_other_transitions():
    old_settled = {"EURUSD": {"state": "filled"}}
    new_settled = {"EURUSD": {"state": "closed_after_fill"}}
    assert _detect_newly_filled_symbols(old_settled, new_settled) == []


def test_determine_close_cause_prefers_a_real_applied_clerk_exit():
    story = _story_with_events(("tactical_action", {"tier": "exit", "applied": True}))
    entry = {"stop_loss": 1.05, "take_profit": 1.20}
    trade = _closed_trade(close_price=1.13)  # doesn't match SL or TP -- exit event should still win
    assert _determine_close_cause(entry, trade, story) == "clerk_tactical_exit"


def test_determine_close_cause_ignores_an_unapplied_exit_verdict():
    story = _story_with_events(("tactical_action", {"tier": "exit", "applied": False}))
    entry = {"stop_loss": 1.13, "take_profit": 1.20}
    trade = _closed_trade(close_price=1.13)
    assert _determine_close_cause(entry, trade, story) == "stop_loss_hit"


def test_determine_close_cause_stop_loss_hit_from_price():
    entry = {"stop_loss": 1.10, "take_profit": 1.20}
    trade = _closed_trade(close_price=1.1005)  # within 0.1% tolerance of the stop
    assert _determine_close_cause(entry, trade, None) == "stop_loss_hit"


def test_determine_close_cause_take_profit_hit_from_price():
    entry = {"stop_loss": 1.05, "take_profit": 1.20}
    trade = _closed_trade(close_price=1.2)
    assert _determine_close_cause(entry, trade, None) == "take_profit_hit"


def test_determine_close_cause_manual_when_price_matches_neither():
    entry = {"stop_loss": 1.00, "take_profit": 1.30}
    trade = _closed_trade(close_price=1.13)
    assert _determine_close_cause(entry, trade, None) == "manual_or_unknown"


def test_determine_close_cause_manual_when_no_closed_trade_data_at_all():
    assert _determine_close_cause({"stop_loss": 1.0}, None, None) == "manual_or_unknown"


def test_format_closed_trade_note_includes_real_numbers_and_thesis():
    entry = {"side": "buy", "stop_loss": 1.08, "reason": "oversold bounce off H4 support"}
    text = _format_closed_trade_note("EURUSD", entry, _closed_trade())
    assert "EURUSD" in text
    assert "buy" in text
    assert "+300.00" in text
    assert "+1.50R" in text
    assert "oversold bounce off H4 support" in text
    assert "[[EURUSD]]" in text
    assert "[[Trade Journal]]" in text


def test_format_closed_trade_note_includes_invalidation_condition_when_present():
    entry = {"side": "buy", "stop_loss": 1.08, "reason": "r", "invalidation_condition": "H1 closes below 1.09"}
    text = _format_closed_trade_note("EURUSD", entry, _closed_trade())
    assert "H1 closes below 1.09" in text


def test_format_closed_trade_note_degrades_honestly_without_a_matched_deal():
    entry = {"side": "buy", "stop_loss": 1.08, "price": 1.10, "take_profit": 1.20, "reason": "r"}
    text = _format_closed_trade_note("EURUSD", entry, None)
    assert "could not be matched" in text
    assert "Intended entry: 1.1" in text
    # Never fabricates a real P&L/R figure it doesn't have.
    assert "Realized P&L" not in text


def test_export_closed_trade_notes_writes_a_real_file(tmp_path):
    with (
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path)),
        patch("ai.clerk_execution.get_history_deals") as mock_deals,
        patch("ai.clerk_execution.group_closed_trades") as mock_group,
    ):
        mock_deals.return_value = []
        mock_group.return_value = [_closed_trade()]
        old_settled = {"EURUSD": {"entry": {"side": "buy", "stop_loss": 1.08, "reason": "r"}}}
        _export_closed_trade_notes(["EURUSD"], old_settled)

    files = list((tmp_path / "Trades").glob("EURUSD*.md"))
    assert len(files) == 1
    assert "EURUSD" in files[0].read_text(encoding="utf-8")


def test_closure_deal_lookups_never_pass_an_explicit_date_to(tmp_path):
    # Real bug 2026-09-22: passing date_to=now_utc excluded the just-closed
    # deal (broker server clock runs ahead of UTC), so MSFT/WHEAT closures
    # were journaled with no P&L. The default date_to reaches a day into the
    # future on purpose -- both callers must use it.
    from ai.clerk_execution import _record_trade_journal_closures

    with (
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path)),
        patch("ai.clerk_execution.get_history_deals", return_value=[]) as mock_deals,
        patch("ai.clerk_execution.group_closed_trades", return_value=[]),
    ):
        _export_closed_trade_notes(["EURUSD"], {"EURUSD": {"entry": {}}})
        _record_trade_journal_closures(["EURUSD"], {"EURUSD": {"entry": {}}})
    assert mock_deals.call_count == 2
    for call in mock_deals.call_args_list:
        assert len(call.args) == 1 and "date_to" not in call.kwargs


def test_reconcile_journal_unknown_closures_survives_null_tactical_and_entry(tmp_path):
    # Real bug 2026-09-24: WHEAT's settlement record had "tactical": None,
    # which crashed the cause resolver and left its P&L unreconciled.
    from datetime import datetime, timedelta, timezone

    from ai import trade_journal as tj
    from ai.clerk_execution import _reconcile_journal_unknown_closures

    with (
        patch.object(config, "TRADE_JOURNAL_DIR", str(tmp_path / "j")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "v")),
    ):
        tj.record_proposals({"immediate_allocation": {"EURUSD": {"pct": 1.0, "side": "buy", "price": 1.1}}, "pending_setups": []})
        tj.record_filled("EURUSD", 42, 1.1)
        tj.record_closed("EURUSD", None, "manual_or_unknown")
        story = next(s for s in tj.list_all_stories() if s.symbol == "EURUSD")
        closed_at = datetime.fromisoformat(story.events[-1].timestamp_utc)
        trade = _closed_trade(position_id=42, closed_at=closed_at + timedelta(hours=1), profit=-10.0, close_price=1.09)
        with (
            patch("ai.clerk_execution.get_history_deals", return_value=[]),
            patch("ai.clerk_execution.group_closed_trades", return_value=[trade]),
        ):
            _reconcile_journal_unknown_closures({"EURUSD": {"entry": None, "tactical": None}})
        assert next(s for s in tj.list_all_stories() if s.symbol == "EURUSD").status == "closed_lost"


def test_export_closed_trade_notes_no_op_when_nothing_closed(tmp_path):
    with (
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path)),
        patch("ai.clerk_execution.get_history_deals") as mock_deals,
    ):
        _export_closed_trade_notes([], {})
        mock_deals.assert_not_called()
    assert not (tmp_path / "Trades").exists()


def test_export_closed_trade_notes_degrades_when_mt5_deal_fetch_fails(tmp_path):
    with (
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path)),
        patch("ai.clerk_execution.get_history_deals", side_effect=RuntimeError("MT5 down")),
    ):
        _export_closed_trade_notes(["EURUSD"], {"EURUSD": {"entry": {}}})
    # Degrades to intended-terms-only rather than raising -- a real note still gets written.
    files = list((tmp_path / "Trades").glob("EURUSD*.md"))
    assert len(files) == 1


def test_export_closed_trade_notes_never_raises_when_file_write_itself_fails(tmp_path):
    with (
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path)),
        patch("ai.clerk_execution.get_history_deals", return_value=[]),
        patch("ai.clerk_execution.group_closed_trades", return_value=[]),
        patch("pathlib.Path.write_text", side_effect=OSError("disk full")),
    ):
        _export_closed_trade_notes(["EURUSD"], {"EURUSD": {"entry": {"side": "buy"}}})
    # No exception propagated -- swallowed exactly like _export_chart_overlay's own pattern.


@patch("ai.clerk_execution.group_closed_trades")
@patch("ai.clerk_execution.get_history_deals")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch", return_value=[])
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_journals_a_real_close_end_to_end(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_deals, mock_group, _fixed_files, tmp_path,
):
    # End-to-end wiring check: a symbol tracked as "filled" last poll that
    # is no longer held this poll must produce a real vault note via the
    # actual run_clerk_execution_check hook, not just via the unit-tested
    # helpers called directly.
    from data.mt5_source import AccountSummary
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_group.return_value = [_closed_trade()]

    generated_utc = datetime.now(timezone.utc).isoformat()
    _write_suggestion(_fixed_files, generated_utc=generated_utc)
    _seed_settled(
        _fixed_files, "EURUSD", "filled", generated_utc,
        entry={"pct": 1.0, "price": 1.10, "stop_loss": 1.08, "take_profit": None, "side": "buy", "reason": "r"},
    )

    with patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path)):
        run_clerk_execution_check()

    files = list((tmp_path / "Trades").glob("EURUSD*.md"))
    assert len(files) == 1
    assert "+1.50R" in files[0].read_text(encoding="utf-8")
    # The settlement record itself must have advanced to closed_after_fill.
    assert read_settlement()["settled"]["EURUSD"]["state"] == "closed_after_fill"


@patch("ai.clerk_execution.get_history_deals")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch", return_value=[])
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_no_vault_activity_when_nothing_closed(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_deals, _fixed_files, tmp_path,
):
    from data.mt5_source import AccountSummary
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    _write_suggestion(_fixed_files)

    with patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path)):
        run_clerk_execution_check()

    mock_deals.assert_not_called()
    assert not (tmp_path / "Trades").exists()


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


def test_carried_forward_allocation_honors_a_persisted_tactical_reduction():
    # Real bug found live 2026-09-03 (USDCHF oscillated 0.13/0.09/0.13/
    # 0.06/0.13 lots across three tickets in about an hour, on a chart
    # with no real trend): a tactical DEFEND's reduced size only ever
    # lasted one poll before this fix, since the baseline was always
    # rebuilt from the mega session's own original, unreduced pct.
    settled = {
        "USDCHF": {
            "origin": "immediate",
            "state": "filled",
            "entry": {"pct": 0.4, "side": "buy", "price": 0.8075, "stop_loss": 0.8045},
            "order_ticket": 100,
            "tactical": {
                "tier_reached": "defend",
                "persisted_pct": 0.2,
                "persisted_stop_loss": 0.8065,
                "persisted_take_profit": 0.8138,
            },
        }
    }
    immediate_allocation_raw = {"USDCHF": {"pct": 0.4, "side": "buy", "price": 0.8075, "stop_loss": 0.8045, "take_profit": 0.8138}}
    carried = _build_carried_forward_allocation(immediate_allocation_raw, settled)
    assert carried["USDCHF"].pct == 0.2
    assert carried["USDCHF"].stop_loss == 0.8065
    assert carried["USDCHF"].take_profit == 0.8138


def test_carried_forward_allocation_ignores_stale_persisted_pct_not_smaller_than_target():
    # A persisted_pct left over that's no longer below the (possibly
    # since-updated) mega-session target shouldn't ever INCREASE the
    # baseline beyond what the current session actually asked for.
    settled = {
        "USDCHF": {
            "origin": "immediate",
            "state": "filled",
            "entry": {"pct": 0.4, "side": "buy"},
            "order_ticket": 100,
            "tactical": {"tier_reached": "defend", "persisted_pct": 0.5},
        }
    }
    immediate_allocation_raw = {"USDCHF": {"pct": 0.4, "side": "buy"}}
    carried = _build_carried_forward_allocation(immediate_allocation_raw, settled)
    assert carried["USDCHF"].pct == 0.4


def test_carried_forward_allocation_unaffected_by_absent_tactical_state():
    settled = {"USDCHF": {"origin": "immediate", "state": "filled", "entry": {"pct": 0.4, "side": "buy"}, "order_ticket": 100}}
    immediate_allocation_raw = {"USDCHF": {"pct": 0.4, "side": "buy"}}
    carried = _build_carried_forward_allocation(immediate_allocation_raw, settled)
    assert carried["USDCHF"].pct == 0.4


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


def test_held_symbol_stuck_in_order_placed_is_still_carried_forward():
    # Real incident, 2026-09-08/09: a Pending-Setup-originated symbol can
    # have a real, already-filled position AND a separate, still-resting
    # "top-up" pending order on the SAME symbol at once. settlement
    # tracks exactly one record per symbol, and the pending order's own
    # still-resting state kept the one record stuck on "order_placed"
    # even though a real position genuinely existed. Without `held_symbols`,
    # this is indistinguishable from test_pending_setup_order_placed_
    # symbol_stays_excluded_from_the_merge above (a fired-but-never-filled
    # setup with zero real volume) — the two cases must resolve
    # differently, and only `held_symbols` (real, live MT5 truth) can
    # tell them apart.
    settled = {
        "USDCAD": {
            "origin": "pending_setup", "state": "order_placed",
            "entry": {"pct": 0.13, "side": "sell", "price": 1.3815, "stop_loss": 1.385, "take_profit": 1.3751},
            "order_ticket": 999,
        }
    }
    carried = _build_carried_forward_allocation({}, settled, held_symbols=frozenset({"USDCAD"}))
    assert carried["USDCAD"] == AllocationEntry(
        pct=0.13, price=1.3815, stop_loss=1.385, take_profit=1.3751, side="sell"
    )


def test_order_placed_symbol_without_held_symbols_membership_stays_excluded():
    # The exact same record as above, but the symbol is genuinely NOT
    # held (held_symbols omitted) — confirms `held_symbols` only RESCUES
    # a stuck record when a real position backs it, never blanket-includes
    # every order_placed symbol regardless of live truth.
    settled = {
        "USDCAD": {
            "origin": "pending_setup", "state": "order_placed",
            "entry": {"pct": 0.13, "side": "sell", "price": 1.3815, "stop_loss": 1.385},
            "order_ticket": 999,
        }
    }
    carried = _build_carried_forward_allocation({}, settled)
    assert "USDCAD" not in carried


# --- real INTC incident, 2026-09-10: a stale immediate_allocation cancel ---
# --- directive must never shadow a separately-triggered, now-filled     ---
# --- Pending-Setup position on the SAME symbol.                        ---


def test_stale_immediate_allocation_cancel_never_shadows_a_filled_pending_setup():
    # Real sequence: the mega session's own CURRENT suggestion still says
    # "INTC: pct=0" (cancel a now-superseded, long-unfilled pending buy-
    # limit) — but hours later, in the SAME cycle, a SEPARATE Pending
    # Setup for INTC triggered, filled, and became a real position. The
    # settlement record's own origin=="pending_setup" is the proof this
    # real position did NOT come from the immediate_allocation entry —
    # that entry's pct=0 is simply stale and must not force-close it.
    immediate_allocation_raw = {"INTC": {"pct": 0, "side": "buy", "price": 100.0, "stop_loss": 95.5}}
    settled = {
        "INTC": {
            "origin": "pending_setup", "state": "filled",
            "entry": {"pct": 0.18, "side": "buy", "price": 101.90, "stop_loss": 98.50, "take_profit": 110.0},
            "order_ticket": 555,
        }
    }
    carried = _build_carried_forward_allocation(immediate_allocation_raw, settled)
    assert carried["INTC"] == AllocationEntry(
        pct=0.18, price=101.90, stop_loss=98.50, take_profit=110.0, side="buy"
    )


def test_stale_immediate_allocation_cancel_never_shadows_a_held_pending_setup_stuck_in_order_placed():
    # Same real gap, caught the instant it fills (before _reconcile_
    # settlement has transitioned state to "filled" yet) via held_symbols
    # — mirrors test_held_symbol_stuck_in_order_placed_is_still_carried_
    # forward's own reasoning, just with a stale SHADOWING immediate_
    # allocation entry present too this time.
    immediate_allocation_raw = {"INTC": {"pct": 0, "side": "buy", "price": 100.0, "stop_loss": 95.5}}
    settled = {
        "INTC": {
            "origin": "pending_setup", "state": "order_placed",
            "entry": {"pct": 0.18, "side": "buy", "price": 101.90, "stop_loss": 98.50, "take_profit": 110.0},
            "order_ticket": 555,
        }
    }
    carried = _build_carried_forward_allocation(immediate_allocation_raw, settled, held_symbols=frozenset({"INTC"}))
    assert carried["INTC"] == AllocationEntry(
        pct=0.18, price=101.90, stop_loss=98.50, take_profit=110.0, side="buy"
    )


def test_immediate_origin_symbol_still_wins_normally_even_when_filled():
    # The FIX must not overreach: a symbol whose settlement record's own
    # origin IS "immediate" (a real, ordinary top-up/hold/close case) must
    # keep using immediate_allocation_raw exactly as before — only
    # origin=="pending_setup" triggers the new shadow-avoidance path.
    immediate_allocation_raw = {"XAUUSD": {"pct": 0.4, "side": "buy", "price": 2000.0, "stop_loss": 1950.0}}
    settled = {
        "XAUUSD": {
            "origin": "immediate", "state": "filled",
            "entry": {"pct": 0.3, "side": "buy", "price": 1990.0, "stop_loss": 1940.0},
            "order_ticket": 777,
        }
    }
    carried = _build_carried_forward_allocation(immediate_allocation_raw, settled)
    assert carried["XAUUSD"].pct == 0.4  # from immediate_allocation_raw, unchanged behavior


def test_pending_setup_origin_not_yet_filled_and_not_held_still_uses_immediate_allocation():
    # A pending_setup-origin record that's neither "filled" nor in
    # held_symbols (still genuinely just a resting, unfilled order) must
    # NOT trigger the shadow-avoidance path -- immediate_allocation_raw's
    # own pct is still the right answer for a symbol with zero real
    # position backing it.
    immediate_allocation_raw = {"INTC": {"pct": 0, "side": "buy", "price": 100.0, "stop_loss": 95.5}}
    settled = {
        "INTC": {
            "origin": "pending_setup", "state": "order_placed",
            "entry": {"pct": 0.18, "side": "buy", "price": 101.90, "stop_loss": 98.50, "take_profit": 110.0},
            "order_ticket": 555,
        }
    }
    carried = _build_carried_forward_allocation(immediate_allocation_raw, settled)
    assert carried["INTC"].pct == 0  # from immediate_allocation_raw -- not yet actually held


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


# --- external stop-loss drift detection (2026-09-08 XAUUSD incident) ---


def test_stop_drift_detected_when_live_stop_diverges_from_recorded():
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4390.03
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.00}}}
    # No fresh carried_forward target for this symbol this cycle — the
    # real 2026-09-08 XAUUSD trade's own mega session never changed its
    # stop proposal, so this matches that real scenario exactly.
    warnings = _detect_external_stop_drift(positions_by_symbol, settled, {}, tolerance_pct=0.05)
    assert len(warnings) == 1
    assert "XAUUSD" in warnings[0]
    assert "4390.03" in warnings[0]
    assert "4378.0" in warnings[0]


def test_no_drift_warning_when_live_stop_matches_recorded():
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4378.00
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.001}}}  # trivial rounding noise, within tolerance
    assert _detect_external_stop_drift(positions_by_symbol, settled, {}, tolerance_pct=0.05) == []


def test_no_drift_warning_for_symbol_with_no_settlement_record():
    # A genuinely manual/untracked position this app has no history for —
    # nothing to compare against, so no warning (not a false positive).
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4390.03
    assert _detect_external_stop_drift(positions_by_symbol, {}, {}) == []


def test_no_drift_warning_when_recorded_stop_loss_is_missing():
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4390.03
    settled = {"XAUUSD": {"entry": {}}}
    assert _detect_external_stop_drift(positions_by_symbol, settled, {}) == []


def test_no_drift_warning_when_live_position_has_no_stop():
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = None
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.00}}}
    assert _detect_external_stop_drift(positions_by_symbol, settled, {}) == []


def test_no_drift_warning_when_a_fresh_mega_session_already_changed_the_target():
    # Real bug found on self-review, fixed 2026-09-09: settlement's own
    # entry.stop_loss is NEVER refreshed once a record exists (see
    # _backfill_settlement_for_held_positions' own docstring) — so a
    # perfectly legitimate, app-driven amend_position (a FRESH mega
    # session proposing a new stop for an already-held position) would
    # otherwise look identical to genuine external drift, since the old
    # settlement record and the new live-intended target simply disagree
    # by design, not because anything unexplained happened. carried_
    # forward already reflects the fresh target — when it disagrees with
    # what was recorded, that's normal pending execution work, not drift.
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4378.00  # still the OLD stop, amend hasn't executed yet this poll
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.00}}}
    carried_forward = {"XAUUSD": AllocationEntry(pct=1.0, price=4400.0, stop_loss=4360.00, side="buy")}
    assert _detect_external_stop_drift(positions_by_symbol, settled, carried_forward) == []


def test_drift_still_detected_when_fresh_target_agrees_with_what_was_recorded():
    # The other half of the fix above: when this app's OWN fresh target
    # matches what was already recorded (nothing about its own intent
    # changed), a live-stop mismatch has no other explanation and must
    # still be flagged.
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4390.03
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.00}}}
    carried_forward = {"XAUUSD": AllocationEntry(pct=1.0, price=4400.0, stop_loss=4378.00, side="buy")}
    warnings = _detect_external_stop_drift(positions_by_symbol, settled, carried_forward)
    assert len(warnings) == 1
    assert "XAUUSD" in warnings[0]


def test_drift_check_prefers_tactical_persisted_stop_over_stale_entry():
    # A tactical DEFEND already tightened this position once this cycle
    # (persisted_stop_loss) — that's the real last-known-good value this
    # app itself set, not the original, now-superseded entry.stop_loss.
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4385.00  # matches the tactical persistence, not the stale original entry
    settled = {
        "XAUUSD": {
            "entry": {"stop_loss": 4378.00},
            "tactical": {"persisted_stop_loss": 4385.00},
        }
    }
    assert _detect_external_stop_drift(positions_by_symbol, settled, {}) == []


# --- external stop-loss drift AUTO-RESTORE (2026-09-14 XAGUSD incident) ---
# Direct user instruction: "do not wait for any human intervention just
# act and save the equity" — _restore_external_stop_drift shares
# _iter_stop_drift's own detection core with _detect_external_stop_drift
# above, so every "no drift" case already covered there applies here too;
# these tests focus on what the restore actually DOES once drift is real.


def test_iter_stop_drift_returns_the_symbol_position_and_recorded_stop():
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4390.03
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.00}}}
    drifted = _iter_stop_drift(positions_by_symbol, settled, {}, tolerance_pct=0.05)
    assert len(drifted) == 1
    symbol, position, recorded_sl = drifted[0]
    assert symbol == "XAUUSD"
    assert position is positions_by_symbol["XAUUSD"]
    assert recorded_sl == 4378.00


@patch("ai.clerk_execution.modify_position_sltp")
def test_restore_writes_back_the_recorded_stop_when_drift_detected(mock_modify):
    mock_modify.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=1)
    positions_by_symbol = {"XAGUSD": _position(symbol="XAGUSD", side="buy", ticket=541112790)}
    positions_by_symbol["XAGUSD"].sl = 63.13  # drifted back down to the original stop
    settled = {
        "XAGUSD": {
            "entry": {"stop_loss": 63.13},
            "tactical": {"persisted_stop_loss": 63.97},  # Clerk's own tightened, break-even+ stop
        }
    }
    outcomes = _restore_external_stop_drift(positions_by_symbol, settled, {})
    assert len(outcomes) == 1
    assert "XAGUSD" in outcomes[0]
    assert "restored" in outcomes[0].lower()
    mock_modify.assert_called_once()
    called_position, kwargs = mock_modify.call_args.args[0], mock_modify.call_args.kwargs
    assert called_position is positions_by_symbol["XAGUSD"]
    assert kwargs["stop_loss"] == 63.97
    assert kwargs["take_profit"] == positions_by_symbol["XAGUSD"].tp


@patch("ai.clerk_execution.modify_position_sltp")
def test_restore_does_nothing_and_calls_nothing_when_no_drift(mock_modify):
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4378.00
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.00}}}
    assert _restore_external_stop_drift(positions_by_symbol, settled, {}) == []
    mock_modify.assert_not_called()


@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.modify_position_sltp")
def test_restore_falls_back_to_closing_the_position_when_the_amend_is_rejected(mock_modify, mock_close):
    # Direct follow-up instruction after replaying the XAGUSD incident:
    # if the intended restore level is no longer valid (price already
    # moved past it), close at market rather than leave the position
    # exposed at an unintended stop — "what worst could happen, i would
    # lose some more cash but its still better than bigger disaster."
    mock_modify.return_value = OrderResult(success=False, retcode=10016, comment="Invalid stops", ticket=None)
    mock_close.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=99)
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4390.03
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.00}}}
    outcomes = _restore_external_stop_drift(positions_by_symbol, settled, {})
    assert len(outcomes) == 1
    assert "Invalid stops" in outcomes[0]
    assert "closed the position at market" in outcomes[0]
    assert "STILL OPEN" not in outcomes[0]
    mock_close.assert_called_once_with(positions_by_symbol["XAUUSD"])


@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.modify_position_sltp")
def test_restore_reports_still_open_when_both_the_amend_and_the_close_fallback_fail(mock_modify, mock_close):
    mock_modify.return_value = OrderResult(success=False, retcode=10016, comment="Invalid stops", ticket=None)
    mock_close.return_value = OrderResult(success=False, retcode=10018, comment="Market closed", ticket=None)
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4390.03
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.00}}}
    outcomes = _restore_external_stop_drift(positions_by_symbol, settled, {})
    assert len(outcomes) == 1
    assert "STILL OPEN" in outcomes[0]
    assert "Invalid stops" in outcomes[0]
    assert "Market closed" in outcomes[0]


@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.modify_position_sltp")
def test_restore_falls_back_to_closing_when_mt5_connection_fails_without_raising(mock_modify, mock_close):
    from data.mt5_source import MT5ConnectionError

    mock_modify.side_effect = MT5ConnectionError("terminal unreachable")
    mock_close.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=99)
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4390.03
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.00}}}
    outcomes = _restore_external_stop_drift(positions_by_symbol, settled, {})
    assert len(outcomes) == 1
    assert "terminal unreachable" in outcomes[0]
    assert "closed the position at market" in outcomes[0]
    mock_close.assert_called_once_with(positions_by_symbol["XAUUSD"])


@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.modify_position_sltp")
def test_restore_reports_still_open_when_both_restore_and_close_raise_connection_errors(mock_modify, mock_close):
    from data.mt5_source import MT5ConnectionError

    mock_modify.side_effect = MT5ConnectionError("terminal unreachable")
    mock_close.side_effect = MT5ConnectionError("still unreachable")
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4390.03
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.00}}}
    outcomes = _restore_external_stop_drift(positions_by_symbol, settled, {})
    assert len(outcomes) == 1
    assert "STILL OPEN" in outcomes[0]
    assert "terminal unreachable" in outcomes[0]
    assert "still unreachable" in outcomes[0]


@patch("ai.clerk_execution.modify_position_sltp")
def test_restore_acts_regardless_of_drift_direction(mock_modify):
    # The 2026-09-08 XAUUSD incident drifted TIGHTER (a premature-stop-out
    # risk); the 2026-09-14 XAGUSD incident drifted LOOSER (a bigger-loss
    # risk). Both must be corrected the same way: restore what Clerk
    # itself last decided, regardless of which direction is "worse".
    mock_modify.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=1)
    positions_by_symbol = {"XAUUSD": _position(symbol="XAUUSD", side="buy")}
    positions_by_symbol["XAUUSD"].sl = 4390.03  # drifted TIGHTER than recorded
    settled = {"XAUUSD": {"entry": {"stop_loss": 4378.00}}}
    outcomes = _restore_external_stop_drift(positions_by_symbol, settled, {})
    assert len(outcomes) == 1
    mock_modify.assert_called_once()
    assert mock_modify.call_args.kwargs["stop_loss"] == 4378.00


# --- _detect_ollama_outage (2026-09-16, direct user request) ---


def test_detect_ollama_outage_fires_when_every_real_attempt_failed():
    results = [
        (PendingSetup(symbol="MSFT", side="buy", pct=0.2, price=490.0, stop_loss=485.0, trigger_condition="x"),
         False, OLLAMA_FAILED_MESSAGE),
        ("XAGUSD", False, OLLAMA_FAILED_MESSAGE),
        ("NVDA", TacticalVerdict(tier="hold"), OLLAMA_FAILED_MESSAGE),
    ]
    note = _detect_ollama_outage(results)
    assert "unreachable" in note
    assert "3/3" in note


def test_detect_ollama_outage_silent_when_at_least_one_succeeded():
    results = [
        ("MSFT", False, OLLAMA_FAILED_MESSAGE),
        ("XAGUSD", True, "FINAL_VERDICT: CONFIRMED"),
    ]
    assert _detect_ollama_outage(results) == ""


def test_detect_ollama_outage_silent_when_no_attempts_at_all():
    assert _detect_ollama_outage([]) == ""


def test_detect_ollama_outage_excludes_deterministic_circuit_breaker_verdicts():
    # A poll made up ENTIRELY of hard-exit/trend-flip circuit breakers
    # never calls Ollama at all -- must never be misread as "every real
    # attempt failed" just because none of them are real attempts.
    results = [
        ("NVDA", TacticalVerdict(tier="exit", hard_exit=True),
         "[deterministic circuit-breaker — no model call made] adverse move past ceiling."),
    ]
    assert _detect_ollama_outage(results) == ""


def test_detect_ollama_outage_excludes_circuit_breakers_from_the_denominator():
    # One real Ollama failure alongside an unrelated circuit-breaker
    # verdict this same poll -- the circuit breaker must not count
    # toward EITHER the numerator or denominator.
    results = [
        ("MSFT", False, OLLAMA_FAILED_MESSAGE),
        ("NVDA", TacticalVerdict(tier="exit", hard_exit=True),
         "[deterministic circuit-breaker — no model call made] adverse move past ceiling."),
    ]
    note = _detect_ollama_outage(results)
    assert "1/1" in note


# --- settlement reset: cancels unfilled orders, resets tracking ---


@patch("ai.clerk_execution.cancel_pending_order")
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


@patch("ai.clerk_execution.cancel_pending_order")
def test_reset_does_not_cancel_filled_or_closed_after_fill_symbols(mock_cancel):
    settled = {
        "EURUSD": {"origin": "immediate", "state": "filled", "entry": {}, "order_ticket": 100},
        "XAUUSD": {"origin": "pending_setup", "state": "closed_after_fill", "entry": {}, "order_ticket": 200},
    }
    _reset_settlement_for_new_session(settled)
    mock_cancel.assert_not_called()


@patch("ai.clerk_execution.cancel_pending_order")
def test_reset_clears_tracking_entirely(mock_cancel):
    mock_cancel.return_value = OrderResult(success=True, retcode=10009, comment="", ticket=100)
    settled = {"EURUSD": {"origin": "immediate", "state": "order_placed", "entry": {}, "order_ticket": 100}}
    result = _reset_settlement_for_new_session(settled)
    assert result == {}


@patch("ai.clerk_execution.cancel_pending_order", side_effect=Exception("should not be reached via non-MT5ConnectionError"))
def test_reset_logs_and_continues_when_cancel_raises_mt5_connection_error(mock_cancel, caplog):
    from data.mt5_source import MT5ConnectionError

    mock_cancel.side_effect = MT5ConnectionError("terminal unreachable")
    settled = {"EURUSD": {"origin": "immediate", "state": "order_placed", "entry": {}, "order_ticket": 100}}
    result = _reset_settlement_for_new_session(settled)  # must not raise
    assert result == {}


# --- interval due-check: rolling cooldown from the last run's own -----
# completion, replacing a fixed-clock-window scheme 2026-09-21 (direct
# user report — see ai.clerk_execution's own module comment above
# is_execution_due for the full incident) ------------------------------


def _utc(y, m, d, h, mi):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def test_is_execution_due_true_when_never_run_before():
    assert is_execution_due(_utc(2026, 8, 23, 15, 0), state={}) is True


def test_is_execution_due_false_before_the_interval_has_elapsed():
    state = {"last_run_completed_utc": _utc(2026, 8, 23, 15, 0).isoformat()}
    # Default interval is 15 minutes -- 14 minutes after completion is
    # still within the cooldown.
    assert is_execution_due(_utc(2026, 8, 23, 15, 14), state=state) is False


def test_is_execution_due_true_exactly_at_the_interval_boundary():
    state = {"last_run_completed_utc": _utc(2026, 8, 23, 15, 0).isoformat()}
    assert is_execution_due(_utc(2026, 8, 23, 15, 15), state=state) is True


def test_is_execution_due_true_well_past_the_interval():
    # A slow run, or the PC asleep through several intervals, must not
    # need to "catch up" to any clock alignment -- simply due the moment
    # enough real time has passed since the last completion.
    state = {"last_run_completed_utc": _utc(2026, 8, 23, 15, 0).isoformat()}
    assert is_execution_due(_utc(2026, 8, 23, 18, 47), state=state) is True


def test_is_execution_due_true_on_corrupt_completion_timestamp():
    state = {"last_run_completed_utc": "not a real timestamp"}
    assert is_execution_due(_utc(2026, 8, 23, 15, 0), state=state) is True


def test_is_execution_due_a_slow_run_never_silently_consumes_the_next_windows_slot(_fixed_files):
    # Real incident this whole scheme replaced: under the OLD fixed-
    # clock-window design, a run that started in one window but finished
    # inside the NEXT one marked that NEXT window "already ran" purely by
    # completing there -- silently consuming a fresh window's own due
    # slot. A rolling cooldown timed from real completion has no such
    # concept to get confused by: exactly `interval` minutes after a run
    # that took an unusually long 6 minutes to finish, the next check
    # must be due -- neither early (mid-cooldown) nor silently skipped.
    set_clerk_execution_interval_minutes(5)
    completed_at = _utc(2026, 8, 23, 15, 6)  # a run that started at ~15:00, finished at 15:06
    state = {"last_run_completed_utc": completed_at.isoformat()}
    assert is_execution_due(_utc(2026, 8, 23, 15, 10), state=state) is False  # 4 min since completion -- not yet
    assert is_execution_due(_utc(2026, 8, 23, 15, 11), state=state) is True  # exactly 5 min since completion


def test_pre_weekend_cleanup_due_on_friday_at_configured_hour():
    # 2026-09-11 is a real Friday; default CLERK_PRE_WEEKEND_CLEANUP_
    # HOUR_UTC is 18.
    assert is_pre_weekend_cleanup_due(_utc(2026, 9, 11, 18, 0)) is True


def test_pre_weekend_cleanup_not_due_before_the_configured_hour():
    assert is_pre_weekend_cleanup_due(_utc(2026, 9, 11, 17, 59)) is False


def test_pre_weekend_cleanup_not_due_on_a_non_friday():
    # 2026-09-10 is a real Thursday.
    assert is_pre_weekend_cleanup_due(_utc(2026, 9, 10, 18, 0)) is False


def test_pre_weekend_cleanup_still_due_later_the_same_friday_deliberately_no_dedup():
    # Real gap caught on a self-recheck (2026-09-12): an earlier version
    # of this function deduped to "once per Friday." This account's mega
    # session can be re-run manually at any time -- if it's re-run AFTER
    # a one-time window already fired and proposes a fresh non-crypto
    # pending setup, a dedup'd version would let it sit uncaught through
    # the whole weekend, silently recreating the exact incident this
    # feature exists to prevent. Deliberately fires again and again for
    # the rest of Friday -- safe, since cancelling an already-gone order
    # is a harmless no-op.
    assert is_pre_weekend_cleanup_due(_utc(2026, 9, 11, 19, 0)) is True
    assert is_pre_weekend_cleanup_due(_utc(2026, 9, 11, 23, 59)) is True


def test_pre_weekend_cleanup_disabled_via_config():
    with patch.object(config, "CLERK_PRE_WEEKEND_CLEANUP_ENABLED", False):
        assert is_pre_weekend_cleanup_due(_utc(2026, 9, 11, 18, 0)) is False


# --- enable/disable toggle + review-frequency override (2026-08-23) ---


def test_read_clerk_execution_enabled_true_when_file_missing(_fixed_files):
    assert read_clerk_execution_enabled() is True


def test_read_clerk_execution_enabled_true_on_corrupt_file(_fixed_files, tmp_path):
    corrupt_path = tmp_path / "corrupt_enabled.json"
    corrupt_path.write_text("{not valid json")
    with patch.object(config, "CLERK_EXECUTION_ENABLED_FILE", str(corrupt_path)):
        assert read_clerk_execution_enabled() is True


def test_set_clerk_execution_enabled_false_then_read_round_trips(_fixed_files):
    set_clerk_execution_enabled(False)
    assert read_clerk_execution_enabled() is False


def test_read_clerk_execution_interval_minutes_falls_back_to_config_when_file_missing(_fixed_files):
    # _fixed_files patches CLERK_EXECUTION_CHECK_INTERVAL_MINUTES to 15.
    assert read_clerk_execution_interval_minutes() == 15


def test_set_clerk_execution_interval_minutes_then_read_round_trips(_fixed_files):
    set_clerk_execution_interval_minutes(30)
    assert read_clerk_execution_interval_minutes() == 30


def test_read_clerk_execution_interval_minutes_falls_back_on_non_positive_value(_fixed_files, tmp_path):
    bad_path = tmp_path / "bad_interval.json"
    bad_path.write_text(json.dumps({"minutes": 0}))
    with patch.object(config, "CLERK_EXECUTION_INTERVAL_FILE", str(bad_path)):
        assert read_clerk_execution_interval_minutes() == 15


def test_is_execution_due_uses_the_overridden_interval(_fixed_files):
    set_clerk_execution_interval_minutes(30)
    state = {"last_run_completed_utc": _utc(2026, 8, 23, 15, 0).isoformat()}
    # Under the default 15-minute interval this would already be due;
    # the override to 30 minutes must push the real cooldown out to
    # match it, proving is_execution_due reads the live override, not
    # the config constant captured at import time.
    assert is_execution_due(_utc(2026, 8, 23, 15, 20), state=state) is False
    assert is_execution_due(_utc(2026, 8, 23, 15, 30), state=state) is True


def test_next_execution_check_utc_is_now_when_due(_fixed_files):
    now = _utc(2026, 8, 23, 15, 5)
    assert next_execution_check_utc(now, state={}) == now


def test_next_execution_check_utc_is_interval_minutes_after_last_completion(_fixed_files):
    now = _utc(2026, 8, 23, 15, 5)
    state = {"last_run_completed_utc": now.isoformat()}
    assert next_execution_check_utc(now, state=state) == _utc(2026, 8, 23, 15, 20)


# --- read_execution_state / read_execution_progress: safe defaults ---


def test_read_execution_state_empty_dict_when_file_missing():
    assert read_execution_state() == {}


def test_read_execution_progress_empty_dict_when_file_missing():
    assert read_execution_progress() == {}


def test_read_execution_state_empty_dict_on_corrupt_file(tmp_path):
    corrupt_path = tmp_path / "corrupt_state.json"
    corrupt_path.write_text("{not valid json")
    with patch.object(config, "CLERK_EXECUTION_STATE_FILE", str(corrupt_path)):
        assert read_execution_state() == {}


# --- _write_execution_state: preserves last_run_completed_utc, merges last_verdicts ---


def test_write_execution_state_preserves_last_run_completed_utc_across_calls(_fixed_files):
    from ai.clerk_execution import _mark_interval_ran, _write_execution_state

    marked_at = datetime(2026, 8, 23, 15, 5, tzinfo=timezone.utc)
    _mark_interval_ran(marked_at)
    # A later write for an unrelated outcome (e.g. a "no_suggestion" poll
    # on a later poll) must not erase the rolling-cooldown marker.
    _write_execution_state("no_suggestion")
    assert read_execution_state()["last_run_completed_utc"] == marked_at.isoformat()


def test_write_execution_state_merges_last_verdicts_instead_of_replacing(_fixed_files):
    from ai.clerk_execution import _write_execution_state

    _write_execution_state("success", last_verdicts={"EURUSD": {"confirmed": False}})
    # A later poll that only checks a DIFFERENT symbol (e.g. because the
    # pending-setups cap truncated the list) must not discard EURUSD's
    # still-relevant verdict from the earlier poll.
    _write_execution_state("success", last_verdicts={"XAUUSD": {"confirmed": True}})
    verdicts = read_execution_state()["last_verdicts"]
    assert verdicts["EURUSD"]["confirmed"] is False
    assert verdicts["XAUUSD"]["confirmed"] is True


def test_write_execution_state_fresh_verdict_overwrites_the_same_symbols_own_prior_one(_fixed_files):
    from ai.clerk_execution import _write_execution_state

    _write_execution_state("success", last_verdicts={"EURUSD": {"confirmed": False}})
    _write_execution_state("success", last_verdicts={"EURUSD": {"confirmed": True}})
    assert read_execution_state()["last_verdicts"]["EURUSD"]["confirmed"] is True


def test_write_execution_state_merges_last_execution_results_instead_of_replacing(_fixed_files):
    from ai.clerk_execution import _write_execution_state

    _write_execution_state("success", last_execution_results={"NVDA": {"success": False, "detail": "Market closed"}})
    _write_execution_state("success", last_execution_results={"BTCUSD": {"success": True, "detail": "placed"}})
    results = read_execution_state()["last_execution_results"]
    assert results["NVDA"]["success"] is False
    assert results["BTCUSD"]["success"] is True


def test_write_execution_state_fresh_execution_result_overwrites_the_same_symbols_own_prior_one(_fixed_files):
    from ai.clerk_execution import _write_execution_state

    _write_execution_state("success", last_execution_results={"NVDA": {"success": False, "detail": "Market closed"}})
    _write_execution_state("success", last_execution_results={"NVDA": {"success": True, "detail": "placed"}})
    assert read_execution_state()["last_execution_results"]["NVDA"]["success"] is True


# --- _describe_elapsed ---


def test_describe_elapsed_unknown_when_generated_utc_missing():
    from ai.clerk_execution import _describe_elapsed

    assert _describe_elapsed(None, datetime.now(timezone.utc)) == "unknown"


def test_describe_elapsed_minutes_only_under_an_hour():
    from ai.clerk_execution import _describe_elapsed

    generated = "2026-08-23T10:00:00+00:00"
    now = _utc(2026, 8, 23, 10, 25)
    assert _describe_elapsed(generated, now) == "25 minute(s)"


def test_describe_elapsed_hours_and_minutes_over_an_hour():
    from ai.clerk_execution import _describe_elapsed

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
    # config.CLERK_EXECUTION_RUN_TIMEOUT_SECONDS ceiling (600s default).
    from datetime import timedelta

    old_but_within_run_budget = (datetime.now(timezone.utc) - timedelta(minutes=8)).isoformat()
    progress = {"updated_utc": old_but_within_run_budget}
    state = {"last_attempt_utc": "2020-01-01T00:00:00+00:00"}
    assert execution_check_is_live(progress, state) is True


# --- run_clerk_execution_check: end-to-end with everything mocked ---


def _write_suggestion(tmp_path, immediate_allocation=None, pending_setups=None, generated_utc=None):
    payload = {
        "generated_utc": generated_utc or datetime.now(timezone.utc).isoformat(),
        "immediate_allocation": immediate_allocation or {},
        "pending_setups": pending_setups or [],
    }
    Path(config.MEGA_ANALYSIS_LATEST_SUGGESTION_FILE).write_text(json.dumps(payload))


def test_run_clerk_execution_check_skips_when_disabled(_fixed_files):
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": {"pct": 1.0, "side": "buy"}})
    set_clerk_execution_enabled(False)
    with patch("ai.clerk_execution.connect") as mock_connect:
        run_clerk_execution_check()
        mock_connect.assert_not_called()  # disabled must short-circuit before any MT5 connection
    assert read_execution_state()["last_status"] == "disabled"


def test_run_clerk_execution_check_noop_when_no_suggestion(_fixed_files):
    run_clerk_execution_check()
    assert read_execution_state()["last_status"] == "no_suggestion"


@patch("ai.clerk_execution._mega_session_is_live", return_value=True)
def test_run_clerk_execution_check_skips_when_mega_session_live(mock_live, _fixed_files):
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": {"pct": 1.0, "side": "buy"}})
    run_clerk_execution_check()
    assert read_execution_state()["last_status"] == "skipped_mega_live"


@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_executes_immediate_allocation(
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

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            immediate_allocation={"EURUSD": {"pct": 1.0, "side": "buy", "price": 1.0900, "stop_loss": 1.0850}},
        )
        run_clerk_execution_check()

    mock_open.assert_called_once()
    state = read_execution_state()
    assert state["last_status"] == "success"


@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_reanchors_a_stale_entry_to_the_real_m5_zone_end_to_end(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_close, mock_open, _fixed_files,
):
    # End-to-end replay of the real WHEAT/UKOIL shape (2026-09-22): an entry proposed far
    # from where real intraday structure sits is STALE (>3.5x M5 ATR from live price). The
    # FULL pipeline — real analyze_ftmo_asset_live with the M5 reads — must re-anchor it to
    # the real M5 zone, not just the guard in isolation. (An entry that is NOT stale is left
    # exactly as Claude proposed it — see the unit tests.)
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="TESTSYM", description="Test", bid=99.98, ask=100.0)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_open.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=556)

    sine_history = _m5_sine_wave_history()  # real oscillation, ~95-105, for every timeframe requested

    with (
        patch("ai.clerk_execution.get_contract_spec") as mock_spec,
        patch("ai.clerk_execution.fetch_mt5_price_history", return_value=sine_history),
        patch("ai.ftmo_suggest.fetch_mt5_price_history", return_value=sine_history),
    ):
        # trade_contract_size=1.0 (share-CFD style) — a forex-style 100,000 would make a $4
        # stop on a ~$100 instrument infeasible at the minimum lot / 1% risk, unrelated here.
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=1.0, currency_margin="USD", margin_initial=100.0,
        )
        _write_suggestion(
            _fixed_files,
            immediate_allocation={
                "TESTSYM": {"pct": 1.0, "side": "buy", "price": 90.0, "stop_loss": 86.0, "take_profit": 130.0}
            },
        )
        run_clerk_execution_check()

    mock_open.assert_called_once()
    placed_price = mock_open.call_args[0][3]  # open_position(symbol, side, volume, price, stop_loss, take_profit)
    assert placed_price != 90.0
    assert 94.0 < placed_price < 96.0  # the real M5 support band, not the original distant number


@patch("ai.clerk_execution.get_symbol_category")
@patch("ai.clerk_execution.is_pre_weekend_cleanup_due", return_value=True)
@patch("ai.clerk_execution.cancel_pending_order")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders")
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_cancels_non_crypto_pending_order_pre_weekend(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_cancel, mock_due, mock_category, _fixed_files,
):
    # Real incident this guards against: an INTC and a USDCHF pending
    # order both survived into a weekend because the only prior cancel
    # trigger never fired in time. EURUSD's pending order here matches
    # its own allocation target exactly -- absent the pre-weekend check
    # it would just "hold" -- confirming the cancel is driven by the new
    # check, not a pre-existing mismatch.
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    # Terms deliberately match _pending_order's own defaults (price_open=
    # 1.09, sl=1.08, tp=1.11) exactly, so absent the pre-weekend check
    # this would cleanly resolve to "hold" -- isolating the cancel as
    # coming from the new check, not an unrelated amend-tolerance mismatch.
    mock_pending.return_value = [_pending_order(symbol="EURUSD", ticket=321)]
    mock_cancel.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=321)
    mock_category.return_value = "Forex"

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            immediate_allocation={
                "EURUSD": {"pct": 1.0, "side": "buy", "price": 1.09, "stop_loss": 1.08, "take_profit": 1.11}
            },
        )
        run_clerk_execution_check()

    mock_cancel.assert_called_once_with(321)
    state = read_execution_state()
    assert state["last_status"] == "success"


@patch("ai.clerk_execution.get_symbol_category")
@patch("ai.clerk_execution.is_pre_weekend_cleanup_due", return_value=True)
@patch("ai.clerk_execution.cancel_pending_order")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders")
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_leaves_crypto_pending_order_alone_pre_weekend(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_cancel, mock_due, mock_category, _fixed_files,
):
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="BTCUSD", description="Bitcoin", bid=59999.0, ask=60000.0)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_pending.return_value = [
        PendingOrder(symbol="BTCUSD", volume=1.0, order_type="buy limit", price_open=60000.0,
                     sl=58000.0, tp=62000.0, ticket=654)
    ]
    mock_category.return_value = "Crypto I CFD"

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=1.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            immediate_allocation={
                # pct=2.0: with a $100k equity, $2000 stop distance, and
                # a 1.0 contract size, this sizes to EXACTLY the pending
                # order's own 1.0-lot volume -- avoiding an unrelated
                # amend-tolerance mismatch that would independently
                # trigger its own cancel-then-reopen.
                "BTCUSD": {
                    "pct": 2.0, "side": "buy", "price": 60000.0,
                    "stop_loss": 58000.0, "take_profit": 62000.0,
                }
            },
        )
        run_clerk_execution_check()

    mock_cancel.assert_not_called()


class _SaturdayDatetime(datetime):
    """A real datetime subclass whose .now() always lands on an actual
    Saturday -- used to test the pre_weekend_due widening below without
    disturbing anything else in run_clerk_execution_check that calls
    datetime.now() (isinstance/arithmetic all still work normally, since
    this is a real subclass, not a bare MagicMock)."""

    @classmethod
    def now(cls, tz=None):
        base = datetime(2026, 9, 12, 10, 0)
        return base.replace(tzinfo=tz) if tz else base


@patch("ai.clerk_execution.datetime", _SaturdayDatetime)
@patch("ai.clerk_execution.is_symbol_tradable_now")
@patch("ai.clerk_execution.is_pre_weekend_cleanup_due", return_value=False)
@patch("ai.clerk_execution.cancel_pending_order")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders")
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_cancels_non_crypto_pending_order_on_saturday_even_when_friday_trigger_not_due(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_cancel, mock_due, mock_tradable, _fixed_files,
):
    # Real gap this closes: the Friday-only trigger wouldn't catch a
    # fresh non-crypto pending order created mid-weekend (e.g. the mega
    # session re-run on a Saturday) until the FOLLOWING Friday. The
    # pre_weekend_due check must also fire on an actual Saturday/Sunday,
    # independent of is_pre_weekend_cleanup_due's own Friday-hour gate
    # (mocked False here specifically to isolate this). The reactive
    # Sat/Sun path calls is_symbol_tradable_now directly (not the
    # crypto-only category check the Friday path uses) -- mocked False
    # here (genuinely closed, e.g. actual Saturday) to isolate that.
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_pending.return_value = [_pending_order(symbol="EURUSD", ticket=321)]
    mock_cancel.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=321)
    mock_tradable.return_value = False

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            immediate_allocation={
                "EURUSD": {"pct": 1.0, "side": "buy", "price": 1.09, "stop_loss": 1.08, "take_profit": 1.11}
            },
        )
        run_clerk_execution_check()

    mock_cancel.assert_called_once_with(321)


@patch("ai.clerk_execution.datetime", _SaturdayDatetime)
@patch("ai.clerk_execution.is_symbol_tradable_now")
@patch("ai.clerk_execution.is_pre_weekend_cleanup_due", return_value=False)
@patch("ai.clerk_execution.cancel_pending_order")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders")
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_leaves_reopened_forex_pending_order_alone_on_sunday(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_cancel, mock_due, mock_tradable, _fixed_files,
):
    # Real bug caught on a self-recheck: the reactive Sat/Sun backstop
    # must NOT use a blunt crypto-only rule the way the proactive Friday
    # trigger does -- forex reopens Sunday evening UTC, before every
    # other category. A crypto-only rule here would keep cancelling a
    # legitimate forex pending order the mega session correctly proposes
    # AFTER forex has already reopened, directly contradicting the "mega
    # session should suggest based on available assets" goal this whole
    # feature exists for. is_symbol_tradable_now=True (forex genuinely
    # reopened) must leave this pending order alone even though it's
    # still the weekend.
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_pending.return_value = [_pending_order(symbol="EURUSD", ticket=321)]
    mock_tradable.return_value = True

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            immediate_allocation={
                "EURUSD": {"pct": 1.0, "side": "buy", "price": 1.09, "stop_loss": 1.08, "take_profit": 1.11}
            },
        )
        run_clerk_execution_check()

    mock_cancel.assert_not_called()


@patch("ai.clerk_execution.is_pre_weekend_cleanup_due", return_value=False)
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.cancel_pending_order")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders")
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_still_cancels_when_heat_blocked_but_skips_new_risk(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_cancel, mock_open, mock_due, _fixed_files,
):
    # Real gap this fixes: a heat-blocked poll used to return BEFORE
    # compute_rebalance_plan ever ran, silently preventing ANY pure
    # risk-reducing cancel (pre-weekend cleanup included) from executing
    # on exactly the days aggregate heat is already too high -- the days
    # freeing that risk-budget matters most. A cancel (EURUSD, no longer
    # wanted) must still go through; a brand-new open (XAUUSD, real NEW
    # risk) must still be skipped.
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [
        MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900),
        MarketAsset(symbol="XAUUSD", description="Gold", bid=1999.0, ask=2000.0),
    ]
    # Tight headroom: XAUUSD's own 2% pct alone (see allocation below)
    # already exceeds 50% of a 1.0% remaining daily-loss headroom.
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=1.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_pending.return_value = [_pending_order(symbol="EURUSD", ticket=999)]
    mock_cancel.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=999)

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            immediate_allocation={
                # pct: 0 -- mega session no longer wants EURUSD; the
                # existing pending order should be cancelled regardless
                # of the heat block, since a cancel can only reduce risk.
                "EURUSD": {"pct": 0.0, "side": "buy"},
                # A genuinely NEW position -- real added risk, must be
                # skipped while heat-blocked.
                "XAUUSD": {"pct": 2.0, "side": "buy", "price": 2000.0, "stop_loss": 1900.0},
            },
        )
        run_clerk_execution_check()

    mock_cancel.assert_called_once_with(999)
    mock_open.assert_not_called()
    results = read_execution_state()["last_execution_results"]
    assert results["XAUUSD"]["success"] is False
    assert "heat-blocked" in results["XAUUSD"]["detail"].lower()


@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_records_a_failed_immediate_allocation_attempt(
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
        "ai.clerk_execution.compute_rebalance_plan",
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
        run_clerk_execution_check()

    mock_open.assert_called_once()
    results = read_execution_state()["last_execution_results"]
    assert results["NVDA"]["success"] is False
    assert results["NVDA"]["detail"] == "Market closed"


@patch("ai.clerk_execution.check_execution_safety_gates", return_value=(False, "Execution blocked: test reason."))
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_never_executes_when_safety_gate_blocks(
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

    with patch("ai.clerk_execution.open_position") as mock_open:
        _write_suggestion(
            _fixed_files,
            immediate_allocation={"EURUSD": {"pct": 1.0, "side": "buy", "price": 1.0900, "stop_loss": 1.0850}},
        )
        run_clerk_execution_check()
        mock_open.assert_not_called()

    state = read_execution_state()
    assert state["last_status"] == "blocked"


# --- local-model primary/backup chain (_run_clerk_prompt) ---
# Direct user request 2026-08-30: Copilot CLI + the OpenRouter backup
# chain are removed entirely, replaced by two models on a local Ollama
# server — config.CLERK_PRIMARY_MODEL tried first, config.CLERK_BACKUP_
# MODEL as the fallback (both thinking-disabled; see config.py's own
# comments for which specific models and why — swapped 2026-08-31).


@patch("ai.clerk_execution.run_ollama")
def test_run_clerk_prompt_uses_primary_when_available(mock_run_ollama):
    mock_run_ollama.return_value = "Real reasoning.\nFINAL_VERDICT: NOT_CONFIRMED"
    result = _run_clerk_prompt("some prompt", timeout=150)
    assert result == "Real reasoning.\nFINAL_VERDICT: NOT_CONFIRMED"
    mock_run_ollama.assert_called_once_with("some prompt", model=config.CLERK_PRIMARY_MODEL, timeout=150, keep_alive=config.CLERK_LLM_KEEP_ALIVE, max_tokens=config.CLERK_LLM_MAX_TOKENS)


@patch("ai.clerk_execution.run_ollama")
def test_run_clerk_prompt_falls_back_to_backup_model_on_primary_failure(mock_run_ollama):
    mock_run_ollama.side_effect = [OLLAMA_FAILED_MESSAGE, "Backup reasoning.\nFINAL_VERDICT: CONFIRMED"]
    result = _run_clerk_prompt("some prompt", timeout=150)
    assert config.CLERK_PRIMARY_MODEL in result
    assert config.CLERK_BACKUP_MODEL in result
    assert "Backup reasoning.\nFINAL_VERDICT: CONFIRMED" in result
    assert mock_run_ollama.call_count == 2
    models_tried = [call.kwargs["model"] for call in mock_run_ollama.call_args_list]
    assert models_tried == [config.CLERK_PRIMARY_MODEL, config.CLERK_BACKUP_MODEL]


@patch("ai.clerk_execution.run_ollama", return_value=OLLAMA_FAILED_MESSAGE)
def test_run_clerk_prompt_returns_primarys_own_failure_when_both_models_fail(mock_run_ollama):
    result = _run_clerk_prompt("some prompt", timeout=150)
    assert result == OLLAMA_FAILED_MESSAGE
    assert mock_run_ollama.call_count == 2
    assert parse_clerk_verdict(result) is False


def test_parse_clerk_verdict_still_works_on_a_backup_attributed_response():
    text = (
        f"[{config.CLERK_PRIMARY_MODEL} unavailable — backup model {config.CLERK_BACKUP_MODEL} responded]\n\n"
        "Some reasoning about the condition.\nFINAL_VERDICT: CONFIRMED"
    )
    assert parse_clerk_verdict(text) is True


@patch("ai.clerk_execution.is_symbol_tradable_now", return_value=True)
@patch("ai.clerk_execution._run_clerk_verdict")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_executes_a_confirmed_pending_setup(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_verdict, mock_tradable, _fixed_files,
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

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
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
        run_clerk_execution_check()

    mock_open.assert_called_once()
    state = read_execution_state()
    assert state["last_verdicts"]["XAUUSD"]["confirmed"] is True


@patch("ai.clerk_execution.is_symbol_tradable_now", return_value=True)
@patch("ai.clerk_execution._run_clerk_verdict")
@patch("ai.clerk_execution._fetch_technical_context")
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_a_structured_trigger_is_evaluated_in_python_and_fires_a_market_entry_without_the_llm(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_verdict, mock_tradable, _fixed_files,
):
    import pandas as pd
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=2000.1, ask=2000.3)]
    bars = pd.DataFrame([(1999.0, 1997.0, 1998.0), (2000.4, 1998.0, 1999.8)], columns=["High", "Low", "Close"])
    mock_fetch_ctx.return_value = (
        _fake_ftmo_analysis(symbol="XAUUSD", m5_stats=_fake_technical_stats(atr=2.0), m5_recent=bars), "fake technical context"
    )
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_open.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=778)

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            pending_setups=[
                {"symbol": "XAUUSD", "side": "buy", "pct": 1.0, "trigger_condition": "M5 closes above 2000",
                 "price": 2000.0, "stop_loss": 1990.0, "take_profit": 2020.0, "reason": "r",
                 "trigger": {"kind": "range_break", "level": 2000.0}}
            ],
        )
        run_clerk_execution_check()

    mock_verdict.assert_not_called()
    mock_open.assert_called_once()
    assert mock_open.call_args.kwargs["kind"] == "market"
    verdict = read_execution_state()["last_verdicts"]["XAUUSD"]
    assert verdict["confirmed"] is True and "[structured trigger]" in verdict["raw_text"]


@patch("ai.clerk_execution.is_symbol_tradable_now", return_value=False)
@patch("ai.clerk_execution._run_clerk_verdict")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_skips_pending_setup_trigger_when_market_closed(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_verdict, mock_tradable, _fixed_files,
):
    # Real gap this closes, caught on a self-recheck: this is the one
    # Clerk decision point that places a genuinely NEW order. Without
    # this gate, Clerk's weak local model would evaluate a pending
    # setup's trigger_condition against frozen weekend data with zero
    # awareness the market is closed (its tactical prompt deliberately
    # excludes market-status -- see include_market_status's own
    # docstring) and could confirm it anyway, placing a real resting
    # order the reactive backstop would only catch a full poll later.
    # is_symbol_tradable_now=False must skip the LLM call entirely --
    # no verdict, no order, no state entry at all for this poll.
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=1999.0, ask=2000.0)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    setup = PendingSetup(
        symbol="XAUUSD", side="buy", pct=1.0, trigger_condition="cond",
        price=2000.0, stop_loss=1980.0, take_profit=2050.0,
    )
    mock_verdict.return_value = (setup, True, "FINAL_VERDICT: CONFIRMED")

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
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
        run_clerk_execution_check()

    mock_verdict.assert_not_called()
    mock_open.assert_not_called()
    state = read_execution_state()
    assert "XAUUSD" not in state["last_verdicts"]


@patch("ai.clerk_execution.is_symbol_tradable_now", return_value=True)
@patch("ai.clerk_execution._run_clerk_verdict")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_run_clerk_execution_check_does_not_execute_an_unconfirmed_pending_setup(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_verdict, mock_tradable, _fixed_files,
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
    run_clerk_execution_check()

    mock_open.assert_not_called()
    state = read_execution_state()
    assert state["last_verdicts"]["XAUUSD"]["confirmed"] is False


# --- watched positions: the Clerk's own invalidation-condition check (2026-08-23) ---


def _seed_settled(fixed_files, symbol, state, generated_utc, entry=None):
    Path(config.CLERK_EXECUTION_SETTLEMENT_FILE).write_text(json.dumps({
        "generated_utc": generated_utc,
        "settled": {
            symbol: {
                "origin": "immediate", "state": state,
                "entry": entry or {"pct": 1.0, "price": 1.09, "stop_loss": 1.08, "take_profit": None, "side": "buy"},
                "order_ticket": 100,
            }
        },
    }))


@patch("ai.clerk_execution._run_clerk_invalidation_check")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.modify_position_sltp")
@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_confirmed_invalidation_forces_pct_zero_and_closes_the_position(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_close, mock_modify, mock_fetch_ctx, mock_invalidation, _fixed_files,
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
    # This test's own position fixture (_position's default sl=1.08) and
    # its settled entry (stop_loss=1.0850) incidentally differ beyond
    # AMEND_TOLERANCE_PCT — real, unrelated drift-restore noise this test
    # isn't about; mocked to succeed so it doesn't fall through to the
    # close-fallback and confuse mock_close's own assertion below.
    mock_modify.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=200)
    mock_invalidation.return_value = ("EURUSD", True, "FINAL_VERDICT: CONFIRMED")

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {
        "pct": 1.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None,
        "side": "buy", "reason": "r", "invalidation_condition": "H1 closes below 1.0800",
    }
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "EURUSD", "filled", generated_utc, entry=entry)

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_clerk_execution_check()

    mock_invalidation.assert_called_once()
    mock_close.assert_called_once()
    state = read_execution_state()
    assert state["last_verdicts"]["EURUSD"]["confirmed"] is True


# --- tactical-defense candidates: must be driven by REAL live positions, not settlement bookkeeping ---


@patch("ai.clerk_execution._run_clerk_tactical_check")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(symbol="USDCAD"), "fake technical context"),
)
@patch("ai.clerk_execution.modify_position_sltp")
@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders")
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_tactical_check_runs_for_a_held_position_despite_a_stuck_settlement_record(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch, mock_status,
    mock_permitted, mock_open, mock_close, mock_modify, mock_fetch_ctx, mock_tactical, _fixed_files,
):
    # Real incident, 2026-09-08: USDCAD had an already-filled short AND a
    # separate, later-triggered "top-up" Pending Setup resting on the
    # SAME symbol. settlement tracks exactly one record per symbol, and
    # the newer pending order's own "order_placed" state clobbered all
    # trace of the older filled position at that same symbol key. The
    # tactical-defense candidate list used to require
    # settlement["settled"][symbol]["state"] == "filled" directly, so
    # this real, live, held position got ZERO tactical checks for hours
    # while it ran to 86.6% of its own target and gave much of the gain
    # back before anything could act on it. This test seeds exactly that
    # stuck settlement state and confirms the fix — driving the candidate
    # list off real live MT5 positions instead of settlement's own
    # bookkeeping — checks it anyway.
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_positions.return_value = [_position(symbol="USDCAD", side="sell", volume=0.02, ticket=536730345)]
    mock_pending.return_value = [_pending_order(symbol="USDCAD", ticket=536856927)]
    mock_watch.return_value = [MarketAsset(symbol="USDCAD", description="USD/CAD", bid=1.3780, ask=1.3781)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    for m in (mock_open, mock_close, mock_modify):
        m.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=1)
    mock_tactical.return_value = ("USDCAD", TacticalVerdict(tier="hold"), "FINAL_VERDICT: HOLD")

    generated_utc = datetime.now(timezone.utc).isoformat()
    # The settlement record is stuck tracking the NEWER pending "top-up"
    # order (536856927), not the older, already-filled position
    # (536730345) — exactly the real, observed clobbering.
    _seed_settled(
        _fixed_files, "USDCAD", "order_placed", generated_utc,
        entry={"pct": 0.13, "price": 1.3815, "stop_loss": 1.385, "take_profit": 1.3751, "side": "sell"},
    )

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(_fixed_files, generated_utc=generated_utc)  # no fresh immediate_allocation this cycle
        run_clerk_execution_check()

    mock_tactical.assert_called_once()
    assert mock_tactical.call_args.args[0] == "USDCAD"


@patch("ai.clerk_execution._run_clerk_invalidation_check")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.modify_position_sltp")
@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_not_confirmed_invalidation_leaves_merged_allocation_unchanged(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_close, mock_modify, mock_fetch_ctx, mock_invalidation, _fixed_files,
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
    # See test_confirmed_invalidation_forces_pct_zero_and_closes_the_
    # position's own comment: this fixture's live sl (1.08) vs recorded
    # stop_loss (1.0850) is incidental drift-restore noise, unrelated to
    # what this test is actually about — mocked to succeed so it can't
    # fall through to the close-fallback and trip mock_close's assertion.
    mock_modify.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=200)
    mock_invalidation.return_value = ("EURUSD", False, "FINAL_VERDICT: NOT_CONFIRMED")

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {
        "pct": 1.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None,
        "side": "buy", "reason": "r", "invalidation_condition": "H1 closes below 1.0800",
    }
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "EURUSD", "filled", generated_utc, entry=entry)

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_clerk_execution_check()

    mock_invalidation.assert_called_once()
    mock_close.assert_not_called()
    state = read_execution_state()
    assert state["last_verdicts"]["EURUSD"]["confirmed"] is False


@patch("ai.clerk_execution._run_clerk_invalidation_check")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
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

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec, patch("ai.clerk_execution.close_position"):
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_clerk_execution_check()

    mock_invalidation.assert_not_called()


@patch("ai.clerk_execution._run_clerk_invalidation_check")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
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

    run_clerk_execution_check()

    mock_invalidation.assert_not_called()


@patch("ai.clerk_execution._run_clerk_invalidation_check")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
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
    Path(config.CLERK_EXECUTION_SETTLEMENT_FILE).write_text(json.dumps({
        "generated_utc": generated_utc,
        "settled": {
            "EURUSD": {"origin": "immediate", "state": "filled", "entry": entry_eur, "order_ticket": 200},
            "GBPUSD": {"origin": "immediate", "state": "filled", "entry": entry_gbp, "order_ticket": 201},
        },
    }))

    with (
        patch.object(config, "CLERK_EXECUTION_MAX_WATCHED_POSITIONS", 1),
        patch("ai.clerk_execution.get_contract_spec") as mock_spec,
        patch("ai.clerk_execution.close_position"),
    ):
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_clerk_execution_check()

    mock_invalidation.assert_called_once()


@patch("ai.clerk_execution.is_symbol_tradable_now", return_value=True)
@patch("ai.clerk_execution._run_clerk_invalidation_check")
@patch("ai.clerk_execution._run_clerk_verdict")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_pending_setup_verdicts_and_invalidation_verdicts_both_processed_in_one_poll(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_verdict, mock_invalidation, mock_tradable,
    _fixed_files,
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

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_clerk_execution_check()

    mock_verdict.assert_called_once()
    mock_invalidation.assert_called_once()
    state = read_execution_state()
    assert state["last_verdicts"]["XAUUSD"]["confirmed"] is True
    assert state["last_verdicts"]["EURUSD"]["confirmed"] is False


# --- regression: cancelling a STRAY pending order must never wipe the ---
# settlement record for an unrelated, still-held "filled" position on the
# same symbol (real bug found on self-review 2026-08-24)


@patch("ai.clerk_execution.cancel_pending_order")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders")
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
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

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_clerk_execution_check()

    mock_cancel.assert_called_once_with(999)
    settled = read_settlement()["settled"]
    assert "EURUSD" in settled
    assert settled["EURUSD"]["state"] == "filled"


# --- regression: an outstanding "increase" order must never force-close ---
# the already-held base position it was topping up (real bug found on audit)


@patch("ai.clerk_execution.modify_position_sltp")
@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders")
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_held_position_with_outstanding_increase_order_is_never_force_closed(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_close, mock_modify, _fixed_files,
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
    # This fixture's live sl (1.08, from _position's default) vs the
    # settled entry's stop_loss (1.0850) is incidental drift-restore
    # noise unrelated to what this test is about — mocked to succeed so
    # it can't fall through to the close-fallback and trip mock_close's
    # own "never force-closed" assertion below.
    mock_modify.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=1)

    generated_utc = datetime.now(timezone.utc).isoformat()
    _write_suggestion(
        _fixed_files,
        immediate_allocation={"EURUSD": {"pct": 2.0, "side": "buy", "price": 1.0900, "stop_loss": 1.0850}},
        generated_utc=generated_utc,
    )
    # Pre-seed settlement: the SAME mega-session cycle already placed an
    # "increase" order for EURUSD (ticket 100), still unfilled.
    Path(config.CLERK_EXECUTION_SETTLEMENT_FILE).write_text(json.dumps({
        "generated_utc": generated_utc,
        "settled": {
            "EURUSD": {
                "origin": "immediate", "state": "order_placed",
                "entry": {"pct": 2.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None, "side": "buy"},
                "order_ticket": 100,
            }
        },
    }))

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_clerk_execution_check()

    # The real, already-filled position must never be force-closed just
    # because an unrelated top-up order is still pending.
    mock_close.assert_not_called()
    # And no DUPLICATE order gets stacked on top of the already-
    # outstanding one for the same symbol.
    mock_open.assert_not_called()


# --- _apply_correlation_guard (2026-09-02, real gap found on audit) ---
# Real incident this closes: ETHUSD, BTCUSD, and XAGUSD each hit their
# own correctly-sized stop-loss within 2 minutes of each other on
# 2026-08-28, compounding into 71% of this account's entire realized
# loss over the audited period, because nothing checked a NEW candidate
# against what was already exposed for correlation before firing it.
#
# Deterministic, hand-verified correlated/uncorrelated daily-close
# series (not live randomness) so these tests can never be flaky:
# _CORRELATED_A/_B share the same underlying trend plus small
# independent noise (verified r=0.98); _UNCORRELATED_A/_B are two
# independent random walks (verified r=0.02).
_CORR_RNG_TREND = np.cumsum(np.random.default_rng(42).normal(0.5, 1.0, 90)) + 100
_CORRELATED_A = pd.Series(_CORR_RNG_TREND + np.random.default_rng(42).normal(0, 0.1, 90))
_CORRELATED_B = pd.Series(_CORR_RNG_TREND * 1.5 + np.random.default_rng(43).normal(0, 0.1, 90))
_UNCORRELATED_A = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 1, 90)) + 200)
_UNCORRELATED_B = pd.Series(np.cumsum(np.random.default_rng(99).normal(0, 1, 90)) + 300)
# Negatively correlated with _CORRELATED_A (verified r=-0.97) — used to
# test the direction-aware hedge logic below.
_NEGATIVELY_CORRELATED_B = pd.Series(
    -_CORR_RNG_TREND * 1.5 + 500 + np.random.default_rng(43).normal(0, 0.1, 90)
)


def _closes_by_symbol(mapping: dict) -> "callable":
    return lambda symbol: mapping.get(symbol)


def test_correlation_guard_reduces_new_candidate_correlated_with_held_position():
    allocation = {
        "BTCUSD": AllocationEntry(pct=0.5, price=79000.0, stop_loss=77600.0, take_profit=82811.0, side="buy", reason="btc thesis"),
    }
    positions = [_position(symbol="ETHUSD")]
    with patch(
        "ai.clerk_execution._fetch_correlation_closes",
        side_effect=_closes_by_symbol({"BTCUSD": _CORRELATED_A, "ETHUSD": _CORRELATED_B}),
    ):
        result = _apply_correlation_guard(allocation, positions)

    assert result["BTCUSD"].pct == pytest.approx(0.5 * config.CLERK_CORRELATION_SIZE_FACTOR)  # trimmed from 0.5
    assert "Correlation guard" in result["BTCUSD"].reason
    assert "btc thesis" in result["BTCUSD"].reason  # original reason preserved, not replaced
    # every other field carried through unchanged
    assert result["BTCUSD"].price == 79000.0
    assert result["BTCUSD"].stop_loss == 77600.0
    assert result["BTCUSD"].take_profit == 82811.0
    assert result["BTCUSD"].side == "buy"


def test_correlation_guard_leaves_uncorrelated_candidate_unchanged():
    allocation = {
        "EURUSD": AllocationEntry(pct=0.5, price=1.09, stop_loss=1.08, side="buy", reason="eur thesis"),
    }
    positions = [_position(symbol="XAUUSD")]
    with patch(
        "ai.clerk_execution._fetch_correlation_closes",
        side_effect=_closes_by_symbol({"EURUSD": _UNCORRELATED_A, "XAUUSD": _UNCORRELATED_B}),
    ):
        result = _apply_correlation_guard(allocation, positions)

    assert result["EURUSD"] == AllocationEntry(pct=0.5, price=1.09, stop_loss=1.08, side="buy", reason="eur thesis")


def test_correlation_guard_catches_two_brand_new_correlated_candidates_in_the_same_poll():
    # Real gap this specifically covers: not just "new stacks on old",
    # but two correlated symbols BOTH confirming fresh in the same poll,
    # neither one already held yet.
    allocation = {
        "BTCUSD": AllocationEntry(pct=0.5, price=79000.0, stop_loss=77600.0, side="buy", reason="btc"),
        "ETHUSD": AllocationEntry(pct=0.3, price=2480.0, stop_loss=2440.0, side="buy", reason="eth"),
    }
    with patch(
        "ai.clerk_execution._fetch_correlation_closes",
        side_effect=_closes_by_symbol({"BTCUSD": _CORRELATED_A, "ETHUSD": _CORRELATED_B}),
    ):
        result = _apply_correlation_guard(allocation, [])

    # Processed in sorted order (BTCUSD before ETHUSD) -- the first one
    # seen has nothing exposed yet to correlate against, so it stays
    # full size; the second sees the first as already-exposed and gets
    # reduced.
    assert result["BTCUSD"].pct == pytest.approx(0.5)
    assert result["ETHUSD"].pct == pytest.approx(0.3 * config.CLERK_CORRELATION_SIZE_FACTOR)
    assert "Correlation guard" in result["ETHUSD"].reason


def test_correlation_guard_never_touches_an_already_held_symbol_itself():
    # The guard protects against NEW stacking -- it must never resize a
    # position that's already open, even if it's highly correlated with
    # something else in the allocation.
    allocation = {
        "BTCUSD": AllocationEntry(pct=0.5, price=79000.0, stop_loss=77600.0, side="buy", reason="btc, already held"),
        "ETHUSD": AllocationEntry(pct=0.3, price=2480.0, stop_loss=2440.0, side="buy", reason="eth, new"),
    }
    positions = [_position(symbol="BTCUSD")]  # BTCUSD is already an open position
    with patch(
        "ai.clerk_execution._fetch_correlation_closes",
        side_effect=_closes_by_symbol({"BTCUSD": _CORRELATED_A, "ETHUSD": _CORRELATED_B}),
    ):
        result = _apply_correlation_guard(allocation, positions)

    assert result["BTCUSD"].pct == pytest.approx(0.5)  # untouched -- not a "candidate"
    assert result["ETHUSD"].pct == pytest.approx(0.3 * config.CLERK_CORRELATION_SIZE_FACTOR)  # the new one is still reduced


def test_correlation_guard_ignores_cash():
    allocation = {"CASH": AllocationEntry(pct=99.0, side="buy", reason="")}
    result = _apply_correlation_guard(allocation, [])
    assert result["CASH"].pct == 99.0


def test_correlation_guard_skips_pairs_with_missing_price_data_without_crashing():
    allocation = {
        "BTCUSD": AllocationEntry(pct=0.5, price=79000.0, stop_loss=77600.0, side="buy", reason="btc"),
    }
    positions = [_position(symbol="ETHUSD")]
    with patch("ai.clerk_execution._fetch_correlation_closes", return_value=None):
        result = _apply_correlation_guard(allocation, positions)  # must not raise

    assert result["BTCUSD"].pct == pytest.approx(0.5)  # can't assess -> proceeds at full size


def test_correlation_guard_returns_unchanged_when_there_are_no_candidates():
    # Every symbol either already held or at pct<=0 -- nothing to check.
    allocation = {
        "BTCUSD": AllocationEntry(pct=0.0, side="buy", reason="closing"),
    }
    positions = [_position(symbol="ETHUSD")]
    result = _apply_correlation_guard(allocation, positions)
    assert result is allocation  # same object, not just equal -- confirms the early-return path


def test_fetch_correlation_closes_returns_none_when_data_is_too_short():
    empty_df = pd.DataFrame({"Close": pd.Series(dtype=float)})
    with patch("ai.clerk_execution.fetch_mt5_price_history", return_value=empty_df):
        assert _fetch_correlation_closes("EURUSD") is None


def test_fetch_correlation_closes_returns_series_when_data_is_sufficient():
    long_df = pd.DataFrame({"Close": pd.Series(range(200), dtype=float)})
    with patch("ai.clerk_execution.fetch_mt5_price_history", return_value=long_df):
        result = _fetch_correlation_closes("EURUSD")
    assert result is not None
    assert len(result) == 200


# --- direction-aware hedge detection: a real correctness bug caught on
# self-review before this ever shipped. Raw |correlation| alone can't
# tell a genuine risk-stack apart from a genuine hedge — the trade
# DIRECTION on both legs matters too. ---


def test_correlation_guard_does_not_reduce_a_hedge_same_side_negative_correlation():
    # Both BUY, but the two instruments are negatively correlated
    # (r=-0.97) — when one tends to rise the other tends to fall, so
    # holding both LONG is a natural hedge, not a stacked bet. Must NOT
    # be reduced.
    allocation = {
        "BTCUSD": AllocationEntry(pct=0.5, price=79000.0, stop_loss=77600.0, side="buy", reason="btc"),
    }
    positions = [_position(symbol="ETHUSD", side="buy")]
    with patch(
        "ai.clerk_execution._fetch_correlation_closes",
        side_effect=_closes_by_symbol({"BTCUSD": _CORRELATED_A, "ETHUSD": _NEGATIVELY_CORRELATED_B}),
    ):
        result = _apply_correlation_guard(allocation, positions)

    assert result["BTCUSD"].pct == pytest.approx(0.5)  # unchanged -- this is a hedge, not a stack


def test_correlation_guard_does_not_reduce_a_hedge_opposite_side_positive_correlation():
    # Positively correlated (r=0.98), but one is BUY and the other SELL
    # — betting the pair moves apart, a hedge, not a stack. Must NOT be
    # reduced.
    allocation = {
        "BTCUSD": AllocationEntry(pct=0.5, price=79000.0, stop_loss=81000.0, side="sell", reason="btc short"),
    }
    positions = [_position(symbol="ETHUSD", side="buy")]
    with patch(
        "ai.clerk_execution._fetch_correlation_closes",
        side_effect=_closes_by_symbol({"BTCUSD": _CORRELATED_A, "ETHUSD": _CORRELATED_B}),
    ):
        result = _apply_correlation_guard(allocation, positions)

    assert result["BTCUSD"].pct == pytest.approx(0.5)  # unchanged -- this is a hedge, not a stack


def test_correlation_guard_reduces_a_real_stack_opposite_side_negative_correlation():
    # Negatively correlated (r=-0.97), but BOTH legs bet the SAME way:
    # long the one that tends to rise, short the one that tends to fall
    # WITH it (per the negative correlation, they fall/rise together in
    # opposite directions) -- both legs win or lose together. This IS a
    # real stack and must be reduced.
    allocation = {
        "BTCUSD": AllocationEntry(pct=0.5, price=79000.0, stop_loss=81000.0, side="sell", reason="btc short"),
    }
    positions = [_position(symbol="ETHUSD", side="buy")]
    with patch(
        "ai.clerk_execution._fetch_correlation_closes",
        side_effect=_closes_by_symbol({"BTCUSD": _CORRELATED_A, "ETHUSD": _NEGATIVELY_CORRELATED_B}),
    ):
        result = _apply_correlation_guard(allocation, positions)

    assert result["BTCUSD"].pct == pytest.approx(0.5 * config.CLERK_CORRELATION_SIZE_FACTOR)  # trimmed -- this really is a stacked bet


# --- _apply_atr_stop_floor_guard (2026-09-16, real EURUSD 2026-09-07 incident) ---


def test_atr_stop_floor_guard_widens_a_too_tight_buy_stop():
    # Real EURUSD-shaped numbers: entry 1.16231, stop 1.1636 is on the
    # WRONG side for a buy (that's a sell-shaped stop) -- use a real buy
    # shape instead: H1 ATR 0.001 (0.06% of 1.16231, per the real
    # audit), floor = 1.5x = 0.0015, stop only 0.0004 away (way
    # tighter than the floor) -- must widen to exactly the floor below price.
    allocation = {
        "EURUSD": AllocationEntry(pct=0.5, price=1.16231, stop_loss=1.16191, side="buy", reason="thesis"),
    }
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", h1_atr=0.001), "")}
    result = _apply_atr_stop_floor_guard(allocation, [], cache)
    expected_stop = 1.16231 - 1.5 * 0.001
    assert result["EURUSD"].stop_loss == pytest.approx(expected_stop)
    assert "ATR stop-floor guard" in result["EURUSD"].reason


def test_atr_stop_floor_guard_widens_a_too_tight_sell_stop_on_the_correct_side():
    allocation = {
        "EURUSD": AllocationEntry(pct=0.5, price=1.16231, stop_loss=1.16271, side="sell", reason="thesis"),
    }
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", h1_atr=0.001), "")}
    result = _apply_atr_stop_floor_guard(allocation, [], cache)
    expected_stop = 1.16231 + 1.5 * 0.001
    assert result["EURUSD"].stop_loss == pytest.approx(expected_stop)
    assert result["EURUSD"].stop_loss > 1.16231  # correct side for a sell


def test_atr_stop_floor_guard_leaves_a_stop_already_at_the_floor_untouched():
    atr = 0.001
    price, stop = 1.16231, 1.16231 - 1.5 * atr
    allocation = {"EURUSD": AllocationEntry(pct=0.5, price=price, stop_loss=stop, side="buy", reason="thesis")}
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", h1_atr=atr), "")}
    result = _apply_atr_stop_floor_guard(allocation, [], cache)
    assert result["EURUSD"].stop_loss == pytest.approx(stop)
    assert result["EURUSD"].reason == "thesis"  # untouched, no guard note appended


def test_atr_stop_floor_guard_leaves_a_stop_wider_than_the_floor_untouched():
    atr = 0.000697
    price, stop = 1.16231, 1.16231 - 3.0 * atr  # already much wider than the 1.5x floor
    allocation = {"EURUSD": AllocationEntry(pct=0.5, price=price, stop_loss=stop, side="buy", reason="thesis")}
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", h1_atr=atr), "")}
    result = _apply_atr_stop_floor_guard(allocation, [], cache)
    assert result["EURUSD"].stop_loss == pytest.approx(stop)


def test_atr_stop_floor_guard_skips_symbol_missing_from_cache():
    allocation = {"EURUSD": AllocationEntry(pct=0.5, price=1.16231, stop_loss=1.16225, side="buy", reason="thesis")}
    result = _apply_atr_stop_floor_guard(allocation, [], {})
    assert result["EURUSD"].stop_loss == 1.16225


# --- Intraday decision-tier guards (2026-09-24; replaces the 2026-09-22 unconditional
# M5 entry refinement that produced the H1-stop/M5-entry MSFT loss) ---


def _m5_sine_wave_history(n_cycles: int = 8, period: int = 12) -> pd.DataFrame:
    """A clean, real oscillation on 5-minute bars — oscillates between ~95 and ~105, giving
    compute_chart_structure a real support band near 95 and a real resistance band near 105."""
    import math

    prices = [100.0 + 5.0 * math.sin(2 * math.pi * k / period) for k in range(n_cycles * period)]
    idx = pd.date_range("2026-01-01", periods=len(prices), freq="5min")
    close = pd.Series(prices, index=idx)
    pad = close * 0.0005
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]), "High": close + pad, "Low": close - pad,
            "Close": close, "Volume": [100.0] * len(close),
        },
        index=idx,
    )


def _intraday_analysis(symbol="TESTSYM", ask=100.0, bid=99.98, m5_atr=2.0, m5_history=None, **overrides):
    """A cached analysis carrying REAL M5 structure (from the sine history) and a chosen M5 ATR
    (None -> no M5 read at all, so the H1-fallback paths can be exercised)."""
    from ai.ftmo_suggest import _compute_divergence_for_history
    from analysis.chart_structure import compute_chart_structure
    from analysis.technical import compute_technical_stats
    from analysis.timeframe_profiles import M5_PROFILE

    hist = m5_history if m5_history is not None else _m5_sine_wave_history()
    m5_stats = compute_technical_stats(hist["Close"], history=hist)
    if m5_atr is None:
        m5_stats = _fake_technical_stats()
    else:
        m5_stats.atr = m5_atr
        m5_stats.atr_pct = (m5_atr / ask * 100) if ask else None
    fields = dict(
        base=AssetAnalysis(symbol=symbol, description=symbol, bid=bid, ask=ask, display_name=None),
        m5_stats=m5_stats,
        m5_structure=compute_chart_structure(hist, profile=M5_PROFILE),
        m5_divergence=_compute_divergence_for_history(hist),
    )
    fields.update(overrides)
    return _fake_ftmo_analysis(symbol=symbol, h1_atr=4.0, **fields)


def _buy(price=90.0, stop=86.0, tp=130.0, pct=1.0, **kw):
    return AllocationEntry(pct=pct, price=price, stop_loss=stop, take_profit=tp, side="buy", reason="thesis", **kw)


def test_stale_entry_reanchor_moves_a_far_entry_to_the_real_m5_zone():
    # Entry 90 vs live 100 = 5x the 2.0 M5 ATR (limit 3.5x) -> stale. Real M5 support band sits near 95.
    cache = {"TESTSYM": (_intraday_analysis(), "")}
    result = _apply_stale_entry_reanchor({"TESTSYM": _buy()}, [], cache)
    entry = result["TESTSYM"]
    assert 94.0 < entry.price < 96.0
    assert entry.take_profit == 130.0  # Claude's own target kept
    assert abs(entry.price - entry.stop_loss) == pytest.approx(4.0)  # Claude's own stop DISTANCE kept
    assert "Stale-entry re-anchor" in entry.reason


def test_stale_entry_reanchor_mirrors_for_a_sell():
    cache = {"TESTSYM": (_intraday_analysis(), "")}
    sell = AllocationEntry(pct=1.0, price=110.0, stop_loss=114.0, take_profit=70.0, side="sell", reason="thesis")
    entry = _apply_stale_entry_reanchor({"TESTSYM": sell}, [], cache)["TESTSYM"]
    assert 104.0 < entry.price < 106.0  # the real M5 resistance band
    assert entry.stop_loss - entry.price == pytest.approx(4.0)


def test_a_non_stale_entry_is_left_exactly_as_claude_proposed_it():
    # Entry 99 vs live 100 = 0.5x M5 ATR (< the 3.5x stale limit): untouched — this is the
    # whole point of the upgrade (Claude's H1/M5 numbers are not overridden).
    cache = {"TESTSYM": (_intraday_analysis(), "")}
    result = _apply_stale_entry_reanchor({"TESTSYM": _buy(price=99.0, stop=97.0, tp=104.0)}, [], cache)
    assert result["TESTSYM"].price == 99.0
    assert result["TESTSYM"].reason == "thesis"


def test_an_entry_whose_target_was_already_reached_is_rejected_not_chased():
    cache = {"TESTSYM": (_intraday_analysis(ask=100.0), "")}
    result = _apply_stale_entry_reanchor({"TESTSYM": _buy(price=97.0, stop=95.0, tp=99.5)}, [], cache)
    assert result["TESTSYM"].pct == 0.0
    assert "already reached the target" in result["TESTSYM"].reason


def test_stale_entry_reanchor_leaves_the_candidate_alone_when_no_real_m5_zone_exists():
    idx = pd.date_range("2026-01-01", periods=100, freq="5min")
    close = pd.Series([100.0] * 100, index=idx)
    flat = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": [100.0] * 100}, index=idx)
    cache = {"TESTSYM": (_intraday_analysis(m5_history=flat), "")}
    result = _apply_stale_entry_reanchor({"TESTSYM": _buy()}, [], cache)
    assert result["TESTSYM"].price == 90.0 and result["TESTSYM"].reason == "thesis"


def test_stale_entry_reanchor_skips_missing_atr_missing_target_missing_cache_and_held_symbols():
    no_atr = {"TESTSYM": (_intraday_analysis(m5_atr=None), "")}
    assert _apply_stale_entry_reanchor({"TESTSYM": _buy()}, [], no_atr)["TESTSYM"].price == 90.0
    cache = {"TESTSYM": (_intraday_analysis(), "")}
    assert _apply_stale_entry_reanchor({"TESTSYM": _buy(tp=None)}, [], cache)["TESTSYM"].price == 90.0
    assert _apply_stale_entry_reanchor({"TESTSYM": _buy()}, [], {})["TESTSYM"].price == 90.0
    held = [_position(symbol="TESTSYM", side="buy")]
    assert _apply_stale_entry_reanchor({"TESTSYM": _buy()}, held, cache)["TESTSYM"].price == 90.0


def _event(symbol_currency="USD", minutes_from_now=15, impact="High"):
    from data import economic_calendar as ec

    when = datetime.now(timezone.utc) + timedelta(minutes=minutes_from_now)
    return ec.CalendarEvent(title="FOMC Rate Decision", currency=symbol_currency, time_utc=when, impact=impact)


def test_event_blackout_zeroes_a_new_entry_inside_the_window_only():
    now = datetime.now(timezone.utc)
    inside = _apply_event_blackout_guard({"MSFT": _buy()}, [], now, [_event(minutes_from_now=15)])
    assert inside["MSFT"].pct == 0.0 and "Event blackout" in inside["MSFT"].reason
    outside = _apply_event_blackout_guard({"MSFT": _buy()}, [], now, [_event(minutes_from_now=90)])
    assert outside["MSFT"].pct == 1.0
    other_currency = _apply_event_blackout_guard({"LVMH": _buy()}, [], now, [_event(minutes_from_now=15)])
    assert other_currency["LVMH"].pct == 1.0  # EUR name, USD event


def test_event_blackout_fails_open_without_calendar_data_and_never_touches_held_symbols():
    now = datetime.now(timezone.utc)
    assert _apply_event_blackout_guard({"MSFT": _buy()}, [], now, [])["MSFT"].pct == 1.0
    assert _apply_event_blackout_guard({"MSFT": _buy()}, [], now, None)["MSFT"].pct == 1.0
    held = [_position(symbol="MSFT", side="buy")]
    assert _apply_event_blackout_guard({"MSFT": _buy()}, held, now, [_event()])["MSFT"].pct == 1.0


def test_event_blackout_can_be_disabled_by_config():
    with patch.object(config, "EVENT_BLACKOUT_ENABLED", False):
        assert _apply_event_blackout_guard(
            {"MSFT": _buy()}, [], datetime.now(timezone.utc), [_event()]
        )["MSFT"].pct == 1.0


def test_size_scalar_downsizes_only_in_an_elevated_volatility_tape():
    hot = _intraday_analysis(m5_atr=2.0, m5_atr_pct_median=1.0)  # atr% = 2.0 vs median 1.0 -> x0.5
    hot.m5_stats.atr_pct = 2.0
    calm = _intraday_analysis(m5_atr=2.0, m5_atr_pct_median=4.0)  # current below median -> never sized UP
    calm.m5_stats.atr_pct = 2.0
    now = datetime.now(timezone.utc)
    hot_result = _apply_intraday_size_scalar({"A": _buy()}, [], {"A": (hot, "")}, now, [])
    assert hot_result["A"].pct == pytest.approx(0.5)
    assert "Intraday size scalar" in hot_result["A"].reason
    calm_result = _apply_intraday_size_scalar({"A": _buy()}, [], {"A": (calm, "")}, now, [])
    assert calm_result["A"].pct == 1.0


def test_size_scalar_is_clamped_to_its_configured_floor_and_halves_into_an_event_runup():
    very_hot = _intraday_analysis(m5_atr=2.0, m5_atr_pct_median=0.2)
    very_hot.m5_stats.atr_pct = 2.0  # ratio 0.1 -> clamped to the 0.5 floor
    now = datetime.now(timezone.utc)
    assert _apply_intraday_size_scalar({"A": _buy()}, [], {"A": (very_hot, "")}, now, [])["A"].pct == pytest.approx(0.5)
    # An event 90 minutes out (inside the 2h run-up, outside the 30-min blackout) halves risk.
    plain = _intraday_analysis()
    result = _apply_intraday_size_scalar({"MSFT": _buy()}, [], {"MSFT": (plain, "")}, now, [_event(minutes_from_now=90)])
    assert result["MSFT"].pct == pytest.approx(0.5)


def test_resting_order_hysteresis_keeps_the_exact_resting_terms_for_small_drift():
    cache = {"TESTSYM": (_intraday_analysis(m5_atr=2.0), "")}
    settled = {"TESTSYM": {"state": "order_placed", "entry": {
        "side": "buy", "price": 95.0, "stop_loss": 91.0, "take_profit": 110.0, "pct": 1.0}}}
    drifted = _buy(price=95.4, stop=91.5, tp=110.3, pct=1.1)  # all within 0.5 x 2.0 = 1.0 and 20%
    entry = _stabilize_resting_orders({"TESTSYM": drifted}, [], cache, settled)["TESTSYM"]
    assert (entry.price, entry.stop_loss, entry.take_profit, entry.pct) == (95.0, 91.0, 110.0, 1.0)


def test_resting_order_hysteresis_lets_a_material_change_through():
    cache = {"TESTSYM": (_intraday_analysis(m5_atr=2.0), "")}
    settled = {"TESTSYM": {"state": "order_placed", "entry": {
        "side": "buy", "price": 95.0, "stop_loss": 91.0, "take_profit": 110.0, "pct": 1.0}}}
    moved = _buy(price=97.5, stop=93.5, tp=110.0)  # entry moved 2.5 > 1.0 tolerance
    assert _stabilize_resting_orders({"TESTSYM": moved}, [], cache, settled)["TESTSYM"].price == 97.5
    rejected = _buy(pct=0.0, price=95.0, stop=91.0, tp=110.0)  # a rejection (pct 0) is never frozen
    assert _stabilize_resting_orders({"TESTSYM": rejected}, [], cache, settled)["TESTSYM"].pct == 0.0
    filled = {"TESTSYM": {"state": "filled", "entry": settled["TESTSYM"]["entry"]}}
    assert _stabilize_resting_orders({"TESTSYM": _buy(price=95.4)}, [], cache, filled)["TESTSYM"].price == 95.4


def test_stale_pending_setup_orders_are_cancelled_after_the_intraday_age_ceiling():
    now = datetime.now(timezone.utc)
    old = _pending_order(symbol="XAUUSD", ticket=77)
    old.time_setup = now - timedelta(hours=5)
    fresh = _pending_order(symbol="BTCUSD", ticket=78)
    fresh.time_setup = now - timedelta(minutes=30)
    settled = {
        "XAUUSD": {"origin": "pending_setup", "state": "order_placed"},
        "BTCUSD": {"origin": "pending_setup", "state": "order_placed"},
    }
    plan = _append_stale_pending_setup_cancels([], [old, fresh], settled, now, 3.0)
    assert [(o.symbol, o.action, o.pending_tickets_to_cancel) for o in plan] == [("XAUUSD", "cancel", [77])]
    # Symbols the plan already covers, immediate-origin orders and filled setups are left alone.
    from risk.apply_suggestion import PlannedOrder

    covered = [PlannedOrder(symbol="XAUUSD", action="hold", side="buy", volume=1.0, order_type="none", price=None, stop_loss=None)]
    assert _append_stale_pending_setup_cancels(list(covered), [old], settled, now, 3.0) == covered
    immediate = {"XAUUSD": {"origin": "immediate", "state": "order_placed"}}
    assert _append_stale_pending_setup_cancels([], [old], immediate, now, 3.0) == []


def test_tactical_event_note_lists_only_relevant_upcoming_events():
    now = datetime.now(timezone.utc)
    assert _tactical_event_note("MSFT", now, []) is None
    note = _tactical_event_note("MSFT", now, [_event(minutes_from_now=45)])
    assert "FOMC Rate Decision" in note and "in 45 min" in note.replace("in 44 min", "in 45 min")
    assert _tactical_event_note("LVMH", now, [_event(minutes_from_now=45)]) is None  # EUR name, USD event


def test_atr_stop_floor_guard_skips_symbol_with_no_h1_atr():
    allocation = {"EURUSD": AllocationEntry(pct=0.5, price=1.16231, stop_loss=1.16225, side="buy", reason="thesis")}
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", h1_atr=None), "")}
    result = _apply_atr_stop_floor_guard(allocation, [], cache)
    assert result["EURUSD"].stop_loss == 1.16225


def test_atr_stop_floor_guard_skips_entry_missing_price_or_stop():
    allocation = {"EURUSD": AllocationEntry(pct=0.5, price=None, stop_loss=None, side="buy", reason="thesis")}
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", h1_atr=0.0007), "")}
    result = _apply_atr_stop_floor_guard(allocation, [], cache)
    assert result["EURUSD"].price is None
    assert result["EURUSD"].stop_loss is None


def test_atr_stop_floor_guard_never_touches_an_already_held_symbol():
    allocation = {"EURUSD": AllocationEntry(pct=0.5, price=1.16231, stop_loss=1.16225, side="buy", reason="thesis")}
    positions = [_position(symbol="EURUSD", side="buy")]
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", h1_atr=0.000697), "")}
    result = _apply_atr_stop_floor_guard(allocation, positions, cache)
    assert result["EURUSD"].stop_loss == 1.16225  # not a "candidate" -- already held


def test_atr_stop_floor_guard_ignores_cash():
    allocation = {"CASH": AllocationEntry(pct=99.0, side="buy", reason="")}
    result = _apply_atr_stop_floor_guard(allocation, [], {})
    assert result["CASH"].pct == 99.0


# --- _apply_reward_risk_floor_guard (2026-09-17, real INTC 1:1 incident) ---


def _fake_trade_cost(spread_pct_of_price=0.01, **overrides):
    defaults = dict(
        category="Equities I CFD", spread_pct_of_price=spread_pct_of_price,
        swap_long_pct_per_day=None, swap_short_pct_per_day=None, min_stop_distance_pct=0.0,
    )
    return TradeCost(**{**defaults, **overrides})


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_rejects_a_ratio_below_the_floor(mock_cost):
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=100.0, stop_loss=95.0, take_profit=105.0, side="buy", reason="thesis"
        )
    }
    result = _apply_reward_risk_floor_guard(allocation, [])
    assert result["EURUSD"].pct == 0.0
    assert result["EURUSD"].price == 100.0
    assert result["EURUSD"].stop_loss == 95.0
    assert result["EURUSD"].take_profit == 105.0
    assert "reward:risk floor guard" in result["EURUSD"].reason
    assert "1.00:1" in result["EURUSD"].reason


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_leaves_a_ratio_at_or_above_the_floor_untouched(mock_cost):
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    # risk=5, reward=9 -> 1.8:1, exactly at the default floor.
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=100.0, stop_loss=95.0, take_profit=109.0, side="buy", reason="thesis"
        )
    }
    result = _apply_reward_risk_floor_guard(allocation, [])
    assert result["EURUSD"].pct == 0.5
    assert result["EURUSD"].reason == "thesis"


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_replays_the_real_intc_incident(mock_cost):
    # Real 2026-09-17 numbers: entry 108.75, stop 104.31, target 113.19 —
    # risk and reward both 4.44, net ~1.0:1, well below the 1.8 floor.
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "INTC": AllocationEntry(
            pct=0.35, price=108.75, stop_loss=104.31, take_profit=113.19, side="buy",
            reason="U.S. government stake and Apple foundry talks validate the turnaround thesis",
        )
    }
    result = _apply_reward_risk_floor_guard(allocation, [])
    assert result["INTC"].pct == 0.0
    assert "1.00:1" in result["INTC"].reason


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_nets_real_spread_cost_against_reward(mock_cost):
    # risk=5, gross reward=10 (2.0:1 gross) but a 5%-of-price spread costs
    # 5.0 in price terms -> net reward=5, net R:R=1.0:1, below the floor.
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=5.0)
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=100.0, stop_loss=95.0, take_profit=110.0, side="buy", reason="thesis"
        )
    }
    result = _apply_reward_risk_floor_guard(allocation, [])
    assert result["EURUSD"].pct == 0.0
    assert "1.00:1" in result["EURUSD"].reason


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_works_for_sell_side_too(mock_cost):
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=100.0, stop_loss=105.0, take_profit=95.0, side="sell", reason="thesis"
        )
    }
    result = _apply_reward_risk_floor_guard(allocation, [])
    assert result["EURUSD"].pct == 0.0
    assert "1.00:1" in result["EURUSD"].reason


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_skips_symbol_missing_take_profit(mock_cost):
    allocation = {
        "EURUSD": AllocationEntry(pct=0.5, price=100.0, stop_loss=95.0, take_profit=None, side="buy", reason="")
    }
    result = _apply_reward_risk_floor_guard(allocation, [])
    assert result["EURUSD"].pct == 0.5
    mock_cost.assert_not_called()


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_skips_symbol_missing_stop_loss(mock_cost):
    allocation = {
        "EURUSD": AllocationEntry(pct=0.5, price=100.0, stop_loss=None, take_profit=105.0, side="buy", reason="")
    }
    result = _apply_reward_risk_floor_guard(allocation, [])
    assert result["EURUSD"].pct == 0.5
    mock_cost.assert_not_called()


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_skips_symbol_with_no_real_trade_cost(mock_cost):
    mock_cost.return_value = None
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=100.0, stop_loss=95.0, take_profit=100.5, side="buy", reason="thesis"
        )
    }
    result = _apply_reward_risk_floor_guard(allocation, [])
    assert result["EURUSD"].pct == 0.5


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_skips_zero_risk_distance(mock_cost):
    mock_cost.return_value = _fake_trade_cost()
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=100.0, stop_loss=100.0, take_profit=105.0, side="buy", reason="thesis"
        )
    }
    result = _apply_reward_risk_floor_guard(allocation, [])
    assert result["EURUSD"].pct == 0.5


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_never_touches_an_already_held_symbol(mock_cost):
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=100.0, stop_loss=95.0, take_profit=105.0, side="buy", reason="thesis"
        )
    }
    result = _apply_reward_risk_floor_guard(allocation, [_position(symbol="EURUSD")])
    assert result["EURUSD"].pct == 0.5
    mock_cost.assert_not_called()


def test_rr_floor_guard_ignores_cash():
    allocation = {"CASH": AllocationEntry(pct=99.0, side="buy", reason="")}
    result = _apply_reward_risk_floor_guard(allocation, [])
    assert result["CASH"].pct == 99.0


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_runs_after_stop_floor_widening(mock_cost):
    # The ATR stop-floor guard would widen a too-tight stop from 95 to 96
    # (1.5x a 2.667 ATR = 4 -> stop 96, tighter than the original 95-away
    # stop was wide already reversed here for clarity: proposed stop 99
    # is tighter than the 1.5x-ATR floor of 4, so it gets widened to 96).
    # After widening, risk becomes 4 and reward (105-100=5) gives 1.25:1
    # net — below the floor — proving the reward:risk guard must see the
    # WIDENED stop, not the original tighter one (which would have shown
    # risk=1, reward=5, a misleadingly generous 5:1).
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=100.0, stop_loss=99.0, take_profit=105.0, side="buy", reason="thesis"
        )
    }
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", h1_atr=4.0 / config.ENTRY_ATR_STOP_FLOOR_MULTIPLE), "")}
    allocation = _apply_atr_stop_floor_guard(allocation, [], cache)
    assert allocation["EURUSD"].stop_loss == 96.0  # widened from 99 to 96 (4 below entry)

    result = _apply_reward_risk_floor_guard(allocation, [])
    assert result["EURUSD"].pct == 0.0
    assert "1.25:1" in result["EURUSD"].reason


# --- _apply_reward_risk_floor_guard: win-rate-aware floor (2026-09-18, direct user request) ---
def _null_baselines():
    """A random-entry baseline (mean -0.05R, sd 1.2R) so a 0.3R / 100-trade fake setup is SUPPORTED (z ~ +2.9)."""
    from analysis.edge_stats import EdgeBaseline

    return dict(
        null_baseline_buy=EdgeBaseline("buy", -0.05, 1.2, 800, 27.0),
        null_baseline_sell=EdgeBaseline("sell", -0.05, 1.2, 800, 27.0),
    )




def _fake_rsi_backtest(win_rate_pct, trades=100, **overrides):
    defaults = dict(
        condition="oversold", threshold=30.0, trades=trades, wins=0, losses=0, timeouts=0,
        win_rate_pct=win_rate_pct, avg_r_multiple=0.3, stop_atr_multiple=1.5, target_atr_multiple=3.0,
        max_holding_bars=16,
    )
    return RSIReactionBacktest(**{**defaults, **overrides})


def _fake_sr_backtest(support_win_rate_pct=None, resistance_win_rate_pct=None, support_tests=100, resistance_tests=100, **overrides):
    defaults = dict(
        support_tests=support_tests, support_wins=0, support_losses=0, support_timeouts=0,
        support_win_rate_pct=support_win_rate_pct, support_avg_r_multiple=0.3,
        resistance_tests=resistance_tests, resistance_wins=0, resistance_losses=0, resistance_timeouts=0,
        resistance_win_rate_pct=resistance_win_rate_pct, resistance_avg_r_multiple=0.3,
        stop_atr_multiple=1.5, target_atr_multiple=3.0, max_holding_bars=16,
    )
    return SupportResistanceBacktest(**{**defaults, **overrides})


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_allows_a_lower_ratio_when_real_win_rate_is_high(mock_cost):
    # 45% win rate implies breakeven (1-0.45)/0.45 ~= 1.222, x1.3 margin
    # ~= 1.589 required -- net R:R of 1.65 passes here even though it
    # would have been REJECTED under the flat 1.8 default. This is the
    # actual "small dagger" property this part exists to demonstrate.
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    # risk=100, reward=165 -> gross/net R:R = 1.65:1 (no cost netted)
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=1000.0, stop_loss=900.0, take_profit=1165.0, side="buy", reason="thesis"
        )
    }
    intraday = IntradayBacktests(rsi_oversold_backtest=_fake_rsi_backtest(win_rate_pct=45.0), **_null_baselines())
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", intraday_backtests=intraday), "")}

    result = _apply_reward_risk_floor_guard(allocation, [], cache)

    assert result["EURUSD"].pct == 0.5
    assert result["EURUSD"].reason == "thesis"


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_never_raises_the_bar_above_the_flat_default_on_a_low_win_rate(mock_cost):
    # Real incident fixed 2026-09-21, direct user report: a genuine MSFT
    # double-bottom setup with its own cited 44% D1 win rate was rejected
    # here purely because this same symbol's UNRELATED M15 scalp win
    # rate read 23% — that M15 evidence is about a different, generic
    # RSI-reaction/support-touch strategy, not necessarily this
    # candidate's own real entry thesis. A 20% win rate implies breakeven
    # (1-0.2)/0.2 = 4.0, x1.3 margin = 5.2 — far above the flat 1.8
    # default — but the guard must now cap at the flat default rather
    # than raise the bar on a mismatched sample: a net R:R of 1.9 (above
    # the flat 1.8 default) must PASS, using the flat floor as its basis.
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=1000.0, stop_loss=900.0, take_profit=1190.0, side="buy", reason="thesis"
        )
    }
    intraday = IntradayBacktests(rsi_oversold_backtest=_fake_rsi_backtest(win_rate_pct=20.0))
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", intraday_backtests=intraday), "")}

    result = _apply_reward_risk_floor_guard(allocation, [], cache)

    assert result["EURUSD"].pct == 0.5
    assert result["EURUSD"].reason == "thesis"


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_ignores_a_thin_win_rate_sample(mock_cost):
    # High win rate, but far too few real trades to trust it -- falls
    # back to the flat 1.8 floor, so a 1.65:1 net ratio (which would have
    # passed under the win-rate-derived floor) is correctly rejected.
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=1000.0, stop_loss=900.0, take_profit=1165.0, side="buy", reason="thesis"
        )
    }
    intraday = IntradayBacktests(
        rsi_oversold_backtest=_fake_rsi_backtest(win_rate_pct=45.0, trades=config.MIN_RESOLVED_TRADES_FOR_WIN_RATE_FLOOR - 1)
    )
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", intraday_backtests=intraday), "")}

    result = _apply_reward_risk_floor_guard(allocation, [], cache)

    assert result["EURUSD"].pct == 0.0
    assert "flat" in result["EURUSD"].reason


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_ignores_a_100_percent_win_rate(mock_cost):
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    # net R:R 0.5:1 would pass at any real win-rate-derived floor near
    # 100%, but a "100% win rate" is untrustworthy by construction --
    # must fall back to the flat 1.8 floor and reject this.
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=1000.0, stop_loss=900.0, take_profit=1050.0, side="buy", reason="thesis"
        )
    }
    intraday = IntradayBacktests(rsi_oversold_backtest=_fake_rsi_backtest(win_rate_pct=100.0))
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", intraday_backtests=intraday), "")}

    result = _apply_reward_risk_floor_guard(allocation, [], cache)

    assert result["EURUSD"].pct == 0.0
    assert "flat" in result["EURUSD"].reason


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_uses_the_worse_of_rsi_and_sr_win_rates_when_both_qualify(mock_cost):
    # RSI reads 70% (breakeven*margin ~0.56, clamped to the 1.0 absolute
    # floor); S/R reads a worse 50% (breakeven*margin = 1.3) -- BOTH
    # still below the flat 1.8 default (so the 2026-09-21 "never raise
    # above flat" cap doesn't mask this case), but the worse (S/R, 50%)
    # reading must still govern over the better RSI-only reading: a net
    # R:R of 1.2 (which would PASS under the RSI-only 1.0 floor) must
    # still be rejected against the real 1.3 floor the worse reading
    # implies.
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=1000.0, stop_loss=900.0, take_profit=1120.0, side="buy", reason="thesis"
        )
    }
    intraday = IntradayBacktests(
        rsi_oversold_backtest=_fake_rsi_backtest(win_rate_pct=70.0),
        **_null_baselines(),
        support_resistance_backtest=_fake_sr_backtest(support_win_rate_pct=50.0),
    )
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", intraday_backtests=intraday), "")}

    result = _apply_reward_risk_floor_guard(allocation, [], cache)

    assert result["EURUSD"].pct == 0.0
    assert "50%" in result["EURUSD"].reason


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_falls_back_to_flat_floor_when_symbol_missing_from_cache(mock_cost):
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=1000.0, stop_loss=900.0, take_profit=1165.0, side="buy", reason="thesis"
        )
    }
    result = _apply_reward_risk_floor_guard(allocation, [], {})
    assert result["EURUSD"].pct == 0.0
    assert "flat" in result["EURUSD"].reason


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_never_goes_below_the_absolute_backstop(mock_cost):
    # An implausibly-high (but not exactly 100%) win rate of 95% implies
    # breakeven (1-0.95)/0.95 ~= 0.053, x1.3 ~= 0.068 -- the 1.0 absolute
    # backstop must govern instead, so a net R:R of 0.9 still gets
    # rejected even though it would pass the raw win-rate-derived number.
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "EURUSD": AllocationEntry(
            pct=0.5, price=1000.0, stop_loss=900.0, take_profit=1090.0, side="buy", reason="thesis"
        )
    }
    intraday = IntradayBacktests(rsi_oversold_backtest=_fake_rsi_backtest(win_rate_pct=95.0), **_null_baselines())
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", intraday_backtests=intraday), "")}

    result = _apply_reward_risk_floor_guard(allocation, [], cache)

    assert result["EURUSD"].pct == 0.0
    assert "1.00:1" in result["EURUSD"].reason  # the absolute backstop value itself


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_replays_the_real_intc_incident_with_no_win_rate_evidence(mock_cost):
    # Re-run of the real INTC regression case with an EMPTY cache entry
    # (no M15 data for INTC in this scenario) -- confirms this change
    # doesn't alter the already-verified real-world case when no
    # win-rate evidence is available.
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "INTC": AllocationEntry(
            pct=0.35, price=108.75, stop_loss=104.31, take_profit=113.19, side="buy",
            reason="U.S. government stake and Apple foundry talks validate the turnaround thesis",
        )
    }
    intraday = IntradayBacktests()  # empty -- no qualifying win rate
    cache = {"INTC": (_fake_ftmo_analysis(symbol="INTC", intraday_backtests=intraday), "")}

    result = _apply_reward_risk_floor_guard(allocation, [], cache)

    assert result["INTC"].pct == 0.0
    assert "1.00:1" in result["INTC"].reason
    assert "flat" in result["INTC"].reason


@patch("ai.clerk_execution.get_trade_economics")
def test_rr_floor_guard_replays_the_real_msft_incident_and_now_passes(mock_cost):
    # Real, live incident 2026-09-21, direct user report ("out of 20
    # symbols, not even one good trade" — traced to this exact guard):
    # a genuine MSFT double-bottom setup, backed by its own cited 44%
    # D1 win rate, netted 3.41:1 reward:risk — comfortably above the
    # flat 1.8:1 default — but was rejected here because this same
    # symbol's UNRELATED M15 scalp win rate happened to read 23%,
    # implying a 4.3:1+ floor that has nothing to do with this trade's
    # actual (D1 pattern-based, not M15-scalp-based) thesis. Must now
    # pass, using the flat default as its basis.
    mock_cost.return_value = _fake_trade_cost(spread_pct_of_price=0.0)
    allocation = {
        "MSFT": AllocationEntry(
            pct=1.2, price=494.065, stop_loss=489.515, take_profit=509.80, side="buy",
            reason="Liquidity-sweep-confirmed D1 double bottom, 44% win rate D1 backtest",
        )
    }
    intraday = IntradayBacktests(rsi_oversold_backtest=_fake_rsi_backtest(win_rate_pct=23.0))
    cache = {"MSFT": (_fake_ftmo_analysis(symbol="MSFT", intraday_backtests=intraday), "")}

    result = _apply_reward_risk_floor_guard(allocation, [], cache)

    assert result["MSFT"].pct == 1.2
    assert "Liquidity-sweep-confirmed" in result["MSFT"].reason


# --- Clerk news feed (2026-09-17: closes the gap where local Ollama
# models can't fulfil the verdict prompt's own "check for major news"
# instruction) -------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_shared_symbol_news_cache(tmp_path):
    # config.NEWS_FETCH_CACHE_FILE (added 2026-09-20, the cross-process
    # disk mirror of the in-memory cache) is isolated here too — without
    # this, a real fresh fetch in these tests writes into the real
    # production records/news_fetch_cache.json, and a stale real entry
    # there could silently short-circuit a test's own mocked fetch.
    symbol_news._symbol_news_cache.clear()
    with patch.object(config, "NEWS_FETCH_CACHE_FILE", str(tmp_path / "news_fetch_cache.json")):
        yield
    symbol_news._symbol_news_cache.clear()


# Real bug fixed 2026-09-20: _fetch_clerk_news_block used to resolve via
# data.underlying.resolve_yahoo_ticker (a PMEX-symbol keyword map) —
# confirmed live that 17 of 22 real symbols in this account's current mix
# silently got NO ticker at all under it. These tests now exercise the
# real fix: resolution via data.mt5_source.get_symbol_category + the
# shared, FTMO-native data.symbol_news.resolve_ftmo_yahoo_ticker.


def test_fetch_clerk_news_block_returns_real_headlines(tmp_path):
    with (
        patch.object(config, "SYMBOL_NEWS_DIR", str(tmp_path / "ftmo_symbol_news")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch("ai.clerk_execution.get_symbol_category", return_value="Metals CFD"),
        patch(
            "data.symbol_news.fetch_recent_news",
            return_value=[{"title": "Gold hits record high", "summary": "", "source": "", "published": ""}],
        ) as fetch,
    ):
        block = _fetch_clerk_news_block("XAUUSD", "Gold vs US Dollar")
    assert block == "- Gold hits record high"
    fetch.assert_called_once_with("XAUUSD=X", limit=config.NEWS_HEADLINES_PER_ASSET)


def test_fetch_clerk_news_block_empty_when_ticker_does_not_resolve_and_description_has_nothing_useful(tmp_path):
    # "Spot CFD" alone strips down to "" via derive_generic_search_name
    # (both words are known trading-type boilerplate) -- genuinely
    # nothing left to search on, so the real fetch is never even
    # attempted, unlike test_fetch_clerk_news_block_uses_the_generic_
    # description_fallback_when_category_is_unmapped below.
    with (
        patch.object(config, "SYMBOL_NEWS_DIR", str(tmp_path / "ftmo_symbol_news")),
        patch("ai.clerk_execution.get_symbol_category", return_value="Commodities"),
        patch("data.symbol_news.fetch_recent_news") as fetch,
    ):
        block = _fetch_clerk_news_block("UNKNOWN.c", "Spot CFD")
    assert block == ""
    fetch.assert_not_called()


def test_fetch_clerk_news_block_uses_the_generic_description_fallback_when_category_is_unmapped(tmp_path):
    # Real gap found 2026-09-20, direct user challenge ("how is it
    # possible the 3 food items and oil does not have news feed, you
    # have to do intelligent news searching"): a category with no known
    # ticker convention (e.g. Agriculture/Cash CFD) used to mean zero
    # news, forever, for that symbol. Now a real Google search is built
    # from MT5's own real description instead of giving up. "SUGAR.c" is
    # deliberately NOT one of the hardcoded commodity futures (unlike
    # COFFEE.c/COCOA.c/WHEAT.c/UKOIL.cash) so this exercises the GENERIC
    # fallback specifically, not the precise hardcoded-ticker path.
    with (
        patch.object(config, "SYMBOL_NEWS_DIR", str(tmp_path / "ftmo_symbol_news")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch("ai.clerk_execution.get_symbol_category", return_value="Agriculture"),
        patch("data.symbol_news.fetch_recent_news", return_value=[]),
        patch(
            "data.symbol_news.fetch_google_news",
            return_value=[{"title": "Sugar prices rally", "summary": "", "source": "Reuters", "published": ""}],
        ) as google,
    ):
        block = _fetch_clerk_news_block("SUGAR.c", "Sugar vs US Dollar, Spot CFD")
    assert block == "- Sugar prices rally"
    google.assert_called_once_with("Sugar", limit=config.NEWS_HEADLINES_PER_ASSET)


def test_fetch_clerk_news_block_empty_when_fetch_returns_nothing(tmp_path):
    with (
        patch.object(config, "SYMBOL_NEWS_DIR", str(tmp_path / "ftmo_symbol_news")),
        patch("ai.clerk_execution.get_symbol_category", return_value="Metals CFD"),
        patch("data.symbol_news.fetch_recent_news", return_value=[]),
        patch("data.symbol_news.fetch_google_news", return_value=[]),
    ):
        block = _fetch_clerk_news_block("XAUUSD", "Gold vs US Dollar")
    assert block == ""


def test_fetch_clerk_news_block_is_cached_within_the_configured_window(tmp_path):
    with (
        patch.object(config, "SYMBOL_NEWS_DIR", str(tmp_path / "ftmo_symbol_news")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch("ai.clerk_execution.get_symbol_category", return_value="Metals CFD"),
        patch(
            "data.symbol_news.fetch_recent_news",
            return_value=[{"title": "headline one", "summary": "", "source": "", "published": ""}],
        ) as fetch,
    ):
        first = _fetch_clerk_news_block("XAUUSD", "Gold vs US Dollar")
        second = _fetch_clerk_news_block("XAUUSD", "Gold vs US Dollar")
    assert first == second == "- headline one"
    fetch.assert_called_once()


def test_fetch_clerk_news_block_refetches_after_the_cache_window_expires(tmp_path):
    from datetime import timedelta

    with (
        patch.object(config, "SYMBOL_NEWS_DIR", str(tmp_path / "ftmo_symbol_news")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "obsidian_vault")),
        patch("ai.clerk_execution.get_symbol_category", return_value="Metals CFD"),
    ):
        with patch(
            "data.symbol_news.fetch_recent_news",
            return_value=[{"title": "old headline", "summary": "", "source": "", "published": ""}],
        ):
            _fetch_clerk_news_block("XAUUSD", "Gold vs US Dollar")
        stale_time = datetime.now(timezone.utc) - timedelta(minutes=config.SYMBOL_NEWS_CACHE_MINUTES + 1)
        symbol_news._symbol_news_cache["XAUUSD=X"] = (
            stale_time,
            [{"title": "old headline", "summary": "", "source": "", "published": ""}],
            config.NEWS_HEADLINES_PER_ASSET,
        )
        with patch(
            "data.symbol_news.fetch_recent_news",
            return_value=[{"title": "new headline", "summary": "", "source": "", "published": ""}],
        ) as fetch:
            refreshed = _fetch_clerk_news_block("XAUUSD", "Gold vs US Dollar")
    assert refreshed == "- new headline"
    fetch.assert_called_once()


def test_verdict_prompt_embeds_real_news_block_not_a_web_access_instruction():
    setup = PendingSetup(symbol="XAUUSD", side="buy", pct=1.0, trigger_condition="cond")
    prompt = _build_verdict_prompt(setup, "technical context here", 100000.0, "1 hour(s)", "- Gold hits record high")
    assert "- Gold hits record high" in prompt
    assert "live web access" not in prompt


def test_verdict_prompt_shows_honest_placeholder_when_no_news_found():
    setup = PendingSetup(symbol="XAUUSD", side="buy", pct=1.0, trigger_condition="cond")
    prompt = _build_verdict_prompt(setup, "technical context here", 100000.0, "1 hour(s)", "")
    assert "(no recent headlines found)" in prompt


def test_repair_closed_trade_note_rewrites_the_pnl_less_note_in_place(tmp_path):
    from ai import trade_journal as tj
    from ai.clerk_execution import _repair_closed_trade_note

    with (
        patch.object(config, "TRADE_JOURNAL_DIR", str(tmp_path / "j")),
        patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path / "v")),
    ):
        tj.record_proposals({"immediate_allocation": {"EURUSD": {"pct": 1.0, "side": "buy", "price": 1.1, "stop_loss": 1.08, "reason": "why"}}, "pending_setups": []})
        tj.record_closed("EURUSD", None, "manual_or_unknown")
        story = next(s for s in tj.list_all_stories() if s.symbol == "EURUSD")
        stamp = datetime.fromisoformat(story.events[-1].timestamp_utc).strftime("%Y-%m-%d_%H%M%S")
        trades_dir = tmp_path / "v" / "Trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        note = trades_dir / f"EURUSD {stamp} buy.md"
        note.write_text("could not be matched", encoding="utf-8")

        _repair_closed_trade_note(story, _closed_trade(profit=-7.5), None)
        text = note.read_text(encoding="utf-8")
        assert "Realized P&L: -7.50" in text and "could not be matched" not in text
        assert "why" in text  # thesis carried from the journal story when settlement has no entry


# --- ATR stop-floor guard on the M5 decision tier (2026-09-24) ---


def test_atr_stop_floor_guard_uses_the_m5_atr_multiple_when_available():
    # H1 ATR 4.0 would demand 6.0; the M5 ATR 0.5 x 2.0 demands only 1.0 — the M5 basis governs.
    cache = {"TESTSYM": (_intraday_analysis(m5_atr=0.5, ask=100.0, bid=100.0), "")}
    ok = AllocationEntry(pct=1.0, price=100.0, stop_loss=99.0, take_profit=103.0, side="buy", reason="t")
    assert _apply_atr_stop_floor_guard({"TESTSYM": ok}, [], cache)["TESTSYM"].stop_loss == 99.0
    tight = AllocationEntry(pct=1.0, price=100.0, stop_loss=99.7, take_profit=103.0, side="buy", reason="t")
    widened = _apply_atr_stop_floor_guard({"TESTSYM": tight}, [], cache)["TESTSYM"]
    assert widened.stop_loss == pytest.approx(99.0)
    assert "M5 ATR" in widened.reason


def test_atr_stop_floor_guard_also_enforces_the_spread_and_minimum_stop_pct_floors():
    # M5 ATR tiny (0.01): the ATR floor is 0.02. But spread 1.0 x4 = 4.0 wins.
    wide_spread = {"TESTSYM": (_intraday_analysis(m5_atr=0.01, ask=101.0, bid=100.0), "")}
    e = AllocationEntry(pct=1.0, price=101.0, stop_loss=100.5, take_profit=110.0, side="buy", reason="t")
    assert _apply_stale_entry_reanchor is not None  # (import sanity)
    out = _apply_atr_stop_floor_guard({"TESTSYM": e}, [], wide_spread)["TESTSYM"]
    assert 101.0 - out.stop_loss >= 4.0 - 1e-9
    # The pipeline's own 0.1% minimum is a floor too (would otherwise be an infeasible order).
    tiny_everything = {"TESTSYM": (_intraday_analysis(m5_atr=0.0001, ask=100.0, bid=100.0), "")}
    e2 = AllocationEntry(pct=1.0, price=100.0, stop_loss=99.99, take_profit=101.0, side="buy", reason="t")
    out2 = _apply_atr_stop_floor_guard({"TESTSYM": e2}, [], tiny_everything)["TESTSYM"]
    assert 100.0 - out2.stop_loss >= config.MIN_STOP_DISTANCE_PCT / 100 * 100.0


def test_atr_stop_floor_guard_falls_back_to_h1_atr_when_there_is_no_m5_read():
    cache = {"TESTSYM": (_intraday_analysis(m5_atr=None, ask=100.0, bid=100.0), "")}  # h1 atr 4.0 -> floor 6.0
    e = AllocationEntry(pct=1.0, price=100.0, stop_loss=98.0, take_profit=120.0, side="buy", reason="t")
    out = _apply_atr_stop_floor_guard({"TESTSYM": e}, [], cache)["TESTSYM"]
    assert out.stop_loss == pytest.approx(94.0)
    assert "H1 ATR" in out.reason


def test_fetch_technical_context_passes_the_calendar_events_and_time_into_the_formatter():
    asset = MarketAsset(symbol="EURUSD", description="Euro vs US Dollar", bid=1.1, ask=1.1005)
    events = [_event(symbol_currency="EUR", minutes_from_now=60)]
    with (
        patch("ai.clerk_execution.analyze_ftmo_asset_live", return_value=_fake_ftmo_analysis()),
        patch("ai.clerk_execution.economic_calendar.fetch_calendar_events", return_value=events),
        patch("ai.clerk_execution.format_ftmo_asset_context", return_value="ctx") as mock_format,
    ):
        _fetch_technical_context("EURUSD", {"EURUSD": asset}, 10_000.0)
    assert mock_format.call_args.kwargs["calendar_events"] == events
    assert mock_format.call_args.kwargs["now_utc"] is not None


def test_fetch_technical_context_survives_a_calendar_failure():
    asset = MarketAsset(symbol="EURUSD", description="Euro vs US Dollar", bid=1.1, ask=1.1005)
    with (
        patch("ai.clerk_execution.analyze_ftmo_asset_live", return_value=_fake_ftmo_analysis()),
        patch("ai.clerk_execution.economic_calendar.fetch_calendar_events", side_effect=RuntimeError("feed down")),
        patch("ai.clerk_execution.format_ftmo_asset_context", return_value="ctx") as mock_format,
    ):
        result = _fetch_technical_context("EURUSD", {"EURUSD": asset}, 10_000.0)
    assert result is not None and mock_format.call_args.kwargs["calendar_events"] == []


def test_guard_note_regex_extracts_only_the_guard_annotations():
    from ai.clerk_execution import _GUARD_NOTE_RE

    reason = (
        "thesis [Stale-entry re-anchor: moved 90 -> 95] more text [ATR stop-floor guard: widened 1 -> 2] "
        "[unrelated bracket] [Intraday size scalar x0.50: vol; event]"
    )
    assert _GUARD_NOTE_RE.findall(reason) == [
        "[Stale-entry re-anchor: moved 90 -> 95]",
        "[ATR stop-floor guard: widened 1 -> 2]",
        "[Intraday size scalar x0.50: vol; event]",
    ]


def test_broker_clock_offset_rounds_a_live_tick_offset_to_the_quarter_hour():
    from datetime import timedelta

    from ai.clerk_execution import _broker_clock_offset

    with patch("ai.clerk_execution.get_server_time_offset", return_value=timedelta(hours=2, minutes=59, seconds=59.97)):
        assert _broker_clock_offset(["BTCUSD"]) == timedelta(hours=3)


def test_broker_clock_offset_skips_a_stale_tick_and_falls_back_to_the_next_symbol_then_zero():
    from datetime import timedelta

    from ai.clerk_execution import _broker_clock_offset

    def fake(symbol):
        return {"MSFT": timedelta(days=-1, hours=16, minutes=4, seconds=53), "BTCUSD": timedelta(hours=3, seconds=-0.03)}.get(symbol)

    with patch("ai.clerk_execution.get_server_time_offset", side_effect=fake):
        assert _broker_clock_offset(["MSFT", "BTCUSD"]) == timedelta(hours=3)
        assert _broker_clock_offset(["MSFT"]) == timedelta(0)  # the closed market's stale tick is not trusted
        assert _broker_clock_offset(["NOPE"]) == timedelta(0)
    with patch("ai.clerk_execution.get_server_time_offset", side_effect=RuntimeError("no mt5")):
        assert _broker_clock_offset(["BTCUSD"]) == timedelta(0)


def test_pending_setup_age_uses_the_brokers_server_clock_not_naive_utc():
    # An order placed 2h ago carries a server-clock stamp 3h ahead of UTC, so naive-UTC age
    # would read as -1h (looks brand new). With the server-clock "now" it is 2h — under the
    # 3h ceiling, kept; the same order at 4h is cancelled.
    now_utc = datetime.now(timezone.utc)
    broker_now = now_utc + timedelta(hours=3)
    two_h = _pending_order(symbol="XAUUSD", ticket=1)
    two_h.time_setup = broker_now - timedelta(hours=2)
    four_h = _pending_order(symbol="BTCUSD", ticket=2)
    four_h.time_setup = broker_now - timedelta(hours=4)
    settled = {s: {"origin": "pending_setup", "state": "order_placed"} for s in ("XAUUSD", "BTCUSD")}
    plan = _append_stale_pending_setup_cancels([], [two_h, four_h], settled, broker_now, 3.0)
    assert [o.symbol for o in plan] == ["BTCUSD"]


def test_guard_note_regex_matches_the_real_lowercase_reward_risk_note():
    from ai.clerk_execution import _GUARD_NOTE_RE

    real = "claude [reward:risk floor guard: net R:R 0.99:1 is below the 1.80:1 floor (flat 1.8:1 default) - rejected, not downsized.]"
    assert len(_GUARD_NOTE_RE.findall(real)) == 1


def test_stale_threshold_default_is_the_fill_math_value_and_spares_a_deliberate_pullback():
    # 3.5 M5 ATRs (= 2.5 M15 ATRs at the measured 1.42 ratio): a deliberate 2-ATR pullback entry (entry 96
    # vs live 100, ATR 2.0 -> 2.0x) is left to Claude; a 5-ATR one (entry 90) is re-anchored.
    assert config.STALE_ENTRY_M5_ATR_MULTIPLE == 3.5
    cache = {"TESTSYM": (_intraday_analysis(m5_atr=2.0), "")}
    kept = _apply_stale_entry_reanchor({"TESTSYM": _buy(price=96.0, stop=92.0, tp=110.0)}, [], cache)
    assert kept["TESTSYM"].price == 96.0 and kept["TESTSYM"].reason == "thesis"
    moved = _apply_stale_entry_reanchor({"TESTSYM": _buy(price=90.0, stop=86.0, tp=130.0)}, [], cache)
    assert moved["TESTSYM"].price != 90.0


def test_weekly_close_note_only_on_a_friday_near_the_close_and_never_for_crypto():
    from ai.clerk_execution import _weekly_close_note

    friday_late = datetime(2026, 9, 25, 19, 30, tzinfo=timezone.utc)  # a Friday, 19:30 UTC
    with patch("ai.clerk_execution.get_symbol_category", return_value="Equities I CFD"):
        note = _weekly_close_note("MSFT", friday_late)
        assert note is not None and "weekend" in note
        assert _weekly_close_note("MSFT", datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)) is None  # too early
        assert _weekly_close_note("MSFT", datetime(2026, 9, 23, 19, 30, tzinfo=timezone.utc)) is None  # a Wednesday
    with patch("ai.clerk_execution.get_symbol_category", return_value="Crypto I"):
        assert _weekly_close_note("BTCUSD", friday_late) is None
    with patch("ai.clerk_execution.get_symbol_category", side_effect=RuntimeError("no mt5")):
        assert _weekly_close_note("MSFT", friday_late) is None


# --- Guard-rejection cooldown (2026-09-25): no place/cancel churn on a setup a guard keeps rejecting ---


def test_guard_rejection_latches_an_unfilled_symbol_and_the_cooldown_then_holds_it_at_zero():
    from ai.clerk_execution import _apply_guard_cooldown, _record_guard_rejections

    now = datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc)
    cooldowns: dict = {}
    before = {"SOLUSD": 0.8, "CASH": 99.2}
    after = {"SOLUSD": _buy(pct=0.0), "CASH": _buy(pct=99.2)}  # the R:R guard zeroed SOLUSD
    latched = _record_guard_rejections(before, after, [], cooldowns, now)
    assert latched == ["SOLUSD"] and "CASH" not in cooldowns
    assert datetime.fromisoformat(cooldowns["SOLUSD"]["until_utc"]) == now + timedelta(minutes=config.GUARD_REJECTION_COOLDOWN_MINUTES)

    # Ten minutes later the setup is proposed again at a "passing" ratio: still held at 0, no new order.
    later = now + timedelta(minutes=10)
    held = _apply_guard_cooldown({"SOLUSD": _buy(pct=0.8)}, [], cooldowns, later)
    assert held["SOLUSD"].pct == 0.0
    # Once the cooldown expires the symbol is evaluated normally again and the record is dropped.
    expired = _apply_guard_cooldown({"SOLUSD": _buy(pct=0.8)}, [], cooldowns, now + timedelta(minutes=61))
    assert expired["SOLUSD"].pct == 0.8 and "SOLUSD" not in cooldowns


def test_guard_rejection_never_latches_a_held_position_or_re_latches_an_active_cooldown():
    from ai.clerk_execution import _apply_guard_cooldown, _record_guard_rejections

    now = datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc)
    cooldowns: dict = {}
    zeroed = {"EURUSD": _buy(pct=0.0)}
    # EURUSD is HELD (a position exists): the guards never manage a held side, so nothing is latched.
    assert _record_guard_rejections({"EURUSD": 1.0}, zeroed, [_position("EURUSD")], cooldowns, now) == []
    assert cooldowns == {}
    # A symbol already cooling down is not re-latched (its clock is not extended every poll).
    _record_guard_rejections({"SOLUSD": 0.8}, {"SOLUSD": _buy(pct=0.0)}, [], cooldowns, now)
    first_until = cooldowns["SOLUSD"]["until_utc"]
    assert _record_guard_rejections({"SOLUSD": 0.8}, {"SOLUSD": _buy(pct=0.0)}, [], cooldowns, now + timedelta(minutes=20)) == []
    assert cooldowns["SOLUSD"]["until_utc"] == first_until
    # The cooldown never gates a held symbol's own management.
    kept = _apply_guard_cooldown({"SOLUSD": _buy(pct=0.8)}, [_position("SOLUSD")], cooldowns, now + timedelta(minutes=1))
    assert kept["SOLUSD"].pct == 0.8


def test_guard_rejection_ignores_a_symbol_that_was_already_zero_going_in():
    from ai.clerk_execution import _record_guard_rejections

    now = datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc)
    cooldowns: dict = {}
    assert _record_guard_rejections({"XAUUSD": 0.0}, {"XAUUSD": _buy(pct=0.0)}, [], cooldowns, now) == []
    assert cooldowns == {}


def test_a_corrupt_cooldown_record_is_dropped_not_fatal():
    from ai.clerk_execution import _apply_guard_cooldown

    now = datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc)
    cooldowns = {"SOLUSD": {"until_utc": "not a date"}, "NVDA": {}}
    result = _apply_guard_cooldown({"SOLUSD": _buy(pct=0.8)}, [], cooldowns, now)
    assert result["SOLUSD"].pct == 0.8 and cooldowns == {}


# --- Clerk improvements 2026-09-25: market-closed backoff, pre-close guard, re-anchor R:R margin ---


def _session_history(close_hour_utc=20, last_bar=None, days=4, offset_hours=3):
    """M5 bars for `days` trading days ending each day at close_hour_utc, stamped on a server clock that runs
    `offset_hours` ahead of UTC — the shape MT5 returns for a US equity CFD."""
    import pandas as pd

    frames = []
    for d in range(days, 0, -1):
        day = pd.Timestamp("2026-09-24") - pd.Timedelta(days=d - 1)
        start = day + pd.Timedelta(hours=13, minutes=30)
        end = day + pd.Timedelta(hours=close_hour_utc) - pd.Timedelta(minutes=5)
        if d == 1 and last_bar is not None:
            end = pd.Timestamp(last_bar)
        frames.append(pd.date_range(start, end, freq="5min"))
    idx = frames[0].append(frames[1:]) + pd.Timedelta(hours=offset_hours)
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)


def test_minutes_to_session_close_is_learned_from_where_the_last_sessions_ended():
    from ai.clerk_execution import _minutes_to_session_close

    history = _session_history(last_bar="2026-09-24 19:35")
    now = datetime(2026, 9, 24, 19, 40, tzinfo=timezone.utc)
    assert _minutes_to_session_close(history, now, timedelta(hours=3)) == pytest.approx(20.0, abs=1.0)
    early = datetime(2026, 9, 24, 17, 0, tzinfo=timezone.utc)
    assert _minutes_to_session_close(_session_history(last_bar="2026-09-24 16:55"), early, timedelta(hours=3)) == pytest.approx(180.0, abs=1.0)


def test_minutes_to_session_close_is_none_for_24h_markets_stale_data_and_thin_history():
    import pandas as pd
    from ai.clerk_execution import _minutes_to_session_close

    idx = pd.date_range("2026-09-21", periods=1500, freq="5min")
    continuous = pd.DataFrame({"Close": 1.0}, index=idx)
    now = datetime(2026, 9, 24, 19, 40, tzinfo=timezone.utc)
    assert _minutes_to_session_close(continuous, now, timedelta(0)) is None              # crypto-like: no daily close
    stale = _session_history(last_bar="2026-09-24 16:00")
    assert _minutes_to_session_close(stale, now, timedelta(hours=3)) is None             # not trading right now
    assert _minutes_to_session_close(continuous.head(10), now, timedelta(0)) is None
    assert _minutes_to_session_close(None, now, timedelta(0)) is None


def _pre_close(alloc, positions=(), pending=(), category="Equities I CFD", minutes=10.0):
    from ai.clerk_execution import _apply_pre_close_guard

    now = datetime(2026, 9, 24, 19, 50, tzinfo=timezone.utc)
    with (
        patch("ai.clerk_execution.get_symbol_category", return_value=category),
        patch("ai.clerk_execution._minutes_to_session_close", return_value=minutes),
    ):
        return _apply_pre_close_guard(alloc, list(positions), list(pending), now, timedelta(hours=3), history_fn=lambda s: None)


def test_pre_close_guard_blocks_only_a_new_order_close_to_the_session_end():
    blocked = _pre_close({"NVDA": _buy(pct=0.4), "CASH": _buy(pct=99.6)})
    assert blocked["NVDA"].pct == 0.0 and blocked["CASH"].pct == 99.6
    assert _pre_close({"NVDA": _buy(pct=0.4)}, minutes=45.0)["NVDA"].pct == 0.4            # plenty of session left
    assert _pre_close({"NVDA": _buy(pct=0.4)}, minutes=None)["NVDA"].pct == 0.4            # no estimate -> never blocks
    assert _pre_close({"BTCUSD": _buy(pct=0.4)}, category="Crypto I CFD")["BTCUSD"].pct == 0.4
    # An order already resting or a held position is never touched by this guard.
    assert _pre_close({"NVDA": _buy(pct=0.4)}, pending=[_pending_order(symbol="NVDA")])["NVDA"].pct == 0.4
    assert _pre_close({"NVDA": _buy(pct=0.4)}, positions=[_position("NVDA")])["NVDA"].pct == 0.4


def test_pre_close_guard_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "NO_NEW_ORDER_MINUTES_BEFORE_CLOSE", 0)
    assert _pre_close({"NVDA": _buy(pct=0.4)})["NVDA"].pct == 0.4


def test_market_closed_rejection_starts_a_backoff_and_a_success_clears_it():
    from ai.clerk_execution import _learn_order_outcome, _order_backoff_until

    now = datetime(2026, 9, 25, 1, 0, tzinfo=timezone.utc)
    backoff: dict = {}
    _learn_order_outcome(backoff, "NVDA", OrderResult(False, 10018, "Market closed", None), now)
    until = _order_backoff_until(backoff, "NVDA", now + timedelta(minutes=1))
    assert until == now + timedelta(minutes=config.MARKET_CLOSED_BACKOFF_MINUTES)
    # some other rejection does not start one
    _learn_order_outcome(backoff, "SOLUSD", OrderResult(False, 10016, "Invalid stops", None), now)
    assert "SOLUSD" not in backoff
    # expiry, then success clears
    assert _order_backoff_until(backoff, "NVDA", now + timedelta(minutes=31)) is None and "NVDA" not in backoff
    _learn_order_outcome(backoff, "NVDA", OrderResult(False, 10018, "market closed", None), now)
    _learn_order_outcome(backoff, "NVDA", OrderResult(True, 10009, "ok", 7), now)
    assert "NVDA" not in backoff
    backoff["X"] = {"until_utc": "garbage"}
    assert _order_backoff_until(backoff, "X", now) is None and "X" not in backoff


@patch("ai.clerk_execution.get_symbol_category")
@patch("ai.clerk_execution.is_pre_weekend_cleanup_due", return_value=True)
@patch("ai.clerk_execution.cancel_pending_order")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders")
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_a_market_closed_cancel_is_not_retried_every_poll(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_cancel, mock_due, mock_category, _fixed_files,
):
    # Real incident (2026-09-25): an NVDA cancel failed with "Market closed" 56 times in one night, once per poll.
    from data.mt5_source import AccountSummary, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_pending.return_value = [_pending_order(symbol="EURUSD", ticket=321)]
    mock_cancel.return_value = OrderResult(success=False, retcode=10018, comment="Market closed", ticket=None)
    mock_category.return_value = "Forex"

    from data.mt5_source import ContractSpec

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": {"pct": 1.0, "side": "buy", "price": 1.09, "stop_loss": 1.08, "take_profit": 1.11}})
        run_clerk_execution_check()
        assert mock_cancel.call_count == 1
        run_clerk_execution_check()
        run_clerk_execution_check()
    assert mock_cancel.call_count == 1  # the next polls defer instead of hammering the broker
    assert "order_backoff" in read_settlement() and "EURUSD" in read_settlement()["order_backoff"]


def test_a_re_anchored_entry_must_clear_the_reward_risk_floor_by_the_margin():
    from ai.clerk_execution import _apply_reward_risk_floor_guard

    def _run(reason):
        entry = AllocationEntry(pct=1.0, price=100.0, stop_loss=98.0, take_profit=104.02, side="buy", reason=reason)
        with patch("ai.clerk_execution.get_trade_economics", return_value=_fake_trade_cost(spread_pct_of_price=0.0)):
            return _apply_reward_risk_floor_guard({"X": entry}, [], None)["X"]

    # net R:R = 2.01: clears the flat 1.8 floor as-is...
    assert _run("thesis").pct == 1.0
    # ...but a re-anchored entry needs 1.8 + 0.3 = 2.1 and is rejected at 2.01.
    assert _run("thesis [Stale-entry re-anchor: moved to a zone]").pct == 0.0


# --- Entry-mode guard and execution wiring (2026-09-25) --------------------------------------------------

def _mode_analysis(symbol="EURUSD", atr=0.20, min_stop_pct=0.0):
    trade_cost = _make_trade_cost_for_mode(min_stop_pct)
    return _fake_ftmo_analysis(symbol=symbol, m5_stats=_fake_technical_stats(atr=atr), trade_cost=trade_cost)


def _make_trade_cost_for_mode(min_stop_pct):
    return TradeCost(
        category="Forex", spread_pct_of_price=0.002, swap_long_pct_per_day=0.0, swap_short_pct_per_day=0.0,
        min_stop_distance_pct=min_stop_pct,
    )


def _mode_entry(mode, price=100.0, stop=99.2, tp=101.6, side="buy", pct=1.0):
    return AllocationEntry(pct=pct, price=price, stop_loss=stop, take_profit=tp, side=side, reason="thesis", entry_mode=mode)


def _run_mode_guard(entry, pending=None, used=None, quote=(100.00, 100.02), analysis=None, symbol="EURUSD"):
    from ai.clerk_execution import _apply_entry_mode_guard

    cache = {symbol: (analysis or _mode_analysis(symbol), "")}
    prices = {symbol: MarketAsset(symbol=symbol, description=symbol, bid=quote[0], ask=quote[1])}
    return _apply_entry_mode_guard({symbol: entry}, [], pending or [], cache, prices, used or set())


def test_entry_mode_guard_leaves_limit_entries_completely_untouched():
    entry = _mode_entry("limit", price=99.5)
    result, caps = _run_mode_guard(entry)
    assert result["EURUSD"] is entry and caps == {}


def test_entry_mode_guard_reprices_a_valid_market_entry_to_the_live_ask_and_caps_the_send_deviation():
    result, caps = _run_mode_guard(_mode_entry("market", price=100.0))
    e = result["EURUSD"]
    assert e.entry_mode == "market" and e.price == 100.02 and e.pct == 1.0
    assert "[Entry-mode guard: entry_mode market: market entry at 100.02" in e.reason
    assert caps == {"EURUSD": pytest.approx(config.MARKET_ENTRY_SEND_DEVIATION_ATR * 0.20)}


def test_entry_mode_guard_downgrades_a_chased_market_entry_to_the_original_limit():
    result, caps = _run_mode_guard(_mode_entry("market", price=99.5))  # ask 100.02 is 2.6 ATR worse
    e = result["EURUSD"]
    assert e.entry_mode == "limit" and e.price == 99.5 and e.pct == 1.0 and "not chasing" in e.reason
    assert caps == {}


def test_entry_mode_guard_rejects_a_dead_setup_instead_of_sending_it():
    result, _ = _run_mode_guard(_mode_entry("market", price=100.0, stop=100.10))  # already through the stop
    e = result["EURUSD"]
    assert e.pct == 0.0 and e.entry_mode == "limit" and "through the stop" in e.reason


def test_entry_mode_guard_blocks_a_second_market_entry_for_the_same_symbol_today():
    result, caps = _run_mode_guard(_mode_entry("market", price=100.0), used={"EURUSD"})
    assert result["EURUSD"].entry_mode == "limit" and "already used" in result["EURUSD"].reason and caps == {}


def test_entry_mode_guard_places_a_breakout_stop_and_rejects_an_extended_or_far_one():
    ok, _ = _run_mode_guard(_mode_entry("stop", price=100.20))
    assert ok["EURUSD"].entry_mode == "stop" and ok["EURUSD"].price == 100.20
    extended, _ = _run_mode_guard(_mode_entry("stop", price=99.70))
    assert extended["EURUSD"].pct == 0.0 and "extended" in extended["EURUSD"].reason
    far, _ = _run_mode_guard(_mode_entry("stop", price=100.90))
    assert far["EURUSD"].pct == 0.0 and "too far" in far["EURUSD"].reason


def test_entry_mode_guard_does_not_renudge_a_stop_order_that_already_rests():
    resting = PendingOrder(symbol="EURUSD", volume=1.0, order_type="buy stop", price_open=100.20, sl=99.2, tp=101.6, ticket=9, time_setup=None)
    entry = _mode_entry("stop", price=100.21)
    result, _ = _run_mode_guard(entry, pending=[resting])
    assert result["EURUSD"] is entry  # untouched: the resting order keeps its own terms


def test_entry_mode_guard_falls_back_to_limit_without_a_quote_or_analysis():
    from ai.clerk_execution import _apply_entry_mode_guard

    entry = _mode_entry("market")
    result, caps = _apply_entry_mode_guard({"EURUSD": entry}, [], [], {}, {}, set())
    assert result["EURUSD"].entry_mode == "limit" and result["EURUSD"].pct == 1.0 and caps == {}


def test_entry_mode_guard_kill_switch(monkeypatch):
    monkeypatch.setattr(config, "NEW_ENTRY_KINDS_ENABLED", False)
    result, caps = _run_mode_guard(_mode_entry("market", price=100.0))
    assert result["EURUSD"].entry_mode == "limit" and "kill switch" in result["EURUSD"].reason and caps == {}


def test_entry_mode_guard_never_touches_a_held_symbol():
    from ai.clerk_execution import _apply_entry_mode_guard

    held = Position(symbol="EURUSD", volume=1.0, side="buy", price_open=100.0, price_current=100.0, sl=99.0, profit=0.0,
                    opened_at=datetime.now(), ticket=1)
    entry = _mode_entry("market")
    result, _ = _apply_entry_mode_guard({"EURUSD": entry}, [held], [], {}, {}, set())
    assert result["EURUSD"] is entry


def test_stale_reanchor_ignores_verified_stop_and_market_entries():
    from ai.clerk_execution import _apply_stale_entry_reanchor

    analysis = _mode_analysis()
    analysis.base.ask = analysis.base.bid = 100.0
    # A market/stop entry far from "market" would look stale to the limit logic; it must be left alone.
    entry = _mode_entry("stop", price=90.0, stop=85.0, tp=120.0)
    result = _apply_stale_entry_reanchor({"EURUSD": entry}, [], {"EURUSD": (analysis, "")})
    assert result["EURUSD"] is entry


def test_stabilizer_never_smooths_a_change_of_order_kind():
    from ai.clerk_execution import _stabilize_resting_orders

    analysis = _mode_analysis()
    entry = _mode_entry("market", price=100.02)
    settled = {"EURUSD": {"state": "order_placed", "entry": {"side": "buy", "price": 100.0, "stop_loss": 99.2, "take_profit": 101.6, "pct": 1.0, "entry_mode": "limit"}}}
    result = _stabilize_resting_orders({"EURUSD": entry}, [], {"EURUSD": (analysis, "")}, settled)
    assert result["EURUSD"].price == 100.02  # not reset to the resting limit's 100.0


def test_with_changes_preserves_the_entry_mode():
    from ai.clerk_execution import _with_changes

    assert _with_changes(_mode_entry("stop"), pct=0.5).entry_mode == "stop"


@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_market_entry_reaches_open_position_with_its_kind_and_cap_and_is_recorded_once_per_day(
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
    mock_open.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=777)
    analysis = _fake_ftmo_analysis(
        symbol="EURUSD", m5_stats=_fake_technical_stats(atr=0.0004),
        base=AssetAnalysis(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900, display_name=None),
    )
    with (
        patch("ai.clerk_execution.get_contract_spec") as mock_spec,
        patch("ai.clerk_execution.analyze_ftmo_asset_live", return_value=analysis),
        patch("ai.clerk_execution.format_ftmo_asset_context", return_value="ctx"),
    ):
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            immediate_allocation={"EURUSD": {"pct": 0.5, "side": "buy", "price": 1.0899, "stop_loss": 1.0850,
                                              "take_profit": 1.0990, "entry_mode": "market"}},
        )
        run_clerk_execution_check()

    mock_open.assert_called_once()
    call = mock_open.call_args
    assert call.kwargs["kind"] == "market"
    assert call.kwargs["max_deviation_price"] == pytest.approx(config.MARKET_ENTRY_SEND_DEVIATION_ATR * 0.0004)
    assert call.args[3] == 1.0900  # the live ask, not the planned 1.0899
    settlement = json.loads(Path(config.CLERK_EXECUTION_SETTLEMENT_FILE).read_text())
    assert list(settlement["market_entries"]) == ["EURUSD"]
    assert settlement["settled"]["EURUSD"]["entry"]["entry_mode"] == "market"


# --- resting-order expiry before the session close (2026-09-25) --------------------------------------------------

def test_resting_order_expiry_is_the_learned_session_close_minus_the_margin(monkeypatch):
    import pandas as pd
    from ai.clerk_execution import _resting_order_expiry_hours

    monkeypatch.setattr(config, "RESTING_ORDER_EXPIRY_BEFORE_CLOSE_MINUTES", 5.0)
    now = datetime(2026, 9, 25, 14, 0, tzinfo=timezone.utc)
    with patch("ai.clerk_execution._minutes_to_session_close", return_value=125.0):
        assert _resting_order_expiry_hours("NVDA", now, timedelta(hours=3), history_fn=lambda s: pd.DataFrame()) == pytest.approx(2.0)
    with patch("ai.clerk_execution._minutes_to_session_close", return_value=None):  # a 24h market: plain GTC
        assert _resting_order_expiry_hours("BTCUSD", now, timedelta(hours=3), history_fn=lambda s: pd.DataFrame()) is None
    with patch("ai.clerk_execution._minutes_to_session_close", return_value=12.0):  # too close to be useful
        assert _resting_order_expiry_hours("NVDA", now, timedelta(hours=3), history_fn=lambda s: pd.DataFrame()) is None


def test_resting_order_expiry_degrades_to_none_on_any_failure_and_can_be_switched_off(monkeypatch):
    from ai.clerk_execution import _resting_order_expiry_hours

    now = datetime(2026, 9, 25, 14, 0, tzinfo=timezone.utc)

    def boom(symbol):
        raise RuntimeError("terminal down")

    monkeypatch.setattr(config, "RESTING_ORDER_EXPIRY_BEFORE_CLOSE_MINUTES", 5.0)
    assert _resting_order_expiry_hours("NVDA", now, timedelta(hours=3), history_fn=boom) is None
    monkeypatch.setattr(config, "RESTING_ORDER_EXPIRY_BEFORE_CLOSE_MINUTES", -1.0)
    with patch("ai.clerk_execution._minutes_to_session_close", return_value=300.0):
        assert _resting_order_expiry_hours("NVDA", now, timedelta(hours=3), history_fn=lambda s: None) is None


def test_guard_note_extraction_cooldown_reason_and_the_entry_mode_exemption():
    from ai.clerk_execution import _guard_blocks_this_poll, _last_guard_note, _record_guard_rejections

    reason = "thesis [Live re-check: ok] [Entry-mode guard: breakout 0.99 M5 ATR beyond the trigger (cap 0.6) - extended, not chasing - rejected.]"
    assert _last_guard_note(reason).startswith("Entry-mode guard: breakout 0.99")
    assert _last_guard_note("plain thesis [Live re-check: verified]") is None and _last_guard_note(None) is None

    now = datetime(2026, 9, 25, 14, 0, tzinfo=timezone.utc)
    before = {"XAGUSD": 0.45, "META": 0.4, "CASH": 99.0}
    after = {
        "XAGUSD": AllocationEntry(pct=0.0, price=63.8, stop_loss=64.5, side="sell", reason=reason),
        "META": AllocationEntry(pct=0.0, price=752.0, stop_loss=742.0, side="buy", reason="thesis [Stale-entry guard: target reached - rejected.]"),
        "CASH": AllocationEntry(pct=99.0),
    }
    cooldowns: dict = {}
    latched = _record_guard_rejections(before, after, [], cooldowns, now, skip={"XAGUSD"})
    assert latched == ["META"] and "XAGUSD" not in cooldowns  # a transient entry-mode rejection is re-judged every poll
    assert cooldowns["META"]["reason"].startswith("Stale-entry guard")
    blocks = _guard_blocks_this_poll(before, after, [], cooldowns)
    assert set(blocks) == {"XAGUSD", "META"} and "extended" in blocks["XAGUSD"]
    # a cooldown-held symbol (no fresh guard note) reports the cooldown and its original reason
    held_back = {"META": AllocationEntry(pct=0.0, price=752.0, stop_loss=742.0, side="buy", reason="thesis [Live re-check: ok]")}
    text = _guard_blocks_this_poll({"META": 0.4}, held_back, [], cooldowns)["META"]
    assert text.startswith("cooldown until") and "Stale-entry guard" in text


@patch("ai.clerk_execution.is_symbol_tradable_now", return_value=True)
@patch("ai.clerk_execution._run_clerk_verdict")
@patch("ai.clerk_execution._fetch_technical_context")
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_the_status_line_separates_the_quick_gates_from_the_llm_lane(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_verdict, mock_tradable, _fixed_files,
):
    import pandas as pd
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=2004.0, ask=2004.3)]
    bars = pd.DataFrame([(1999.0, 1997.0, 1998.0), (2000.4, 1998.0, 1999.8)], columns=["High", "Low", "Close"])
    mock_fetch_ctx.return_value = (
        _fake_ftmo_analysis(symbol="XAUUSD", m5_stats=_fake_technical_stats(atr=2.0), m5_recent=bars), "ctx"
    )
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0,
            trade_contract_size=100.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            pending_setups=[
                {"symbol": "XAUUSD", "side": "buy", "pct": 1.0, "trigger_condition": "M5 closes above 2000",
                 "price": 2000.0, "stop_loss": 1990.0, "take_profit": 2020.0, "reason": "r",
                 "trigger": {"kind": "range_break", "level": 2000.0}}
            ],
        )
        run_clerk_execution_check()
    mock_verdict.assert_not_called()
    detail = read_execution_state()["last_detail"]
    assert detail.startswith("0 order(s) sent | quick gates: XAUUSD trigger not fired")
    assert "extended, not chasing" in detail  # 2004.3 is 2.15 ATR past the level
    assert "LLM: nothing needed a model this poll" in detail


def _xag_like(median=1.0):
    from data.mt5_source import ContractSpec

    analysis = _intraday_analysis(m5_atr=2.0, m5_atr_pct_median=median)
    analysis.m5_stats.atr_pct = 1.7  # ratio 0.59
    analysis.base.contract_spec = ContractSpec(
        volume_min=0.01, volume_step=0.01, volume_max=100.0, trade_contract_size=5000.0,
        currency_margin="USD", margin_initial=6405.4, money_per_price_unit=5000.0,
    )
    return analysis


def _xag_entry(pct=0.45):
    return AllocationEntry(pct=pct, price=63.818, stop_loss=64.547, take_profit=62.36, side="sell", reason="thesis")


def test_size_scalar_never_shrinks_an_entry_below_the_minimum_lot_when_the_ceiling_allows_it():
    now = datetime.now(timezone.utc)
    analysis = _xag_like()
    # $10k account: 0.01 lot of silver risks 0.729 x $50 = $36.45 = 0.3645%; the scaled 0.45 x 0.59 = 0.265% buys nothing.
    result = _apply_intraday_size_scalar({"XAGUSD": _xag_entry()}, [], {"XAGUSD": (analysis, "")}, now, [], account_equity=9990.33)
    assert result["XAGUSD"].pct == pytest.approx(0.729 * 50 / 9990.33 * 100 * 1.02, rel=1e-6)
    assert "raised to the minimum-lot risk" in result["XAGUSD"].reason
    # a ceiling below even the minimum lot stays as scaled (infeasible downstream, exactly as before)
    tiny = _apply_intraday_size_scalar({"XAGUSD": _xag_entry(pct=0.30)}, [], {"XAGUSD": (analysis, "")}, now, [], account_equity=9990.33)
    assert tiny["XAGUSD"].pct == pytest.approx(0.30 * 0.59, rel=0.02) and "raised" not in tiny["XAGUSD"].reason
    # no equity given / switched off -> the plain scalar
    assert _apply_intraday_size_scalar({"XAGUSD": _xag_entry()}, [], {"XAGUSD": (analysis, "")}, now, [])["XAGUSD"].pct < 0.3
    with patch.object(config, "SIZE_SCALAR_MIN_LOT_FLOOR", False):
        assert _apply_intraday_size_scalar({"XAGUSD": _xag_entry()}, [], {"XAGUSD": (analysis, "")}, now, [], account_equity=9990.33)["XAGUSD"].pct < 0.3
    # a bigger account needs no floor
    big = _apply_intraday_size_scalar({"XAGUSD": _xag_entry()}, [], {"XAGUSD": (analysis, "")}, now, [], account_equity=100000.0)
    assert big["XAGUSD"].pct == pytest.approx(0.45 * 0.59, rel=0.02) and "raised" not in big["XAGUSD"].reason


def test_correlation_guard_only_acts_on_near_duplicates_by_default():
    assert config.CLERK_CORRELATION_GUARD_THRESHOLD == 0.9 and config.CLERK_CORRELATION_SIZE_FACTOR == 0.75
    allocation = {
        "BTCUSD": AllocationEntry(pct=0.5, price=79000.0, stop_loss=77600.0, side="buy", reason="btc"),
        "ETHUSD": AllocationEntry(pct=0.3, price=2480.0, stop_loss=2440.0, side="buy", reason="eth"),
    }
    with patch("ai.clerk_execution._fetch_correlation_closes", side_effect=_closes_by_symbol({"BTCUSD": _CORRELATED_A, "ETHUSD": _CORRELATED_B})):
        with patch.object(config, "CLERK_CORRELATION_GUARD_THRESHOLD", 1.01):  # switched off
            result = _apply_correlation_guard(allocation, [])
    assert result["ETHUSD"].pct == pytest.approx(0.3)


def test_pending_partial_symbols_retries_until_the_volume_drops_then_gives_up_after_three_attempts():
    from ai.clerk_execution import _MAX_PARTIAL_ATTEMPTS, _pending_partial_symbols

    def rec(attempts=0, from_volume=0.14):
        return {"tactical": {"partial_pending_from_volume": from_volume, "partial_attempts": attempts,
                             "persisted_pct": 0.3, "persisted_stop_loss": 120.9, "persisted_take_profit": 126.0}}

    full = AllocationEntry(pct=0.4, price=122.0, stop_loss=119.4, take_profit=126.6, side="buy", reason="r")
    settled = {"SOLUSD": rec()}
    merged = {"SOLUSD": full}
    still_held = {"SOLUSD": _position(symbol="SOLUSD", side="buy")}
    still_held["SOLUSD"].volume = 0.14
    assert _pending_partial_symbols(settled, still_held, merged) == {"SOLUSD"}
    assert settled["SOLUSD"]["tactical"]["partial_attempts"] == 1
    # the retry re-pins the already-reduced risk and stop so a full-size carried baseline cannot undo the partial
    assert merged["SOLUSD"].pct == 0.3 and merged["SOLUSD"].stop_loss == 120.9
    # it went through: volume dropped -> flag cleared, no longer pending
    reduced = {"SOLUSD": _position(symbol="SOLUSD", side="buy")}
    reduced["SOLUSD"].volume = 0.11
    assert _pending_partial_symbols(settled, reduced, merged) == set()
    assert settled["SOLUSD"]["tactical"]["partial_pending_from_volume"] is None
    # never went through: gives up after the cap
    stuck = {"SOLUSD": rec(attempts=_MAX_PARTIAL_ATTEMPTS)}
    assert _pending_partial_symbols(stuck, still_held, {"SOLUSD": full}) == set()
    assert stuck["SOLUSD"]["tactical"]["partial_pending_from_volume"] is None
    # nothing recorded / no position: nothing pending
    assert _pending_partial_symbols({"SOLUSD": {"tactical": {}}}, still_held, {}) == set()
    assert _pending_partial_symbols(settled, {}, {}) == set()


def test_a_pending_setup_origin_position_keeps_its_tactical_stop_and_reduced_size_in_the_baseline():
    settled = {"XAUUSD": {
        "state": "filled", "origin": "pending_setup",
        "entry": {"pct": 0.4, "price": 2000.0, "stop_loss": 1990.0, "take_profit": 2030.0, "side": "buy", "reason": "r"},
        "tactical": {"persisted_pct": 0.1, "persisted_stop_loss": 2004.0, "persisted_take_profit": 2030.0},
    }}
    carried = _build_carried_forward_allocation({}, settled, held_symbols=frozenset({"XAUUSD"}))
    assert (carried["XAUUSD"].pct, carried["XAUUSD"].stop_loss) == (0.1, 2004.0)
    settled["XAUUSD"]["origin"] = "immediate"  # the same record read through the second (non-immediate) loop
    assert _build_carried_forward_allocation({}, settled, held_symbols=frozenset({"XAUUSD"}))["XAUUSD"].stop_loss == 2004.0


def test_invalidation_already_true_reads_the_mechanical_line_and_the_side():
    from ai.clerk_execution import invalidation_already_true

    cond = "M5 closes above 1.1397"
    assert "already above the invalidation level 1.1397" in invalidation_already_true("sell", cond, 1.14, 1.1401)  # the EURUSD case
    assert invalidation_already_true("sell", cond, 1.1385, 1.1386) is None
    assert invalidation_already_true("buy", cond, 1.14, 1.1401) is None  # points the other way for a buy: never blocks
    assert "already below" in invalidation_already_true("buy", "M5 closes below 745.54", 740.0, 740.2)
    assert invalidation_already_true("buy", "H1 closes back below $1,250.5", 1200.0, 1200.4) is not None
    assert invalidation_already_true("sell", "trend flips and RSI turns", 1.14, 1.1401) is None  # unparseable: no block
    assert invalidation_already_true("sell", None, 1.14, 1.1401) is None and invalidation_already_true("sell", cond, None, 1.14) is None


def test_invalidation_guard_zeroes_only_new_entries_that_are_already_dead_and_never_a_held_position():
    from ai.clerk_execution import _apply_invalidation_guard
    from data.mt5_source import MarketAsset

    dead = AllocationEntry(pct=0.25, price=1.1386, stop_loss=1.1405, take_profit=1.1347, side="sell", reason="thesis", invalidation_condition="M5 closes above 1.1397")
    alive = AllocationEntry(pct=0.25, price=1.1386, stop_loss=1.1405, take_profit=1.1347, side="sell", reason="thesis", invalidation_condition="M5 closes above 1.1450")
    prices = {"EURUSD": MarketAsset("EURUSD", "e", 1.14, 1.1401), "GBPUSD": MarketAsset("GBPUSD", "g", 1.14, 1.1401)}
    result = _apply_invalidation_guard({"EURUSD": dead, "GBPUSD": alive}, [], prices)
    assert result["EURUSD"].pct == 0.0 and "Invalidation guard" in result["EURUSD"].reason and result["GBPUSD"].pct == 0.25
    held = _apply_invalidation_guard({"EURUSD": dead}, [_position(symbol="EURUSD", side="sell")], prices)
    assert held["EURUSD"].pct == 0.25  # an open position is the invalidation check's job, not this guard's


def test_a_marketable_limit_is_resolved_as_a_market_entry_instead_of_failing_invalid_price_forever():
    # buy limit 100.10 while the ask is 100.02: it cannot rest (the EURUSD "Invalid price" loop) and its price is BETTER than planned
    buy, caps = _run_mode_guard(_mode_entry("limit", price=100.10))
    e = buy["EURUSD"]
    assert e.entry_mode == "market" and e.price == 100.02 and e.pct == 1.0 and "already marketable" not in e.reason
    assert "market entry at 100.02" in e.reason and "EURUSD" in caps
    # a sell limit at 99.90 with the bid at 100.00 is the mirror
    sell, sell_caps = _run_mode_guard(_mode_entry("limit", price=99.90, stop=100.80, tp=98.40, side="sell"))
    assert sell["EURUSD"].entry_mode == "market" and sell["EURUSD"].price == 100.00 and "EURUSD" in sell_caps
    # a normal resting limit and an already-resting limit order are left alone
    assert _run_mode_guard(_mode_entry("limit", price=99.50))[0]["EURUSD"].entry_mode == "limit"
    resting = PendingOrder(symbol="EURUSD", volume=1.0, order_type="buy limit", price_open=100.10, sl=99.2, tp=101.6, ticket=9, time_setup=None)
    entry = _mode_entry("limit", price=100.10)
    assert _run_mode_guard(entry, pending=[resting])[0]["EURUSD"] is entry
    # the same caps as any market entry: a second one the same day downgrades to the limit, a dead setup is rejected
    used, _ = _run_mode_guard(_mode_entry("limit", price=100.10), used={"EURUSD"})
    assert used["EURUSD"].entry_mode == "limit" and "already used" in used["EURUSD"].reason
    dead, _ = _run_mode_guard(_mode_entry("limit", price=100.10, stop=100.10))
    assert dead["EURUSD"].pct == 0.0 and "through the stop" in dead["EURUSD"].reason


def test_fast_lane_has_work_only_when_the_suggestion_gives_it_something():
    from clerk_fast_job import fast_lane_has_work

    assert not fast_lane_has_work(None, {}) and not fast_lane_has_work({"immediate_allocation": {"CASH": {"pct": 99}}, "pending_setups": []}, {})
    assert fast_lane_has_work({"immediate_allocation": {"EURUSD": {"pct": 0.25}}, "pending_setups": []}, {})
    assert not fast_lane_has_work({"immediate_allocation": {"EURUSD": {"pct": 0.0}}, "pending_setups": [{"symbol": "X"}]}, {})
    assert fast_lane_has_work({"immediate_allocation": {}, "pending_setups": [{"symbol": "X", "trigger": {"kind": "range_break", "level": 1}}]}, {})
    assert fast_lane_has_work({"immediate_allocation": {}, "pending_setups": []}, {"settled": {"SOL": {"state": "filled"}}})


@patch("ai.clerk_execution.is_symbol_tradable_now", return_value=True)
@patch("ai.clerk_execution._run_clerk_verdict")
@patch("ai.clerk_execution._fetch_technical_context")
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_the_fast_lane_places_an_entry_without_a_model_and_leaves_the_full_polls_headline_and_countdown_alone(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_verdict, mock_tradable, _fixed_files,
):
    import pandas as pd
    from ai.clerk_execution import read_execution_progress
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=2004.0, ask=2004.3)]
    bars = pd.DataFrame([(1999.0, 1997.0, 1998.0), (2000.4, 1998.0, 1999.8)], columns=["High", "Low", "Close"])
    mock_fetch_ctx.return_value = (_fake_ftmo_analysis(symbol="XAUUSD", m5_stats=_fake_technical_stats(atr=2.0), m5_recent=bars), "ctx")
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_open.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=901)
    from ai.clerk_execution import _write_execution_state

    _write_execution_state("success", "full poll headline")  # what the desk shows from the last FULL poll
    before = read_execution_state()
    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0, trade_contract_size=100.0, currency_margin="USD", margin_initial=1000.0,
        )
        _write_suggestion(
            _fixed_files,
            immediate_allocation={"XAUUSD": {"pct": 0.5, "price": 2004.3, "stop_loss": 1994.0, "take_profit": 2030.0, "side": "buy", "reason": "r", "entry_mode": "market"}},
        )
        run_clerk_execution_check(fast=True)
    mock_verdict.assert_not_called()  # no model call in the fast lane
    mock_open.assert_called_once()  # ... and the entry still went out
    assert mock_open.call_args.kwargs["kind"] == "market"
    after = read_execution_state()
    assert after["last_detail"] == before["last_detail"] == "full poll headline"
    assert after["last_run_completed_utc"] == before["last_run_completed_utc"]  # the full poll's countdown is untouched
    assert read_execution_progress() == {}  # no live-progress flicker on the desk


# --- thinking vs acting (2026-09-26) ------------------------------------------------------------------------------------

def _split_poll_scaffold(mock_account, mock_watch, mock_fetch_ctx, mock_status, mock_open):
    import pandas as pd
    from data.mt5_source import AccountSummary, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=1999.0, ask=2000.0)]
    bars = pd.DataFrame([(1999.0, 1997.0, 1998.0), (2000.4, 1998.0, 1999.8)], columns=["High", "Low", "Close"])
    mock_fetch_ctx.return_value = (_fake_ftmo_analysis(symbol="XAUUSD", m5_stats=_fake_technical_stats(atr=2.0), m5_recent=bars), "ctx")
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_open.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=902)


_FREE_TEXT_SETUP = {"symbol": "XAUUSD", "side": "buy", "pct": 1.0, "trigger_condition": "cond",
                    "price": 2000.0, "stop_loss": 1980.0, "take_profit": 2050.0, "reason": "r"}


@patch("ai.clerk_execution.is_symbol_tradable_now", return_value=True)
@patch("ai.clerk_execution._run_clerk_verdict")
@patch("ai.clerk_execution._fetch_technical_context")
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions", return_value=[])
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_the_thinking_pass_stores_verdicts_and_touches_nothing_else_then_an_acting_pass_uses_them_once(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_verdict, mock_tradable, _fixed_files, monkeypatch,
):
    from ai import clerk_thinking
    from ai.clerk_execution import _load_settlement, _write_execution_state
    from ai.portfolio_suggest import PendingSetup
    from data.mt5_source import ContractSpec

    monkeypatch.setattr(config, "CLERK_THINK_SPLIT_ENABLED", True)
    _split_poll_scaffold(mock_account, mock_watch, mock_fetch_ctx, mock_status, mock_open)
    setup = PendingSetup(symbol="XAUUSD", side="buy", pct=1.0, trigger_condition="cond", price=2000.0, stop_loss=1980.0, take_profit=2050.0, reason="r")
    mock_verdict.return_value = (setup, True, "FINAL_VERDICT: CONFIRMED")
    _write_execution_state("success", "headline before")
    before_state = read_execution_state()
    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(volume_min=0.01, volume_step=0.01, volume_max=100.0, trade_contract_size=100.0, currency_margin="USD", margin_initial=1000.0)
        _write_suggestion(_fixed_files, pending_setups=[_FREE_TEXT_SETUP])

        # 1) the THINKING pass: the model is asked, the verdict is stored, and NOTHING is acted on or recorded
        with patch("ai.clerk_execution._restore_external_stop_drift") as mock_drift:
            run_clerk_execution_check(think=True)
        mock_drift.assert_not_called()  # restoring a stop modifies a live position: an action, never in the thinking pass
        mock_verdict.assert_called_once()
        mock_open.assert_not_called()
        assert read_execution_state() == before_state  # desk status untouched
        assert _load_settlement()["settled"] == {}  # settlement untouched
        stored = clerk_thinking.load_cache()["items"]["pending:XAUUSD"]
        assert stored["confirmed"] is True and stored["consumed_utc"] is None

        # 2) an ACTING pass never calls the model, takes the stored verdict, and places the order
        mock_verdict.reset_mock()
        run_clerk_execution_check()
        mock_verdict.assert_not_called()
        mock_open.assert_called_once()
        assert clerk_thinking.load_cache()["items"]["pending:XAUUSD"]["consumed_utc"]
        assert "thinker verdict(s) applied this pass" in read_execution_state()["last_detail"]

    # 3) a stale verdict is never used
    cache = clerk_thinking.load_cache()
    item = cache["items"]["pending:XAUUSD"]
    item["consumed_utc"] = None
    item["checked_utc"] = (datetime.now(timezone.utc) - timedelta(minutes=config.CLERK_THINK_CACHE_TTL_MINUTES + 5)).isoformat()
    clerk_thinking.save_cache(cache)
    results, applied = __import__("ai.clerk_execution", fromlist=["x"])._cached_thinking_results(
        [(setup, "ctx", "d")], [], [], 100000.0, cache["generated_utc"]
    )
    assert results == [] and applied == 0


@patch("ai.clerk_execution.is_symbol_tradable_now", return_value=False)
@patch("ai.clerk_execution._restore_external_stop_drift", return_value=[])
@patch("ai.clerk_execution._fetch_technical_context")
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_the_stop_drift_restore_is_not_attempted_against_a_closed_market(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_fetch_ctx, mock_restore, mock_tradable, _fixed_files,
):
    _split_poll_scaffold(mock_account, mock_watch, mock_fetch_ctx, mock_status, mock_open)
    mock_positions.return_value = [_position(symbol="XAUUSD", side="buy")]
    _write_suggestion(_fixed_files, immediate_allocation={})
    run_clerk_execution_check(fast=True)
    assert mock_restore.call_args.args[0] == {}  # the only held symbol is on a closed market: nothing is attempted


# --- the thinking diet (2026-09-26) -------------------------------------------------------------------------------------

def _bars_with_last_close(close):
    import pandas as pd

    return pd.DataFrame([(64.6, 64.2, 64.3), (64.7, 64.3, close)], columns=["High", "Low", "Close"])


def test_a_purely_mechanical_m5_invalidation_is_decided_in_python_from_the_last_completed_close(monkeypatch):
    from ai.clerk_execution import deterministic_invalidation

    monkeypatch.setattr(config, "CLERK_DETERMINISTIC_INVALIDATION", True)
    sell = {"side": "sell"}
    hit = deterministic_invalidation(sell, "M5 closes above 64.49", _fake_ftmo_analysis(m5_recent=_bars_with_last_close(64.52)))
    assert hit[0] is True and "HAS closed beyond it" in hit[1] and hit[1].endswith("FINAL_VERDICT: CONFIRMED")
    assert hit[1].startswith("[deterministic circuit-breaker")  # the outage detector and the thinking cache both ignore it
    miss = deterministic_invalidation(sell, "M5 closes above 64.49.", _fake_ftmo_analysis(m5_recent=_bars_with_last_close(64.30)))
    assert miss[0] is False and miss[1].endswith("FINAL_VERDICT: NOT_CONFIRMED")
    buy = deterministic_invalidation({"side": "buy"}, "the M5 candle closes below $1,250.5", _fake_ftmo_analysis(m5_recent=_bars_with_last_close(1249.0)))
    assert buy[0] is True
    # anything that is not a bare mechanical line, or points the wrong way for the side, still goes to the model
    for cond, side in (
        ("M5 closes above 64.49 with RSI below 40", "sell"), ("H1 closes above 64.49", "sell"), ("M5 closes below 64.49", "sell"),
        ("M5 closes above 64.49 or volume dries up", "sell"), ("", "sell"), (None, "sell"),
    ):
        assert deterministic_invalidation({"side": side}, cond, _fake_ftmo_analysis(m5_recent=_bars_with_last_close(64.6))) is None
    assert deterministic_invalidation(sell, "M5 closes above 64.49", _fake_ftmo_analysis()) is None  # no completed bars: model decides
    monkeypatch.setattr(config, "CLERK_DETERMINISTIC_INVALIDATION", False)
    assert deterministic_invalidation(sell, "M5 closes above 64.49", _fake_ftmo_analysis(m5_recent=_bars_with_last_close(64.6))) is None


def test_a_condition_written_on_a_higher_timeframe_gets_the_full_context_and_an_m5_one_keeps_the_compact_context(monkeypatch):
    from ai.clerk_execution import _condition_needs_htf, _context_for_condition

    monkeypatch.setattr(config, "CLERK_COMPACT_CONTEXT", True)
    assert _condition_needs_htf("H1 closes back above 85080.00") and _condition_needs_htf("daily close below 100")
    assert not _condition_needs_htf("M5 closes above 1.1397") and not _condition_needs_htf(None)
    analysis = _fake_ftmo_analysis()
    assert _context_for_condition(analysis, "COMPACT", "M5 closes above 1.1397", 10_000.0) == "COMPACT"
    with patch("ai.clerk_execution.format_ftmo_asset_context", return_value="FULL") as full, patch("ai.clerk_execution.economic_calendar.fetch_calendar_events", return_value=[]):
        assert _context_for_condition(analysis, "COMPACT", "H1 closes above 1.1397", 10_000.0) == "FULL"
        full.assert_called_once()
    with patch("ai.clerk_execution.format_ftmo_asset_context", side_effect=RuntimeError("boom")):
        assert _context_for_condition(analysis, "COMPACT", "H4 trend flips", 10_000.0) == "COMPACT"  # never blocks a check
    monkeypatch.setattr(config, "CLERK_COMPACT_CONTEXT", False)
    assert _context_for_condition(analysis, "ORIGINAL", "H1 closes above 1.1397", 10_000.0) == "ORIGINAL"


def test_the_clerk_prompts_ask_for_short_reasoning():
    from ai.clerk_execution import _build_invalidation_prompt, _build_verdict_prompt
    from ai.portfolio_suggest import PendingSetup

    inv = _build_invalidation_prompt("XAGUSD", {"side": "sell"}, "M5 closes above 64.49", "ctx", 1000.0, "1 hour(s)", "filled")
    assert "Keep your reasoning SHORT - at most two sentences" in inv
    setup = PendingSetup(symbol="X", side="buy", pct=1.0, trigger_condition="c")
    assert "Keep your reasoning short (at most four sentences)" in _build_verdict_prompt(setup, "ctx", 1000.0, "1 hour(s)", "")


@patch("ai.clerk_execution._run_clerk_invalidation_check")
@patch("ai.clerk_execution.modify_position_sltp")
@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_a_mechanical_invalidation_closes_the_position_without_any_model_call(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_close, mock_modify, mock_invalidation, _fixed_files, monkeypatch,
):
    import pandas as pd
    from data.mt5_source import AccountSummary, ContractSpec, MarketAsset
    from risk.ftmo_rules import FtmoStatus

    monkeypatch.setattr(config, "CLERK_DETERMINISTIC_INVALIDATION", True)
    bars = pd.DataFrame([(1.0902, 1.0895, 1.0898), (1.0899, 1.0888, 1.0890)], columns=["High", "Low", "Close"])
    analysis = _fake_ftmo_analysis(symbol="EURUSD", m5_recent=bars)
    mock_account.return_value = AccountSummary(balance=100000.0, equity=100000.0, free_margin=90000.0, currency="USD")
    mock_positions.return_value = [_position(symbol="EURUSD", side="buy", volume=1.0, ticket=200)]
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0889, ask=1.0890)]
    mock_status.return_value = FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=3.0, trailing_max_loss_floor=90000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )
    mock_close.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=200)
    mock_modify.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=200)
    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {"pct": 1.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None, "side": "buy", "reason": "r",
             "invalidation_condition": "M5 closes below 1.0895"}
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "EURUSD", "filled", generated_utc, entry=entry)
    with patch("ai.clerk_execution._fetch_technical_context", return_value=(analysis, "ctx")), patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = ContractSpec(
            volume_min=0.01, volume_step=0.01, volume_max=100.0, trade_contract_size=100000.0, currency_margin="USD", margin_initial=1000.0,
        )
        run_clerk_execution_check()
    mock_invalidation.assert_not_called()  # no model needed for "M5 closes below 1.0895" against a last close of 1.0890
    mock_close.assert_called_once()
    verdict = read_execution_state()["last_verdicts"]["EURUSD"]
    assert verdict["confirmed"] is True and "last completed M5 close is 1.089" in verdict["raw_text"]


def test_the_tactical_prompt_drops_the_book_rationale_lines_only_with_the_compact_setting(monkeypatch):
    from ai.clerk_execution import _build_tactical_prompt, _compute_tactical_signals

    def prompt():
        pos = _position(symbol="EURUSD", side="buy", volume=1.0, ticket=1)
        entry = AllocationEntry(pct=1.0, price=1.09, stop_loss=1.085, take_profit=1.1, side="buy", reason="r")
        signals = _compute_tactical_signals(pos, entry, _fake_technical_stats(), _fake_technical_stats(), _empty_chart_structure(), _fake_ftmo_analysis().base, None)
        return _build_tactical_prompt("EURUSD", entry, pos, "ctx", 1000.0, None, signals)

    monkeypatch.setattr(config, "CLERK_COMPACT_CONTEXT", True)
    brief = prompt()
    monkeypatch.setattr(config, "CLERK_COMPACT_CONTEXT", False)
    full = prompt()
    assert "Why:" not in brief and "Why:" in full and len(brief) < len(full) - 2000
    assert "Keep your reasoning short - at most three sentences" in brief


def test_fetch_technical_context_asks_for_the_lean_analysis_only_when_told_to():
    asset = MarketAsset(symbol="EURUSD", description="Euro vs US Dollar", bid=1.1, ask=1.1005)
    with (
        patch("ai.clerk_execution.analyze_ftmo_asset_live", return_value=_fake_ftmo_analysis()) as mock_live,
        patch("ai.clerk_execution.economic_calendar.fetch_calendar_events", return_value=[]),
        patch("ai.clerk_execution.format_ftmo_asset_context", return_value="ctx"),
    ):
        _fetch_technical_context("EURUSD", {"EURUSD": asset}, 10_000.0)
        assert "lean" not in mock_live.call_args.kwargs
        _fetch_technical_context("EURUSD", {"EURUSD": asset}, 10_000.0, lean=True)
        assert mock_live.call_args.kwargs["lean"] is True


# --- Continuation Watch: registering a watch right where a WINNING close is detected (2026-09-28) ---------

def test_maybe_register_continuation_watch_registers_only_a_real_win(tmp_path):
    from ai.clerk_execution import _maybe_register_continuation_watch
    from ai.continuation_hunter import load_state

    trade = _closed_trade(symbol="XAUUSD", side="buy", close_price=1.13, profit=300.0)
    n = 20
    idx = pd.DatetimeIndex([trade.closed_at - timedelta(minutes=5 * (n - i)) for i in range(n)])
    m5_bars = pd.DataFrame({"Open": [1.10] * n, "High": [1.12] * n, "Low": [1.09] * n, "Close": [1.11] * n}, index=idx)
    with patch("ai.clerk_execution.fetch_mt5_price_history_range", return_value=m5_bars) as mock_fetch:
        _maybe_register_continuation_watch("XAUUSD", "XAUUSD_2026-09-10_080000", trade)

    state = load_state()
    assert state["watching"]["XAUUSD"]["side"] == "buy"
    assert state["watching"]["XAUUSD"]["exit_price"] == 1.13
    assert state["watching"]["XAUUSD"]["story_id"] == "XAUUSD_2026-09-10_080000"
    assert state["watching"]["XAUUSD"]["atr_at_close"] is not None
    mock_fetch.assert_called_once()
    call = mock_fetch.call_args
    assert call.args[0] == "XAUUSD" and call.args[1] == "M5"
    assert call.args[3] == trade.closed_at  # the fetch window ends exactly at the close


def test_maybe_register_continuation_watch_never_raises_on_a_bad_atr_fetch():
    from ai.clerk_execution import _maybe_register_continuation_watch

    trade = _closed_trade(symbol="XAUUSD")
    with patch("ai.clerk_execution.fetch_mt5_price_history_range", side_effect=RuntimeError("MT5 down")):
        _maybe_register_continuation_watch("XAUUSD", "XAUUSD_2026-09-10_080000", trade)  # must not raise


def test_maybe_register_continuation_watch_respects_its_own_kill_switch():
    from ai.clerk_execution import _maybe_register_continuation_watch
    from ai.continuation_hunter import load_state

    trade = _closed_trade(symbol="XAUUSD")
    with (
        patch.object(config, "CONTINUATION_WATCH_ENABLED", False),
        patch("ai.clerk_execution.fetch_mt5_price_history_range") as mock_fetch,
    ):
        _maybe_register_continuation_watch("XAUUSD", "XAUUSD_2026-09-10_080000", trade)
    mock_fetch.assert_not_called()
    assert "XAUUSD" not in load_state()["watching"]


def test_record_trade_journal_closures_registers_a_watch_only_for_a_real_win(tmp_path):
    from ai.clerk_execution import _record_trade_journal_closures
    from ai.continuation_hunter import load_state
    from ai import trade_journal

    def _payload(symbol, price, stop_loss):
        return {
            "immediate_allocation": {
                symbol: {"pct": 1.0, "price": price, "stop_loss": stop_loss, "take_profit": None, "side": "buy", "reason": "r", "invalidation_condition": None}
            },
            "pending_setups": [],
        }

    with patch.object(config, "OBSIDIAN_VAULT_PATH", str(tmp_path)):
        trade_journal.record_proposals(_payload("XAUUSD", 1.10, 1.08))
        trade_journal.record_order_result("XAUUSD", "open", True, "placed")
        trade_journal.record_filled("XAUUSD", 1, 1.10)
        trade_journal.record_proposals(_payload("GBPUSD", 1.20, 1.18))
        trade_journal.record_order_result("GBPUSD", "open", True, "placed")
        trade_journal.record_filled("GBPUSD", 2, 1.20)

        win = _closed_trade(symbol="XAUUSD", side="buy", close_price=1.13, profit=300.0)
        loss = _closed_trade(symbol="GBPUSD", side="buy", close_price=1.15, profit=-50.0)
        n = 20
        idx = pd.DatetimeIndex([win.closed_at - timedelta(minutes=5 * (n - i)) for i in range(n)])
        m5_bars = pd.DataFrame({"Open": [1.10] * n, "High": [1.13] * n, "Low": [1.09] * n, "Close": [1.12] * n}, index=idx)
        with (
            patch("ai.clerk_execution.get_history_deals", return_value=[]),
            patch("ai.clerk_execution.group_closed_trades", return_value=[win, loss]),
            patch("ai.clerk_execution.fetch_mt5_price_history_range", return_value=m5_bars),
        ):
            _record_trade_journal_closures(["XAUUSD", "GBPUSD"], {"XAUUSD": {"entry": {}}, "GBPUSD": {"entry": {}}})

    state = load_state()
    assert "XAUUSD" in state["watching"]
    assert "GBPUSD" not in state["watching"]  # a loser is never watched for continuation
