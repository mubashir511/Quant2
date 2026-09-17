import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import config
from ai.ollama_client import FAILED_MESSAGE as OLLAMA_FAILED_MESSAGE
from ai.clerk_execution import (
    SymbolSettlement,
    TacticalVerdict,
    _apply_atr_stop_floor_guard,
    _apply_correlation_guard,
    _build_carried_forward_allocation,
    _build_verdict_prompt,
    _clerk_news_cache,
    _compute_realized_r,
    _detect_external_stop_drift,
    _detect_newly_closed_symbols,
    _detect_ollama_outage,
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
from ai.ftmo_suggest import FtmoAssetAnalysis
from ai.portfolio_suggest import AllocationEntry, AssetAnalysis, PendingSetup
from analysis.chart_structure import ChartStructureSnapshot
from analysis.technical import TechnicalStats
from data.mt5_execution import OrderResult
from data.mt5_source import ClosedTrade, MarketAsset, PendingOrder, Position


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
        patch.object(config, "CLERK_EXECUTION_GRACE_MINUTES", 10),
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
    from ai.clerk_execution import _interval_start

    state = {"last_run_interval_utc": _interval_start(_utc(2026, 8, 23, 15, 0)).isoformat()}
    assert is_execution_due(_utc(2026, 8, 23, 15, 5), state=state) is False


def test_is_execution_due_true_again_next_interval():
    from ai.clerk_execution import _interval_start

    state = {"last_run_interval_utc": _interval_start(_utc(2026, 8, 23, 15, 0)).isoformat()}
    # 15:20 falls in the NEXT 15-minute window (15:15-15:29), distinct
    # from the one already marked as run.
    assert is_execution_due(_utc(2026, 8, 23, 15, 20), state=state) is True


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


def test_interval_start_uses_the_overridden_interval(_fixed_files):
    from ai.clerk_execution import _interval_start

    set_clerk_execution_interval_minutes(30)
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
    from ai.clerk_execution import _interval_start

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
    with patch.object(config, "CLERK_EXECUTION_STATE_FILE", str(corrupt_path)):
        assert read_execution_state() == {}


# --- _write_execution_state: preserves last_run_interval_utc, merges last_verdicts ---


def test_write_execution_state_preserves_last_run_interval_utc_across_calls(_fixed_files):
    from ai.clerk_execution import _interval_start, _mark_interval_ran, _write_execution_state

    marked_at = datetime(2026, 8, 23, 15, 5, tzinfo=timezone.utc)
    _mark_interval_ran(marked_at)
    # A later write for an unrelated outcome (e.g. a "no_suggestion" poll
    # in a different interval) must not erase the interval dedup marker.
    _write_execution_state("no_suggestion")
    assert read_execution_state()["last_run_interval_utc"] == _interval_start(marked_at).isoformat()


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
    mock_run_ollama.assert_called_once_with("some prompt", model=config.CLERK_PRIMARY_MODEL, timeout=150)


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

    assert result["BTCUSD"].pct == pytest.approx(0.25)  # halved from 0.5
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
    assert result["ETHUSD"].pct == pytest.approx(0.15)
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
    assert result["ETHUSD"].pct == pytest.approx(0.15)  # the new one is still reduced


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

    assert result["BTCUSD"].pct == pytest.approx(0.25)  # halved -- this really is a stacked bet


# --- _apply_atr_stop_floor_guard (2026-09-16, real EURUSD 2026-09-07 incident) ---


def test_atr_stop_floor_guard_widens_a_too_tight_buy_stop():
    # Real EURUSD-shaped numbers: entry 1.16231, stop 1.1636 is on the
    # WRONG side for a buy (that's a sell-shaped stop) -- use a real buy
    # shape instead: H1 ATR 0.000697 (0.06% of 1.16231, per the real
    # audit), floor = 1.5x = 0.0010455, stop only 0.0004 away (way
    # tighter than the floor) -- must widen to exactly the floor below price.
    allocation = {
        "EURUSD": AllocationEntry(pct=0.5, price=1.16231, stop_loss=1.16191, side="buy", reason="thesis"),
    }
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", h1_atr=0.000697), "")}
    result = _apply_atr_stop_floor_guard(allocation, [], cache)
    expected_stop = 1.16231 - 1.5 * 0.000697
    assert result["EURUSD"].stop_loss == pytest.approx(expected_stop)
    assert "ATR stop-floor guard" in result["EURUSD"].reason


def test_atr_stop_floor_guard_widens_a_too_tight_sell_stop_on_the_correct_side():
    allocation = {
        "EURUSD": AllocationEntry(pct=0.5, price=1.16231, stop_loss=1.16271, side="sell", reason="thesis"),
    }
    cache = {"EURUSD": (_fake_ftmo_analysis(symbol="EURUSD", h1_atr=0.000697), "")}
    result = _apply_atr_stop_floor_guard(allocation, [], cache)
    expected_stop = 1.16231 + 1.5 * 0.000697
    assert result["EURUSD"].stop_loss == pytest.approx(expected_stop)
    assert result["EURUSD"].stop_loss > 1.16231  # correct side for a sell


def test_atr_stop_floor_guard_leaves_a_stop_already_at_the_floor_untouched():
    atr = 0.000697
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


# --- Clerk news feed (2026-09-17: closes the gap where local Ollama
# models can't fulfil the verdict prompt's own "check for major news"
# instruction) -------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_clerk_news_cache():
    _clerk_news_cache.clear()
    yield
    _clerk_news_cache.clear()


def test_fetch_clerk_news_block_returns_real_headlines():
    with patch("ai.clerk_execution.resolve_yahoo_ticker", return_value=("Gold", "GC=F")):
        with patch("ai.clerk_execution.fetch_recent_headlines", return_value=["Gold hits record high"]) as fetch:
            block = _fetch_clerk_news_block("XAUUSD", "Gold vs US Dollar")
    assert block == "- Gold hits record high"
    fetch.assert_called_once_with("GC=F", limit=config.NEWS_HEADLINES_PER_ASSET)


def test_fetch_clerk_news_block_empty_when_ticker_does_not_resolve():
    with patch("ai.clerk_execution.resolve_yahoo_ticker", return_value=None):
        with patch("ai.clerk_execution.fetch_recent_headlines") as fetch:
            block = _fetch_clerk_news_block("UNKNOWN.c", "Some CFD")
    assert block == ""
    fetch.assert_not_called()


def test_fetch_clerk_news_block_empty_when_fetch_returns_nothing():
    with patch("ai.clerk_execution.resolve_yahoo_ticker", return_value=("Gold", "GC=F")):
        with patch("ai.clerk_execution.fetch_recent_headlines", return_value=[]):
            block = _fetch_clerk_news_block("XAUUSD", "Gold vs US Dollar")
    assert block == ""


def test_fetch_clerk_news_block_is_cached_within_the_configured_window():
    with patch("ai.clerk_execution.resolve_yahoo_ticker", return_value=("Gold", "GC=F")):
        with patch("ai.clerk_execution.fetch_recent_headlines", return_value=["headline one"]) as fetch:
            first = _fetch_clerk_news_block("XAUUSD", "Gold vs US Dollar")
            second = _fetch_clerk_news_block("XAUUSD", "Gold vs US Dollar")
    assert first == second == "- headline one"
    fetch.assert_called_once()


def test_fetch_clerk_news_block_refetches_after_the_cache_window_expires():
    from datetime import timedelta

    with patch("ai.clerk_execution.resolve_yahoo_ticker", return_value=("Gold", "GC=F")):
        with patch("ai.clerk_execution.fetch_recent_headlines", return_value=["old headline"]):
            _fetch_clerk_news_block("XAUUSD", "Gold vs US Dollar")
        stale_time = datetime.now(timezone.utc) - timedelta(minutes=config.CLERK_NEWS_CACHE_MINUTES + 1)
        _clerk_news_cache["XAUUSD"] = (stale_time, "- old headline")
        with patch("ai.clerk_execution.fetch_recent_headlines", return_value=["new headline"]) as fetch:
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
