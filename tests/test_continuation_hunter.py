"""Tests for ai/continuation_hunter.py — Continuation Watch, phase 2 (the orchestration).

All model/order side effects are mocked; the goal here is the WIRING: which gate stops a candidate where, what
gets written to the trade journal, and — critically — that a "proposed" decision in LOG_ONLY mode (the
default) never calls anything that could place a real order."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import config
from ai import continuation_hunter as ch
from ai import trade_journal
from data.mt5_source import MarketAsset


def _m5_frame(start: datetime, closes: list[float]) -> pd.DataFrame:
    idx = [start + timedelta(minutes=5 * i) for i in range(len(closes))]
    rows = [{"Open": c, "High": c + 0.05, "Low": c - 0.05, "Close": c} for c in closes]
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def _register(symbol="XAUUSD", side="buy", exit_price=100.0, exit_time=None, atr=1.0, story_id=None):
    exit_time = exit_time or datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    story_id = story_id or f"{symbol}_2026-01-01_110000"
    ch.register_watch(symbol, story_id, side, exit_price, exit_time, atr)
    return exit_time, story_id


def _open_story(symbol: str, story_id: str) -> None:
    """A real journal story to attach the continuation_watch event to (record_continuation_watch looks it up
    by story_id directly, same as record_audit does for an already-closed story)."""
    trade_journal._save_story(trade_journal.TradeStory(symbol=symbol, story_id=story_id, status="closed_won", events=[]))


# --- register_watch / state -----------------------------------------------------------------------------

def test_register_watch_writes_state_and_skips_without_a_usable_atr():
    exit_time, story_id = _register()
    state = ch.load_state()
    assert state["watching"]["XAUUSD"]["side"] == "buy"
    assert state["watching"]["XAUUSD"]["story_id"] == story_id

    ch.register_watch("EURUSD", "EURUSD_x", "sell", 1.1, exit_time, atr_at_close=None)
    assert "EURUSD" not in ch.load_state()["watching"]
    ch.register_watch("EURUSD", "EURUSD_x", "sell", 1.1, exit_time, atr_at_close=0.0)
    assert "EURUSD" not in ch.load_state()["watching"]


def test_a_fresh_registration_on_the_same_symbol_replaces_the_old_watch():
    _register(symbol="XAUUSD", exit_price=100.0)
    _register(symbol="XAUUSD", exit_price=200.0, story_id="XAUUSD_new")
    assert ch.load_state()["watching"]["XAUUSD"]["exit_price"] == 200.0
    assert ch.load_state()["watching"]["XAUUSD"]["story_id"] == "XAUUSD_new"


# --- run_continuation_watch_check: gate 1 (phase-1 filter) ----------------------------------------------

def test_no_watches_touches_no_mt5_and_no_model():
    with patch("ai.continuation_hunter.fetch_mt5_price_history_range") as mock_fetch:
        summary = ch.run_continuation_watch_check(datetime.now(timezone.utc))
    assert summary["checked"] == 0
    mock_fetch.assert_not_called()


def test_still_waiting_leaves_the_watch_in_place_and_writes_no_journal_event():
    exit_time, story_id = _register()
    _open_story("XAUUSD", story_id)
    only_one_bar = _m5_frame(exit_time + timedelta(minutes=5), [100.5])
    with patch("ai.continuation_hunter.fetch_mt5_price_history_range", return_value=only_one_bar):
        summary = ch.run_continuation_watch_check(exit_time + timedelta(minutes=6))
    assert summary == {"checked": 1, "still_waiting": 1, "expired": 0, "vetoed_cost": 0, "proposed": 0, "skipped": 0}
    assert "XAUUSD" in ch.load_state()["watching"]
    story = trade_journal.find_story_by_id(story_id)
    assert not any(e.type == "continuation_watch" for e in story.events)


def test_expired_without_enough_movement_resolves_as_no_continuation_and_never_calls_a_model():
    exit_time, story_id = _register(atr=1.0)
    _open_story("XAUUSD", story_id)
    flat_bars = _m5_frame(exit_time + timedelta(minutes=5), [100.1, 100.1, 100.1])
    with (
        patch("ai.continuation_hunter.fetch_mt5_price_history_range", return_value=flat_bars),
        patch("ai.continuation_hunter._ask_local_then_openrouter") as mock_ask,
    ):
        summary = ch.run_continuation_watch_check(exit_time + timedelta(minutes=20))
    assert summary["expired"] == 1 and summary["checked"] == 1
    mock_ask.assert_not_called()
    assert "XAUUSD" not in ch.load_state()["watching"]
    story = trade_journal.find_story_by_id(story_id)
    event = next(e for e in story.events if e.type == "continuation_watch")
    assert event.data["outcome"] == "no_continuation"


# --- gate 2: cost-drag veto, before any model call --------------------------------------------------------

def _passing_bars(exit_time: datetime) -> pd.DataFrame:
    return _m5_frame(exit_time + timedelta(minutes=5), [105.0, 108.0, 110.0])  # +10 ATR move on atr=1.0


def test_a_real_continuation_still_gets_vetoed_on_cost_and_never_reaches_a_model():
    exit_time, story_id = _register(atr=1.0)
    _open_story("XAUUSD", story_id)
    asset = MarketAsset(symbol="XAUUSD", description="Gold", bid=110.0, ask=110.05)
    with (
        patch("ai.continuation_hunter.fetch_mt5_price_history_range", return_value=_passing_bars(exit_time)),
        patch("data.mt5_source.get_market_watch", return_value=[asset]),
        patch("ai.continuation_hunter.analyze_ftmo_asset_live", return_value=SimpleNamespace(m5_stats=SimpleNamespace(atr=1.0))),
        patch("ai.continuation_hunter.cost_drag_r", return_value=config.COST_DRAG_VETO_R + 0.05),
        patch("ai.continuation_hunter._ask_local_then_openrouter") as mock_ask,
    ):
        summary = ch.run_continuation_watch_check(exit_time + timedelta(minutes=16))
    assert summary["vetoed_cost"] == 1
    mock_ask.assert_not_called()
    assert "XAUUSD" not in ch.load_state()["watching"]
    event = next(e for e in trade_journal.find_story_by_id(story_id).events if e.type == "continuation_watch")
    assert event.data["outcome"] == "vetoed_cost"


def _pass_cost_gate(stack):
    """Enters patches (via the given ExitStack) for analyze_ftmo_asset_live/cost_drag_r/format_clerk_context/
    get_market_watch so a candidate always clears gate 2 and reaches the model stage."""
    asset = MarketAsset(symbol="XAUUSD", description="Gold", bid=110.0, ask=110.05)
    stack.enter_context(patch("data.mt5_source.get_market_watch", return_value=[asset]))
    stack.enter_context(patch("ai.continuation_hunter.analyze_ftmo_asset_live", return_value=SimpleNamespace(m5_stats=SimpleNamespace(atr=1.0))))
    stack.enter_context(patch("ai.continuation_hunter.cost_drag_r", return_value=0.05))
    stack.enter_context(patch("ai.continuation_hunter.format_clerk_context", return_value="[technical context]"))


# --- gate 3: the model decides -- and LOG_ONLY never places an order -------------------------------------

def test_a_model_yes_is_logged_as_proposed_and_never_places_an_order():
    exit_time, story_id = _register(atr=1.0)
    _open_story("XAUUSD", story_id)
    raw = "CONTINUE: yes\nENTRY: 111.0\nSTOP: 109.5\nTARGET: 115.0\nREASON: real structure, still running"
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("ai.continuation_hunter.fetch_mt5_price_history_range", return_value=_passing_bars(exit_time)))
        _pass_cost_gate(stack)
        stack.enter_context(patch("ai.continuation_hunter._ask_local_then_openrouter", return_value=(raw, config.CLERK_PRIMARY_MODEL)))
        assert config.CONTINUATION_WATCH_LOG_ONLY is True  # the default this test relies on
        summary = ch.run_continuation_watch_check(exit_time + timedelta(minutes=16))
    assert summary["proposed"] == 1
    assert "XAUUSD" not in ch.load_state()["watching"]
    event = next(e for e in trade_journal.find_story_by_id(story_id).events if e.type == "continuation_watch")
    assert event.data["outcome"] == "proposed"
    assert event.data["propose"] is True and event.data["entry"] == 111.0 and event.data["stop"] == 109.5
    assert event.data["log_only"] is True
    assert event.data["model_used"] == config.CLERK_PRIMARY_MODEL


def test_a_model_no_is_logged_as_skipped():
    exit_time, story_id = _register(atr=1.0)
    _open_story("XAUUSD", story_id)
    raw = "CONTINUE: no\nREASON: already extended, RSI overbought"
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("ai.continuation_hunter.fetch_mt5_price_history_range", return_value=_passing_bars(exit_time)))
        _pass_cost_gate(stack)
        stack.enter_context(patch("ai.continuation_hunter._ask_local_then_openrouter", return_value=(raw, config.CLERK_PRIMARY_MODEL)))
        summary = ch.run_continuation_watch_check(exit_time + timedelta(minutes=16))
    assert summary["skipped"] == 1
    event = next(e for e in trade_journal.find_story_by_id(story_id).events if e.type == "continuation_watch")
    assert event.data["outcome"] == "skipped" and event.data["propose"] is False


def test_every_model_failing_is_logged_as_skipped_not_a_crash():
    exit_time, story_id = _register(atr=1.0)
    _open_story("XAUUSD", story_id)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("ai.continuation_hunter.fetch_mt5_price_history_range", return_value=_passing_bars(exit_time)))
        _pass_cost_gate(stack)
        stack.enter_context(patch("ai.continuation_hunter._ask_local_then_openrouter", return_value=("", "none (local and every OpenRouter fallback failed)")))
        summary = ch.run_continuation_watch_check(exit_time + timedelta(minutes=16))
    assert summary["skipped"] == 1
    event = next(e for e in trade_journal.find_story_by_id(story_id).events if e.type == "continuation_watch")
    assert event.data["propose"] is False and "every model failed" in event.data["reason"]


# --- local-then-OpenRouter escalation ---------------------------------------------------------------------

def test_escalates_to_openrouter_only_when_the_local_model_is_down_or_unclear():
    with (
        patch("ai.continuation_hunter.run_ollama", return_value=ch.OLLAMA_FAILED_MESSAGE),
        patch("ai.continuation_hunter.run_openrouter", return_value="CONTINUE: yes\nENTRY: 1\nSTOP: 2\nTARGET: 3\nREASON: x") as mock_or,
    ):
        raw, used = ch._ask_local_then_openrouter("prompt")
    assert used.startswith("openrouter:") and mock_or.called

    with patch("ai.continuation_hunter.run_ollama", return_value="CONTINUE: no\nREASON: fine as is"):
        raw, used = ch._ask_local_then_openrouter("prompt")
    assert used == config.CLERK_PRIMARY_MODEL


def test_a_local_answer_that_does_not_parse_at_all_also_escalates():
    with (
        patch("ai.continuation_hunter.run_ollama", return_value="I'm not sure what to make of this."),
        patch("ai.continuation_hunter.run_openrouter", return_value="CONTINUE: no\nREASON: no edge") as mock_or,
    ):
        raw, used = ch._ask_local_then_openrouter("prompt")
    assert mock_or.called and used.startswith("openrouter:")


# --- decision parsing --------------------------------------------------------------------------------------

def test_parse_continuation_decision_accepts_well_formed_yes():
    d = ch._parse_continuation_decision("CONTINUE: yes\nENTRY: 111.0 (fresh breakout)\nSTOP: 109.5\nTARGET: 115.0\nREASON: x")
    assert d.propose and d.entry == 111.0 and d.stop == 109.5 and d.target == 115.0


def test_parse_continuation_decision_rejects_a_yes_missing_numbers():
    d = ch._parse_continuation_decision("CONTINUE: yes\nREASON: looks good")
    assert not d.propose and "no usable entry" in d.reason


def test_parse_continuation_decision_handles_no_and_garbage():
    assert not ch._parse_continuation_decision("CONTINUE: no\nREASON: too extended").propose
    assert not ch._parse_continuation_decision("garbled nonsense with no verdict at all").propose
