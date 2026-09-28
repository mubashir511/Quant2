import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import config
from ai.ollama_client import FAILED_MESSAGE as OLLAMA_FAILED_MESSAGE
from ai.clerk_execution import (
    TacticalSignals,
    TacticalVerdict,
    _backfill_settlement_for_held_positions,
    _build_tactical_prompt,
    _compute_tactical_signals,
    _reconcile_settlement,
    _run_clerk_tactical_check,
    _validate_and_apply_tactical_verdict,
    parse_tactical_verdict,
    read_execution_state,
    read_settlement,
    read_tactical_defense_enabled,
    run_clerk_execution_check,
    set_clerk_execution_enabled,
    set_tactical_defense_enabled,
)
from ai.ftmo_suggest import FtmoAssetAnalysis
from ai.portfolio_suggest import AllocationEntry, AssetAnalysis, PendingSetup
from analysis.backtest import FavorableExcursionStats, RSIReactionBacktest
from analysis.chart_structure import (
    ChartStructureSnapshot,
    LiquiditySweepEvent,
    SRLevel,
    SRLevelsResult,
    StructureBreak,
)
from analysis.technical import TechnicalStats
from data.mt5_execution import OrderResult
from data.mt5_source import ContractSpec, Position, TradeCost


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
        patch.object(config, "CLERK_EXECUTION_MAX_TACTICAL_CANDIDATES", 10),
        patch.object(config, "AMEND_TOLERANCE_PCT", 0.05),
        patch.object(config, "CLERK_EXECUTION_ENABLED_FILE", str(tmp_path / "clerk_execution_enabled.json")),
        patch.object(config, "CLERK_EXECUTION_INTERVAL_FILE", str(tmp_path / "clerk_execution_interval.json")),
        patch.object(config, "CLERK_TACTICAL_DEFENSE_ENABLED_FILE", str(tmp_path / "clerk_tactical_defense_enabled.json")),
        patch.object(config, "CLERK_TACTICAL_DEFEND_COOLDOWN_MINUTES", 60),
        patch.object(config, "CLERK_TACTICAL_DEFEND_MIN_RETRIGGER_PCT", 0.3),
        patch.object(config, "CLERK_TACTICAL_MIN_PARTIAL_CLOSE_FRACTION", 0.10),
        patch.object(config, "CLERK_TACTICAL_MAX_PARTIAL_CLOSE_FRACTION", 0.75),
    ):
        yield tmp_path


def _position(symbol="XAUUSD", side="buy", volume=1.0, ticket=1, sl=1950.0, tp=None, price_open=2000.0, price_current=1980.0, profit=-20.0):
    return Position(
        symbol=symbol, volume=volume, side=side, price_open=price_open, price_current=price_current,
        sl=sl, profit=profit, opened_at=datetime.now(), ticket=ticket, tp=tp,
    )


def _spec(volume_min=0.01, volume_step=0.01, trade_contract_size=100.0):
    return ContractSpec(
        volume_min=volume_min, volume_step=volume_step, volume_max=100.0,
        trade_contract_size=trade_contract_size, currency_margin="USD", margin_initial=1000.0,
    )


def _fake_trade_cost(spread_pct_of_price=0.01, **overrides):
    defaults = dict(
        category="Equities I CFD", spread_pct_of_price=spread_pct_of_price,
        swap_long_pct_per_day=None, swap_short_pct_per_day=None, min_stop_distance_pct=0.0,
    )
    return TradeCost(**{**defaults, **overrides})


def _get_spec_for(specs: dict):
    return lambda symbol: specs.get(symbol)


def _signals(atr_stop_candidate=None, **overrides) -> TacticalSignals:
    defaults = dict(
        favorable_move_pct=0.0, h1_atr=None, velocity_tier=None,
        atr_stop_multiple_used=config.CLERK_TACTICAL_ATR_STOP_MULTIPLE,
        atr_stop_candidate=atr_stop_candidate, atr_stop_is_tighter_than_current=None,
        profit_lock_due=False, target_captured_pct=None, days_held=None,
        partial_profit_due=False, hard_exit_required=False,
        h1_rsi=None, h1_rsi_tier=None, h4_rsi=None, h4_rsi_tier=None,
    )
    return TacticalSignals(**{**defaults, **overrides})


def _write_suggestion(tmp_path, immediate_allocation=None, pending_setups=None, generated_utc=None):
    payload = {
        "generated_utc": generated_utc or datetime.now(timezone.utc).isoformat(),
        "immediate_allocation": immediate_allocation or {},
        "pending_setups": pending_setups or [],
    }
    Path(config.MEGA_ANALYSIS_LATEST_SUGGESTION_FILE).write_text(json.dumps(payload))


def _seed_settled(tmp_path, symbol, state, generated_utc, entry=None):
    Path(config.CLERK_EXECUTION_SETTLEMENT_FILE).write_text(json.dumps({
        "generated_utc": generated_utc,
        "settled": {
            symbol: {
                "origin": "immediate", "state": state,
                "entry": entry or {"pct": 5.0, "price": 2000.0, "stop_loss": 1950.0, "take_profit": None, "side": "buy"},
                "order_ticket": 100,
            }
        },
    }))


# --- parse_tactical_verdict: the full fail-safe branch table ---


def test_parse_tactical_verdict_hold():
    v = parse_tactical_verdict("Some reasoning.\nFINAL_VERDICT: HOLD")
    assert v.tier == "hold"


def test_parse_tactical_verdict_defend_with_stop_only():
    text = (
        "Reasoning.\nFINAL_VERDICT: DEFEND\nNEW_STOP_LOSS: 1960.0\n"
        "PARTIAL_CLOSE_FRACTION: NONE\nRULE: O'Neil - tighten stop at +15% gain\n"
        "NUMBERS: position up 16%, tightening stop from 1950 to 1960"
    )
    v = parse_tactical_verdict(text)
    assert v.tier == "defend"
    assert v.new_stop_loss == 1960.0
    assert v.partial_close_fraction is None
    assert "O'Neil" in v.rule_citation
    assert "1960" in v.numbers_citation


def test_parse_tactical_verdict_defend_with_fraction_only():
    text = "FINAL_VERDICT: DEFEND\nPARTIAL_CLOSE_FRACTION: 0.5\nRULE: Schwager - 50-60% of target fast\nNUMBERS: hit 55% of target in 2 days"
    v = parse_tactical_verdict(text)
    assert v.tier == "defend"
    assert v.new_stop_loss is None
    assert v.partial_close_fraction == 0.5


def test_parse_tactical_verdict_exit_with_citation():
    text = "FINAL_VERDICT: EXIT\nRULE: Schwager - cut size during losing streak\nNUMBERS: 3rd consecutive adverse move, down 4.2%"
    v = parse_tactical_verdict(text)
    assert v.tier == "exit"
    assert "Schwager" in v.rule_citation


def test_parse_tactical_verdict_fails_safe_on_empty_string():
    assert parse_tactical_verdict("").tier == "hold"


def test_parse_tactical_verdict_fails_safe_on_whitespace_only():
    assert parse_tactical_verdict("   \n  ").tier == "hold"


def test_parse_tactical_verdict_fails_safe_when_ollama_unavailable():
    assert parse_tactical_verdict(OLLAMA_FAILED_MESSAGE).tier == "hold"


def test_parse_tactical_verdict_fails_safe_when_no_token_present():
    assert parse_tactical_verdict("I think this position looks fine.").tier == "hold"


def test_parse_tactical_verdict_defend_missing_rule_fails_safe():
    text = "FINAL_VERDICT: DEFEND\nNEW_STOP_LOSS: 1960.0\nNUMBERS: x"
    assert parse_tactical_verdict(text).tier == "hold"


def test_parse_tactical_verdict_defend_missing_numbers_fails_safe():
    text = "FINAL_VERDICT: DEFEND\nNEW_STOP_LOSS: 1960.0\nRULE: x"
    assert parse_tactical_verdict(text).tier == "hold"


def test_parse_tactical_verdict_exit_missing_citation_fails_safe():
    assert parse_tactical_verdict("FINAL_VERDICT: EXIT").tier == "hold"


def test_parse_tactical_verdict_defend_both_fields_none_fails_safe():
    text = "FINAL_VERDICT: DEFEND\nNEW_STOP_LOSS: NONE\nPARTIAL_CLOSE_FRACTION: NONE\nRULE: x\nNUMBERS: y"
    assert parse_tactical_verdict(text).tier == "hold"


def test_parse_tactical_verdict_defend_fraction_above_max_rejected():
    text = "FINAL_VERDICT: DEFEND\nPARTIAL_CLOSE_FRACTION: 0.95\nRULE: x\nNUMBERS: y"
    assert parse_tactical_verdict(text).tier == "hold"


def test_parse_tactical_verdict_defend_fraction_below_min_rejected():
    text = "FINAL_VERDICT: DEFEND\nPARTIAL_CLOSE_FRACTION: 0.05\nRULE: x\nNUMBERS: y"
    assert parse_tactical_verdict(text).tier == "hold"


def test_parse_tactical_verdict_defend_fraction_within_bounds_accepted():
    text = "FINAL_VERDICT: DEFEND\nPARTIAL_CLOSE_FRACTION: 0.5\nRULE: x\nNUMBERS: y"
    v = parse_tactical_verdict(text)
    assert v.tier == "defend"
    assert v.partial_close_fraction == 0.5


def test_parse_tactical_verdict_last_occurrence_wins():
    text = "FINAL_VERDICT: DEFEND\nRULE: x\nNUMBERS: y\n\nActually, on reflection, FINAL_VERDICT: HOLD"
    assert parse_tactical_verdict(text).tier == "hold"


def test_parse_tactical_verdict_case_insensitive():
    text = "final_verdict: exit\nrule: x\nnumbers: y"
    assert parse_tactical_verdict(text).tier == "exit"


# --- _validate_and_apply_tactical_verdict: the deterministic guardrail layer ---


def test_validate_hold_makes_no_change():
    verdict = TacticalVerdict(tier="hold")
    position = _position(side="buy", volume=1.0, sl=1950.0)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({}),
    )
    assert new_entry is None
    assert tactical_state is None
    assert rejected_reason == ""  # HOLD is never a "rejection" -- it's a legitimate verdict


def test_validate_hold_still_persists_a_changed_trend_flip_count():
    # Real gap this closes: the NVDA incident was dozens of consecutive
    # HOLD verdicts -- without this, the counter would never advance
    # across a HOLD-heavy streak.
    verdict = TacticalVerdict(tier="hold")
    position = _position(side="buy", volume=1.0, sl=1950.0)
    signals = _signals(trend_flip_against_count=3)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, {"trend_flip_against_count": 2}, 100_000.0, _get_spec_for({}),
        signals=signals,
    )
    assert new_entry is None  # HOLD never touches the allocation
    assert tactical_state == {"trend_flip_against_count": 3}
    assert rejected_reason == ""


def test_validate_hold_persists_nothing_when_the_count_is_unchanged():
    verdict = TacticalVerdict(tier="hold")
    position = _position(side="buy", volume=1.0, sl=1950.0)
    signals = _signals(trend_flip_against_count=2)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, {"trend_flip_against_count": 2}, 100_000.0, _get_spec_for({}),
        signals=signals,
    )
    assert new_entry is None
    assert tactical_state is None


def test_validate_exit_forces_pct_zero_and_cites_the_verdict():
    verdict = TacticalVerdict(tier="exit", rule_citation="O'Neil", numbers_citation="stop hit at 1950")
    position = _position(side="buy", volume=1.0, sl=1950.0)
    existing = AllocationEntry(pct=5.0, price=2000.0, stop_loss=1950.0, take_profit=2100.0, side="buy", reason="orig")
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, None, 100_000.0, _get_spec_for({}),
    )
    assert new_entry.pct == 0.0
    assert "Tactical EXIT" in new_entry.reason
    assert "O'Neil" in new_entry.reason
    assert tactical_state["tier_reached"] == "exit"
    assert tactical_state["trend_flip_against_count"] == 0  # no signals given -- degrades to prior (absent) value
    assert rejected_reason == ""  # a successful EXIT is not a rejection


def test_validate_exit_persists_the_fresh_trend_flip_count_from_signals():
    verdict = TacticalVerdict(tier="exit", rule_citation="O'Neil", numbers_citation="stop hit at 1950")
    position = _position(side="buy", volume=1.0, sl=1950.0)
    existing = AllocationEntry(pct=5.0, price=2000.0, stop_loss=1950.0, take_profit=2100.0, side="buy", reason="orig")
    signals = _signals(trend_flip_against_count=6)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, {"trend_flip_against_count": 5}, 100_000.0, _get_spec_for({}),
        signals=signals,
    )
    assert tactical_state["trend_flip_against_count"] == 6


def test_validate_defend_tightens_stop_and_resizes_pct_to_preserve_lots():
    # 1.0 lot held; tightening the stop from 1950 to 1960 while keeping
    # the SAME 1.0 lot requires a smaller pct than the original (smaller
    # stop distance / same lots = less risk-%) -- the exact correctness
    # trap pct_for_target_lots exists to close. Sizing uses the position's
    # own real, fixed price_open (2000), NOT the live price_current (1980)
    # -- real coupling bug found live 2026-09-08: compute_rebalance_plan's
    # own sizing for an ALREADY-HELD position anchors to price_open (see
    # its own "sizing_price" comment for the incident this fixes: a live-
    # tracking anchor made an EURUSD position's implied lot size thrash
    # every poll from ordinary tick noise, opening/closing 9 separate
    # tickets in 3.5 hours), so this round-trip must use the SAME anchor
    # or a stop-tighten would silently imply a different lot count than
    # intended and get misread as a real reallocation.
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1960.0, rule_citation="Schwager", numbers_citation="x")
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1980.0)
    existing = AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0, take_profit=2100.0, side="buy")
    spec = _spec(trade_contract_size=100.0)
    new_entry, tactical_state, _reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, None, 100_000.0, _get_spec_for({"XAUUSD": spec}),
    )
    assert new_entry is not None
    assert new_entry.stop_loss == 1960.0
    assert new_entry.price == 2000.0
    assert new_entry.take_profit == 2100.0  # DEFEND never touches the target
    # stop distance (2000-1960=40) x 1.0 lot x contract_size 100 = $4,000 = 4% of 100k
    assert new_entry.pct == pytest.approx(4.0)
    assert "Tactical DEFEND" in new_entry.reason
    assert tactical_state["tier_reached"] == "defend"
    assert tactical_state["defend_count"] == 1
    # Real bug found live 2026-09-03: without persisting the resolved
    # size/stop/target here, the next poll's carried-forward baseline
    # would rebuild straight from the mega session's own unreduced
    # original target and buy the reduction right back.
    assert tactical_state["persisted_pct"] == pytest.approx(4.0)
    assert tactical_state["persisted_stop_loss"] == 1960.0
    assert tactical_state["persisted_take_profit"] == 2100.0


def test_validate_defend_preserves_the_trend_flip_count_from_signals():
    # Real gap this closes: this dict does NOT spread prior_tactical the
    # way the EXIT branch does -- without an explicit line for it, a
    # real DEFEND would silently drop an already-accumulating trend-flip
    # streak back to invisible (read as 0 next poll).
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1960.0, rule_citation="Schwager", numbers_citation="x")
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1980.0)
    existing = AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0, take_profit=2100.0, side="buy")
    spec = _spec(trade_contract_size=100.0)
    signals = _signals(trend_flip_against_count=2)
    new_entry, tactical_state, _reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, {"trend_flip_against_count": 1}, 100_000.0,
        _get_spec_for({"XAUUSD": spec}), signals=signals,
    )
    assert tactical_state["trend_flip_against_count"] == 2


def test_validate_defend_uses_price_open_not_existing_entry_or_live_price_for_sizing():
    # The real bug this now closes (found live 2026-09-08, reversing this
    # test's own former assumption): sizing must use the position's own
    # real, fixed price_open, NEITHER existing_entry.price (the mega
    # session's own possibly-stale suggestion) NOR position.price_current
    # (a live-tracking value) -- compute_rebalance_plan's own sizing for
    # an ALREADY-HELD position anchors to price_open (see its own
    # "sizing_price" comment): a live-tracking anchor made an EURUSD
    # position's implied lot size thrash every poll from ordinary tick
    # noise alone, opening/closing 9 separate tickets in 3.5 hours. This
    # round-trip must match that SAME anchor or a stop-tighten would
    # silently imply the wrong lot count. Three genuinely different
    # prices here (existing.price=1500, position.price_open=2000,
    # position.price_current=1700) make it unambiguous which one wins.
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1650.0, rule_citation="Schwager", numbers_citation="x")
    position = _position(side="buy", volume=2.0, sl=1600.0, price_open=2000.0, price_current=1700.0)
    existing = AllocationEntry(pct=10.0, price=1500.0, stop_loss=1600.0, side="buy")
    spec = _spec(trade_contract_size=100.0)
    new_entry, _, _reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, None, 100_000.0, _get_spec_for({"XAUUSD": spec}),
    )
    assert new_entry is not None
    assert new_entry.price == 2000.0  # price_open -- neither the stale 1500 nor the live 1700
    # (2000-1650=350) x 2.0 lots x 100 contract size = $70,000 = 70% of 100k
    assert new_entry.pct == pytest.approx(70.0)


def test_validate_defend_rejects_a_stop_placed_past_the_live_price_for_a_buy():
    # "More protective than the OLD stop" alone doesn't rule out a stop
    # placed past the CURRENT price -- an over-eager tighten proposing
    # 1960 while the live price has already fallen to 1950 would not be
    # a valid protective stop at all (it sits on the wrong side of the
    # market). Must be rejected explicitly, not merely "more protective."
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1960.0, rule_citation="x", numbers_citation="y")
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1955.0)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert new_entry is None
    assert tactical_state is None
    assert "wrong side" in rejected_reason


def test_validate_defend_rejects_a_stop_placed_past_the_live_price_for_a_sell():
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1990.0, rule_citation="x", numbers_citation="y")
    position = _position(side="sell", volume=1.0, sl=2000.0, price_open=2000.0, price_current=1995.0)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert new_entry is None
    assert tactical_state is None
    assert "wrong side" in rejected_reason


def test_validate_defend_substitutes_atr_fallback_when_proposed_stop_is_on_the_wrong_side():
    # Real, repeated incident (2026-09-09/10): the local tactical model
    # proposed a SELL's stop below the live price six times in one real
    # session -- every DEFEND discarded outright, including its own
    # correct underlying judgment that the position needed defending.
    # signals.atr_stop_candidate (always valid by construction) must now
    # be substituted instead of discarding the whole DEFEND.
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1990.0, rule_citation="x", numbers_citation="y")
    position = _position(side="sell", volume=1.0, sl=2000.0, price_open=2000.0, price_current=1995.0)
    existing = AllocationEntry(pct=5.0, price=2000.0, stop_loss=2000.0, side="sell")
    signals = _signals(atr_stop_candidate=1998.0)  # above 1995 (sane side), below 2000 (more protective)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
        signals=signals,
    )
    assert new_entry is not None
    assert new_entry.stop_loss == 1998.0  # the fallback, NOT the model's own invalid 1990.0
    assert rejected_reason == ""
    assert tactical_state["tier_reached"] == "defend"


def test_validate_defend_substitutes_atr_fallback_when_proposed_stop_widens_the_current_one():
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1900.0, rule_citation="x", numbers_citation="y")
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1980.0)
    existing = AllocationEntry(pct=5.0, price=2000.0, stop_loss=1950.0, side="buy")
    signals = _signals(atr_stop_candidate=1960.0)  # tighter than 1950, below live price 1980
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
        signals=signals,
    )
    assert new_entry is not None
    assert new_entry.stop_loss == 1960.0
    assert rejected_reason == ""


def test_validate_defend_still_rejects_when_the_fallback_stop_is_also_invalid():
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1990.0, rule_citation="x", numbers_citation="y")
    position = _position(side="sell", volume=1.0, sl=2000.0, price_open=2000.0, price_current=1995.0)
    signals = _signals(atr_stop_candidate=1992.0)  # ALSO below the 1995 live price -- also wrong side
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
        signals=signals,
    )
    assert new_entry is None
    assert tactical_state is None
    assert "wrong side" in rejected_reason
    assert "no valid ATR-based fallback" in rejected_reason


def test_validate_defend_no_fallback_available_rejects_exactly_as_before():
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1990.0, rule_citation="x", numbers_citation="y")
    position = _position(side="sell", volume=1.0, sl=2000.0, price_open=2000.0, price_current=1995.0)
    signals = _signals(atr_stop_candidate=None)  # no H1 ATR reading available -- no fallback possible
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
        signals=signals,
    )
    assert new_entry is None
    assert "wrong side" in rejected_reason
    assert "no valid ATR-based fallback" in rejected_reason


def test_validate_defend_rejects_as_before_when_no_signals_given():
    # Backward-compatible default: signals=None (every existing caller
    # before this fix, and every other test in this file) behaves
    # exactly like it did before this fallback existed.
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1990.0, rule_citation="x", numbers_citation="y")
    position = _position(side="sell", volume=1.0, sl=2000.0, price_open=2000.0, price_current=1995.0)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert new_entry is None
    assert "wrong side" in rejected_reason


def test_validate_defend_rejects_a_widened_stop_for_a_buy():
    # 1900 is LESS protective than the current 1950 stop for a buy.
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1900.0, rule_citation="x", numbers_citation="y")
    position = _position(side="buy", volume=1.0, sl=1950.0)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert new_entry is None
    assert tactical_state is None
    assert "never-widen-stop" in rejected_reason


def test_validate_defend_rejects_a_widened_stop_for_a_sell():
    # For a sell, a HIGHER stop is less protective.
    verdict = TacticalVerdict(tier="defend", new_stop_loss=2050.0, rule_citation="x", numbers_citation="y")
    position = _position(side="sell", volume=1.0, sl=2000.0, price_open=2000.0, price_current=1980.0)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert new_entry is None
    assert tactical_state is None
    assert "never-widen-stop" in rejected_reason


def test_validate_defend_accepts_a_tighter_stop_for_a_sell():
    # For a sell, a LOWER stop is more protective.
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1990.0, rule_citation="x", numbers_citation="y")
    position = _position(side="sell", volume=1.0, sl=2000.0, price_open=2000.0, price_current=1985.0)
    existing = AllocationEntry(pct=5.0, price=2000.0, stop_loss=2000.0, side="sell")
    new_entry, _, _reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert new_entry is not None
    assert new_entry.stop_loss == 1990.0


def test_validate_defend_rejected_when_no_current_or_proposed_stop():
    verdict = TacticalVerdict(tier="defend", partial_close_fraction=0.5, rule_citation="x", numbers_citation="y")
    position = _position(side="buy", volume=1.0, sl=None)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert new_entry is None
    assert tactical_state is None
    assert rejected_reason != ""


def test_validate_defend_partial_close_resizes_pct_for_remaining_lots():
    # A 40% partial close (fraction=0.4) on 1.0 held lot targets 0.6
    # lots remaining, at the SAME (unchanged) stop. Sizing anchors to
    # the position's own real price_open (2000), not its live
    # price_current -- see the "sizing_price"/"sizing_entry_price"
    # comments this round-trip must match.
    verdict = TacticalVerdict(tier="defend", partial_close_fraction=0.4, rule_citation="Schwager", numbers_citation="x")
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=2020.0)  # in profit: a partial close takes profit
    existing = AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0, side="buy")
    spec = _spec(trade_contract_size=100.0)
    new_entry, _, _reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, None, 100_000.0, _get_spec_for({"XAUUSD": spec}),
    )
    assert new_entry is not None
    assert new_entry.stop_loss == 1950.0  # unchanged -- no new stop proposed
    # 0.6 lots x (2000-1950=50)-point stop x 100 contract size = $3,000 = 3% of 100k
    assert new_entry.pct == pytest.approx(3.0)


def test_validate_defend_suppressed_within_cooldown_without_further_deterioration():
    now = datetime.now(timezone.utc)
    prior_tactical = {"last_action_utc": now.isoformat(), "adverse_move_pct_at_last_action": 1.0, "defend_count": 1}
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1960.0, rule_citation="x", numbers_citation="y")
    # adverse_move_pct barely above 1.0 (well under the 0.3-point min retrigger)
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1998.0)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, prior_tactical, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert new_entry is None
    assert tactical_state is None
    assert "cooldown" in rejected_reason


def test_validate_defend_fires_within_cooldown_when_genuinely_worsened():
    now = datetime.now(timezone.utc)
    prior_tactical = {"last_action_utc": now.isoformat(), "adverse_move_pct_at_last_action": 0.0, "defend_count": 1}
    # new_stop (1955) sits between the old stop (1950) and the live price
    # (1965) -- a genuinely valid tighter stop, not past the market.
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1955.0, rule_citation="x", numbers_citation="y")
    # adverse_move_pct ~ (2000-1965)/2000*100 = 1.75%, well above the 0.3 minimum re-trigger
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1965.0)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, prior_tactical, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert new_entry is not None
    assert tactical_state["defend_count"] == 2
    assert rejected_reason == ""


def test_validate_defend_hard_exit_bypasses_the_cooldown_gate_entirely():
    # Real bug found on self-audit (2026-09-16), before the trend-flip
    # partial-reduction verdict ever ran live: this is the EXACT scenario
    # from test_validate_defend_suppressed_within_cooldown_without_
    # further_deterioration above (a real, unrelated DEFEND fired
    # moments ago, position hasn't worsened enough to normally re-fire)
    # -- but with hard_exit=True (the trend-flip circuit-breaker's own
    # flag), it must NOT be suppressed, and it must still persist
    # tactical_state (a suppressed DEFEND persists nothing, which would
    # otherwise freeze trend_flip_against_count's own counter forever).
    now = datetime.now(timezone.utc)
    prior_tactical = {
        "last_action_utc": now.isoformat(), "adverse_move_pct_at_last_action": 1.0, "defend_count": 1,
        "trend_flip_against_count": 3,
    }
    verdict = TacticalVerdict(
        tier="defend", partial_close_fraction=0.5, rule_citation="x", numbers_citation="y", hard_exit=True,
    )
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1998.0)
    signals = _signals(trend_flip_against_count=3)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, prior_tactical, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
        signals=signals,
    )
    assert new_entry is not None  # NOT suppressed by cooldown
    assert rejected_reason == ""
    assert tactical_state["trend_flip_against_count"] == 3  # persisted, not frozen


def test_validate_defend_fires_after_cooldown_elapses_even_without_deterioration():
    long_ago = datetime.now(timezone.utc) - timedelta(minutes=120)
    prior_tactical = {"last_action_utc": long_ago.isoformat(), "adverse_move_pct_at_last_action": 5.0, "defend_count": 1}
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1960.0, rule_citation="x", numbers_citation="y")
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1998.0)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, prior_tactical, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert new_entry is not None
    assert tactical_state["defend_count"] == 2
    assert rejected_reason == ""


def test_validate_defend_rejected_when_sizing_fails():
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1960.0, rule_citation="x", numbers_citation="y")
    position = _position(side="buy", volume=1.0, sl=1950.0)
    new_entry, tactical_state, rejected_reason = _validate_and_apply_tactical_verdict(
        "UNKNOWN", verdict, position, None, None, 100_000.0, _get_spec_for({}),
    )
    assert new_entry is None
    assert tactical_state is None
    assert "sized" in rejected_reason


# --- _backfill_settlement_for_held_positions: the real gap this closes ---
# (_reset_settlement_for_new_session wipes ALL tracking on every new mega
# session, including "filled" records, and neither "hold" nor "amend_
# position" ever re-creates one -- without this backfill, an already-held
# position the fresh mega session simply reaffirms unchanged would be
# invisible to the tactical-defense check for the rest of that cycle.)


def test_backfill_creates_filled_record_from_fresh_immediate_allocation():
    positions = [_position(symbol="XAUUSD", side="buy", volume=1.0, sl=1950.0)]
    raw = {"pct": 5.0, "price": 2000.0, "stop_loss": 1950.0, "take_profit": 2100.0, "side": "buy", "reason": "Gold long"}
    backfilled = _backfill_settlement_for_held_positions(
        {}, positions, {"XAUUSD": raw}, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert backfilled["XAUUSD"]["state"] == "filled"
    assert backfilled["XAUUSD"]["origin"] == "immediate"
    assert backfilled["XAUUSD"]["entry"]["pct"] == 5.0
    assert backfilled["XAUUSD"]["entry"]["stop_loss"] == 1950.0


def test_backfill_reproduces_held_size_when_not_in_fresh_suggestion():
    # A Pending-Setup-originated position the fresh mega session doesn't
    # mention at all -- pct must be back-computed to EXACTLY reproduce
    # the position's own already-held 2.0 lots, so compute_rebalance_
    # plan resolves this to "hold," not an accidental resize. Sizing
    # uses the position's own fixed price_open (2000), not its live
    # price_current (1980, _position's own default) -- must match
    # compute_rebalance_plan's own "sizing_price" anchor for an already-
    # held position (see that function's own comment for the real
    # 2026-09-08 incident this fixes), or the round-trip breaks.
    positions = [_position(symbol="XAUUSD", side="buy", volume=2.0, sl=1950.0, price_open=2000.0)]
    spec = _spec(trade_contract_size=100.0)
    backfilled = _backfill_settlement_for_held_positions(
        {}, positions, {}, 100_000.0, _get_spec_for({"XAUUSD": spec}),
    )
    assert backfilled["XAUUSD"]["state"] == "filled"
    assert backfilled["XAUUSD"]["origin"] == "pending_setup"
    assert backfilled["XAUUSD"]["entry"]["price"] == 2000.0
    # 2.0 lots x (2000-1950=50) stop x 100 contract size = $10,000 = 10% of 100k
    assert backfilled["XAUUSD"]["entry"]["pct"] == pytest.approx(10.0)


def test_backfill_skips_symbol_when_sizing_is_not_possible():
    positions = [_position(symbol="XAUUSD", side="buy", volume=1.0, sl=None)]
    backfilled = _backfill_settlement_for_held_positions(
        {}, positions, {}, 100_000.0, _get_spec_for({}),
    )
    assert "XAUUSD" not in backfilled


def test_backfill_never_overwrites_an_existing_record():
    positions = [_position(symbol="XAUUSD", side="buy", volume=1.0, sl=1950.0)]
    existing = {"XAUUSD": {"origin": "immediate", "state": "filled", "entry": {"pct": 99.0}, "order_ticket": None}}
    backfilled = _backfill_settlement_for_held_positions(
        existing, positions, {"XAUUSD": {"pct": 5.0}}, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert backfilled["XAUUSD"]["entry"]["pct"] == 99.0


def test_backfill_ignores_cash():
    positions = [_position(symbol="XAUUSD", side="buy", volume=1.0, sl=1950.0)]
    backfilled = _backfill_settlement_for_held_positions(
        {}, positions, {"CASH": {"pct": 50.0}}, 100_000.0, _get_spec_for({}),
    )
    assert "CASH" not in backfilled


def test_reconcile_then_backfill_survives_a_mega_session_reset_for_an_unchanged_position():
    # The exact real-world sequence: a symbol was "filled" under the OLD
    # mega session; a brand-new session resets settlement to {} (see
    # _reset_settlement_for_new_session); on the very next poll,
    # _reconcile_settlement alone leaves it with NO record at all (it
    # only advances EXISTING records) -- _backfill_settlement_for_held_
    # positions must restore it in the same poll.
    positions = [_position(symbol="XAUUSD", side="buy", volume=1.0, sl=1950.0)]
    reconciled = _reconcile_settlement({}, positions, [])
    assert "XAUUSD" not in reconciled  # confirms reconcile alone does NOT fix this
    raw = {"pct": 5.0, "price": 2000.0, "stop_loss": 1950.0, "side": "buy", "reason": "Gold long"}
    backfilled = _backfill_settlement_for_held_positions(
        reconciled, positions, {"XAUUSD": raw}, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert backfilled["XAUUSD"]["state"] == "filled"


# --- enable/disable toggle: opt-in, defaults False ---


def test_tactical_defense_defaults_disabled(_fixed_files):
    assert read_tactical_defense_enabled() is False


def test_tactical_defense_enable_disable_round_trip(_fixed_files):
    set_tactical_defense_enabled(True)
    assert read_tactical_defense_enabled() is True
    set_tactical_defense_enabled(False)
    assert read_tactical_defense_enabled() is False


# --- full-poll integration: shadow mode, live application, and the collision rule ---


def _account():
    from data.mt5_source import AccountSummary
    return AccountSummary(balance=100_000.0, equity=100_000.0, free_margin=90_000.0, currency="USD")


def _ftmo_status():
    from risk.ftmo_rules import FtmoStatus
    # Generous headroom -- these tests are about the tactical-defense
    # mechanism, not the (separately, fully tested) FTMO heat gate.
    return FtmoStatus(
        daily_loss_limit_pct=3.0, today_realized_pl=0.0, today_floating_pl=0.0, today_total_pl=0.0,
        daily_loss_headroom_pct=30.0, trailing_max_loss_floor=90_000.0, max_loss_headroom_pct=10.0,
        best_day_pl=None, total_positive_days_pl=None, best_day_rule_pct=None,
    )


@patch("ai.clerk_execution._run_clerk_tactical_check")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.modify_position_sltp")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_tactical_defend_shadow_mode_does_not_apply_when_disabled(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_amend, mock_fetch_ctx, mock_tactical, _fixed_files,
):
    from data.mt5_source import MarketAsset

    mock_account.return_value = _account()
    mock_positions.return_value = [_position(symbol="XAUUSD", side="buy", volume=1.0, ticket=300, sl=1950.0)]
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=1979.0, ask=1980.0)]
    mock_status.return_value = _ftmo_status()
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1960.0, rule_citation="Schwager", numbers_citation="x")
    mock_tactical.return_value = ("XAUUSD", verdict, "FINAL_VERDICT: DEFEND")

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {"pct": 5.0, "price": 2000.0, "stop_loss": 1950.0, "take_profit": None, "side": "buy", "reason": "Gold long"}
    _write_suggestion(_fixed_files, immediate_allocation={"XAUUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "XAUUSD", "filled", generated_utc, entry=entry)
    # read_tactical_defense_enabled() defaults False -- no explicit set call.

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = _spec()
        run_clerk_execution_check()

    mock_amend.assert_not_called()
    state = read_execution_state()
    tactical = state["last_tactical_verdicts"]["XAUUSD"]
    assert tactical["tier"] == "defend"
    assert tactical["applied"] is False
    assert tactical["shadow_mode"] is True
    # Shadow mode must leave EVERY piece of real state untouched, not
    # just skip the order -- the settlement record's own tactical
    # history (defend_count/last_stop_loss/etc.) must also stay
    # unwritten, or a later real enable would start from a phantom
    # cooldown/escalation history that never actually happened.
    settlement = read_settlement()
    assert "tactical" not in settlement["settled"]["XAUUSD"]


@patch("ai.clerk_execution._run_clerk_tactical_check")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_hard_exit_bypasses_shadow_mode_even_when_tactical_defense_disabled(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_close, mock_fetch_ctx, mock_tactical, _fixed_files,
):
    # Mirror image of test_tactical_defend_shadow_mode_does_not_apply_
    # when_disabled above: a verdict.hard_exit=True verdict is O'Neil's
    # "no exceptions" hard stop-loss ceiling -- a genuine circuit-
    # breaker, same category as the FTMO compliance safety gates, which
    # already apply unconditionally. It must NOT sit inert in shadow mode
    # the way an ordinary discretionary DEFEND/EXIT does -- confirms the
    # merge loop's `if tactical_enabled or verdict.hard_exit:` fix.
    from data.mt5_source import MarketAsset

    mock_account.return_value = _account()
    mock_positions.return_value = [_position(symbol="XAUUSD", side="buy", volume=1.0, ticket=300, sl=1950.0, price_open=2000.0, price_current=1850.0)]
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=1849.0, ask=1850.0)]
    mock_status.return_value = _ftmo_status()
    mock_close.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=300)
    verdict = TacticalVerdict(
        tier="exit", rule_citation="William O'Neil (How to Make Money in Stocks)",
        numbers_citation="Adverse move 7.50% at/past the 7.0% hard-stop ceiling — no exceptions.",
        hard_exit=True,
    )
    mock_tactical.return_value = ("XAUUSD", verdict, "[deterministic circuit-breaker — no model call made]")

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {"pct": 5.0, "price": 2000.0, "stop_loss": 1950.0, "take_profit": None, "side": "buy", "reason": "Gold long"}
    _write_suggestion(_fixed_files, immediate_allocation={"XAUUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "XAUUSD", "filled", generated_utc, entry=entry)
    # read_tactical_defense_enabled() defaults False -- no explicit set call,
    # exactly the case that leaves an ordinary DEFEND/EXIT in shadow mode.

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = _spec()
        run_clerk_execution_check()

    mock_close.assert_called_once()
    state = read_execution_state()
    tactical = state["last_tactical_verdicts"]["XAUUSD"]
    assert tactical["tier"] == "exit"
    assert tactical["applied"] is True
    assert tactical["shadow_mode"] is False


@patch("ai.clerk_execution._run_clerk_tactical_check")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.modify_position_sltp")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_tactical_defend_applies_via_amend_position_when_enabled(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_amend, mock_fetch_ctx, mock_tactical, _fixed_files,
):
    from data.mt5_source import MarketAsset

    mock_account.return_value = _account()
    mock_positions.return_value = [_position(symbol="XAUUSD", side="buy", volume=1.0, ticket=300, sl=1950.0, price_open=2000.0, price_current=1980.0)]
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=1979.0, ask=1980.0)]
    mock_status.return_value = _ftmo_status()
    mock_amend.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=300)
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1960.0, rule_citation="Schwager", numbers_citation="up 16%, tighten 1950->1960")
    mock_tactical.return_value = ("XAUUSD", verdict, "FINAL_VERDICT: DEFEND")

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {"pct": 5.0, "price": 2000.0, "stop_loss": 1950.0, "take_profit": None, "side": "buy", "reason": "Gold long"}
    _write_suggestion(_fixed_files, immediate_allocation={"XAUUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "XAUUSD", "filled", generated_utc, entry=entry)
    set_tactical_defense_enabled(True)

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = _spec(trade_contract_size=100.0)
        run_clerk_execution_check()

    mock_amend.assert_called_once()
    _called_position, called_sl, _called_tp = mock_amend.call_args[0]
    assert called_sl == 1960.0

    state = read_execution_state()
    assert state["last_tactical_verdicts"]["XAUUSD"]["applied"] is True
    settlement = read_settlement()
    assert settlement["settled"]["XAUUSD"]["tactical"]["tier_reached"] == "defend"
    assert settlement["settled"]["XAUUSD"]["tactical"]["defend_count"] == 1


@patch("ai.clerk_execution._run_clerk_tactical_check")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.modify_position_sltp")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_tactical_defend_rejection_is_distinguishable_from_a_pending_shadow_action(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_amend, mock_fetch_ctx, mock_tactical, _fixed_files,
):
    # Found on audit: even with tactical defense ENABLED, a DEFEND that
    # the guardrail genuinely REJECTS (here: proposed stop 1900 widens
    # the current 1950 stop) must never be silently indistinguishable
    # from a valid action merely waiting on the toggle -- both look like
    # "not applied" without the fix, but only one of them will EVER fire.
    from data.mt5_source import MarketAsset

    mock_account.return_value = _account()
    mock_positions.return_value = [_position(symbol="XAUUSD", side="buy", volume=1.0, ticket=300, sl=1950.0)]
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=1979.0, ask=1980.0)]
    mock_status.return_value = _ftmo_status()
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1900.0, rule_citation="Schwager", numbers_citation="x")
    mock_tactical.return_value = ("XAUUSD", verdict, "FINAL_VERDICT: DEFEND")

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {"pct": 5.0, "price": 2000.0, "stop_loss": 1950.0, "take_profit": None, "side": "buy", "reason": "Gold long"}
    _write_suggestion(_fixed_files, immediate_allocation={"XAUUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "XAUUSD", "filled", generated_utc, entry=entry)
    set_tactical_defense_enabled(True)  # enabled -- yet this must still never fire

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = _spec(trade_contract_size=100.0)
        run_clerk_execution_check()

    mock_amend.assert_not_called()
    state = read_execution_state()
    tactical = state["last_tactical_verdicts"]["XAUUSD"]
    assert tactical["applied"] is False
    assert tactical["shadow_mode"] is False  # NOT shadow mode -- it was genuinely rejected
    assert "never-widen-stop" in tactical["skipped_reason"]


@patch("ai.clerk_execution._run_clerk_tactical_check")
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
def test_tactical_check_survives_a_new_mega_session_reset_for_an_unchanged_hold(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_fetch_ctx, mock_tactical, _fixed_files,
):
    # The end-to-end proof of the backfill fix: settlement was tracking
    # XAUUSD as "filled" under an OLD mega session. A BRAND-NEW session
    # arrives (different generated_utc) that reaffirms the EXACT same
    # pct/price/stop -- compute_rebalance_plan will resolve this to
    # "hold" (no order placed at all), which -- before the backfill fix
    # -- would leave XAUUSD with NO settlement record and therefore
    # invisible to the tactical check for the rest of this new cycle.
    from data.mt5_source import MarketAsset

    mock_account.return_value = _account()
    mock_positions.return_value = [_position(symbol="XAUUSD", side="buy", volume=1.0, ticket=300, sl=1950.0, price_open=2000.0, price_current=1980.0)]
    mock_watch.return_value = [MarketAsset(symbol="XAUUSD", description="Gold", bid=1979.0, ask=1980.0)]
    mock_status.return_value = _ftmo_status()
    mock_tactical.return_value = ("XAUUSD", TacticalVerdict(tier="hold"), "FINAL_VERDICT: HOLD")

    old_generated_utc = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    new_generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {"pct": 5.0, "price": 2000.0, "stop_loss": 1950.0, "take_profit": None, "side": "buy", "reason": "Gold long, still holds"}
    # Fresh suggestion under a NEW generated_utc, same terms as before.
    _write_suggestion(_fixed_files, immediate_allocation={"XAUUSD": entry}, generated_utc=new_generated_utc)
    # But the settlement file is still stamped with the OLD generated_utc
    # -- this is exactly what triggers _reset_settlement_for_new_session.
    _seed_settled(_fixed_files, "XAUUSD", "filled", old_generated_utc, entry=entry)

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = _spec(trade_contract_size=100.0)
        run_clerk_execution_check()

    mock_tactical.assert_called_once()  # the real proof: it was checked at all, this same poll
    state = read_execution_state()
    assert state["last_tactical_verdicts"]["XAUUSD"]["tier"] == "hold"
    settlement = read_settlement()
    assert settlement["settled"]["XAUUSD"]["state"] == "filled"


@patch("ai.clerk_execution.is_symbol_tradable_now", return_value=True)
@patch("ai.clerk_execution._run_clerk_tactical_check")
@patch("ai.clerk_execution._run_clerk_verdict")
@patch("ai.clerk_execution._run_clerk_invalidation_check")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.modify_position_sltp")
@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.open_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_pending_setup_invalidation_and_tactical_defend_all_fire_together(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_close, mock_amend, mock_fetch_ctx,
    mock_invalidation, mock_verdict, mock_tactical, mock_tradable, _fixed_files,
):
    from data.mt5_source import MarketAsset

    mock_account.return_value = _account()
    mock_positions.return_value = [
        _position(symbol="EURUSD", side="buy", volume=1.0, ticket=200, sl=1.0850, price_open=1.0900, price_current=1.0899),
        _position(symbol="XAUUSD", side="buy", volume=1.0, ticket=300, sl=1950.0, price_open=2000.0, price_current=1980.0),
    ]
    mock_watch.return_value = [
        MarketAsset(symbol="GBPUSD", description="Cable", bid=1.2699, ask=1.2700),
        MarketAsset(symbol="EURUSD", description="Euro", bid=1.0899, ask=1.0900),
        MarketAsset(symbol="XAUUSD", description="Gold", bid=1979.0, ask=1980.0),
    ]
    mock_status.return_value = _ftmo_status()
    mock_open.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=777)
    mock_close.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=200)
    mock_amend.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=300)

    setup = PendingSetup(symbol="GBPUSD", side="buy", pct=1.0, trigger_condition="cond", price=1.2700, stop_loss=1.2600, take_profit=1.2900)
    mock_verdict.return_value = (setup, True, "FINAL_VERDICT: CONFIRMED")
    mock_invalidation.return_value = ("EURUSD", True, "FINAL_VERDICT: CONFIRMED")
    # EURUSD is ALSO a tactical candidate (a filled position, no
    # invalidation_condition gate on that list) -- keep its own tactical
    # check a plain HOLD so this test isolates one DEFEND (XAUUSD) from
    # one invalidation CONFIRMED (EURUSD), each on its own symbol.
    tactical_verdict = TacticalVerdict(tier="defend", new_stop_loss=1960.0, rule_citation="Schwager", numbers_citation="x")

    def _tactical_side_effect(symbol, entry, position, technical_context, account_equity, prior_tactical, signals):
        if symbol == "XAUUSD":
            return "XAUUSD", tactical_verdict, "FINAL_VERDICT: DEFEND"
        return symbol, TacticalVerdict(tier="hold"), "FINAL_VERDICT: HOLD"

    mock_tactical.side_effect = _tactical_side_effect

    generated_utc = datetime.now(timezone.utc).isoformat()
    eurusd_entry = {
        "pct": 1.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None,
        "side": "buy", "reason": "r", "invalidation_condition": "H1 closes below 1.0800",
    }
    xauusd_entry = {"pct": 5.0, "price": 2000.0, "stop_loss": 1950.0, "take_profit": None, "side": "buy", "reason": "Gold long"}
    _write_suggestion(
        _fixed_files,
        immediate_allocation={"EURUSD": eurusd_entry, "XAUUSD": xauusd_entry},
        pending_setups=[{"symbol": "GBPUSD", "side": "buy", "pct": 1.0, "trigger_condition": "cond",
                          "price": 1.2700, "stop_loss": 1.2600, "take_profit": 1.2900, "reason": "r"}],
        generated_utc=generated_utc,
    )
    Path(config.CLERK_EXECUTION_SETTLEMENT_FILE).write_text(json.dumps({
        "generated_utc": generated_utc,
        "settled": {
            "EURUSD": {"origin": "immediate", "state": "filled", "entry": eurusd_entry, "order_ticket": 200},
            "XAUUSD": {"origin": "immediate", "state": "filled", "entry": xauusd_entry, "order_ticket": 300},
        },
    }))
    set_tactical_defense_enabled(True)

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.side_effect = lambda symbol: {
            "GBPUSD": _spec(trade_contract_size=100_000.0),
            "EURUSD": _spec(trade_contract_size=100_000.0),
            "XAUUSD": _spec(trade_contract_size=100.0),
        }.get(symbol)
        run_clerk_execution_check()

    mock_open.assert_called_once()
    mock_close.assert_called_once()
    mock_amend.assert_called_once()

    state = read_execution_state()
    assert state["last_verdicts"]["GBPUSD"]["confirmed"] is True
    assert state["last_verdicts"]["EURUSD"]["confirmed"] is True
    assert state["last_tactical_verdicts"]["XAUUSD"]["tier"] == "defend"
    assert state["last_tactical_verdicts"]["XAUUSD"]["applied"] is True


@patch("ai.clerk_execution._run_clerk_tactical_check")
@patch("ai.clerk_execution._run_clerk_invalidation_check")
@patch(
    "ai.clerk_execution._fetch_technical_context",
    return_value=(_fake_ftmo_analysis(), "fake technical context"),
)
@patch("ai.clerk_execution.close_position")
@patch("ai.clerk_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.clerk_execution.fetch_ftmo_status")
@patch("ai.clerk_execution.get_market_watch")
@patch("ai.clerk_execution.get_pending_orders", return_value=[])
@patch("ai.clerk_execution.get_open_positions")
@patch("ai.clerk_execution.get_account_summary")
@patch("ai.clerk_execution.connect")
def test_tactical_exit_skipped_when_invalidation_already_forced_pct_zero_same_symbol_same_poll(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_close, mock_fetch_ctx, mock_invalidation, mock_tactical, _fixed_files,
):
    # EURUSD carries BOTH an invalidation_condition (so it's also a
    # watched position) AND is a plain filled position (so it's also a
    # tactical candidate) -- a real symbol legitimately eligible for
    # both checks in the same poll. The invalidation branch is submitted
    # (and therefore merged) BEFORE the tactical branch, so by the time
    # the tactical EXIT verdict is processed, merged_allocation["EURUSD"]
    # is already at pct=0 from invalidation -- "0% wins," tactical must
    # be skipped, not double-applied.
    from data.mt5_source import MarketAsset

    mock_account.return_value = _account()
    mock_positions.return_value = [_position(symbol="EURUSD", side="buy", volume=1.0, ticket=200, sl=1.0850, price_open=1.0900, price_current=1.0700)]
    mock_watch.return_value = [MarketAsset(symbol="EURUSD", description="Euro", bid=1.0699, ask=1.0700)]
    mock_status.return_value = _ftmo_status()
    mock_close.return_value = OrderResult(success=True, retcode=10009, comment="ok", ticket=200)
    mock_invalidation.return_value = ("EURUSD", True, "FINAL_VERDICT: CONFIRMED")
    tactical_verdict = TacticalVerdict(tier="exit", rule_citation="Murphy", numbers_citation="unlimited downside on a losing short")
    mock_tactical.return_value = ("EURUSD", tactical_verdict, "FINAL_VERDICT: EXIT")

    generated_utc = datetime.now(timezone.utc).isoformat()
    entry = {
        "pct": 1.0, "price": 1.0900, "stop_loss": 1.0850, "take_profit": None,
        "side": "buy", "reason": "r", "invalidation_condition": "H1 closes below 1.0800",
    }
    _write_suggestion(_fixed_files, immediate_allocation={"EURUSD": entry}, generated_utc=generated_utc)
    _seed_settled(_fixed_files, "EURUSD", "filled", generated_utc, entry=entry)
    set_tactical_defense_enabled(True)

    with patch("ai.clerk_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = _spec(trade_contract_size=100_000.0)
        run_clerk_execution_check()

    mock_close.assert_called_once()
    state = read_execution_state()
    assert state["last_verdicts"]["EURUSD"]["confirmed"] is True
    tactical = state["last_tactical_verdicts"]["EURUSD"]
    assert tactical["tier"] == "exit"
    assert tactical["applied"] is False
    assert tactical["skipped_reason"]


# --- _compute_tactical_signals: the deterministic pre-screen ---


_NO_OP_ENTRY = AllocationEntry(pct=5.0, price=2000.0, stop_loss=1950.0, side="buy")


def test_compute_tactical_signals_profit_lock_triggers_at_threshold_not_below():
    at_threshold = _position(side="buy", price_open=2000.0, price_current=2300.0)  # +15.0%
    below_threshold = _position(side="buy", price_open=2000.0, price_current=2290.0)  # +14.5%
    assert _compute_tactical_signals(at_threshold, _NO_OP_ENTRY, None).profit_lock_due is True
    assert _compute_tactical_signals(below_threshold, _NO_OP_ENTRY, None).profit_lock_due is False


def test_compute_tactical_signals_hard_exit_triggers_at_threshold_not_below():
    at_threshold = _position(side="buy", price_open=2000.0, price_current=1860.0)  # -7.0%
    below_threshold = _position(side="buy", price_open=2000.0, price_current=1870.0)  # -6.5%
    assert _compute_tactical_signals(at_threshold, _NO_OP_ENTRY, None).hard_exit_required is True
    assert _compute_tactical_signals(below_threshold, _NO_OP_ENTRY, None).hard_exit_required is False


def test_compute_tactical_signals_atr_stop_candidate_buy_side_is_signed_correctly():
    # Never-widen-stop convention (same as _validate_and_apply_tactical_
    # verdict's own sign check): for a BUY, a HIGHER stop is tighter.
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    h1_stats = _fake_technical_stats(atr=10.0)  # 1.5x ATR = 15 -> candidate 1965, above the 1950 sl
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats)
    assert signals.h1_atr == 10.0
    assert signals.atr_stop_candidate == pytest.approx(1965.0)
    assert signals.atr_stop_is_tighter_than_current is True


def test_compute_tactical_signals_atr_stop_candidate_sell_side_is_signed_correctly():
    # For a SELL, a LOWER stop is tighter.
    position = _position(side="sell", sl=2000.0, price_current=1980.0)
    h1_stats = _fake_technical_stats(atr=10.0)  # 1.5x ATR = 15 -> candidate 1995, below the 2000 sl
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats)
    assert signals.atr_stop_candidate == pytest.approx(1995.0)
    assert signals.atr_stop_is_tighter_than_current is True


def test_compute_tactical_signals_missing_atr_degrades_stop_candidate_to_none():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    signals_no_stats = _compute_tactical_signals(position, _NO_OP_ENTRY, None)
    signals_no_atr = _compute_tactical_signals(position, _NO_OP_ENTRY, _fake_technical_stats(atr=None))
    for signals in (signals_no_stats, signals_no_atr):
        assert signals.h1_atr is None
        assert signals.atr_stop_candidate is None
        assert signals.atr_stop_is_tighter_than_current is None


# --- velocity-tiered ATR stop multiple (2026-09-09, real XAUUSD/USDCAD incident) ---


def test_compute_tactical_signals_uses_wider_multiple_for_a_fast_instrument():
    # Real XAUUSD-range H1 atr_pct (~0.33%/hour) — above the 0.25% fast
    # threshold, so the wider CLERK_TACTICAL_ATR_STOP_MULTIPLE_FAST (2.5x
    # by default) applies instead of the standard 1.5x.
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    h1_stats = _fake_technical_stats(atr=10.0, atr_pct=0.33)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats)
    assert signals.velocity_tier == "fast"
    assert signals.atr_stop_multiple_used == config.CLERK_TACTICAL_ATR_STOP_MULTIPLE_FAST
    assert signals.atr_stop_candidate == pytest.approx(1980.0 - config.CLERK_TACTICAL_ATR_STOP_MULTIPLE_FAST * 10.0)


def test_compute_tactical_signals_uses_standard_multiple_for_a_slow_instrument():
    # Real USDCAD-range H1 atr_pct (~0.10%/hour) — below the fast
    # threshold, standard 1.5x multiple applies, same as before velocity
    # tiering existed.
    position = _position(side="sell", sl=2000.0, price_current=1980.0)
    h1_stats = _fake_technical_stats(atr=10.0, atr_pct=0.10)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats)
    assert signals.velocity_tier == "slow"
    assert signals.atr_stop_multiple_used == config.CLERK_TACTICAL_ATR_STOP_MULTIPLE
    assert signals.atr_stop_candidate == pytest.approx(1980.0 + config.CLERK_TACTICAL_ATR_STOP_MULTIPLE * 10.0)


def test_compute_tactical_signals_defaults_to_standard_multiple_without_an_atr_pct_reading():
    # No live atr_pct at all (e.g. thin history) — never guess a tier;
    # degrade to exactly the same behavior this had before velocity
    # tiering existed.
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    h1_stats = _fake_technical_stats(atr=10.0)  # atr_pct defaults to None
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats)
    assert signals.velocity_tier is None
    assert signals.atr_stop_multiple_used == config.CLERK_TACTICAL_ATR_STOP_MULTIPLE
    assert signals.atr_stop_candidate == pytest.approx(1965.0)


# --- RSI tier pre-computation (2026-09-09, real incident: qwen3:8b called RSI 11/15 "overbought") ---


def test_compute_tactical_signals_precomputes_oversold_rsi_tiers_for_the_model():
    # The exact real scenario: a deeply oversold RSI on both timeframes,
    # which the model itself repeatedly mislabeled "overbought" in free
    # text across three consecutive live polls. This must now arrive
    # pre-classified so the model never has to derive it.
    position = _position(side="sell")
    h1_stats = _fake_technical_stats(rsi=37)
    h4_stats = _fake_technical_stats(rsi=11)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats, h4_stats)
    assert signals.h1_rsi == 37
    assert signals.h1_rsi_tier == "neutral"
    assert signals.h4_rsi == 11
    assert signals.h4_rsi_tier == "oversold"


def test_compute_tactical_signals_precomputes_overbought_rsi_tier():
    position = _position(side="buy")
    h1_stats = _fake_technical_stats(rsi=82)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats)
    assert signals.h1_rsi_tier == "overbought"


def test_compute_tactical_signals_rsi_fields_degrade_to_none_without_h4_stats():
    # h4_stats is optional (defaults to None) so no existing caller breaks.
    position = _position(side="buy")
    h1_stats = _fake_technical_stats(rsi=50)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats)
    assert signals.h1_rsi == 50
    assert signals.h1_rsi_tier == "neutral"
    assert signals.h4_rsi is None
    assert signals.h4_rsi_tier is None


def test_compute_tactical_signals_rsi_fields_none_without_any_stats():
    position = _position(side="buy")
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None)
    assert signals.h1_rsi is None
    assert signals.h1_rsi_tier is None
    assert signals.h4_rsi is None
    assert signals.h4_rsi_tier is None


def test_compute_tactical_signals_picks_nearest_resistance_and_support():
    # Added 2026-09-09 -- the tactical clerk previously had NO access to
    # real structural S/R at all in its deterministic pre-screen (only as
    # buried prose in technical_context). Must pick the NEAREST level on
    # each side, not the strongest (most-touches) one, which is a
    # different ranking real select_curiosity-style callers already need.
    position = _position(side="buy")
    structure = ChartStructureSnapshot(
        fibonacci=None,
        sr_levels=SRLevelsResult(
            resistance_levels=[
                SRLevel(price=1.30, touches=1, distance_pct=+2.0),
                SRLevel(price=1.11, touches=1, distance_pct=+0.3),  # nearest, fewer touches
            ],
            support_levels=[
                SRLevel(price=1.00, touches=5, distance_pct=-5.0),
                SRLevel(price=1.08, touches=1, distance_pct=-0.5),  # nearest, fewer touches
            ],
        ),
        trendlines=None, patterns=[],
    )
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None, None, structure)
    assert signals.nearest_resistance.price == pytest.approx(1.11)
    assert signals.nearest_support.price == pytest.approx(1.08)


def test_compute_tactical_signals_sr_fields_none_without_structure():
    position = _position(side="buy")
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None)
    assert signals.nearest_resistance is None
    assert signals.nearest_support is None

    empty_structure = _empty_chart_structure()
    signals_empty = _compute_tactical_signals(position, _NO_OP_ENTRY, None, None, empty_structure)
    assert signals_empty.nearest_resistance is None
    assert signals_empty.nearest_support is None


def test_compute_tactical_signals_picks_the_most_recent_structure_break_and_sweep():
    # Added 2026-09-21, Phase 5c of the charting-expert upgrade — must
    # pick the MOST RECENT (smallest bars_ago) event on each list, not
    # just the first one, same "nearest, not strongest/first" discipline
    # as nearest_resistance/nearest_support above.
    position = _position(side="buy")
    structure = ChartStructureSnapshot(
        fibonacci=None, sr_levels=None, trendlines=None, patterns=[],
        structure_breaks=[
            StructureBreak(kind="BOS", direction="bullish", broken_level=1.10, break_price=1.11, bars_ago=8),
            StructureBreak(kind="CHOCH", direction="bearish", broken_level=1.05, break_price=1.04, bars_ago=2),
        ],
        liquidity_sweeps=[
            LiquiditySweepEvent(level_price=1.09, direction="swept_below", wick_penetration_pct=0.3, bars_ago=5, volume_ratio=1.8),
            LiquiditySweepEvent(level_price=1.12, direction="swept_above", wick_penetration_pct=0.1, bars_ago=1, volume_ratio=None),
        ],
    )
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None, None, structure)
    assert signals.nearest_structure_break.kind == "CHOCH"
    assert signals.nearest_structure_break.bars_ago == 2
    assert signals.nearest_liquidity_sweep.direction == "swept_above"
    assert signals.nearest_liquidity_sweep.bars_ago == 1


def test_compute_tactical_signals_structure_break_fields_none_without_structure():
    position = _position(side="buy")
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None)
    assert signals.nearest_structure_break is None
    assert signals.nearest_liquidity_sweep is None

    empty_structure = _empty_chart_structure()
    signals_empty = _compute_tactical_signals(position, _NO_OP_ENTRY, None, None, empty_structure)
    assert signals_empty.nearest_structure_break is None
    assert signals_empty.nearest_liquidity_sweep is None


def test_tactical_prompt_shows_the_liquidity_pool_tag_and_real_level_numbers():
    position = _position(side="buy")
    structure = ChartStructureSnapshot(
        fibonacci=None,
        sr_levels=SRLevelsResult(
            resistance_levels=[SRLevel(price=1.2345, touches=3, distance_pct=+0.42, is_liquidity_pool=True)],
            support_levels=[SRLevel(price=1.2000, touches=1, distance_pct=-2.4, is_liquidity_pool=False)],
        ),
        trendlines=None, patterns=[],
    )
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None, None, structure)
    prompt = _build_tactical_prompt(
        "EURUSD", _NO_OP_ENTRY, position, "fake technical context", 10000.0, None, signals,
    )
    assert "1.23450" in prompt
    assert "LIQUIDITY POOL" in prompt
    assert "3x real confirmed touches" in prompt
    assert "1.20000" in prompt


def test_tactical_prompt_shows_the_most_recent_structure_break_and_sweep():
    position = _position(side="buy")
    structure = ChartStructureSnapshot(
        fibonacci=None, sr_levels=None, trendlines=None, patterns=[],
        structure_breaks=[
            StructureBreak(kind="CHOCH", direction="bearish", broken_level=1.05000, break_price=1.04000, bars_ago=2),
        ],
        liquidity_sweeps=[
            LiquiditySweepEvent(level_price=1.12, direction="swept_above", wick_penetration_pct=0.15, bars_ago=1, volume_ratio=None),
        ],
    )
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None, None, structure)
    prompt = _build_tactical_prompt(
        "EURUSD", _NO_OP_ENTRY, position, "fake technical context", 10000.0, None, signals,
    )
    assert "CHOCH 2 bars ago" in prompt
    assert "1.05000" in prompt
    assert "liquidity sweep 1 bars ago" in prompt
    assert "swept_above" in prompt


def test_tactical_prompt_states_the_valid_stop_range_for_a_buy():
    # Added 2026-09-10 -- real, repeated incident: the local model
    # proposed a SELL's stop on the wrong side of live price six times in
    # one session. The prompt must now state the actual numeric window
    # explicitly rather than leaving the model to derive it from "tighter".
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None)
    prompt = _build_tactical_prompt("XAUUSD", _NO_OP_ENTRY, position, "ctx", 10000.0, None, signals)
    assert "Valid NEW_STOP_LOSS range for this BUY position" in prompt
    assert "1950.00000" in prompt
    assert "1980.00000" in prompt


def test_tactical_prompt_states_the_valid_stop_range_for_a_sell():
    position = _position(side="sell", sl=2000.0, price_current=1995.0)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None)
    prompt = _build_tactical_prompt("XAUUSD", _NO_OP_ENTRY, position, "ctx", 10000.0, None, signals)
    assert "Valid NEW_STOP_LOSS range for this SELL position" in prompt
    assert "1995.00000" in prompt
    assert "2000.00000" in prompt


def test_tactical_prompt_states_a_one_sided_valid_range_without_a_current_stop():
    position = _position(side="sell", sl=None, price_current=1995.0)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None)
    prompt = _build_tactical_prompt("XAUUSD", _NO_OP_ENTRY, position, "ctx", 10000.0, None, signals)
    assert "strictly ABOVE the current live price (1995.00000)" in prompt


def test_tactical_prompt_shows_the_precomputed_rsi_tier_not_just_the_raw_number():
    # Direct regression test for the real incident: the prompt must hand
    # the model an unambiguous, already-classified label rather than
    # leaving it to derive "overbought"/"oversold" itself from a bare
    # number — that's exactly what it got wrong three polls running.
    position = _position(side="sell")
    h1_stats = _fake_technical_stats(rsi=37)
    h4_stats = _fake_technical_stats(rsi=11)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats, h4_stats)
    prompt = _build_tactical_prompt(
        "USDCAD", _NO_OP_ENTRY, position, "fake technical context", 10000.0, None, signals,
    )
    assert "H4 RSI: 11" in prompt
    assert "OVERSOLD" in prompt
    # Must not present the oversold reading as overbought anywhere in the
    # deterministic pre-screen block this function builds.
    assert "H4 RSI: 11 — ALREADY CLASSIFIED AS **OVERBOUGHT**" not in prompt
    assert "do not describe an oversold reading as overbought" in prompt.lower()


def test_tactical_prompt_includes_supported_backtest_favorability_line():
    position = _position(side="buy")
    signals = _signals(backtest_favorability="supported", favorable_excursion_median_r=1.5)
    prompt = _build_tactical_prompt("XAUUSD", _NO_OP_ENTRY, position, "ctx", 10000.0, None, signals)
    assert "SUPPORTS its underlying thesis" in prompt
    assert "1.50R median" in prompt


def test_tactical_prompt_includes_contradicted_backtest_favorability_line():
    position = _position(side="buy")
    signals = _signals(backtest_favorability="contradicted")
    prompt = _build_tactical_prompt("XAUUSD", _NO_OP_ENTRY, position, "ctx", 10000.0, None, signals)
    assert "CONTRADICTS its underlying thesis" in prompt


def test_tactical_prompt_omits_backtest_favorability_line_for_mixed():
    position = _position(side="buy")
    signals = _signals(backtest_favorability="mixed")
    prompt = _build_tactical_prompt("XAUUSD", _NO_OP_ENTRY, position, "ctx", 10000.0, None, signals)
    assert "SUPPORTS its underlying thesis" not in prompt
    assert "CONTRADICTS its underlying thesis" not in prompt


def test_tactical_prompt_omits_backtest_favorability_line_for_none():
    # Every pre-existing test above builds a prompt via _signals()'s own
    # default (backtest_favorability unset) -- confirms no regression:
    # no line, no crash, no stray "None" leaking into the prompt text.
    position = _position(side="buy")
    signals = _signals()
    prompt = _build_tactical_prompt("XAUUSD", _NO_OP_ENTRY, position, "ctx", 10000.0, None, signals)
    assert "SUPPORTS its underlying thesis" not in prompt
    assert "CONTRADICTS its underlying thesis" not in prompt


def test_compute_tactical_signals_missing_take_profit_degrades_target_captured_to_none():
    no_tp = _position(side="buy", price_open=2000.0, price_current=2100.0, tp=None)
    degenerate_tp = _position(side="buy", price_open=2000.0, price_current=2100.0, tp=2000.0)
    assert _compute_tactical_signals(no_tp, _NO_OP_ENTRY, None).target_captured_pct is None
    assert _compute_tactical_signals(degenerate_tp, _NO_OP_ENTRY, None).target_captured_pct is None


def test_compute_tactical_signals_partial_profit_due_requires_both_pct_and_days():
    # +50% of the 200-point target (2000 -> 2200) captured, opened 2 days ago.
    fresh_and_captured = _position(side="buy", price_open=2000.0, price_current=2100.0, tp=2200.0)
    fresh_and_captured.opened_at = datetime.now() - timedelta(days=2)
    signals = _compute_tactical_signals(fresh_and_captured, _NO_OP_ENTRY, None)
    assert signals.target_captured_pct == pytest.approx(50.0)
    assert signals.partial_profit_due is True

    # Same capture, but held past the day cutoff.
    stale_and_captured = _position(side="buy", price_open=2000.0, price_current=2100.0, tp=2200.0)
    stale_and_captured.opened_at = datetime.now() - timedelta(days=10)
    assert _compute_tactical_signals(stale_and_captured, _NO_OP_ENTRY, None).partial_profit_due is False

    # Fresh, but under the capture threshold.
    fresh_not_captured = _position(side="buy", price_open=2000.0, price_current=2050.0, tp=2200.0)
    fresh_not_captured.opened_at = datetime.now() - timedelta(days=2)
    assert _compute_tactical_signals(fresh_not_captured, _NO_OP_ENTRY, None).partial_profit_due is False


def test_compute_tactical_signals_days_held_computes_a_real_value_for_naive_opened_at():
    # Position.opened_at is always tz-naive in real code (data/mt5_source.
    # py's own datetime.fromtimestamp construction) -- this must compute
    # a real number in that realistic case, not silently degrade to None.
    position = _position(side="buy")
    position.opened_at = datetime.now() - timedelta(days=3, hours=1)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None)
    assert signals.days_held is not None
    assert signals.days_held == pytest.approx(3.04, abs=0.05)


def test_compute_tactical_signals_backtest_favorability_none_when_base_not_given():
    # Backward compatibility: every pre-existing caller/test above calls
    # _compute_tactical_signals without `base` at all -- both new fields
    # must degrade to None, not raise or guess.
    position = _position(side="buy")
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None)
    assert signals.backtest_favorability is None
    assert signals.favorable_excursion_median_r is None


def test_compute_tactical_signals_wires_backtest_favorability_from_base():
    position = _position(side="buy")
    base = AssetAnalysis(
        symbol="XAUUSD", description="Gold", bid=2000.0, ask=2000.5, display_name="Gold",
        rsi_oversold_backtest=RSIReactionBacktest(
            "oversold", 30.0, 10, 7, 3, 0, 70.0, 0.6, 1.5, 3.0, 10,
            excursion=FavorableExcursionStats(sample_size=8, avg_r=2.5, median_r=2.1, horizon_bars=90),
        ),
    )
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None, base=base)
    assert signals.backtest_favorability == "supported"
    assert signals.favorable_excursion_median_r == pytest.approx(2.1)


# --- TacticalSignals.trend_flip_against_count (2026-09-16, real NVDA incident) ---


def test_trend_flip_count_starts_at_one_on_the_first_contradicting_poll():
    position = _position(side="buy")  # long
    h1_stats = _fake_technical_stats(trend="downtrend")
    h4_stats = _fake_technical_stats(trend="downtrend")  # aligned downtrend vs a BUY -- contradicts
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats, h4_stats, prior_tactical=None)
    assert signals.trend_flip_against_count == 1


def test_trend_flip_count_increments_across_consecutive_contradicting_polls():
    position = _position(side="buy")
    h1_stats = _fake_technical_stats(trend="downtrend")
    h4_stats = _fake_technical_stats(trend="downtrend")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, h4_stats, prior_tactical={"trend_flip_against_count": 4}
    )
    assert signals.trend_flip_against_count == 5


def test_trend_flip_count_resets_to_zero_when_trend_realigns():
    position = _position(side="buy")
    h1_stats = _fake_technical_stats(trend="uptrend")
    h4_stats = _fake_technical_stats(trend="uptrend")  # aligned uptrend vs a BUY -- no contradiction
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, h4_stats, prior_tactical={"trend_flip_against_count": 5}
    )
    assert signals.trend_flip_against_count == 0


def test_trend_flip_count_resets_to_zero_when_h1_h4_disagree_with_each_other():
    # No real, un-hedged aligned read at all -- nothing to contradict with.
    position = _position(side="buy")
    h1_stats = _fake_technical_stats(trend="downtrend")
    h4_stats = _fake_technical_stats(trend="uptrend")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, h4_stats, prior_tactical={"trend_flip_against_count": 5}
    )
    assert signals.trend_flip_against_count == 0


def test_trend_flip_count_zero_for_a_sell_position_when_aligned_trend_is_down():
    # A SELL is not contradicted by a downtrend -- it agrees with it.
    position = _position(side="sell")
    h1_stats = _fake_technical_stats(trend="downtrend")
    h4_stats = _fake_technical_stats(trend="downtrend")
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats, h4_stats, prior_tactical=None)
    assert signals.trend_flip_against_count == 0


def test_trend_flip_count_zero_by_default_when_no_prior_tactical_given():
    position = _position(side="buy")
    h1_stats = _fake_technical_stats(trend="uptrend")
    h4_stats = _fake_technical_stats(trend="uptrend")
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats, h4_stats)
    assert signals.trend_flip_against_count == 0


# --- _run_clerk_tactical_check: the hard-exit circuit breaker ---


@patch("ai.clerk_execution.run_ollama")
def test_hard_exit_required_never_calls_the_model(mock_run_ollama):
    position = _position(side="buy", price_open=2000.0, price_current=1850.0)  # -7.5%, past the 7.0% ceiling
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None)
    assert signals.hard_exit_required is True

    symbol, verdict, raw = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )

    mock_run_ollama.assert_not_called()
    assert symbol == "XAUUSD"
    assert verdict.tier == "exit"
    assert verdict.hard_exit is True
    assert verdict.rule_citation
    assert verdict.numbers_citation


# --- _run_clerk_tactical_check: sustained trend-flip circuit breaker (2026-09-16, real NVDA incident) ---


@patch("ai.clerk_execution.run_ollama")
def test_trend_flip_partial_fires_exactly_at_the_configured_threshold(mock_run_ollama):
    position = _position(side="buy", price_open=2000.0, price_current=1980.0)  # -1%, well under hard-exit
    h1_stats = _fake_technical_stats(trend="downtrend")
    h4_stats = _fake_technical_stats(trend="downtrend")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, h4_stats,
        prior_tactical={"trend_flip_against_count": config.CLERK_TREND_FLIP_PARTIAL_AFTER_POLLS - 1},
    )
    assert signals.trend_flip_against_count == config.CLERK_TREND_FLIP_PARTIAL_AFTER_POLLS

    symbol, verdict, raw = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )

    mock_run_ollama.assert_not_called()
    assert verdict.tier == "defend"
    assert verdict.hard_exit is True
    assert verdict.new_stop_loss is None
    assert verdict.partial_close_fraction == pytest.approx(config.CLERK_TREND_FLIP_PARTIAL_REDUCE_PCT / 100.0)
    assert verdict.rule_citation
    assert verdict.numbers_citation


@patch("ai.clerk_execution.run_ollama")
def test_trend_flip_does_not_re_fire_partial_between_the_two_thresholds(mock_run_ollama):
    # One poll past the partial threshold but still short of the exit
    # threshold -- must fall through to the normal LLM call, not force
    # another action (never compounds 50% -> 25% -> 12.5% ...).
    mock_run_ollama.return_value = "FINAL_VERDICT: HOLD"
    position = _position(side="buy", price_open=2000.0, price_current=1980.0)
    h1_stats = _fake_technical_stats(trend="downtrend")
    h4_stats = _fake_technical_stats(trend="downtrend")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, h4_stats,
        prior_tactical={"trend_flip_against_count": config.CLERK_TREND_FLIP_PARTIAL_AFTER_POLLS},
    )
    assert signals.trend_flip_against_count == config.CLERK_TREND_FLIP_PARTIAL_AFTER_POLLS + 1

    symbol, verdict, raw = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )

    mock_run_ollama.assert_called_once()
    assert verdict.tier == "hold"


@patch("ai.clerk_execution.run_ollama")
def test_trend_flip_exit_fires_at_the_configured_threshold(mock_run_ollama):
    position = _position(side="buy", price_open=2000.0, price_current=1980.0)
    h1_stats = _fake_technical_stats(trend="downtrend")
    h4_stats = _fake_technical_stats(trend="downtrend")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, h4_stats,
        prior_tactical={"trend_flip_against_count": config.CLERK_TREND_FLIP_EXIT_AFTER_POLLS - 1},
    )
    assert signals.trend_flip_against_count == config.CLERK_TREND_FLIP_EXIT_AFTER_POLLS

    symbol, verdict, raw = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )

    mock_run_ollama.assert_not_called()
    assert verdict.tier == "exit"
    assert verdict.hard_exit is True


@patch("ai.clerk_execution.run_ollama")
def test_trend_flip_exit_keeps_firing_past_the_threshold(mock_run_ollama):
    # Same "keep insisting" philosophy as hard_exit_required's own >= check.
    position = _position(side="buy", price_open=2000.0, price_current=1980.0)
    h1_stats = _fake_technical_stats(trend="downtrend")
    h4_stats = _fake_technical_stats(trend="downtrend")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, h4_stats,
        prior_tactical={"trend_flip_against_count": config.CLERK_TREND_FLIP_EXIT_AFTER_POLLS + 5},
    )
    symbol, verdict, raw = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )
    mock_run_ollama.assert_not_called()
    assert verdict.tier == "exit"


@patch("ai.clerk_execution.run_ollama")
def test_hard_exit_required_takes_priority_over_trend_flip_escalation(mock_run_ollama):
    # A real, large loss is a more urgent circuit-breaker than a sustained
    # trend-flip streak -- both being true this poll must still resolve
    # to the O'Neil hard-exit citation, not the trend-flip one.
    position = _position(side="buy", price_open=2000.0, price_current=1850.0)  # -7.5%, past hard-exit
    h1_stats = _fake_technical_stats(trend="downtrend")
    h4_stats = _fake_technical_stats(trend="downtrend")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, h4_stats,
        prior_tactical={"trend_flip_against_count": config.CLERK_TREND_FLIP_EXIT_AFTER_POLLS + 1},
    )
    assert signals.hard_exit_required is True

    symbol, verdict, raw = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )

    mock_run_ollama.assert_not_called()
    assert verdict.tier == "exit"
    assert "O'Neil" in verdict.rule_citation
    assert raw


# --- _run_clerk_tactical_check: deterministic quick profit-lock (2026-09-18, direct user request) ---


@patch("ai.clerk_execution.run_ollama")
def test_quick_profit_lock_fires_once_cost_multiple_is_covered(mock_run_ollama):
    # spread 0.02% -> threshold 3x = 0.06%; +0.25% favorable move clears it.
    position = _position(side="buy", price_open=2000.0, price_current=2005.0)
    h1_stats = _fake_technical_stats(trend="flat")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, trade_cost=_fake_trade_cost(spread_pct_of_price=0.02),
    )
    assert signals.quick_profit_lock_due is True

    symbol, verdict, raw = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )

    mock_run_ollama.assert_not_called()
    assert verdict.tier == "defend"
    assert verdict.hard_exit is True
    assert verdict.new_stop_loss is None
    assert verdict.partial_close_fraction == pytest.approx(config.CLERK_QUICK_PROFIT_LOCK_REDUCE_PCT / 100.0)
    assert verdict.rule_citation
    assert verdict.numbers_citation


@patch("ai.clerk_execution.run_ollama")
def test_quick_profit_lock_does_not_fire_below_the_cost_multiple(mock_run_ollama):
    mock_run_ollama.return_value = "FINAL_VERDICT: HOLD"
    # +0.025% favorable move, well under the 0.06% (3x 0.02%) threshold.
    position = _position(side="buy", price_open=2000.0, price_current=2000.5)
    h1_stats = _fake_technical_stats(trend="flat")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, trade_cost=_fake_trade_cost(spread_pct_of_price=0.02),
    )
    assert signals.quick_profit_lock_due is False

    symbol, verdict, raw = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )

    mock_run_ollama.assert_called_once()
    assert verdict.tier == "hold"


@patch("ai.clerk_execution.run_ollama")
def test_quick_profit_lock_does_not_re_fire_once_already_done(mock_run_ollama):
    mock_run_ollama.return_value = "FINAL_VERDICT: HOLD"
    position = _position(side="buy", price_open=2000.0, price_current=2005.0)
    h1_stats = _fake_technical_stats(trend="flat")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, trade_cost=_fake_trade_cost(spread_pct_of_price=0.02),
        prior_tactical={"quick_profit_lock_done": True},
    )
    assert signals.quick_profit_lock_due is False

    symbol, verdict, raw = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )

    mock_run_ollama.assert_called_once()
    assert verdict.tier == "hold"


def test_quick_profit_lock_skipped_when_trade_cost_unavailable():
    position = _position(side="buy", price_open=2000.0, price_current=2005.0)
    h1_stats = _fake_technical_stats(trend="flat")
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats, trade_cost=None)
    assert signals.quick_profit_lock_due is False


@patch("ai.clerk_execution.run_ollama")
def test_hard_exit_required_takes_priority_over_quick_profit_lock(mock_run_ollama):
    # A real, large adverse move (short side, price ran up) must still win
    # over a same-poll favorable-move reading elsewhere in the check —
    # constructed via a sell position deep in loss.
    position = _position(side="sell", price_open=2000.0, price_current=2150.0, sl=2100.0)  # -7.5% for a sell
    h1_stats = _fake_technical_stats(trend="flat")
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1_stats, trade_cost=_fake_trade_cost())
    assert signals.hard_exit_required is True
    assert signals.quick_profit_lock_due is False  # adverse move, not favorable -- can't both be true

    symbol, verdict, raw = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )

    mock_run_ollama.assert_not_called()
    assert verdict.tier == "exit"
    assert "O'Neil" in verdict.rule_citation


@patch("ai.clerk_execution.run_ollama")
def test_quick_profit_lock_flag_survives_a_hold_verdict(mock_run_ollama):
    mock_run_ollama.return_value = "FINAL_VERDICT: HOLD"
    # Fire the lock once (DEFEND).
    position = _position(side="buy", price_open=2000.0, price_current=2005.0)
    h1_stats = _fake_technical_stats(trend="flat")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, trade_cost=_fake_trade_cost(spread_pct_of_price=0.02),
    )
    _, verdict, _ = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )
    _, tactical_state, _ = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}), signals=signals,
    )
    assert tactical_state["quick_profit_lock_done"] is True

    # A later poll that HOLDs but also changes trend_flip_against_count
    # (forcing the spread-based persistence path, not the early no-op
    # return) must still carry the flag forward.
    h4_stats = _fake_technical_stats(trend="downtrend")
    h1_stats_2 = _fake_technical_stats(trend="downtrend")
    signals_2 = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats_2, h4_stats, prior_tactical=tactical_state,
    )
    assert signals_2.trend_flip_against_count == 1  # changed from 0 -> triggers HOLD's own persistence path
    _, verdict_2, _ = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals_2,
    )
    assert verdict_2.tier == "hold"
    _, tactical_state_2, _ = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict_2, position, None, tactical_state, 100_000.0, _get_spec_for({"XAUUSD": _spec()}), signals=signals_2,
    )
    assert tactical_state_2["quick_profit_lock_done"] is True


@patch("ai.clerk_execution.run_ollama")
def test_quick_profit_lock_flag_survives_an_exit_verdict(mock_run_ollama):
    # Fire the lock once (DEFEND), then a later hard-exit (EXIT, which
    # spreads prior_tactical) must still carry the flag forward.
    position = _position(side="buy", price_open=2000.0, price_current=2005.0)
    h1_stats = _fake_technical_stats(trend="flat")
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_stats, trade_cost=_fake_trade_cost(spread_pct_of_price=0.02),
    )
    _, verdict, _ = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "fake technical context", 100_000.0, None, signals,
    )
    _, tactical_state, _ = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}), signals=signals,
    )
    assert tactical_state["quick_profit_lock_done"] is True

    losing_position = _position(side="buy", price_open=2000.0, price_current=1850.0)  # -7.5%, past hard-exit
    signals_2 = _compute_tactical_signals(
        losing_position, _NO_OP_ENTRY, h1_stats, prior_tactical=tactical_state,
    )
    assert signals_2.hard_exit_required is True
    _, verdict_2, _ = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, losing_position, "fake technical context", 100_000.0, None, signals_2,
    )
    assert verdict_2.tier == "exit"
    _, tactical_state_2, _ = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict_2, losing_position, None, tactical_state, 100_000.0, _get_spec_for({"XAUUSD": _spec()}), signals=signals_2,
    )
    assert tactical_state_2["quick_profit_lock_done"] is True


# --- Decision-tier (M15/M5) tactical signals, intraday decision-tier upgrade 2026-09-24 ---


@pytest.fixture(autouse=True)
def _no_real_economic_calendar():
    with (
        patch("ai.clerk_execution.economic_calendar.fetch_calendar_events", return_value=[]),
        patch("ai.clerk_execution.get_server_time_offset", return_value=None),
    ):
        yield


def _m15_structure(support=None, resistance=None):
    return ChartStructureSnapshot(
        fibonacci=None,
        sr_levels=SRLevelsResult(
            resistance_levels=[resistance] if resistance else [], support_levels=[support] if support else []
        ),
        trendlines=None, patterns=[],
    )


def test_atr_stop_candidate_uses_the_m5_atr_multiple_when_an_m5_read_exists():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    h1 = _fake_technical_stats(atr=10.0)    # H1-era 1.5x would give 1980 - 15 = 1965
    m5 = _fake_technical_stats(atr=4.0, rsi=25.0)  # 2.0 x 4 = 8 -> 1972
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1, m5_stats=m5)
    assert signals.atr_timeframe == "M5" and signals.m5_atr == 4.0
    assert signals.atr_stop_candidate == pytest.approx(1972.0)
    assert signals.atr_stop_multiple_used == config.CLERK_TACTICAL_M5_ATR_STOP_MULTIPLE
    assert signals.h1_atr == 10.0  # H1 stays available as the anchor/regime read
    assert signals.m5_rsi_tier == "oversold"


def test_fast_tier_uses_the_wider_m5_multiple():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    h1 = _fake_technical_stats(atr=10.0, atr_pct=0.5)  # 0.5%/h -> "fast" tier
    m5 = _fake_technical_stats(atr=4.0)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, h1, m5_stats=m5)
    assert signals.velocity_tier == "fast"
    assert signals.atr_stop_multiple_used == config.CLERK_TACTICAL_M5_ATR_STOP_MULTIPLE_FAST
    assert signals.atr_stop_candidate == pytest.approx(1980.0 - 3.0 * 4.0)


def test_atr_stop_candidate_falls_back_to_h1_without_an_m5_read():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, _fake_technical_stats(atr=10.0))
    assert signals.atr_timeframe == "H1"
    assert signals.atr_stop_candidate == pytest.approx(1965.0)


def test_structure_trail_candidate_sits_just_beyond_the_nearest_m5_support_for_a_buy():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    support = SRLevel(price=1972.0, touches=3, distance_pct=-0.4, low=1971.0, high=1973.0)
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, None, m5_stats=_fake_technical_stats(atr=4.0), m5_structure=_m15_structure(support=support)
    )
    assert signals.nearest_m5_support == support
    assert signals.structure_trail_candidate == pytest.approx(1971.0 - 0.4 * 4.0)  # band low minus 0.4 M5 ATR
    assert signals.structure_trail_is_tighter is True  # above the 1950 stop


def test_structure_trail_candidate_mirrors_for_a_sell_and_is_dropped_on_the_wrong_side_of_price():
    position = _position(side="sell", sl=2000.0, price_current=1980.0)
    resistance = SRLevel(price=1988.0, touches=2, distance_pct=0.4, low=1987.0, high=1989.0)
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, None, m5_stats=_fake_technical_stats(atr=4.0), m5_structure=_m15_structure(resistance=resistance)
    )
    assert signals.structure_trail_candidate == pytest.approx(1989.0 + 0.4 * 4.0)
    assert signals.structure_trail_is_tighter is True  # below the 2000 stop
    # A buy whose only M5 support is ABOVE live price has no valid trail candidate.
    buy = _position(side="buy", sl=1950.0, price_current=1980.0)
    above = SRLevel(price=1990.0, touches=2, distance_pct=0.5, low=1989.0, high=1991.0)
    none = _compute_tactical_signals(
        buy, _NO_OP_ENTRY, None, m5_stats=_fake_technical_stats(atr=4.0), m5_structure=_m15_structure(support=above)
    )
    assert none.structure_trail_candidate is None


def test_tactical_prompt_shows_decision_tier_lines_roles_note_and_event_risk():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    support = SRLevel(price=1972.0, touches=3, distance_pct=-0.4, low=1971.0, high=1973.0)
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, _fake_technical_stats(atr=10.0),
        m5_stats=_fake_technical_stats(atr=4.0, rsi=62.0),
        m5_structure=_m15_structure(support=support), event_note="High-impact event(s): USD CPI in 20 min",
    )
    prompt = _build_tactical_prompt("XAUUSD", _NO_OP_ENTRY, position, "ctx", 10000.0, None, signals)
    assert "TIMEFRAME ROLES" in prompt and "DECISION TIER" in prompt
    assert "M5 RSI: 62" in prompt and "M15" not in prompt
    assert "Nearest M5 structural support" in prompt and "1971.00000-1973.00000" in prompt
    assert "M5 structure-trail stop candidate" in prompt
    assert "Event risk: High-impact event(s): USD CPI in 20 min" in prompt
    assert "M5 ATR-based stop candidate (2.0x M5 ATR" in prompt


def test_tactical_prompt_without_m5_data_renders_h1_lines_only():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, _fake_technical_stats(atr=10.0))
    prompt = _build_tactical_prompt("XAUUSD", _NO_OP_ENTRY, position, "ctx", 10000.0, None, signals)
    assert "H1 ATR-based stop candidate (1.5x H1 ATR" in prompt
    assert "M5 structure-trail" not in prompt and "Event risk" not in prompt


def test_roles_note_lets_the_clerk_judge_older_conditions_written_on_a_higher_timeframe():
    from ai.clerk_execution import _TIMEFRAME_ROLES_NOTE

    flat = " ".join(_TIMEFRAME_ROLES_NOTE.split())
    assert "explicitly wrote on H1, H4 or D1" in flat and "judged against that named timeframe's read" in flat


def test_verdict_and_invalidation_prompts_carry_the_timeframe_roles_note():
    from ai.clerk_execution import _build_invalidation_prompt, _build_verdict_prompt

    setup = PendingSetup(symbol="EURUSD", side="buy", pct=1.0, trigger_condition="M15 closes above 1.1",
                         price=1.1, stop_loss=1.09, take_profit=1.12, reason="r")
    verdict = _build_verdict_prompt(setup, "ctx", 10000.0, "10 minutes", "")
    invalidation = _build_invalidation_prompt("EURUSD", {"side": "buy"}, "M15 closes below 1.09", "ctx", 10000.0, "10 minutes", "filled")
    for prompt in (verdict, invalidation):
        assert "TIMEFRAME ROLES" in prompt and "CONTEXT TIER" in prompt


# --- M5-ONLY risk management (2026-09-24): velocity, structure and the trend-flip breaker read M5 ---


def test_velocity_tier_is_classified_from_the_median_m5_atr_pct_not_the_h1_one():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    h1_slow = _fake_technical_stats(atr=10.0, atr_pct=0.1)    # would classify "slow" on H1
    fast = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_slow, m5_stats=_fake_technical_stats(atr=4.0), m5_atr_pct_median=0.10
    )
    assert fast.velocity_tier == "fast" and fast.velocity_timeframe == "M5"
    assert fast.atr_stop_multiple_used == config.CLERK_TACTICAL_M5_ATR_STOP_MULTIPLE_FAST
    h1_fast = _fake_technical_stats(atr=10.0, atr_pct=0.6)    # would classify "fast" on H1
    slow = _compute_tactical_signals(
        position, _NO_OP_ENTRY, h1_fast, m5_stats=_fake_technical_stats(atr=4.0), m5_atr_pct_median=0.03
    )
    assert slow.velocity_tier == "slow"
    assert slow.atr_stop_multiple_used == config.CLERK_TACTICAL_M5_ATR_STOP_MULTIPLE


def test_velocity_tier_ignores_a_quiet_hour_dip_of_the_instantaneous_m5_atr_pct():
    # Gold-shaped: typical M5 ATR% 0.10 (fast), but the CURRENT window is quiet (0.05, below the cut).
    # The audit found the instantaneous value flips fast<->slow on 18-28% of bars; only the median counts.
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    quiet = _fake_technical_stats(atr=1.5, atr_pct=0.05)
    signals = _compute_tactical_signals(position, _NO_OP_ENTRY, None, m5_stats=quiet, m5_atr_pct_median=0.10)
    assert signals.velocity_tier == "fast"


def test_velocity_falls_back_to_the_h1_tier_when_there_is_no_m5_baseline():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, _fake_technical_stats(atr=10.0, atr_pct=0.5), m5_stats=_fake_technical_stats(atr=4.0, atr_pct=0.02)
    )
    assert signals.velocity_tier == "fast" and signals.velocity_timeframe == "H1"


def test_nearest_levels_and_structure_events_come_from_the_m5_structure_with_h1_only_as_fallback():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    m5_res = SRLevel(price=1985.0, touches=2, distance_pct=0.3, low=1984.0, high=1986.0)
    h1_res = SRLevel(price=2010.0, touches=5, distance_pct=1.5, low=2009.0, high=2011.0)
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, _fake_technical_stats(atr=10.0), h1_structure=_m15_structure(resistance=h1_res),
        m5_stats=_fake_technical_stats(atr=4.0), m5_structure=_m15_structure(resistance=m5_res),
    )
    assert signals.structure_timeframe == "M5" and signals.nearest_resistance == m5_res
    fallback = _compute_tactical_signals(
        position, _NO_OP_ENTRY, _fake_technical_stats(atr=10.0), h1_structure=_m15_structure(resistance=h1_res)
    )
    assert fallback.structure_timeframe == "H1" and fallback.nearest_resistance == h1_res


def test_trend_flip_counter_reads_the_m5_trend_and_regime_and_needs_both_to_agree():
    buy = _position(side="buy", sl=1950.0, price_current=1980.0)
    down = _fake_technical_stats(atr=4.0, trend="downtrend", market_regime="trending_down")
    signals = _compute_tactical_signals(buy, _NO_OP_ENTRY, None, m5_stats=down, prior_tactical={"trend_flip_against_count": 2})
    assert signals.trend_flip_basis == "M5" and signals.trend_flip_against_count == 3
    # M5 trend down but the 60-bar regime still up -> not a confirmed flip: the streak resets.
    mixed = _fake_technical_stats(atr=4.0, trend="downtrend", market_regime="trending_up")
    reset = _compute_tactical_signals(buy, _NO_OP_ENTRY, None, m5_stats=mixed, prior_tactical={"trend_flip_against_count": 2})
    assert reset.trend_flip_against_count == 0
    # A sell is only contradicted by an aligned M5 UP read.
    sell = _position(side="sell", sl=2000.0, price_current=1980.0)
    assert _compute_tactical_signals(sell, _NO_OP_ENTRY, None, m5_stats=down).trend_flip_against_count == 0


def test_trend_flip_counter_falls_back_to_h1_h4_only_when_there_is_no_m5_read():
    buy = _position(side="buy", sl=1950.0, price_current=1980.0)
    h1 = _fake_technical_stats(atr=10.0, trend="downtrend")
    h4 = _fake_technical_stats(trend="downtrend")
    signals = _compute_tactical_signals(buy, _NO_OP_ENTRY, h1, h4)
    assert signals.trend_flip_basis == "H1+H4" and signals.trend_flip_against_count == 1
    # With an M5 read present the H1+H4 agreement is context and does NOT drive the breaker.
    with_m5 = _compute_tactical_signals(
        buy, _NO_OP_ENTRY, h1, h4, m5_stats=_fake_technical_stats(atr=4.0, trend="uptrend", market_regime="trending_up")
    )
    assert with_m5.trend_flip_basis == "M5" and with_m5.trend_flip_against_count == 0


def test_m5_trend_flip_breaker_message_names_the_m5_basis():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    signals = _signals(
        trend_flip_against_count=config.CLERK_TREND_FLIP_EXIT_AFTER_POLLS, trend_flip_basis="M5",
    )
    _sym, verdict, raw = _run_clerk_tactical_check("XAUUSD", _NO_OP_ENTRY, position, "ctx", 10000.0, None, signals)
    assert verdict.tier == "exit" and verdict.hard_exit
    assert "confirmed M5 trend" in raw and "M5 trend has read opposite" in verdict.numbers_citation


def test_tactical_prompt_labels_the_nearest_levels_with_their_timeframe_and_h1_h4_rsi_as_context():
    position = _position(side="buy", sl=1950.0, price_current=1980.0)
    support = SRLevel(price=1972.0, touches=3, distance_pct=-0.4, low=1971.0, high=1973.0)
    signals = _compute_tactical_signals(
        position, _NO_OP_ENTRY, _fake_technical_stats(atr=10.0, rsi=55.0), m5_stats=_fake_technical_stats(atr=4.0),
        m5_structure=_m15_structure(support=support), m5_atr_pct_median=0.12,
    )
    prompt = _build_tactical_prompt("XAUUSD", _NO_OP_ENTRY, position, "ctx", 10000.0, None, signals)
    assert "Nearest M5 structural support: 1971.00000-1973.00000" in prompt
    assert "Nearest H1 structural" not in prompt
    assert "(context only)" in prompt
    assert "typical M5 ATR classifies it as a FAST-tier mover" in prompt


# --- A discretionary partial close is a PROFIT-taking action (real MSFT incident, 2026-09-22) ---


def test_partial_close_on_a_position_not_in_profit_is_dropped_but_the_stop_change_is_kept():
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1975.0, partial_close_fraction=0.25, rule_citation="O'Neil",
                              numbers_citation="A 15% gain was already achieved")
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1980.0)   # 1% underwater
    existing = AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0, side="buy")
    new_entry, state, reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, None, 100_000.0, _get_spec_for({"XAUUSD": _spec(trade_contract_size=100.0)}),
    )
    assert reason == "" and new_entry is not None and state is not None
    assert new_entry.stop_loss == 1975.0
    # the FULL 1.0 lot is kept (no 25% cut): 1.0 lot x 25-point stop x 100 = $2,500 = 2.5% of 100k
    assert new_entry.pct == pytest.approx(2.5)


def test_a_partial_close_only_defend_on_a_losing_position_is_rejected():
    verdict = TacticalVerdict(tier="defend", partial_close_fraction=0.25, rule_citation="O'Neil", numbers_citation="x")
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1980.0)
    new_entry, state, reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert new_entry is None and state is None and "not in profit" in reason


def test_a_partial_close_in_profit_still_works_and_a_hard_exit_cut_is_exempt():
    profitable = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=2030.0)
    ok = TacticalVerdict(tier="defend", partial_close_fraction=0.25, rule_citation="Schwager", numbers_citation="x")
    entry, _, reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", ok, profitable, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert reason == "" and entry is not None
    # the deterministic trend-flip cut reduces a LOSING position on purpose
    losing = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1980.0)
    cut = TacticalVerdict(tier="defend", partial_close_fraction=0.5, rule_citation="trend flip", numbers_citation="x", hard_exit=True)
    entry2, _, reason2 = _validate_and_apply_tactical_verdict(
        "XAUUSD", cut, losing, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert reason2 == "" and entry2 is not None
    # a SELL is in profit when price fell below its open
    sell_profit = _position(side="sell", volume=1.0, sl=2050.0, price_open=2000.0, price_current=1970.0)
    entry3, _, reason3 = _validate_and_apply_tactical_verdict(
        "XAUUSD", ok, sell_profit, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert reason3 == "" and entry3 is not None


def test_the_profit_requirement_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "CLERK_PARTIAL_CLOSE_REQUIRES_PROFIT", False)
    verdict = TacticalVerdict(tier="defend", partial_close_fraction=0.25, rule_citation="x", numbers_citation="x")
    losing = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1980.0)
    entry, _, reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, losing, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}),
    )
    assert reason == "" and entry is not None


# --- Deterministic profit trail (2026-09-25) ---------------------------------------------------------------

def _trail_signals(position, entry=None, prior=None, atr=2.0, trade_cost=None):
    entry = entry or AllocationEntry(pct=1.0, price=position.price_open, stop_loss=position.sl, side=position.side)
    return _compute_tactical_signals(
        position, entry, None, m5_stats=_fake_technical_stats(atr=atr), prior_tactical=prior, trade_cost=trade_cost,
    )


def test_trail_waits_for_plus_one_r_then_sits_one_and_a_half_m5_atr_behind_the_price():
    # entry 2000, original stop 1996 (R0 = 4 = 2 ATR), ATR 2.0.
    below = _trail_signals(_position(price_open=2000.0, price_current=2003.9, sl=1996.0))
    assert below.profit_trail_r == pytest.approx(0.975) and below.profit_trail_stop is None
    at = _trail_signals(_position(price_open=2000.0, price_current=2004.0, sl=1996.0))
    assert at.profit_trail_r == pytest.approx(1.0)
    assert at.profit_trail_stop == pytest.approx(2004.0 - 1.5 * 2.0)  # 2001.0: already past breakeven
    assert at.profit_trail_r0 == pytest.approx(4.0)


def test_trail_for_a_sell_mirrors_the_buy():
    signals = _trail_signals(_position(side="sell", price_open=2000.0, price_current=1995.0, sl=2004.0))
    assert signals.profit_trail_r == pytest.approx(1.25)
    assert signals.profit_trail_stop == pytest.approx(1995.0 + 3.0)


def test_trail_only_ratchets_and_ignores_micro_improvements():
    already_tight = _trail_signals(_position(price_open=2000.0, price_current=2006.0, sl=2002.9))
    assert already_tight.profit_trail_stop is None  # candidate 2003.0: only 0.1 above the stop, step is 0.5
    improved = _trail_signals(_position(price_open=2000.0, price_current=2006.0, sl=2002.0))
    assert improved.profit_trail_stop == pytest.approx(2003.0)
    looser = _trail_signals(_position(price_open=2000.0, price_current=2006.0, sl=2004.0))
    assert looser.profit_trail_stop is None  # the existing stop is already tighter: never widen


def test_trail_uses_the_persisted_original_risk_not_a_later_tightened_stop():
    # The stop has since been tightened to 2001 (1 unit from entry); measuring "R" against THAT would say +3R.
    position = _position(price_open=2000.0, price_current=2003.0, sl=2001.0)
    naive = _trail_signals(position, entry=AllocationEntry(pct=1.0, price=2000.0, stop_loss=2001.0, side="buy"))
    assert naive.profit_trail_r == pytest.approx(3.0)
    persisted = _trail_signals(position, entry=AllocationEntry(pct=1.0, price=2000.0, stop_loss=2001.0, side="buy"),
                               prior={"profit_trail_r0": 4.0})
    assert persisted.profit_trail_r == pytest.approx(0.75) and persisted.profit_trail_stop is None


def test_trail_respects_the_brokers_minimum_stop_distance():
    cost = _fake_trade_cost(min_stop_distance_pct=0.3)  # 0.3% of ~2006 = 6.0 > 1.5 ATR (3.0)
    signals = _trail_signals(_position(price_open=2000.0, price_current=2006.0, sl=1996.0), trade_cost=cost)
    assert signals.profit_trail_stop == pytest.approx(2006.0 - 2006.0 * 0.003 * 1.02)


def test_trail_needs_an_m5_atr_a_stop_and_the_switch(monkeypatch):
    position = _position(price_open=2000.0, price_current=2006.0, sl=1996.0)
    assert _compute_tactical_signals(position, _NO_OP_ENTRY, None).profit_trail_stop is None  # no M5 read
    assert _trail_signals(_position(price_open=2000.0, price_current=2006.0, sl=None)).profit_trail_stop is None
    monkeypatch.setattr(config, "CLERK_PROFIT_TRAIL_ENABLED", False)
    assert _trail_signals(position).profit_trail_stop is None


@patch("ai.clerk_execution.run_ollama")
def test_trail_verdict_is_a_stop_only_deterministic_defend_that_never_calls_the_model(mock_run_ollama):
    position = _position(price_open=2000.0, price_current=2006.0, sl=1996.0)
    signals = _trail_signals(position)
    symbol, verdict, raw = _run_clerk_tactical_check(
        "XAUUSD", _NO_OP_ENTRY, position, "ctx", 100_000.0, None, signals,
    )
    mock_run_ollama.assert_not_called()
    assert verdict.tier == "defend" and verdict.hard_exit is True and verdict.partial_close_fraction is None
    assert verdict.new_stop_loss == pytest.approx(2003.0) and "1.50R" in verdict.numbers_citation
    assert raw.startswith("[deterministic circuit-breaker")


def test_trail_defend_validates_persists_the_original_risk_and_sizes_the_same_lots():
    position = _position(price_open=2000.0, price_current=2006.0, sl=1996.0, volume=2.0)
    signals = _trail_signals(position)
    verdict = TacticalVerdict(tier="defend", new_stop_loss=signals.profit_trail_stop, rule_citation="r", numbers_citation="n", hard_exit=True)
    entry = AllocationEntry(pct=2.0, price=2000.0, stop_loss=1996.0, side="buy")
    new_entry, state, rejected = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, entry, None, 100_000.0, _get_spec_for({"XAUUSD": _spec()}), signals,
    )
    assert rejected == "" and new_entry.stop_loss == pytest.approx(2003.0)
    assert state["profit_trail_r0"] == pytest.approx(4.0) and state["persisted_stop_loss"] == pytest.approx(2003.0)


@patch("ai.clerk_execution.run_ollama")
def test_a_position_not_yet_at_plus_one_r_still_goes_to_the_model(mock_run_ollama):
    mock_run_ollama.return_value = "FINAL_VERDICT: HOLD"
    position = _position(price_open=2000.0, price_current=2002.0, sl=1996.0)
    _run_clerk_tactical_check("XAUUSD", _NO_OP_ENTRY, position, "ctx", 100_000.0, None, _trail_signals(position))
    mock_run_ollama.assert_called()


# --- thinking pass: do not re-ask the model about a HOLD when nothing moved (2026-09-26) -------------------------------------------

def _fingerprint(position=None, **signal_overrides):
    from ai.clerk_execution import _tactical_fingerprint
    return _tactical_fingerprint(position or _position(price_current=1980.0), _signals(m5_atr=2.0, **signal_overrides))


def _cache_with(symbol, fingerprint, tier="hold", minutes_old=5.0):
    now = datetime.now(timezone.utc)
    return {"items": {f"tactical:{symbol}": {
        "verdict": {"tier": tier}, "fingerprint": fingerprint, "checked_utc": (now - timedelta(minutes=minutes_old)).isoformat(),
    }}}, now


def test_fingerprint_ignores_tiny_price_moves_but_sees_real_ones():
    base = _fingerprint()
    assert _fingerprint(_position(price_current=1980.3)) == base  # inside the same half-ATR bucket
    assert _fingerprint(_position(price_current=1976.0)) != base  # a real move
    assert _fingerprint(_position(sl=1960.0)) != base  # the position's own terms changed
    assert _fingerprint(trend_flip_against_count=1) != base
    assert _fingerprint(profit_lock_due=True) != base
    assert _fingerprint(m5_rsi_tier="overbought") != base


def test_an_unchanged_hold_is_reused_but_never_a_defend_an_old_or_a_moved_one():
    from ai.clerk_execution import _tactical_unchanged
    fp = _fingerprint()
    cache, now = _cache_with("XAUUSD", fp)
    assert _tactical_unchanged(cache, "XAUUSD", fp, now)
    assert not _tactical_unchanged(cache, "XAUUSD", _fingerprint(trend_flip_against_count=2), now)
    assert not _tactical_unchanged(cache, "EURUSD", fp, now)
    defend, now = _cache_with("XAUUSD", fp, tier="defend")
    assert not _tactical_unchanged(defend, "XAUUSD", fp, now)
    old, now = _cache_with("XAUUSD", fp, minutes_old=config.CLERK_THINK_REUSE_MAX_MINUTES + 1)
    assert not _tactical_unchanged(old, "XAUUSD", fp, now)
    legacy, now = _cache_with("XAUUSD", None)  # an item stored before fingerprints existed
    assert not _tactical_unchanged(legacy, "XAUUSD", fp, now)


# --- a partial close is impossible on a minimum-lot position (XAGUSD 2026-09-26: "infeasible" every poll, stop change lost) ---

def test_a_partial_close_below_the_minimum_lot_is_dropped_but_the_stop_change_is_kept():
    verdict = TacticalVerdict(tier="defend", new_stop_loss=2010.0, partial_close_fraction=0.5, rule_citation="Schwager", numbers_citation="x")
    position = _position(side="buy", volume=0.01, sl=1950.0, price_open=2000.0, price_current=2030.0)  # 0.01 lot = the minimum
    new_entry, state, reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec(trade_contract_size=100.0)}),
    )
    assert reason == "" and new_entry is not None and state is not None
    assert new_entry.stop_loss == 2010.0
    assert state["partial_pending_from_volume"] is None  # no partial to insist on / retry
    # the whole 0.01 lot is kept: 0.01 x 10-point stop x 100 = $10 = 0.01% of 100k
    assert new_entry.pct == pytest.approx(0.01)


def test_a_partial_only_defend_on_a_minimum_lot_position_does_nothing_and_a_quick_lock_is_marked_done():
    verdict = TacticalVerdict(tier="defend", partial_close_fraction=0.2, rule_citation="quick lock", numbers_citation="x")
    position = _position(side="buy", volume=0.01, sl=1950.0, price_open=2000.0, price_current=2030.0)
    spec = _get_spec_for({"XAUUSD": _spec(trade_contract_size=100.0)})
    new_entry, state, reason = _validate_and_apply_tactical_verdict("XAUUSD", verdict, position, None, None, 100_000.0, spec)
    assert new_entry is None and state is None and reason == ""
    signals = _signals(quick_profit_lock_due=True)
    new_entry, state, reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, {"defend_count": 1}, 100_000.0, spec, signals,
    )
    assert new_entry is None and reason == "" and state["quick_profit_lock_done"] is True and state["defend_count"] == 1


def test_a_partial_close_that_leaves_at_least_the_minimum_lot_still_works():
    verdict = TacticalVerdict(tier="defend", partial_close_fraction=0.5, rule_citation="Schwager", numbers_citation="x")
    position = _position(side="buy", volume=0.04, sl=1950.0, price_open=2000.0, price_current=2030.0)
    new_entry, state, reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, None, None, 100_000.0, _get_spec_for({"XAUUSD": _spec(trade_contract_size=100.0)}),
    )
    assert reason == "" and new_entry is not None and state["partial_pending_from_volume"] == 0.04
