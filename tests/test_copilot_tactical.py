import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import config
from ai.copilot_cli import CLI_FAILED_PREFIX as COPILOT_FAILED_PREFIX
from ai.copilot_cli import CLI_MISSING_MESSAGE as COPILOT_MISSING_MESSAGE
from ai.copilot_execution import (
    TacticalVerdict,
    _backfill_settlement_for_held_positions,
    _reconcile_settlement,
    _validate_and_apply_tactical_verdict,
    parse_tactical_verdict,
    read_execution_state,
    read_settlement,
    read_tactical_defense_enabled,
    run_copilot_execution_check,
    set_copilot_execution_enabled,
    set_tactical_defense_enabled,
)
from ai.portfolio_suggest import AllocationEntry, PendingSetup
from data.mt5_execution import OrderResult
from data.mt5_source import ContractSpec, Position


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
        patch.object(config, "COPILOT_EXECUTION_MAX_TACTICAL_CANDIDATES", 10),
        patch.object(config, "AMEND_TOLERANCE_PCT", 0.05),
        patch.object(config, "COPILOT_EXECUTION_ENABLED_FILE", str(tmp_path / "copilot_execution_enabled.json")),
        patch.object(config, "COPILOT_EXECUTION_INTERVAL_FILE", str(tmp_path / "copilot_execution_interval.json")),
        patch.object(config, "COPILOT_TACTICAL_DEFENSE_ENABLED_FILE", str(tmp_path / "copilot_tactical_defense_enabled.json")),
        patch.object(config, "COPILOT_TACTICAL_DEFEND_COOLDOWN_MINUTES", 60),
        patch.object(config, "COPILOT_TACTICAL_DEFEND_MIN_RETRIGGER_PCT", 0.3),
        patch.object(config, "COPILOT_TACTICAL_MIN_PARTIAL_CLOSE_FRACTION", 0.10),
        patch.object(config, "COPILOT_TACTICAL_MAX_PARTIAL_CLOSE_FRACTION", 0.75),
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


def _get_spec_for(specs: dict):
    return lambda symbol: specs.get(symbol)


def _write_suggestion(tmp_path, immediate_allocation=None, pending_setups=None, generated_utc=None):
    payload = {
        "generated_utc": generated_utc or datetime.now(timezone.utc).isoformat(),
        "immediate_allocation": immediate_allocation or {},
        "pending_setups": pending_setups or [],
    }
    Path(config.MEGA_ANALYSIS_LATEST_SUGGESTION_FILE).write_text(json.dumps(payload))


def _seed_settled(tmp_path, symbol, state, generated_utc, entry=None):
    Path(config.COPILOT_EXECUTION_SETTLEMENT_FILE).write_text(json.dumps({
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


def test_parse_tactical_verdict_fails_safe_when_cli_missing():
    assert parse_tactical_verdict(COPILOT_MISSING_MESSAGE).tier == "hold"


def test_parse_tactical_verdict_fails_safe_when_cli_failed():
    assert parse_tactical_verdict(f"{COPILOT_FAILED_PREFIX} (timed out).").tier == "hold"


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
    assert rejected_reason == ""  # a successful EXIT is not a rejection


def test_validate_defend_tightens_stop_and_resizes_pct_to_preserve_lots():
    # 1.0 lot held; tightening the stop from 1950 to 1960 while keeping
    # the SAME 1.0 lot requires a smaller pct than the original (smaller
    # stop distance / same lots = less risk-%) -- the exact correctness
    # trap pct_for_target_lots exists to close. Sizing uses the LIVE
    # price_current (1980), NOT the stale existing_entry.price (2000) --
    # the entry's own original suggestion price can have drifted well
    # past compute_rebalance_plan's own sanity-clamp band by the time a
    # tactical DEFEND fires (found on audit).
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1960.0, rule_citation="Schwager", numbers_citation="x")
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0, price_current=1980.0)
    existing = AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0, take_profit=2100.0, side="buy")
    spec = _spec(trade_contract_size=100.0)
    new_entry, tactical_state, _reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, None, 100_000.0, _get_spec_for({"XAUUSD": spec}),
    )
    assert new_entry is not None
    assert new_entry.stop_loss == 1960.0
    assert new_entry.price == 1980.0
    assert new_entry.take_profit == 2100.0  # DEFEND never touches the target
    # stop distance (1980-1960=20) x 1.0 lot x contract_size 100 = $2,000 = 2% of 100k
    assert new_entry.pct == pytest.approx(2.0)
    assert "Tactical DEFEND" in new_entry.reason
    assert tactical_state["tier_reached"] == "defend"
    assert tactical_state["defend_count"] == 1


def test_validate_defend_uses_live_price_not_stale_entry_price_for_sizing():
    # The real bug this closes: existing_entry.price is what the mega
    # session suggested potentially DAYS ago -- exactly the value most
    # likely to have drifted past compute_rebalance_plan's own 5% price-
    # sanity-clamp band in precisely the "mega session hasn't run in
    # days, position deteriorated" scenario this whole feature exists
    # to defend against. If sizing used that stale price while
    # compute_rebalance_plan later clamps to the live one, the two would
    # size against DIFFERENT stop distances, breaking the "same lots"
    # invariant. Here the entry's stale price (2000) is >15% away from
    # the live price (1700) -- sizing must follow the live price.
    verdict = TacticalVerdict(tier="defend", new_stop_loss=1650.0, rule_citation="Schwager", numbers_citation="x")
    position = _position(side="buy", volume=2.0, sl=1600.0, price_open=2000.0, price_current=1700.0)
    existing = AllocationEntry(pct=10.0, price=2000.0, stop_loss=1600.0, side="buy")
    spec = _spec(trade_contract_size=100.0)
    new_entry, _, _reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, None, 100_000.0, _get_spec_for({"XAUUSD": spec}),
    )
    assert new_entry is not None
    assert new_entry.price == 1700.0  # the live price, not the stale 2000
    # (1700-1650=50) x 2.0 lots x 100 contract size = $10,000 = 10% of 100k
    assert new_entry.pct == pytest.approx(10.0)


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
    # lots remaining, at the SAME (unchanged) stop. price_current
    # defaults to 1980 (see _position's own default).
    verdict = TacticalVerdict(tier="defend", partial_close_fraction=0.4, rule_citation="Schwager", numbers_citation="x")
    position = _position(side="buy", volume=1.0, sl=1950.0, price_open=2000.0)
    existing = AllocationEntry(pct=10.0, price=2000.0, stop_loss=1950.0, side="buy")
    spec = _spec(trade_contract_size=100.0)
    new_entry, _, _reason = _validate_and_apply_tactical_verdict(
        "XAUUSD", verdict, position, existing, None, 100_000.0, _get_spec_for({"XAUUSD": spec}),
    )
    assert new_entry is not None
    assert new_entry.stop_loss == 1950.0  # unchanged -- no new stop proposed
    # 0.6 lots x (1980-1950=30)-point stop x 100 contract size = $1,800 = 1.8% of 100k
    assert new_entry.pct == pytest.approx(1.8)


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
    # uses the LIVE price_current (1980, _position's own default), not
    # the fixed price_open (2000) -- same stale-price fix as
    # _validate_and_apply_tactical_verdict's own entry_price resolution.
    positions = [_position(symbol="XAUUSD", side="buy", volume=2.0, sl=1950.0, price_open=2000.0)]
    spec = _spec(trade_contract_size=100.0)
    backfilled = _backfill_settlement_for_held_positions(
        {}, positions, {}, 100_000.0, _get_spec_for({"XAUUSD": spec}),
    )
    assert backfilled["XAUUSD"]["state"] == "filled"
    assert backfilled["XAUUSD"]["origin"] == "pending_setup"
    assert backfilled["XAUUSD"]["entry"]["price"] == 1980.0
    # 2.0 lots x (1980-1950=30) stop x 100 contract size = $6,000 = 6% of 100k
    assert backfilled["XAUUSD"]["entry"]["pct"] == pytest.approx(6.0)


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


@patch("ai.copilot_execution._run_copilot_tactical_check")
@patch("ai.copilot_execution._fetch_technical_context", return_value="fake technical context")
@patch("ai.copilot_execution.modify_position_sltp")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
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

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = _spec()
        run_copilot_execution_check()

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


@patch("ai.copilot_execution._run_copilot_tactical_check")
@patch("ai.copilot_execution._fetch_technical_context", return_value="fake technical context")
@patch("ai.copilot_execution.modify_position_sltp")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
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

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = _spec(trade_contract_size=100.0)
        run_copilot_execution_check()

    mock_amend.assert_called_once()
    _called_position, called_sl, _called_tp = mock_amend.call_args[0]
    assert called_sl == 1960.0

    state = read_execution_state()
    assert state["last_tactical_verdicts"]["XAUUSD"]["applied"] is True
    settlement = read_settlement()
    assert settlement["settled"]["XAUUSD"]["tactical"]["tier_reached"] == "defend"
    assert settlement["settled"]["XAUUSD"]["tactical"]["defend_count"] == 1


@patch("ai.copilot_execution._run_copilot_tactical_check")
@patch("ai.copilot_execution._fetch_technical_context", return_value="fake technical context")
@patch("ai.copilot_execution.modify_position_sltp")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
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

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = _spec(trade_contract_size=100.0)
        run_copilot_execution_check()

    mock_amend.assert_not_called()
    state = read_execution_state()
    tactical = state["last_tactical_verdicts"]["XAUUSD"]
    assert tactical["applied"] is False
    assert tactical["shadow_mode"] is False  # NOT shadow mode -- it was genuinely rejected
    assert "never-widen-stop" in tactical["skipped_reason"]


@patch("ai.copilot_execution._run_copilot_tactical_check")
@patch("ai.copilot_execution._fetch_technical_context", return_value="fake technical context")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
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

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = _spec(trade_contract_size=100.0)
        run_copilot_execution_check()

    mock_tactical.assert_called_once()  # the real proof: it was checked at all, this same poll
    state = read_execution_state()
    assert state["last_tactical_verdicts"]["XAUUSD"]["tier"] == "hold"
    settlement = read_settlement()
    assert settlement["settled"]["XAUUSD"]["state"] == "filled"


@patch("ai.copilot_execution._run_copilot_tactical_check")
@patch("ai.copilot_execution._run_copilot_verdict")
@patch("ai.copilot_execution._run_copilot_invalidation_check")
@patch("ai.copilot_execution._fetch_technical_context", return_value="fake technical context")
@patch("ai.copilot_execution.modify_position_sltp")
@patch("ai.copilot_execution.close_position")
@patch("ai.copilot_execution.open_position")
@patch("ai.copilot_execution.is_trading_permitted", return_value=(True, ""))
@patch("ai.copilot_execution.fetch_ftmo_status")
@patch("ai.copilot_execution.get_market_watch")
@patch("ai.copilot_execution.get_pending_orders", return_value=[])
@patch("ai.copilot_execution.get_open_positions")
@patch("ai.copilot_execution.get_account_summary")
@patch("ai.copilot_execution.connect")
def test_pending_setup_invalidation_and_tactical_defend_all_fire_together(
    mock_connect, mock_account, mock_positions, mock_pending, mock_watch,
    mock_status, mock_permitted, mock_open, mock_close, mock_amend, mock_fetch_ctx,
    mock_invalidation, mock_verdict, mock_tactical, _fixed_files,
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

    def _tactical_side_effect(symbol, entry, position, technical_context, account_equity, prior_tactical):
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
    Path(config.COPILOT_EXECUTION_SETTLEMENT_FILE).write_text(json.dumps({
        "generated_utc": generated_utc,
        "settled": {
            "EURUSD": {"origin": "immediate", "state": "filled", "entry": eurusd_entry, "order_ticket": 200},
            "XAUUSD": {"origin": "immediate", "state": "filled", "entry": xauusd_entry, "order_ticket": 300},
        },
    }))
    set_tactical_defense_enabled(True)

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.side_effect = lambda symbol: {
            "GBPUSD": _spec(trade_contract_size=100_000.0),
            "EURUSD": _spec(trade_contract_size=100_000.0),
            "XAUUSD": _spec(trade_contract_size=100.0),
        }.get(symbol)
        run_copilot_execution_check()

    mock_open.assert_called_once()
    mock_close.assert_called_once()
    mock_amend.assert_called_once()

    state = read_execution_state()
    assert state["last_verdicts"]["GBPUSD"]["confirmed"] is True
    assert state["last_verdicts"]["EURUSD"]["confirmed"] is True
    assert state["last_tactical_verdicts"]["XAUUSD"]["tier"] == "defend"
    assert state["last_tactical_verdicts"]["XAUUSD"]["applied"] is True


@patch("ai.copilot_execution._run_copilot_tactical_check")
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

    with patch("ai.copilot_execution.get_contract_spec") as mock_spec:
        mock_spec.return_value = _spec(trade_contract_size=100_000.0)
        run_copilot_execution_check()

    mock_close.assert_called_once()
    state = read_execution_state()
    assert state["last_verdicts"]["EURUSD"]["confirmed"] is True
    tactical = state["last_tactical_verdicts"]["EURUSD"]
    assert tactical["tier"] == "exit"
    assert tactical["applied"] is False
    assert tactical["skipped_reason"]
