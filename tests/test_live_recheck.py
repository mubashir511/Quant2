import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import config
from ai import live_recheck as lr
from ai.portfolio_suggest import AllocationEntry, PendingSetup

NOW = datetime(2026, 9, 24, 18, 30, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(config, "LIVE_RECHECK_ENABLED", True)


def _rec(side="buy", entry=115.85, stop=115.07, target=117.70, pct=0.8, bid=117.18, ask=117.19, atr=0.29,
         floor=0.58, status=None, kind="immediate", anchors=None, symbol="SOLUSD"):
    rec = lr.SymbolRecheck(symbol, kind, side, entry, stop, target, pct, "ok", bid=bid, ask=ask, atr=atr, floor_distance=floor)
    rec.anchors = anchors if anchors is not None else [
        ("M5 support band 116.80-116.95", 116.80, 116.95),
        ("M5 resistance band 117.80-117.95", 117.80, 117.95),
        ("session high 117.87", 117.87, 117.87),
        ("live ask 117.19", ask, ask),
        ("live bid 117.18", bid, bid),
    ]
    rec.status, rec.drift_atr = lr._classify(side, entry, stop, target, bid, ask, atr)
    if status:
        rec.status = status
    return rec


def _sheet(*recs):
    return lr.RecheckSheet(NOW.isoformat(), {r.symbol: r for r in recs}, "block")


def _draft_alloc(side="buy", price=115.85, stop=115.07, tp=117.70, pct=0.8):
    return {"SOLUSD": AllocationEntry(pct=pct, price=price, stop_loss=stop, take_profit=tp, side=side, reason="draft thesis",
                                      invalidation_condition="M5 closes below 115.07")}


def _payload(price=115.85, stop=115.07, tp=117.70, pct=0.8, side="buy", pending=None):
    return {
        "immediate_allocation": {"SOLUSD": {"pct": pct, "price": price, "stop_loss": stop, "take_profit": tp, "side": side,
                                            "reason": "final thesis", "invalidation_condition": "M5 closes below 115.07"}},
        "pending_setups": pending or [],
    }


# --- parsing / classification ---------------------------------------------------------------------------


def test_snapshot_prices_are_parsed_from_the_summary_headers():
    summary = "- SOLUSD (Solana vs US Dollar, Crypto): bid 116.32, ask 116.34\n  execution: x\n- MSFT (Microsoft, Spot CFD): bid 494.58, ask 494.93\n"
    prices = lr._snapshot_prices(summary)
    assert prices["SOLUSD"] == pytest.approx(116.33) and prices["MSFT"] == pytest.approx(494.755)


def test_classify_flags_only_real_drift_and_dead_levels():
    # buy, entry 115.85, ask 117.19, ATR 0.29 -> 4.6 ATR below the ask
    assert lr._classify("buy", 115.85, 115.07, 117.70, 117.18, 117.19, 0.29)[0] == "DRIFTED"
    assert lr._classify("buy", 116.60, 115.80, 118.0, 117.18, 117.19, 0.29)[0] == "ok"           # 2.0 ATR
    assert lr._classify("buy", 117.30, 116.50, 118.5, 117.18, 117.19, 0.29)[0] == "MARKETABLE"   # market is through it, mildly
    status, drift = lr._classify("buy", 118.50, 116.0, 120.0, 117.18, 117.19, 0.29)
    assert status == "DRIFTED" and drift < 0                                                      # market 4.5 ATR THROUGH the entry
    assert lr._classify("buy", 115.85, 117.20, 118.0, 117.18, 117.19, 0.29)[0] == "STOP_BREACHED"
    assert lr._classify("buy", 115.85, 115.07, 117.10, 117.18, 117.19, 0.29)[0] == "TARGET_PASSED"
    assert lr._classify("sell", 118.6, 119.4, 116.0, 117.18, 117.19, 0.29)[0] == "DRIFTED"        # entry 4.8 ATR above the bid
    assert lr._classify("sell", 117.0, 117.9, 115.0, 117.18, 117.19, 0.29)[0] == "MARKETABLE"
    assert lr._classify("buy", 116.0, 115.0, 118.0, 117.18, 117.19, None)[0] == "NO_LIVE_DATA"


# --- the deterministic level checks ---------------------------------------------------------------------


def test_a_grounded_reissue_passes_every_check():
    rec = _rec()
    # entry on the fresh support band (1.1 ATR below the ask), stop at the minimum distance, target on the fresh resistance band
    problems, notes = lr._check_levels("buy", 116.87, 116.29, 117.94, rec, 117.18, 117.19, (115.85, 115.07, 117.70), True)
    assert problems == [], problems
    assert any("entry anchored to M5 support band" in n for n in notes) and any("target anchored to M5 resistance band" in n for n in notes)


@pytest.mark.parametrize(
    "entry,stop,target,needle",
    [
        (116.90, 116.35, 117.87, "tighter than the minimum"),         # 0.55 stop < 0.58 floor
        (115.85, 115.00, 117.87, "entry sits"),                       # 4.6 ATR from the live price
        (116.87, 116.20, 117.55, "target 117.55"),                    # target is not a real level
        (116.87, 116.29, 117.94, None),                               # passes (control)
        (116.87, 116.30, 117.87, "net reward:risk"),                  # 1.0 vs 0.57 risk -> ratio too low
        (116.13, 116.20, 117.87, "not ordered"),                      # entry below stop
    ],
)
def test_reissue_violations_are_each_caught(entry, stop, target, needle):
    rec = _rec()
    problems, _ = lr._check_levels("buy", entry, stop, target, rec, 117.18, 117.19, (115.85, 115.07, 117.70), True)
    if needle is None:
        assert problems == []
    else:
        assert any(needle in p for p in problems), problems


def test_an_unanchored_entry_is_rejected_even_when_it_is_near_the_market():
    rec = _rec()
    problems, _ = lr._check_levels("buy", 116.55, 115.90, 117.87, rec, 117.18, 117.19, (115.85, 115.07, 117.70), True)
    assert any("entry 116.55 is not within tolerance" in p for p in problems)


def test_the_reach_limit_and_max_stop_are_enforced():
    rec = _rec(anchors=[("far target", 120.0, 120.0), ("live ask 117.19", 117.19, 117.19)])
    problems, _ = lr._check_levels("buy", 117.19, 116.60, 120.0, rec, 117.18, 117.19, (115.85, 115.07, 117.70), True)
    assert any("reach limit" in p for p in problems)
    rec2 = _rec()
    problems2, _ = lr._check_levels("buy", 117.19, 114.0, 117.87, rec2, 117.18, 117.19, (115.85, 115.07, 117.70), True)
    assert any("wider than" in p for p in problems2)


# --- apply_recheck_validation ---------------------------------------------------------------------------


def test_a_verified_reissue_is_kept_and_annotated():
    payload = _payload(price=116.87, stop=116.29, tp=117.94)
    notes = lr.apply_recheck_validation(payload, _draft_alloc(), [], _sheet(_rec()), {"SOLUSD": (117.18, 117.19)})
    item = payload["immediate_allocation"]["SOLUSD"]
    assert (item["price"], item["stop_loss"], item["take_profit"]) == (116.87, 116.29, 117.94)
    assert "re-issue verified against a fresh quote" in item["reason"]
    assert any("re-issue accepted" in n for n in notes)


def test_a_fabricated_reissue_is_reverted_to_the_drafted_levels():
    payload = _payload(price=116.55, stop=115.90, tp=117.87)  # entry not on any real level
    notes = lr.apply_recheck_validation(payload, _draft_alloc(), [], _sheet(_rec()), {"SOLUSD": (117.18, 117.19)})
    item = payload["immediate_allocation"]["SOLUSD"]
    assert (item["price"], item["stop_loss"], item["take_profit"], item["pct"]) == (115.85, 115.07, 117.70, 0.8)
    assert "re-issue REJECTED" in item["reason"] and "kept the drafted levels" in item["reason"]
    assert any("rejected" in n for n in notes)


def test_a_rejected_reissue_of_already_dead_levels_is_dropped_not_reverted():
    rec = _rec(target=117.10, status=None)          # the drafted target is already passed by the live price
    assert rec.status == "TARGET_PASSED"
    payload = _payload(price=116.55, stop=115.90, tp=117.87)
    lr.apply_recheck_validation(payload, _draft_alloc(tp=117.10), [], _sheet(rec), {"SOLUSD": (117.18, 117.19)})
    item = payload["immediate_allocation"]["SOLUSD"]
    assert item["pct"] == 0.0 and "already dead" in item["reason"]


def test_raising_pct_is_clamped_to_the_drafted_risk_and_flipping_side_is_rejected():
    payload = _payload(price=116.87, stop=116.29, tp=117.94, pct=1.5)
    notes = lr.apply_recheck_validation(payload, _draft_alloc(), [], _sheet(_rec()), {"SOLUSD": (117.18, 117.19)})
    item = payload["immediate_allocation"]["SOLUSD"]
    assert item["pct"] == 0.8 and item["price"] == 116.87  # risk never rises, but a valid re-issue is not thrown away for it
    assert "pct clamped from 1.5 to the drafted 0.8" in item["reason"] and any("pct clamped" in n for n in notes)
    payload2 = _payload(price=116.87, stop=116.29, tp=117.94, side="sell")
    lr.apply_recheck_validation(payload2, _draft_alloc(), [], _sheet(_rec()), {"SOLUSD": (117.18, 117.19)})
    assert payload2["immediate_allocation"]["SOLUSD"]["side"] == "buy"


def test_unchanged_and_dropped_entries_are_not_touched_but_a_still_drifted_one_is_noted():
    payload = _payload()
    notes = lr.apply_recheck_validation(payload, _draft_alloc(), [], _sheet(_rec()), {"SOLUSD": (117.18, 117.19)})
    assert payload["immediate_allocation"]["SOLUSD"]["reason"] == "final thesis"
    assert any("still DRIFTED" in n for n in notes)
    dropped = _payload(pct=0.0)
    notes2 = lr.apply_recheck_validation(dropped, _draft_alloc(), [], _sheet(_rec()), {"SOLUSD": (117.18, 117.19)})
    assert dropped["immediate_allocation"]["SOLUSD"]["pct"] == 0.0 and any("dropped it" in n for n in notes2)


def test_the_distance_check_uses_the_fresh_quote_not_the_sheet_quote():
    payload = _payload(price=116.87, stop=116.29, tp=117.94)
    # By validation time price fell to 115.20: the 'near the market' re-issue is now 1.7 ATR ABOVE... i.e. marketable-through
    lr.apply_recheck_validation(payload, _draft_alloc(), [], _sheet(_rec()), {"SOLUSD": (115.19, 115.20)})
    item = payload["immediate_allocation"]["SOLUSD"]
    assert "REJECTED" in item["reason"]


def test_a_valid_move_to_pending_is_kept_and_a_failed_one_is_undone():
    pending_ok = {"symbol": "SOLUSD", "side": "buy", "pct": 0.8, "trigger_condition": "M5 reclaims 116.87", "price": 116.87,
                  "stop_loss": 116.29, "take_profit": 117.94, "reason": "moved"}
    payload = _payload(pct=0.0, pending=[pending_ok])
    lr.apply_recheck_validation(payload, _draft_alloc(), [], _sheet(_rec()), {"SOLUSD": (117.18, 117.19)})
    assert payload["pending_setups"] == [pending_ok] and "verified" in pending_ok["reason"]

    pending_bad = dict(pending_ok, price=116.55, reason="moved")
    payload2 = _payload(pct=0.0, pending=[pending_bad])
    lr.apply_recheck_validation(payload2, _draft_alloc(), [], _sheet(_rec()), {"SOLUSD": (117.18, 117.19)})
    assert payload2["pending_setups"] == []
    restored = payload2["immediate_allocation"]["SOLUSD"]
    assert (restored["price"], restored["pct"]) == (115.85, 0.8) and "move to Pending Setups REJECTED" in restored["reason"]


def test_held_symbols_and_symbols_without_a_quote_are_skipped():
    payload = _payload(price=116.55, stop=115.90, tp=117.87)
    held = _rec(status="HELD")
    assert lr.apply_recheck_validation(payload, _draft_alloc(), [], _sheet(held), {}) == []
    assert payload["immediate_allocation"]["SOLUSD"]["price"] == 116.55  # untouched


# --- build_live_recheck ---------------------------------------------------------------------------------


def _fake_analysis(ask=117.19, bid=117.18, atr=0.29):
    from analysis.chart_structure import ChartStructureSnapshot, SRLevel, SRLevelsResult
    from analysis.intraday_context import IntradayLevels
    from analysis.technical import TechnicalStats

    fields = {f: None for f in TechnicalStats.__dataclass_fields__}
    fields.update(last_price=ask, atr=atr, atr_pct=atr / ask * 100, trend="uptrend", market_regime="trending_up")
    stats = TechnicalStats(**fields)
    structure = ChartStructureSnapshot(
        fibonacci=None, patterns=[], trendlines=None,
        sr_levels=SRLevelsResult(
            support_levels=[SRLevel(price=116.87, touches=3, distance_pct=-0.3, low=116.80, high=116.95)],
            resistance_levels=[SRLevel(price=117.87, touches=2, distance_pct=0.6, low=117.80, high=117.95)],
        ),
    )
    levels = IntradayLevels(ask, 117.9, 115.8, 116.5, 116.4, 117.87, 116.05, 116.9, 1.6, 90.0, 200)
    return SimpleNamespace(
        m5_stats=stats, m5_structure=structure, m5_divergence=None, intraday_levels=levels, h1_stats=stats,
        base=SimpleNamespace(ask=ask, bid=bid), trade_cost=None, m5_level_reliability={}, m5_atr_pct_median=0.2,
    )


DRAFT_TEXT = (
    "Draft.\n```json\n{\"SOLUSD\": {\"side\": \"buy\", \"pct\": 0.8, \"price\": 115.85, \"stop_loss\": 115.07, "
    "\"take_profit\": 117.7, \"reason\": \"pullback\", \"invalidation_condition\": \"M5 closes below 115.07\"}, \"CASH\": 99.2}\n```\n"
)
SUMMARY = "- SOLUSD (Solana vs US Dollar, Crypto): bid 116.32, ask 116.34\n"


@patch("data.mt5_source.get_open_positions", return_value=[])
@patch("data.mt5_source.get_market_watch")
@patch("ai.ftmo_suggest.analyze_ftmo_asset_live")
def test_build_live_recheck_flags_the_drifted_entry_and_prints_only_measured_numbers(mock_analyze, mock_watch, _pos):
    mock_watch.return_value = [SimpleNamespace(symbol="SOLUSD", bid=117.18, ask=117.19, description="Solana")]
    mock_analyze.return_value = _fake_analysis()
    sheet = lr.build_live_recheck(DRAFT_TEXT, SUMMARY, now_utc=NOW, snapshot_utc=datetime(2026, 9, 24, 18, 5, tzinfo=timezone.utc))
    rec = sheet.symbols["SOLUSD"]
    assert rec.status == "DRIFTED" and rec.drift_atr == pytest.approx((117.19 - 115.85) / 0.29)
    assert sheet.flagged() == ["SOLUSD"]
    block = sheet.prompt_block
    assert "about 25 minutes after the data snapshot" in block
    assert "Do NOT use WebSearch/WebFetch, memory, or your own arithmetic" in block
    assert "STATUS: DRIFTED" in block and "4.6 M5 ATR below the live ask" in block
    assert "the price in your data snapshot was ~116.33" in block and "+2.9 M5 ATR" in block
    assert "measured fill odds" in block and "re-checked live at 18:30 UTC" in block
    assert "FRESH M5 chart structure" in block and "Fresh M5 trade-zone candidate" in block
    assert "1 entry need a decision (SOLUSD)" in block


@patch("data.mt5_source.get_open_positions")
@patch("data.mt5_source.get_market_watch")
@patch("ai.ftmo_suggest.analyze_ftmo_asset_live")
def test_build_live_recheck_skips_held_symbols_and_degrades_open(mock_analyze, mock_watch, mock_pos):
    mock_watch.return_value = [SimpleNamespace(symbol="SOLUSD", bid=117.18, ask=117.19, description="Solana")]
    mock_pos.return_value = [SimpleNamespace(symbol="SOLUSD")]
    assert lr.build_live_recheck(DRAFT_TEXT, SUMMARY, now_utc=NOW) is None          # only a held symbol -> nothing to check
    mock_analyze.assert_not_called()
    mock_pos.return_value = []
    mock_analyze.side_effect = RuntimeError("mt5 down")
    sheet = lr.build_live_recheck(DRAFT_TEXT, SUMMARY, now_utc=NOW)
    assert sheet.symbols["SOLUSD"].status == "NO_LIVE_DATA" and "leave this entry exactly as drafted" in sheet.prompt_block


def test_build_live_recheck_returns_none_when_disabled_or_the_draft_is_unparseable(monkeypatch):
    assert lr.build_live_recheck("no json here", SUMMARY, now_utc=NOW) is None
    monkeypatch.setattr(config, "LIVE_RECHECK_ENABLED", False)
    assert lr.build_live_recheck(DRAFT_TEXT, SUMMARY, now_utc=NOW) is None


# --- integration with the suggestion writer ------------------------------------------------------------


@patch("data.mt5_source.get_market_watch")
def test_write_latest_suggestion_verifies_the_final_answer_against_the_recheck(mock_watch, tmp_path, monkeypatch):
    from ai.ftmo_suggest import _write_latest_suggestion

    monkeypatch.setattr(config, "MEGA_ANALYSIS_LATEST_SUGGESTION_FILE", str(tmp_path / "latest.json"))
    mock_watch.return_value = [SimpleNamespace(symbol="SOLUSD", bid=117.18, ask=117.19, description="Solana")]
    final = (
        "Final.\n```json\n{\"SOLUSD\": {\"side\": \"buy\", \"pct\": 0.8, \"price\": 116.55, \"stop_loss\": 115.9, "
        "\"take_profit\": 117.87, \"reason\": \"re-issued\", \"invalidation_condition\": \"M5 closes below 115.9\"}, \"CASH\": 99.2}\n```\n"
    )
    with patch("ai.ftmo_suggest.trade_journal.record_proposals"):
        _write_latest_suggestion(final, recheck=(DRAFT_TEXT, _sheet(_rec())))
    saved = json.loads((tmp_path / "latest.json").read_text())
    sol = saved["immediate_allocation"]["SOLUSD"]
    assert (sol["price"], sol["stop_loss"], sol["take_profit"]) == (115.85, 115.07, 117.7)   # the fabricated level never reached the Clerk
    assert saved["live_recheck"]["statuses"] == {"SOLUSD": "DRIFTED"}
    assert any("rejected" in n for n in saved["live_recheck"]["notes"])


@patch("data.mt5_source.get_market_watch", side_effect=RuntimeError("feed down"))
def test_write_latest_suggestion_still_writes_when_the_validation_itself_fails(_watch, tmp_path, monkeypatch):
    from ai.ftmo_suggest import _write_latest_suggestion

    monkeypatch.setattr(config, "MEGA_ANALYSIS_LATEST_SUGGESTION_FILE", str(tmp_path / "latest.json"))
    with patch("ai.ftmo_suggest.trade_journal.record_proposals"):
        _write_latest_suggestion(DRAFT_TEXT, recheck=(DRAFT_TEXT, _sheet(_rec())))
    saved = json.loads((tmp_path / "latest.json").read_text())
    assert saved["immediate_allocation"]["SOLUSD"]["price"] == 115.85 and "live_recheck" not in saved


# --- entry_mode re-issues (2026-09-25) ------------------------------------------------------------------

_MODE_ANCHORS = [
    ("M5 resistance band 117.50-117.55", 117.50, 117.55),
    ("M5 resistance band 118.60-118.70", 118.60, 118.70),
    ("M5 resistance band 118.25-118.35", 118.25, 118.35),
    ("live ask 117.19", 117.19, 117.19),
    ("live bid 117.18", 117.18, 117.18),
]


def _mode_payload(mode, price, stop, tp):
    payload = _payload(price=price, stop=stop, tp=tp)
    payload["immediate_allocation"]["SOLUSD"]["entry_mode"] = mode
    return payload


def test_a_market_reissue_at_the_live_price_with_a_grounded_target_is_accepted():
    payload = _mode_payload("market", 117.19, 116.61, 118.30)
    notes = lr.apply_recheck_validation(payload, _draft_alloc(), [], _sheet(_rec(anchors=_MODE_ANCHORS)), {"SOLUSD": (117.18, 117.19)})
    item = payload["immediate_allocation"]["SOLUSD"]
    assert item["entry_mode"] == "market" and (item["price"], item["stop_loss"], item["take_profit"]) == (117.19, 116.61, 118.30)
    assert "entry_mode market confirmed on the live quote" in item["reason"]
    assert any("re-issue accepted" in n for n in notes)


def test_a_market_reissue_that_states_a_pullback_price_is_rejected_and_the_draft_mode_restored():
    payload = _mode_payload("market", 116.87, 116.29, 117.94)
    lr.apply_recheck_validation(payload, _draft_alloc(), [], _sheet(_rec()), {"SOLUSD": (117.18, 117.19)})
    item = payload["immediate_allocation"]["SOLUSD"]
    assert item["entry_mode"] == "limit" and (item["price"], item["stop_loss"]) == (115.85, 115.07)
    assert "market entry states" in item["reason"]


def test_a_breakout_stop_reissue_on_a_printed_level_is_accepted_and_one_that_is_too_far_is_not():
    ok = _mode_payload("stop", 117.55, 116.97, 118.65)
    lr.apply_recheck_validation(ok, _draft_alloc(), [], _sheet(_rec(anchors=_MODE_ANCHORS)), {"SOLUSD": (117.18, 117.19)})
    item = ok["immediate_allocation"]["SOLUSD"]
    assert item["entry_mode"] == "stop" and item["price"] == 117.55 and "entry_mode stop confirmed" in item["reason"]

    far_anchors = _MODE_ANCHORS + [("M5 resistance band 117.95-118.00", 117.95, 118.00)]
    far = _mode_payload("stop", 118.00, 117.42, 119.00)  # 2.8 M5 ATR beyond the ask
    lr.apply_recheck_validation(far, _draft_alloc(), [], _sheet(_rec(anchors=far_anchors)), {"SOLUSD": (117.18, 117.19)})
    assert far["immediate_allocation"]["SOLUSD"]["entry_mode"] == "limit"
    assert "stop-order trigger sits" in far["immediate_allocation"]["SOLUSD"]["reason"]


def test_a_stop_trigger_must_be_a_printed_level_not_an_invented_one():
    payload = _mode_payload("stop", 117.75, 117.17, 118.95)
    lr.apply_recheck_validation(payload, _draft_alloc(), [], _sheet(_rec(anchors=_MODE_ANCHORS)), {"SOLUSD": (117.18, 117.19)})
    item = payload["immediate_allocation"]["SOLUSD"]
    assert item["entry_mode"] == "limit" and "not within tolerance" in item["reason"]


def test_entry_mode_on_a_pending_setup_is_not_allowed():
    payload = _payload(pending=[{"symbol": "SOLUSD", "side": "buy", "pct": 0.8, "price": 116.87, "stop_loss": 116.29,
                                 "take_profit": 117.94, "trigger_condition": "M5 close above 117.5", "reason": "x", "entry_mode": "market"}])
    payload["immediate_allocation"]["SOLUSD"]["pct"] = 0.0
    notes = lr.apply_recheck_validation(payload, _draft_alloc(), [], _sheet(_rec()), {"SOLUSD": (117.18, 117.19)})
    assert payload["pending_setups"] == []  # the failed move is undone
    assert any("entry_mode applies to immediate entries only" in n for n in notes)
    assert payload["immediate_allocation"]["SOLUSD"]["entry_mode"] == "limit" and payload["immediate_allocation"]["SOLUSD"]["pct"] == 0.8


def test_the_entry_mode_menu_prints_only_verified_options_from_the_live_quote():
    text = "\n".join(lr._entry_mode_menu(_rec(anchors=_MODE_ANCHORS, entry=116.87)))
    assert "ENTRY MODES available now" in text
    assert '"limit" at your drafted 116.87' in text and "chance it is touched within 3h" in text
    assert '"market" at the live ask 117.19: available' in text
    assert '"stop" (breakout trigger) at the nearest printed resistance edge 117.55' in text and "accepted as a buy stop" in text
    assert lr._entry_mode_menu(_rec(kind="pending")) == []
    assert lr._entry_mode_menu(_rec(status="TARGET_PASSED", target=117.10)) == []


def test_the_rules_tell_claude_how_to_choose_an_entry_mode_and_that_it_is_re_verified():
    assert "ENTRY MODES" in lr._RULES and "\"entry_mode\"" in lr._RULES and "Never invent a trigger" in lr._RULES


def test_fill_odds_interpolate_the_measured_table():
    from analysis.entry_mode import fill_odds_pct

    assert fill_odds_pct(1.0) == pytest.approx(79) and fill_odds_pct(2.3) == pytest.approx(55)
    assert fill_odds_pct(1.3) == pytest.approx(73)  # midway 79 -> 67
    assert fill_odds_pct(0.0) == 100 and fill_odds_pct(-1) == 100 and fill_odds_pct(12) == 26


# --- shortlist gaps (2026-09-25) --------------------------------------------------------------------------

HUNT_SUMMARY = (
    "- SOLUSD (Solana vs US Dollar, Crypto): bid 116.32, ask 116.34\n"
    "POSITION HUNT (deterministic, Python - ranked ...):\n"
    "  1. SOLUSD BUY [tier A] - WITH the D1 and H4 trend; cost drag 0.05R at the tightest stop\n"
    "  2. AMD BUY [tier A] - WITH the D1 and H4 trend; M5 trend+regime agree (up); cost drag 0.05R at the tightest stop\n"
    "  3. XAUUSD SELL [tier B] - partly with the higher-timeframe trend\n"
    "  Vetoed (hard, objective - cite the id to leave one out):\n"
    "    - WHEAT.c SELL [tier A]: V1: round-trip cost is 1.40R\n"
)


def test_the_shortlist_is_parsed_from_the_summary_and_stops_at_the_vetoed_list():
    items = lr._shortlist_from_summary(HUNT_SUMMARY)
    assert [(s, side, tier) for s, side, tier, _ in items] == [("SOLUSD", "buy", "A"), ("AMD", "buy", "A"), ("XAUUSD", "sell", "B")]
    assert lr._shortlist_from_summary("no hunt block here") == []


@patch("data.mt5_source.get_open_positions", return_value=[])
@patch("data.mt5_source.get_market_watch")
@patch("ai.ftmo_suggest.analyze_ftmo_asset_live")
def test_shortlisted_candidates_missing_from_the_draft_are_named_in_the_final_revision_prompt(mock_analyze, mock_watch, _pos):
    mock_watch.return_value = [SimpleNamespace(symbol="SOLUSD", bid=117.18, ask=117.19, description="Solana")]
    mock_analyze.return_value = _fake_analysis()
    sheet = lr.build_live_recheck(DRAFT_TEXT, HUNT_SUMMARY, now_utc=NOW)
    block = sheet.prompt_block
    assert "SHORTLIST GAPS" in block and "- AMD BUY [tier A]" in block and "- XAUUSD SELL [tier B]" in block
    assert "- SOLUSD BUY" not in block.split("SHORTLIST GAPS")[1]  # SOLUSD IS in the draft
    assert "WHEAT" not in block.split("SHORTLIST GAPS")[1]  # vetoed candidates are not gaps
    assert "Do not invent a number" in block and "NO-INFORMATION backtest never is" in block


@patch("data.mt5_source.get_open_positions", return_value=[SimpleNamespace(symbol="AMD")])
def test_a_draft_with_no_entries_still_gets_the_gap_check_and_held_symbols_are_not_gaps(_pos):
    empty_draft = "Draft.\n```json\n{\"CASH\": 100.0}\n```\n"
    sheet = lr.build_live_recheck(empty_draft, HUNT_SUMMARY, now_utc=NOW)
    assert sheet is not None and sheet.symbols == {} and sheet.flagged() == []
    assert "- SOLUSD BUY" in sheet.prompt_block and "- XAUUSD SELL" in sheet.prompt_block
    assert "- AMD BUY" not in sheet.prompt_block  # already an open position


@patch("data.mt5_source.get_open_positions", return_value=[])
def test_no_gaps_and_no_entries_means_no_recheck_at_all(_pos):
    assert lr.build_live_recheck("Draft.\n```json\n{\"CASH\": 100.0}\n```\n", "no hunt block", now_utc=NOW) is None


# --- playbook in the live re-check (2026-09-25) ------------------------------------------------------------

def test_the_range_extremes_are_real_levels_a_breakout_trigger_may_anchor_to():
    from analysis.playbook import RangeRead

    analysis = _fake_analysis()
    analysis.m5_range = RangeRead(high=117.60, low=116.90, bars=24, vol_ratio=1.3, last_close=117.19)
    labels = [label for label, *_ in lr._anchors_from_analysis(analysis, 117.18, 117.19)]
    assert "M5 24-bar range high 117.6" in labels and "M5 24-bar range low 116.9" in labels
    assert not any("range" in label for label, *_ in lr._anchors_from_analysis(_fake_analysis(), 117.18, 117.19))


def test_the_menu_prints_the_playbook_advice_when_the_fresh_read_has_one():
    rec = _rec(anchors=_MODE_ANCHORS)
    rec.playbook_text = "TREND-BREAKOUT: buy-STOP at 117.6 (...)"
    text = "\n".join(lr._entry_mode_menu(rec))
    assert "PLAYBOOK (with-the-trend breakout, Python-measured from this fresh read): TREND-BREAKOUT: buy-STOP at 117.6" in text
    assert "PLAYBOOK" not in "\n".join(lr._entry_mode_menu(_rec(anchors=_MODE_ANCHORS)))


def test_a_failed_move_to_immediate_restores_the_drafted_pending_setup_instead_of_leaving_a_zero_row():
    draft_pending = [PendingSetup(symbol="SOLUSD", side="buy", pct=0.4, trigger_condition="M5 reclaims 116.87", price=116.87,
                                  stop_loss=116.29, take_profit=117.94, reason="drafted", trigger={"kind": "reclaim", "level": 116.87, "within_bars": 8})]
    rec = _rec(kind="pending", entry=116.87, stop=116.29, target=117.94, pct=0.4)
    assert rec.status != "STOP_BREACHED"
    payload = _payload(price=116.55, stop=115.90, tp=117.87, pct=0.4)  # a fabricated move to immediate (not on a real level)
    lr.apply_recheck_validation(payload, {}, draft_pending, _sheet(rec), {"SOLUSD": (117.18, 117.19)})
    assert "SOLUSD" not in payload["immediate_allocation"]
    restored = payload["pending_setups"][0]
    assert restored["symbol"] == "SOLUSD" and restored["pct"] == 0.4 and restored["trigger"]["kind"] == "reclaim"
    assert "kept the drafted Pending Setup" in restored["reason"]


def test_a_failed_move_of_dead_drafted_levels_leaves_nothing_behind_not_a_zero_percent_row():
    draft_pending = [PendingSetup(symbol="SOLUSD", side="buy", pct=0.4, trigger_condition="x", price=115.85, stop_loss=117.20,
                                  take_profit=118.0, reason="drafted")]
    rec = _rec(kind="pending", entry=115.85, stop=117.20, target=118.0, pct=0.4)
    assert rec.status == "STOP_BREACHED"
    payload = _payload(price=116.55, stop=115.90, tp=117.87, pct=0.4)
    notes = lr.apply_recheck_validation(payload, {}, draft_pending, _sheet(rec), {"SOLUSD": (117.18, 117.19)})
    assert "SOLUSD" not in payload["immediate_allocation"] and payload["pending_setups"] == []
    assert any("dead" in n for n in notes)
