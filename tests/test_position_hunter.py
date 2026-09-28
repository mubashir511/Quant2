import pytest

import config
from analysis.position_hunter import HuntFacts, format_position_hunt, hunt


def _f(symbol, **kw):
    defaults = dict(d1_dir="up", h4_dir="up", m5_dir="up", cost_drag_r=0.05, has_structure=True,
                    zone_rr={"buy": 2.5, "sell": None}, market_open=True)
    return HuntFacts(symbol, **{**defaults, **kw})


def _syms(cands):
    return [(c.symbol, c.side) for c in cands]


def test_aligned_symbols_rank_ahead_of_counter_trend_and_take_the_aligned_side():
    result = hunt([
        _f("MSFT", d1_dir="down", h4_dir="down", m5_dir="down", zone_rr={"sell": 2.2}),
        _f("XAUUSD"),
        _f("EURUSD", d1_dir="up", h4_dir="down", m5_dir="up"),  # mixed higher timeframes -> tier C, M5 with the side
    ])
    assert _syms(result.shortlist)[:2] == [("XAUUSD", "buy"), ("MSFT", "sell")] or _syms(result.shortlist)[:2] == [("MSFT", "sell"), ("XAUUSD", "buy")]
    assert result.shortlist[-1].symbol == "EURUSD" and result.shortlist[-1].tier == "C"
    assert all(c.tier == "A" for c in result.shortlist[:2])


def test_no_direction_anywhere_is_no_read_not_a_candidate():
    result = hunt([_f("FLAT", d1_dir="flat", h4_dir="flat", m5_dir=None), _f("NONE", d1_dir=None, h4_dir=None, m5_dir=None)])
    assert result.shortlist == [] and result.no_read == ["FLAT", "NONE"]


def test_a_counter_trend_side_with_a_silent_m5_is_not_a_hunt_candidate():
    result = hunt([_f("X", d1_dir="up", h4_dir="down", m5_dir=None)])
    assert result.shortlist == [] and result.no_read == ["X"]


def test_m5_only_setup_is_tier_m_and_ranks_below_aligned_but_above_counter():
    result = hunt([
        _f("ONLYM5", d1_dir="flat", h4_dir="flat", m5_dir="up"),
        _f("ALIGNED"),
        _f("COUNTER", d1_dir="up", h4_dir="down", m5_dir="up", zone_rr={"buy": 2.0}),
    ])
    assert [c.symbol for c in result.shortlist] == ["ALIGNED", "ONLYM5", "COUNTER"]
    assert [c.tier for c in result.shortlist] == ["A", "M", "C"]


def test_v1_cost_drag_veto_uses_the_configured_limit():
    result = hunt([_f("WHEAT", cost_drag_r=1.4), _f("OK", cost_drag_r=config.COST_DRAG_VETO_R)])
    assert [c.symbol for c in result.shortlist] == ["OK"]  # exactly at the limit is not above it
    assert result.vetoed[0].symbol == "WHEAT" and result.vetoed[0].vetoes[0][0] == "V1"


def test_v2_v3_v7_are_reported_with_their_own_ids_and_can_stack():
    result = hunt([
        _f("A", has_structure=False),
        _f("B", blackout_reason="US CPI in 12 min"),
        _f("C", market_open=False),
        _f("D", minutes_to_close=config.NO_NEW_ORDER_MINUTES_BEFORE_CLOSE - 5),
        _f("E", has_structure=False, market_open=False),
    ])
    ids = {c.symbol: [v[0] for v in c.vetoes] for c in result.vetoed}
    assert ids == {"A": ["V2"], "B": ["V3"], "C": ["V7"], "D": ["V7"], "E": ["V2", "V7"]}
    assert result.shortlist == []


def test_unknown_inputs_never_create_a_veto():
    result = hunt([_f("U", cost_drag_r=None, has_structure=None, market_open=None, minutes_to_close=None,
                      min_viable_pct=None, blackout_reason=None)], risk_budget_pct=0.5)
    assert [c.symbol for c in result.shortlist] == ["U"] and result.vetoed == []


def test_v4_minimum_viable_size_against_the_risk_budget():
    result = hunt([_f("BIG", min_viable_pct=0.9), _f("SMALL", min_viable_pct=0.2)], risk_budget_pct=0.5)
    assert [c.symbol for c in result.shortlist] == ["SMALL"]
    assert result.vetoed[0].vetoes[0][0] == "V4"
    # No budget known (no FTMO status): the size veto cannot fire.
    assert len(hunt([_f("BIG", min_viable_pct=0.9)], risk_budget_pct=None).shortlist) == 1


def test_a_contradicted_backtest_is_context_only_never_a_veto_and_never_moves_the_rank():
    contradicted = {"buy": {"rsi": "contradicted", "sr": "no_information"}}
    supported = {"buy": {"rsi": "supported", "sr": "no_information"}}
    result = hunt([_f("BAD", edge_verdicts=contradicted), _f("GOOD", edge_verdicts=supported), _f("PLAIN")])
    assert {c.symbol for c in result.shortlist} == {"BAD", "GOOD", "PLAIN"} and result.vetoed == []
    scores = {c.symbol: c.score for c in result.shortlist}
    assert scores["BAD"] == scores["GOOD"] == scores["PLAIN"]
    assert any("context for sizing only" in n for n in next(c for c in result.shortlist if c.symbol == "BAD").notes)


def test_no_information_backtest_is_never_a_veto_and_is_labelled_as_such():
    result = hunt([_f("N", edge_verdicts={"buy": {"rsi": "no_information"}})])
    assert len(result.shortlist) == 1
    assert any("no information" in note for note in result.shortlist[0].notes)


def test_correlation_is_an_informational_note_only_and_never_removes_a_candidate():
    pairs = [("GOLD", "SILVER", 0.9), ("USD", "EUR", -0.85)]
    same = hunt([_f("GOLD"), _f("SILVER", zone_rr={"buy": None})], pairs)
    assert [c.symbol for c in same.shortlist] == ["GOLD", "SILVER"] and same.vetoed == []
    note = next(n for n in same.shortlist[1].notes if "correlated" in n)
    assert "GOLD" in note and "informational only" in note
    # the opposite-exposure (hedge) pairs get no note at all
    hedge = hunt([_f("USD"), _f("EUR", zone_rr={"buy": None})], pairs)
    assert {c.symbol for c in hedge.shortlist} == {"USD", "EUR"} and not any("correlated" in n for c in hedge.shortlist for n in c.notes)


def test_a_counter_trend_side_is_a_candidate_when_its_breakout_is_active_and_only_lightly_outranked():
    from analysis.playbook import PlaybookAdvice

    active = {"sell": PlaybookAdvice(name="TREND-BREAKOUT", side="sell", status="ACTIVE", order="stop", entry=1.0, stop=1.1)}
    against = dict(d1_dir="up", h4_dir="down", m5_dir=None, zone_rr={"buy": None, "sell": None})  # mixed HTF: tier C both ways
    live = hunt([_f("CTR", playbook=active, **against)])
    assert [(c.symbol, c.side, c.tier) for c in live.shortlist] == [("CTR", "sell", "C")] and live.no_read == []
    silent = hunt([_f("CTR", **against)])  # nothing points at it and no breakout is live: still not a candidate
    assert silent.shortlist == [] and silent.no_read == ["CTR"]
    # trend alignment is a nudge: a cheaper counter-trend candidate can outrank a costlier aligned one
    ranked = hunt([_f("WITH", cost_drag_r=0.09), _f("AGAINST", d1_dir="down", h4_dir="down", m5_dir="down", cost_drag_r=0.0, zone_rr={"sell": 2.0})])
    assert ranked.shortlist[0].symbol in ("WITH", "AGAINST") and len(ranked.shortlist) == 2


def test_the_shortlist_is_capped_but_the_cap_is_not_a_veto():
    facts = [_f(f"S{i}", cost_drag_r=0.005 * i) for i in range(12)]
    result = hunt(facts, max_candidates=5)
    assert len(result.shortlist) == 5 and result.vetoed == []
    assert [c.symbol for c in result.shortlist] == ["S0", "S1", "S2", "S3", "S4"]  # cheapest first among equals


def test_cheaper_symbols_rank_first_within_a_tier():
    result = hunt([_f("DEAR", cost_drag_r=0.09), _f("CHEAP", cost_drag_r=0.02)])
    assert [c.symbol for c in result.shortlist] == ["CHEAP", "DEAR"]


def test_format_lists_shortlist_vetoes_and_ids():
    result = hunt([_f("GOOD"), _f("WHEAT", cost_drag_r=1.4), _f("FLAT", d1_dir="flat", h4_dir="flat", m5_dir=None)])
    text = format_position_hunt(result)
    assert "1. GOOD BUY [tier A]" in text and "WITH the D1 and H4 trend" in text
    assert "WHEAT BUY" in text and "V1: round-trip cost is 1.40R" in text
    assert "No directional read" in text and "FLAT" in text
    assert "Veto ids: V1" in text and "V7" in text
    empty = format_position_hunt(hunt([]))
    assert "shortlist: none" in empty


def test_a_symbol_aligned_against_its_own_m5_is_hunted_on_the_aligned_side_with_the_m5_caveat():
    result = hunt([_f("MSFT", d1_dir="down", h4_dir="down", m5_dir="up", zone_rr={"buy": None, "sell": 2.0})])
    cand = result.shortlist[0]
    assert (cand.symbol, cand.side, cand.tier, cand.m5_agrees) == ("MSFT", "sell", "A", False)
    assert any("against this side" in note for note in cand.notes)


def test_eligible_candidates_beyond_the_cap_are_listed_not_silently_dropped():
    facts = [_f(f"S{i}", cost_drag_r=0.01 * i) for i in range(10)]
    result = hunt(facts, max_candidates=3)
    assert [c.symbol for c in result.overflow] == [f"S{i}" for i in range(3, 10)]
    text = format_position_hunt(result)
    assert "Also eligible" in text and "S9 BUY" in text


def test_v1_uses_the_measured_ten_percent_r_limit():
    result = hunt([_f("OK", cost_drag_r=0.10), _f("DEAR", cost_drag_r=0.12)])
    assert [c.symbol for c in result.shortlist] == ["OK"]
    assert result.vetoed[0].symbol == "DEAR" and result.vetoed[0].vetoes[0][0] == "V1" and "playbook would use" in result.vetoed[0].vetoes[0][1]
