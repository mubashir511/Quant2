from data.chart_wisdom import CHART_PRINCIPLES, format_chart_wisdom


def test_chart_principles_are_all_attributed_and_have_rationale():
    for p in CHART_PRINCIPLES:
        assert p.principle
        assert p.author
        assert p.rationale


def test_format_chart_wisdom_includes_expected_authors():
    text = format_chart_wisdom()
    assert "advisory context" in text
    assert "Murphy" in text
    assert "Bulkowski" in text
    assert "Pring" in text
    assert "Nison" in text


def test_format_chart_wisdom_includes_reasoning_not_just_rules():
    text = format_chart_wisdom()
    assert text.count("Why:") == len(CHART_PRINCIPLES)


def test_format_chart_wisdom_cites_the_real_deterministic_rules_it_grounds():
    # Every entry should trace back to a REAL, already-shipped/upgraded
    # analysis/*.py rule name, not float free of the code it's meant to
    # explain — a spot-check across all 5 phases of this upgrade.
    text = format_chart_wisdom()
    for real_rule_name in (
        "detect_structure_breaks",
        "detect_breakouts",
        "detect_liquidity_sweeps",
        "detect_rsi_divergence",
        "busted_pattern_reversal",
        "candlestick_reversal_confirmed",
        "SRLevel.low/.high",
    ):
        assert real_rule_name in text
