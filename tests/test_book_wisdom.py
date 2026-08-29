from data.book_wisdom import BOOK_PRINCIPLES, format_book_wisdom, format_trend_wisdom


def test_book_principles_are_all_attributed_and_have_rationale():
    for p in BOOK_PRINCIPLES:
        assert p.principle
        assert p.author
        assert p.rationale


def test_format_book_wisdom_includes_expected_authors():
    text = format_book_wisdom()
    assert "advisory context, not enforced rules" in text
    assert "Bulkowski" in text
    assert "Schwager" in text
    assert "Murphy" in text
    assert "O'Neil" in text


def test_format_book_wisdom_includes_reasoning_not_just_rules():
    text = format_book_wisdom()
    # Each principle should be paired with a "Why:" explanation, not a
    # bare rule with no reasoning.
    assert text.count("Why:") == len(BOOK_PRINCIPLES)


def test_format_book_wisdom_includes_phase3_geopolitical_shock_authors():
    # Phase 3 ("Andrew Lo"/"Graham" previously only appeared in
    # TREND_PRINCIPLES, which format_book_wisdom() does not include) --
    # confirms the new geopolitical-shock entries genuinely made it into
    # BOOK_PRINCIPLES, not just TREND_PRINCIPLES.
    text = format_book_wisdom()
    assert "Andrew Lo" in text
    assert "Graham" in text


def test_sharpened_pring_business_cycle_entry_has_all_six_stages_and_caveat():
    text = format_book_wisdom()
    for stage in ["Stage I", "Stage II", "Stage III", "Stage IV", "Stage V", "Stage VI"]:
        assert stage in text
    assert "little forecasting value" in text


def test_phase3_entries_are_regime_tier():
    phase3_authors = {"Andrew Lo", "Benjamin Graham (with Jason Zweig commentary)"}
    matches = [p for p in BOOK_PRINCIPLES if p.author in phase3_authors]
    assert matches  # sanity: the entries actually exist
    for p in matches:
        assert p.tier == "regime"


def test_format_trend_wisdom_excludes_phase2_and_phase3_regime_content():
    # The Execution Clerk's tactical-defense prompt (ai/copilot_execution.
    # py::_build_tactical_prompt) calls format_trend_wisdom() exclusively —
    # Phase 2's correlation/diversification content and Phase 3's
    # geopolitical-shock/macro-cycle content are portfolio-construction-
    # time, regime-tier wisdom for the Claude mega-session only, and must
    # never leak into the Clerk's narrow, per-position short-term check.
    book_text = format_book_wisdom()
    trend_text = format_trend_wisdom()

    phase2_marker = "October 1997 Asian crisis"  # Murphy's correlation-decoupling entry
    phase3_marker = "reasoning from the headline in isolation"  # Pring's geopolitical-shock entry

    assert phase2_marker in book_text
    assert phase3_marker in book_text
    assert phase2_marker not in trend_text
    assert phase3_marker not in trend_text
