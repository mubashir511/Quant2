from data.book_wisdom import BOOK_PRINCIPLES, format_book_wisdom


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
