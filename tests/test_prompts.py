"""Tests for the answer-parsing rules (no ML dependencies)."""

from atom_slm.prompts import (
    answers_match,
    ensure_answer_marker,
    extract_answer,
    has_answer_marker,
    has_numeric_answer,
    last_number,
    normalize_answer,
)


def test_extract_gsm8k_marker():
    assert extract_answer("She has 5 apples.\n#### 12") == "12"


def test_extract_last_marker_wins():
    text = "Step one gives #### 3\nBut the real answer is:\n#### 42"
    assert extract_answer(text) == "42"


def test_extract_boxed():
    assert extract_answer(r"So the value is $\boxed{\frac{1}{2}}$") == r"\frac{1}{2}"


def test_extract_answer_is_phrase():
    assert extract_answer("Therefore, the answer is: 1,250.") == "1,250"


def test_extract_none_without_marker():
    # No marker and no "answer is" phrase: do not reward a random last number.
    assert extract_answer("The result is probably 17.") is None


def test_extract_handles_none():
    assert extract_answer(None) is None
    assert extract_answer("") is None


def test_normalize_answer():
    assert normalize_answer(" $1,200. ") == "1200"
    assert normalize_answer(None) == ""


def test_answers_match_numbers():
    assert answers_match("1,200", "1200")
    assert answers_match("3.0", "3")
    assert answers_match(" 42 ", "42.")
    assert not answers_match("42", "43")
    assert not answers_match(None, "42")


def test_answers_match_fractions_are_exact_strings():
    # 3/4 and 0.75 are semantically equal but we keep the comparison honest.
    assert not answers_match("3/4", "0.75")


def test_ensure_answer_marker_appends_expected_answer():
    result = ensure_answer_marker("Add 2 and 2 to get 4.", expected="4")
    assert result.endswith("#### 4")


def test_ensure_answer_marker_rewrites_metamath_style():
    result = ensure_answer_marker("We add them. The answer is: 18.")
    assert "#### 18" in result
    assert "The answer is" not in result


def test_ensure_answer_marker_keeps_existing_marker():
    text = "Work.\n#### 7"
    assert ensure_answer_marker(text, expected="999") == text


def test_ensure_answer_marker_keeps_unparseable_solution():
    text = "A solution with no numeric answer."
    assert ensure_answer_marker(text) == text
    assert not has_answer_marker(text)


def test_last_number_is_diagnostic_only():
    assert last_number("we tried 3 then 9.5 and finally 12") == "12"


def test_has_numeric_answer_accepts_numbers():
    assert has_numeric_answer("We get 12.\n#### 12")
    assert has_numeric_answer("Work.\n#### $1,200")
    assert has_numeric_answer("Work.\n#### 3.5")


def test_has_numeric_answer_rejects_prose_and_missing_markers():
    assert not has_numeric_answer("Work.\n#### twelve dollars")
    assert not has_numeric_answer("No marker here, answer 12")
    assert not has_numeric_answer("")
