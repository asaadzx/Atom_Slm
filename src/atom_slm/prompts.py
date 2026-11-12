"""Prompt formatting and answer parsing helpers.

This module is deliberately free of torch / transformers / datasets imports so
the parsing rules (the part that decides whether an answer is "correct") can be
unit-tested quickly and reused by the evaluator and the data pipeline.

Answer conventions supported
----------------------------
* GSM8K style:      ``#### 42``
* LaTeX style:      ``\\boxed{42}``
* MetaMathQA style: ``The answer is: 42``
"""

from __future__ import annotations

import math
import re

ANSWER_MARKER = "####"

# Instruction prepended to every training/eval question. Keeping it identical
# between training and evaluation is what makes the output format reliable.
INSTRUCTION = (
    "Solve step by step. End with the final answer on its own line as: "
    f"{ANSWER_MARKER} <answer>"
)

_MARKER_RE = re.compile(r"####\s*(.+)")
_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*(.+)", re.IGNORECASE
)
_TRAILING_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\-]?\s*([^\n]+?)\s*\.?\s*$",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def extract_answer(text: str | None) -> str | None:
    """Return the final answer contained in ``text``, or ``None``.

    Precedence: explicit ``####`` marker, then ``\\boxed{}``, then the last
    "answer is ..." phrase. We intentionally do *not* fall back to "the last
    number in the text": that inflates accuracy by rewarding rambling.
    """
    if not text:
        return None

    matches = _MARKER_RE.findall(text)
    if matches:
        # Everything after the *last* marker wins (models sometimes restate).
        return _clean_answer(text.split(ANSWER_MARKER)[-1])

    boxes = _BOXED_RE.findall(text)
    if boxes:
        return _clean_answer(boxes[-1])

    matches = _ANSWER_IS_RE.findall(text)
    if matches:
        return _clean_answer(matches[-1])

    return None


def normalize_answer(text: str | None) -> str:
    """Normalize an answer for comparison (case, spaces, ``$``, commas)."""
    if text is None:
        return ""
    value = text.strip().lower()
    value = value.replace(",", "").replace("$", "").replace("%", "")
    value = re.sub(r"\s+", "", value)
    return value.strip(".")


def answers_match(prediction: str | None, gold: str | None, rel_tol: float = 1e-4) -> bool:
    """Compare a predicted answer with the gold answer.

    Numbers are compared with a small relative tolerance (so ``1,200`` matches
    ``1200`` and ``3.0`` matches ``3``). Anything else is compared as a
    normalized string.
    """
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    if not pred or not target:
        return False
    if pred == target:
        return True
    try:
        return math.isclose(float(pred), float(target), rel_tol=rel_tol, abs_tol=1e-8)
    except ValueError:
        return False


def ensure_answer_marker(
    solution: str | None,
    expected: str | None = None,
    marker: str = ANSWER_MARKER,
) -> str:
    """Rewrite a reference solution so it ends with ``#### <answer>``.

    Handles the three common dataset shapes:

    * already contains ``####``            -> returned unchanged
    * has a separate ``expected`` field    -> marker appended
    * ends with "The answer is: 18"        -> phrase replaced by the marker

    If none of these apply the solution is returned unchanged (callers can use
    :func:`has_answer_marker` to count how often that happens).
    """
    text = (solution or "").strip()
    if not text:
        return text
    if marker in text:
        return text

    if expected is not None and str(expected).strip():
        return f"{text}\n\n{marker} {str(expected).strip()}"

    match = _TRAILING_ANSWER_IS_RE.search(text)
    if match:
        answer = _clean_answer(match.group(1))
        if answer:
            body = text[: match.start()].rstrip()
            return f"{body}\n\n{marker} {answer}"

    return text


def has_answer_marker(text: str | None, marker: str = ANSWER_MARKER) -> bool:
    return bool(text) and marker in text


def last_number(text: str | None) -> str | None:
    """Return the last number in ``text`` (used only for diagnostics)."""
    if not text:
        return None
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def _clean_answer(raw: str) -> str | None:
    """Strip decoration around an extracted answer."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return None
    value = lines[0]
    value = value.strip("*`_ ")
    value = value.removeprefix("$").removesuffix("$").strip()
    value = value.rstrip(".,;:").strip()
    value = value.removeprefix("\\(").removesuffix("\\)").strip()
    return value or None
