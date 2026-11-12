"""Identity-data tests: valid JSONL, consistent facts, safe mixing."""

import json
from pathlib import Path

from atom_slm.data import load_identity_rows

IDENTITY = Path(__file__).resolve().parents[1] / "data" / "identity" / "atom_identity.jsonl"


def test_identity_file_is_valid_jsonl():
    rows = [json.loads(line) for line in IDENTITY.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) >= 25
    for row in rows:
        assert row["question"].strip()
        assert row["solution"].strip()


def test_identity_questions_are_unique():
    rows = load_identity_rows(IDENTITY)
    questions = [row["question"].lower() for row in rows]
    assert len(questions) == len(set(questions))


def test_every_answer_names_the_model_and_stays_marker_free():
    for row in load_identity_rows(IDENTITY):
        assert "Atom SLM" in row["solution"]
        # `####` is a math-answer convention; identity rows must not dilute it.
        assert "####" not in row["solution"]


def test_identity_mentions_the_developer():
    rows = load_identity_rows(IDENTITY)
    assert sum("Asaadzx" in row["solution"] for row in rows) >= 5


def test_identity_rejects_other_vendors_as_itself():
    rows = load_identity_rows(IDENTITY)
    joined = " ".join(row["solution"] for row in rows)
    assert "not made by OpenAI" in joined
    assert "not affiliated with Anthropic" in joined


def test_load_identity_rows_skips_blanks_and_reports_empty(tmp_path):
    path = tmp_path / "identity.jsonl"
    path.write_text(
        '{"question": "Who are you?", "solution": "I am Atom SLM."}\n\n'
        '{"question": "", "solution": "ignored"}\n',
        encoding="utf-8",
    )
    rows = load_identity_rows(path)
    assert rows == [{"question": "Who are you?", "solution": "I am Atom SLM."}]

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    try:
        load_identity_rows(empty)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("empty identity file should raise")


def test_identity_rows_are_mixed_into_train_only(monkeypatch):
    """Identity rows must not leak into validation, and must stay marker-free."""
    from datasets import Dataset

    import atom_slm.data as data

    math_rows = [
        {
            "query": f"Add {i} and {i}. How many?",
            "response": f"We add {i} + {i} = {2 * i}.\n#### {2 * i}",
        }
        for i in range(1, 51)
    ]
    math_rows.append({"query": "What is one plus one?", "response": "The answer is: two."})
    monkeypatch.setattr(data, "_load_raw", lambda config: Dataset.from_list(math_rows))

    class Tok:
        chat_template = None

        def __call__(self, text):
            return {"input_ids": [0] * len(text.split())}

    config = {
        "seed": 42,
        "data": {
            "name": "metamath",
            "max_samples": 51,
            "val_samples": 5,
            "streaming": False,
            "seed": 42,
            "require_numeric_answer": True,
            "identity_path": str(IDENTITY),
            "identity_repeat": 2,
        },
    }
    train, validation = data.build_sft_datasets(config, Tok())

    assert all("####" in row["text"] for row in validation)
    identity = [row for row in train if "Who are you?" in row["text"]]
    assert len(identity) == 2  # one question, repeated twice
    assert all("####" not in row["text"] for row in identity)
