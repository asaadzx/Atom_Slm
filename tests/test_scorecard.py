"""Smoke tests for the ATOM Index scorecard."""

import numpy as np
import pandas as pd

from atom_slm.scorecard import build_scorecard, format_markdown


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "tiny-but-weak",
                "q4_mb": 90,
                "gsm8k": 1,
                "math": 0,
                "mmlu": 30,
                "arc_e": 40,
                "hellaswag": 40,
                "piqa": 60,
                "winogrande": 50,
                "tok_s_cpu": 40,
            },
            {
                "model": "mid",
                "q4_mb": 250,
                "gsm8k": 10,
                "math": 2,
                "mmlu": 40,
                "arc_e": 50,
                "hellaswag": 50,
                "piqa": 70,
                "winogrande": 60,
                "tok_s_cpu": 25,
            },
            {
                "model": "big-strong",
                "q4_mb": 800,
                "gsm8k": 35,
                "math": 12,
                "mmlu": 55,
                "arc_e": 65,
                "hellaswag": 65,
                "piqa": 78,
                "winogrande": 70,
                "tok_s_cpu": 8,
            },
        ]
    )


def test_atom_index_is_bounded():
    scored = build_scorecard(_frame())
    assert scored["ATOM"].between(0, 100).all()
    assert np.isfinite(scored["ATOM"]).all()


def test_stronger_model_scores_higher_on_math():
    scored = build_scorecard(_frame()).set_index("model")
    assert scored.loc["big-strong", "MathScore"] > scored.loc["tiny-but-weak", "MathScore"]


def test_math_per_mb_penalizes_size():
    scored = build_scorecard(_frame()).set_index("model")
    assert scored.loc["tiny-but-weak", "Math_per_MB"] > scored.loc["big-strong", "Math_per_MB"]


def test_missing_columns_are_tolerated():
    small = _frame().drop(columns=["piqa", "winogrande", "tok_s_cpu"])
    scored = build_scorecard(small)
    assert "ATOM" in scored.columns
    assert scored["ATOM"].notna().any()


def test_markdown_renders_table():
    scored = build_scorecard(_frame())
    table = format_markdown(scored)
    assert table.startswith("| model |")
    assert "ATOM" in table
    assert len(table.splitlines()) == 5  # header + separator + 3 models
