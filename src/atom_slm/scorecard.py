"""ATOM Index scorecard (plan §17.5).

Reads a CSV with one row per model and one column per benchmark, then computes
robust-normalized category scores and the composite ATOM Index::

    N(m,b)     = 100 * clip((S - p10) / (p90 - p10), 0, 1)
    MathScore  = mean(N(gsm8k), N(math), N(in_domain))
    TextScore  = mean(N(mmlu), N(arc_e), N(hellaswag), N(piqa), N(winogrande))
    EffScore   = 0.5 * N(-q4_mb) + 0.5 * N(tok_s_cpu)
    ATOM       = 0.45 * MathScore + 0.35 * TextScore + 0.20 * EffScore

Missing columns are skipped and weights are renormalized, so partial
scorecards can be produced while a model is still being evaluated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MATH_COLUMNS = ["gsm8k", "math", "in_domain"]
TEXT_COLUMNS = ["mmlu", "arc_e", "hellaswag", "piqa", "winogrande"]
EFFICIENCY_COLUMNS = ["q4_mb", "tok_s_cpu"]

DEFAULT_WEIGHTS = {"math": 0.45, "text": 0.35, "efficiency": 0.20}


def robust_normalize(series: pd.Series) -> pd.Series:
    """Scale a benchmark to 0–100 using its 10th/90th percentiles."""
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if len(valid) == 0:
        return pd.Series(np.nan, index=series.index)
    if len(valid) < 2:
        return values.notna().astype(float) * 50.0
    low, high = np.percentile(valid, 10), np.percentile(valid, 90)
    if high - low < 1e-9:
        return values.notna().astype(float) * 50.0
    scaled = 100.0 * np.clip((values - low) / (high - low), 0.0, 1.0)
    return scaled


def _category_score(normalized: dict[str, pd.Series], columns: list[str]) -> pd.Series:
    available = [normalized[c] for c in columns if c in normalized]
    if not available:
        return pd.Series(np.nan, index=next(iter(normalized.values())).index)
    return pd.concat(available, axis=1).mean(axis=1, skipna=True)


CATEGORY_WEIGHT_KEYS = {"MathScore": "math", "TextScore": "text", "EffScore": "efficiency"}


def _weighted_index(scores: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Weighted mean that renormalizes over the categories that are present."""
    frame = pd.DataFrame(scores)
    weight_series = pd.Series(weights)
    available = [c for c in frame.columns if frame[c].notna().any()]
    if not available:
        return pd.Series(np.nan, index=frame.index)

    total_weight = float(sum(weight_series[CATEGORY_WEIGHT_KEYS[c]] for c in available))
    if total_weight <= 0:
        return pd.Series(np.nan, index=frame.index)

    result = pd.Series(0.0, index=frame.index)
    used = pd.Series(0.0, index=frame.index)
    for column in available:
        weight = weight_series[CATEGORY_WEIGHT_KEYS[column]] / total_weight
        mask = frame[column].notna()
        result[mask] += weight * frame.loc[mask, column]
        used[mask] += weight
    return (result / used.replace(0, np.nan)).clip(0, 100)


def build_scorecard(
    frame: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    frame = frame.copy()

    normalized: dict[str, pd.Series] = {}
    for column in MATH_COLUMNS + TEXT_COLUMNS + EFFICIENCY_COLUMNS:
        if column in frame.columns:
            normalized[column] = robust_normalize(frame[column])

    if "q4_mb" in normalized:
        normalized["q4_mb"] = robust_normalize(-pd.to_numeric(frame["q4_mb"], errors="coerce"))

    scores = {
        "MathScore": _category_score(normalized, MATH_COLUMNS),
        "TextScore": _category_score(normalized, TEXT_COLUMNS),
        "EffScore": _category_score(normalized, EFFICIENCY_COLUMNS),
    }
    for name, series in scores.items():
        frame[name] = series.round(2)

    frame["ATOM"] = _weighted_index(scores, weights).round(2)
    if "q4_mb" in frame.columns:
        frame["Math_per_MB"] = (
            pd.to_numeric(frame["MathScore"], errors="coerce")
            / pd.to_numeric(frame["q4_mb"], errors="coerce")
        ).round(3)

    return frame


def format_markdown(frame: pd.DataFrame) -> str:
    display = frame.sort_values("ATOM", ascending=False, na_position="last")
    preferred = [
        "model",
        "params",
        "q4_mb",
        "gsm8k",
        "math",
        "mmlu",
        "arc_e",
        "hellaswag",
        "ppl",
        "tok_s_cpu",
        "MathScore",
        "TextScore",
        "EffScore",
        "ATOM",
        "Math_per_MB",
    ]
    columns = [c for c in preferred if c in display.columns]
    display = display[columns + [c for c in display.columns if c not in columns]]

    def render(value: object) -> str:
        if isinstance(value, float) and not np.isnan(value):
            return f"{value:.2f}"
        return "" if pd.isna(value) else str(value)

    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in display[columns].itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the ATOM Index scorecard.")
    parser.add_argument("--results", required=True, help="Input CSV, one row per model.")
    parser.add_argument("--output", default=None, help="Where to write the scored CSV.")
    parser.add_argument("--math-weight", type=float, default=DEFAULT_WEIGHTS["math"])
    parser.add_argument("--text-weight", type=float, default=DEFAULT_WEIGHTS["text"])
    parser.add_argument("--efficiency-weight", type=float, default=DEFAULT_WEIGHTS["efficiency"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.results)
    weights = {
        "math": args.math_weight,
        "text": args.text_weight,
        "efficiency": args.efficiency_weight,
    }
    scored = build_scorecard(frame, weights)
    scored = scored.sort_values("ATOM", ascending=False, na_position="last")

    print(f"\nWeights: math={weights['math']} text={weights['text']} "
          f"efficiency={weights['efficiency']}\n")
    print(format_markdown(scored))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output, index=False)
        print(f"\nSaved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
