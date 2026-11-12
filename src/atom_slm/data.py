"""Dataset loading, chat formatting and SFT dataset construction.

Datasets used
-------------
* ``meta-math/MetaMathQA``      — small, fast, great for M1.
* ``nvidia/OpenMathInstruct-2`` — ~14M rows, streamed and subsampled for M2.
* ``openai/gsm8k``              — evaluation only (never trained on).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from .prompts import INSTRUCTION, ensure_answer_marker, has_answer_marker

LOGGER = logging.getLogger(__name__)

# Dataset-specific column names and the id used on the Hub.
DATASET_SPECS: dict[str, dict[str, str]] = {
    "metamath": {
        "path": "meta-math/MetaMathQA",
        "question": "query",
        "solution": "response",
        "expected": "",
    },
    "openmathinstruct": {
        "path": "nvidia/OpenMathInstruct-2",
        "question": "problem",
        "solution": "generated_solution",
        "expected": "expected_answer",
    },
}


def load_gsm8k(split: str = "test") -> Dataset:
    """Load GSM8K, tolerating the old and new Hub dataset ids."""
    errors = []
    for path in ("openai/gsm8k", "gsm8k"):
        try:
            return load_dataset(path, "main", split=split)
        except Exception as exc:  # noqa: BLE001 - try the next id
            errors.append(f"{path}: {exc}")
    raise RuntimeError("Could not load GSM8K:\n" + "\n".join(errors))


def user_content(question: str, instruction: str = INSTRUCTION) -> str:
    """Instruction-then-question layout shared by training and evaluation."""
    return f"{instruction}\n\nQuestion:\n{question}"


def build_prompt_text(tokenizer: Any, question: str, instruction: str = INSTRUCTION) -> str:
    """Render a question as a chat prompt (assistant turn left open)."""
    content = user_content(question, instruction)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"### Question:\n{content}\n\n### Solution:\n"


def format_example(
    tokenizer: Any,
    question: str,
    solution: str,
    instruction: str = INSTRUCTION,
) -> str:
    """Render one ``(question, solution)`` pair as a full chat transcript."""
    messages = [
        {"role": "user", "content": user_content(question, instruction)},
        {"role": "assistant", "content": solution},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"### Question:\n{question}\n\n### Solution:\n{solution or ''}<|endoftext|>"


def _load_raw(config: dict[str, Any]) -> Dataset:
    data_cfg = config["data"]
    name = data_cfg["name"]
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(DATASET_SPECS)}")

    spec = DATASET_SPECS[name]
    max_samples = data_cfg.get("max_samples")
    streaming = bool(data_cfg.get("streaming", True)) and name == "openmathinstruct"

    if streaming:
        LOGGER.info("Streaming %s (taking %s rows)...", spec["path"], max_samples)
        stream = load_dataset(spec["path"], split="train", streaming=True)
        rows = list(itertools.islice(stream, int(max_samples or 100_000)))
        return Dataset.from_list(rows)

    dataset = load_dataset(spec["path"], split="train")
    if name == "openmathinstruct" and "problem" not in dataset.column_names:
        # OpenMathInstruct-2 ships 32 parquet shards; keep the code path explicit.
        LOGGER.warning("Unexpected OpenMathInstruct-2 columns: %s", dataset.column_names)
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(int(max_samples)))
    return dataset


def _standardize(dataset: Dataset, config: dict[str, Any]) -> Dataset:
    """Map dataset-specific columns onto ``question`` / ``solution`` / ``answer``."""
    spec = DATASET_SPECS[config["data"]["name"]]
    question_col = spec["question"]
    solution_col = spec["solution"]
    expected_col = spec["expected"] or None

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        question = (row.get(question_col) or "").strip()
        raw_solution = row.get(solution_col) or ""
        expected = row.get(expected_col) if expected_col else None
        solution = ensure_answer_marker(raw_solution, expected)
        return {"question": question, "solution": solution}

    columns = [c for c in (question_col, solution_col, expected_col) if c]
    dataset = dataset.map(
        convert,
        remove_columns=[c for c in dataset.column_names if c in columns],
        desc="Standardizing columns",
    )
    return dataset


def build_sft_datasets(
    config: dict[str, Any],
    tokenizer: Any,
) -> tuple[Dataset, Dataset]:
    """Return ``(train, validation)`` datasets with a ``text`` field.

    The ``text`` field contains the full chat transcript produced by the base
    model's own chat template, so the SFT model learns exactly the format it
    will be served with.
    """
    data_cfg = config["data"]
    instruction = data_cfg.get("instruction", INSTRUCTION)
    seed = int(data_cfg.get("seed", config.get("seed", 42)))

    dataset = _load_raw(config)
    dataset = _standardize(dataset, config)

    # Basic hygiene: drop empty / degenerate rows before spending tokens on them.
    dataset = dataset.filter(
        lambda row: len(row["question"]) >= 8 and len(row["solution"]) >= 16,
        desc="Filtering empty rows",
    )
    dataset = dataset.shuffle(seed=seed)

    missing = sum(0 if has_answer_marker(text) else 1 for text in dataset["solution"])
    if missing:
        LOGGER.warning(
            "%s of %s solutions have no '####' marker; the model may not learn the format.",
            missing,
            len(dataset),
        )

    def render(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": format_example(
                tokenizer, row["question"], row["solution"], instruction=instruction
            )
        }

    dataset = dataset.map(render, desc="Applying chat template")

    val_size = int(data_cfg.get("val_samples", 1000))
    val_size = max(0, min(val_size, len(dataset) // 10))
    if val_size == 0:
        raise ValueError("Dataset too small to split off a validation set.")

    validation = dataset.select(range(val_size))
    train = dataset.select(range(val_size, len(dataset)))
    LOGGER.info("SFT datasets ready: %s train / %s validation rows", len(train), len(validation))
    return train, validation


def build_dataset_dict(config: dict[str, Any], tokenizer: Any) -> DatasetDict:
    train, validation = build_sft_datasets(config, tokenizer)
    return DatasetDict({"train": train, "validation": validation})
