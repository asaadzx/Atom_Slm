"""GSM8K evaluation for ATOM-SLM checkpoints.

This is the *fast, local* evaluator used inside Kaggle sessions. For the public
scorecard we also run the official ``lm-eval`` harness (see
``scripts/eval_lm_eval.sh``) so numbers are comparable with other models.

Usage
-----
::

    python -m atom_slm.evaluate --model Asaadzx/atom-slm-math-135m --limit 200
    python -m atom_slm.evaluate --model ./runs/m1 --limit 0 --output results/m1.json
    python -m atom_slm.evaluate --compare results/base.json results/m1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from .data import build_prompt_text, load_gsm8k
from .prompts import INSTRUCTION, answers_match, extract_answer
from .utils import set_seed

LOGGER = logging.getLogger("atom_slm.evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K.")
    parser.add_argument("--model", default=None, help="Hub id or local checkpoint path.")
    parser.add_argument("--limit", type=int, default=200, help="0 = full 1319-item test set.")
    parser.add_argument("--num-fewshot", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output", default=None, help="Where to write the JSON report.")
    parser.add_argument("--compare", nargs="+", default=None, help="Compare existing JSON reports.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def build_fewshot_prefix(num_fewshot: int) -> str:
    """Standard GSM8K CoT demonstrations taken from the train split."""
    if num_fewshot <= 0:
        return ""
    train = load_gsm8k("train")
    shots = []
    for row in train.select(range(min(num_fewshot, len(train)))):
        answer = row["answer"].split("####")[-1].strip()
        solution = row["answer"].split("####")[0].strip()
        shots.append(
            f"Question:\n{row['question']}\n\nAnswer:\n{solution}\n\n#### {answer}"
        )
    return "\n\n".join(shots) + "\n\n"


def build_eval_prompt(tokenizer: Any, question: str, fewshot_prefix: str) -> str:
    content = f"{INSTRUCTION}\n\n{fewshot_prefix}Question:\n{question}\n\nAnswer:\n"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"### Question:\n{content}\n\n### Solution:\n"


def stop_token_ids(tokenizer: Any) -> list[int]:
    """All end-of-turn tokens, not just ``eos_token``."""
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    for token in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
            ids.append(token_id)
    return ids


def repetition_ratio(text: str, n: int = 4) -> float:
    """Fraction of repeated word n-grams (0 = no repetition, 1 = degenerate)."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 20:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def clean_generation(text: str) -> str:
    for stop in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype_map[args.dtype]}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_kwargs["device_map"] = {"": 0} if device == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    return model, tokenizer


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not args.model:
        raise SystemExit("--model is required unless you only use --compare")

    set_seed(args.seed)
    LOGGER.info("Loading %s ...", args.model)
    model, tokenizer = load_model_and_tokenizer(args)

    test = load_gsm8k("test")
    if args.limit and args.limit > 0:
        test = test.select(range(min(args.limit, len(test))))

    fewshot_prefix = build_fewshot_prefix(args.num_fewshot)
    prompts = [build_eval_prompt(tokenizer, row["question"], fewshot_prefix) for row in test]
    stops = stop_token_ids(tokenizer)

    predictions: list[dict[str, Any]] = []
    correct = 0
    formatted = 0
    started = time.time()

    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        batch_rows = test.select(range(start, start + len(batch_prompts)))
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=stops or None,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for row, output in zip(batch_rows, outputs):
            text = clean_generation(output)
            gold = row["answer"].split("####")[-1].strip()
            prediction = extract_answer(text)
            is_correct = answers_match(prediction, gold)
            correct += int(is_correct)
            formatted += int(prediction is not None)
            predictions.append(
                {
                    "question": row["question"],
                    "gold": gold,
                    "prediction": prediction,
                    "correct": is_correct,
                    "repetition": round(repetition_ratio(text), 4),
                    "output": text[:2000],
                }
            )

        done = len(predictions)
        elapsed = time.time() - started
        LOGGER.info(
            "GSM8K %d/%d | acc %.2f%% | %.1f s | %.2f s/item",
            done,
            len(prompts),
            100 * correct / done,
            elapsed,
            elapsed / done,
        )

    total = len(predictions)
    result = {
        "model": args.model,
        "benchmark": "gsm8k",
        "split": "test",
        "limit": len(prompts),
        "num_fewshot": args.num_fewshot,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "format_rate": round(formatted / total, 6) if total else 0.0,
        "repetition_rate": round(
            sum(1 for p in predictions if p["repetition"] > 0.5) / total, 6
        )
        if total
        else 0.0,
        "mean_repetition": round(
            sum(p["repetition"] for p in predictions) / total, 6
        )
        if total
        else 0.0,
        "elapsed_seconds": round(time.time() - started, 1),
        "predictions": predictions,
    }

    LOGGER.info(
        "Result: GSM8K accuracy %.2f%% (%d/%d) | format %.1f%% | degenerate %.1f%%",
        100 * result["accuracy"],
        correct,
        total,
        100 * result["format_rate"],
        100 * result["repetition_rate"],
    )
    return result


def compare_reports(paths: list[str]) -> None:
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append(
            {
                "model": data.get("model", Path(path).stem),
                "gsm8k": 100 * data.get("accuracy", 0.0),
                "format": 100 * data.get("format_rate", 0.0),
                "degenerate": 100 * data.get("repetition_rate", 0.0),
                "n": data.get("limit", 0),
            }
        )

    print("\n| Model | GSM8K (%) | Format (%) | Degenerate (%) | N |")
    print("|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['model']} | {row['gsm8k']:.2f} | {row['format']:.1f} "
            f"| {row['degenerate']:.1f} | {row['n']} |"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    if args.compare:
        compare_reports(args.compare)
        return 0

    result = run_evaluation(args)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        LOGGER.info("Report written to %s", output)
    else:
        summary = {k: v for k, v in result.items() if k != "predictions"}
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
