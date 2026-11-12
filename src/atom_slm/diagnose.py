"""Diagnose a finished run: training curve, checkpoint delta, sample outputs.

Run this before trusting a GSM8K number. It answers the three questions that
matter after a fine-tune:

1. Did training actually move? (loss curve + eval losses from ``trainer_state.json``)
2. Is the checkpoint different from the base model? (weight delta)
3. Does the model answer *this* prompt? (raw generations, training-style prompt)

Usage
-----
::

    python -m atom_slm.diagnose --run runs/m1-smollm2-135m-metamath
    python -m atom_slm.diagnose --run runs/m1 --base HuggingFaceTB/SmolLM2-135M --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .prompts import answers_match, extract_answer
from .utils import model_dtype_kwarg, set_seed

LOGGER = logging.getLogger("atom_slm.diagnose")

# Small, cheap probes: embeddings and one MLP block per model.
PROBE_KEYS = (
    "model.embed_tokens.weight",
    "model.layers.0.mlp.down_proj.weight",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a training run.")
    parser.add_argument("--run", required=True, help="Run directory produced by atom_slm.train.")
    parser.add_argument("--base", default=None, help="Base model id (read from run_config.yaml if omitted).")
    parser.add_argument("--limit", type=int, default=3, help="GSM8K questions to sample.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _run_config(run_dir: Path) -> dict[str, Any]:
    for name in ("run_config.yaml", "config.yaml"):
        path = run_dir / name
        if path.exists():
            import yaml

            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def summarize_training(run_dir: Path) -> None:
    path = run_dir / "trainer_state.json"
    if not path.exists():
        print(f"[1/3] No trainer_state.json in {run_dir} — cannot verify training.")
        return

    state = json.loads(path.read_text(encoding="utf-8"))
    history = state.get("log_history", [])
    losses = [h["loss"] for h in history if isinstance(h.get("loss"), (int, float))]
    evals = [(h.get("step"), h["eval_loss"]) for h in history if "eval_loss" in h]

    print("[1/3] Training")
    print(f"  global_step : {state.get('global_step')}")
    print(f"  epoch       : {state.get('epoch')}")
    if losses:
        print(f"  loss        : {losses[0]:.4f} (first) -> {losses[-1]:.4f} (last)")
        print(f"  last 5      : {[round(x, 4) for x in losses[-5:]]}")
    if evals:
        print(f"  eval_loss   : {[(step, round(v, 4)) for step, v in evals]}")
    if state.get("best_metric") is not None:
        print(f"  best_metric : {state['best_metric']} ({state.get('best_model_checkpoint')})")

    if losses and losses[-1] > losses[0] - 0.05:
        print("  WARNING: loss barely moved — training may have been too short or frozen.")


def checkpoint_delta(run_dir: Path, base: str | None) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    if not base:
        print("[2/3] Skipped: no --base given and no run_config.yaml found.")
        return

    dtype_kwarg = model_dtype_kwarg()
    print(f"[2/3] Weight delta vs {base}")
    try:
        tuned = AutoModelForCausalLM.from_pretrained(
            str(run_dir), low_cpu_mem_usage=True, **{dtype_kwarg: torch.float16}
        )
        reference = AutoModelForCausalLM.from_pretrained(
            base, low_cpu_mem_usage=True, **{dtype_kwarg: torch.float16}
        )
    except Exception as exc:  # noqa: BLE001 - diagnosis, not control flow
        print(f"  could not compare weights: {exc}")
        return

    tuned_sd = tuned.state_dict()
    base_sd = reference.state_dict()
    for key in PROBE_KEYS:
        if key not in tuned_sd or key not in base_sd:
            continue
        delta = (tuned_sd[key].float() - base_sd[key].float()).abs().mean().item()
        scale = base_sd[key].float().abs().mean().item()
        print(f"  {key}: mean|delta| = {delta:.6f} (base scale {scale:.6f})")

    del tuned, reference


def sample_generations(run_dir: Path, limit: int, max_new_tokens: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .data import build_prompt_text, load_gsm8k

    print(f"[3/3] {limit} greedy generations (training-style prompt)")
    tokenizer = AutoTokenizer.from_pretrained(str(run_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs: dict[str, Any] = {model_dtype_kwarg(): torch.float16, "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(str(run_dir), **load_kwargs)
    model.eval()

    dataset = load_gsm8k("test").select(range(min(limit, 1319)))
    for row in dataset:
        gold = row["answer"].split("####")[-1].strip()
        prompt = build_prompt_text(tokenizer, row["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        prediction = extract_answer(text)
        print("\n" + "-" * 78)
        print(f"question : {row['question'][:160]}")
        print(f"gold     : {gold}")
        print(f"pred     : {prediction!r}  |  correct={answers_match(prediction, gold)}")
        print(f"output   : {text[:600].replace(chr(10), ' ')}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    set_seed(args.seed)

    run_dir = Path(args.run).expanduser()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    config = _run_config(run_dir)
    base = args.base or config.get("base_model")
    print(f"Run     : {run_dir}")
    print(f"Base    : {base or 'unknown'}")
    print(f"Files   : {sorted(p.name for p in run_dir.iterdir())[:12]}\n")

    summarize_training(run_dir)
    checkpoint_delta(run_dir, base)
    sample_generations(run_dir, args.limit, args.max_new_tokens)

    print("\nIf the samples above stop mid-sentence or loop, the problem is the")
    print("generation prompt or undertraining — not the answer parser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
