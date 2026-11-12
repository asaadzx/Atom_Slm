"""Math SFT training entry point (Track A).

Examples
--------
Full fine-tune of SmolLM2-135M on a MetaMathQA subset (M1)::

    python -m atom_slm.train --config configs/m1_smolm2_135m_metamath.yaml

LoRA on SmolLM2-360M (M3)::

    python -m atom_slm.train --config configs/m3_smolm2_360m_lora.yaml

Smoke test the whole pipeline in a couple of minutes::

    python -m atom_slm.train --config configs/debug.yaml --smoke-test

The script is written against the *stable* TRL/Transformers surface and
filters unsupported keyword arguments, so a Kaggle image update does not kill a
multi-hour run over a renamed config field.
"""

from __future__ import annotations

import os

# Kaggle "T4 x2" exposes two GPUs to a single process. Hugging Face ``Trainer``
# then wraps the model in ``nn.DataParallel``, which TRL's SFTTrainer refuses.
# A 135M full fine-tune does not need two GPUs, so pin GPU 0 unless the user is
# launching a real distributed run via torchrun/accelerate (those set LOCAL_RANK).
if "LOCAL_RANK" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    _PINNED_SINGLE_GPU = True
else:
    _PINNED_SINGLE_GPU = False

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any

from .config import load_config, save_config
from .data import build_sft_datasets
from .utils import (
    detect_precision,
    filter_kwargs,
    model_dtype_kwarg,
    pick_alias,
    set_seed,
)

LOGGER = logging.getLogger("atom_slm.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ATOM-SLM with SFT.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override the run directory.")
    parser.add_argument("--base-model", default=None, help="Override the base model id.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override the data subset size.")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--method", choices=["full", "lora"], default=None)
    parser.add_argument("--no-push", action="store_true", help="Disable Hub uploads.")
    parser.add_argument(
        "--resume",
        choices=["auto", "none"],
        default="auto",
        help="Resume from the last local checkpoint when present.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Tiny run to validate the pipeline.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.base_model:
        overrides["base_model"] = args.base_model
    if args.method:
        overrides["method"] = args.method
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.max_samples is not None:
        overrides.setdefault("data", {})["max_samples"] = args.max_samples
    if args.learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.no_push:
        overrides.setdefault("hub", {})["push_to_hub"] = False

    if args.smoke_test:
        overrides.setdefault("training", {}).update(
            {
                "max_steps": 5,
                "max_length": 512,
                "logging_steps": 1,
                "save_steps": 5,
                "eval_steps": 5,
            }
        )
        overrides.setdefault("data", {}).update({"max_samples": 64, "val_samples": 16})
        overrides.setdefault("hub", {})["push_to_hub"] = False
        overrides["run_name"] = f"{overrides.get('run_name', 'smoke')}-smoke"
    return overrides


def build_model_and_tokenizer(config: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    precision = detect_precision(config["precision"])
    config["precision"] = precision
    method = config["method"]
    base_model = config["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # fp16 full fine-tuning needs fp32 master weights (GradScaler + autocast).
    # For LoRA fp16 weights are fine and halve the base-model footprint.
    if precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp16" and method == "lora":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    load_kwargs: dict[str, Any] = {
        model_dtype_kwarg(): torch_dtype,
        "attn_implementation": "sdpa",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        LOGGER.info("Loaded %s with SDPA attention (%s).", base_model, torch_dtype)
    except (ValueError, TypeError) as exc:
        LOGGER.warning("SDPA unavailable (%s); falling back to the default attention.", exc)
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    if config["training"].get("gradient_checkpointing", True):
        model.config.use_cache = False

    if method == "lora":
        from peft import LoraConfig, get_peft_model

        lora_cfg = config["lora"]
        peft_config = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora_cfg["target_modules"]),
        )
        try:
            model = get_peft_model(model, peft_config)
        except ValueError as exc:
            LOGGER.warning("LoRA targets not found (%s); using all linear layers.", exc)
            peft_config.target_modules = "all-linear"
            model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    return model, tokenizer


def build_sft_config(config: dict[str, Any], output_dir: Path, push_to_hub: bool):
    from trl import SFTConfig

    train_cfg = config["training"]
    hub_cfg = config["hub"]

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": config["run_name"],
        "num_train_epochs": float(train_cfg["epochs"]),
        "max_steps": int(train_cfg.get("max_steps", -1)),
        "per_device_train_batch_size": int(train_cfg["per_device_batch_size"]),
        "per_device_eval_batch_size": int(train_cfg["per_device_batch_size"]),
        "gradient_accumulation_steps": int(train_cfg["grad_accum"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "lr_scheduler_type": train_cfg["lr_scheduler"],
        "warmup_ratio": float(train_cfg["warmup_ratio"]),
        "weight_decay": float(train_cfg["weight_decay"]),
        "max_grad_norm": float(train_cfg["max_grad_norm"]),
        "optim": train_cfg["optim"],
        "gradient_checkpointing": bool(train_cfg["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "fp16": config["precision"] == "fp16",
        "bf16": config["precision"] == "bf16",
        "logging_steps": int(train_cfg["logging_steps"]),
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": int(train_cfg["save_steps"]),
        "save_total_limit": int(train_cfg["save_total_limit"]),
        "load_best_model_at_end": bool(train_cfg.get("load_best_model_at_end", False)),
        "metric_for_best_model": train_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": bool(train_cfg.get("greater_is_better", False)),
        "eval_strategy": "steps",
        "eval_steps": int(train_cfg["eval_steps"]),
        "dataloader_num_workers": int(train_cfg["num_workers"]),
        "dataloader_pin_memory": True,
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "report_to": "none",
        # TRL / SFT specifics
        "dataset_text_field": "text",
        "packing": bool(train_cfg["packing"]),
        "max_seq_length": int(train_cfg["max_length"]),
    }

    if os.environ.get("WANDB_API_KEY") and train_cfg.get("report_to") == "wandb":
        kwargs["report_to"] = "wandb"

    if push_to_hub:
        kwargs["push_to_hub"] = True
        kwargs["hub_model_id"] = hub_cfg["repo_id"]
        kwargs["hub_private_repo"] = bool(hub_cfg.get("private", False))
        kwargs["hub_strategy"] = hub_cfg.get("strategy", "every_save")

    # Normalize renamed arguments across TRL versions.
    max_len_key = pick_alias(SFTConfig, ("max_length", "max_seq_length"))
    if max_len_key:
        kwargs[max_len_key] = kwargs.pop("max_seq_length")

    # transformers v5 dropped `warmup_ratio`: `warmup_steps` accepts a float
    # below 1 and interprets it as a ratio, so the value carries over.
    warmup_key = pick_alias(SFTConfig, ("warmup_ratio", "warmup_steps"))
    if warmup_key:
        kwargs[warmup_key] = kwargs.pop("warmup_ratio")
    else:
        LOGGER.warning("No warmup argument supported by this TRL version; skipping warmup.")
        kwargs.pop("warmup_ratio", None)

    eval_key = pick_alias(SFTConfig, ("eval_strategy", "evaluation_strategy"))
    if eval_key:
        kwargs[eval_key] = kwargs.pop("eval_strategy")

    return SFTConfig(**filter_kwargs(SFTConfig, kwargs))


def build_trainer(model, tokenizer, train_dataset, eval_dataset, sft_config):
    from trl import SFTTrainer

    kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        kwargs["processing_class"] = tokenizer
    else:  # older TRL
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**filter_kwargs(SFTTrainer, kwargs))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    config = load_config(args.config, build_overrides(args))

    if _PINNED_SINGLE_GPU:
        LOGGER.info(
            "Pinned CUDA_VISIBLE_DEVICES=0 (single-process training; TRL rejects "
            "nn.DataParallel on multi-GPU hosts)."
        )

    output_dir = Path(config.get("output_dir") or f"runs/{config['run_name']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))
    save_config(config, output_dir / "run_config.yaml")
    LOGGER.info("Run directory: %s", output_dir.resolve())

    push_to_hub = bool(config["hub"].get("push_to_hub")) and not args.no_push
    if push_to_hub and not os.environ.get("HF_TOKEN"):
        LOGGER.warning("HF_TOKEN is not set: disabling Hub uploads.")
        push_to_hub = False

    model, tokenizer = build_model_and_tokenizer(config)
    # Re-save after precision detection so the recorded config matches reality.
    save_config(config, output_dir / "run_config.yaml")
    LOGGER.info("Resolved precision: %s", config["precision"])
    train_dataset, eval_dataset = build_sft_datasets(config, tokenizer)

    sft_config = build_sft_config(config, output_dir, push_to_hub=push_to_hub)
    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset, sft_config)

    checkpoint = None
    if args.resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint

        if output_dir.exists():
            checkpoint = get_last_checkpoint(str(output_dir))
        if checkpoint:
            LOGGER.info("Resuming from checkpoint: %s", checkpoint)
        else:
            LOGGER.info("No local checkpoint found; starting from the base model.")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if push_to_hub and trainer.args.push_to_hub:
        trainer.push_to_hub(commit_message="ATOM-SLM math SFT: final model")

    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Done. Metrics written to %s", output_dir / "metrics.json")
    LOGGER.info("Artifacts: %s", ", ".join(sorted(p.name for p in output_dir.glob("*"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
