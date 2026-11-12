"""Configuration loading and CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "run_name": "atom-slm-run",
    "seed": 42,
    "base_model": "HuggingFaceTB/SmolLM2-135M",
    "method": "full",  # "full" | "lora"
    "precision": "fp16",  # "fp16" | "bf16" | "no"  (T4/P100 need fp16)
    "output_dir": None,
    "data": {
        "name": "metamath",  # "metamath" | "openmathinstruct"
        "max_samples": 60000,
        "val_samples": 1000,
        "streaming": True,
        "seed": 42,
        "require_numeric_answer": False,
    },
    "training": {
        "max_length": 1024,
        "epochs": 1.0,
        "per_device_batch_size": 8,
        "grad_accum": 8,
        "learning_rate": 5.0e-5,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
        "gradient_checkpointing": True,
        "packing": True,
        "logging_steps": 20,
        "save_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "num_workers": 2,
    },
    "lora": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "hub": {
        "push_to_hub": False,
        "repo_id": "Asaadzx/atom-slm-math-135m",
        "private": False,
        "strategy": "every_save",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and merge it on top of :data:`DEFAULTS`."""
    with open(path, "r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    config = deep_merge(DEFAULTS, user_config)
    if overrides:
        config = deep_merge(config, overrides)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path
