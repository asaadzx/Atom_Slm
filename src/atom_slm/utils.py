"""Kernel-agnostic helpers shared by the training/eval entry points."""

from __future__ import annotations

import inspect
import logging
import os
import random
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (torch is optional at import time)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        pass


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keyword arguments a callable does not accept.

    Used to stay compatible with the fast-moving TRL/Transformers APIs: a
    renamed argument is logged instead of crashing a multi-hour Kaggle run.
    """
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs

    accepted: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in kwargs.items():
        if key in parameters:
            accepted[key] = value
        else:
            dropped.append(key)
    if dropped:
        LOGGER.warning("Ignoring unsupported arguments for %s: %s", callable_obj, dropped)
    return accepted


def pick_alias(callable_obj: Any, candidates: tuple[str, ...]) -> str | None:
    """Return the first parameter name that ``callable_obj`` accepts."""
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return candidates[0]
    for name in candidates:
        if name in parameters:
            return name
    return None


def detect_precision(requested: str) -> str:
    """Validate the requested precision against the available GPU.

    T4 / P100 (compute capability < 8.0) do not support bf16, so we fall back to
    fp16 with a warning instead of silently producing garbage.
    """
    import torch

    if requested in {"no", "none", "fp32"}:
        return "no"
    if requested == "fp16":
        return "fp16"

    if requested == "bf16":
        if not torch.cuda.is_available():
            LOGGER.warning("No CUDA device: bf16 requested, using fp32.")
            return "no"
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            LOGGER.warning(
                "GPU does not support bf16 (compute capability %s.x): falling back to fp16.",
                major,
            )
            return "fp16"
        return "bf16"

    raise ValueError(f"Unknown precision {requested!r}; use 'fp16', 'bf16' or 'no'.")


def human_bytes(num_bytes: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def model_dtype_kwarg() -> str:
    """Return the dtype keyword accepted by ``from_pretrained``.

    Transformers v5 renamed ``torch_dtype`` to ``dtype`` (the old name still
    works but warns). Keeping this in one place avoids deprecation spam in
    every entry point.
    """
    try:
        from transformers import __version__ as version
    except ImportError:  # pragma: no cover - transformers is a runtime dep
        return "torch_dtype"
    try:
        major = int(str(version).split(".", 1)[0])
    except ValueError:
        return "torch_dtype"
    return "dtype" if major >= 5 else "torch_dtype"
