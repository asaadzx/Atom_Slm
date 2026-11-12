"""Source-level guards for the training entry point.

These tests never import torch/trl, so they run anywhere. They protect the two
Kaggle-specific invariants that cost real debugging time:

* ``CUDA_VISIBLE_DEVICES`` must be pinned before torch is imported.
* ``warmup_ratio`` must be translated for transformers v5, which dropped it.
"""

import ast
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1] / "src" / "atom_slm" / "train.py"


def _tree() -> ast.Module:
    return ast.parse(TRAIN.read_text(encoding="utf-8"))


def _line_of_first(pattern: str) -> int | None:
    for number, line in enumerate(TRAIN.read_text(encoding="utf-8").splitlines(), 1):
        if pattern in line:
            return number
    return None


def test_cuda_visible_devices_pinned_before_torch_import():
    tree = _tree()
    pin_line = _line_of_first('os.environ["CUDA_VISIBLE_DEVICES"] = "0"')
    assert pin_line is not None, "missing single-GPU pin"

    torch_imports = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        and any(alias.name.split(".")[0] == "torch" for alias in node.names)
    ]
    torch_imports += [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch"
    ]
    if torch_imports:
        assert pin_line < min(torch_imports), "CUDA_VISIBLE_DEVICES must be set before importing torch"


def test_pin_is_skipped_for_distributed_launches():
    source = TRAIN.read_text(encoding="utf-8")
    assert "LOCAL_RANK" in source, "torchrun/accelerate must be able to keep multi-GPU"
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in source


def test_warmup_ratio_is_mapped_to_a_supported_argument():
    source = TRAIN.read_text(encoding="utf-8")
    assert 'pick_alias(SFTConfig, ("warmup_ratio", "warmup_steps"))' in source
    assert 'kwargs[warmup_key] = kwargs.pop("warmup_ratio")' in source


def test_dtype_keyword_is_version_aware():
    source = TRAIN.read_text(encoding="utf-8")
    assert "model_dtype_kwarg()" in source
    assert '"torch_dtype": torch_dtype' not in source, "deprecated torch_dtype= on transformers v5"
