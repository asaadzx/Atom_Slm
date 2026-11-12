"""Structural tests for the Kaggle notebook (no Jupyter install needed)."""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "atom_slm_kaggle_t4.ipynb"


def _load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_valid_json_with_expected_format():
    notebook = _load()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert notebook["cells"], "notebook has no cells"


def test_all_code_cells_compile():
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell.get("id", "cell"), "exec")


def test_code_cells_have_no_saved_outputs():
    # A clean notebook pushes faster and avoids leaking stale logs.
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_kaggle_gpu_metadata():
    metadata = _load()["metadata"]["kaggle"]
    assert metadata["isGpuEnabled"] is True
    assert metadata["isInternetEnabled"] is True
    assert metadata["accelerator"] == "nvidiaTeslaT4"


def test_t4_safe_defaults_and_smoke_toggle():
    source = "\n".join(
        "".join(cell["source"]) for cell in _load()["cells"] if cell["cell_type"] == "code"
    )
    assert '"precision": "fp16"' in source
    assert "SMOKE_TEST" in source
    assert "gradient_checkpointing" in source


def test_notebook_has_no_local_absolute_paths():
    source = "\n".join(
        "".join(cell["source"]) for cell in _load()["cells"] if cell["cell_type"] == "code"
    )
    assert "/home/" not in source
