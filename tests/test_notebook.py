"""Structural tests for the Kaggle notebooks (no Jupyter install needed)."""

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOK = NOTEBOOKS / "atom_slm_kaggle_t4.ipynb"
TEST_NOTEBOOK = NOTEBOOKS / "test_notebook.ipynb"
STAGE2_NOTEBOOK = NOTEBOOKS / "atomslm_dev_stage2.ipynb"


def _load(path: Path = NOTEBOOK) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(path: Path = NOTEBOOK) -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in _load(path)["cells"] if cell["cell_type"] == "code"
    )


def test_notebook_is_valid_json_with_expected_format():
    notebook = _load()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert notebook["cells"], "notebook has no cells"


def test_all_notebooks_are_valid_and_compile():
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    assert len(paths) >= 3
    for path in paths:
        notebook = _load(path)
        assert notebook["nbformat"] == 4
        assert notebook["cells"], f"{path.name} has no cells"
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:{cell.get('id', 'cell')}", "exec")


def test_all_code_cells_have_no_saved_outputs():
    # A clean notebook pushes faster and avoids leaking stale logs.
    for path in NOTEBOOKS.glob("*.ipynb"):
        for cell in _load(path)["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [], f"{path.name}:{cell['id']}"
                assert cell["execution_count"] is None, f"{path.name}:{cell['id']}"


def test_kaggle_gpu_metadata():
    for path in (NOTEBOOK, TEST_NOTEBOOK, STAGE2_NOTEBOOK):
        metadata = _load(path)["metadata"]["kaggle"]
        assert metadata["isGpuEnabled"] is True
        assert metadata["isInternetEnabled"] is True
        assert metadata["accelerator"] == "nvidiaTeslaT4"


def test_t4_safe_defaults_and_smoke_toggle():
    source = _source()
    assert '"precision": "fp16"' in source
    assert "SMOKE_TEST" in source
    assert "gradient_checkpointing" in source


def test_notebooks_have_no_local_absolute_paths():
    for path in NOTEBOOKS.glob("*.ipynb"):
        assert "/home/" not in _source(path), path.name


def test_stage2_uses_the_fixed_recipe():
    source = _source(STAGE2_NOTEBOOK)
    assert "--num-fewshot 0" in source
    assert '"require_numeric_answer": True' in source
    assert '"load_best_model_at_end": True' in source
    assert "git\", \"-C\", str(REPO_DIR), \"pull\", \"--ff-only\"" in source
    assert "atom_slm.diagnose" in source


def test_test_notebook_loads_from_the_hub_and_parses_answers():
    source = _source(TEST_NOTEBOOK)
    assert "from_pretrained(model_id" in source
    assert "Asaadzx/atom-slm-math-135m" in source
    assert "### Solution:" in source
    assert "extract_answer" in source
    assert "openai/gsm8k" in source
