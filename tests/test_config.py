"""Tests for config loading/merging."""

import yaml

from atom_slm.config import DEFAULTS, deep_merge, load_config, save_config


def test_deep_merge_is_recursive_and_non_mutating():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"c": 20}, "e": 5}
    merged = deep_merge(base, override)

    assert merged == {"a": {"b": 1, "c": 20}, "d": 3, "e": 5}
    assert base["a"]["c"] == 2  # original untouched


def test_load_config_merges_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"run_name": "test-run", "training": {"epochs": 2}}),
        encoding="utf-8",
    )
    config = load_config(path)

    assert config["run_name"] == "test-run"
    assert config["training"]["epochs"] == 2
    # untouched defaults survive
    assert config["training"]["optim"] == DEFAULTS["training"]["optim"]
    assert config["base_model"] == DEFAULTS["base_model"]


def test_load_config_applies_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("run_name: base\n", encoding="utf-8")
    config = load_config(path, {"data": {"max_samples": 10}})
    assert config["data"]["max_samples"] == 10


def test_save_config_roundtrip(tmp_path):
    target = tmp_path / "nested" / "run_config.yaml"
    save_config({"run_name": "x", "training": {"epochs": 1}}, target)
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["run_name"] == "x"
