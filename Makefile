PY ?= python
CONFIG ?= configs/m1_smolm2_135m_metamath.yaml
MODEL ?= runs/m1-smollm2-135m-metamath
LIMIT ?= 200
OUTPUT ?= results/eval.json
RESULTS ?= results/results.csv

.PHONY: help install test lint smoke train eval scorecard clean

help:
	@echo "install    Install the package in editable mode with dev tools"
	@echo "test       Run the unit tests"
	@echo "lint       Run ruff"
	@echo "smoke      5-step pipeline check on 64 examples"
	@echo "train      Train with CONFIG=$(CONFIG)"
	@echo "eval       Evaluate MODEL=$(MODEL) on GSM8K (LIMIT=$(LIMIT))"
	@echo "scorecard  Build the ATOM Index from RESULTS=$(RESULTS)"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

smoke:
	$(PY) -m atom_slm.train --config configs/debug.yaml --smoke-test

train:
	$(PY) -m atom_slm.train --config $(CONFIG)

eval:
	$(PY) -m atom_slm.evaluate --model $(MODEL) --limit $(LIMIT) --output $(OUTPUT)

scorecard:
	$(PY) -m atom_slm.scorecard --results $(RESULTS) --output results/scorecard.csv

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
