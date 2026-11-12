#!/usr/bin/env bash
# Fast local GSM8K check.
#   ./scripts/eval_gsm8k.sh <model> [limit] [output.json]
set -euo pipefail

MODEL="${1:?usage: eval_gsm8k.sh <model-or-path> [limit] [output.json]}"
LIMIT="${2:-200}"
OUTPUT="${3:-results/$(basename "${MODEL}")-gsm8k.json}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python -m atom_slm.evaluate --model "${MODEL}" --limit "${LIMIT}" --output "${OUTPUT}"
