#!/usr/bin/env bash
# Build the ATOM Index scorecard from a results CSV (plan §17.5).
#   ./scripts/scorecard.sh results/results.csv
set -euo pipefail

RESULTS="${1:-results/results.csv}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python -m atom_slm.scorecard --results "${RESULTS}" --output results/scorecard.csv
