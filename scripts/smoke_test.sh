#!/usr/bin/env bash
# Tier-1 pipeline smoke test — run this on Kaggle (or any machine with the
# training stack installed) before spending GPU hours on a real run.
#
#   ./scripts/smoke_test.sh
#   ATOM_SMOKE_DIR=/kaggle/working/atom-smoke ./scripts/smoke_test.sh
#
# It checks, in order:
#   1. unit tests (parsing, config, scorecard, notebook integrity)
#   2. 5-step training on 64 rows (data -> chat template -> packing -> checkpoint)
#   3. GSM8K eval on the freshly saved checkpoint (local model loading)
#   4. GSM8K sanity eval on the base model (Hub loading, generation, parsing)
#
# Expected runtime: ~5–10 min on a T4. Nothing here needs a full epoch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# Same single-GPU rule as scripts/train.sh (Kaggle T4 x2).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT_ROOT="${ATOM_SMOKE_DIR:-/kaggle/working/atom-smoke}"
mkdir -p "${OUT_ROOT}"

echo "== [1/5] Unit tests =="
if python -c "import pytest" >/dev/null 2>&1; then
  python -m pytest -q tests
else
  echo "pytest not installed — skipping (pip install pytest)"
fi

echo "== [2/5] 5-step training run =="
python -m atom_slm.train \
  --config configs/debug.yaml \
  --smoke-test \
  --output-dir "${OUT_ROOT}/model"

echo "== [3/5] Eval the smoke checkpoint (pipeline only) =="
python -m atom_slm.evaluate \
  --model "${OUT_ROOT}/model" \
  --limit 5 \
  --max-new-tokens 128 \
  --output "${OUT_ROOT}/smoke-model-gsm8k.json"

echo "== [4/5] Base-model GSM8K sanity eval =="
python -m atom_slm.evaluate \
  --model HuggingFaceTB/SmolLM2-135M \
  --limit 20 \
  --output "${OUT_ROOT}/base-gsm8k.json"

echo "== [5/5] Verify artifacts =="
for artifact in \
  "${OUT_ROOT}/model/config.json" \
  "${OUT_ROOT}/model/metrics.json" \
  "${OUT_ROOT}/smoke-model-gsm8k.json" \
  "${OUT_ROOT}/base-gsm8k.json"; do
  if [ ! -f "${artifact}" ]; then
    echo "MISSING: ${artifact}" >&2
    exit 1
  fi
  echo "ok: ${artifact}"
done

python - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("\nArtifacts written to:", root)
print("model files:", sorted(p.name for p in (root / "model").glob("*")))

base = json.loads((root / "base-gsm8k.json").read_text())
print(f"base GSM8K ({base['limit']} items): {100 * base['accuracy']:.2f}% "
      f"| format {100 * base['format_rate']:.1f}%")

smoke = json.loads((root / "smoke-model-gsm8k.json").read_text())
print(f"smoke checkpoint eval ran on {smoke['limit']} items (score is meaningless)")
print("\nSMOKE TEST PASSED — the pipeline is wired correctly.")
print("Next: open the Kaggle notebook, keep SMOKE_TEST = False, run the real M1.")
PY
