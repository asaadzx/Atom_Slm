#!/usr/bin/env bash
# Train with a YAML config. Extra arguments are forwarded to atom_slm.train,
# e.g. ./scripts/train.sh configs/m1_smolm2_135m_metamath.yaml --smoke-test
set -euo pipefail

CONFIG="${1:-configs/m1_smolm2_135m_metamath.yaml}"
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# TRL's SFTTrainer rejects nn.DataParallel; keep single-process runs on one GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m atom_slm.train --config "${CONFIG}" "$@"
