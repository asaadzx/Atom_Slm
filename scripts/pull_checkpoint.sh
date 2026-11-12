#!/usr/bin/env bash
# Pull a training run (including optimizer-state checkpoints) back from the Hub.
# Use this at the start of a new Kaggle session, then train with --resume auto.
set -euo pipefail

REPO="${1:?usage: pull_checkpoint.sh <hub-repo-id> [local-dir]}"
DEST="${2:-runs/resume}"

mkdir -p "${DEST}"
if command -v hf >/dev/null 2>&1; then
  hf download "${REPO}" --repo-type model --include "checkpoint-*/*" --local-dir "${DEST}"
else
  huggingface-cli download "${REPO}" --repo-type model --include "checkpoint-*/*" --local-dir "${DEST}"
fi

echo "Checkpoints downloaded to ${DEST}"
