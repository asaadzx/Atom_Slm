#!/usr/bin/env bash
# Official lm-eval harness run (plan §17.3). Use the SAME command for every
# model in the competitor pool.
#
# Note: plan §17.2 asks for per-benchmark few-shot counts (GSM8K 8-shot, MMLU
# 5-shot, ...). A single `--num_fewshot` applies to all tasks, so for the final
# scorecard run each benchmark separately with its stated shot count.
set -euo pipefail

MODEL="${1:?usage: eval_lm_eval.sh <hf-repo-or-path> [output-dir]}"
OUTPUT_DIR="${2:-results/lm_eval/$(basename "${MODEL}")}"
mkdir -p "${OUTPUT_DIR}"

lm_eval --model hf \
  --model_args "pretrained=${MODEL},dtype=float16" \
  --tasks gsm8k,mmlu,arc_easy,arc_challenge,hellaswag,piqa,winogrande \
  --num_fewshot 5 \
  --batch_size auto \
  --seed 42 \
  --output_path "${OUTPUT_DIR}"

echo "Raw JSON written to ${OUTPUT_DIR}"
