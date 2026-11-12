#!/usr/bin/env bash
# Convert a merged HF checkpoint to GGUF and quantize it (plan M4).
#   ./scripts/convert_gguf.sh runs/m1-smollm2-135m-metamath [output-basename]
#
# Requires a llama.cpp checkout with built binaries (see README / M4).
set -euo pipefail

MODEL_DIR="${1:?usage: convert_gguf.sh <merged-model-dir> [output-basename]}"
BASENAME="${2:-atom-slm-math}"
LLAMA_DIR="${LLAMA_CPP_DIR:-third_party/llama.cpp}"

if [ ! -d "${LLAMA_DIR}" ]; then
  mkdir -p "$(dirname "${LLAMA_DIR}")"
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "${LLAMA_DIR}"
fi

F16="${BASENAME}-f16.gguf"
python "${LLAMA_DIR}/convert_hf_to_gguf.py" "${MODEL_DIR}" --outtype f16 --outfile "${F16}"

QUANTIZE_BIN="${LLAMA_DIR}/build/bin/llama-quantize"
if [ ! -x "${QUANTIZE_BIN}" ]; then
  echo "warning: ${QUANTIZE_BIN} not found; build llama.cpp first" >&2
  echo "  cmake -B ${LLAMA_DIR}/build -S ${LLAMA_DIR} && cmake --build ${LLAMA_DIR}/build -j" >&2
  exit 1
fi

for QUANT in Q8_0 Q5_K_M Q4_K_M; do
  "${QUANTIZE_BIN}" "${F16}" "${BASENAME}-${QUANT}.gguf" "${QUANT}"
done

ls -lh "${BASENAME}"-*.gguf
