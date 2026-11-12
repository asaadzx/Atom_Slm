#!/usr/bin/env bash
# Install the training stack inside a Kaggle GPU notebook.
# Torch is preinstalled by Kaggle — reinstalling it is the fastest way to break
# the image (CUDA mismatch), so it is deliberately not listed here.
set -euo pipefail

python -m pip install -q -U \
  "transformers>=4.46" \
  "trl>=0.13" \
  "peft>=0.13" \
  "accelerate>=1.0" \
  "datasets>=3.0" \
  "huggingface-hub>=0.24" \
  "sentencepiece>=0.2" \
  "protobuf>=4.25" \
  "bitsandbytes>=0.44" \
  "lm-eval>=0.4.5" \
  "pytest>=8.0" \
  "pyyaml>=6.0"

python - <<'PY'
import torch, transformers, trl, peft, datasets
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | gpus {torch.cuda.device_count()}")
print(f"transformers {transformers.__version__} | trl {trl.__version__} | peft {peft.__version__}")
print(f"datasets {datasets.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        print(f"GPU {i}: {name} (sm_{major}{minor})")
PY
