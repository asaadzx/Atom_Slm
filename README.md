# ATOM-SLM

A **~135M-parameter math SLM** that runs locally on a phone or a student laptop:
no GPU, ~100 MB after 4-bit quantization, 4k–8k context, and decent at
grade-school → early high-school word problems (GSM8K-style).

This repo is the reproducible codebase for the project plan in
[`docs/Project ATOM-SLM - Plan.md`](docs/Project%20ATOM-SLM%20-%20Plan.md):
data formatting, SFT training, GSM8K evaluation, the ATOM Index scorecard, and
a ready-to-run **Kaggle T4** notebook.

> **Status:** M1 pipeline wired and ready to run. Old 68M `atom-slm` weights are
> retired — see [plan §2](docs/Project%20ATOM-SLM%20-%20Plan.md) for the
> post-mortem.

---

## Why this repo exists

The first ATOM (68.7M params) collapsed into `"fox fox fox…"`: 95% of its
parameters were the 262k-token embedding table, it had 4 layers, 512 context,
and almost no training. ATOM-SLM starts from a proven tiny base
(**SmolLM2-135M / 360M**), fixes the vocabulary and depth, and trains on
step-by-step math data with a strict answer format.

Locked design decisions (details in plan §4–§5):

| | |
|---|---|
| Architecture | dense (not MoE — MoE saves compute, not RAM) |
| Position / attention | RoPE, GQA (~2–4 KV heads) |
| Norm / activation | pre-RMSNorm, SwiGLU |
| Vocabulary | ~49k (reuse the base tokenizer) |
| Depth | 16–20 layers for the from-scratch model |
| Deployment | GGUF Q4_K_M via llama.cpp |

## Repository layout

```
.
├── configs/                  # M1 / M2 / M3 / debug training configs
├── docs/
│   ├── Project ATOM-SLM - Plan.md   # the full project plan
│   └── kaggle.md                    # Kaggle T4 playbook (limits, resume, tips)
├── kaggle/
│   └── kernel-metadata.json         # template for `kaggle kernels push`
├── notebooks/
│   └── atom_slm_kaggle_t4.ipynb     # end-to-end training on a free T4
├── results/                  # eval JSONs, scorecards, example CSV
├── scripts/                  # thin shell wrappers (train, eval, GGUF, resume)
├── src/atom_slm/
│   ├── prompts.py            # answer parsing + `####` normalization (no deps)
│   ├── data.py               # MetaMathQA / OpenMathInstruct-2 → chat format
│   ├── config.py             # YAML config with defaults + overrides
│   ├── train.py              # SFT trainer (full FT or LoRA, fp16 on T4)
│   ├── evaluate.py           # fast GSM8K eval + base-vs-tuned report
│   ├── merge.py              # merge LoRA adapters (M4)
│   └── scorecard.py          # ATOM Index (plan §17.5)
└── tests/                    # unit tests for parsing, config, scorecard
```

## Quickstart (local)

```bash
git clone https://github.com/asaadzx/Atom_Slm.git
cd Atom_Slm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q                                             # fast, no model download
python -m atom_slm.train --config configs/debug.yaml --smoke-test
```

The smoke test runs 5 steps on 64 MetaMathQA rows and verifies the whole
pipeline (data → chat template → packing → training → checkpoint) in minutes.

## Train on Kaggle (free T4)

**The notebook:** [`notebooks/atom_slm_kaggle_t4.ipynb`](notebooks/atom_slm_kaggle_t4.ipynb)

1. Import it into Kaggle.
2. Settings: **GPU T4 x2**, **Internet on**.
3. Secrets: `HF_TOKEN` (write access), optionally `WANDB_API_KEY`.
4. Edit the config cell (`METHOD`, `MAX_SAMPLES`, `HF_REPO_ID`, …) and run all.

It clones this repo, installs the training stack, trains, evaluates GSM8K
**before and after**, pushes checkpoints to the Hub, and prints a comparison
table. Full walkthrough: [`docs/kaggle.md`](docs/kaggle.md).

```bash
# from a local machine, the same thing without notebooks:
./scripts/setup_kaggle.sh        # Kaggle only; never reinstalls torch
./scripts/train.sh configs/m1_smolm2_135m_metamath.yaml
```

### Training configs

| Config | Milestone | Base | Method | Data |
|---|---|---|---|---|
| `configs/debug.yaml` | — | SmolLM2-135M | full | 128 MetaMathQA rows |
| `configs/m1_smolm2_135m_metamath.yaml` | M1 | SmolLM2-135M | full FT | 60k MetaMathQA |
| `configs/m2_smolm2_135m_openmath.yaml` | M2 | SmolLM2-135M | full FT | 500k OpenMathInstruct-2 |
| `configs/m3_smolm2_360m_lora.yaml` | M3 | SmolLM2-360M | LoRA r=32 | 500k OpenMathInstruct-2 |

Key T4 settings: **fp16 + GradScaler** (no bf16 on Turing), **SDPA attention**
(not FlashAttention-2), gradient checkpointing, sequence packing, effective
batch 64 via gradient accumulation.

### Answer format

Every solution is normalized to end with:

```
#### <answer>
```

`atom_slm.prompts.ensure_answer_marker` rewrites MetaMathQA's
`The answer is: 18` and appends OpenMathInstruct-2's `expected_answer`, so
training and evaluation share one format. The evaluator accepts `####`,
`\boxed{}`, and "answer is …", and compares numbers with a small relative
tolerance.

## Evaluate

Fast, local, honest (no last-number fallback):

```bash
python -m atom_slm.evaluate --model runs/m1-smollm2-135m-metamath --limit 200 \
  --output results/m1-gsm8k.json

python -m atom_slm.evaluate --compare results/base.json results/m1-gsm8k.json
```

For the public scorecard, run the official harness with the **same command for
every model** (plan §17):

```bash
./scripts/eval_lm_eval.sh HuggingFaceTB/SmolLM2-135M
```

Then build the composite index:

```bash
cp results/results.example.csv results/results.csv   # fill with measured numbers
python -m atom_slm.scorecard --results results/results.csv --output results/scorecard.csv
```

```
ATOM Index = 0.45·MathScore + 0.35·TextScore + 0.20·EffScore
Math-per-MB = MathScore / Q4_size_MB
```

## Deploy (M4)

```bash
# 1) merge adapters (no-op for full fine-tunes)
python -m atom_slm.merge \
  --base HuggingFaceTB/SmolLM2-135M \
  --adapter Asaadzx/atom-slm-math-360m-lora \
  --output runs/merged

# 2) convert + quantize to GGUF (Q8_0 / Q5_K_M / Q4_K_M)
./scripts/convert_gguf.sh runs/merged atom-slm-math

# 3) serve and test
llama-server -m atom-slm-math-Q4_K_M.gguf -c 4096 --port 8080
```

Target: **90–100 MB** Q4_K_M for 135M, ~250 MB for 360M, usable tokens/sec on a
mid-range CPU. Always report the Q4 vs Q8 quality delta.

## Roadmap

Mirrors plan §12 — check boxes as milestones land.

- [ ] **M1** — Recreate pipeline: SmolLM2-135M + MetaMathQA on Kaggle; working GSM8K eval
- [ ] **M2** — Scale to an OpenMathInstruct-2 subset (~0.5–1M rows)
- [ ] **M3** — 135M vs 360M + distillation; pick the base
- [ ] **M4** — Merge → GGUF → quantize → deploy → push to HF
- [ ] **M5** — *(Track B)* custom ~135M dense model from scratch
- [ ] **M6** — Context extension to 8192 (continued pretrain + YaRN)
- [ ] **M7** — Model card, datacard, blog post
- [ ] **BM1 / BM2** — ATOM Index scorecard after M4 and M6

## Development

```bash
pytest -q            # tests need only pytest, pyyaml, numpy, pandas
ruff check src tests
```

CI runs the same tests on every push (`.github/workflows/ci.yml`). Training
always writes `run_config.yaml` and `metrics.json` into the run directory, so
every number in a write-up is traceable to an exact config.

## References

- [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) ·
  [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) ·
  [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
- [TinyStories](https://arxiv.org/abs/2305.07759) ·
  [RoPE](https://arxiv.org/abs/2104.09864) ·
  [YaRN](https://arxiv.org/abs/2309.00071) ·
  [GQA](https://arxiv.org/abs/2305.13245) ·
  [LoRA](https://arxiv.org/abs/2106.09685) ·
  [QLoRA](https://arxiv.org/abs/2305.14314)
- [llama.cpp / GGUF](https://github.com/ggerganov/llama.cpp) ·
  [old model card](https://huggingface.co/Asaadzx/atom-slm)

## License

MIT — see [LICENSE](LICENSE).
