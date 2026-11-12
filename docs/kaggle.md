# Running ATOM-SLM on Kaggle (T4)

Everything in this guide is free-tier friendly: **T4 x2** (or P100), ~30 GPU
hours per week, 9–12 hour sessions, ~20 GB of working disk.

## 1. One-time setup

1. Create a Kaggle Notebook → **File → Import Notebook** → upload
   `notebooks/atom_slm_kaggle_t4.ipynb`.
2. In **Settings**:
   - Accelerator: **GPU T4 x2** (the notebook uses one GPU by default;
     `torchrun` is only needed if you explicitly want DDP),
   - Internet: **On**,
   - Persistence: **Files only**.
3. Add secrets (**Add-ons → Secrets**):
   - `HF_TOKEN` — a Hugging Face token with **write** access,
   - `WANDB_API_KEY` — optional.

`HF_TOKEN` is what makes a run survive a session reset: the trainer pushes
checkpoints to the Hub every `save_steps`.

## 2. What the notebook does

| Section | Purpose |
|---|---|
| 1 | GPU/driver check + dependency install (torch is *not* reinstalled) |
| 2 | Clone this repo into `/kaggle/working` and put `src/` on `sys.path` |
| 3 | Write a run config from a few obvious variables (`SMOKE_TEST` lives here) |
| 4 | Hugging Face login via Kaggle Secrets |
| 5 | Optional baseline GSM8K eval (know your starting point) |
| 6 | Optional resume of checkpoints from the Hub |
| 7 | Training (`atom_slm.train`) — full FT or LoRA |
| 8 | Post-training GSM8K eval + base-vs-tuned table |
| 9 | Artifact summary |
| 10 | Optional merge + GGUF export |

### Run the smoke test first

In section 3 set `SMOKE_TEST = True`, then Run all. That trains 5 steps on 64
rows, evaluates 50 GSM8K items, uses a `-smoke` run directory and disables Hub
pushes. If it finishes without errors (~10 min), the pipeline is healthy — set
`SMOKE_TEST = False` and start the real M1 run.

The equivalent from a terminal is `./scripts/smoke_test.sh`.

## 3. T4 rules of thumb

- **No bf16.** The code auto-detects compute capability and falls back to
  fp16. Do not force `bf16` — you will train NaNs.
- **Do not `pip install torch`.** Kaggle's image is already CUDA-matched.
- **SDPA attention**, not FlashAttention-2. SDPA is built into PyTorch and
  works on Turing; `flash-attn` needs Ampere+ for the fast kernels.
- **Gradient checkpointing on**, `max_length` 1024 for M1, 2048 for M2.
- **Effective batch 64** via `per_device_batch_size × grad_accum`.
- A 135M full fine-tune on 60k MetaMathQA rows takes roughly **1–3 h** on T4.

## 4. Surviving the 9–12 h session limit

The trainer writes checkpoints to `/kaggle/working` and (with
`hub.push_to_hub: true`) uploads each one to the Hub.

New session:

```bash
./scripts/pull_checkpoint.sh Asaadzx/atom-slm-math-135m runs/m1-smollm2-135m-metamath
python -m atom_slm.train --config configs/m1_smolm2_135m_metamath.yaml \
  --output-dir runs/m1-smollm2-135m-metamath --resume auto
```

`--resume auto` finds `checkpoint-*` and restores optimizer/scheduler state.

## 5. Pushing the notebook from a local machine (optional)

`kaggle/kernel-metadata.json` is a template. Update `id` to your Kaggle
username, then:

```bash
kaggle kernels push -p kaggle/     # expects the .ipynb next to kernel-metadata.json
kaggle kernels status <username>/atom-slm-kaggle-t4
```

To push the notebook itself as a Kaggle *dataset*-free kernel, copy
`notebooks/atom_slm_kaggle_t4.ipynb` into `kaggle/` first:

```bash
cp notebooks/atom_slm_kaggle_t4.ipynb kaggle/
```

## 6. Disk budget

| Item | Approx. |
|---|---|
| Base 135M weights (fp32, full FT) | 0.5 GB |
| Two checkpoints with AdamW state | ~4.5 GB |
| MetaMathQA download | ~0.3 GB |
| OpenMathInstruct-2 (streamed 500k rows) | ~2–4 GB in RAM, not disk |

If disk gets tight, set `save_total_limit: 1` and rely on Hub uploads.
