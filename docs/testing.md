# Testing & acceptance criteria

Four tiers, cheapest first. Everything runs on **Kaggle** — the local machine
only needs the repo (no installs).

| Tier | What it proves | Where | Time |
|---|---|---|---|
| **T0** | unit logic: answer parsing, config merge, scorecard, notebook integrity | any Python | seconds |
| **T1** | end-to-end wiring: data → template → packing → train → checkpoint → eval | Kaggle T4 | ~5–10 min |
| **T2** | M1 result: fine-tuned model beats base on GSM8K | Kaggle T4 | ~1–3 h |
| **T3** | public scorecard: same harness as other models | Kaggle / CPU | ~1–4 h |

## T0 — unit tests

```bash
pip install pytest pyyaml numpy pandas    # CI does this; no torch needed
PYTHONPATH=src pytest -q
```

Expected: `50 passed`. These tests need no GPU, no model download, no network.

## T1 — pipeline smoke test

**Option A (notebook):** open `notebooks/atom_slm_kaggle_t4.ipynb`, set
`SMOKE_TEST = True` in the config cell, Run all. It trains 5 steps on 64 rows,
evaluates 50 GSM8K items, and disables Hub pushes.

**Stage 2 (stronger recipe):** `notebooks/atomslm_dev_stage2.ipynb` — 100k rows,
2 epochs, LR 1e-4, numeric-answer filter, best-checkpoint selection, and
evaluation on the exact training prompt. Use it after M1 accepts.

**Single-checkpoint test:** `notebooks/test_notebook.ipynb` loads any Hub
checkpoint, asks it questions, prints the raw answer and the parsed final
answer, and runs a 20-item GSM8K sanity check. No repo clone needed.

**Option B (script):**

```bash
./scripts/smoke_test.sh
```

It runs the unit tests, a 5-step training run, an eval of the saved smoke
checkpoint, a 20-item base-model eval, and checks that every artifact exists.
Expected final line: `SMOKE TEST PASSED`.

Pass criteria:

- [ ] training reaches step 5 and writes `model/config.json` + `model/metrics.json`
- [ ] the saved checkpoint loads and generates without error
- [ ] the base model is downloadable and evaluated (accuracy is ~0–5% on 20 items)
- [ ] no traceback, no CUDA OOM, no NaN loss

If this passes, the notebook is safe to run with `SMOKE_TEST = False`.

## T2 — M1 acceptance criteria

Run the notebook with `SMOKE_TEST = False` (60k MetaMathQA rows, 1 epoch). Then
check:

| Criterion | Target | Where to look |
|---|---|---|
| Training completes | final checkpoint + `metrics.json` written | section 9 |
| Validation loss falls | first eval > last eval, no flatline | training log |
| Answer format learned | **format rate ≥ 90%** | tuned GSM8K report |
| Beats the base | **GSM8K > base** (base is usually ~1%) | comparison table |
| No degeneration | **degenerate rate < 5%** | tuned GSM8K report |
| Qualitative sanity | step-by-step math, no `"fox fox fox"` loop | sample predictions |

M1 is a *pipeline* milestone: a jump from ~1% to 5–15% GSM8K is a success.
Bigger gains belong to M2 (more data) and M3 (larger base/distillation).

```bash
# what the notebook runs, if you prefer the CLI:
python -m atom_slm.evaluate --model runs/m1-smollm2-135m-metamath \
  --limit 200 --num-fewshot 0 --output results/m1-gsm8k.json
python -m atom_slm.evaluate --compare results/base-gsm8k.json results/m1-gsm8k.json
```

Keep `--num-fewshot 0` (the default): training rows are 0-shot and end with
`### Solution:`, so few-shot prompts change the task and can drop accuracy to
near zero even for a good checkpoint.

```bash
# if a number looks wrong, inspect the run before tuning hyperparameters:
python -m atom_slm.diagnose --run runs/m1-smollm2-135m-metamath
```

It prints the loss curve, the mean weight change vs the base model, and a few
raw generations, which separates "training did not work" from "evaluation
asked the wrong question".

## T3 — comparable scorecard

Numbers are only comparable if produced by the **same harness, prompts, seeds
and few-shot counts**. Use:

```bash
./scripts/eval_lm_eval.sh HuggingFaceTB/SmolLM2-135M
./scripts/eval_lm_eval.sh runs/m1-smollm2-135m-metamath
```

Record every model in `results/results.csv`, then:

```bash
python -m atom_slm.scorecard --results results/results.csv --output results/scorecard.csv
```

## Failure triage

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: atom_slm` | `src/` not on path | run via `scripts/*.sh` or `export PYTHONPATH=src` |
| CUDA OOM on T4 | batch/length too high | halve `PER_DEVICE_BATCH`, double `GRAD_ACCUM` |
| `SFTTrainer does not support nn.DataParallel` | both T4s visible to one process | trainer pins GPU 0; rerun, or set `CUDA_VISIBLE_DEVICES=0` |
| Loss = NaN | bf16 on T4, or bad LR | `precision: fp16`; LR 5e-5 (full) / 2e-4 (LoRA) |
| Loss flat from step 0 | chat template mismatch | print a formatted sample; check `tokenizer.chat_template` |
| Format rate ~0% | `####` normalization missing | inspect dataset `solution` text before training |
| Format rate low + heavy repetition after training | eval prompt differs from training prompt | evaluate with `--num-fewshot 0`; the prompt must end with `### Solution:` |
| Accuracy ~base after training | checkpoint saved but barely trained | `python -m atom_slm.diagnose --run <run>` (loss curve + weight delta) |
| Model repeats a word | undertrained / LR too high | train longer, lower LR, verify data quality |
| Hub push 401/403 | token missing or read-only | create a **write** token as `HF_TOKEN` |
| Eval extremely slow | running full 1319 items by accident | `--limit 200` (or `EVAL_LIMIT`) |

## Recording results

- Keep the raw JSON from every eval (`results/*.json` is ignored by git only
  for runs — commit small summaries if you want them versioned).
- Put the base and tuned numbers side by side; never report a tuned score
  without its baseline.
- Note the resolved `transformers`/`trl` versions from the notebook's first cell.
