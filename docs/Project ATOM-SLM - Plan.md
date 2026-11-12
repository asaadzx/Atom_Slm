# Project ATOM-SLM — Plan

> Building a small, math-capable language model that runs on phones and student laptops.
> Companion note: [[Small Language Models - Concepts & Resources]]

**Status:** planning / restart
**Owner:** Asaad
**Updated:** 2026-10-12

---

## 1. Vision

A **dense ~135M-parameter math SLM** that:

- Runs locally on a phone or a student laptop (CPU, no GPU required),
- Fits in **~100 MB** after 4-bit quantization,
- Handles **4k–8k context**,
- Is decent at grade-school → early high-school **math word problems** (GSM8K-style),
- Is fully reproducible on **free Kaggle GPUs**.

## 2. Retrospective: why the first ATOM failed

Facts from `atom_slm_f16.gguf` (uploaded at `Asaadzx/atom-slm`):

| Property   | Value                         | Problem                                           |
| ---------- | ----------------------------- | ------------------------------------------------- |
| Params     | 68.7M                         | tiny                                              |
| Layers     | 4                             | far too shallow for reasoning                     |
| Hidden     | 256                           | too narrow                                        |
| Vocab      | 262,144 (Gemma tokenizer)     | **65.5M of 68.7M params are the embedding table** |
| Embeddings | tied                          | fine                                              |
| Context    | 512                           | tiny                                              |
| Output     | *"fox fox fox…"* (degenerate) | undertrained / didn't converge                    |

**Root causes:** (1) undertrained, (2) vocab dominates the model, (3) too few layers, (4) no chat/CoT data.

**Decision:** do **not** continue the old weights. Start from a proven tiny base and fine-tune. Learn from the lessons above.

## 3. Strategy

Two-track approach — do the cheap win first, then the "own architecture" goal:

- **Track A (fast, recommended first):** fine-tune an existing tiny base (**SmolLM2-135M** or **SmolLM2-360M**) on math data for GSM8K-style reasoning. Best result-per-effort.
- **Track B (build-your-own):** train a custom dense SLM from scratch with the corrected architecture. Slower, more educational, lower ceiling without lots of pretraining data.

We do **Track A to learn the pipeline**, then apply the knowledge to Track B.

## 4. Design decisions (locked)

| Decision         | Choice                                              | Why                                                                           |
| ---------------- | --------------------------------------------------- | ----------------------------------------------------------------------------- |
| Architecture     | **Dense** (not MoE)                                 | MoE saves *compute*, not memory; all experts must fit in RAM. Bad for mobile. |
| Position         | **RoPE**                                            | standard, separable freq control                                              |
| Context strategy | **Train at target length**, optional **YaRN** later | tiny models don't extrapolate well                                            |
| Attention        | **GQA** (~2–4 KV heads)                             | 4× smaller KV cache → mobile long-context                                     |
| Norm             | **RMSNorm**, pre-norm                               | cheap, stable                                                                 |
| Activation       | **SwiGLU**                                          | +1–2% quality                                                                 |
| Vocab            | **~49k** (reuse a base tokenizer)                   | big memory savings vs 262k                                                    |
| Quantization     | **Q4_K_M GGUF**                                     | mobile target                                                                 |
| Depth            | **16–20 layers**                                    | depth helps reasoning                                                         |

## 5. Proposed architecture (Track B reference)

```
hidden_size        = 768
num_layers         = 16
num_heads          = 12      # head_dim = 64
num_kv_heads       = 3       # GQA (4 q-heads per kv-head)
intermediate_size  = 2048    # SwiGLU
max_position       = 4096    # trained; YaRN -> 8192 optional
rope_theta         = 10000
vocab_size         = 49152
tie_embeddings     = true
total params       ~ 135M (embed ~38M + body ~97M)
```

Target deploy size ≈ **90–100 MB** (Q4_K_M).

## 6. Data plan

| Dataset | Role | Notes |
|---|---|---|
| `nvidia/OpenMathInstruct-2` | main math SFT | ~14M examples, 32 parquet shards; **subsample** (start ~1M) |
| `meta-math/MetaMathQA` | extra math SFT | 395K examples, single JSON (`query`, `response`, `type`) |
| General web text (e.g. FineWeb-Edu / SmolLM2 data) | optional continued-pretrain (Track B) | improves coherence before SFT |

- **Format** into chat/CoT: `question -> step-by-step solution -> final answer`.
- **Packing** on, with correct attention masks.
- Hold out a clean validation set; keep GSM8K test **untouched** for eval.
- **Balance**: math SFT only can degrade general language — mix in ~10–20% general instruction data if we care about non-math chat.

## 7. Training pipeline (algorithms)

**Stage 0 — Setup:** tokenizer (reuse base's), chat template, loss = causal cross-entropy.

**Stage 1 — (Track A) Math SFT:**
- Base: SmolLM2-135M/360M.
- Full fine-tune if budget allows, else **LoRA** (r=16–64) or **QLoRA**.
- Optimizer **AdamW**, cosine schedule, warmup ~2–3%.
- Precision **fp16 + GradScaler** (T4/P100), **FlashAttention-2**, **gradient checkpointing**, **packing**.

**Stage 2 — (Optional) Distillation:**
- Use a strong teacher (e.g. Qwen2.5-Math or a large model) to generate/re-rank solutions; train student on teacher outputs. Cheap way to boost a small model.

**Stage 3 — (Optional) DPO/preference** for answer formatting and correctness.

**Stage 4 — Deploy:** merge adapters → convert to **GGUF** → quantize `Q4_K_M`/`Q8_0` → test in llama.cpp/LM Studio → push to HF.

## 8. Tech stack

| Layer           | Tool                                                                |
| --------------- | ------------------------------------------------------------------- |
| Framework       | **PyTorch** + **Hugging Face Transformers**                         |
| Training        | **TRL** (`SFTTrainer`), **Accelerate**                              |
| PEFT            | **LoRA / QLoRA** (`peft`, `bitsandbytes`)                           |
| Speed           | **Unsloth** (optional), FlashAttention-2                            |
| Data            | `datasets`, `tokenizers`, `polars`/`pandas`                         |
| Eval            | `lm-eval-harness`, custom GSM8K script                              |
| Convert/Deploy  | `llama.cpp` (`convert_hf_to_gguf.py`, `llama-quantize`), LM Studio  |
| Mobile runtimes | llama.cpp, MLC-LLM / WebLLM, ONNX Runtime Mobile, ExecuTorch        |
| Compute         | **Kaggle** (free T4/P100)                                           |
| Storage         | HF Hub (models + datasets), external drive `Hard-Asaad-yad` (local) |
|                 |                                                                     |

Local machine already has: `uv`, `transformers`, `trl`, `accelerate`, `unsloth`, `hf` CLI.

## 9. Kaggle workflow & limits

- Free tier: **T4 (often 2×) or P100**, **~30 GPU-hrs/week**, **9–12h sessions**, ~20 GB working disk.
- **T4/P100 = no bf16 → use fp16 + GradScaler.**
- Keep datasets on `/kaggle/input` (or download a subset); checkpoints to `/kaggle/working`; push to HF Hub for persistence.
- Session dies → resume from last HF-pushed checkpoint. Always save adapters/checkpoints regularly.
- `OpenMathInstruct-2` is huge → **stream/subsample**, don't download all.

## 10. Evaluation plan

- **Primary:** GSM8K exact-match accuracy (compare base vs fine-tuned).
- **Secondary:** MATH (subset), held-out MetaMathQA, perplexity.
- **Sanity:** qualitative prompts; check for repetition/degeneracy (the old failure mode).
- **Report:** always show base baseline next to the tuned model.

## 11. Deployment targets

- **Phone / laptop:** GGUF `Q4_K_M` via llama.cpp; optional WebLLM/MLC for browser demos.
- **Size budget:** 135M → ~90–100 MB Q4; 360M → ~250 MB Q4.
- **Speed check:** tokens/sec on CPU (target a few tok/s+ on a mid laptop).

## 12. Roadmap

- [ ] **M1** — Recreate pipeline: fine-tune SmolLM2-135M on MetaMathQA (small) on Kaggle; get a working GSM8K eval. *(Track A)*
- [ ] **M2** — Scale data to OpenMathInstruct-2 subset (~1M); tune; measure gain.
- [ ] **M3** — Compare 135M vs 360M; pick base. Add distillation.
- [ ] **M4** — Convert + quantize + deploy; push to HF; demo on phone/laptop.
- [ ] **M5** — *(Track B)* Build the custom ~135M dense model (spec §5); continued-pretrain on general text, then math SFT.
- [ ] **M6** — Context extension to 8192 (continued pretrain + YaRN); re-eval.
- [ ] **M7** — Write-up: architecture, data, results, lessons vs the old ATOM-1.
- [ ] **BM1** — Benchmark after M4: run §17 scorecard vs open SLMs (math + text + efficiency); rank by ATOM Index. *(Track A)*
- [ ] **BM2** — Benchmark after M6: rerun §17 on the long-context custom model; compare vs BM1. *(Track B)*

**Dependencies:** `M0 → M1 → M2 → M3 → M4` (Track A). `M5 → M6` (Track B) can run in parallel using the pipeline from M1. `M7` last. Benchmarking (§17) runs **after M4 and again after M6**.

## 13. Roadmap — Detailed Walkthrough (M0–M7)

> This section expands the checklist above. Each milestone documents its **objective**, **detailed steps**, **hyperparameters**, **commands**, **success criteria**, **pitfalls**, and **artifacts**. Do **M0** once — it applies to everything.

### M0 — Environment & housekeeping (do once, reuse everywhere)

**Objective:** a reproducible Kaggle environment that survives session resets.

**Steps**
1. Create a Kaggle Notebook → **Accelerator: GPU T4 x2**, **Internet: On**.
2. Add **secrets**: `HF_TOKEN` (a Hugging Face *write* token), optionally `WANDB_API_KEY`.
3. Install libraries (pin versions for reproducibility):

   ```bash
   !pip -q install -U transformers trl peft accelerate bitsandbytes datasets evaluate sentencepiece protobuf
   !pip -q install lm-eval                          # for benchmarking
   !pip -q install flash-attn --no-build-isolation  # optional; T4 may not support — skip if it fails
   ```

4. Log in to the Hub:

   ```python
   import os
   from huggingface_hub import login
   login(os.environ["HF_TOKEN"])
   ```

5. **Persistence plan:** write checkpoints to `/kaggle/working`, and `push_to_hub` every N steps. Kaggle sessions die at ~9–12h and working dirs are wiped.
6. Set seeds (`transformers.set_seed(42)`, `torch.manual_seed`).

**Success:** `!nvidia-smi` shows the GPU; all imports work; HF login OK.
**Pitfalls:** T4/P100 **don't support bf16** → use fp16 + GradScaler. Don't `pip install torch` (pre-installed).
**Artifacts:** a reusable notebook template you clone per milestone.
**Compute:** ~10 min.

---

### M1 — Recreate the pipeline (Track A baseline)

**Objective:** prove end-to-end training works and beat the base model on GSM8K. This is the *learning milestone* — keep it small.

**Steps**
1. Load tokenizer + base model:

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   base = "HuggingFaceTB/SmolLM2-135M"
   tok = AutoTokenizer.from_pretrained(base)
   model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="float16")
   ```

2. Load `meta-math/MetaMathQA` (~395K). Convert each row (`query`, `response`) to **chat/CoT** format using the **base model's chat template** (`tokenizer.apply_chat_template`).
3. Tokenize with **packing** (`packing=True` in `SFTConfig`) and correct attention masks. Split 98/2 train/val.
4. Configure `SFTTrainer`/`SFTConfig`:
```
   | Hyperparameter | Start value |
   |----------------|-------------|
   | learning rate  | `2e-5` (LoRA) – `5e-5` (full FT) |
   | schedule       | cosine + 3% warmup |
   | epochs         | 1–3 |
   | effective batch| 64–128 (via grad accumulation) |
   | max_seq_len    | 1024 |
   | optimizer      | AdamW (8-bit optional) |
   | precision      | fp16 |
   | gradient 
      checkpointing | on |
   | method         | full FT if 135M fits, else LoRA (r=16) |

```
5. Train; log loss. Expect val loss to drop steadily; if it plateaus immediately, learn rate or template is wrong.
6. **Evaluate GSM8K before and after** with `lm-eval-harness` (see §17 for the command).
7. Push the model/adapter to `Asaadzx/atom-slm-math-135m`.

**Success:** coherent generations (no repetition), GSM8K strictly > base.
**Pitfalls:** wrong chat template; packing without masking (tokens "leak" across examples); CoT answers where the model never emits a final answer format.
**Artifacts:** trained checkpoint, eval table (base vs tuned).
**Compute:** ~1–3 GPU-hrs.

---

### M2 — Scale the data

**Objective:** more (cleaner) math data → bigger gain.

**Steps**
1. Stream `nvidia/OpenMathInstruct-2` (32 parquet shards) instead of downloading all. Take a **subset** (start 500K–1M rows).
2. Mix with MetaMathQA; **dedup** (MinHash/exact), filter very long/short CoT, enforce one consistent answer format (`#### <answer>` or `\boxed{}`).
3. Optional **curriculum**: easy problems first, then hard.
4. Re-run M1 training with the new mix; run the **same** eval for an apples-to-apples comparison.
5. Ablation: data size vs GSM8K to find where returns flatten.

**Hyperparameters:** same as M1; try `max_seq_len 2048`, 1–2 epochs (bigger data).
**Success:** measurable improvement over M1 on GSM8K + held-out MetaMathQA.
**Pitfalls:** noisy/duplicated solutions hurt; over-fitting small subsets; token-length explosion.
**Artifacts:** dataset mix recipe, scaling curve.
**Compute:** ~2–6 GPU-hrs.

---

### M3 — Compare bases + add distillation

**Objective:** choose the base, and squeeze more from it via a teacher.

**Steps**
1. Repeat M2 SFT on **SmolLM2-360M** (and optionally **Qwen2.5-0.5B**).
2. Compare on GSM8K/MATH plus **size, RAM and CPU speed** (a 360M may not be worth 2.7× the size for +3% accuracy).
3. **Distillation (optional but strong):** use a teacher (e.g. **Qwen2.5-Math-1.5B/7B**) to generate or re-rank step-by-step solutions for the *training* prompts; train the student on teacher outputs (off-policy sequence KD). Optionally logit-level KD.
4. Pick the winner (accuracy-per-MB) as the Track A deliverable.

**Success:** a defensible base choice with data; student benefits from teacher.
**Pitfalls:** teacher leakage into eval prompts (keep test sets clean); distillation adds pipeline complexity.
**Artifacts:** base-comparison table, distilled checkpoint.
**Compute:** ~3–8 GPU-hrs.

---

### M4 — Convert, quantize, deploy

**Objective:** turn the checkpoint into a small, runnable artifact.

**Steps**
1. Merge LoRA adapters into base weights.
2. Convert HF → GGUF:

   ```bash
   python llama.cpp/convert_hf_to_gguf.py ./merged --outfile atom-slm-math.gguf --outtype f16
   llama-quantize atom-slm-math.gguf atom-slm-math-Q8_0.gguf  Q8_0
   llama-quantize atom-slm-math.gguf atom-slm-math-Q5_K_M.gguf Q5_K_M
   llama-quantize atom-slm-math.gguf atom-slm-math-Q4_K_M.gguf Q4_K_M
   ```

3. Test each quant: `llama-server -m <file> -c 4096`, then prompt with sample math.
4. Measure **file size, RAM, and tokens/sec on CPU**.
5. Push to HF with a **multi-quant** repo + README (include the benchmark table from §17).
6. Demo: load in **LM Studio**; optionally run on a phone (llama.cpp Android / MLC).
7. Re-run GSM8K on the **quantized** model (quality can drop at low bits) and report the Q4 vs Q8 delta.

**Success:** ~90–250 MB GGUF that answers simple math offline at usable speed.
**Pitfalls:** quant degradation at Q4 for tiny models; chat-template must be embedded in GGUF metadata.
**Artifacts:** GGUF files, deployment README, speed table.
**Compute:** ~30 min CPU.

---

### M5 — Build the custom dense model (Track B)

**Objective:** train *your own* architecture (spec §5) from scratch.

**Steps**
1. Define the config: 16 layers, hidden 768, GQA 12/3, FFN 2048 (SwiGLU), RMSNorm, RoPE, vocab ~49k, tie embeddings.
2. Tokenizer: reuse SmolLM2's (49,152) for comparability, or train your own BPE (~49k) on your corpus.
3. **Stage 1 — continued/pretraining:** train on general text (e.g. a FineWeb-Edu subset or the SmolLM2 training mix). Budget ≈ **Chinchilla ~20 tokens/param** → ~2.7B tokens for 135M. Use fp16, grad clipping (1.0), warmup, cosine.
4. Watch for **loss spikes**; checkpoint often; validate on held-out text (perplexity).
5. **Stage 2 — math SFT:** reuse the M2 pipeline.
6. Compare head-to-head against Track A (§17).

**Success:** a coherent base (perplexity in a sensible range) that improves after math SFT.
**Pitfalls:** from-scratch models are easy to break (NaN, dead neurons); insufficient tokens → another "atom-slm".
**Artifacts:** training code, configs, checkpoints, loss curves.
**Compute:** ~20–60 GPU-hrs (spread across Kaggle weeks).

---

### M6 — Context extension to 8192

**Objective:** longer context without wrecking quality.

**Steps**
1. **Continued-pretrain** at 4096 (long packing), then at **8192**, on general + long-math data.
2. Set `rope_theta` (e.g. 10000 → 500000 for long ctx) or apply **YaRN**:

   ```bash
   llama-server -m atom-slm-math.gguf -c 8192 \
     --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 4096
   ```

3. Re-run math evals at short and long context; quantify any degradation.
4. Report max usable context honestly (where quality holds).

**Success:** stable quality at 8k; documented context limit.
**Pitfalls:** tiny models degrade fast with scaling — always *train* at the target length first.
**Artifacts:** long-context checkpoint, `--rope` config.
**Compute:** ~5–15 GPU-hrs.

---

### M7 — Write-up & release

**Objective:** package everything as a portfolio-grade project.

**Steps**
1. Write the **model card**: architecture, data, training recipe, and the **benchmark/scorecard from §17**.
2. Include a short **architecture diagram** and the **ablations** (data size, base choice, quant quality).
3. Publish a **dataset/datacard** and a **GitHub repo** with reproducible scripts.
4. Blog/HF post: *"What I learned: from a broken 68M toy to a useful math SLM"*, linking the old `Asaadzx/atom-slm`.
5. Optional: post to r/LocalLLaMA / HF community; collect feedback.

**Success:** anyone can reproduce your numbers from the repo.
**Artifacts:** model card, repo, blog post.

## 14. Risks & mitigations

| Risk                              | Mitigation                                          |
| --------------------------------- | --------------------------------------------------- |
| Undertraining again               | enough tokens; monitor val loss; train longer       |
| Repetition/degeneracy             | proper tokenizer, enough data, sampling/eval checks |
| Math-only data ruins general chat | mix general instruction data                        |
| Kaggle timeouts                   | checkpoint to HF often; resumable training          |
| Context scaling hurts quality     | train at target length; validate after YaRN         |
| Overfitting small datasets        | hold-out set; early stopping; LoRA regularization   |

## 15. Expected outcomes (ballpark)

> Estimates only — real numbers depend on data, budget and luck.

- **Track A (SmolLM2-360M + math SFT):** meaningful GSM8K gains over base; target **~15–35%** GSM8K. Good, useful demo.
- **Track B (custom 135M from scratch):** modest without large-scale pretraining; likely **low GSM8K**, but a clean, well-understood model and a strong learning artifact.
- **Deployable artifact:** ~100 MB Q4 GGUF that answers simple math step-by-step offline.
- **Learning outcome (the real win):** hands-on understanding of tokenizers, GQA, RoPE, SFT/LoRA, quantization, and evals.

## 16. Key corrections to remember

- **MoE does not save memory** → use dense for mobile.
- **More layers (depth) → better reasoning**, but keep head_dim 64–128 and enough width.
- **RoPE scaling is a supplement**; train at the target context length.
- **Vocab size is a memory tax** — keep it small (~50k).
- **Data quality + enough tokens** beat parameter count (TinyStories).

## 17. Benchmarking & Competitive Scoring (vs. open SLMs)

> **When:** once after **M4** (Track A), and again after **M6** (Track B). The goal is a fair, reproducible **scorecard** that places ATOM-SLM against other open small models on **math** and **general text**, plus a size/speed penalty so a "big-for-its-class" model can't win on accuracy alone.

### 17.1 Principle: fairness first

Numbers on model cards are **not comparable** (different few-shot counts, prompts, and harnesses). For a valid scorecard you must **run the same harness, versions, seeds and prompts on every model yourself**. Cite the exact `lm-eval` version.

### 17.2 What to measure

| Category | Benchmark | Setting |
|---|---|---|
| **Math** | GSM8K | 8-shot CoT |
| Math | MATH | 4-shot |
| Math | held-out MetaMathQA / OpenMathInstruct test | own split |
| **General text** | MMLU | 5-shot |
| General | ARC-easy / ARC-challenge | 25-shot / 25-shot |
| General | HellaSwag | 10-shot |
| General | PIQA / WinoGrande | 0-shot / 5-shot |
| General | Perplexity | held-out text |
| **Efficiency** | params, Q4_K_M size (MB), CPU tok/s, RAM, max context | measured |
| **Robustness** | repetition / degeneracy rate | custom probe |

### 17.3 How to run (protocol)

```bash
# 1) HF model (fp16) — same command for EVERY model
lm_eval --model hf \
  --model_args pretrained=<repo_id>,dtype=float16 \
  --tasks gsm8k,mmlu,arc_easy,arc_challenge,hellaswag,piqa,winogrande \
  --num_fewshot 5 --batch_size auto \
  --output_path results/$(basename <repo_id>)

# 2) Quantized GGUF (fair deployment test) — serve, then evaluate
llama-server -m atom-slm-math-Q4_K_M.gguf -c 4096 --port 8080 &
lm_eval --model local-completions \
  --model_args base_url=http://127.0.0.1:8080/v1/completions,model=local \
  --tasks gsm8k,arc_easy --num_fewshot 5 \
  --output_path results/atom-slm-math-Q4
```

Fix seeds; keep few-shot counts identical across models; store the raw JSON.

### 17.4 Competitor pool

| Model | Params | Q4 size | Math focus | Notes |
|---|---|---|---|---|
| SmolLM2-135M | 135M | ~100 MB | no | our Track A base |
| SmolLM2-360M | 360M | ~250 MB | no | Track A alternative |
| SmolLM2-1.7B | 1.7B | ~1.1 GB | no | upper reference |
| Qwen2.5-0.5B | 0.5B | ~350 MB | no | strong general tiny |
| Qwen2.5-1.5B | 1.5B | ~1 GB | no | strong general |
| Qwen2.5-Math-1.5B | 1.5B | ~1 GB | **yes** | math specialist ceiling |
| TinyLlama-1.1B | 1.1B | ~700 MB | no | popular reference |
| Llama-3.2-1B | 1.2B | ~800 MB | no | modern reference |
| Gemma-3-270M | 270M | ~200 MB | partial | Google tiny |
| MobileLLM-125M / 350M | 125M/350M | ~90/250 MB | no | mobile-first design |
| Phi-1.5 | 1.3B | ~900 MB | partial | text/code |

Add any new strong SOTA tiny model you find at release time.

### 17.5 Scoring methodology (composite index)

For each model `m` and benchmark `b`, robust-normalize across the pool:

```
N(m,b) = 100 * clamp( (S(m,b) - p10_b) / (p90_b - p10_b), 0, 1 )
```

Then build category scores and a single index:

```
MathScore = mean( N(GSM8K), N(MATH), N(in-domain) )
TextScore = mean( N(MMLU), N(ARC-e), N(HellaSwag), N(PIQA), N(WinoGrande) )
EffScore  = 0.5*norm(inverse Q4_size) + 0.5*norm(CPU_tokens_per_sec)

ATOM Index (0–100) = 0.45*MathScore + 0.35*TextScore + 0.20*EffScore
```

Also report **Math-per-MB** `= MathScore / Q4_size` — the real question for a mobile math model. Rank all models by ATOM Index and by Math-per-MB.

Small helper (pandas):

```python
import pandas as pd, numpy as np
df = pd.read_csv("results.csv")  # one row per model, one column per benchmark
def norm(col):
    lo, hi = np.percentile(col, 10), np.percentile(col, 90)
    return (100*np.clip((col-lo)/(hi-lo+1e-9), 0, 1))
n = {c: norm(df[c]) for c in ["gsm8k","math","mmlu","arc_e","hellaswag","piqa","winogrande"]}
df["MathScore"] = (n["gsm8k"]+n["math"])/2
df["TextScore"] = (n["mmlu"]+n["arc_e"]+n["hellaswag"]+n["piqa"]+n["winogrande"])/5
df["EffScore"]  = 0.5*norm(-df["q4_mb"]) + 0.5*norm(df["tok_s_cpu"])
df["ATOM"] = 0.45*df.MathScore + 0.35*df.TextScore + 0.20*df.EffScore
df["Math_per_MB"] = df.MathScore/df.q4_mb
print(df.sort_values("ATOM", ascending=False)[["model","ATOM","MathScore","TextScore","EffScore","Math_per_MB"]])
```

Weights (0.45/0.35/0.20) are a starting point — adjust to taste and **state them**.

### 17.6 Scorecard template (fill after M4 / M6)

> Values below are **placeholders / reported ballparks** — replace every cell with your own measured numbers.

| Model | Params | Q4 MB | GSM8K | MATH | MMLU | ARC-e | HellaSwag | PPL | tok/s CPU | ATOM |
|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 135M | ~100 | ~1% | ~0 | ~30% | ~40% | ~40% | — | — | — |
| SmolLM2-360M | 360M | ~250 | ~3% | ~1% | ~35% | ~45% | ~45% | — | — | — |
| Qwen2.5-0.5B | 0.5B | ~350 | ~20% | ~5% | ~45% | ~50% | ~50% | — | — | — |
| **ATOM-SLM (ours)** | … | … | **TBD** | TBD | TBD | TBD | TBD | TBD | TBD | **TBD** |

### 17.7 Reporting

- Put the completed scorecard in the **model card** and the **repo**, with the exact `lm-eval` version and raw JSON links.
- Show **progress**: M4 scorecard vs M6 scorecard.
- Be explicit about limitations (few-shot sensitivity, quant degradation, English-only, synthetic-data leakage risk).

---

## References

- SmolLM2 — https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- OpenMathInstruct-2 — https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
- MetaMathQA — https://huggingface.co/datasets/meta-math/MetaMathQA
- TinyStories — https://arxiv.org/abs/2305.07759
- RoPE — https://arxiv.org/abs/2104.09864 · YaRN — https://arxiv.org/abs/2309.00071
- GQA — https://arxiv.org/abs/2305.13245 · LoRA — https://arxiv.org/abs/2106.09685 · QLoRA — https://arxiv.org/abs/2305.14314
- llama.cpp GGUF — https://github.com/ggerganov/llama.cpp
- Model card of old model — https://huggingface.co/Asaadzx/atom-slm
