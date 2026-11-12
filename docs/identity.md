# Model identity — teaching Atom SLM who it is

A model has no built-in notion of its name or author. Identity is **learned
text**, exactly like the `#### <answer>` format. There are three layers, and
only one of them actually works at inference time.

| Layer | What it does | Does the model "know" it? |
|---|---|---|
| Model card / README / `config.json` | documents the model for humans | **No** |
| System prompt at inference | tells the model in-context | only if it was trained to follow one |
| **SFT rows in the training data** | bakes the facts into the weights | **Yes** |

## The catch for this base model

`HuggingFaceTB/SmolLM2-135M` has **no chat template** (verified: `chat_template`
is absent from its `tokenizer_config.json`). So there is no `system` role and a
"system prompt" is just text the model has never seen. Identity advice that
starts with "add a system message" does not apply here.

Two ways forward:

1. **Bake it in** (used here): add identity Q&A rows to SFT in the same plain
   format the model already uses. Works with the current checkpoint.
2. **Switch the base** to `SmolLM2-135M-Instruct`, which *does* ship a chat
   template (`<|im_end|>`), then a system prompt works normally. This means
   redoing the math fine-tune on a different base, so it is a separate
   experiment, not a drop-in change.

## How it is wired in this repo

`data/identity/atom_identity.jsonl` contains short, consistent Q&A pairs:

```json
{"question": "Who are you?", "solution": "I am Atom SLM, an open-source math language model developed by Asaadzx. ..."}
```

They are rendered with the normal training template, so the model sees:

```
### Question:
Who are you?

### Solution:
I am Atom SLM, an open-source math language model developed by Asaadzx. ...
```

Enable mixing in any config:

```yaml
data:
  identity_path: data/identity/atom_identity.jsonl
  identity_repeat: 50      # 34 rows x 50 = 1,700 identity rows
```

`identity_repeat` controls the ratio. 34 rows repeated 50× is ~1.7% of a 100k
run; repeated 100× next to 25k math rows is ~11%. Rules of thumb:

| Goal | Ratio | Notes |
|---|---|---|
| keep it subtle | 1–3% | safest for math accuracy |
| make it reliable | 5–10% | best balance at 135M |
| drill it hard | > 20% | risks losing math quality and format |

Identity rows intentionally do **not** end with `####`; that marker stays a math
answer convention. Keep the persona block small so the model does not start
answering every question with self-description.

## Recipe used here

`configs/stage3_identity_polish.yaml`:

- start from the Stage 2 checkpoint (`base_model: Asaadzx/atom-slm-math-135m-v2`)
- 25k math rows (replay) + 3.4k identity rows (~11%)
- 1 epoch, LR **2e-5** (lower than math training — this is a small correction)
- best eval-loss checkpoint kept, ~10–20 min on a T4

Replay matters: identity-only training on a 135M model quickly forgets the math
format and starts answering *everything* with "I am Atom SLM...".

## Verify it

Ask the model these (same plain prompt, 0-shot):

```
Who are you?
What is your name?
Who made you?
Are you ChatGPT?
What can you do?
```

Pass criteria:

- [ ] always calls itself **Atom SLM**, never ChatGPT/Claude/Gemini
- [ ] names **Asaadzx** as the developer
- [ ] describes itself as an open-source **math** model
- [ ] still solves 3 GSM8K items with steps and `#### <number>`
- [ ] no "I am Atom SLM" when asked a math question

## Consistency rules

- Spell it **Atom SLM** everywhere: JSONL, model card, README, tokenizer config.
- Keep facts stable (name, author, purpose, MIT license, "small math model").
  A 135M model cannot hold a nuanced backstory; contradictions leak.
- Do not claim training details you cannot verify (dates, token counts,
  benchmark numbers). "Fine-tuned on math word problems" is enough.
- Update `README.md` on the Hub too — it is what users read, even though it does
  not change the weights.
