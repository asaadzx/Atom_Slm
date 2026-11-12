# Results

Raw evaluation artifacts live here. Model weights and datasets stay on the Hub,
never in git.

`results.example.csv` shows the shape expected by `atom_slm.scorecard`:

| column | meaning |
|---|---|
| `model` | display name |
| `params` | parameter count (e.g. `135M`) |
| `q4_mb` | Q4_K_M GGUF size in MB |
| `gsm8k`, `math`, `in_domain` | math scores, **same unit** (percent) |
| `mmlu`, `arc_e`, `hellaswag`, `piqa`, `winogrande` | general scores (percent) |
| `ppl` | held-out perplexity |
| `tok_s_cpu` | CPU tokens/sec |
| `ctx` | max context used for the speed test |

All benchmark cells must be measured with the same harness, prompts, seeds and
few-shot counts — see `scripts/eval_lm_eval.sh` and plan §17.

```bash
cp results/results.example.csv results/results.csv
# fill in your measured numbers, then:
python -m atom_slm.scorecard --results results/results.csv --output results/scorecard.csv
```
