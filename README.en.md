# Text-to-SQL Fine-tuning — QLoRA on Qwen2.5-Coder-7B

Fine-tuning an open LLM to translate natural language questions into SQL
queries. An honest, fully documented path from zero-shot baseline to a
result surpassing a comparable published work — including the technical
bugs found and fixed along the way, not just the final metric.

*Русская версия: [README.md](./README.md) · Українська версія: [README.uk.md](./README.uk.md)*

## Result

| Stage | Execution Accuracy |
|---|---|
| Zero-shot baseline (no fine-tuning) | 69.5% |
| Naive SFT fine-tuning (before bug fixes) | 63.0–67.0% (worse than baseline) |
| SFT after fixing 3 technical bugs | 70.0–73.5% |
| **Final** (SFT + 3 inference-time techniques, epochs found via sweep) | **78.5%** |

**+9.0 points over the baseline**, surpassing a comparable published
work (Llama 3 8B, same dataset, similar QLoRA approach, 77.2% EX) by
1.3 points.

## Why this isn't "just running a tutorial"

Getting to the final result required systematically finding and fixing
**7+ real technical bugs** — not cosmetic issues, but ones that
materially affected the outcome:

- Loss was computed over the entire prompt (including the DB schema),
  not just the target SQL — the model was spending gradient updates on
  "copying" context instead of learning to generate SQL
- No LR schedule (warmup + cosine decay)
- Optimizer state silently reset when resuming training after a session drop
- The same bug class, twice — post-processing quote handling corrupted
  string literals like `'USA'`, mistaking them for hallucinated column
  names

Full write-up of every finding: [`technical_lessons_learned.md`](./technical_lessons_learned.md).

## Approach

1. **An independent hyperparameter sweep**, not copying published
   values — number of training epochs (1–4) and LoRA rank (16 vs 32)
   were tested from scratch; the optimum for this model/dataset turned
   out to differ from the reference work's choice
2. **Diagnose before fixing** — real error-type classification on 200
   examples before writing any correction code
3. **Five inference-time techniques** tested individually and in
   combination: SQL post-processing, targeted database value lookup,
   sample rows in the schema, few-shot retrieval, advanced
   self-correction — not all of them helped, negative results are
   documented alongside positive ones
4. **Regression analysis after every change** — not just the aggregate
   metric, per-example comparison

Full methodology and findings:
[`implementation_priority_plan.md`](./implementation_priority_plan.md),
[`error_analysis_final.md`](./error_analysis_final.md).

## Tech stack

Python · PyTorch · Unsloth · QLoRA/PEFT · Qwen2.5-Coder-7B-Instruct ·
Google Colab (T4/GPU) · Spider dataset

## Repository structure

```
src/            # Core code (dataset, model, evaluator, post-processing techniques)
scripts/        # Entry points (train.py, evaluate.py, diagnostic scripts)
configs/        # Training and evaluation configuration
data/README.md  # Instructions for obtaining the datasets (Spider, BIRD-SQL)
```

## Reproduction

See [`RUNBOOK_milestone2.md`](./RUNBOOK_milestone2.md) — step-by-step
commands for Google Colab.

```bash
python scripts/evaluate.py --model lora --postprocess --value-retrieval --sample-values
```

## Data and model license

Spider dataset — CC BY-SA 4.0. Qwen2.5-Coder-7B-Instruct — Apache 2.0.
