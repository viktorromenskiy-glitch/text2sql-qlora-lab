# Text-to-SQL Fine-tuning — QLoRA on Qwen2.5-Coder-7B

A research experiment: fine-tuning an open 7B model (QLoRA) to translate natural-language questions into SQL on the Spider benchmark, with every run, configuration and negative result documented — including the technical bugs found and fixed along the way, not just the final metric.

*Русская версия: [README.ru.md](./README.ru.md) · Українська версія: [README.uk.md](./README.uk.md)*

**In short:** Qwen2.5-Coder-7B-Instruct fine-tuned with QLoRA (r=16, 3 epochs on Spider train) plus three inference-time techniques reaches **78.5% execution accuracy (157/200)** on the first 200 examples of the Spider dev set, against 69.5% (139/200) for the same model without fine-tuning. Stack: PyTorch, Unsloth, QLoRA/PEFT, Google Colab.

This is a benchmark experiment on one public dataset, not a ready-made product: accuracy on a company's own database schema has to be measured separately.

## Result

All numbers: execution accuracy on the **first 200 examples of Spider dev** (a fixed subset, not a random sample), one run per configuration, SQLite databases.

| Configuration | Execution Accuracy |
|---|---|
| Base model, no fine-tuning (zero-shot) | 69.5% (139/200) |
| Fine-tuned adapter (r=16, 3 epochs), no inference techniques | 73.5% (147/200) |
| + SQL post-processing + value retrieval | 74.5% (149/200) |
| + sample values in the schema (**final**) | **78.5% (157/200)** |

**+9.0 points over the zero-shot baseline** (18 more examples correct out of 200).

Single-technique effects on the same adapter (same 200 examples):

| Added technique | Accuracy | Change vs. plain (73.5%) |
|---|---|---|
| SQL post-processing | 74.0% | +0.5 |
| Value retrieval | 74.0% | +0.5 |
| Sample values in the schema | 77.0% | +3.5 |
| Few-shot retrieval | 67.5% | −6.0 |

Few-shot retrieval lowered accuracy and is not used in the final configuration.

### How to read these numbers

- The subset is small: **1–3 examples (0.5–1.5 points) are within noise**. Only the larger differences (sample values, few-shot) are clearly outside it.
- The techniques and the number of epochs were chosen by looking at results on these same 200 examples, so 78.5% is likely somewhat optimistic compared with unseen data.
- Published results usually use the full dev set (1,034 examples); for reference, Llama 3 8B with QLoRA is reported at 77.2% there. The two numbers are on different subsets and are **not a like-for-like comparison**.
- Spider test labels are not available locally, so no test-set result is reported.
- Only the final adapter (r=16, 3 epochs) is kept: its weights have SHA-256 `1ab75e6db713820c7a3fb06dff0e0ddb5119f3e550eb1688e263849904118a0f`. Earlier adapters from the hyperparameter experiments are no longer stored, so those runs cannot be re-run.

## Why this isn't "just running a tutorial"

Getting to the final result required finding and fixing real technical bugs — not cosmetic ones, but ones that materially affected the outcome:

- Loss was computed over the entire prompt (including the DB schema), not just the target SQL — the model was spending gradient updates on "copying" context instead of learning to generate SQL
- No LR schedule (warmup + cosine decay)
- Resuming training after a Colab session drop silently overwrote earlier checkpoints (checkpoint numbers restarted at 0) and filled the Drive; fixed with a step offset and automatic cleanup of old checkpoints
- Post-processing quote handling corrupted string literals like `'USA'`, mistaking them for hallucinated column names

Further notes on pitfalls: [`technical_lessons_learned.md`](./technical_lessons_learned.md) (in Russian).

## Approach

1. **Hyperparameter experiments** — number of training epochs (1–4) and LoRA rank (16 vs 32) were compared on the same 200 examples; r=16 with 3 epochs was kept
2. **Diagnose before fixing** — error-type classification on 200 examples before writing any correction code
3. **Six inference-time techniques** tested individually and in combination: SQL post-processing, targeted database value lookup, sample rows in the schema, few-shot retrieval, schema pruning, advanced self-correction. Not all of them helped; negative results are documented alongside positive ones
4. **Regression analysis** — per-example comparison, not only the aggregate metric. For the final configuration, post-processing caused 0 regressions and 3 improvements (`scripts/final_regression_and_errors.py`); 43 of 200 answers remain wrong

Methodology and findings: [`implementation_priority_plan.md`](./implementation_priority_plan.md), [`error_analysis_final.md`](./error_analysis_final.md).

## Tech stack

Python · PyTorch · Unsloth · QLoRA/PEFT · Qwen2.5-Coder-7B-Instruct · Google Colab (T4/GPU) · Spider dataset

## Repository structure

```
src/            # Core code (dataset, model, evaluator, post-processing techniques)
scripts/        # Entry points (train.py, evaluate.py, final_regression_and_errors.py, diagnostic scripts)
configs/        # Training and evaluation configuration
data/README.md  # Instructions for obtaining the datasets (Spider, BIRD-SQL)
```

## Reproduction

See [`RUNBOOK_milestone2.md`](./RUNBOOK_milestone2.md) — step-by-step commands for Google Colab. `configs/eval_config.yaml` points to the final adapter folder.

```bash
python scripts/evaluate.py --model lora --postprocess --value-retrieval --sample-values
```

Note: the evaluation uses the first 200 dev examples in file order (the `seed=42` in the code does not affect which examples are picked).

## Data and model license

Spider dataset — CC BY-SA 4.0. Qwen2.5-Coder-7B-Instruct — Apache 2.0.

## About the author

Viktor Romenskiy — GenAI/ML engineer (RAG systems, LLM evaluation, fine-tuning). GitHub: [viktorromenskiy-glitch](https://github.com/viktorromenskiy-glitch)
