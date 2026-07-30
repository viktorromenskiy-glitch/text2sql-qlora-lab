"""Cheap A/B test of two prompt/post-processing improvements found via
literature review (Ветка "basic adjustments", branching_points_analysis.md):

1. Levenshtein-based table name correction (post-processing)
2. Sample data values embedded in the schema (prompt change)

Tests all 4 combinations (neither / just #1 / just #2 / both) on a small
subset (50 examples, not 200) of the BASELINE model - matching how the
reference write-up tested these as "basic adjustments" before any SFT,
and keeping this cheap since it needs 4x the generations of a normal run.
"""
from __future__ import annotations

import json

from src.dataset import get_spider_db_dir, load_spider
from src.evaluator import evaluate_example
from src.model import load_base_model
from src.prompt_formatter import format_prompt, format_prompt_with_samples
from src.sanitizer import extract_sql
from src.sql_postprocess import correct_table_names
from src.utils import set_seed, setup_logging

CHEAP_SUBSET_SIZE = 50  # smaller than the usual 200 - this test runs 4x the generations
DATA_DIR = "data/spider"
TIMEOUT = 10


def generate_sql(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str | None:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_sql(raw_output)


def main() -> None:
    setup_logging()
    set_seed(42)

    print("Loading Spider dev split...")
    examples = load_spider("dev", data_dir=DATA_DIR)[:CHEAP_SUBSET_SIZE]
    db_dir = get_spider_db_dir("dev", data_dir=DATA_DIR)
    print(f"Cheap test subset: {len(examples)} examples")

    print("Loading base model...")
    model, tokenizer = load_base_model()

    results_plain = []
    results_levenshtein = []
    results_samples = []
    results_both = []

    for i, example in enumerate(examples):
        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"
        gold_sql = example["query"]
        difficulty = example["difficulty"]
        table_names = example["schema"]["table_names_original"]

        # Condition A: plain (current baseline, no changes) - reuse for both #1 and #3
        prompt_plain = format_prompt(example["schema"], example["question"])
        sql_plain = generate_sql(model, tokenizer, prompt_plain)
        results_plain.append(evaluate_example(sql_plain, gold_sql, db_path, difficulty, TIMEOUT))

        # Condition B: plain generation + Levenshtein post-processing only
        sql_corrected = correct_table_names(sql_plain, table_names) if sql_plain else sql_plain
        results_levenshtein.append(evaluate_example(sql_corrected, gold_sql, db_path, difficulty, TIMEOUT))

        # Condition C: sample-values prompt, no post-processing
        prompt_samples = format_prompt_with_samples(example["schema"], example["question"], db_path)
        sql_samples = generate_sql(model, tokenizer, prompt_samples)
        results_samples.append(evaluate_example(sql_samples, gold_sql, db_path, difficulty, TIMEOUT))

        # Condition D: sample-values prompt + Levenshtein post-processing (both combined)
        sql_both = correct_table_names(sql_samples, table_names) if sql_samples else sql_samples
        results_both.append(evaluate_example(sql_both, gold_sql, db_path, difficulty, TIMEOUT))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(examples)} done")

    def acc(results: list[dict]) -> float:
        return sum(r["correct"] for r in results) / len(results)

    summary = {
        "n_examples": len(examples),
        "plain_baseline": acc(results_plain),
        "with_levenshtein_only": acc(results_levenshtein),
        "with_samples_only": acc(results_samples),
        "with_both": acc(results_both),
    }

    print("\n=== CHEAP TEST RESULTS (basic adjustments, baseline model, 50 examples) ===")
    print(json.dumps(summary, indent=2))

    with open("results_cheap_improvements_test.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved to results_cheap_improvements_test.json")


if __name__ == "__main__":
    main()
