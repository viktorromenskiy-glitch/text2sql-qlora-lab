"""Milestone 2: evaluate the fine-tuned LoRA model on the same Spider dev
sample used for the Milestone 1 baseline - honest before/after comparison.

Reuses the identical 200-example sample (same random_state) and identical
evaluation pipeline as scripts/evaluate_baseline.py, so the only variable
that changes between the two runs is the model itself - see
technical_assignment.md, "Zero-shot / few-shot fairness" requirement.

GPU-only script - see model.py's module docstring for the same caveat.
"""
from __future__ import annotations

import json

from src.dataset import get_spider_db_dir, load_spider
from src.evaluator import aggregate_results, evaluate_example
from src.model import load_lora_model
from src.prompt_formatter import format_prompt
from src.sanitizer import extract_sql
from src.utils import set_seed, setup_logging

SAMPLE_SIZE = 200  # MUST match evaluate_baseline.py - same examples, fair comparison
DATA_DIR = "data/spider"
CHECKPOINT_PATH = "/content/drive/MyDrive/text2sql-checkpoints/checkpoint-1500"  # latest from interrupted run - training stopped early on loss plateau (see technical_lessons_learned.md)


def main() -> None:
    setup_logging()
    set_seed(42)  # same seed as evaluate_baseline.py -> same 200 examples

    print("Loading Spider dev split...")
    examples = load_spider("dev", data_dir=DATA_DIR)
    examples = examples[:SAMPLE_SIZE]
    db_dir = get_spider_db_dir("dev", data_dir=DATA_DIR)
    print(f"Evaluating on {len(examples)} examples (same sample as baseline)")

    print(f"Loading LoRA model from {CHECKPOINT_PATH}...")
    model, tokenizer = load_lora_model(CHECKPOINT_PATH)

    results = []
    for i, example in enumerate(examples):
        prompt = format_prompt(example["schema"], example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predicted_sql = extract_sql(raw_output)
        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"
        result = evaluate_example(
            predicted_sql=predicted_sql,
            gold_sql=example["query"],
            db_path=db_path,
            difficulty=example["difficulty"],
        )
        results.append(result)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(examples)} done")

    aggregated = aggregate_results(results)
    print("\n=== LoRA (fine-tuned) RESULTS ===")
    print(json.dumps(aggregated, indent=2))

    with open("results_lora.json", "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print("\nSaved to results_lora.json")

    # Honest side-by-side comparison, printed right here - no separate
    # analysis step needed to see whether fine-tuning actually helped.
    try:
        with open("results_baseline.json", encoding="utf-8") as f:
            baseline = json.load(f)
        print("\n=== BASELINE vs LoRA ===")
        print(f"Overall:  {baseline['overall_accuracy']:.3f} -> {aggregated['overall_accuracy']:.3f}")
        for difficulty in ["easy", "medium", "hard", "extra"]:
            b = baseline["accuracy_by_difficulty"].get(difficulty, 0.0)
            l = aggregated["accuracy_by_difficulty"].get(difficulty, 0.0)
            print(f"{difficulty:8s}: {b:.3f} -> {l:.3f}")
    except FileNotFoundError:
        print("\n(results_baseline.json not found in current directory - "
              "copy it here to see the side-by-side comparison)")


if __name__ == "__main__":
    main()
