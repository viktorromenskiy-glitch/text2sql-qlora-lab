"""Cheap test of execution-guided self-correction (Ветка 2,
branching_points_analysis.md): does retrying failed SQL with error
feedback help specifically on hard/extra examples?

Deliberately restricted to the hard+extra subset of the same 200-example
fixed sample (not all 200) - cheaper to run, and targets exactly where
the LoRA model is weakest (see technical_lessons_learned.md: hard/extra
consistently underperform baseline across all three training attempts).

Runs BOTH direct generation and self-correction on the SAME examples, so
the comparison isolates the effect of correction alone, not sampling
variance from a different subset.
"""
from __future__ import annotations

import json

from src.dataset import get_spider_db_dir, load_spider
from src.evaluator import evaluate_example
from src.model import load_lora_model
from src.prompt_formatter import format_prompt
from src.sanitizer import extract_sql
from src.self_correction import generate_with_correction
from src.utils import set_seed, setup_logging

SAMPLE_SIZE = 200  # same fixed sample as baseline/lora eval - seed=42 below matches
DATA_DIR = "data/spider"
CHECKPOINT_DIR = "/content/drive/MyDrive/text2sql-checkpoints-v2"
CHECKPOINT = "final"
TIMEOUT = 10


def generate_direct(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str | None:
    """Single-attempt generation, no correction - for the "before" column."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_sql(raw_output)


def main() -> None:
    setup_logging()
    set_seed(42)

    print("Loading Spider dev split...")
    examples = load_spider("dev", data_dir=DATA_DIR)[:SAMPLE_SIZE]
    db_dir = get_spider_db_dir("dev", data_dir=DATA_DIR)

    hard_extra = [e for e in examples if e["difficulty"] in ("hard", "extra")]
    print(f"Cheap test subset: {len(hard_extra)} hard/extra examples (out of {len(examples)} total)")

    print(f"Loading LoRA model from {CHECKPOINT_DIR}/{CHECKPOINT}...")
    model, tokenizer = load_lora_model(f"{CHECKPOINT_DIR}/{CHECKPOINT}")

    results_direct = []
    results_corrected = []
    flipped_count = 0

    for i, example in enumerate(hard_extra):
        prompt = format_prompt(example["schema"], example["question"])
        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"

        direct_sql = generate_direct(model, tokenizer, prompt)
        direct_result = evaluate_example(direct_sql, example["query"], db_path, example["difficulty"], TIMEOUT)
        results_direct.append(direct_result)

        corrected_sql, was_corrected = generate_with_correction(model, tokenizer, prompt, db_path, TIMEOUT)
        corrected_result = evaluate_example(corrected_sql, example["query"], db_path, example["difficulty"], TIMEOUT)
        results_corrected.append(corrected_result)

        if not direct_result["correct"] and corrected_result["correct"]:
            flipped_count += 1
            print(f"  [{i+1}] FLIPPED wrong->correct (difficulty={example['difficulty']})")

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(hard_extra)} done")

    direct_acc = sum(r["correct"] for r in results_direct) / len(results_direct)
    corrected_acc = sum(r["correct"] for r in results_corrected) / len(results_corrected)

    print("\n=== SELF-CORRECTION CHEAP TEST (hard+extra only) ===")
    print(f"Direct generation accuracy:    {direct_acc:.3f}")
    print(f"With self-correction accuracy: {corrected_acc:.3f}")
    print(f"Examples flipped wrong->correct: {flipped_count} / {len(hard_extra)}")

    with open("results_self_correction_cheap_test.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_examples": len(hard_extra),
                "direct_accuracy": direct_acc,
                "corrected_accuracy": corrected_acc,
                "flipped_count": flipped_count,
            },
            f,
            indent=2,
        )
    print("\nSaved to results_self_correction_cheap_test.json")


if __name__ == "__main__":
    main()
