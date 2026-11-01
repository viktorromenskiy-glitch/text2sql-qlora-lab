"""Post-processing regression check and error collection on the final configuration.

Configuration: LoRA adapter text2sql-checkpoints-3epoch/final (r=16, alpha=32,
3 epochs), techniques postprocess + value_retrieval + sample_values, no few-shot.
Evaluated on the first 200 examples of Spider dev.

One generation pass per example (saves GPU time). For each example:
1. Build the prompt with sample_values and value_retrieval hints (the final
   configuration that scored 0.785 = 157/200 in scripts/evaluate.py).
2. Generate SQL once.
3. Compare the execution result BEFORE post-processing (raw SQL) with the result
   AFTER (correct_table_names + correct_column_names + normalize_string_quotes),
   which isolates regressions caused by post-processing from the other techniques.
4. For every finally wrong answer, save the details for manual error categorization.

Output: final_regression_and_errors.json with the keys "regressions",
"improvements", "all_wrong_final".
"""
from __future__ import annotations

import json

from src.dataset import get_spider_db_dir, load_spider
from src.evaluator import compare_results, execute_sql
from src.model import load_lora_model
from src.prompt_formatter import CHATML_IM_END, format_prompt_with_samples
from src.sanitizer import extract_sql
from src.sql_postprocess import correct_column_names, correct_table_names, normalize_string_quotes
from src.utils import set_seed, setup_logging
from src.value_retrieval import find_value_hints

SAMPLE_SIZE = 200
DATA_DIR = "data/spider"
TIMEOUT = 10
CHECKPOINT_DIR = "/content/drive/MyDrive/text2sql-checkpoints-3epoch"
CHECKPOINT = "final"


def build_prompt(schema: dict, question: str, db_path: str) -> str:
    """Build the prompt exactly as in the final configuration:
    sample values embedded in the schema, value-retrieval hints added on top."""
    prompt = format_prompt_with_samples(schema, question, db_path)
    hints = find_value_hints(question, schema, db_path)
    if hints:
        marker = f"{CHATML_IM_END}\n"
        insert_at = prompt.rfind(marker)
        prompt = prompt[:insert_at] + "\n" + "\n".join(hints) + prompt[insert_at:]
    return prompt


def main() -> None:
    setup_logging()
    set_seed(42)

    print("Loading Spider dev split...")
    examples = load_spider("dev", data_dir=DATA_DIR)[:SAMPLE_SIZE]
    db_dir = get_spider_db_dir("dev", data_dir=DATA_DIR)

    print(f"Loading LoRA model from {CHECKPOINT_DIR}/{CHECKPOINT}...")
    model, tokenizer = load_lora_model(f"{CHECKPOINT_DIR}/{CHECKPOINT}")

    regressions = []
    improvements = []
    all_wrong_final = []

    for i, example in enumerate(examples):
        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"
        gold_rows = execute_sql(example["query"], db_path, TIMEOUT)

        prompt = build_prompt(example["schema"], example["question"], db_path)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        raw_sql = extract_sql(raw_output)
        raw_rows = execute_sql(raw_sql, db_path, TIMEOUT) if raw_sql is not None else None
        raw_correct = compare_results(raw_rows, gold_rows)

        corrected_sql = raw_sql
        if corrected_sql is not None:
            corrected_sql = correct_table_names(corrected_sql, example["schema"]["table_names_original"])
            corrected_sql = correct_column_names(corrected_sql, example["schema"])
            corrected_sql = normalize_string_quotes(corrected_sql, example["schema"])
        corrected_rows = execute_sql(corrected_sql, db_path, TIMEOUT) if corrected_sql is not None else None
        corrected_correct = compare_results(corrected_rows, gold_rows)

        if raw_correct and not corrected_correct:
            regressions.append({
                "question": example["question"], "difficulty": example["difficulty"],
                "raw_sql": raw_sql, "corrected_sql": corrected_sql, "gold_sql": example["query"],
            })
        elif not raw_correct and corrected_correct:
            improvements.append({"question": example["question"], "difficulty": example["difficulty"]})

        if not corrected_correct:
            all_wrong_final.append({
                "question": example["question"], "difficulty": example["difficulty"],
                "predicted_sql": corrected_sql, "gold_sql": example["query"],
                "db_id": example["db_id"],
                "category": "no_sql" if corrected_sql is None else ("execution_error" if corrected_rows is None else "wrong_result"),
            })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(examples)} done")

    print(f"\n=== REGRESSIONS from post-processing (correct -> wrong): {len(regressions)} ===")
    for r in regressions:
        print(f"\nQ: {r['question']}")
        print(f"Before: {r['raw_sql']}")
        print(f"After:  {r['corrected_sql']}")

    print(f"\n=== IMPROVEMENTS from post-processing: {len(improvements)} ===")
    for imp in improvements:
        print(f"  [{imp['difficulty']}] {imp['question'][:60]}")

    print(f"\n=== TOTAL wrong final answers (for error analysis): {len(all_wrong_final)} ===")

    with open("final_regression_and_errors.json", "w", encoding="utf-8") as f:
        json.dump({
            "regressions": regressions,
            "improvements": improvements,
            "all_wrong_final": all_wrong_final,
        }, f, indent=2)
    print("\nSaved to final_regression_and_errors.json")


if __name__ == "__main__":
    main()
