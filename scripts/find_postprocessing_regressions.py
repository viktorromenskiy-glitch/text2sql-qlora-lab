"""Находит примеры, где постобработка (correct_table_names +
correct_column_names + normalize_string_quotes) ИСПОРТИЛА уже верный
ответ - "регрессии". Обнаружено: overall_accuracy упала с 0.700 до
0.690 при включении постобработки, особенно заметно на easy (-9.8 п.п.)
- нужно найти конкретные случаи, не только агрегат.

Генерирует SQL ОДИН раз на пример (экономим GPU-время), затем сравнивает
три варианта: без постобработки / с ней - на уже сгенерированном тексте,
не перегенерируя.
"""
from __future__ import annotations

import json

from src.dataset import get_spider_db_dir, load_spider
from src.evaluator import compare_results, execute_sql
from src.model import load_lora_model
from src.prompt_formatter import format_prompt
from src.sanitizer import extract_sql
from src.sql_postprocess import correct_column_names, correct_table_names, normalize_string_quotes
from src.utils import set_seed, setup_logging

SAMPLE_SIZE = 200
DATA_DIR = "data/spider"
TIMEOUT = 10
CHECKPOINT_DIR = "/content/drive/MyDrive/text2sql-checkpoints-v2"
CHECKPOINT = "final"


def apply_postprocessing(sql: str, schema: dict) -> str:
    sql = correct_table_names(sql, schema["table_names_original"])
    sql = correct_column_names(sql, schema)
    sql = normalize_string_quotes(sql, schema)
    return sql


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

    for i, example in enumerate(examples):
        prompt = format_prompt(example["schema"], example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        original_sql = extract_sql(raw_output)
        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"
        gold_rows = execute_sql(example["query"], db_path, TIMEOUT)

        if original_sql is None:
            continue  # no SQL to compare either way

        processed_sql = apply_postprocessing(original_sql, example["schema"])

        original_rows = execute_sql(original_sql, db_path, TIMEOUT)
        processed_rows = execute_sql(processed_sql, db_path, TIMEOUT)

        original_correct = compare_results(original_rows, gold_rows)
        processed_correct = compare_results(processed_rows, gold_rows)

        if original_correct and not processed_correct:
            regressions.append({
                "question": example["question"],
                "difficulty": example["difficulty"],
                "original_sql": original_sql,
                "processed_sql": processed_sql,
                "gold_sql": example["query"],
            })
        elif not original_correct and processed_correct:
            improvements.append({
                "question": example["question"],
                "difficulty": example["difficulty"],
            })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(examples)} done")

    print(f"\n=== РЕГРЕССИИ (было верно -> стало неверно): {len(regressions)} ===")
    for r in regressions:
        print(f"\nQ: {r['question']}")
        print(f"Difficulty: {r['difficulty']}")
        print(f"До:    {r['original_sql']}")
        print(f"После: {r['processed_sql']}")

    print(f"\n=== УЛУЧШЕНИЯ (было неверно -> стало верно): {len(improvements)} ===")
    for imp in improvements:
        print(f"  [{imp['difficulty']}] {imp['question'][:60]}")

    with open("postprocessing_regressions.json", "w", encoding="utf-8") as f:
        json.dump({"regressions": regressions, "improvements": improvements}, f, indent=2)
    print("\nSaved to postprocessing_regressions.json")


if __name__ == "__main__":
    main()
