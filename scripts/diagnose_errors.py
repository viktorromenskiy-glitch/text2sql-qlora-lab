"""Priority 1, Этап 1 (диагностика) — реальное разделение ошибок LoRA-модели
по типам, прежде чем строить механизм коррекции.

Per Gemini's cheap-test methodology: не строить Execution Guard вслепую -
сначала узнать, СКОЛЬКО реальных ошибок вообще относятся к типу
"SQL не выполнился" (то, что Guard может исправить), а не к типу
"выполнился, но неверно" (то, что Guard не может исправить в принципе).

parse_failure_rate был 0.0 во всех прошлых прогонах - но это отдельная
метрика (sanitizer не нашёл ```sql-блок). НЕ отслеживалась отдельно доля
случаев, когда SQL извлечён, но execute_sql() вернул None (реальная
ошибка SQLite: несуществующая колонка/таблица, синтаксис) - это и есть
недостающая диагностика.
"""
from __future__ import annotations

import json
from collections import Counter

from src.dataset import get_spider_db_dir, load_spider
from src.evaluator import execute_sql
from src.model import load_lora_model
from src.prompt_formatter import format_prompt
from src.sanitizer import extract_sql
from src.utils import set_seed, setup_logging

SAMPLE_SIZE = 200
DATA_DIR = "data/spider"
TIMEOUT = 10
CHECKPOINT_DIR = "/content/drive/MyDrive/text2sql-checkpoints-v2"
CHECKPOINT = "final"


def main() -> None:
    setup_logging()
    set_seed(42)

    print("Loading Spider dev split...")
    examples = load_spider("dev", data_dir=DATA_DIR)[:SAMPLE_SIZE]
    db_dir = get_spider_db_dir("dev", data_dir=DATA_DIR)

    print(f"Loading LoRA model from {CHECKPOINT_DIR}/{CHECKPOINT}...")
    model, tokenizer = load_lora_model(f"{CHECKPOINT_DIR}/{CHECKPOINT}")

    categories = Counter()
    detail_log = []

    for i, example in enumerate(examples):
        prompt = format_prompt(example["schema"], example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predicted_sql = extract_sql(raw_output)
        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"
        gold_rows = execute_sql(example["query"], db_path, TIMEOUT)

        if predicted_sql is None:
            category = "no_sql_extracted"
            predicted_rows = None
        else:
            predicted_rows = execute_sql(predicted_sql, db_path, TIMEOUT)
            if predicted_rows is None:
                category = "sqlite_execution_error"
            elif sorted(predicted_rows) == sorted(gold_rows) if gold_rows is not None else False:
                category = "correct"
            elif len(predicted_rows) == 0 and (gold_rows is None or len(gold_rows) > 0):
                category = "executes_but_empty_wrong"
            else:
                category = "executes_but_wrong_result"

        categories[category] += 1
        if category not in ("correct",):
            detail_log.append({
                "question": example["question"],
                "difficulty": example["difficulty"],
                "category": category,
                "predicted_sql": predicted_sql,
                "gold_sql": example["query"],
            })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(examples)} done")

    print("\n=== ERROR CATEGORY BREAKDOWN (LoRA model, 200 examples) ===")
    total = sum(categories.values())
    for category, count in categories.most_common():
        print(f"  {category}: {count} ({count/total*100:.1f}%)")

    fixable_by_guard = categories["sqlite_execution_error"] + categories["no_sql_extracted"]
    print(f"\nPotentially fixable by Execution Guard: {fixable_by_guard} ({fixable_by_guard/total*100:.1f}%)")
    print(f"NOT fixable by Execution Guard (wrong but valid SQL): "
          f"{categories['executes_but_wrong_result'] + categories['executes_but_empty_wrong']} "
          f"({(categories['executes_but_wrong_result'] + categories['executes_but_empty_wrong'])/total*100:.1f}%)")

    with open("error_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump({"categories": dict(categories), "details": detail_log}, f, indent=2)
    print("\nSaved to error_diagnosis.json")


if __name__ == "__main__":
    main()
