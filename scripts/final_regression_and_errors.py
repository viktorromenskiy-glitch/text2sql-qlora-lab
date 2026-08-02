"""Регрессионный анализ (Вариант A) + сбор данных для Error Analysis
(Вариант E.1) — на нашем лучшем чекпоинте (3 эпохи) и лучшей комбинации
техник (postprocess + value_retrieval + sample_values, БЕЗ few-shot).

Один проход генерации на пример (экономим GPU) - для каждого:
1. Строим промпт СРАЗУ с sample_values + value_retrieval (как в
   финальной, лучшей конфигурации - 0.785)
2. Генерируем SQL один раз
3. Сравниваем: результат ДО постобработки (сырой SQL) vs ПОСЛЕ
   (correct_table_names + correct_column_names + normalize_string_quotes)
   - находим регрессии (было верно -> стало неверно) именно от
   постобработки, изолированно от остальных техник
4. Для всех финально неверных ответов - сохраняем детали для
   последующей ручной категоризации ошибок (Error Analysis)
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
    """Собирает промпт точно как в лучшей, финальной конфигурации:
    sample_values встроен в схему + value_retrieval подсказки поверх."""
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

    print(f"\n=== РЕГРЕССИИ от постобработки (было верно -> стало неверно): {len(regressions)} ===")
    for r in regressions:
        print(f"\nQ: {r['question']}")
        print(f"До:    {r['raw_sql']}")
        print(f"После: {r['corrected_sql']}")

    print(f"\n=== УЛУЧШЕНИЯ от постобработки: {len(improvements)} ===")
    for imp in improvements:
        print(f"  [{imp['difficulty']}] {imp['question'][:60]}")

    print(f"\n=== ВСЕГО неверных финальных ответов (для Error Analysis): {len(all_wrong_final)} ===")

    with open("final_regression_and_errors.json", "w", encoding="utf-8") as f:
        json.dump({
            "regressions": regressions,
            "improvements": improvements,
            "all_wrong_final": all_wrong_final,
        }, f, indent=2)
    print("\nSaved to final_regression_and_errors.json")


if __name__ == "__main__":
    main()
