"""Проверка correct_column_names + normalize_string_quotes именно на
16 известных sqlite_execution_error примерах из error_diagnosis.json -
не абстрактный тест, а прогон на реальных, уже найденных ошибках.

Читает error_diagnosis.json (должен быть в текущей папке, из
diagnose_errors.py), для каждого примера категории
"sqlite_execution_error" - применяет обе функции постобработки к
predicted_sql, реально выполняет исправленный запрос через sqlite3,
сравнивает с gold. Не полагается на визуальное сходство с gold -
честная execution accuracy проверка, как и весь остальной пайплайн.
"""
from __future__ import annotations

import json

from src.dataset import get_spider_db_dir, load_spider
from src.evaluator import compare_results, execute_sql
from src.sql_postprocess import correct_column_names, normalize_string_quotes

DATA_DIR = "data/spider"
TIMEOUT = 10


def main() -> None:
    with open("error_diagnosis.json", encoding="utf-8") as f:
        diagnosis = json.load(f)

    sqlite_errors = [d for d in diagnosis["details"] if d["category"] == "sqlite_execution_error"]
    print(f"Проверяем {len(sqlite_errors)} известных ошибок выполнения SQLite")

    # Нужна полная информация о примере (db_id, schema) - её нет в
    # error_diagnosis.json (там только question/sql/category) - находим
    # по вопросу в исходном датасете, тем же фиксированным сэмплом.
    examples = load_spider("dev", data_dir=DATA_DIR)[:200]
    examples_by_question = {e["question"]: e for e in examples}
    db_dir = get_spider_db_dir("dev", data_dir=DATA_DIR)

    fixed_count = 0
    still_broken = []

    for error in sqlite_errors:
        example = examples_by_question.get(error["question"])
        if example is None:
            print(f"  ПРОПУСК (не найден в датасете): {error['question'][:50]}")
            continue

        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"
        original_sql = error["predicted_sql"]

        corrected_sql = correct_column_names(original_sql, example["schema"])
        corrected_sql = normalize_string_quotes(corrected_sql, example["schema"])

        corrected_rows = execute_sql(corrected_sql, db_path, TIMEOUT)
        gold_rows = execute_sql(example["query"], db_path, TIMEOUT)

        is_fixed = compare_results(corrected_rows, gold_rows)
        if is_fixed:
            fixed_count += 1
            print(f"  ИСПРАВЛЕНО: {error['question'][:60]}")
        else:
            still_broken.append(error["question"])
            print(f"  всё ещё неверно: {error['question'][:60]}")

    print(f"\n=== ИТОГ ===")
    print(f"Исправлено: {fixed_count} / {len(sqlite_errors)}")
    print(f"Из общих 200 примеров это даёт: +{fixed_count/200*100:.1f}% к overall_accuracy")


if __name__ == "__main__":
    main()
