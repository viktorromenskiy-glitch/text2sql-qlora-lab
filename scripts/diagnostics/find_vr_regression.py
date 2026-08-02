"""Находит конкретный medium-пример, где Value Retrieval поверх
постобработки СЛОМАЛ уже верный ответ - overall упал с 0.720 до 0.715,
причём только в категории medium, easy/hard/extra не изменились.

Генерирует SQL дважды на каждый medium-пример: без Value Retrieval и с
ним (постобработка включена в обоих случаях, чтобы изолировать именно
эффект подсказок, не смешивать с уже подтверждённым эффектом
постобработки) - находит регрессию и печатает саму вставленную
подсказку, чтобы понять, была ли она ложной.
"""
from __future__ import annotations

from src.dataset import get_spider_db_dir, load_spider
from src.evaluator import compare_results, execute_sql
from src.model import load_lora_model
from src.prompt_formatter import CHATML_IM_END, format_prompt
from src.sanitizer import extract_sql
from src.sql_postprocess import correct_column_names, correct_table_names, normalize_string_quotes
from src.utils import set_seed, setup_logging
from src.value_retrieval import find_value_hints

DATA_DIR = "data/spider"
TIMEOUT = 10
CHECKPOINT_DIR = "/content/drive/MyDrive/text2sql-checkpoints-v2"
CHECKPOINT = "final"


def generate_and_postprocess(model, tokenizer, prompt, schema, db_path):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    sql = extract_sql(raw_output)
    if sql is not None:
        sql = correct_table_names(sql, schema["table_names_original"])
        sql = correct_column_names(sql, schema)
        sql = normalize_string_quotes(sql, schema)
    return sql


def main() -> None:
    setup_logging()
    set_seed(42)

    examples = load_spider("dev", data_dir=DATA_DIR)[:200]
    medium_examples = [e for e in examples if e["difficulty"] == "medium"]
    db_dir = get_spider_db_dir("dev", data_dir=DATA_DIR)
    print(f"Проверяем {len(medium_examples)} medium-примеров")

    model, tokenizer = load_lora_model(f"{CHECKPOINT_DIR}/{CHECKPOINT}")

    for i, example in enumerate(medium_examples):
        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"
        gold_rows = execute_sql(example["query"], db_path, TIMEOUT)

        # Без Value Retrieval
        prompt_plain = format_prompt(example["schema"], example["question"])
        sql_plain = generate_and_postprocess(model, tokenizer, prompt_plain, example["schema"], db_path)
        rows_plain = execute_sql(sql_plain, db_path, TIMEOUT) if sql_plain else None
        correct_plain = compare_results(rows_plain, gold_rows)

        # С Value Retrieval
        hints = find_value_hints(example["question"], example["schema"], db_path)
        prompt_vr = prompt_plain
        if hints:
            marker = f"{CHATML_IM_END}\n"
            insert_at = prompt_vr.rfind(marker)
            prompt_vr = prompt_vr[:insert_at] + "\n" + "\n".join(hints) + prompt_vr[insert_at:]
        sql_vr = generate_and_postprocess(model, tokenizer, prompt_vr, example["schema"], db_path)
        rows_vr = execute_sql(sql_vr, db_path, TIMEOUT) if sql_vr else None
        correct_vr = compare_results(rows_vr, gold_rows)

        if correct_plain and not correct_vr:
            print(f"\n=== РЕГРЕССИЯ найдена ===")
            print(f"Q: {example['question']}")
            print(f"Подсказки (Value Retrieval): {hints}")
            print(f"SQL без подсказок (верно):    {sql_plain}")
            print(f"SQL с подсказками (неверно):  {sql_vr}")
            print(f"Gold: {example['query']}")

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(medium_examples)} done")

    print("\nГотово")


if __name__ == "__main__":
    main()
