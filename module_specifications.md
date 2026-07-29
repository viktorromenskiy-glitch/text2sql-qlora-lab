# Спецификация модулей — что должен делать каждый файл

Дерево каталогов само по себе недостаточно для Code — файл может лежать
в правильном месте и при этом реализовывать не ту логику, не тот
интерфейс, или не согласовываться с соседними модулями. Ниже — контракт
для каждого файла: ответственность, интерфейс, связи.

---

## `configs/`

### `train_config.yaml`
Гиперпараметры QLoRA, ничего больше — никакого кода не подразумевает.
```yaml
model_name: "Qwen/Qwen2.5-Coder-7B-Instruct"
lora:
  r: 16
  alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
quantization:
  type: nf4
  double_quant: true
training:
  learning_rate: 2e-4
  batch_size: 4
  gradient_checkpointing: true
  num_epochs: 3
checkpoint_dir: "/content/drive/MyDrive/text2sql-checkpoints"
```

### `eval_config.yaml`
Параметры валидации — не код, только значения.
```yaml
sql_timeout_seconds: 10
datasets:
  spider_test: "data/spider/test.json"
  bird_dev: "data/bird/dev.json"
report_by_difficulty: true
output_path: "results/eval_report.json"
```

---

## `src/` — импортируемые модули, без побочных эффектов при импорте

### `utils.py`
- `setup_logging() -> None` — единый формат логов для всех скриптов
- `set_seed(seed: int) -> None` — фиксирует random/numpy/torch для воспроизводимости
- `measure_vram() -> dict` — текущее использование VRAM (для метрики из ТЗ "16-bit vs 4-bit")
- `Timer` (context manager) — замер latency генерации, токенов/сек

### `prompt_formatter.py`
Единый источник правды для промптов (как в вашем описании). Один
публичный интерфейс:
- `schema_to_ddl(schema: dict) -> str` — JSON-схема Spider/BIRD → минифицированный `CREATE TABLE`
- `format_prompt(schema: dict, question: str) -> str` — оборачивает DDL + вопрос в ChatML-шаблон Qwen2.5-Coder с системной инструкцией из ТЗ
- Используется в `dataset.py` (обучение), `scripts/evaluate.py` (инференс), `app.py` (демо) — **не дублировать логику форматирования нигде больше**

### `dataset.py`
- `load_spider(split: str) -> list[dict]`, `load_bird(split: str) -> list[dict]` — читают сырые файлы датасетов
- `build_training_examples(raw_data, tokenizer) -> Dataset` — применяет `prompt_formatter`, токенизирует, возвращает объект, готовый для `SFTTrainer`

### `model.py`
- `load_base_model() -> (model, tokenizer)` — Qwen2.5-Coder-7B-Instruct через Unsloth, без LoRA (для zero-shot baseline)
- `load_lora_model(checkpoint_path: str) -> (model, tokenizer)` — базовая модель + обученный LoRA-адаптер
- `apply_lora_config(model, config: dict)` — накладывает параметры из `train_config.yaml`

### `evaluator.py`
- `execute_sql(sql: str, db_path: str, timeout: int) -> list | None` — выполняет запрос, `None` при ошибке/таймауте
- `compare_results(predicted: list, gold: list) -> bool` — сравнение **с сортировкой** перед сравнением (см. техническую спецификацию)
- `evaluate_example(predicted_sql, gold_sql, db_path, difficulty) -> dict` — один пример → результат + метка сложности
- `aggregate_results(results: list[dict]) -> dict` — Execution Accuracy в целом и **по каждому уровню сложности отдельно**

### `sanitizer.py`
- `extract_sql(raw_model_output: str) -> str | None` — regex-извлечение из ```sql-обёртки; `None`, если не найдено (это осознанный, а не случайный результат — см. тех.спецификацию)

---

## `scripts/` — точки входа, только оркестрация, не бизнес-логика

### `train.py`
Читает `train_config.yaml` → вызывает `dataset.py` + `model.py` → запускает `SFTTrainer` → чекпоинты в Google Drive (путь из конфига) → логирует прогресс через `utils.setup_logging()`.

### `evaluate.py`
Читает `eval_config.yaml` → прогоняет **и** `load_base_model()`, **и** `load_lora_model()` через `evaluator.py` на Spider test **и** BIRD-SQL dev → сохраняет сравнительный отчёт (Веха 2 и Веха 3 из плана подготовки).

**⚠️ РАСХОЖДЕНИЕ С РЕАЛИЗАЦИЕЙ (зафиксировано после Вехи 2)**: по факту
вместо единого `evaluate.py` + `eval_config.yaml` созданы два отдельных
скрипта — `evaluate_baseline.py` и `evaluate_lora.py`, оба жёстко
заточены под Spider dev, без конфиг-файла (пути захардкожены). BIRD-SQL
не задействован вообще. `eval_config.yaml` не существует.

**Решение нужно принять перед Вехой 3** (первое реальное использование
BIRD-SQL): либо (а) переписать в единый `evaluate.py`+`eval_config.yaml`
по исходной спецификации, либо (б) осознанно принять текущий паттерн
(отдельные скрипты) и обновить эту спецификацию под факт, добавив
`evaluate_bird.py` по той же схеме. Не решать по инерции — обсудить
явно в начале Вехи 3.

---

## `tests/`

### `test_prompt_formatter.py`
Фиксированный тестовый JSON-схемы на входе → assert точного совпадения минифицированного DDL со строкой-эталоном.

### `test_sanitizer.py`
Набор образцов сырого вывода модели (корректная обёртка ```sql, обёртка с лишним текстом до/после, полное отсутствие обёртки) → assert правильного извлечения или правильного `None`.

---

## `app.py`
Gradio: выпадающий список БД (Spider/BIRD) → поле вопроса → кнопка Generate → side-by-side: (base model SQL + результат выполнения) vs (LoRA model SQL + результат выполнения) + итоговая таблица. Использует `prompt_formatter.py`, `model.py`, `evaluator.py`, `sanitizer.py` — не дублирует их логику.

## `notebooks/`
- `01_eda_spider_bird.ipynb` — только исследование (размер, распределение по сложности, примеры) — не часть продакшен-пайплайна
- `02_demo_inference_test.ipynb` — ручная проверка на нескольких вопросах перед сборкой `app.py`

## `RUNBOOK.md`
Пошаговые команды именно для Colab (монтирование Drive, установка зависимостей, запуск `scripts/train.py` → `scripts/evaluate.py` → `app.py`) — по аналогии с прошлым проектом, но другой набор команд (не PowerShell).

## `.github/workflows/`
CI: на каждый push — `ruff check` (или `flake8`) + `pytest tests/`.
