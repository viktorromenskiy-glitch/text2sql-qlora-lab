# RUNBOOK — Milestone 2 (QLoRA fine-tuning)

С учётом всех уроков вчерашней ночи (см. technical_lessons_learned.md,
разделы 8-14). GPU подключается ПОСЛЕДНИМ. Google Drive монтируется и
ПРОВЕРЯЕТСЯ реальной записью до начала обучения.

## Шаг 0 — новый, чистый блокнот Colab

Открыть новый блокнот. Не переиспользовать вчерашний.

## Шаг 1 — установка зависимостей (БЕЗ GPU)

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes gdown pyyaml
```

## Шаг 2 — скачать Spider (БЕЗ GPU)

```python
!gdown --id 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O spider.zip
!unzip -q spider.zip -d data
!mv data/spider_data data/spider
!ls data/spider
```

## Шаг 3 — получить код с GitHub (БЕЗ GPU)

```python
!git clone https://github.com/viktorromenskiy-glitch/text2sql-qlora-lab.git
!ls text2sql-qlora-lab
```

Если файлы окажутся не в папке `src/` (уже случалось) — переместить:
```python
%cd text2sql-qlora-lab
!mkdir -p src
!mv __init__.py dataset.py evaluator.py model.py prompt_formatter.py sanitizer.py spider_hardness.py utils.py src/ 2>&1
!ls src
%cd ..
!cp -r text2sql-qlora-lab/src /content/src
!cp -r text2sql-qlora-lab/configs /content/configs
!cp -r text2sql-qlora-lab/scripts /content/scripts
!ls /content/src /content/configs /content/scripts
```

## Шаг 4 — скачать вчерашний results_baseline.json (для сравнения)

Загрузить через кнопку (📁 → upload) файл `results_baseline.json`,
сохранённый вчера на компьютер, в `/content/`.

## Шаг 5 — смонтировать Google Drive И ПРОВЕРИТЬ запись (НОВЫЙ шаг, критичный)

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Обязательная проверка перед тем как доверять Drive чекпоинтам:**
```python
import os
test_dir = "/content/drive/MyDrive/text2sql-checkpoints"
os.makedirs(test_dir, exist_ok=True)
with open(f"{test_dir}/test.txt", "w") as f:
    f.write("test")
print(open(f"{test_dir}/test.txt").read())
!ls "{test_dir}"
```

Если это выполнилось без ошибок и показало "test" — Drive реально
работает. Если ошибка — НЕ переходить дальше, разобраться сначала.

## Шаг 6 — подключить GPU (ТОЛЬКО ТЕПЕРЬ, в самом конце)

Runtime → Change runtime type → T4 GPU → Save.

**Важно (см. урок 11)**: это стирает всё, что было в сессии. Шаги 1-5
придётся выполнить заново после этого переключения — это не опция, а
гарантированная необходимость. Альтернатива: выбрать T4 GPU СРАЗУ на
Шаге 0, до выполнения шагов 1-5, чтобы не переключать тип среды посреди
работы (это может быть быстрее в сумме, чем переключать и терять
прогресс).

## Шаг 7 — подтвердить GPU

```python
!nvidia-smi
```

## Шаг 8 — запустить обучение

```python
%cd /content
!python scripts/train.py
```

Ожидаемое время: 20-60 минут. Прогресс печатается каждые 10 шагов
(logging_steps в конфиге). Чекпоинты сохраняются в Drive каждые 50 шагов
— если сессия оборвётся, прогресс не теряется полностью.

## Шаг 9 — оценить дообученную модель

```python
!python scripts/evaluate_lora.py
```

Выведет честное сравнение baseline vs LoRA по каждому уровню сложности.

## Шаг 10 — сохранить результаты

Скачать `results_lora.json` на компьютер (📁 → download), тем же
способом, что вчера с `results_baseline.json`.
