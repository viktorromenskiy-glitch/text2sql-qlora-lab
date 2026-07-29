# data/ — источники датасетов

Сами файлы датасетов НЕ хранятся в git (см. .gitignore) — слишком
большие. Здесь описано, откуда их получить и куда положить, чтобы код
из `src/dataset.py` нашёл их по ожидаемым путям.

## Spider

**Источник**: официальный архив проекта Yale (подтверждён напрямую
сверкой с официальной страницей https://yale-lily.github.io/spider,
раздел "Getting Started" — тот же файл, что указан там).

**Скачивание** (проверено, реально работает):
```bash
!pip install gdown -q
!gdown --id 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O spider.zip
!unzip -q spider.zip -d data
!mv data/spider_data data/spider
```

**Ожидаемая структура после распаковки** (`data/spider/`):
```
database/              # .sqlite файлы, по одному на train/dev (database/<db_id>/<db_id>.sqlite)
test_database/         # отдельные .sqlite файлы для test-сплита (другой набор баз)
tables.json            # схемы для train/dev
test_tables.json       # схемы для test (другой набор баз, не путать с tables.json)
train_spider.json      # обучающие примеры (вопрос + SQL + db_id)
dev.json               # dev-сплит
test.json              # test-сплит
dev_gold.sql, train_gold.sql, test_gold.sql, train_others.json, README.txt  # не используются кодом
```

**Важно**: "test" использует ПОЛНОСТЬЮ ДРУГОЙ набор баз данных, чем
"train"/"dev" — отсюда разделение на `tables.json`/`database/` (для
train/dev) и `test_tables.json`/`test_database/` (для test). Это уже
учтено в `src/dataset.py` (`SPIDER_SPLIT_CONFIG`) — не нужно ничего
чинить, но важно знать, если вносите изменения в код.

## BIRD-SQL

**Источник**: https://bird-bench.github.io/

**Статус**: НЕ реализовано автоматическое скачивание. На момент
написания не было подтверждённого стабильного прямого URL для
программного скачивания (в отличие от Spider). Полный датасет с
эталонными ответами частично требует отдельного запроса
разработчикам — не хардкодить URL наугад.

**Ручной шаг**: скачать `dev.json` (dev-сплит, единственный
используемый в проекте — см. `technical_assignment.md`, честная
проверка на обобщение, без дообучения на BIRD) с официального сайта,
положить в `data/bird/dev.json`.

`src/dataset.py::download_bird()` намеренно вызывает
`NotImplementedError` с этим же объяснением — не баг, осознанное
решение не гадать с URL.

## Зачем вообще два датасета

Spider — обучение и основная оценка. BIRD-SQL — независимая, более
"грязная", реалистичная проверка на обобщение (используется только на
инференсе, без дообучения на нём) — см. `technical_assignment.md`.
