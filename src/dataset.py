"""Loading and training-example assembly for Spider and BIRD-SQL.

Design decisions worth flagging explicitly (see Веха 1 discussion):

- Spider ships each example with its SQL pre-parsed (`sql` field - see
  taoyds/spider README, "Question, SQL, and Parsed SQL"), so `load_spider`
  computes an official-methodology `difficulty` label per example via
  `spider_hardness.eval_hardness` at load time, rather than deferring that
  to the evaluation step.
- `load_spider` also loads `tables.json` and attaches the matching schema
  to every example (keyed by `db_id`), because `module_specifications.md`
  gives `build_training_examples(raw_data, tokenizer)` only two arguments -
  the schema has to already be present on each example for
  `prompt_formatter.format_prompt` to be callable from there.
- BIRD-SQL ships its own `difficulty` label directly on each example
  (three tiers: simple/moderate/challenging - not the same four-tier
  scale as Spider's easy/medium/hard/extra), so `load_bird` uses that
  field as-is rather than re-deriving it. BIRD does not ship pre-parsed
  SQL the way Spider does, so `spider_hardness.eval_hardness` does not
  apply to it. Whether BIRD ships a `tables.json`-equivalent schema file
  the same way Spider does has not been verified against real downloaded
  BIRD files yet - `load_bird` currently does NOT attach a schema, and
  that needs to be confirmed before Milestone 3 (BIRD generalization
  check), not before Milestone 1 (Spider baseline only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.prompt_formatter import CHATML_IM_END, format_prompt
from src.spider_hardness import eval_hardness

# Confirmed against a real download of the official archive
# (https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view):
# `test.json` uses a DIFFERENT set of databases than `train_spider.json`/
# `dev.json` - its schemas live in `test_tables.json`, not `tables.json`,
# and its .sqlite files live in `test_database/`, not `database/`. Using
# `tables.json` for the "test" split (the original assumption here) would
# raise KeyError on every example, since none of test.json's db_ids are
# present in tables.json.
SPIDER_SPLIT_CONFIG = {
    "train": {"examples_file": "train_spider.json", "tables_file": "tables.json", "db_dir": "database"},
    "dev": {"examples_file": "dev.json", "tables_file": "tables.json", "db_dir": "database"},
    "test": {"examples_file": "test.json", "tables_file": "test_tables.json", "db_dir": "test_database"},
}


def load_spider(split: str, data_dir: str = "data/spider") -> list[dict]:
    """Load a Spider split and attach schema + difficulty to each example.

    Args:
        split: One of "train", "dev", or "test" (see `SPIDER_SPLIT_CONFIG`).
            Note "test" uses a disjoint set of databases from
            "train"/"dev" - see module-level comment above
            `SPIDER_SPLIT_CONFIG`.
        data_dir: Directory containing the tables file(s) and split file
            for this split, per `data/README.md`'s documented layout.

    Returns:
        A list of example dicts. Each dict is the original Spider entry
        (`db_id`, `question`, `query`, `sql`, ...) plus two added keys:
        `schema` (this example's entry from the split's tables file,
        ready for `prompt_formatter.schema_to_ddl`) and `difficulty`
        (easy/medium/hard/extra, from `spider_hardness.eval_hardness`).

    Raises:
        ValueError: If `split` is not a recognized Spider split.
    """
    if split not in SPIDER_SPLIT_CONFIG:
        raise ValueError(f"Unknown Spider split '{split}', expected one of {list(SPIDER_SPLIT_CONFIG)}")

    config = SPIDER_SPLIT_CONFIG[split]
    data_path = Path(data_dir)
    with open(data_path / config["tables_file"], encoding="utf-8") as f:
        tables = json.load(f)
    schema_by_db_id = {table["db_id"]: table for table in tables}

    with open(data_path / config["examples_file"], encoding="utf-8") as f:
        examples = json.load(f)

    for example in examples:
        example["schema"] = schema_by_db_id[example["db_id"]]
        example["difficulty"] = eval_hardness(example["sql"])

    return examples


def get_spider_db_dir(split: str, data_dir: str = "data/spider") -> str:
    """Return the directory containing .sqlite files for a Spider split.

    Args:
        split: One of "train", "dev", or "test" (see `SPIDER_SPLIT_CONFIG`).
        data_dir: The same `data_dir` passed to `load_spider`.

    Returns:
        Path to the split's database directory (e.g. `data/spider/database`
        for "train"/"dev", `data/spider/test_database` for "test").
    """
    if split not in SPIDER_SPLIT_CONFIG:
        raise ValueError(f"Unknown Spider split '{split}', expected one of {list(SPIDER_SPLIT_CONFIG)}")
    return str(Path(data_dir) / SPIDER_SPLIT_CONFIG[split]["db_dir"])


def load_bird(split: str = "dev", data_dir: str = "data/bird") -> list[dict]:
    """Load a BIRD-SQL split, attaching schema per example - same shape
    contract as `load_spider` (adds a `schema` key per example, ready for
    `prompt_formatter.schema_to_ddl`).

    IMPORTANT - schema format is UNVERIFIED against a real download.
    Many text-to-SQL codebases process Spider and BIRD with shared
    tooling, which suggests `dev_tables.json` likely uses the same field
    names as Spider's `tables.json` (`table_names_original`,
    `column_names_original`, `column_types`, `primary_keys`,
    `foreign_keys`) - but this was not confirmed by inspecting a real
    BIRD file (see data/README.md, technical_lessons_learned.md section
    19: don't state unconfirmed things as fact). `_validate_bird_schema`
    below checks for these exact keys and raises a clear, actionable
    error immediately if they're missing, rather than silently producing
    wrong DDL - fix the mapping in one place if the real format differs.

    Args:
        split: Only "dev" is currently supported - BIRD-SQL's dev split is
            the only one this project uses (held-out generalization check,
            Milestone 3, never trained on - see technical_assignment.md).
        data_dir: Directory containing `dev.json` and `dev_tables.json`,
            per `data/README.md`.

    Returns:
        List of BIRD-SQL example dicts (`db_id`, `question`, `SQL`,
        `evidence`, `difficulty`, ...) plus an added `schema` key per
        example.

    Raises:
        ValueError: If `split` is not "dev".
        KeyError: If `dev_tables.json`'s actual format doesn't match the
            Spider-style keys assumed above - see error message for which
            key is missing.
    """
    if split != "dev":
        raise ValueError(f"Unknown BIRD-SQL split '{split}', only 'dev' is supported")

    data_path = Path(data_dir)
    with open(data_path / "dev_tables.json", encoding="utf-8") as f:
        tables = json.load(f)
    for table in tables:
        _validate_bird_schema(table)
    schema_by_db_id = {table["db_id"]: table for table in tables}

    with open(data_path / "dev.json", encoding="utf-8") as f:
        examples = json.load(f)

    for example in examples:
        example["schema"] = schema_by_db_id[example["db_id"]]

    return examples


def _validate_bird_schema(table: dict) -> None:
    """Fail loudly and specifically if dev_tables.json doesn't match the
    Spider-style format assumed by load_bird() - see its docstring."""
    required_keys = ["db_id", "table_names_original", "column_names_original", "column_types", "primary_keys", "foreign_keys"]
    missing = [k for k in required_keys if k not in table]
    if missing:
        raise KeyError(
            f"dev_tables.json is missing expected key(s) {missing} - "
            f"BIRD's schema format may differ from Spider's assumed "
            f"format (see load_bird docstring). Inspect a real entry "
            f"and update this function's field mapping accordingly, "
            f"don't guess further."
        )


def get_bird_db_dir(data_dir: str = "data/bird") -> str:
    """Return the directory containing .sqlite files for BIRD dev.

    Mirrors `get_spider_db_dir` - see `data/README.md` for the expected
    layout after extracting `dev_databases.zip`.
    """
    return str(Path(data_dir) / "dev_databases")


def download_spider(data_dir: str = "data/spider") -> None:
    """Download Spider's question/SQL annotations (not database files).

    Fetches `train`/`validation` splits and `tables.json`-equivalent
    schema info from the `xlangai/spider` Hugging Face dataset (per
    technical_assignment.md) and writes them to `data_dir` in the layout
    `load_spider` expects.

    IMPORTANT - not yet a complete Step 4 of RUNBOOK.md: the HF Datasets
    mirror of Spider provides the JSON annotations only. The actual
    `database/<db_id>/<db_id>.sqlite` files that `evaluator.execute_sql`
    needs for Milestone 1 are distributed separately by the original
    Spider release (see https://yale-lily.github.io/spider) and are NOT
    fetched by this function - downloading those still needs a manual
    step or a separate, verified download path before evaluate.py can
    actually run. Flagging this now rather than guessing a URL blind.

    Args:
        data_dir: Destination directory, per `data/README.md`'s layout.
    """
    from datasets import load_dataset  # local import: heavy, only needed here

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_split = load_dataset("xlangai/spider", split="train")
    test_split = load_dataset("xlangai/spider", split="validation")

    with open(data_path / "train_spider.json", "w", encoding="utf-8") as f:
        json.dump(train_split.to_list(), f)
    with open(data_path / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_split.to_list(), f)


def download_bird(data_dir: str = "data/bird") -> None:
    """Not implemented - BIRD-SQL has no confirmed stable programmatic download.

    Unlike Spider, BIRD-SQL's official distribution (bird-bench.github.io)
    was not confirmed to have a stable direct-download URL suitable for
    unattended scripting when this was written, and the full dataset with
    ground truth reportedly requires emailing the maintainers for some
    fields. Rather than hardcode a URL that might be wrong or break,
    this is left as an explicit manual step - see `data/README.md` for
    where to get it. Only relevant for Milestone 3 (BIRD generalization
    check), not Milestone 1.

    Raises:
        NotImplementedError: Always - see docstring.
    """
    raise NotImplementedError(
        "BIRD-SQL download is a manual step - see data/README.md "
        "(https://bird-bench.github.io/). Not needed until Milestone 3."
    )


def build_training_examples(raw_data: list[dict], tokenizer: Any) -> Any:
    """Turn loaded Spider examples into a tokenized dataset for SFTTrainer.

    Args:
        raw_data: Output of `load_spider` - each example must already have
            a `schema` key (see `load_spider`).
        tokenizer: A HuggingFace tokenizer (e.g. from
            `model.load_base_model`), called on each full prompt+target
            string. Only `tokenizer(text)` returning a dict with
            `input_ids` is required, so this also works with any
            tokenizer-like stub for local testing.

    Returns:
        A `datasets.Dataset` with `text`, `input_ids`, and
        `attention_mask` columns, ready for `SFTTrainer`. The target
        appended after the prompt is the gold SQL wrapped in the same
        ```sql fence the model is instructed to produce (see
        `prompt_formatter.SYSTEM_PROMPT`), so training teaches the exact
        output format `sanitizer.extract_sql` expects at inference time.
    """
    from datasets import Dataset  # local import: heavy, only needed here

    texts = []
    for example in raw_data:
        prompt = format_prompt(example["schema"], example["question"])
        target = f"```sql\n{example['query']}\n```{CHATML_IM_END}"
        texts.append(prompt + target)

    encodings = tokenizer(texts)
    return Dataset.from_dict(
        {
            "text": texts,
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
    )
