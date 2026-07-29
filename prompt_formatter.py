"""Single source of truth for prompt construction.

Used by dataset.py (training), scripts/evaluate.py (inference) and app.py
(demo) so that the zero-shot baseline and the LoRA-tuned model are always
compared on exactly the same prompt — see technical_assignment.md.
"""

from __future__ import annotations

# Spider/BIRD `tables.json` uses a small, fixed set of column type labels.
# SQLite itself is dynamically typed, but the schema is rendered as a
# CREATE TABLE so the model sees familiar, conventional SQL types rather
# than the dataset's internal type vocabulary.
COLUMN_TYPE_TO_SQL: dict[str, str] = {
    "number": "INT",
    "text": "TEXT",
    "time": "TEXT",
    "boolean": "BOOLEAN",
    "others": "TEXT",
}
DEFAULT_SQL_TYPE = "TEXT"

# The "*" column that Spider/BIRD prepend to every schema (table index -1)
# represents "all columns" for SQL parsing purposes and is not a real column.
WILDCARD_TABLE_INDEX = -1

SYSTEM_PROMPT = (
    "Given the database schema, write a single SQLite query that answers "
    "the question. Respond with ONLY the SQL query wrapped in ```sql code "
    "fences, no explanation."
)

# ChatML special tokens used by Qwen2.5-Coder-Instruct.
CHATML_IM_START = "<|im_start|>"
CHATML_IM_END = "<|im_end|>"


def schema_to_ddl(schema: dict) -> str:
    """Convert a Spider/BIRD `tables.json` schema entry to minified DDL.

    Args:
        schema: One entry from `tables.json`, with keys
            `table_names_original`, `column_names_original`,
            `column_types`, `primary_keys`, `foreign_keys` (the raw Spider/
            BIRD schema format — see data_prep notes in module_specifications.md).

    Returns:
        One `CREATE TABLE ...` statement per table, joined by newlines.
        No comments, no example values, no extra whitespace — kept compact
        to reduce prompt length (BIRD-SQL schemas can be long enough to
        risk OOM on a T4, see technical_assignment.md).
    """
    table_names = schema["table_names_original"]
    column_entries = schema["column_names_original"]
    column_types = schema["column_types"]
    primary_keys = schema["primary_keys"]
    foreign_keys = schema["foreign_keys"]

    primary_key_columns = _flatten_primary_keys(primary_keys)

    statements = []
    for table_index, table_name in enumerate(table_names):
        columns_sql = _build_columns_sql(
            table_index, column_entries, column_types, primary_key_columns
        )
        composite_pk_sql = _build_composite_primary_key_sql(
            table_index, column_entries, primary_keys
        )
        foreign_keys_sql = _build_foreign_keys_sql(
            table_index, column_entries, table_names, foreign_keys
        )

        clauses = columns_sql + composite_pk_sql + foreign_keys_sql
        statements.append(f"CREATE TABLE {table_name} ({', '.join(clauses)})")

    return "\n".join(statements)


def format_prompt(schema: dict, question: str) -> str:
    """Wrap a DDL schema and a question in the Qwen2.5-Coder ChatML template.

    Args:
        schema: Spider/BIRD `tables.json` schema entry (see `schema_to_ddl`).
        question: Natural language question to answer with SQL.

    Returns:
        A ChatML-formatted prompt string (system + user turns, ending with
        an open assistant turn) ready to feed to the tokenizer/model.
    """
    ddl = schema_to_ddl(schema)
    user_content = f"Schema:\n{ddl}\n\nQuestion: {question}"

    return (
        f"{CHATML_IM_START}system\n{SYSTEM_PROMPT}{CHATML_IM_END}\n"
        f"{CHATML_IM_START}user\n{user_content}{CHATML_IM_END}\n"
        f"{CHATML_IM_START}assistant\n"
    )


def _flatten_primary_keys(primary_keys: list) -> set[int]:
    """Return the set of column indices that are single-column primary keys.

    Spider's `primary_keys` mixes plain ints (single-column PK) with lists
    of ints (composite PK) in the same list — composite keys are handled
    separately via `_build_composite_primary_key_sql` because they render
    as a trailing `PRIMARY KEY (a, b)` clause, not an inline column modifier.
    """
    return {pk for pk in primary_keys if isinstance(pk, int)}


def _build_columns_sql(
    table_index: int,
    column_entries: list,
    column_types: list[str],
    primary_key_columns: set[int],
) -> list[str]:
    """Render `name TYPE [PRIMARY KEY]` for every column of one table."""
    columns_sql = []
    for column_index, (owner_table_index, column_name) in enumerate(column_entries):
        if owner_table_index != table_index:
            continue
        sql_type = COLUMN_TYPE_TO_SQL.get(
            column_types[column_index], DEFAULT_SQL_TYPE
        )
        column_sql = f"{column_name} {sql_type}"
        if column_index in primary_key_columns:
            column_sql += " PRIMARY KEY"
        columns_sql.append(column_sql)
    return columns_sql


def _build_composite_primary_key_sql(
    table_index: int, column_entries: list, primary_keys: list
) -> list[str]:
    """Render a trailing `PRIMARY KEY (a, b)` clause for composite keys."""
    clauses = []
    for pk in primary_keys:
        if not isinstance(pk, list):
            continue
        pk_column_names = [
            column_entries[col_index][1]
            for col_index in pk
            if column_entries[col_index][0] == table_index
        ]
        if pk_column_names:
            clauses.append(f"PRIMARY KEY ({', '.join(pk_column_names)})")
    return clauses


def _build_foreign_keys_sql(
    table_index: int,
    column_entries: list,
    table_names: list[str],
    foreign_keys: list,
) -> list[str]:
    """Render `FOREIGN KEY (col) REFERENCES table(col)` for one table."""
    clauses = []
    for from_column_index, to_column_index in foreign_keys:
        from_table_index, from_column_name = column_entries[from_column_index]
        if from_table_index != table_index:
            continue
        to_table_index, to_column_name = column_entries[to_column_index]
        to_table_name = table_names[to_table_index]
        clauses.append(
            f"FOREIGN KEY ({from_column_name}) "
            f"REFERENCES {to_table_name}({to_column_name})"
        )
    return clauses
