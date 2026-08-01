"""Schema Pruning - Priority 3, implementation_priority_plan.md.

Removes tables from the schema that are very unlikely to be relevant to
the question, before building the DDL prompt - targets specifically the
`extra` difficulty category, where our error analysis (16 real
sqlite_execution_error cases, technical_lessons_learned.md) found
repeated confusion between similarly-named tables in wide schemas
(car_makers/model_list/car_names/cars_data - 10 of 16 real errors on
one such schema).

Conservative, cheap heuristic (per Gemini's own revised estimate:
"реально сработает только на базах с >8 таблицами" - only worth
applying on wide schemas, and only removing tables with NO textual
overlap with the question AND no foreign-key link to a kept table):
1. Score each table by word overlap between the question and the
   table/column names.
2. Keep the top-scoring tables, but ALWAYS also keep any table
   connected via foreign key to an already-kept table (protects JOIN
   chains from being broken).
3. Only prune anything at all if the schema has more tables than a
   threshold - small schemas (Spider's common case) are left untouched.
"""
from __future__ import annotations

import re

WORD_PATTERN = re.compile(r"[a-zA-Z]+")


def _words(text: str) -> set[str]:
    """Lowercased, naively-stemmed word set from an identifier or
    question - splits on both whitespace and underscores/camelCase
    boundaries (pet_type, PetType, pettype - all seen in our own error
    analysis), and strips a trailing "s" for simple plural/singular
    matching.

    CRITICAL FIX: without stemming, "singers" (question) vs "singer"
    (table name) had ZERO word overlap - found via testing, this broke
    the single most basic, common case (questions ask about "singers",
    tables are named "singer" in the singular, as is standard DB
    convention) and would have made the whole technique nearly useless
    in practice. Naive trailing-"s" stripping isn't a real stemmer, but
    handles the overwhelmingly common English plural case cheaply.
    """
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    words = {w.lower() for w in WORD_PATTERN.findall(spaced) if len(w) > 2}
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}


def prune_schema(schema: dict, question: str, min_tables_to_prune: int = 8, keep_top_n: int = 5) -> dict:
    """Return a copy of `schema` with low-relevance tables removed.

    Args:
        schema: Original schema entry (table_names_original,
            column_names_original, column_types, primary_keys,
            foreign_keys - same format as schema_to_ddl expects).
        question: The natural-language question - used to score table
            relevance by word overlap.
        min_tables_to_prune: Schemas with this many tables or fewer are
            returned UNCHANGED - per Gemini's own revised estimate,
            pruning only helps on wide schemas; small ones aren't worth
            the risk of removing something needed.
        keep_top_n: How many top-scoring tables to keep as the "core"
            set before adding foreign-key-connected tables.

    Returns:
        A new schema dict with the same structure as the input, but
        possibly fewer tables/columns - safe to pass directly to
        schema_to_ddl. Returns the ORIGINAL schema unchanged if there
        are too few tables to bother, or if pruning would remove
        everything (safety fallback - never return an empty schema).
    """
    table_names = schema["table_names_original"]
    if len(table_names) <= min_tables_to_prune:
        return schema

    question_words = _words(question)

    def table_score(table_index: int) -> int:
        score = len(_words(table_names[table_index]) & question_words)
        for col_table_index, col_name in schema["column_names_original"]:
            if col_table_index == table_index and col_name != "*":
                score += len(_words(col_name) & question_words)
        return score

    scores = [(table_score(i), i) for i in range(len(table_names))]
    scores.sort(reverse=True)
    kept_indices = {i for _, i in scores[:keep_top_n]}

    # Expand with any table directly foreign-key-connected to a kept
    # table - protects JOIN chains (e.g. cars_data <-> car_names) from
    # being severed by pruning one side of a needed join.
    column_to_table = {ci: table_idx for ci, (table_idx, _) in enumerate(schema["column_names_original"])}
    for from_col, to_col in schema.get("foreign_keys", []):
        from_table, to_table = column_to_table[from_col], column_to_table[to_col]
        if from_table in kept_indices or to_table in kept_indices:
            kept_indices.add(from_table)
            kept_indices.add(to_table)

    if len(kept_indices) >= len(table_names):
        return schema  # nothing actually pruned - avoid pointless copy/remap

    return _rebuild_schema(schema, kept_indices)


def _rebuild_schema(schema: dict, kept_table_indices: set[int]) -> dict:
    """Build a new schema dict containing only the kept tables/columns,
    with all indices correctly remapped - table_names_original,
    column_names_original, column_types, primary_keys, foreign_keys all
    reference each other by position, so dropping tables requires
    consistently renumbering everything that refers to them."""
    old_to_new_table = {}
    new_table_names = []
    for old_index, name in enumerate(schema["table_names_original"]):
        if old_index in kept_table_indices:
            old_to_new_table[old_index] = len(new_table_names)
            new_table_names.append(name)

    old_to_new_column = {}
    new_column_names = [[-1, "*"]]
    new_column_types = [schema["column_types"][0]]
    for old_index, (old_table_index, col_name) in enumerate(schema["column_names_original"]):
        if old_index == 0:
            continue  # the "*" entry, already added above
        if old_table_index in kept_table_indices:
            old_to_new_column[old_index] = len(new_column_names)
            new_column_names.append([old_to_new_table[old_table_index], col_name])
            new_column_types.append(schema["column_types"][old_index])

    new_primary_keys = [old_to_new_column[c] for c in schema.get("primary_keys", []) if c in old_to_new_column]
    new_foreign_keys = [
        [old_to_new_column[a], old_to_new_column[b]]
        for a, b in schema.get("foreign_keys", [])
        if a in old_to_new_column and b in old_to_new_column
    ]

    return {
        "table_names_original": new_table_names,
        "column_names_original": new_column_names,
        "column_types": new_column_types,
        "primary_keys": new_primary_keys,
        "foreign_keys": new_foreign_keys,
    }
