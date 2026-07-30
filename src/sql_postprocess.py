"""Post-processing corrections for generated SQL - applied AFTER
generation, before execution/scoring. No retraining needed.

Levenshtein-based table name correction, per a real, documented gain
reported in an independent QLoRA/Spider write-up (Victor Dubus-Chanson,
Hexamind, 2024) - see branching_points_analysis.md and
technical_lessons_learned.md. Models sometimes hallucinate a plausible
but non-existent table name (e.g. "singers" instead of "singer") - this
catches and fixes that specific, narrow failure mode without touching
anything else about the query.
"""
from __future__ import annotations

import re

# Matches a table name directly after FROM or JOIN - deliberately simple
# (not a full SQL parser). Good enough for Spider/BIRD-style queries,
# which don't use exotic FROM-clause syntax (subqueries in FROM are the
# main known gap - see correct_table_names docstring).
TABLE_REFERENCE_PATTERN = re.compile(r"\b(FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def levenshtein_distance(a: str, b: str) -> int:
    """Standard dynamic-programming edit distance between two strings.

    Args:
        a, b: strings to compare (case-sensitive - caller should
            lowercase both sides first if case-insensitive matching is
            wanted, see correct_table_names).

    Returns:
        Minimum number of single-character insertions, deletions, or
        substitutions to turn `a` into `b`.
    """
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a):
        current_row = [i + 1]
        for j, char_b in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (char_a != char_b)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def correct_table_names(sql: str, real_table_names: list[str]) -> str:
    """Replace hallucinated table names in `sql` with the closest real one.

    Only touches identifiers immediately after FROM/JOIN - does not
    touch column names, string literals, or table names inside
    subqueries in other clause positions. If a referenced name already
    matches a real table exactly (case-insensitive), it's left alone.

    Known gap: doesn't handle FROM (SELECT ...) subquery syntax or table
    aliases introducing a second name for the same table - out of scope
    for this narrow, cheap fix; not expected to be common in Spider/BIRD
    gold-style queries.

    Args:
        sql: The generated SQL query (already extracted from the model's
            raw output, e.g. via `sanitizer.extract_sql`).
        real_table_names: The actual table names for this example's
            database (from `schema["table_names_original"]`).

    Returns:
        The SQL with any non-matching FROM/JOIN table names replaced by
        their closest real match. Unchanged if every referenced name
        already matches, or if `real_table_names` is empty.
    """
    if not real_table_names:
        return sql

    real_names_lower = {name.lower(): name for name in real_table_names}

    def replace_match(match: re.Match) -> str:
        keyword, referenced_name = match.group(1), match.group(2)
        if referenced_name.lower() in real_names_lower:
            return match.group(0)  # already correct (case-insensitive) - leave as-is

        closest_name = min(
            real_table_names,
            key=lambda real_name: levenshtein_distance(referenced_name.lower(), real_name.lower()),
        )
        return f"{keyword} {closest_name}"

    return TABLE_REFERENCE_PATTERN.sub(replace_match, sql)
