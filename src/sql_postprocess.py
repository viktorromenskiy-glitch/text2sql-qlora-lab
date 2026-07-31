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


# Real, documented failure mode found via diagnose_errors.py on our own
# LoRA model (16 sqlite_execution_error cases inspected by hand - see
# technical_lessons_learned.md): the model generates plausible-looking
# but wrong-case/wrong-format column names (PetAge vs pet_age, StuID vs
# stuid) - a DIFFERENT, separate failure mode from table-name
# hallucination. Not every SQL keyword/alias should be treated as a
# candidate column - this list is deliberately conservative (Spider/BIRD
# gold queries don't use exotic SQL features).
SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "JOIN", "ON", "AND", "OR", "NOT", "AS",
    "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "DISTINCT", "ASC", "DESC",
    "IN", "LIKE", "IS", "NULL", "BETWEEN", "EXISTS", "UNION", "EXCEPT",
    "INTERSECT", "INNER", "LEFT", "RIGHT", "OUTER", "COUNT", "AVG",
    "MIN", "MAX", "SUM", "CASE", "WHEN", "THEN", "ELSE", "END", "ALL",
}
# Table aliases (T1, T2, ...) are extremely common in Spider/BIRD gold
# SQL - must not be "corrected" into a column name by accident.
ALIAS_PATTERN = re.compile(r"^T\d+$", re.IGNORECASE)
# CRITICAL FIX (found via find_postprocessing_regressions.py on a real
# run: 'USA' was being "corrected" to 'uid', 'UAL' to 'uid' - 6 real
# regressions on previously-correct examples): the identifier pattern
# alone can't distinguish a bare identifier from the CONTENTS of a
# single-quoted string literal ('USA' contains the letters U-S-A, which
# look like a candidate identifier if matched in isolation). Fixed by
# matching quoted strings as a whole (and leaving them untouched)
# alongside identifiers, in one pass, so string contents are never
# individually examined - see technical_lessons_learned.md.
IDENTIFIER_OR_STRING_PATTERN = re.compile(r"'[^']*'|\b[A-Za-z_][A-Za-z0-9_]*\b")


def correct_column_names(sql: str, schema: dict) -> str:
    """Replace hallucinated/miscased column names with the closest real one.

    Two-pass, conservative approach:
    1. Exact case-insensitive match (e.g. "StuID" -> "stuid") - safe,
       handles most cases observed in practice (see diagnose_errors.py
       findings, technical_lessons_learned.md).
    2. Levenshtein fallback ONLY for identifiers within a small edit
       distance of a real column (e.g. "PetAge" -> "pet_age", distance 1)
       - deliberately conservative threshold to avoid corrupting
       legitimate SQL that just happens to share a similar-looking word.

    Does NOT touch: SQL keywords, table names (see correct_table_names),
    table aliases (T1, T2, ...), identifiers already exactly correct, or
    the CONTENTS of single-quoted string literals (see
    IDENTIFIER_OR_STRING_PATTERN - a real, found-in-production bug where
    'USA' was corrupted to 'uid').

    Args:
        sql: The generated SQL query.
        schema: This example's schema entry (needs
            `column_names_original`, same format as `schema_to_ddl`).

    Returns:
        The SQL with column-name identifiers corrected where a confident
        match was found. Unchanged if no correction was needed/possible.
    """
    real_column_names = [name for _, name in schema["column_names_original"] if name != "*"]
    if not real_column_names:
        return sql

    real_names_lower = {name.lower(): name for name in real_column_names}
    table_names_lower = {name.lower() for name in schema.get("table_names_original", [])}

    def replace_identifier(match: re.Match) -> str:
        identifier = match.group(0)

        if identifier.startswith("'"):
            return identifier  # matched a whole string literal - leave untouched entirely

        identifier_lower = identifier.lower()

        if identifier.upper() in SQL_KEYWORDS or ALIAS_PATTERN.match(identifier):
            return identifier
        if identifier_lower in table_names_lower:
            return identifier  # let correct_table_names handle these separately
        if identifier_lower in real_names_lower:
            return real_names_lower[identifier_lower]  # exact case-insensitive fix

        closest_name = min(real_column_names, key=lambda c: levenshtein_distance(identifier_lower, c.lower()))
        distance = levenshtein_distance(identifier_lower, closest_name.lower())
        # Conservative threshold: only correct if genuinely close (e.g.
        # "PetAge"->"pet_age" is distance 1) - not confident guesses.
        if distance <= 2 and distance < len(identifier_lower):
            return closest_name
        return identifier  # not confident enough - leave untouched

    return IDENTIFIER_OR_STRING_PATTERN.sub(replace_identifier, sql)


DOUBLE_QUOTED_STRING = re.compile(r'"([^"]*)"')


def normalize_string_quotes(sql: str, schema: dict) -> str:
    """Convert double-quoted VALUE literals to single quotes.

    Real, observed failure mode (diagnose_errors.py): the model sometimes
    writes `WHERE pet_type = "cat"` - in SQLite, double quotes denote an
    IDENTIFIER, not a string literal, so this raises "no such column:
    cat" at execution time, not a logic error - a purely mechanical fix.

    Only converts quotes whose content does NOT match a real table/column
    name (case-insensitive) - genuine double-quoted identifiers (rare,
    but valid SQL) are left untouched, since those aren't the bug.

    Args:
        sql: The generated SQL query.
        schema: This example's schema entry - used to distinguish
            legitimate double-quoted identifiers from mistaken value
            literals.

    Returns:
        The SQL with likely-mistaken double-quoted values converted to
        single quotes. Unchanged if no such case is found.
    """
    known_identifiers = {name.lower() for _, name in schema["column_names_original"] if name != "*"}
    known_identifiers |= {name.lower() for name in schema.get("table_names_original", [])}

    def replace_quotes(match: re.Match) -> str:
        content = match.group(1)
        if content.lower() in known_identifiers:
            return match.group(0)  # genuine identifier reference - leave as-is
        return f"'{content}'"

    return DOUBLE_QUOTED_STRING.sub(replace_quotes, sql)
