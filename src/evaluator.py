"""Execution-based comparison of predicted vs. gold SQL queries.

Implements Execution Accuracy (EX): run both the predicted and reference
SQL against the same SQLite database, compare the resulting row sets
(sorted, since SQL without ORDER BY doesn't guarantee row order — see
technical_assignment.md and precoding_preparation_plan.md, Этап 0).
"""

from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict


def execute_sql(sql: str, db_path: str, timeout: int) -> list | None:
    """Execute a SQL query against a SQLite database.

    Args:
        sql: The SQL query to run.
        db_path: Path to the .sqlite database file.
        timeout: Max seconds to allow the query to run before it is
            interrupted — protects against accidental heavy/hanging
            queries (see technical_assignment.md validation pipeline).

    Returns:
        The list of result rows (as tuples), or `None` if the query is
        invalid, raises, or times out. Callers never see an exception —
        `None` is an honest "this query did not produce a usable result"
        outcome, not a script crash.
    """
    connection = sqlite3.connect(db_path)
    # sqlite3 has no native query timeout; interrupting from a background
    # thread after `timeout` seconds is the standard workaround.
    timer = threading.Timer(timeout, connection.interrupt)
    try:
        timer.start()
        rows = connection.execute(sql).fetchall()
        return rows
    except sqlite3.Error:
        return None
    finally:
        timer.cancel()
        connection.close()


def compare_results(predicted: list | None, gold: list | None) -> bool:
    """Compare two result sets for Execution Accuracy.

    Rows are sorted before comparison: SQL without ORDER BY does not
    guarantee row order, so comparing unsorted lists would falsely mark a
    semantically-correct query as wrong.

    Args:
        predicted: Rows returned by the predicted SQL, or `None` if it
            failed to execute.
        gold: Rows returned by the reference SQL.

    Returns:
        True if the sorted row sets match exactly.
    """
    if predicted is None or gold is None:
        return False
    return sorted(predicted) == sorted(gold)


def evaluate_example(
    predicted_sql: str | None,
    gold_sql: str,
    db_path: str,
    difficulty: str,
    timeout: int = 10,
) -> dict:
    """Evaluate one predicted SQL query against its gold reference.

    Args:
        predicted_sql: The SQL extracted from the model's raw output, or
            `None` if `sanitizer.extract_sql` found no fenced block. A
            `None` here is a format failure, tracked separately from
            wrong-logic failures (see technical_assignment.md metrics).
        gold_sql: The reference SQL query.
        db_path: Path to the SQLite database for this example.
        difficulty: Spider/BIRD difficulty label (easy/medium/hard/extra hard).
        timeout: Max seconds per query execution.

    Returns:
        A dict with `correct` (bool), `difficulty` (str), and
        `parse_failure` (bool) — `parse_failure` is True exactly when
        `predicted_sql` was `None`.
    """
    if predicted_sql is None:
        return {"correct": False, "difficulty": difficulty, "parse_failure": True}

    predicted_rows = execute_sql(predicted_sql, db_path, timeout)
    gold_rows = execute_sql(gold_sql, db_path, timeout)
    correct = compare_results(predicted_rows, gold_rows)
    return {"correct": correct, "difficulty": difficulty, "parse_failure": False}


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate per-example results into Execution Accuracy metrics.

    Args:
        results: List of dicts as returned by `evaluate_example`.

    Returns:
        A dict with `overall_accuracy`, `accuracy_by_difficulty` (dict
        keyed by difficulty label), and `parse_failure_rate`. Difficulty
        and parse-failure are reported separately from the overall figure
        so a single aggregate number can't hide that the model handles
        easy questions well but fails hard ones, or that failures are a
        format problem rather than a logic problem (same principle as the
        "Сбой сервиса" case in the Customer Support Agent project).
    """
    if not results:
        return {
            "overall_accuracy": 0.0,
            "accuracy_by_difficulty": {},
            "parse_failure_rate": 0.0,
        }

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    parse_failure_count = sum(1 for r in results if r["parse_failure"])

    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_difficulty[r["difficulty"]].append(r)

    accuracy_by_difficulty = {
        difficulty: sum(1 for r in group if r["correct"]) / len(group)
        for difficulty, group in by_difficulty.items()
    }

    return {
        "overall_accuracy": correct_count / total,
        "accuracy_by_difficulty": accuracy_by_difficulty,
        "parse_failure_rate": parse_failure_count / total,
    }
