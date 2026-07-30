"""Execution-guided self-correction for SQL generation.

If the model's first SQL attempt fails to execute or returns an empty
result, retry once with the error/emptiness fed back into the prompt -
see branching_points_analysis.md, Ветка 2, and technical_lessons_learned.md
section 19 (literature: LitE-SQL, ZAS-SQL, SQL-PaLM all report gains
from execution-guided self-correction).

Deliberately capped at ONE retry, not unlimited - keeps inference cost
bounded and matches how the cited papers describe the technique (not a
long agentic loop).
"""
from __future__ import annotations

from typing import Any

from src.evaluator import execute_sql
from src.sanitizer import extract_sql

CORRECTION_PROMPT_TEMPLATE = (
    "Your previous SQL query failed: {error_description}\n"
    "Original query: {failed_sql}\n"
    "Please provide a corrected SQLite query for the same question, "
    "wrapped in ```sql code fences."
)


def describe_failure(sql: str | None, rows: list | None) -> str | None:
    """Classify why a SQL attempt failed, for the correction prompt.

    CHANGED after a real cheap test (75 hard/extra examples) showed
    self-correction making accuracy WORSE (0.480 -> 0.453), with ZERO
    wrong->correct flips - see technical_lessons_learned.md. The bug: an
    empty result set (`len(rows) == 0`) was treated as a failure needing
    correction, but a correct query legitimately CAN return zero rows
    (e.g. "singers older than 200" - the honest answer is nothing).
    Retrying a query that was already correct just because it happened
    to return no rows very plausibly explains the correct->wrong
    regressions observed. Now only genuine failure signals trigger a
    retry - no SQL extracted, or the query didn't execute at all.

    Returns:
        A short, human-readable description if the attempt failed, or
        `None` if it succeeded (rows is not None, regardless of whether
        it's empty) - the caller uses `None` as the signal to skip
        correction entirely.
    """
    if sql is None:
        return "no SQL query was found in your response (missing ```sql fence)"
    if rows is None:
        return "the query failed to execute (syntax error or timeout)"
    return None


def generate_with_correction(
    model: Any,
    tokenizer: Any,
    initial_prompt: str,
    db_path: str,
    timeout: int,
    max_new_tokens: int = 256,
) -> tuple[str | None, bool]:
    """Generate SQL, retry once with error feedback if the first attempt fails.

    Args:
        model, tokenizer: loaded model, as returned by `src.model` functions.
        initial_prompt: the standard prompt from `prompt_formatter.format_prompt`.
        db_path: path to the .sqlite database for execution.
        timeout: per-query execution timeout (seconds).
        max_new_tokens: generation length cap, same as the main eval scripts.

    Returns:
        `(final_sql, was_corrected)` - `final_sql` is the SQL to actually
        score (either the first attempt if it succeeded, or the retry if
        one was made), `was_corrected` is True iff a retry happened -
        useful for reporting how often correction was needed/helped.
    """
    first_sql, first_rows = _generate_and_execute(model, tokenizer, initial_prompt, db_path, timeout, max_new_tokens)
    failure = describe_failure(first_sql, first_rows)

    if failure is None:
        return first_sql, False

    correction_prompt = initial_prompt + CORRECTION_PROMPT_TEMPLATE.format(
        error_description=failure,
        failed_sql=first_sql or "(no query extracted)",
    )
    second_sql, _ = _generate_and_execute(model, tokenizer, correction_prompt, db_path, timeout, max_new_tokens)
    return second_sql, True


def _generate_and_execute(
    model: Any, tokenizer: Any, prompt: str, db_path: str, timeout: int, max_new_tokens: int
) -> tuple[str | None, list | None]:
    """One generate+extract+execute cycle - shared by the initial attempt
    and the correction retry, so the two stay in sync if generation
    parameters change."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    sql = extract_sql(raw_output)
    rows = execute_sql(sql, db_path, timeout) if sql is not None else None
    return sql, rows
