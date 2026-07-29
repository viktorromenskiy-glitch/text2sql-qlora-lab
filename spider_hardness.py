"""Spider/BIRD SQL difficulty (hardness) classification.

Vendored, trimmed port of `Evaluator.eval_hardness` and its helper
functions from `evaluation.py` in taoyds/test-suite-sql-eval (Apache
License 2.0, Copyright 2024 XLANG NLP Lab):
https://github.com/taoyds/test-suite-sql-eval/blob/master/evaluation.py

Why vendored instead of hand-written from scratch: `technical_assignment.md`
requires the official evaluation methodology, not a homemade metric, so
that our per-difficulty Execution Accuracy numbers are directly comparable
to published Spider/BIRD benchmark results (see Веха 1 discussion in chat).
Why trimmed instead of importing the package whole: the original
`evaluation.py` imports `exec_eval` (execution-accuracy-with-value-matching,
which we don't use - our own `evaluator.py` implements execution accuracy
directly) and depends on `process_sql.py`, which in turn requires `nltk`.
None of that is needed here: Spider/BIRD `train.json`/`dev.json`/`test.json`
already ship each example's SQL pre-parsed in the `sql` field (see
taoyds/spider README, "Question, SQL, and Parsed SQL"), so hardness can be
computed directly from `example["sql"]` without re-parsing anything.

Only the pure, dependency-free structural analysis is kept below.
"""

from __future__ import annotations

AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")


def _has_agg(unit: tuple) -> bool:
    return unit[0] != AGG_OPS.index("none")


def _count_agg(units: list) -> int:
    return len([unit for unit in units if _has_agg(unit)])


def _get_nested_sql(sql: dict) -> list[dict]:
    """Collect nested sub-queries appearing in WHERE/HAVING/JOIN conditions
    and in INTERSECT/EXCEPT/UNION clauses."""
    nested = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
        if isinstance(cond_unit[4], dict):
            nested.append(cond_unit[4])
    if sql["intersect"] is not None:
        nested.append(sql["intersect"])
    if sql["except"] is not None:
        nested.append(sql["except"])
    if sql["union"] is not None:
        nested.append(sql["union"])
    return nested


def _count_component1(sql: dict) -> int:
    """Count WHERE/GROUP BY/ORDER BY/LIMIT/JOIN/OR/LIKE occurrences."""
    count = 0
    if len(sql["where"]) > 0:
        count += 1
    if len(sql["groupBy"]) > 0:
        count += 1
    if len(sql["orderBy"]) > 0:
        count += 1
    if sql["limit"] is not None:
        count += 1
    if len(sql["from"]["table_units"]) > 0:  # JOIN
        count += len(sql["from"]["table_units"]) - 1

    and_or = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += len([token for token in and_or if token == "or"])
    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index("like")])

    return count


def _count_component2(sql: dict) -> int:
    """Count nested sub-queries (INTERSECT/EXCEPT/UNION, sub-SELECTs)."""
    return len(_get_nested_sql(sql))


def _count_others(sql: dict) -> int:
    """Count aggregation/multi-column/multi-condition complexity signals."""
    count = 0
    agg_count = _count_agg(sql["select"][1])
    agg_count += _count_agg(sql["where"][::2])
    agg_count += _count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        agg_count += _count_agg(
            [unit[1] for unit in sql["orderBy"][1] if unit[1]]
            + [unit[2] for unit in sql["orderBy"][1] if unit[2]]
        )
    agg_count += _count_agg(sql["having"])
    if agg_count > 1:
        count += 1
    if len(sql["select"][1]) > 1:
        count += 1
    if len(sql["where"]) > 1:
        count += 1
    if len(sql["groupBy"]) > 1:
        count += 1
    return count


def eval_hardness(sql: dict) -> str:
    """Classify a parsed Spider/BIRD SQL query as easy/medium/hard/extra.

    Args:
        sql: The pre-parsed `sql` field from a Spider/BIRD dataset example
            (see module docstring - not a raw SQL string).

    Returns:
        One of "easy", "medium", "hard", "extra" - matches the official
        Spider/BIRD hardness buckets exactly, so results are comparable to
        published benchmark numbers.
    """
    count_comp1 = _count_component1(sql)
    count_comp2 = _count_component2(sql)
    count_others = _count_others(sql)

    if count_comp1 <= 1 and count_others == 0 and count_comp2 == 0:
        return "easy"
    if (count_others <= 2 and count_comp1 <= 1 and count_comp2 == 0) or (
        count_comp1 <= 2 and count_others < 2 and count_comp2 == 0
    ):
        return "medium"
    if (
        (count_others > 2 and count_comp1 <= 2 and count_comp2 == 0)
        or (2 < count_comp1 <= 3 and count_others <= 2 and count_comp2 == 0)
        or (count_comp1 <= 1 and count_others == 0 and count_comp2 <= 1)
    ):
        return "hard"
    return "extra"
