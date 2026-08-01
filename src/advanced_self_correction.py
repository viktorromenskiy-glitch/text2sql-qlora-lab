"""Advanced self-correction (N-candidate majority voting) - Priority 5,
implementation_priority_plan.md.

DIFFERENT from src/self_correction.py (already tested, 0 effect - single
greedy generation + 1 retry only on genuine execution failure, and our
errors are mostly "valid SQL, wrong logic", which that version can't
catch at all). This version generates N candidates via SAMPLING
(temperature>0, not greedy do_sample=False used everywhere else in this
project) and picks by majority vote on the EXECUTION RESULT, not the SQL
text - can in principle catch "wrong logic" errors too, if enough of the
N samples happen to get it right and agree with each other, unlike the
single-attempt/single-retry version.

Expensive: N generations per example instead of 1 - only justified if
Priorities 1-4 (0.740 EX) leave enough remaining error budget to be
worth the extra GPU time; see technical_lessons_learned.md.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def canonical_result(rows: list | None) -> tuple | None:
    """Turn a query result into a hashable, order-independent signature
    for grouping - mirrors evaluator.compare_results's row-order-blind
    comparison, so voting groups are consistent with how we actually
    score correctness.

    Args:
        rows: Query result rows, or None if execution failed.

    Returns:
        A hashable signature: None stays None (all failures group
        together as their own "result"), successful results become a
        sorted tuple of rows (order-independent).
    """
    if rows is None:
        return None
    return tuple(sorted(rows))


def majority_vote(candidates: list[tuple[str | None, list | None]]) -> str | None:
    """Pick the SQL whose execution result is most common among candidates.

    Args:
        candidates: List of (sql, rows) pairs - one per generated
            candidate, already executed (rows=None if that candidate's
            SQL failed to execute or wasn't extracted).

    Returns:
        The SQL string from the winning (most common result) group, or
        None if candidates is empty. Ties broken by: prefer a
        non-failure group over an all-failure group; among remaining
        ties, prefer the first-generated candidate (earliest index) -
        deterministic, not random, so results are reproducible.
    """
    if not candidates:
        return None

    signatures = [canonical_result(rows) for _, rows in candidates]
    counts = Counter(signatures)

    def group_priority(signature: tuple | None) -> tuple:
        # Sort key: (vote count descending, is-not-failure descending) -
        # Python's max() with this key picks the most-voted, preferring
        # real results over grouped failures on a tie.
        return (counts[signature], signature is not None)

    winning_signature = max(set(signatures), key=group_priority)

    for sql, rows in candidates:
        if canonical_result(rows) == winning_signature:
            return sql
    return None  # unreachable in practice, but keeps type-checkers happy


def generate_n_candidates(
    model: Any, tokenizer: Any, prompt: str, n: int = 5, max_new_tokens: int = 256,
    temperature: float = 0.7, top_p: float = 0.95,
) -> list[str | None]:
    """Generate N candidate completions via sampling (not greedy).

    Args:
        model, tokenizer: loaded model, as returned by src.model functions.
        prompt: the full ChatML prompt.
        n: number of candidates to generate.
        max_new_tokens: generation length cap, matches the rest of the pipeline.
        temperature, top_p: sampling parameters - per Gemini's proposal
            (implementation_priority_plan.md), not independently tuned.

    Returns:
        List of n extracted SQL strings (or None per-candidate if no
        ```sql block was found in that generation).
    """
    from src.sanitizer import extract_sql  # local import - keeps this testable without src on path at module level

    candidates = []
    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p,
        )
        raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        candidates.append(extract_sql(raw_output))
    return candidates
