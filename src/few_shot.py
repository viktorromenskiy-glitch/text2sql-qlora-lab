"""Few-shot examples - Priority 4, implementation_priority_plan.md.

Deliberately starts with the CHEAPEST possible retrieval (word overlap,
no embeddings model, no new heavy dependencies) rather than immediately
building a full Sentence-Transformers pipeline - literature is
contradictory on whether few-shot helps at all on top of SFT (DAIL-SQL:
+2-4%; a separate Llama 3 8B write-up: "inconclusive... lowered or
stayed the same" - see technical_lessons_learned.md, gemini_challenge
discussion). If this cheap version shows no benefit, the heavier
embeddings version likely wouldn't be worth building either.

Retrieves from the Spider TRAIN split only (never dev - that would be
leakage, since dev is what we evaluate on).
"""
from __future__ import annotations

import re

WORD_PATTERN = re.compile(r"[a-zA-Z]+")


def _words(text: str) -> set[str]:
    """Same naive stemming as schema_pruning._words - kept as a small,
    separate copy rather than a shared import, since the two modules
    are independent, optional techniques (see implementation_priority_plan.md)
    and neither should require the other to function standalone."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    words = {w.lower() for w in WORD_PATTERN.findall(spaced) if len(w) > 2}
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}


def find_similar_examples(question: str, train_examples: list[dict], k: int = 2) -> list[dict]:
    """Find the k most similar training examples by question word overlap.

    Args:
        question: The dev-set question to find similar train examples for.
        train_examples: Spider TRAIN split examples (each needs
            "question" and "query" keys) - NEVER pass dev examples here,
            that would leak the answer.
        k: How many similar examples to return.

    Returns:
        Up to k training examples (dicts with at least "question" and
        "query"), most similar first. Empty list if train_examples is
        empty.
    """
    if not train_examples:
        return []

    question_words = _words(question)

    def overlap_score(example: dict) -> int:
        return len(_words(example["question"]) & question_words)

    scored = sorted(train_examples, key=overlap_score, reverse=True)
    return scored[:k]


def format_few_shot_block(similar_examples: list[dict]) -> str:
    """Format similar examples as a few-shot block for prompt insertion.

    Args:
        similar_examples: Output of find_similar_examples.

    Returns:
        A text block like:
        "Example 1:\nQuestion: ...\nSQL: ...\n\nExample 2:\n..."
        Empty string if similar_examples is empty (caller should skip
        insertion entirely in that case, not insert an empty block).
    """
    if not similar_examples:
        return ""

    blocks = []
    for i, example in enumerate(similar_examples, start=1):
        blocks.append(f"Example {i}:\nQuestion: {example['question']}\nSQL: {example['query']}")
    return "\n\n".join(blocks)
