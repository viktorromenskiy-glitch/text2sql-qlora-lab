"""Value Retrieval - Priority 2, implementation_priority_plan.md.

Searches the question for candidate value mentions (proper nouns / quoted
phrases), looks them up against REAL column values in the target
database, and returns hints to inject into the prompt - e.g. the model
sees "Note: value 'London' was found in table offices, column city"
instead of guessing blindly whether 'london'/'London'/'LONDON' is the
correct casing/spelling stored in the database.

Addresses a specific, plausible failure mode identified by Gemini
(Value Mismatch) and confirmed relevant by our own error analysis
(diagnose_errors.py): the model can write syntactically valid SQL that
returns wrong/empty results purely because the literal value doesn't
match what's actually stored (case, exact wording).
"""
from __future__ import annotations

import re
import sqlite3

# Candidate value mentions in a question - capitalized words/phrases
# (proper nouns tend to be the values that matter for WHERE clauses:
# city/country/person names, abbreviations). Deliberately simple - not
# full NER - matches our "cheap, dependency-free" approach throughout
# this project.
CAPITALIZED_PHRASE_PATTERN = re.compile(r"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b")

# Common sentence-initial words that are capitalized but NOT value
# candidates - excluding these avoids noisy, useless hints (e.g. every
# question starting with "What"/"Find"/"List" would otherwise trigger a
# DB lookup for the word "What" itself).
QUESTION_STARTER_WORDS = {
    "What", "Which", "Who", "Where", "When", "How", "Find", "List",
    "Show", "Give", "Return", "Name", "Count", "Get", "Select",
}


def extract_candidate_values(question: str) -> list[str]:
    """Extract capitalized word/phrase candidates from a question.

    Args:
        question: The natural-language question.

    Returns:
        Candidate substrings that might be literal values referenced in
        the question (e.g. "London", "General Motors", "USA") - may
        include false positives (sentence-initial capitalized common
        words are filtered, but not exhaustively); false positives are
        cheap here since find_value_hints only keeps ones that actually
        match real DB content.
    """
    candidates = CAPITALIZED_PHRASE_PATTERN.findall(question)
    return [c for c in candidates if c not in QUESTION_STARTER_WORDS]


def find_value_hints(question: str, schema: dict, db_path: str, max_hints: int = 5) -> list[str]:
    """Look up candidate values from the question against real DB content.

    Only checks text-type columns (per schema["column_types"]) - numeric
    value matching is a different, not-yet-addressed problem.

    Args:
        question: The natural-language question.
        schema: This example's schema entry (needs column_names_original,
            column_types, table_names_original).
        db_path: Path to the .sqlite database for this example.
        max_hints: Cap on returned hints - keeps the prompt addition
            bounded even if a question has many capitalized words.

    Returns:
        List of hint strings, e.g. ['Note: value "London" was found in
        table "offices", column "city".'] - empty if no candidate
        matched any real value, or if there's nothing to check.
    """
    candidates = extract_candidate_values(question)
    if not candidates:
        return []

    table_names = schema["table_names_original"]
    column_entries = schema["column_names_original"]
    column_types = schema["column_types"]

    hints = []
    try:
        connection = sqlite3.connect(db_path)
        for column_index, (table_index, column_name) in enumerate(column_entries):
            if column_name == "*" or column_types[column_index] != "text":
                continue
            table_name = table_names[table_index]

            for candidate in candidates:
                if len(hints) >= max_hints:
                    connection.close()
                    return hints
                try:
                    cursor = connection.execute(
                        f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
                        f'WHERE "{column_name}" = ? COLLATE NOCASE LIMIT 1',
                        (candidate,),
                    )
                    row = cursor.fetchone()
                except sqlite3.Error:
                    continue
                if row is not None:
                    hints.append(f'Note: value "{row[0]}" was found in table "{table_name}", column "{column_name}".')
        connection.close()
    except sqlite3.Error:
        return []

    return hints
