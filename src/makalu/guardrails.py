"""
SQL guardrails for the LLM research interface.
"""

from __future__ import annotations

import re

_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE|COPY|VACUUM|ATTACH|DETACH)\b",
    re.IGNORECASE,
)
_HAS_LIMIT = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)
_IMPLICIT_CROSS_JOIN = re.compile(r"\bFROM\b[^;]+,\s*\w+(?!\s*ON\b)", re.IGNORECASE)

MAX_ROWS = 10_000
MAX_DATE_SPAN_DAYS = 365


class GuardrailError(ValueError):
    pass


def validate(sql: str) -> str:
    stripped = sql.strip().rstrip(";")
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        raise GuardrailError("Only SELECT queries are allowed.")

    match = _WRITE_KEYWORDS.search(stripped)
    if match:
        raise GuardrailError(f"Query contains forbidden keyword: {match.group().upper()}")

    if _IMPLICIT_CROSS_JOIN.search(stripped):
        raise GuardrailError(
            "Query looks like an implicit cross join (comma-separated tables). Use explicit JOIN ... ON syntax."
        )

    if not _HAS_LIMIT.search(stripped):
        stripped = f"{stripped}\nLIMIT {MAX_ROWS}"
    else:
        def _cap(m: re.Match[str]) -> str:
            n = int(m.group(1))
            return f"LIMIT {min(n, MAX_ROWS)}"

        stripped = re.sub(r"\bLIMIT\s+(\d+)", _cap, stripped, flags=re.IGNORECASE)

    return stripped
