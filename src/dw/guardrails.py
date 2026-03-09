"""
SQL guardrails for the LLM research interface.

Enforces read-only access, row limits, and basic structural safety
before any SQL is executed against the warehouse.
"""

from __future__ import annotations

import re

# Keywords that indicate write or schema-modifying operations
_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE|COPY|VACUUM|ATTACH|DETACH)\b",
    re.IGNORECASE,
)

# Detect LIMIT clause already present
_HAS_LIMIT = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)

# Detect FROM ... , ... (implicit cross join — two or more tables with no ON)
_IMPLICIT_CROSS_JOIN = re.compile(r"\bFROM\b[^;]+,\s*\w+(?!\s*ON\b)", re.IGNORECASE)

MAX_ROWS = 10_000
MAX_DATE_SPAN_DAYS = 365


class GuardrailError(ValueError):
    """Raised when a query violates a guardrail."""


def validate(sql: str) -> str:
    """
    Validate and sanitise a SQL query.

    Returns the (possibly modified) SQL string with a LIMIT injected if
    absent, or raises GuardrailError for policy violations.
    """
    stripped = sql.strip().rstrip(";")

    # Must be a SELECT
    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        raise GuardrailError("Only SELECT queries are allowed.")

    # Block write / DDL keywords anywhere in the query
    match = _WRITE_KEYWORDS.search(stripped)
    if match:
        raise GuardrailError(f"Query contains forbidden keyword: {match.group().upper()}")

    # Warn on implicit cross joins (two table names after FROM with a comma
    # but no explicit JOIN ... ON)
    if _IMPLICIT_CROSS_JOIN.search(stripped):
        raise GuardrailError(
            "Query looks like an implicit cross join (comma-separated tables). "
            "Use explicit JOIN ... ON syntax."
        )

    # Inject or cap LIMIT
    if not _HAS_LIMIT.search(stripped):
        stripped = f"{stripped}\nLIMIT {MAX_ROWS}"
    else:
        # Cap existing LIMIT if it exceeds MAX_ROWS
        def _cap(m: re.Match) -> str:
            n = int(m.group(1))
            return f"LIMIT {min(n, MAX_ROWS)}"

        stripped = re.sub(r"\bLIMIT\s+(\d+)", _cap, stripped, flags=re.IGNORECASE)

    return stripped
