"""
LLM research agent for the data warehouse.

The agent answers natural language questions about market microstructure data
by generating SQL, executing it via the Engine, and explaining the results.
It supports multi-turn conversation and enforces guardrails on every query.

Quick start (OpenAI):
    from dw import Engine, Agent, OpenAIBackend

    engine = Engine()
    agent = Agent(engine, OpenAIBackend(api_key="sk-...", model="gpt-4o"))
    print(agent.ask("Show the average spread for BINANCE XRP_USDT in July 2025"))
    print(agent.ask("Break that down by hour of day"))  # follow-up works naturally

Quick start (Ollama):
    backend = OpenAIBackend(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="llama3.1",
    )
    agent = Agent(engine, backend)

Quick start (Claude):
    from dw import Agent, ClaudeBackend
    backend = ClaudeBackend(api_key="sk-ant-...", model="claude-sonnet-4-6")
    agent = Agent(engine, backend)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backends import LLMBackend, ToolCall
from .engine import Engine
from .guardrails import GuardrailError, validate

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFINITIONS_PATH = _REPO_ROOT / "doc" / "dataset_definitions.md"

# How many result rows to include in the model's context
_MAX_SAMPLE_ROWS = 20

_RUN_SQL_TOOL = {
    "name": "run_sql",
    "description": (
        "Execute a DuckDB SQL SELECT query against the data warehouse. "
        "Returns a JSON object with columns, row_count, and sample rows. "
        "Always filter by exchange, symbol, and date range when possible to avoid scanning too much data. "
        "View names are lowercase dataset names (e.g. VolSpread -> volspread, Chart_CANDLE -> chart_candle)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "The SQL SELECT query to execute.",
            },
            "rationale": {
                "type": "string",
                "description": "One sentence explaining what this query is computing and why.",
            },
        },
        "required": ["sql"],
    },
}


def _build_system_prompt(engine: Engine) -> str:
    catalog_lines = ["## Available Datasets\n"]
    for entry in engine._catalog:
        name = entry["dataset_name"]
        if name not in engine._datasets:
            continue
        partitions = ", ".join(entry.get("partition_columns", []))
        catalog_lines.append(
            f"- **{name}** (layer={entry['layer']}, grain={entry['grain']}, "
            f"view=`{name.lower()}`, partitions: {partitions})"
        )

    definitions = ""
    if _DEFINITIONS_PATH.exists():
        definitions = _DEFINITIONS_PATH.read_text()

    return f"""You are a quantitative research assistant for a crypto market microstructure data warehouse.
You answer questions by generating DuckDB SQL queries, running them, and explaining the results.

{chr(10).join(catalog_lines)}

## Column Definitions

{definitions}

## Rules

1. Only use `run_sql` to fetch data — never guess or fabricate numbers.
2. Use exact column names from the definitions above.
3. View names in SQL are lowercase dataset names (e.g. VolSpread → `volspread`).
4. Always include exchange, symbol, and a date range filter when possible.
5. Only SELECT queries are allowed. No DDL or data modification.
6. If a query returns no rows, say so clearly and suggest why (wrong date range, wrong symbol, etc.).
7. After showing results, suggest one or two follow-up analyses.
"""


def _format_result(df: pd.DataFrame) -> str:
    """Serialise a DataFrame into a compact JSON string for the model."""
    result: dict = {
        "row_count": len(df),
        "columns": list(df.columns),
        "rows": json.loads(df.head(_MAX_SAMPLE_ROWS).to_json(orient="records")),
    }
    if len(df) > _MAX_SAMPLE_ROWS:
        result["note"] = f"Showing first {_MAX_SAMPLE_ROWS} of {len(df)} rows."
    return json.dumps(result)


class Agent:
    """
    Multi-turn research agent backed by any LLMBackend.

    Conversation history is owned by the backend so tool call formatting
    is correct per-provider. reset() starts a fresh conversation.
    """

    def __init__(self, engine: Engine, backend: LLMBackend):
        self._engine = engine
        self._backend = backend
        self._system = _build_system_prompt(engine)

    def reset(self) -> None:
        """Clear conversation history."""
        self._backend.reset()

    def ask(self, question: str, *, verbose: bool = False) -> str:
        """
        Ask a natural language question and return the agent's response.

        Set verbose=True to print each SQL query and row count as it executes.
        """
        user_msg: str | None = question

        for _ in range(10):  # cap agentic loop at 10 steps
            response = self._backend.step(user_msg, self._system, [_RUN_SQL_TOOL])
            user_msg = None  # subsequent steps are continuations, not new user messages

            if not response.tool_calls:
                return response.text or ""

            for tc in response.tool_calls:
                result = self._execute_tool(tc, verbose=verbose)
                self._backend.record_tool_result(tc, result)

        return "Agent loop limit reached without a final answer."

    def _execute_tool(self, tc: ToolCall, verbose: bool) -> str:
        if tc.name != "run_sql":
            return json.dumps({"error": f"Unknown tool: {tc.name}"})

        sql = tc.arguments.get("sql", "")
        rationale = tc.arguments.get("rationale", "")

        if verbose:
            print(f"\n[SQL] {rationale}")
            print(f"      {sql.strip()}")

        try:
            safe_sql = validate(sql)
        except GuardrailError as e:
            return json.dumps({"error": f"Guardrail blocked query: {e}"})

        try:
            df = self._engine.query(safe_sql)
        except Exception as e:
            return json.dumps({"error": f"Query failed: {e}"})

        if verbose:
            print(f"      → {len(df)} rows")

        return _format_result(df)
