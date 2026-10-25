from __future__ import annotations

import json

from makalu.engine import Engine


def register_cli(subparsers) -> None:
    p = subparsers.add_parser(
        "query", help="Query the Makalu data warehouse via DuckDB"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--sql", help="SQL SELECT query to execute")
    group.add_argument(
        "--list", action="store_true", help="List all registered datasets"
    )
    p.add_argument("--format", choices=["json", "table"], default="json")
    p.set_defaults(func=run)


def run(args) -> None:
    try:
        engine = Engine()
    except Exception as e:
        print(json.dumps({"error": f"Failed to initialize engine: {e}"}))
        return

    if args.list:
        df = engine.list()
        if args.format == "table":
            print(df.to_string(index=False))
        else:
            print(json.dumps({"datasets": df.to_dict(orient="records")}, default=str))
        return

    try:
        df = engine.query(args.sql)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return

    if args.format == "table":
        print(df.to_string(index=False))
    else:
        rows = df.head(200).to_dict(orient="records")
        result = {"row_count": len(df), "columns": list(df.columns), "rows": rows}
        if len(df) > 200:
            result["truncated"] = True
            result["note"] = (
                f"Showing first 200 of {len(df)} rows. Add LIMIT to your query."
            )
        print(json.dumps(result, default=str))
