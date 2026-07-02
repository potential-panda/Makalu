import argparse
import json
import sys
from logging import getLogger
from pathlib import Path

import qate.env.sys_env as sys_env

LOG = getLogger(__name__)


def _volspread_base(exchange: str, symbol: str) -> Path:
    return Path(sys_env.PARQUET_ROOT_DIR) / "gold" / "dataset=VolSpread" / f"exchange={exchange}" / f"symbol={symbol}"


def _discover_execution_size(sym_dir: Path, requested: str | None) -> "Path | None":
    if not sym_dir.exists():
        return None
    if requested:
        d = sym_dir / f"execution_size={requested}"
        return d if d.exists() else None
    dirs = [d for d in sym_dir.iterdir() if d.is_dir() and d.name.startswith("execution_size=")]
    return dirs[0] if dirs else None


def execute_list_volspread(args) -> None:
    """List available param_ids for VolSpread (called when --list-params is set).

    Always exits 0 and returns JSON. Returns {"param_ids": [], "needs_populate": true}
    when no data exists so the agent can trigger a populate flow.
    """
    sym_dir = _volspread_base(args.exchange, args.symbol)
    es_dir = _discover_execution_size(sym_dir, getattr(args, "execution_size", None))
    if es_dir is None:
        # No data on disk — signal the agent to populate first
        print(
            json.dumps(
                {
                    "param_ids": [],
                    "needs_populate": True,
                    "message": (
                        f"No VolSpread data found for {args.exchange} {args.symbol}. "
                        "Run data-check then populate VolSpread (requires MarketPrice first)."
                    ),
                }
            )
        )
        return
    param_ids = sorted(
        p.name.split("=", 1)[1] for p in es_dir.iterdir() if p.is_dir() and p.name.startswith("param_id=")
    )
    print(json.dumps({"param_ids": param_ids, "execution_size": es_dir.name.split("=", 1)[1]}))


def execute_plot_volspread(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    sym_dir = _volspread_base(args.exchange, args.symbol)
    es_dir = _discover_execution_size(sym_dir, getattr(args, "execution_size", None))
    if es_dir is None:
        print(json.dumps({"error": f"No VolSpread data for {args.exchange} {args.symbol}"}))
        sys.exit(1)

    param_dir = es_dir / f"param_id={args.param_id}"
    if not param_dir.exists():
        print(
            json.dumps(
                {"error": f"param_id '{args.param_id}' not found. Run with --list-params to see available values."}
            )
        )
        sys.exit(1)

    dfs = []
    for date_dir in sorted(param_dir.iterdir()):
        if not date_dir.name.startswith("date="):
            continue
        date_str = date_dir.name.split("=", 1)[1]  # YYYY-MM-DD
        if args.start and date_str < args.start:
            continue
        if args.end and date_str > args.end:
            continue
        for f in sorted(date_dir.glob("*.parquet")):
            dfs.append(pd.read_parquet(f))

    if not dfs:
        print(json.dumps({"error": "No data found for the specified date range."}))
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True).sort_values("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="s")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["dt"], df["volatility_ema"], label="volatility_ema", linewidth=0.8, color="#e74c3c", alpha=0.85)
    ax.plot(df["dt"], df["spread_pct_ema"], label="spread_pct_ema", linewidth=0.8, color="#3498db", alpha=0.85)
    ax.set_title(f"VolSpread — {args.exchange} {args.symbol} | {args.param_id}\n{args.start} → {args.end}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    LOG.info(f"Plot saved: {output_path} ({len(df)} rows)")
    print(json.dumps({"path": str(output_path)}))


def register_cli(parser: argparse.ArgumentParser) -> None:
    p = parser.add_parser(
        "plot",
        help="Generate a PNG plot from a parquet dataset",
        description=(
            "Read parquet data and render a matplotlib PNG.\n\n"
            "Supported datasets: VolSpread\n\n"
            "Two modes:\n"
            "  --list-params   List available param_ids for the given market (no --param-id needed)\n"
            "  normal          Generate a PNG (requires --param-id and --output)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=["VolSpread"], help="Dataset to plot")
    p.add_argument("--exchange", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD")
    p.add_argument(
        "--execution-size", dest="execution_size", help="Execution size partition (auto-discovered if omitted)"
    )
    p.add_argument("--param-id", dest="param_id", help="param_id partition value (required unless --list-params)")
    p.add_argument("--output", help="Output PNG file path (required unless --list-params)")
    p.add_argument("--list-params", action="store_true", help="List available param_ids and exit")
    p.set_defaults(func=_dispatch)


def _dispatch(args) -> None:
    if args.dataset == "VolSpread":
        if args.list_params:
            execute_list_volspread(args)
        else:
            if not args.param_id:
                print(
                    json.dumps({"error": "VolSpread requires --param-id. Use --list-params to see available values."})
                )
                sys.exit(1)
            if not args.output:
                print(json.dumps({"error": "--output is required"}))
                sys.exit(1)
            execute_plot_volspread(args)
    else:
        print(json.dumps({"error": f"Unsupported dataset: {args.dataset}"}))
        sys.exit(1)
