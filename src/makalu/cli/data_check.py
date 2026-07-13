import argparse
from logging import getLogger

from makalu.data_access import dataset_availability

LOG = getLogger(__name__)


def execute(args):
    dataset = args.dataset
    partitions = {
        "exchange": args.exchange,
        "symbol": args.symbol,
        "execution_size": args.execution_size,
        "param_id": args.param_id,
        "market_a": args.market_a,
        "market_b": args.market_b,
    }
    info = dataset_availability(dataset, partitions=partitions, start=args.start, end=args.end)

    if info.get("state") == "unknown_dataset":
        LOG.error(info["error"])
        return
    if info.get("state") == "missing_dataset":
        LOG.warning(f"No data found for {dataset} at {info['base_path']}")
        return
    if info.get("state") == "needs_partition":
        print(f"Available {info['next_key']}:")
        for v in info.get("available_values", []):
            print(f"  {v}")
        return
    if info.get("state") == "missing_partition":
        LOG.warning(
            f"No data for {info['missing_key']}={info['missing_value']} under {info['path']}"
        )
        return
    if info.get("state") == "summary":
        partition_desc = " / ".join(
            f"{k}={v}" for k, v in info["resolved_partitions"].items() if v is not None
        )
        print(f"Dataset : {info['dataset']}")
        print(f"Location: {info['path']}")
        print(f"Partition: {partition_desc}")
        print(f"Min date: {info['min_date']}")
        print(f"Max date: {info['max_date']}")
        print(f"Total   : {info['total_days']} days on disk")
        if args.start or args.end:
            print(f"Window  : {info['window_start']} → {info['window_end']}")
            print(f"Missing : {info['missing_days']} days")
            for r in info["missing_ranges"]:
                print(f"  {r}")
        else:
            print(f"Missing : {info['missing_days']} days (between min and max)")
            for r in info["missing_ranges"]:
                print(f"  {r}")


_DESCRIPTION = """\
READ-ONLY. Inspect what parquet data is on disk for a given dataset.

Uses progressive disclosure: omit partition arguments to list available values.
  --dataset only              → list available exchanges
  + --exchange                → list available symbols
  + --symbol (+ extra keys)   → list available extra partition values (execution_size, param_id)
  all partitions provided     → show date summary (min, max, total days, missing dates)

OUTPUT FORMAT (when all partitions are provided)
  Dataset  : <name>
  Location : <path on disk>
  Partition: <key=value / ...>
  Min date : YYYY-MM-DD
  Max date : YYYY-MM-DD
  Total    : N days on disk
  Missing  : N days          (optionally followed by date ranges of missing days)

Available datasets (feature/research layers excluded):
  Trade, OrderBook                                    bronze — raw tick data
  BookChange, BookState, TradeAggression, MarketPrice silver — derived metrics
  Chart_CANDLE, Chart_HEIKIN_ASHI, Chart_RANGE        gold   — bar charts
  VolSpread                                           gold   — vol/spread signals
  LeadLag                                             gold   — cross-market lead-lag
"""

_EPILOG = """\
Examples:
  # List available exchanges for Trade
  python cli.py data-check --dataset Trade

  # List available symbols for GMO
  python cli.py data-check --dataset Trade --exchange GMO

  # Full date summary
  python cli.py data-check --dataset Trade --exchange GMO --symbol BTC_JPY

  # Check a specific window for missing dates
  python cli.py data-check --dataset Trade --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-31

  # MarketPrice requires --execution-size
  python cli.py data-check --dataset MarketPrice --exchange GMO --symbol BTC_JPY --execution-size 0.002
"""


def register_cli(sub):
    p = sub.add_parser(
        "data-check",
        help="Inspect data availability on disk (read-only, safe to run anytime)",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name. Run without other args to discover available exchanges.",
    )
    p.add_argument("--exchange", default=None, help="Exchange name, e.g. GMO, BINANCE")
    p.add_argument("--symbol", default=None, help="Symbol name, e.g. BTC_JPY, BTC_USDT")
    p.add_argument("--execution-size", dest="execution_size", default=None, help="Required for MarketPrice, VolSpread")
    p.add_argument("--param-id", dest="param_id", default=None, help="Required for VolSpread, Chart_*, LeadLag")
    p.add_argument("--market-a", dest="market_a", default=None, help="Required for LeadLag (e.g. BINANCE:BTC_USDT)")
    p.add_argument("--market-b", dest="market_b", default=None, help="Required for LeadLag")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD — narrows the missing-date check window")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD — narrows the missing-date check window")
    p.set_defaults(func=execute)
