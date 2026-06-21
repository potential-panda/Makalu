import argparse
from datetime import date, timedelta
from logging import getLogger
from pathlib import Path

from .common import DATASETS, get_dataset_root

LOG = getLogger(__name__)


def _list_partition_values(path: Path, key: str) -> list[str]:
    """Return sorted values for a partition key at the given path level."""
    prefix = f"{key}="
    start = len(prefix)
    return sorted(d.name[start:] for d in path.iterdir() if d.is_dir() and d.name.startswith(prefix))


def _collect_dates(path: Path) -> list[str]:
    """Return sorted date strings under a fully-resolved partition path."""
    prefix = "date="
    dates = []
    for d in path.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            parquet = d / "data.parquet"
            if parquet.exists():
                start = len(prefix)
                dates.append(d.name[start:])
    return sorted(dates)


def _missing_dates(dates: list[str], start: str | None, end: str | None) -> list[str]:
    if not dates:
        return []
    d_start = date.fromisoformat(start) if start else date.fromisoformat(dates[0])
    d_end = date.fromisoformat(end) if end else date.fromisoformat(dates[-1])
    existing = set(dates)
    missing = []
    cur = d_start
    while cur <= d_end:
        s = cur.isoformat()
        if s not in existing:
            missing.append(s)
        cur += timedelta(days=1)
    return missing


def _to_ranges(dates: list[str]) -> list[str]:
    """Compress a sorted list of date strings into human-readable ranges."""
    if not dates:
        return []
    ranges = []
    start = prev = date.fromisoformat(dates[0])
    for d in dates[1:]:
        cur = date.fromisoformat(d)
        if (cur - prev).days == 1:
            prev = cur
        else:
            ranges.append(_range_str(start, prev))
            start = prev = cur
    ranges.append(_range_str(start, prev))
    return ranges


def _range_str(start: date, end: date) -> str:
    if start == end:
        return str(start)
    return f"{start} → {end}"


def execute(args):
    dataset = args.dataset
    if dataset not in DATASETS:
        LOG.error(f"Unknown dataset '{dataset}'. Available: {', '.join(sorted(DATASETS))}")
        return

    layer, keys = DATASETS[dataset]
    base = get_dataset_root(layer, dataset)

    if not base.exists():
        LOG.warning(f"No data found for {dataset} at {base}")
        return

    # Build mapping of key -> provided value
    provided: dict[str, str | None] = {}
    for key in keys:
        provided[key] = getattr(args, key, None)

    # Walk partition keys one by one; stop and list when a value is missing
    path = base
    for key in keys:
        val = provided[key]
        if val is None:
            values = _list_partition_values(path, key)
            if not values:
                LOG.warning(f"No data under {path}")
                return
            print(f"Available {key}:")
            for v in values:
                print(f"  {v}")
            return
        next_path = path / f"{key}={val}"
        if not next_path.exists():
            LOG.warning(f"No data for {key}={val} under {path}")
            return
        path = next_path

    # All partition keys resolved — show date summary
    dates = _collect_dates(path)
    if not dates:
        LOG.warning(f"No parquet files found under {path}")
        return

    missing = _missing_dates(dates, args.start, args.end)

    partition_desc = " / ".join(f"{k}={provided[k]}" for k in keys)
    print(f"Dataset : {dataset}")
    print(f"Location: {path}")
    print(f"Partition: {partition_desc}")
    print(f"Min date: {dates[0]}")
    print(f"Max date: {dates[-1]}")
    print(f"Total   : {len(dates)} days on disk")
    if args.start or args.end:
        window_start = args.start or dates[0]
        window_end = args.end or dates[-1]
        print(f"Window  : {window_start} → {window_end}")
        print(f"Missing : {len(missing)} days")
        if missing:
            # Show missing as compact ranges
            ranges = _to_ranges(missing)
            for r in ranges:
                print(f"  {r}")
    else:
        print(f"Missing : {len(missing)} days (between min and max)")
        if missing:
            for r in _to_ranges(missing):
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
