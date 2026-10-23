import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from logging import getLogger
from pathlib import Path

from makalu import settings as sys_env

LOG = getLogger(__name__)

RAW_ROOT = Path(sys_env.MARKET_DATA_DIR)
PARQUET_ROOT = Path(sys_env.PARQUET_ROOT_DIR)


@dataclass(frozen=True)
class RawMarketDataFile:
    path: Path
    exchange: str
    symbol: str
    date_str: str

    @property
    def market_id(self) -> str:
        return f"{self.exchange}:{self.symbol}"

    @property
    def partition_date(self) -> str:
        return f"{self.date_str[:4]}-{self.date_str[4:6]}-{self.date_str[6:8]}"


def _parse_raw_file(path: Path) -> RawMarketDataFile | None:
    if path.suffix != ".msgpack":
        return None

    stem = path.stem
    prefix, sep, date_str = stem.rpartition("_")
    if not sep or len(date_str) != 8 or not date_str.isdigit():
        return None

    exchange, sep, symbol = prefix.partition("_")
    if not sep or not exchange or not symbol:
        return None

    return RawMarketDataFile(path=path, exchange=exchange, symbol=symbol, date_str=date_str)


def _bronze_partition_exists(dataset: str, raw: RawMarketDataFile) -> bool:
    path = (
        PARQUET_ROOT
        / "bronze"
        / f"dataset={dataset}"
        / f"exchange={raw.exchange}"
        / f"symbol={raw.symbol}"
        / f"date={raw.partition_date}"
        / "data.parquet"
    )
    return path.exists()


def _is_recent(raw: RawMarketDataFile, keep_days: int, today: date) -> bool:
    raw_date = datetime.strptime(raw.date_str, "%Y%m%d").date()
    cutoff = today - timedelta(days=keep_days - 1)
    return raw_date >= cutoff


def _matches_market(raw: RawMarketDataFile, market: str | None) -> bool:
    if market is None:
        return True
    return raw.market_id == market.upper()


def _format_size(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} TiB"


def execute(args) -> None:
    if args.keep_days < 1:
        raise ValueError("--keep-days must be >= 1")

    today = date.today()
    candidates: list[tuple[RawMarketDataFile, int]] = []
    skipped_recent = 0
    skipped_unpopulated = 0
    skipped_unmatched = 0
    skipped_market = 0

    for path in sorted(RAW_ROOT.glob("*.msgpack")):
        raw = _parse_raw_file(path)
        if raw is None:
            skipped_unmatched += 1
            continue
        if not _matches_market(raw, args.market):
            skipped_market += 1
            continue
        if _is_recent(raw, args.keep_days, today):
            skipped_recent += 1
            continue
        if not (_bronze_partition_exists("Trade", raw) and _bronze_partition_exists("OrderBook", raw)):
            skipped_unpopulated += 1
            continue

        candidates.append((raw, path.stat().st_size))

    total_size = sum(size for _, size in candidates)
    action = "DELETE" if args.apply else "WOULD DELETE"

    for raw, size in candidates:
        LOG.info(f"{action:<12} {raw.market_id:<22} {raw.partition_date}  {_format_size(size):>10}  {raw.path}")

    if args.apply:
        deleted = 0
        for raw, _ in candidates:
            try:
                raw.path.unlink()
                deleted += 1
            except Exception as e:
                LOG.error(f"Error deleting {raw.path}: {e}")
    else:
        deleted = 0

    summary = {
        "status": "done",
        "dry_run": not args.apply,
        "raw_root": str(RAW_ROOT),
        "parquet_root": str(PARQUET_ROOT),
        "keep_days": args.keep_days,
        "market": args.market,
        "matched_for_delete": len(candidates),
        "deleted": deleted,
        "bytes": total_size,
        "human_size": _format_size(total_size),
        "skipped_recent": skipped_recent,
        "skipped_unpopulated": skipped_unpopulated,
        "skipped_unmatched": skipped_unmatched,
        "skipped_other_market": skipped_market,
    }
    LOG.info(json.dumps(summary))

    if not args.apply:
        LOG.info("Dry-run. Run with --apply to delete.")


def register_cli(sub) -> None:
    p = sub.add_parser(
        "market-data-cleanup",
        help="Delete raw msgpack files that are already populated to bronze (dry-run by default)",
        description=(
            "Deletes raw /panda/market-data/*.msgpack files only when the same\n"
            "exchange, symbol, and date exist in both bronze Trade and bronze\n"
            "OrderBook parquet. Recent files are kept to avoid racing daily ingest.\n\n"
            "Policy by default:\n"
            "  delete only raw files whose date exists in both bronze Trade and bronze OrderBook\n"
            "  keep the most recent 3 calendar days\n\n"
            "IMPORTANT: Always run without --apply first to review what would be deleted."
        ),
        epilog=(
            "Examples:\n"
            "  python src/cli.py market-data-cleanup\n"
            "  python src/cli.py market-data-cleanup --market GMO:BTC_JPY\n"
            "  python src/cli.py market-data-cleanup --apply"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--apply", action="store_true", help="Actually delete files (default: dry-run)")
    p.add_argument("--keep-days", type=int, default=3, help="Keep this many recent calendar days (default: 3)")
    p.add_argument("--market", help="Optional EXCHANGE:SYMBOL filter, e.g. GMO:BTC_JPY")
    p.set_defaults(func=execute)
