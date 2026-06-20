import argparse
import json
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qate.env import sys_env
from qate.env.sys_env import get_s3_client
from qate.store.s3 import list_files

from .market_data import BUCKETS

CACHE_PATH = Path(sys_env.PARQUET_ROOT_DIR) / ".cache" / "market_data_s3_inventory.json"


def _market_from_local_name(name: str) -> tuple[str, str] | None:
    prefix, sep, date_str = name.rpartition("_")
    if not sep or len(date_str) != 8 or not date_str.isdigit():
        return None

    exchange, sep, symbol = prefix.partition("_")
    if not sep or not exchange or not symbol:
        return None

    return f"{exchange}:{symbol}", date_str


def _scan_s3_inventory() -> dict[str, Any]:
    client = get_s3_client()
    markets: dict[str, dict[str, Any]] = defaultdict(lambda: {"dates": set(), "source_objects": 0})
    scanned_sources = []

    for host, (bucket, env) in BUCKETS.items():
        prefix = f"{env}_MarketData"
        scanned_sources.append({"host": host, "bucket": bucket, "prefix": prefix})
        files = list_files(client, bucket, prefix=prefix)
        for name, items in files.items():
            parsed = _market_from_local_name(name)
            if parsed is None:
                continue
            market, date_str = parsed
            markets[market]["dates"].add(date_str)
            markets[market]["source_objects"] += len(items)

    out_markets = {}
    for market, data in sorted(markets.items()):
        dates = sorted(data["dates"])
        out_markets[market] = {
            "dates": dates,
            "source_objects": data["source_objects"],
            "days": len(dates),
            "first_date": _fmt_date(dates[0]) if dates else None,
            "last_date": _fmt_date(dates[-1]) if dates else None,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_path": str(CACHE_PATH),
        "scanned_sources": scanned_sources,
        "markets": out_markets,
    }


def _load_or_refresh_inventory(refresh: bool) -> tuple[dict[str, Any], bool]:
    if CACHE_PATH.exists() and not refresh:
        return json.loads(CACHE_PATH.read_text()), True

    inventory = _scan_s3_inventory()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(inventory, indent=2, sort_keys=True))
    return inventory, False


def _fmt_date(date_str: str) -> str:
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def _parse_date(value: str) -> date:
    if "-" in value:
        return date.fromisoformat(value)
    return datetime.strptime(value, "%Y%m%d").date()


def _date_window(dates: list[str], start: str | None, end: str | None) -> tuple[date, date]:
    if not dates and (not start or not end):
        raise ValueError("--start and --end are required when the selected market has no available dates")
    start_dt = _parse_date(start) if start else _parse_date(dates[0])
    end_dt = _parse_date(end) if end else _parse_date(dates[-1])
    if end_dt < start_dt:
        raise ValueError("--end must be >= --start")
    return start_dt, end_dt


def _missing_dates(dates: list[str], start: str | None, end: str | None) -> list[str]:
    start_dt, end_dt = _date_window(dates, start, end)
    existing = {_parse_date(d) for d in dates}
    missing = []
    cur = start_dt
    while cur <= end_dt:
        if cur not in existing:
            missing.append(cur.isoformat())
        cur += timedelta(days=1)
    return missing


def _to_ranges(dates: list[str]) -> list[str]:
    if not dates:
        return []

    ranges = []
    start = prev = date.fromisoformat(dates[0])
    for value in dates[1:]:
        cur = date.fromisoformat(value)
        if cur == prev + timedelta(days=1):
            prev = cur
        else:
            ranges.append(_range_str(start, prev))
            start = prev = cur
    ranges.append(_range_str(start, prev))
    return ranges


def _range_str(start: date, end: date) -> str:
    if start == end:
        return start.isoformat()
    return f"{start.isoformat()} -> {end.isoformat()}"


def _print_market_table(inventory: dict[str, Any], from_cache: bool) -> None:
    source = "cache" if from_cache else "s3"
    print(f"Source      : {source}")
    print(f"Generated at: {inventory['generated_at']}")
    print(f"Cache       : {inventory['cache_path']}")
    print(f"Markets     : {len(inventory['markets'])}")
    print()
    print(f"{'Market':<24} {'First':<10} {'Last':<10} {'Days':>6} {'Source objects':>14}")
    print("-" * 72)
    for market, data in inventory["markets"].items():
        print(
            f"{market:<24} {data['first_date'] or '-':<10} {data['last_date'] or '-':<10} "
            f"{data['days']:>6} {data['source_objects']:>14}"
        )


def _print_market_detail(inventory: dict[str, Any], market: str, start: str | None, end: str | None) -> None:
    market = market.upper()
    data = inventory["markets"].get(market)
    if data is None:
        print(f"No S3 raw market-data found for {market}")
        return

    dates = data["dates"]
    missing = _missing_dates(dates, start, end)
    window_start, window_end = _date_window(dates, start, end)

    print(f"Market        : {market}")
    print(f"First date    : {data['first_date']}")
    print(f"Last date     : {data['last_date']}")
    print(f"Available days: {data['days']}")
    print(f"Source objects: {data['source_objects']}")
    print(f"Window        : {window_start.isoformat()} -> {window_end.isoformat()}")
    print(f"Missing days  : {len(missing)}")
    for r in _to_ranges(missing):
        print(f"  {r}")


def execute(args) -> None:
    inventory, from_cache = _load_or_refresh_inventory(args.refresh)

    if args.json:
        payload = {
            "from_cache": from_cache,
            "inventory": inventory,
        }
        if args.market:
            market = args.market.upper()
            data = inventory["markets"].get(market)
            payload["market"] = market
            payload["market_data"] = data
            if data is not None:
                payload["missing_dates"] = _missing_dates(data["dates"], args.start, args.end)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.market:
        _print_market_detail(inventory, args.market, args.start, args.end)
    else:
        _print_market_table(inventory, from_cache)


def register_cli(sub) -> None:
    p = sub.add_parser(
        "market-data-s3",
        help="Inspect raw market-data availability in S3 (read-only, cached)",
        description=(
            "Lists raw market-data availability in S3 without downloading files.\n"
            "A full scan paginates S3 prefixes for all known market-data hosts, so\n"
            "the result is cached at /panda/makalu-data/.cache/market_data_s3_inventory.json.\n\n"
            "Run with --refresh to rescan S3. Without --refresh, the cached inventory\n"
            "is used when present."
        ),
        epilog=(
            "Examples:\n"
            "  python src/cli.py market-data-s3 --refresh\n"
            "  python src/cli.py market-data-s3\n"
            "  python src/cli.py market-data-s3 --market GMO:BTC_SPOT\n"
            "  python src/cli.py market-data-s3 --market GMO:BTC_SPOT --start 2026-04-01 --end 2026-04-15"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--refresh", action="store_true", help="Rescan S3 and overwrite the local cache")
    p.add_argument("--market", help="Optional EXCHANGE:SYMBOL detail view, e.g. GMO:BTC_SPOT")
    p.add_argument("--start", help="Optional detail-view start date, YYYY-MM-DD or YYYYMMDD")
    p.add_argument("--end", help="Optional detail-view end date, YYYY-MM-DD or YYYYMMDD")
    p.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    p.set_defaults(func=execute)
