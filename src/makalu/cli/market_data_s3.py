import argparse
import json

from qate_market_data.inventory import date_window, load_or_refresh_inventory, missing_dates, to_ranges


def _print_market_table(inventory: dict, from_cache: bool) -> None:
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


def _print_market_detail(inventory: dict, market: str, start: str | None, end: str | None) -> None:
    market = market.upper()
    data = inventory["markets"].get(market)
    if data is None:
        print(f"No S3 raw market-data found for {market}")
        return

    missing = missing_dates(data["dates"], start, end)
    window_start, window_end = date_window(data["dates"], start, end)

    print(f"Market        : {market}")
    print(f"First date    : {data['first_date']}")
    print(f"Last date     : {data['last_date']}")
    print(f"Available days: {data['days']}")
    print(f"Source objects: {data['source_objects']}")
    print(f"Window        : {window_start.isoformat()} -> {window_end.isoformat()}")
    print(f"Missing days  : {len(missing)}")
    for r in to_ranges(missing):
        print(f"  {r}")


def execute(args) -> None:
    inventory, from_cache = load_or_refresh_inventory(args.refresh)

    if args.json:
        payload = {"from_cache": from_cache, "inventory": inventory}
        if args.market:
            market = args.market.upper()
            data = inventory["markets"].get(market)
            payload["market"] = market
            payload["market_data"] = data
            if data is not None:
                payload["missing_dates"] = missing_dates(data["dates"], args.start, args.end)
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--refresh", action="store_true", help="Rescan S3 and overwrite the local cache")
    p.add_argument("--market", help="Optional EXCHANGE:SYMBOL detail view, e.g. GMO:BTC_SPOT")
    p.add_argument("--start", help="Optional detail-view start date, YYYY-MM-DD or YYYYMMDD")
    p.add_argument("--end", help="Optional detail-view end date, YYYY-MM-DD or YYYYMMDD")
    p.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    p.set_defaults(func=execute)
