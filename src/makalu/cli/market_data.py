import argparse
import json
from logging import getLogger

from qate_market_data.download import download_market_data, parse_market, target_dates

LOG = getLogger(__name__)


def execute(args) -> None:
    dates = target_dates(args.start, args.end, args.days)
    markets = [parse_market(args.market)] if args.market else None
    LOG.info(f"Target dates: {dates}")
    result = download_market_data(
        markets=markets,
        dates=dates,
        workers=args.workers,
        keep_merge_cache=args.keep_merge_cache,
    )
    LOG.info(f"market-data complete: {result}")
    print(json.dumps(result))


def register_cli(sub) -> None:
    p = sub.add_parser(
        "market-data",
        help="Download raw market data (msgpack) from S3 to /panda/market-data/",
        description=(
            "Downloads raw market data files (msgpack) from S3 across multiple\n"
            "hosts and merges files when two sources cover the same time window.\n\n"
            "Output dir : /panda/market-data/\n"
            "Idempotent : skips files that already exist on disk.\n\n"
            "Run this before 'populate bronze' to ensure msgpack source files are present.\n"
            "Default markets: BINANCE BTC/ETH/XRP/SOL, GMO BTC/ETH/XRP/SOL."
        ),
        epilog=(
            "Examples:\n"
            "  python cli.py market-data                           # all markets, last 3 days\n"
            "  python cli.py market-data --days 7                  # all markets, last 7 days\n"
            "  python cli.py market-data --start 2025-07-01 --end 2025-07-31\n"
            "  python cli.py market-data --market GMO:BTC_JPY      # single market, last 3 days\n"
            "  python cli.py market-data --market BINANCE:XRP_USDT --days 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--market",
        metavar="EXCHANGE:SYMBOL",
        help="Single market to download (e.g. GMO:BTC_JPY). Default: all standard markets.",
    )
    p.add_argument(
        "--days",
        type=int,
        default=3,
        metavar="N",
        help="Number of trailing days to download when --start/--end are omitted (default: 3).",
    )
    p.add_argument("--start", help="Start date YYYY-MM-DD or YYYYMMDD. Must be used with --end.")
    p.add_argument("--end", help="End date YYYY-MM-DD or YYYYMMDD. Must be used with --start.")
    p.add_argument("--workers", type=int, default=4, metavar="N", help="Parallel workers per market (default: 4).")
    p.add_argument("--keep-merge-cache", action="store_true", help="Keep decompressed source files after merges.")
    p.set_defaults(func=execute)
