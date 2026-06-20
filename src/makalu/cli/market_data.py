import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from logging import WARNING, getLogger
from pathlib import Path

import msgpack
from qate.core.ev_type import EventType
from qate.core.model import ExchangeName, Market, Symbol, TimeSeriesData
from qate.env.sys_env import get_s3_client
from qate.store.market_data_store import MarketDataLoader
from qate.store.s3 import download, list_files
from qate.util.serializer import OrderBookSerializer, TradeSerializer

LOG = getLogger(__name__)

OUTPUT_DIR = "/panda/market-data"
DOWNLOAD_TMP_DIR = "/panda/tmp_download"
MERGE_TMP_DIR = "/panda/tmp_merge"

DEFAULT_MARKETS = [
    Market(ExchangeName.BINANCE, Symbol.BTC_USDT),
    Market(ExchangeName.BINANCE, Symbol.ETH_USDT),
    Market(ExchangeName.BINANCE, Symbol.XRP_USDT),
    Market(ExchangeName.BINANCE, Symbol.SOL_USDT),
    Market(ExchangeName.GMO, Symbol.BTC_JPY),
    Market(ExchangeName.GMO, Symbol.ETH_JPY),
    Market(ExchangeName.GMO, Symbol.XRP_JPY),
    Market(ExchangeName.GMO, Symbol.SOL_JPY),
]

# S3 bucket → [bucket_name, env_name]
BUCKETS = {
    "vineyard-01":   ["panda-cc-i-13100000693141", "CHIHIRO_TAKAO"],
    "vineyard-02":   ["panda-cc-i-14100000694045", "CHLOE_BINGHAM"],
    "vineyard-01_a": ["panda-cc-vineyard-01",      "CHIHIRO_TAKAO"],
    "vineyard-01_b": ["panda-cc-vineyard-01",      "CARLY_GOODWIN"],
    "vineyard-02_a": ["panda-cc-vineyard-02",      "CHLOE_BINGHAM"],
    "vineyard-11":   ["panda-cc-vineyard-11",      "CASSANDRA_HILL"],
    "vineyard-12":   ["panda-cc-vineyard-12",      "CHIHIRO_TAKAO"],
    "margaux":       ["panda-cc-margaux",           "CASSANDRA_HILL"],
    "margaux-v1":    ["panda-cc-i-53100000522554",  "CARLY_GOODWIN"],
}


def _serialize(data: tuple) -> list:
    event_type, obj = data
    if event_type == EventType.MARKET_ORDER_BOOK:
        return OrderBookSerializer.serialize(obj)
    elif event_type == EventType.MARKET_TRADE:
        return TradeSerializer.serialize(obj)
    raise ValueError(f"Unexpected event type {event_type}")


def _merge_step(exchange: ExchangeName) -> float:
    if exchange == ExchangeName.BINANCE:
        return 0.2
    if exchange == ExchangeName.COINCHECK:
        return 1.0
    return 0.5


def _s3_prefix(env: str, exchange: str, symbol: str) -> str:
    return f"{env}_MarketData_{exchange}_{symbol}"


def _list_market_files(s3, market: Market, target_dates: list[str] | None) -> dict[str, list]:
    """Return {base_name: [S3Item, ...]} across all relevant hosts for this market."""
    if market.exchange_name == ExchangeName.BINANCE:
        hosts = ["vineyard-11", "vineyard-12", "margaux-v1"]
    else:
        hosts = ["vineyard-01", "vineyard-02", "vineyard-01_a", "vineyard-01_b", "vineyard-02_a"]

    result: dict[str, list] = {}
    for host in hosts:
        bucket, env = BUCKETS[host]
        prefix = _s3_prefix(env, market.exchange_name.name, market.symbol.name)
        for name, items in list_files(s3, bucket, prefix).items():
            if target_dates and not any(d in name for d in target_dates):
                continue
            result.setdefault(name, []).extend(items)

    return result


def _merge_two_sources(paths: list[Path], market: Market) -> list:
    loader = MarketDataLoader(market)
    series = [loader.load(p) for p in paths]
    if len(series) != 2:
        raise ValueError(f"Expected exactly 2 source files to merge, got {len(series)}")
    if not series[0] or not series[1]:
        raise ValueError("Cannot merge empty source file")

    start_ts = math.floor(min(s[0][1].get_ts() for s in series))
    end_ts = math.ceil(max(s[-1][1].get_ts() for s in series)) + 1.0
    step = _merge_step(market.exchange_name)

    merged = []
    pos_a, pos_b = 0, 0
    series_a, series_b = series[0], series[1]
    current_box = start_ts

    while current_box < end_ts:
        current_box += step
        has_data = False
        while pos_a < len(series_a) and series_a[pos_a][1].get_ts() < current_box:
            merged.append(series_a[pos_a])
            has_data = True
            pos_a += 1
        while pos_b < len(series_b) and series_b[pos_b][1].get_ts() < current_box:
            if not has_data:
                merged.append(series_b[pos_b])
                has_data = True
            pos_b += 1

    return [_serialize(x) for x in merged]


def _download_or_merge_one(task: dict) -> dict:
    getLogger("qate").setLevel(WARNING)
    market = Market(ExchangeName[task["exchange"]], Symbol[task["symbol"]])
    dist = Path(task["dist"])
    items = task["items"]
    keep_merge_cache = task["keep_merge_cache"]

    if dist.exists():
        return {"status": "skipped", "path": str(dist)}

    s3 = get_s3_client()
    if len(items) == 1:
        item = items[0]
        tmp = download(s3, item["bucket"], item["name"], Path(DOWNLOAD_TMP_DIR))
        os.rename(tmp, dist)
        return {"status": "downloaded", "path": str(dist)}

    if len(items) != 2:
        raise ValueError(f"{dist.name}: expected 1 or 2 source files, got {len(items)}")

    merge_paths = []
    for item in items:
        local_name = item["name"][:-4] if item["name"].endswith(".bz2") else item["name"]
        tmp = Path(MERGE_TMP_DIR) / local_name
        merge_paths.append(tmp)
        if not tmp.exists():
            dl = download(s3, item["bucket"], item["name"], Path(DOWNLOAD_TMP_DIR))
            os.rename(dl, tmp)

    merged = _merge_two_sources(merge_paths, market)
    with open(dist, "wb") as f:
        f.write(b"".join(msgpack.packb(row, use_bin_type=True) for row in merged))
    if not keep_merge_cache:
        for path in merge_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return {"status": "merged", "path": str(dist), "rows": len(merged)}


def _process_market(
    s3,
    market: Market,
    target_dates: list[str] | None,
    stats: dict,
    workers: int,
    keep_merge_cache: bool,
) -> None:
    label = f"{market.exchange_name.name}:{market.symbol.name}"
    LOG.info(f"Processing {label}")
    files = _list_market_files(s3, market, target_dates)

    tasks = []
    for name, items in files.items():
        dist = Path(OUTPUT_DIR) / f"{name}.msgpack"
        if dist.exists():
            LOG.debug(f"Skip (exists): {dist.name}")
            stats["skipped"] += 1
            continue
        tasks.append(
            {
                "exchange": market.exchange_name.name,
                "symbol": market.symbol.name,
                "dist": str(dist),
                "items": [{"bucket": item.bucket, "name": item.name} for item in items],
                "keep_merge_cache": keep_merge_cache,
            }
        )

    if not tasks:
        LOG.info(f"{label}: all target files already exist")
        return

    LOG.info(f"{label}: processing {len(tasks)} missing file(s) with {workers} worker(s)")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(_download_or_merge_one, task): task for task in tasks}
        for idx, future in enumerate(as_completed(future_to_task), 1):
            task = future_to_task[future]
            try:
                result = future.result()
                status = result["status"]
                stats[status] += 1
                if status == "merged":
                    LOG.debug(f"Merged: {Path(result['path']).name} ({result['rows']} rows)")
                elif status == "downloaded":
                    LOG.debug(f"Downloaded: {Path(result['path']).name}")
            except Exception as e:
                LOG.error(f"Error processing {task['dist']}: {e}")
                stats["errors"] += 1

            if idx % 10 == 0 or idx == len(tasks):
                LOG.info(f"{label}: {idx}/{len(tasks)} missing file(s) processed")


def _parse_cli_date(value: str) -> date:
    if "-" in value:
        return date.fromisoformat(value)
    return datetime.strptime(value, "%Y%m%d").date()


def _target_dates(args) -> list[str]:
    if args.start or args.end:
        if not args.start or not args.end:
            raise ValueError("--start and --end must be provided together")
        start = _parse_cli_date(args.start)
        end = _parse_cli_date(args.end)
        if end < start:
            raise ValueError("--end must be >= --start")
        dates = []
        cur = start
        while cur <= end:
            dates.append(cur.strftime("%Y%m%d"))
            cur += timedelta(days=1)
        return dates

    today = datetime.now()
    return [
        (today - timedelta(days=i)).strftime("%Y%m%d")
        for i in range(args.days, 0, -1)
    ]


def execute(args) -> None:
    for d in [OUTPUT_DIR, DOWNLOAD_TMP_DIR, MERGE_TMP_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if args.market:
        try:
            exch_str, sym_str = args.market.split(":")
            markets = [Market(ExchangeName[exch_str.upper()], Symbol[sym_str.upper()])]
        except (ValueError, KeyError):
            LOG.error(f"Invalid market '{args.market}'. Expected EXCHANGE:SYMBOL, e.g. GMO:BTC_JPY")
            raise
    else:
        markets = DEFAULT_MARKETS

    target_dates = _target_dates(args)
    LOG.info(f"Target dates: {target_dates}")

    s3 = get_s3_client()
    stats = {"downloaded": 0, "merged": 0, "skipped": 0, "errors": 0}

    for market in markets:
        try:
            _process_market(s3, market, target_dates, stats, args.workers, args.keep_merge_cache)
        except Exception as e:
            LOG.error(f"Error processing {market}: {e}")
            stats["errors"] += 1

    result = {"status": "done", "markets": len(markets), "dates": target_dates, **stats}
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
    p.add_argument(
        "--start",
        help="Start date YYYY-MM-DD or YYYYMMDD. Must be used with --end.",
    )
    p.add_argument(
        "--end",
        help="End date YYYY-MM-DD or YYYYMMDD. Must be used with --start.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel download/merge workers per market (default: 4).",
    )
    p.add_argument(
        "--keep-merge-cache",
        action="store_true",
        help="Keep decompressed source files in /panda/tmp_merge after successful two-source merges.",
    )
    p.set_defaults(func=execute)
