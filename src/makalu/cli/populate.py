import argparse
import sys
from logging import getLogger

import qate.env.sys_env as sys_env
from qate.core.model import Market
from qate.store.market_data_store import MarketDataStore, partition_market_events
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange, TimeFormat

LOG = getLogger(__name__)


def execute_single_market(args):
    from pipeline.framework import Pipeline
    from pipeline.order_book_derived_steps import (
        PopulateBookChangeStep,
        PopulateBookStateStep,
        PopulateMarketPriceStep,
    )
    from pipeline.stages import OrderBookDerivedStage, TradeDerivedStage
    from pipeline.trade_derived_steps import PopulateAggressionStep

    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)

    market_data_store = MarketDataStore(sys_env.MARKET_DATA_DIR)
    parquet_store = ParquetStore(sys_env.PARQUET_ROOT_DIR)
    pipeline = Pipeline(market_data_store, parquet_store)

    dataset = args.dataset
    if dataset == "TradeAggression":
        pipeline.stage(
            TradeDerivedStage(market).step(
                PopulateAggressionStep(windows_sec=[1, 3, 5, 10, 30, 60])
            )
        )
    elif dataset == "BookState":
        pipeline.stage(
            OrderBookDerivedStage(market).step(
                PopulateBookStateStep(
                    volume_imbalance_k=[1, 3, 5, 10],
                    weighted_imbalance_k=[1, 3, 5, 10],
                    depth_imbalance_rate=[0.0001, 0.0002, 0.0005, 0.001],
                    microprice_k=[1, 3, 5, 10],
                )
            )
        )
    elif dataset == "BookChange":
        pipeline.stage(
            OrderBookDerivedStage(market).step(
                PopulateBookChangeStep(k_levels=[1, 3, 5, 10])
            )
        )
    elif dataset == "MarketPrice":
        pipeline.stage(
            OrderBookDerivedStage(market).step(
                PopulateMarketPriceStep(args.execution_size)
            )
        )
    else:
        LOG.error(f"Unknown dataset for populate: {dataset}")
        sys.exit(1)

    LOG.info(f"Populating {dataset}: {market.id}  {args.start} → {args.end}")
    pipeline.process(dt_range)
    LOG.info("Done.")


def execute_bronze(args):
    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)

    LOG.info(f"Populating bronze: {market.id}  {args.start} → {args.end}")
    LOG.info(f"  Source : {sys_env.MARKET_DATA_DIR}")
    LOG.info(f"  Output : {sys_env.PARQUET_ROOT_DIR}")

    loader = MarketDataStore(sys_env.MARKET_DATA_DIR).create_loader(
        market,
        event_types=None,  # load both trades and order books
    )
    dt_range = DtRange(dt_range._start_dt, dt_range._end_dt, TimeFormat.YYYYMMDD)
    all_data = loader.batch_load_mt(dt_range, max_workers=8)
    order_books, trades = partition_market_events(all_data)
    LOG.info(f"  Loaded : {len(trades)} trades, {len(order_books)} order books")

    parquet_store = ParquetStore(sys_env.PARQUET_ROOT_DIR)

    trade_store = parquet_store.create_trade(market)
    for trade in trades:
        trade_store.add_trade(trade)
    trade_store.close()
    LOG.info(f"  Wrote  : {len(trades)} trades")

    ob_store = parquet_store.create_order_book(market)
    for ob in order_books:
        ob_store.add_order_book(ob)
    ob_store.close()
    LOG.info(f"  Wrote  : {len(order_books)} order books")

    LOG.info("Done.")


_DATASET_META = {
    "bronze": {
        "help": "Populate Trade + OrderBook (always together, sourced from raw msgpack files)",
        "description": (
            "Reads raw msgpack files from /panda/market-data/ and writes two parquet datasets:\n"
            "  Trade     — one row per exchange trade (price, size, side)\n"
            "  OrderBook — full 20-level order book snapshots\n\n"
            "Source file pattern: {MARKET_DATA_DIR}/{exchange}_{symbol}_{YYYYMMDD}.msgpack\n"
            "This is always the FIRST step. All silver datasets depend on this."
        ),
        "example": "python cli.py populate bronze --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07",
    },
    "TradeAggression": {
        "help": "Populate rolling trade flow metrics derived from Trade (silver layer)",
        "description": (
            "Computes rolling buy/sell volume and trade flow imbalance metrics\n"
            "attached to each trade event. Windows: 1s, 3s, 5s, 10s, 30s, 60s.\n\n"
            "Prerequisite: bronze must be populated for the same market and date range."
        ),
        "example": "python cli.py populate TradeAggression --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07",
    },
    "BookState": {
        "help": "Populate order book microstructure state derived from OrderBook (silver layer)",
        "description": (
            "Computes best bid/ask, spread, volume imbalance, weighted imbalance,\n"
            "depth imbalance, and microprice at depth levels 1, 3, 5, 10.\n\n"
            "Prerequisite: bronze must be populated for the same market and date range."
        ),
        "example": "python cli.py populate BookState --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07",
    },
    "BookChange": {
        "help": "Populate order book diff metrics derived from OrderBook (silver layer)",
        "description": (
            "Computes depth deltas, pull rates, and refill rates between consecutive\n"
            "order book snapshots at depth levels 1, 3, 5, 10.\n\n"
            "Prerequisite: bronze must be populated for the same market and date range."
        ),
        "example": "python cli.py populate BookChange --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07",
    },
    "MarketPrice": {
        "help": "Populate effective execution prices by walking the order book (silver layer)",
        "description": (
            "Walks the order book by a fixed execution_size to compute effective\n"
            "bid, ask, mid, spread, and spread_bps at each book snapshot.\n"
            "execution_size is stored as a partition key — use the same value consistently.\n\n"
            "Prerequisite: bronze must be populated for the same market and date range.\n"
            "MarketPrice is itself a prerequisite for VolSpread (gold layer)."
        ),
        "example": "python cli.py populate MarketPrice --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07 --execution-size 0.002",
    },
}


def _register_dataset_single_market(sub: argparse.ArgumentParser, dataset: str) -> argparse.ArgumentParser:
    meta = _DATASET_META[dataset]
    p = sub.add_parser(
        dataset,
        help=meta["help"],
        description=meta["description"],
        epilog=f"Example:\n  {meta['example']}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--exchange", required=True, help="Exchange name, e.g. GMO, BINANCE")
    p.add_argument("--symbol", required=True, help="Symbol name, e.g. BTC_JPY, BTC_USDT")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD (defaults to --start if omitted)")
    return p


def _register_market_price(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p = _register_dataset_single_market(sub, "MarketPrice")
    p.add_argument(
        "--execution-size",
        required=True,
        dest="execution_size",
        help="Order size used to walk the book, e.g. 0.002. Becomes a partition key — use consistently.",
    )
    return p


def register_cli(parser: argparse.ArgumentParser):
    p = parser.add_parser(
        "populate",
        help="Write parquet data from raw sources or derived computations (idempotent)",
        description=(
            "Populate parquet datasets from raw msgpack files (bronze) or by deriving\n"
            "metrics from already-populated bronze data (silver).\n\n"
            "All commands are idempotent — safe to re-run. Later dates do not affect earlier ones.\n"
            "Run 'python cli.py scan' after populating to update the manifest."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = p.add_subparsers(dest="dataset", required=True)
    _register_dataset_single_market(sub, "bronze").set_defaults(func=execute_bronze)
    _register_dataset_single_market(sub, "BookChange").set_defaults(func=execute_single_market)
    _register_dataset_single_market(sub, "BookState").set_defaults(func=execute_single_market)
    _register_dataset_single_market(sub, "TradeAggression").set_defaults(func=execute_single_market)
    _register_market_price(sub).set_defaults(func=execute_single_market)
