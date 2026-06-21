import argparse
import sys
from logging import WARNING, getLogger

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
        pipeline.stage(TradeDerivedStage(market).step(PopulateAggressionStep(windows_sec=[1, 3, 5, 10, 30, 60])))
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
        pipeline.stage(OrderBookDerivedStage(market).step(PopulateBookChangeStep(k_levels=[1, 3, 5, 10])))
    elif dataset == "MarketPrice":
        pipeline.stage(OrderBookDerivedStage(market).step(PopulateMarketPriceStep(args.execution_size)))
    else:
        LOG.error(f"Unknown dataset for populate: {dataset}")
        sys.exit(1)

    LOG.info(f"Populating {dataset}: {market.id}  {args.start} → {args.end}")
    pipeline.process(dt_range)
    LOG.info("Done.")


BRONZE_BATCH_DAYS = 3  # days loaded into memory at once before flushing to parquet


def execute_bronze(args):
    getLogger("qate").setLevel(WARNING)
    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)
    dt_range = DtRange(dt_range._start_dt, dt_range._end_dt, TimeFormat.YYYYMMDD)
    days = dt_range.days

    n_batches = (len(days) + BRONZE_BATCH_DAYS - 1) // BRONZE_BATCH_DAYS
    LOG.info(f"Populating bronze: {market.id}  {args.start} → {args.end}")
    LOG.info(f"  Source : {sys_env.MARKET_DATA_DIR}")
    LOG.info(f"  Output : {sys_env.PARQUET_ROOT_DIR}")
    LOG.info(f"  Days   : {len(days)} in {n_batches} batch(es) of {BRONZE_BATCH_DAYS}")

    parquet_store = ParquetStore(sys_env.PARQUET_ROOT_DIR)
    trade_store = parquet_store.create_trade(market)
    ob_store = parquet_store.create_order_book(market)
    total_trades = total_obs = 0

    for batch_idx, i in enumerate(range(0, len(days), BRONZE_BATCH_DAYS)):
        batch = days[i:i + BRONZE_BATCH_DAYS]
        batch_range = DtRange.from_strings(batch[0], batch[-1])
        batch_range = DtRange(batch_range._start_dt, batch_range._end_dt, TimeFormat.YYYYMMDD)

        loader = MarketDataStore(sys_env.MARKET_DATA_DIR).create_loader(market, event_types=None)
        all_data = loader.batch_load_mt(batch_range, max_workers=min(len(batch), 8))
        order_books, trades = partition_market_events(all_data)
        del all_data  # free before writing

        for trade in trades:
            trade_store.add_trade(trade)
        for ob in order_books:
            ob_store.add_order_book(ob)

        total_trades += len(trades)
        total_obs += len(order_books)
        LOG.info(f"  Batch {batch_idx + 1}/{n_batches} ({batch[0]}..{batch[-1]}): {len(trades)} trades, {len(order_books)} order books")
        del order_books, trades

    trade_store.close()
    ob_store.close()
    LOG.info(f"  Wrote  : {total_trades} trades, {total_obs} order books")
    LOG.info("Done.")


def execute_volspread(args):
    from feature_engineering.volatility import VolSpreadConfig
    from pipeline.execution_price_derived_steps import VolSpreadStep
    from pipeline.framework import Pipeline
    from pipeline.stages import ExecutionPriceDerivedStage

    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)

    pipeline = Pipeline(MarketDataStore(sys_env.MARKET_DATA_DIR), ParquetStore(sys_env.PARQUET_ROOT_DIR))
    pipeline.stage(
        ExecutionPriceDerivedStage(market, args.execution_size)
        .step(VolSpreadStep(VolSpreadConfig(60, 300, 120)))
        .step(VolSpreadStep(VolSpreadConfig(60, 900, 300)))
        .step(VolSpreadStep(VolSpreadConfig(180, 900, 300)))
        .step(VolSpreadStep(VolSpreadConfig(300, 900, 300)))
    )

    LOG.info(f"Populating VolSpread: {market.id} (execution_size={args.execution_size}) {args.start} → {args.end}")
    pipeline.process(dt_range)
    LOG.info("Done.")


def execute_return(args):
    from pipeline.framework import Pipeline
    from pipeline.return_steps import ReturnConfig, ReturnStep
    from pipeline.stages import ReturnDerivedStage

    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)
    horizons_sec = tuple(int(h) for h in args.horizons.split(","))
    config = ReturnConfig(horizons_sec=horizons_sec, execution_size_str=args.execution_size)

    pipeline = Pipeline(MarketDataStore(sys_env.MARKET_DATA_DIR), ParquetStore(sys_env.PARQUET_ROOT_DIR))
    pipeline.stage(
        ReturnDerivedStage(market, args.execution_size, max_horizon_sec=max(horizons_sec)).step(ReturnStep(config))
    )

    LOG.info(f"Populating Return: {market.id} (execution_size={args.execution_size}, horizons={horizons_sec}) {args.start} → {args.end}")
    pipeline.process(dt_range)
    LOG.info("Done.")


def execute_leadlag(args):
    from lead_lag.lead_lag import LEAD_LAG_CONFIG_MID, LEAD_LAG_CONFIG_NEARHFT
    from pipeline.framework import Pipeline
    from pipeline.lead_lag_steps import LeadLagStep
    from pipeline.market_pair import EXECUTION_SIZE, LEAD_LAG_PAIRS
    from pipeline.stages import LeadLagStage

    # --pair is the original shorthand for the 4 known BINANCE->GMO-margin
    # relationships LEAD_LAG_PAIRS encodes -- that dict is also load-bearing
    # for silver.py's feature pipeline and training/config.py's ML training
    # config, so it's deliberately left untouched. --lead-market/--lag-market
    # is a separate, explicit path for any other pair (e.g. BINANCE->GMO
    # spot, or same-exchange pairs), added because Kilimanjaro needed
    # lead-lag for markets LEAD_LAG_PAIRS doesn't cover and never will.
    if args.pair and (args.lead_market or args.lag_market):
        LOG.error("--pair is mutually exclusive with --lead-market/--lag-market.")
        sys.exit(1)
    if args.pair:
        pair = LEAD_LAG_PAIRS.get(args.pair)
        if pair is None:
            LOG.error(f"Unknown pair '{args.pair}'. Known: {list(LEAD_LAG_PAIRS)}")
            sys.exit(1)
        market_lead, market_lag = pair
    elif args.lead_market and args.lag_market:
        market_lead = Market.from_str(args.lead_market)
        market_lag = Market.from_str(args.lag_market)
    else:
        LOG.error("Pass either --pair, or both --lead-market and --lag-market.")
        sys.exit(1)

    execution_size = args.execution_size or EXECUTION_SIZE.get(market_lag.symbol)
    if execution_size is None:
        LOG.error(
            f"No execution_size found for {market_lag.id} in pipeline.market_pair.EXECUTION_SIZE "
            "-- pass --execution-size explicitly."
        )
        sys.exit(1)
    dt_range = DtRange.from_strings(args.start, args.end)

    pipeline = Pipeline(MarketDataStore(sys_env.MARKET_DATA_DIR), ParquetStore(sys_env.PARQUET_ROOT_DIR))
    pipeline.stage(
        LeadLagStage(market_lead, market_lag, execution_size)
        .step(LeadLagStep(config=LEAD_LAG_CONFIG_NEARHFT))
        .step(LeadLagStep(config=LEAD_LAG_CONFIG_MID))
    )

    LOG.info(f"Populating LeadLag: {market_lead.id} → {market_lag.id} {args.start} → {args.end}")
    pipeline.process(dt_range)
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
        "example": "python cli.py populate TradeAggression --exchange GMO --symbol BTC_JPY --start 2026-01-01 "
        "--end 2026-01-07",
    },
    "BookState": {
        "help": "Populate order book microstructure state derived from OrderBook (silver layer)",
        "description": (
            "Computes best bid/ask, spread, volume imbalance, weighted imbalance,\n"
            "depth imbalance, and microprice at depth levels 1, 3, 5, 10.\n\n"
            "Prerequisite: bronze must be populated for the same market and date range."
        ),
        "example": "python cli.py populate BookState --exchange GMO --symbol BTC_JPY --start 2026-01-01 "
        "--end 2026-01-07",
    },
    "BookChange": {
        "help": "Populate order book diff metrics derived from OrderBook (silver layer)",
        "description": (
            "Computes depth deltas, pull rates, and refill rates between consecutive\n"
            "order book snapshots at depth levels 1, 3, 5, 10.\n\n"
            "Prerequisite: bronze must be populated for the same market and date range."
        ),
        "example": "python cli.py populate BookChange --exchange GMO --symbol BTC_JPY --start 2026-01-01 "
        "--end 2026-01-07",
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
        "example": "python cli.py populate MarketPrice --exchange GMO --symbol BTC_JPY --start 2026-01-01 "
        "--end 2026-01-07 --execution-size 0.002",
    },
    "VolSpread": {
        "help": "Populate volatility and spread EMA metrics derived from MarketPrice (gold layer)",
        "description": (
            "Computes EMA of realized volatility and spread percentage from execution prices.\n"
            "Produces 4 param_id configs: (shift=60,lookback=300,ema=120), (60,900,300),\n"
            "                             (180,900,300), (300,900,300).\n\n"
            "Prerequisite: MarketPrice must be populated for the same market and execution_size."
        ),
        "example": "python cli.py populate VolSpread --exchange GMO --symbol BTC_JPY --start 2026-01-01 "
        "--end 2026-01-07 --execution-size 0.002",
    },
    "Return": {
        "help": "Populate forward mid-price returns at fixed horizons from MarketPrice (research layer)",
        "description": (
            "Computes simple forward returns of mid price at fixed horizons\n"
            "(e.g. return_30s, return_120s, return_300s), resampled to a regular grid first.\n"
            "A stable, widely-reusable quantity shared across many signal-scanning purposes,\n"
            "rather than each consumer recomputing the same forward join independently.\n\n"
            "Prerequisite: MarketPrice must be populated for the same market and execution_size,\n"
            "through at least (end date + max horizon)."
        ),
        "example": "python cli.py populate Return --exchange GMO --symbol BTC_JPY --start 2026-01-01 "
        "--end 2026-01-07 --execution-size 0.002 --horizons 30,120,300",
    },
    "LeadLag": {
        "help": "Populate lead-lag cross-correlation metrics between a BINANCE lead and GMO lag market (gold layer)",
        "description": (
            "Computes cross-correlation statistics between the lead market (BINANCE) and lag market (GMO)\n"
            "using 100ms resampled returns. Produces two configs: NearHFT and Mid.\n\n"
            "Known pairs: btc (BINANCE:BTC_USDT → GMO:BTC_JPY), eth, xrp, sol.\n\n"
            "Prerequisite: MarketPrice must be populated for both markets in the pair."
        ),
        "example": "python cli.py populate LeadLag --pair btc --start 2026-01-01 --end 2026-01-07",
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


def _register_volspread(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p = _register_dataset_single_market(sub, "VolSpread")
    p.add_argument(
        "--execution-size",
        required=True,
        dest="execution_size",
        help="Execution size matching the MarketPrice partition to read from, e.g. 0.002.",
    )
    return p


def _register_return(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p = _register_dataset_single_market(sub, "Return")
    p.add_argument(
        "--execution-size",
        required=True,
        dest="execution_size",
        help="Execution size matching the MarketPrice partition to read from, e.g. 0.002.",
    )
    p.add_argument(
        "--horizons",
        default="30,120,300",
        help="Comma-separated forward horizons in seconds, e.g. 30,120,300.",
    )
    return p


def _register_leadlag(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
    meta = _DATASET_META["LeadLag"]
    p = sub.add_parser(
        "LeadLag",
        help=meta["help"],
        description=meta["description"],
        epilog=f"Example:\n  {meta['example']}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pair", default=None, help="Asset pair key: btc, eth, xrp, sol (mutually exclusive with --lead-market/--lag-market)")
    p.add_argument("--lead-market", default=None, help="Explicit lead market as EXCHANGE:SYMBOL, e.g. BINANCE:BTC_USDT -- for any pair --pair doesn't cover")
    p.add_argument("--lag-market", default=None, help="Explicit lag market as EXCHANGE:SYMBOL, e.g. GMO:BTC_SPOT")
    p.add_argument("--execution-size", dest="execution_size", default=None, help="Overrides pipeline.market_pair.EXECUTION_SIZE's lookup for the lag market, if needed")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD (defaults to --start if omitted)")
    return p


def register_cli(parser: argparse.ArgumentParser):
    p = parser.add_parser(
        "populate",
        help="Write parquet data from raw sources or derived computations (idempotent)",
        description=(
            "Populate parquet datasets from raw msgpack files (bronze) or by deriving\n"
            "metrics from already-populated bronze data (silver/gold).\n\n"
            "All commands are idempotent — safe to re-run. Later dates do not affect earlier ones.\n"
            "Run 'python cli.py scan' after populating to update the manifest.\n\n"
            "Data flow:\n"
            "  bronze → MarketPrice → VolSpread\n"
            "  bronze → MarketPrice → Return\n"
            "  bronze → BookState / BookChange / TradeAggression\n"
            "  MarketPrice (both markets) → LeadLag"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = p.add_subparsers(dest="dataset", required=True)
    _register_dataset_single_market(sub, "bronze").set_defaults(func=execute_bronze)
    _register_dataset_single_market(sub, "BookChange").set_defaults(func=execute_single_market)
    _register_dataset_single_market(sub, "BookState").set_defaults(func=execute_single_market)
    _register_dataset_single_market(sub, "TradeAggression").set_defaults(func=execute_single_market)
    _register_market_price(sub).set_defaults(func=execute_single_market)
    _register_volspread(sub).set_defaults(func=execute_volspread)
    _register_return(sub).set_defaults(func=execute_return)
    _register_leadlag(sub).set_defaults(func=execute_leadlag)
