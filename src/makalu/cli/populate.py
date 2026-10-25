import argparse
import sys
from logging import getLogger

from qate.core.model import Market
from qate.util.dt_range import DtRange

import makalu.settings as sys_env
from makalu.storage.parquet_store import ParquetStore

LOG = getLogger(__name__)


class PopulateParquetStore(ParquetStore):
    """ParquetStore variant used by populate commands.

    The populate contract is common across all datasets: each store knows the
    exact target data.parquet path at flush time. Unless --recreate is set, an
    existing target file is left untouched and only that write is skipped.
    """

    def __init__(self, root_dir: str, recreate: bool = False):
        super().__init__(root_dir)
        self.recreate = recreate

    def _m(self, store):
        store = super()._m(store)
        original_flush = store.flush

        def flush_without_overwrite():
            if not store._buffer:
                return
            parquet_path = store._get_parquet_path(store._current_time_key)
            if parquet_path.exists() and not self.recreate:
                LOG.info("Skip existing parquet: %s", parquet_path)
                store._buffer.clear()
                return
            original_flush()

        store.flush = flush_without_overwrite
        return store


def _create_parquet_store(args) -> ParquetStore:
    if args.recreate:
        LOG.info("--recreate set; existing parquet files may be overwritten.")
    return PopulateParquetStore(sys_env.PARQUET_ROOT_DIR, recreate=args.recreate)


def execute_single_market(args):
    from pipeline.framework import Pipeline
    from pipeline.order_book_derived_steps import (
        PopulateBookChangeStep,
        PopulateBookStateStep,
    )
    from pipeline.stages import OrderBookDerivedStage, TradeDerivedStage
    from pipeline.trade_derived_steps import PopulateAggressionStep

    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)
    pipeline = Pipeline(_create_parquet_store(args))

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
    else:
        LOG.error(f"Unknown dataset for populate: {dataset}")
        sys.exit(1)

    LOG.info(f"Populating {dataset}: {market.id}  {args.start} → {args.end}")
    pipeline.process(dt_range)
    LOG.info("Done.")


def _volspread_configs():
    # Single source of truth for both execute_volspread (what's actually
    # populated) and execute_activity_curve (what gets read back to build
    # the curve) -- kept in sync with src/silver.py's own VolSpreadStep list.
    from feature_engineering.volatility import VolSpreadConfig

    return [
        VolSpreadConfig(60, 300, 120),
        VolSpreadConfig(60, 900, 300),
        VolSpreadConfig(180, 900, 300),
        VolSpreadConfig(300, 900, 300),
    ]


def execute_volspread(args):
    from pipeline.execution_price_derived_steps import VolSpreadStep
    from pipeline.framework import Pipeline
    from pipeline.stages import ExecutionPriceDerivedStage

    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)

    pipeline = Pipeline(_create_parquet_store(args))
    stage = ExecutionPriceDerivedStage(market, args.execution_size)
    for config in _volspread_configs():
        stage = stage.step(VolSpreadStep(config))
    pipeline.stage(stage)

    LOG.info(
        f"Populating VolSpread: {market.id} (execution_size={args.execution_size}) {args.start} → {args.end}"
    )
    pipeline.process(dt_range)
    LOG.info("Done.")


def execute_activity_curve(args):
    from makalu.storage.parquet_store import Layer
    from pipeline.curve_steps import build_activity_curve_rows

    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)
    param_ids = [c.param_id for c in _volspread_configs()]
    parquet_store = _create_parquet_store(args)

    LOG.info(
        f"Building ActivityCurve: {market.id} (execution_size={args.execution_size}) trained on {args.start} → {args.end}"
    )
    rows = build_activity_curve_rows(
        parquet_store, market, args.execution_size, param_ids, dt_range, args.bin_size
    )

    n_days = len(dt_range.days)
    param_id = f"bin{args.bin_size}s_z{n_days}d_es{args.execution_size}"
    store = parquet_store.create_single_market(
        "ActivityCurve", market, layer=Layer.GOLD, param_id=param_id
    )
    # Partitioned by the as-of date the window itself describes (dt_range's
    # end), not by whenever this command happens to run -- otherwise
    # backfilling many historical windows in one sitting (exactly what
    # learning the right Z refresh cadence needs) would dump every result
    # into a single today's-date partition instead of one per as-of date.
    as_of_ts = dt_range.end_ts
    for row in rows:
        store.add(as_of_ts, row)
    store.close()
    LOG.info(f"Done. param_id={param_id}")


def execute_return(args):
    from pipeline.framework import Pipeline
    from pipeline.return_steps import ReturnConfig, ReturnStep
    from pipeline.stages import ReturnDerivedStage

    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)
    horizons_sec = tuple(int(h) for h in args.horizons.split(","))
    config = ReturnConfig(
        horizons_sec=horizons_sec, execution_size_str=args.execution_size
    )

    pipeline = Pipeline(_create_parquet_store(args))
    pipeline.stage(
        ReturnDerivedStage(
            market, args.execution_size, max_horizon_sec=max(horizons_sec)
        ).step(ReturnStep(config))
    )

    LOG.info(
        f"Populating Return: {market.id} (execution_size={args.execution_size}, horizons={horizons_sec}) {args.start} → {args.end}"
    )
    pipeline.process(dt_range)
    LOG.info("Done.")


def _resolve_leadlag_markets(args) -> tuple[Market, Market, str]:
    """Shared --pair / --lead-market+--lag-market resolution for both
    execute_leadlag and execute_leadlag_curve.

    --pair is the original shorthand for the 4 known BINANCE->GMO-margin
    relationships LEAD_LAG_PAIRS encodes -- that dict is also load-bearing
    for silver.py's feature pipeline and training/config.py's ML training
    config, so it's deliberately left untouched. --lead-market/--lag-market
    is a separate, explicit path for any other pair (e.g. BINANCE->GMO
    spot, or same-exchange pairs), added because Kilimanjaro needed
    lead-lag for markets LEAD_LAG_PAIRS doesn't cover and never will.
    """
    from pipeline.market_pair import EXECUTION_SIZE, LEAD_LAG_PAIRS

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
    return market_lead, market_lag, execution_size


def execute_chart_range(args):
    from pipeline.framework import Pipeline
    from pipeline.stages import TradeDerivedStage
    from pipeline.trade_derived_steps import PopulateRangeBarStep

    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)

    pipeline = Pipeline(_create_parquet_store(args))
    pipeline.stage(
        TradeDerivedStage(market).step(
            PopulateRangeBarStep(box_size_rate=args.box_size_rate)
        )
    )

    LOG.info(
        f"Populating Chart_RANGE: {market.id} (box_size_rate={args.box_size_rate}) {args.start} → {args.end}"
    )
    pipeline.process(dt_range)
    LOG.info("Done.")


def execute_leadlag(args):
    from lead_lag.lead_lag import LEAD_LAG_CONFIG_MID, LEAD_LAG_CONFIG_NEARHFT
    from pipeline.framework import Pipeline
    from pipeline.lead_lag_steps import LeadLagStep
    from pipeline.stages import LeadLagStage

    market_lead, market_lag, execution_size = _resolve_leadlag_markets(args)
    dt_range = DtRange.from_strings(args.start, args.end)

    pipeline = Pipeline(_create_parquet_store(args))
    pipeline.stage(
        LeadLagStage(market_lead, market_lag, execution_size)
        .step(LeadLagStep(config=LEAD_LAG_CONFIG_NEARHFT))
        .step(LeadLagStep(config=LEAD_LAG_CONFIG_MID))
    )

    LOG.info(
        f"Populating LeadLag: {market_lead.id} → {market_lag.id} {args.start} → {args.end}"
    )
    pipeline.process(dt_range)
    LOG.info("Done.")


def execute_leadlag_curve(args):
    from lead_lag.lead_lag import LEAD_LAG_CONFIG_MID, LEAD_LAG_CONFIG_NEARHFT
    from makalu.storage.parquet_store import Layer
    from pipeline.curve_steps import build_leadlag_curve_rows

    market_lead, market_lag, _execution_size = _resolve_leadlag_markets(args)
    dt_range = DtRange.from_strings(args.start, args.end)
    param_ids = [LEAD_LAG_CONFIG_NEARHFT.param_id(), LEAD_LAG_CONFIG_MID.param_id()]
    parquet_store = _create_parquet_store(args)

    LOG.info(
        f"Building LeadLagCurve: {market_lead.id} → {market_lag.id} trained on {args.start} → {args.end}"
    )
    rows = build_leadlag_curve_rows(
        parquet_store, market_lead, market_lag, param_ids, dt_range, args.bin_size
    )

    n_days = len(dt_range.days)
    param_id = f"bin{args.bin_size}s_z{n_days}d"
    store = parquet_store.create_cross_market(
        "LeadLagCurve", market_lead, market_lag, layer=Layer.GOLD, param_id=param_id
    )
    # See execute_activity_curve's comment -- same as-of-date partitioning
    # reasoning applies here.
    as_of_ts = dt_range.end_ts
    for row in rows:
        store.add(as_of_ts, row)
    store.close()
    LOG.info(f"Done. param_id={param_id}")


_DATASET_META = {
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
    "Chart_RANGE": {
        "help": "Populate range bars (price-triggered, not time-triggered) from Trade (gold layer)",
        "description": (
            "Each bar closes when price moves by --box-size-rate (as a fraction of the bar's\n"
            "open price) from the bar's open -- not on a fixed time interval. param_id encodes\n"
            "the rate as rate_<value>, e.g. --box-size-rate 0.0025 -> rate_250.\n\n"
            "Prerequisite: bronze (Trade) must already be populated for this market."
        ),
        "example": "python cli.py populate Chart_RANGE --exchange GMO --symbol BTC_JPY "
        "--box-size-rate 0.0015 --start 2026-01-01 --end 2026-01-07",
    },
    "ActivityCurve": {
        "help": "Populate a weekly (day-of-week x time-of-day) volatility+spread activity curve from VolSpread (gold layer)",
        "description": (
            "Combines VolSpread's volatility_ema AND spread_pct_ema columns (z-score normalized\n"
            "per config, then averaged) into one weekly mean+std curve, binned by\n"
            "(day_of_week, time_of_day // --bin-size). --start/--end is the TRAINING window\n"
            "(how many trailing days of VolSpread data the curve is built from); the curve\n"
            "is written under a date= partition equal to --end -- the as-of date this window\n"
            "describes, NOT the day the command happens to run -- so backfilling many\n"
            "historical windows in one sitting produces one correctly-dated snapshot per\n"
            "window instead of colliding on a single today's-date partition.\n\n"
            "Prerequisite: VolSpread must already be populated for the same market, execution_size,\n"
            "and the same --start/--end window."
        ),
        "example": "python cli.py populate ActivityCurve --exchange GMO --symbol BTC_JPY "
        "--start 2026-01-01 --end 2026-01-14 --execution-size 0.002",
    },
    "LeadLagCurve": {
        "help": "Populate a weekly (day-of-week x time-of-day) lead-lag-strength curve from LeadLag (gold layer)",
        "description": (
            "Same weekly mean+std curve as ActivityCurve, but built from LeadLag's ll_max_corr*\n"
            "columns (absolute correlation strength) instead of VolSpread. --start/--end is the\n"
            "TRAINING window; the curve is written under a date= partition equal to --end (see\n"
            "ActivityCurve's help for why this isn't the day the command runs).\n\n"
            "Prerequisite: LeadLag must already be populated for the same market pair and the\n"
            "same --start/--end window."
        ),
        "example": "python cli.py populate LeadLagCurve --pair btc --start 2026-01-01 --end 2026-01-14",
    },
}


def _register_dataset_single_market(
    sub: argparse.ArgumentParser, dataset: str
) -> argparse.ArgumentParser:
    meta = _DATASET_META[dataset]
    p = sub.add_parser(
        dataset,
        help=meta["help"],
        description=meta["description"],
        epilog=f"Example:\n  {meta['example']}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--exchange", required=True, help="Exchange name, e.g. GMO, BINANCE")
    p.add_argument(
        "--symbol", required=True, help="Symbol name, e.g. BTC_JPY, BTC_USDT"
    )
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD (defaults to --start if omitted)")
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Overwrite existing target data.parquet files instead of skipping them.",
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
    p.add_argument(
        "--pair",
        default=None,
        help="Asset pair key: btc, eth, xrp, sol (mutually exclusive with --lead-market/--lag-market)",
    )
    p.add_argument(
        "--lead-market",
        default=None,
        help="Explicit lead market as EXCHANGE:SYMBOL, e.g. BINANCE:BTC_USDT -- for any pair --pair doesn't cover",
    )
    p.add_argument(
        "--lag-market",
        default=None,
        help="Explicit lag market as EXCHANGE:SYMBOL, e.g. GMO:BTC_SPOT",
    )
    p.add_argument(
        "--execution-size",
        dest="execution_size",
        default=None,
        help="Overrides pipeline.market_pair.EXECUTION_SIZE's lookup for the lag market, if needed",
    )
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD (defaults to --start if omitted)")
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Overwrite existing target data.parquet files instead of skipping them.",
    )
    return p


def _register_chart_range(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p = _register_dataset_single_market(sub, "Chart_RANGE")
    p.add_argument(
        "--box-size-rate",
        type=float,
        required=True,
        dest="box_size_rate",
        help="Fraction of a bar's open price that triggers closing it, e.g. 0.0015.",
    )
    return p


def _register_activity_curve(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p = _register_dataset_single_market(sub, "ActivityCurve")
    p.add_argument(
        "--execution-size",
        required=True,
        dest="execution_size",
        help="Execution size matching the VolSpread partition to read from, e.g. 0.002.",
    )
    p.add_argument(
        "--bin-size",
        type=int,
        default=300,
        dest="bin_size",
        help="Curve time bin width in seconds; must evenly divide 86400 (default: 300).",
    )
    return p


def _register_leadlag_curve(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
    meta = _DATASET_META["LeadLagCurve"]
    p = sub.add_parser(
        "LeadLagCurve",
        help=meta["help"],
        description=meta["description"],
        epilog=f"Example:\n  {meta['example']}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pair",
        default=None,
        help="Asset pair key: btc, eth, xrp, sol (mutually exclusive with --lead-market/--lag-market)",
    )
    p.add_argument(
        "--lead-market",
        default=None,
        help="Explicit lead market as EXCHANGE:SYMBOL, e.g. BINANCE:BTC_USDT",
    )
    p.add_argument(
        "--lag-market",
        default=None,
        help="Explicit lag market as EXCHANGE:SYMBOL, e.g. GMO:BTC_SPOT",
    )
    p.add_argument(
        "--execution-size",
        dest="execution_size",
        default=None,
        help="Overrides pipeline.market_pair.EXECUTION_SIZE's lookup for the lag market, if needed",
    )
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD (defaults to --start if omitted)")
    p.add_argument(
        "--bin-size",
        type=int,
        default=300,
        dest="bin_size",
        help="Curve time bin width in seconds; must evenly divide 86400 (default: 300).",
    )
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Overwrite existing target data.parquet files instead of skipping them.",
    )
    return p


def register_cli(parser: argparse.ArgumentParser):
    p = parser.add_parser(
        "populate",
        help="Derive silver/gold parquet from bronze (idempotent)",
        description=(
            "Derive silver/gold parquet datasets from bronze. Bronze (Trade, OrderBook)\n"
            "is produced by Kitakamakura and read through Komachi; Makalu never writes it.\n"
            "Silver MarketPrice is derived by Hase, not here:\n"
            "  hase derive MarketPrice --market GMO:BTC_JPY --execution-size 0.002 --start ... --end ...\n\n"
            "All commands are idempotent: if a target data.parquet already exists, that write is skipped.\n"
            "Use --recreate to overwrite existing target files.\n"
            "Later dates do not affect earlier ones.\n"
            "Run 'python cli.py scan' after populating to update the manifest.\n\n"
            "Data flow:\n"
            "  bronze → MarketPrice (hase) → VolSpread → ActivityCurve\n"
            "  bronze → MarketPrice (hase) → Return\n"
            "  bronze → BookState / BookChange / TradeAggression\n"
            "  bronze (Trade) → Chart_RANGE / Chart_CANDLE / Chart_HEIKIN_ASHI\n"
            "  MarketPrice (both markets) → LeadLag → LeadLagCurve"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = p.add_subparsers(dest="dataset", required=True)
    _register_dataset_single_market(sub, "BookChange").set_defaults(
        func=execute_single_market
    )
    _register_dataset_single_market(sub, "BookState").set_defaults(
        func=execute_single_market
    )
    _register_dataset_single_market(sub, "TradeAggression").set_defaults(
        func=execute_single_market
    )
    _register_volspread(sub).set_defaults(func=execute_volspread)
    _register_return(sub).set_defaults(func=execute_return)
    _register_leadlag(sub).set_defaults(func=execute_leadlag)
    _register_chart_range(sub).set_defaults(func=execute_chart_range)
    _register_activity_curve(sub).set_defaults(func=execute_activity_curve)
    _register_leadlag_curve(sub).set_defaults(func=execute_leadlag_curve)
