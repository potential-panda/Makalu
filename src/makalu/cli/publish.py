import argparse
import logging

import qate.env.sys_env as sys_env
from qate.core.model import Market
from qate.store.influx_store import InfluxdbStore, convert_to_point
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

LOG = logging.getLogger(__name__)

DEFAULT_BUCKET = "Makalu"


def _create_parquet_store() -> ParquetStore:
    return ParquetStore(sys_env.PARQUET_ROOT_DIR)


def _create_writer(bucket: str):
    return InfluxdbStore(sys_env.get_influxdb()).create_writer(bucket, buffer_size=4096)


def execute_trade(args):
    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)
    store = _create_parquet_store().create_trade(market)
    writer = _create_writer(args.influxdb_bucket)

    total = 0
    for date_str in dt_range.days:
        df = store.read_a_day(date_str)
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            writer.add(convert_to_point(store.convert_to_trade(row).to_metric_object()))
        total += len(df)
        LOG.info(f"Published Trade {market.id} {date_str}: {len(df)} points")

    writer.close()
    LOG.info(f"Done. Total Trade points published: {total}")


def execute_market_price(args):
    market = Market.from_str(f"{args.exchange}:{args.symbol}")
    dt_range = DtRange.from_strings(args.start, args.end)
    store = _create_parquet_store().create_execution_price(market, args.execution_size)
    writer = _create_writer(args.influxdb_bucket)

    total = 0
    for date_str in dt_range.days:
        df = store.read_a_day(date_str)
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            from qate.core.model import MarketPrice
            mp = MarketPrice(
                bid=row.bid,
                ask=row.ask,
                symbol=market.symbol,
                exchange_name=market.exchange_name,
                exchange_ts=row.ts,
            )
            writer.add(convert_to_point(mp.to_metric_object()))
        total += len(df)
        LOG.info(f"Published MarketPrice {market.id} {date_str}: {len(df)} points")

    writer.close()
    LOG.info(f"Done. Total MarketPrice points published: {total}")


def _add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--exchange", required=True, help="Exchange name, e.g. GMO, BINANCE")
    p.add_argument("--symbol", required=True, help="Symbol name, e.g. ETH_JPY, XRP_JPY")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD (defaults to --start if omitted)")
    p.add_argument(
        "--influxdb-bucket",
        default=DEFAULT_BUCKET,
        dest="influxdb_bucket",
        help=f"InfluxDB destination bucket (default: {DEFAULT_BUCKET})",
    )
    return p


def register_cli(parser: argparse.ArgumentParser):
    p = parser.add_parser(
        "publish",
        help="Stream parquet data to InfluxDB (reads existing parquet, skips missing dates)",
        description=(
            "Read already-populated parquet files and stream them to InfluxDB.\n"
            "Uses the same store objects as 'populate' to resolve parquet paths.\n"
            "If a date has no parquet file, it is silently skipped.\n\n"
            "Implemented datasets: Trade (bronze), MarketPrice (silver)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="dataset", required=True)

    _add_common_args(
        sub.add_parser("Trade", help="Publish bronze Trade parquet → InfluxDB TRADE measurement")
    ).set_defaults(func=execute_trade)

    mp = _add_common_args(
        sub.add_parser("MarketPrice", help="Publish silver MarketPrice parquet → InfluxDB MARKET_PRICE measurement")
    )
    mp.add_argument(
        "--execution-size",
        required=True,
        dest="execution_size",
        help="Execution size partition key used when populating, e.g. 0.1 for ETH_JPY",
    )
    mp.set_defaults(func=execute_market_price)
