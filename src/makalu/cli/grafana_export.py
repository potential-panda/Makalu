import argparse
import json
import sys
from logging import getLogger

import qate.env.sys_env as sys_env
from qate.core.model import Market
from qate.store.influx_store import InfluxdbStore
from qate.store.parquet_store import ParquetStore, TimeSeriesStore
from qate.util.dt_range import DtRange

LOG = getLogger(__name__)

GRAFANA_HOST = "http://localhost:3000"
EXPLORER_DASHBOARD_UID = "makalu-explorer"

_SUPPORTED_DATASETS = [
    "Trade", "OrderBook",
    "MarketPrice", "BookState", "BookChange", "TradeAggression",
    "VolSpread",
    "Chart_CANDLE", "Chart_HEIKIN_ASHI", "Chart_RANGE",
    "LeadLag",
    "BacktestPnL",
    "Feature_Mid", "Feature_NearHFT",
]


def _market(exchange: str, symbol: str) -> Market:
    return Market.from_str(f"{exchange}:{symbol}")


def _create_store(ps: ParquetStore, args) -> TimeSeriesStore:
    dataset = args.dataset
    exchange = args.exchange
    symbol = args.symbol

    if dataset == "Trade":
        return ps.create_trade(_market(exchange, symbol))
    if dataset == "OrderBook":
        return ps.create_order_book(_market(exchange, symbol))
    if dataset == "MarketPrice":
        _require(args, "execution_size", dataset)
        return ps.create_execution_price(_market(exchange, symbol), str(args.execution_size))
    if dataset == "VolSpread":
        _require(args, "execution_size", dataset)
        _require(args, "param_id", dataset)
        return ps.create_vol_spread(_market(exchange, symbol), str(args.execution_size), str(args.param_id))
    if dataset == "BacktestPnL":
        _require(args, "strategy", dataset)
        _require(args, "param_id", dataset)
        return ps.create_pnl(_market(exchange, symbol), args.strategy, str(args.param_id))
    if dataset.startswith("Feature_"):
        feature_name = dataset[len("Feature_"):]
        return ps.create_feature(_market(exchange, symbol), feature_name)
    if dataset.startswith("Chart_"):
        chart_type = dataset[len("Chart_"):]
        _require(args, "param_id", dataset)
        return ps.create_chart(chart_type, _market(exchange, symbol), str(args.param_id))
    if dataset == "LeadLag":
        _require(args, "exchange_b", dataset)
        _require(args, "symbol_b", dataset)
        market_b = _market(args.exchange_b, args.symbol_b)
        return ps.create_lead_lag(_market(exchange, symbol), market_b, str(args.param_id or ""))
    raise ValueError(f"Unsupported dataset: {dataset}")


def _require(args, attr: str, dataset: str) -> None:
    if not getattr(args, attr, None):
        print(json.dumps({"error": f"--{attr.replace('_', '-')} is required for dataset {dataset}"}))
        sys.exit(1)


def _grafana_url(bucket: str, measurement: str, start_date: str, end_date: str) -> str:
    return (
        f"{GRAFANA_HOST}/d/{EXPLORER_DASHBOARD_UID}/data-explorer"
        f"?orgId=1"
        f"&var-bucket={bucket}"
        f"&var-measurement={measurement}"
        f"&from={start_date}T00:00:00.000Z"
        f"&to={end_date}T23:59:59.999Z"
        f"&timezone=browser"
    )


def execute(args) -> None:
    try:
        parquet_store = ParquetStore(sys_env.get_parquet_root_dir())
        influx_store = InfluxdbStore(sys_env.get_influxdb())
        store = _create_store(parquet_store, args)
    except SystemExit:
        raise
    except Exception as e:
        print(json.dumps({"error": f"Setup failed: {e}"}))
        sys.exit(1)

    dt_range = DtRange.from_strings(args.start, args.end)
    try:
        store.populate_influxdb(influx_store, dt_range)
    except Exception as e:
        print(json.dumps({"error": f"Export failed: {e}"}))
        sys.exit(1)

    bucket = f"makalu_{store.dataset}"
    measurement = store.dataset
    result = {
        "bucket": bucket,
        "measurement": measurement,
        "grafana_url": _grafana_url(bucket, measurement, args.start, args.end),
    }
    LOG.info(f"grafana-export: {result['grafana_url']}")
    print(json.dumps(result))


def register_cli(parser: argparse.ArgumentParser) -> None:
    p = parser.add_parser(
        "grafana-export",
        help="Export parquet data to InfluxDB and return a Grafana URL",
        description=(
            "Push a date range of parquet data into InfluxDB so it can be\n"
            "rendered in Grafana. Prints JSON: {bucket, measurement, grafana_url}.\n\n"
            f"Supported datasets: {', '.join(_SUPPORTED_DATASETS)}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=_SUPPORTED_DATASETS, help="Dataset to export")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--exchange", help="Exchange (e.g. GMO, BINANCE)")
    p.add_argument("--symbol", help="Symbol (e.g. BTC_JPY, BTC_USDT)")
    p.add_argument("--execution-size", dest="execution_size", help="Required for MarketPrice, VolSpread (e.g. 0.002)")
    p.add_argument("--param-id", dest="param_id", help="Required for VolSpread, Chart_*, BacktestPnL")
    p.add_argument("--strategy", help="Required for BacktestPnL")
    p.add_argument("--exchange-b", dest="exchange_b", help="Second exchange for LeadLag")
    p.add_argument("--symbol-b", dest="symbol_b", help="Second symbol for LeadLag")
    p.set_defaults(func=execute)
