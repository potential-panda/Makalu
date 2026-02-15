import argparse
from logging import getLogger

from qate.core.model import Market
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.store.influx_writer import InfluxdbStore
from qate.store.market_data_store import MarketDataStore
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

from pipeline.framework import Pipeline
from pipeline.market_data_steps import PopulateOrderBookStep, PopulateTradeStep
from pipeline.stages import MarketDataStage

LOG = getLogger(__name__)
ENV_NAME = EnvName.LUIGI_MARIO


def main(market: Market, dt_range: DtRange):
    market_data_store = MarketDataStore(sys_env.get_market_data_dir())
    parquet_store = ParquetStore(sys_env.get_parquet_root_dir())
    influx_store = InfluxdbStore(sys_env.get_influxdb())

    LOG.info(f"Starting Data pipeline from {dt_range.start_str} to {dt_range.end_str}")

    Pipeline(market_data_store, parquet_store, influx_store).stage(
        MarketDataStage(market).step(PopulateTradeStep()).step(PopulateOrderBookStep())
    ).process(dt_range)

    LOG.info("Data pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data pipeline CLI for processing market data.")
    parser.add_argument("--market", required=True, help="Market in format EXCHANGE:SYMBOL (e.g., BINANCE:BTC_USDT)")
    parser.add_argument("--from-dt", required=True, help="Start date in YYYYMMDD format")
    parser.add_argument("--to-dt", help="End date in YYYYMMDD format")
    args = parser.parse_args()

    market = Market.from_str(args.market)
    dt_range = DtRange.from_strings(args.from_dt, args.to_dt)

    env = Env.load_env(sys_env.ENV_DIR, ENV_NAME)
    env.enable_multiprocess_logging = True
    env.visit()
    main(market, dt_range)
