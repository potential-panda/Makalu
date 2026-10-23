import argparse
from logging import getLogger

from qate.core.model import Market
from makalu import settings as sys_env
from qate_env.env import Env
from qate_env.env_name import EnvName
from makalu.storage.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

from k2_data.store import MarketDataStore
from pipeline.framework import Pipeline
from pipeline.market_data_steps import PopulateOrderBookStep, PopulateTradeStep
from pipeline.stages import MarketDataStage

LOG = getLogger(__name__)
ENV_NAME = EnvName("LUIGI_MARIO")


def main(market: Market, dt_range: DtRange, populate=False):
    market_data_store = MarketDataStore(sys_env.get_market_data_dir())
    parquet_store = ParquetStore(sys_env.get_parquet_root_dir())

    if populate:
        LOG.info(f"Starting Data pipeline from {dt_range.start_str} to {dt_range.end_str}")

        Pipeline(market_data_store, parquet_store).stage(
            MarketDataStage(market).step(PopulateTradeStep()).step(PopulateOrderBookStep())
        ).process(dt_range)

        LOG.info("Data pipeline finished.")
    else:
        store = parquet_store.create_trade(market)
        result = store.data_check(dt_range)
        print("Trade")
        for k, v in result.items():
            print("  ", k, v)

        store = parquet_store.create_order_book(market)
        result = store.data_check(dt_range)
        print("OrderBook")
        for k, v in result.items():
            print("  ", k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data pipeline CLI for processing market data.")
    parser.add_argument("--market", required=True, help="Market in format EXCHANGE:SYMBOL (e.g., BINANCE:BTC_USDT)")
    parser.add_argument("--from-dt", required=True, help="Start date in YYYYMMDD format")
    parser.add_argument("--to-dt", help="End date in YYYYMMDD format")
    parser.add_argument(
        "--populate",
        action="store_true",
        default=False,
        help="Run the populate pipeline (default: data_check only)",
    )
    args = parser.parse_args()

    market = Market.from_str(args.market)
    dt_range = DtRange.from_strings(args.from_dt, args.to_dt)

    env = Env(sys_env.get_env_root_dir(), ENV_NAME)
    env.enable_multiprocess_logging = True
    env.visit()
    main(market, dt_range, populate=args.populate)
