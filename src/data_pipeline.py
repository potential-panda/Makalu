import argparse
import logging

from qate.core.model import Market
from qate.core.symbol import Symbol
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.util.dt_range import DtRange

from pipeline.framework import Pipeline
from pipeline.stages import MarketDataStage, TradeDerivedStage
from pipeline.steps import (
    MARKET_DATA_BUCKET,
    TRADE_DERIVED_BUCKET,
    PopulateAggressionStep,
    PopulateBookChangeStep,
    PopulateCandleBarStep,
    PopulateHeikinAshiBarStep,
    PopulateMarketPriceStep,
    PopulateOrderBookStep,
    PopulateOrderImbalanceStep,
    PopulateRangeBarStep,
    PopulateTradeStep,
)

logging.basicConfig(level=logging.INFO, format="{asctime} {levelname:8s} {name} {message}", style="{")

TARGET_SIZE = {
    Symbol.BTC_JPY: 0.002,
    Symbol.XRP_JPY: 150,
}


def create_market_data_stage(market: Market) -> MarketDataStage:
    data_dir = sys_env.get_market_data_dir()
    influx_writer = sys_env.get_influxdb_writer(bucket_name=MARKET_DATA_BUCKET, no_retention=True)
    return (
        MarketDataStage(influx_writer, market, data_dir)
        .step(PopulateTradeStep())
        .step(PopulateMarketPriceStep(target_size=TARGET_SIZE.get(market.symbol, 0.01)))
        .step(PopulateOrderBookStep())
        .step(
            PopulateOrderImbalanceStep(
                volume_imbalance_k=[1, 3, 5, 10],
                weighted_imbalance_k=[1, 3, 5, 10],
                depth_imbalance_rate=[0.0001, 0.0002, 0.0005, 0.001],
                microprice_k=[1, 3, 5, 10],
            )
        )
        .step(PopulateBookChangeStep(k_levels=[3, 5, 10]))
    )


def create_trade_derived_stage(market: Market) -> TradeDerivedStage:
    influx_reader = sys_env.get_influxdb_reader()
    influx_writer = sys_env.get_influxdb_writer(bucket_name=TRADE_DERIVED_BUCKET, no_retention=True)
    return (
        TradeDerivedStage(influx_writer, market, influx_reader)
        .step(PopulateCandleBarStep(interval_seconds=60))  # 1-minute candle bars
        .step(PopulateHeikinAshiBarStep(interval_seconds=60))  # 1-minute Heikin Ashi bars
        .step(PopulateRangeBarStep(box_size_rate=0.001))  # 0.1% range bars (example)
        .step(PopulateAggressionStep(windows_sec=[1, 3, 5]))  # Aggression metrics
    )


def run_pipeline(args):
    env_name = EnvName[args.env]
    env = Env.load_env(sys_env.ENV_DIR, env_name)

    market = Market.from_str(args.market)
    dt_range = DtRange.from_strings(args.from_dt, args.to_dt)

    try:
        env.enter()

        pipeline = Pipeline()
        pipeline.add_stage(create_market_data_stage(market))
        pipeline.add_stage(create_trade_derived_stage(market))

        logging.info(f"Starting data pipeline for {market.id} from {args.from_dt} to {args.to_dt}")
        pipeline.process(dt_range)
        logging.info("Data pipeline finished.")
    finally:
        env.leave()


def main():
    parser = argparse.ArgumentParser(description="Data pipeline CLI for processing market data.")
    parser.add_argument("--env", required=True, choices=EnvName._member_names_, help="Environment name")
    parser.add_argument("--market", required=True, help="Market in format EXCHANGE:SYMBOL (e.g., BINANCE:BTC_USDT)")
    parser.add_argument("--from-dt", required=True, help="Start date in YYYYMMDD format")
    parser.add_argument("--to-dt", required=True, help="End date in YYYYMMDD format")
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
