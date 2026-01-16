import argparse
from logging import getLogger

from qate.core.model import Market
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.util.dt_range import DtRange

from pipeline.framework import Pipeline
from pipeline.market_pair import LEAD_LAG_PAIRS, TARGET_SIZE, find_pair_key
from pipeline.stages import LeadLagStage, MarketDataStage, TradeDerivedStage
from pipeline.steps import (
    LEAD_LAG_BUCKET,
    MARKET_DATA_BUCKET,
    TRADE_DERIVED_BUCKET,
    LeadLagStep,
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

LOG = getLogger(__name__)
ENV_NAME = EnvName.MARIO_MARIO


def create_pipeline(pair_key: str, run_stages: list[str]) -> Pipeline:
    lead_market, lag_market = LEAD_LAG_PAIRS.get(pair_key)
    data_dir = sys_env.get_market_data_dir()
    influx_reader = sys_env.get_influxdb_reader()
    market_writer = sys_env.get_influxdb_writer(bucket_name=MARKET_DATA_BUCKET, no_retention=True)
    derived_writer = sys_env.get_influxdb_writer(bucket_name=TRADE_DERIVED_BUCKET, no_retention=True)
    lead_lag_writer = sys_env.get_influxdb_writer(bucket_name=LEAD_LAG_BUCKET, no_retention=True)

    pipeline = Pipeline()
    if "MarketData" in run_stages:
        pipeline.add_stage(
            MarketDataStage(market_writer, lead_market, data_dir)
            .step(PopulateTradeStep())
            .step(PopulateMarketPriceStep(target_size=TARGET_SIZE.get(lead_market.symbol)))
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

        pipeline.add_stage(
            MarketDataStage(market_writer, lag_market, data_dir)
            .step(PopulateTradeStep())
            .step(PopulateMarketPriceStep(target_size=TARGET_SIZE.get(lag_market.symbol)))
        )

    if "TradeDerived" in run_stages:
        pipeline.add_stage(
            TradeDerivedStage(derived_writer, lead_market, influx_reader)
            .step(PopulateCandleBarStep(interval_seconds=60))  # 1-minute candle bars
            .step(PopulateHeikinAshiBarStep(interval_seconds=60))  # 1-minute Heikin Ashi bars
            .step(PopulateRangeBarStep(box_size_rate=0.001))  # 0.1% range bars (example)
            .step(PopulateAggressionStep(windows_sec=[1, 3, 5]))  # Aggression metrics
        )

    if "LeadLag" in run_stages:
        pipeline.add_stage(
            LeadLagStage(lead_lag_writer, lead_market, lag_market, influx_reader).step(
                LeadLagStep(
                    interval_ms=1000,
                    max_lag=30,
                    resample_interval="1s",
                    window_size_sec=300,
                    window_step_sec=60,
                )
            )
        )

    return pipeline


def run_pipeline(args):
    env = Env.load_env(sys_env.ENV_DIR, ENV_NAME)
    dt_range = DtRange.from_strings(args.from_dt, args.to_dt)
    market = Market.from_str(args.market)
    pair_key = find_pair_key(market)
    try:
        env.enter()
        pipeline = create_pipeline(pair_key, args.stages)
        LOG.info(f"Starting Data pipeline from {args.from_dt} to {args.to_dt}")
        pipeline.process(dt_range)
        LOG.info("Data pipeline finished.")
    finally:
        env.leave()


def main():
    parser = argparse.ArgumentParser(description="Data pipeline CLI for processing market data.")
    parser.add_argument("--market", required=True, help="Market in format EXCHANGE:SYMBOL (e.g., BINANCE:BTC_USDT)")
    parser.add_argument("--from-dt", required=True, help="Start date in YYYYMMDD format")
    parser.add_argument("--to-dt", help="End date in YYYYMMDD format")
    parser.add_argument(
        "--stages",
        nargs="*",
        type=str,
        default=["MarketData", "TradeDerived", "LeadLag"],
        help="Optional list of stages",
    )
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
