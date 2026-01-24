import argparse
from logging import getLogger

from qate.core.model import Market
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.store.influx_writer import InfluxdbStore
from qate.store.market_data_store import MarketDataStore
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

from pipeline.feature_steps import AlignAndBuildFeaturesStep, QueryDataStep
from pipeline.framework import Pipeline
from pipeline.lead_lag_steps import LeadLagStep
from pipeline.market_data_steps import (
    PopulateBookStateStep,
    PopulateMarketPriceStep,
    PopulateOrderBookStep,
    PopulateTradeStep,
)
from pipeline.market_pair import EXECUTION_SIZE, LEAD_LAG_PAIRS, find_pair_key
from pipeline.order_book_derived_steps import PopulateBookChangeStep
from pipeline.stages import (
    FeatureStage,
    LeadLagStage,
    MarketDataStage,
    OrderBookDerivedStage,
    TradeDerivedStage,
)
from pipeline.trade_derived_steps import (
    PopulateAggressionStep,
    PopulateCandleBarStep,
    PopulateHeikinAshiBarStep,
    PopulateRangeBarStep,
)

LOG = getLogger(__name__)
ENV_NAME = EnvName.MARIO_MARIO


def create_pipeline(pair_key: str, run_stages: list[str]) -> Pipeline:
    market_lead, market_lag = LEAD_LAG_PAIRS.get(pair_key)
    market_data_store = MarketDataStore(sys_env.get_market_data_dir())
    parquet_store = ParquetStore(sys_env.get_parquet_root_dir())
    influx_store = InfluxdbStore(sys_env.get_influxdb())

    pipeline = Pipeline(market_data_store, parquet_store, influx_store)
    if "MarketData" in run_stages:
        pipeline.stage(
            MarketDataStage(market_lead)
            .step(PopulateTradeStep())
            .step(PopulateOrderBookStep())
            .step(PopulateMarketPriceStep(EXECUTION_SIZE.get(market_lead.symbol)))
            .step(
                PopulateBookStateStep(
                    volume_imbalance_k=[1, 3, 5, 10],
                    weighted_imbalance_k=[1, 3, 5, 10],
                    depth_imbalance_rate=[0.0001, 0.0002, 0.0005, 0.001],
                    microprice_k=[1, 3, 5, 10],
                )
            )
        ).stage(
            MarketDataStage(market_lag)
            .step(PopulateTradeStep())
            .step(PopulateOrderBookStep())
            .step(PopulateMarketPriceStep(EXECUTION_SIZE.get(market_lag.symbol)))
        )

    if "TradeDerived" in run_stages:
        pipeline.stage(
            TradeDerivedStage(market_lead)
            .step(PopulateCandleBarStep(interval_seconds=60))  # 1-minute candle bars
            .step(PopulateHeikinAshiBarStep(interval_seconds=60))  # 1-minute Heikin Ashi bars
            .step(PopulateRangeBarStep(box_size_rate=0.001))  # 0.1% range bars (example)
            .step(PopulateAggressionStep(windows_sec=[1, 3, 5]))  # Aggression metrics
        )

    if "OrderBookDerived" in run_stages:
        pipeline.stage(OrderBookDerivedStage(market_lead).step(PopulateBookChangeStep(k_levels=[3, 5, 10])))

    if "LeadLag" in run_stages:
        pipeline.stage(
            LeadLagStage(market_lead, market_lag, EXECUTION_SIZE.get(market_lag.symbol))
            .step(LeadLagStep(window_size_sec=3600 * 2, window_step_sec=3600))
            .step(LeadLagStep(window_size_sec=3600 * 4, window_step_sec=3600))
        )

    if "Feature" in run_stages:
        pipeline.stage(FeatureStage(market_lead).step(QueryDataStep()).step(AlignAndBuildFeaturesStep()))

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
        default=["MarketData", "TradeDerived", "OrderBookDerived", "LeadLag"],
        help="Optional list of stages",
    )
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
