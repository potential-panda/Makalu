import argparse
from logging import getLogger

from qate.core.model import Market
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.store.influx_writer import InfluxdbStore
from qate.store.market_data_store import MarketDataStore
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

from labeling.triple_barrier import TripleBarrierConfig
from lead_lag.lead_lag import LEAD_LAG_CONFIG_MID, LEAD_LAG_CONFIG_NEARHFT
from pipeline.features import (
    BookChangeFeatureStep,
    CrossInteractionsFeatureStep,
    ExecutionPriceFeatureStep,
    LoadLeadLagStep,
    OrderBookStateFeatureStep,
    PostprocessAndSaveStep,
    TradeAggressionFeatureStep,
    TrendFeatureStep,
    VolatilityFeatureStep,
)
from pipeline.features.stage import FeatureStage
from pipeline.features.target import FeatureTarget
from pipeline.framework import Pipeline
from pipeline.lead_lag_steps import LeadLagStep
from pipeline.market_pair import EXECUTION_SIZE, LEAD_LAG_PAIRS, find_pair_key
from pipeline.order_book_derived_steps import (
    PopulateBookChangeStep,
    PopulateBookStateStep,
    PopulateMarketPriceStep,
)
from pipeline.stages import (
    LabelingStage,
    LeadLagStage,
    OrderBookDerivedStage,
    TradeDerivedStage,
)
from pipeline.trade_derived_steps import (
    PopulateAggressionStep,
    PopulateCandleBarStep,
    PopulateHeikinAshiBarStep,
    PopulateRangeBarStep,
)
from pipeline.triple_barrier_steps import GenerateLabelsStep

LOG = getLogger(__name__)
ENV_NAME = EnvName.MARIO_MARIO


def create_pipeline(pair_key: str, run_stages: list[str]) -> Pipeline:
    market_lead, market_lag = LEAD_LAG_PAIRS.get(pair_key)
    market_data_store = MarketDataStore(sys_env.get_market_data_dir())
    parquet_store = ParquetStore(sys_env.get_parquet_root_dir())
    influx_store = InfluxdbStore(sys_env.get_influxdb())

    pipeline = Pipeline(market_data_store, parquet_store, influx_store)

    if "TradeDerived" in run_stages:
        pipeline.stage(
            TradeDerivedStage(market_lead).step(
                PopulateAggressionStep(
                    windows_sec=[
                        1,
                        3,
                        5,
                        # 15,
                        # 30,
                        # 60,
                        # 120,
                    ]
                )
            )  # Aggression metrics
        )

    if "TradeChart" in run_stages:
        pipeline.stage(
            TradeDerivedStage(market_lead)
            .step(PopulateCandleBarStep(interval_seconds=60))  # 1-minute candle bars
            .step(PopulateHeikinAshiBarStep(interval_seconds=60))  # 1-minute Heikin Ashi bars
            .step(PopulateRangeBarStep(box_size_rate=0.001))  # 0.1% range bars (example)
        )

    if "OrderBookDerived" in run_stages:
        pipeline.stage(
            OrderBookDerivedStage(market_lead)
            .step(PopulateMarketPriceStep(EXECUTION_SIZE.get(market_lead.symbol)))
            .step(
                PopulateBookStateStep(
                    volume_imbalance_k=[1, 3, 5, 10, 20],
                    weighted_imbalance_k=[1, 3, 5, 10, 20],
                    depth_imbalance_rate=None,
                    microprice_k=[1, 3, 5, 10, 20],
                )
            )
            .step(PopulateBookChangeStep(k_levels=[1, 3, 5, 10, 20]))
        )
        pipeline.stage(
            OrderBookDerivedStage(market_lag).step(PopulateMarketPriceStep(EXECUTION_SIZE.get(market_lag.symbol)))
        )

    if "LeadLag" in run_stages:
        pipeline.stage(
            LeadLagStage(market_lead, market_lag, EXECUTION_SIZE.get(market_lag.symbol))
            .step(LeadLagStep(config=LEAD_LAG_CONFIG_NEARHFT))
            .step(LeadLagStep(config=LEAD_LAG_CONFIG_MID))
        )

    if "Feature_" + FeatureTarget.NEAR_HFT in run_stages:
        pipeline.stage(
            FeatureStage(market_lead, EXECUTION_SIZE.get(market_lead.symbol), target=FeatureTarget.NEAR_HFT)
            .step(TradeAggressionFeatureStep(windows=["5s", "10s", "30s"], rolling_windows=["30s", "1min", "5min"]))
            .step(OrderBookStateFeatureStep(depth_levels=[1, 3, 5, 10]))
            .step(BookChangeFeatureStep(depth_levels=[1, 3, 5, 10], rolling_windows=["30s", "1min", "5min"]))
            .step(ExecutionPriceFeatureStep(rolling_windows=["30s", "1min", "5min"]))
            .step(
                CrossInteractionsFeatureStep(
                    windows=["5s", "10s", "30s"],
                    depth_levels=[1, 3, 5, 10],
                    base_roll_window="1min",
                )
            )
            .step(VolatilityFeatureStep(windows=["5s", "10s", "30s", "60s"]))
            .step(LoadLeadLagStep(market_lag, LEAD_LAG_CONFIG_NEARHFT))
            .step(PostprocessAndSaveStep())
        )

    if "Feature_" + FeatureTarget.MID in run_stages:
        pipeline.stage(
            FeatureStage(market_lead, EXECUTION_SIZE.get(market_lead.symbol), target=FeatureTarget.MID)
            .step(
                TradeAggressionFeatureStep(windows=["30s", "60s", "180s"], rolling_windows=["2min", "5min", "10min"])
            )
            .step(OrderBookStateFeatureStep(depth_levels=[5, 10, 20]))
            .step(BookChangeFeatureStep(depth_levels=[5, 10, 20], rolling_windows=["2min", "5min", "10min"]))
            .step(ExecutionPriceFeatureStep(rolling_windows=["2min", "5min", "10min"]))
            .step(
                CrossInteractionsFeatureStep(
                    windows=["30s", "60s", "180s"],
                    depth_levels=[5, 10, 20],
                    base_roll_window="5min",
                )
            )
            .step(TrendFeatureStep(windows=["30s", "60s", "180s", "600s"]))
            .step(VolatilityFeatureStep(windows=["30s", "60s", "180s", "600s"]))
            .step(LoadLeadLagStep(market_lag, LEAD_LAG_CONFIG_MID))
            .step(PostprocessAndSaveStep())
        )

    if "Label" in run_stages:
        pipeline.stage(
            LabelingStage(market_lag, EXECUTION_SIZE.get(market_lag.symbol)).step(
                GenerateLabelsStep(TripleBarrierConfig())
            )
        )

    return pipeline


def run_pipeline(args):
    env = Env.load_env(sys_env.ENV_DIR, ENV_NAME)
    env.enable_multiprocess_logging = True
    dt_range = DtRange.from_strings(args.from_dt, args.to_dt)
    market = Market.from_str(args.market)
    pair_key = find_pair_key(market)

    env.visit()
    pipeline = create_pipeline(pair_key, args.stages)
    LOG.info(f"Starting Data pipeline from {args.from_dt} to {args.to_dt}")
    pipeline.process(dt_range)
    LOG.info("Data pipeline finished.")


ALL_STEPS = [
    "TradeDerived",
    "TradeChart",
    "OrderBookDerived",
    "LeadLag",
    "Feature_" + FeatureTarget.NEAR_HFT,  # Feature_NearHFT
    "Feature_" + FeatureTarget.MID,  # Feature_MID
    "Label",
]


def main():
    parser = argparse.ArgumentParser(description="Data pipeline CLI for processing market data.")
    parser.add_argument("--market", required=True, help="Market in format EXCHANGE:SYMBOL (e.g., BINANCE:BTC_USDT)")
    parser.add_argument("--from-dt", required=True, help="Start date in YYYYMMDD format")
    parser.add_argument("--to-dt", help="End date in YYYYMMDD format")
    parser.add_argument(
        "--stages",
        nargs="*",
        type=str,
        default=ALL_STEPS,
        help="Optional list of stages",
    )
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
