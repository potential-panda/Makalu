import argparse
from logging import getLogger

from qate.core.model import Market
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.store.market_data_store import MarketDataStore
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

from feature_engineering.volatility import VolSpreadConfig
from labeling.triple_barrier import TripleBarrierConfig
from lead_lag.lead_lag import LEAD_LAG_CONFIG_MID, LEAD_LAG_CONFIG_NEARHFT, LEAD_LAG_CONFIG_FAR
from pipeline.execution_price_derived_steps import VolSpreadStep
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
    ExecutionPriceDerivedStage,
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

    pipeline = Pipeline(market_data_store, parquet_store)

    if "TradeDerived" in run_stages:
        pipeline.stage(TradeDerivedStage(market_lead).step(PopulateAggressionStep(windows_sec=[1, 3, 5, 10, 30, 60])))

    if "TradeChart" in run_stages:
        pipeline.stage(
            TradeDerivedStage(market_lead)
            .step(PopulateCandleBarStep(interval_seconds=60))  # 1-minute candle bars
            .step(PopulateHeikinAshiBarStep(interval_seconds=60))  # 1-minute Heikin Ashi bars
            .step(PopulateRangeBarStep(box_size_rate=0.001))  # 0.1% range bars (example)
        )

    if "ExecutionPrice" in run_stages:
        pipeline.stage(
            OrderBookDerivedStage(market_lead).step(PopulateMarketPriceStep(EXECUTION_SIZE.get(market_lead.symbol)))
        )
        pipeline.stage(
            OrderBookDerivedStage(market_lag).step(PopulateMarketPriceStep(EXECUTION_SIZE.get(market_lag.symbol)))
        )

    if "ExectuionPriceDerived" in run_stages:
        # Param policy:
        # - Keep return_shift_sec responsive for frequent trading.
        # - Require lookback_sec >= return_shift_sec so volatility is estimated on a sensible horizon.
        # - Keep ema_span_sec shorter than lookback_sec so spread smoothing does not become too sluggish.

        # pipeline.stage(
        #     ExecutionPriceDerivedStage(market_lead, EXECUTION_SIZE.get(market_lead.symbol))
        #     .step(VolSpreadStep(VolSpreadConfig(60, 300, 120)))
        #     .step(VolSpreadStep(VolSpreadConfig(60, 900, 300)))
        #     .step(VolSpreadStep(VolSpreadConfig(180, 900, 300)))
        #     .step(VolSpreadStep(VolSpreadConfig(300, 900, 300)))
        # )
        pipeline.stage(
            ExecutionPriceDerivedStage(market_lag, EXECUTION_SIZE.get(market_lag.symbol))
            .step(VolSpreadStep(VolSpreadConfig(60, 300, 120)))
            .step(VolSpreadStep(VolSpreadConfig(60, 900, 300)))
            .step(VolSpreadStep(VolSpreadConfig(180, 900, 300)))
            .step(VolSpreadStep(VolSpreadConfig(300, 900, 300)))
        )

    if "OrderBookDerived" in run_stages:
        pipeline.stage(
            OrderBookDerivedStage(market_lead)
            .step(
                PopulateBookStateStep(
                    volume_imbalance_k=[1, 3, 5, 10],
                    weighted_imbalance_k=[1, 3, 5, 10],
                    depth_imbalance_rate=[0.0001, 0.0002, 0.0005, 0.001],
                    microprice_k=[1, 3, 5, 10],
                )
            )
            .step(PopulateBookChangeStep(k_levels=[1, 3, 5, 10]))
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
            .step(TradeAggressionFeatureStep(windows=["5s", "10s", "30s"], rolling_windows=["30s", "60s", "300s"]))
            .step(OrderBookStateFeatureStep(depth_levels=[1, 3, 5, 10], windows=["5s", "10s", "30s"]))
            .step(BookChangeFeatureStep(depth_levels=[1, 3, 5, 10], rolling_windows=["30s", "60s", "300s"]))
            .step(ExecutionPriceFeatureStep(rolling_windows=["30s", "60s", "300s"]))
            .step(
                CrossInteractionsFeatureStep(
                    windows=["5s", "10s", "30s"],
                    depth_levels=[1, 3, 5, 10],
                    base_roll_window="60s",
                )
            )
            # .step(VolatilityFeatureStep(windows=["5s", "10s", "30s", "60s"]))
            .step(LoadLeadLagStep(market_lag, LEAD_LAG_CONFIG_NEARHFT))
            .step(PostprocessAndSaveStep())
        )

    if "Feature_" + FeatureTarget.MID in run_stages:
        pipeline.stage(
            FeatureStage(market_lead, EXECUTION_SIZE.get(market_lead.symbol), target=FeatureTarget.MID)
            .step(TradeAggressionFeatureStep(windows=["30s", "60s", "180s"], rolling_windows=["120s", "300s", "600s"]))
            .step(OrderBookStateFeatureStep(depth_levels=[5, 10, 20], windows=["30s", "60s", "180s"]))
            .step(BookChangeFeatureStep(depth_levels=[5, 10, 20], rolling_windows=["120s", "300s", "600s"]))
            .step(ExecutionPriceFeatureStep(rolling_windows=["120s", "300s", "600s"]))
            .step(
                CrossInteractionsFeatureStep(
                    windows=["30s", "60s", "180s"],
                    depth_levels=[5, 10, 20],
                    base_roll_window="300s",
                )
            )
            .step(TrendFeatureStep(windows=["30s", "60s", "180s", "600s"]))
            # .step(VolatilityFeatureStep(windows=["30s", "60s", "180s", "600s"]))
            .step(LoadLeadLagStep(market_lag, LEAD_LAG_CONFIG_MID))
            .step(PostprocessAndSaveStep())
        )

    if "Feature_" + FeatureTarget.ORION in run_stages:
        # Orion is a meta-regime filter operating on range-bar close timestamps.
        # Windows are chosen to cover the 5m–2h regime-detection horizon from the spec.
        pipeline.stage(
            FeatureStage(market_lead, EXECUTION_SIZE.get(market_lead.symbol), target=FeatureTarget.ORION)
            .step(TradeAggressionFeatureStep(windows=["300s", "1800s"], rolling_windows=["300s", "1800s", "7200s"]))
            .step(OrderBookStateFeatureStep(depth_levels=[5, 10, 20], windows=["300s", "1800s"]))
            .step(BookChangeFeatureStep(depth_levels=[5, 10, 20], rolling_windows=["300s", "1800s", "7200s"]))
            .step(ExecutionPriceFeatureStep(rolling_windows=["300s", "1800s", "7200s"]))
            .step(VolatilityFeatureStep(windows=["300s", "1800s", "7200s"]))
            .step(TrendFeatureStep(windows=["300s", "1800s", "7200s"]))
            .step(LoadLeadLagStep(market_lag, LEAD_LAG_CONFIG_FAR))
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
    "Feature_" + FeatureTarget.ORION,  # Feature_Orion
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
