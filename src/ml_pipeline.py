import argparse
import logging

from qate.core.model import Market
from qate.env import sys_env
from qate.util.dt_range import DtRange

from pipeline.framework import Pipeline
from pipeline.stages import FeatureStage
from pipeline.steps import (
    FEATURE_BUCKET,
    AlignAndBuildFeaturesStep,
    QueryBookChangeStep,
    QueryOrderBookStateStep,
    QueryTradeAggressionStep,
)

logging.basicConfig(level=logging.INFO, format="{asctime} {levelname:8s} {name} {message}", style="{")


def create_feature_stage(market: Market) -> FeatureStage:
    influx_reader = sys_env.get_influxdb_reader()
    influx_writer = sys_env.get_influxdb_writer(bucket_name=FEATURE_BUCKET, no_retention=True)
    return (
        FeatureStage(influx_writer, market, influx_reader)
        .step(QueryTradeAggressionStep())
        .step(QueryOrderBookStateStep())
        .step(QueryBookChangeStep())
        .step(AlignAndBuildFeaturesStep())
    )


def main():
    parser = argparse.ArgumentParser(description="ML pipeline CLI for building features from market data.")
    parser.add_argument("--market", required=True, help="Market in format EXCHANGE:SYMBOL (e.g., BINANCE:BTC_USDT)")
    parser.add_argument("--from-dt", required=True, help="Start date in YYYYMMDD format")
    parser.add_argument("--to-dt", required=True, help="End date in YYYYMMDD format")
    parser.add_argument("--feature-output-dir", default="/tmp", help="Output directory for feature parquet files")
    args = parser.parse_args()

    market = Market.from_str(args.market)
    dt_range = DtRange.from_strings(args.from_dt, args.to_dt)

    pipeline = Pipeline()
    pipeline.add_stage(create_feature_stage(market))

    # --- Execution ---
    logging.info(f"Starting ML pipeline for {market.id} from {args.from_dt} to {args.to_dt}")
    pipeline.process(dt_range, args={"feature_output_dir": args.feature_output_dir})
    logging.info("ML pipeline finished.")


if __name__ == "__main__":
    main()
