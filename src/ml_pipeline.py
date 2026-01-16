import argparse
import logging

from qate.core.model import Market
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.util.dt_range import DtRange

from pipeline.framework import Pipeline
from pipeline.stages import FeatureStage, LabelStage
from pipeline.steps import (
    FEATURE_BUCKET,
    AlignAndBuildFeaturesStep,
    PnlLabelStep,
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


def create_label_stage(market: Market, pnl_bucket: str) -> LabelStage:
    influx_reader = sys_env.get_influxdb_reader()
    # LabelStage doesn't write to InfluxDB, but framework requires a writer
    # Use a dummy writer or the feature bucket
    influx_writer = sys_env.get_influxdb_writer(bucket_name=FEATURE_BUCKET, no_retention=True)
    return LabelStage(influx_writer, market, influx_reader, pnl_bucket).step(
        PnlLabelStep(
            ema_windows=[5, 10, 20],
            rolling_windows=[10, 20, 50],
        )
    )


def run_pipeline(args):
    env_name = EnvName[args.env]
    env = Env.load_env(sys_env.ENV_DIR, env_name)

    market = Market.from_str(args.market)
    dt_range = DtRange.from_strings(args.from_dt, args.to_dt)

    try:
        env.enter()

        pipeline = Pipeline()
        pipeline.add_stage(create_feature_stage(market))

        # Add LabelStage only if pnl_bucket is specified
        if args.pnl_bucket:
            pipeline.add_stage(create_label_stage(market, args.pnl_bucket))

        logging.info(f"Starting ML pipeline for {market.id} from {args.from_dt} to {args.to_dt}")
        pipeline.process(dt_range, args={"feature_output_dir": args.feature_output_dir})
        logging.info("ML pipeline finished.")
    finally:
        env.leave()


def main():
    parser = argparse.ArgumentParser(description="ML pipeline CLI for building features and labels from market data.")
    parser.add_argument("--env", required=True, choices=EnvName._member_names_, help="Environment name")
    parser.add_argument("--market", required=True, help="Market in format EXCHANGE:SYMBOL (e.g., BINANCE:BTC_USDT)")
    parser.add_argument("--from-dt", required=True, help="Start date in YYYYMMDD format")
    parser.add_argument("--to-dt", required=True, help="End date in YYYYMMDD format")
    parser.add_argument("--feature-output-dir", default="/tmp", help="Output directory for feature parquet files")
    parser.add_argument(
        "--pnl-bucket",
        required=False,
        help="InfluxDB bucket containing PNL data from backtest. If provided, LabelStage will be executed.",
    )
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
