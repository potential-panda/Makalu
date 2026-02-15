import argparse
from logging import getLogger

from qate.core.model import Market
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.util.dt_range import DtRange

from pipeline import (
    FEATURE_DATA_BUCKET,
    AlignAndBuildFeaturesStep,
    PnlLabelStep,
    QueryBookChangeStep,
    QueryOrderBookStateStep,
    QueryTradeAggressionStep,
)
from pipeline.framework import Pipeline
from pipeline.stages import FeatureStage, LabelStage

from .pipeline.market_pair import LEAD_LAG_PAIRS, find_pair_key

LOG = getLogger(__name__)
ENV_NAME = EnvName.LUIGI_MARIO


def create_feature_stage(market: Market) -> FeatureStage:
    influx_reader = sys_env.get_influxdb_reader()
    influx_writer = sys_env.get_influxdb_writer(bucket_name=FEATURE_DATA_BUCKET, no_retention=True)
    return (
        FeatureStage(influx_writer, market, influx_reader)
        .step(QueryTradeAggressionStep())
        .step(QueryOrderBookStateStep())
        .step(QueryBookChangeStep())
        .step(AlignAndBuildFeaturesStep())
    )


def create_label_stage(market: Market, pnl_bucket: str) -> LabelStage:
    influx_reader = sys_env.get_influxdb_reader()
    # Write label data to ${pnl_bucket}_LabelStage bucket
    label_output_bucket = f"{pnl_bucket}_LabelStage"
    influx_writer = sys_env.get_influxdb_writer(bucket_name=label_output_bucket, no_retention=True)
    return LabelStage(influx_writer, market, influx_reader, pnl_bucket).step(
        PnlLabelStep(
            ema_windows=[5, 10, 20],
            rolling_windows=[10, 20, 50],
        )
    )


def create_pipeline(pair_key: str, run_stages: list[str], pnl_bucket: str) -> Pipeline:
    lead_market, lag_market = LEAD_LAG_PAIRS.get(pair_key)
    influx_reader = sys_env.get_influxdb_reader()
    feature_writer = sys_env.get_influxdb_writer(bucket_name=FEATURE_DATA_BUCKET, no_retention=True)

    pipeline = Pipeline()
    if "Feature" in run_stages:
        pipeline.add_stage(
            FeatureStage(feature_writer, lead_market, influx_reader)
            .step(QueryTradeAggressionStep())
            .step(QueryOrderBookStateStep())
            .step(QueryBookChangeStep())
            .step(AlignAndBuildFeaturesStep())
        )
    if pnl_bucket and "Label" in run_stages:
        label_writer = sys_env.get_influxdb_writer(bucket_name=f"{pnl_bucket}_Label", no_retention=True)
        pipeline.add_stage(
            LabelStage(label_writer, lag_market, influx_reader, pnl_bucket).step(
                PnlLabelStep(
                    ema_windows=[5, 10, 20],
                    rolling_windows=[10, 20, 50],
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
        pipeline = create_pipeline(pair_key, args.stages, args.pnl_bucket)
        LOG.info(f"Starting ML pipeline from {args.from_dt} to {args.to_dt}")
        pipeline.process(dt_range)
        LOG.info("ML pipeline finished.")
    finally:
        env.leave()


def main():
    parser = argparse.ArgumentParser(description="ML pipeline CLI for building features and labels from market data.")
    parser.add_argument("--env", required=True, choices=EnvName._member_names_, help="Environment name")
    parser.add_argument("--market", required=True, help="Market in format EXCHANGE:SYMBOL (e.g., BINANCE:BTC_USDT)")
    parser.add_argument("--from-dt", required=True, help="Start date in YYYYMMDD format")
    parser.add_argument("--to-dt", required=True, help="End date in YYYYMMDD format")
    parser.add_argument(
        "--pnl-bucket",
        required=False,
        help="InfluxDB bucket containing PNL data from backtest. If provided, LabelStage will be executed.",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        type=str,
        default=["Feature", "Label"],
        help="Optional list of stages",
    )
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
