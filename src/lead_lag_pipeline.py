"""
Lead-lag pipeline CLI for computing cross-correlation between two markets.

This pipeline analyzes the lead-lag relationship between price movements
on different exchanges/markets using market price data from the market_data bucket.

Usage:
    python lead_lag_pipeline.py --env DEV \
        --market-a BINANCE:BTC_USDT \
        --market-b GMO:BTC_JPY \
        --from-dt 20250601 \
        --to-dt 20250602
"""

import argparse
import logging

from qate.core.model import Market
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.util.dt_range import DtRange

from pipeline.framework import Pipeline
from pipeline.lead_lag_steps import LEAD_LAG_BUCKET, LeadLagStep
from pipeline.stages import LeadLagStage

logging.basicConfig(level=logging.INFO, format="{asctime} {levelname:8s} {name} {message}", style="{")


def create_lead_lag_stage(
    market_a: Market,
    market_b: Market,
    interval_ms: int,
    max_lag: int,
) -> LeadLagStage:
    """
    Create a LeadLagStage with the specified markets and parameters.

    Args:
        market_a: First market (potential leader)
        market_b: Second market (potential follower)
        interval_ms: Aggregation interval in milliseconds
        max_lag: Maximum lag to test in steps

    Returns:
        Configured LeadLagStage
    """
    influx_reader = sys_env.get_influxdb_reader()
    influx_writer = sys_env.get_influxdb_writer(bucket_name=LEAD_LAG_BUCKET, no_retention=True)

    # Convert interval_ms to pandas resample string
    if interval_ms >= 1000:
        resample_interval = f"{interval_ms // 1000}s"
    else:
        resample_interval = f"{interval_ms}ms"

    return LeadLagStage(influx_writer, market_a, market_b, influx_reader).step(
        LeadLagStep(
            interval_ms=interval_ms,
            max_lag=max_lag,
            resample_interval=resample_interval,
        )
    )


def run_pipeline(args):
    """Run the lead-lag pipeline with the specified arguments."""
    env_name = EnvName[args.env]
    env = Env.load_env(sys_env.ENV_DIR, env_name)

    market_a = Market.from_str(args.market_a)
    market_b = Market.from_str(args.market_b)
    dt_range = DtRange.from_strings(args.from_dt, args.to_dt)

    try:
        env.enter()

        pipeline = Pipeline()
        pipeline.add_stage(
            create_lead_lag_stage(
                market_a,
                market_b,
                args.interval_ms,
                args.max_lag,
            )
        )

        logging.info(
            f"Starting lead-lag pipeline: {market_a.id} vs {market_b.id} "
            f"from {args.from_dt} to {args.to_dt}"
        )
        logging.info(f"Parameters: interval={args.interval_ms}ms, max_lag={args.max_lag} steps")

        pipeline.process(dt_range, args={})

        logging.info("Lead-lag pipeline finished.")
    finally:
        env.leave()


def main():
    parser = argparse.ArgumentParser(
        description="Lead-lag pipeline CLI for computing cross-correlation between two markets."
    )
    parser.add_argument(
        "--env",
        required=True,
        choices=EnvName._member_names_,
        help="Environment name",
    )
    parser.add_argument(
        "--market-a",
        required=True,
        help="Market A (potential leader) in format EXCHANGE:SYMBOL (e.g., BINANCE:BTC_USDT)",
    )
    parser.add_argument(
        "--market-b",
        required=True,
        help="Market B (potential follower) in format EXCHANGE:SYMBOL (e.g., GMO:BTC_JPY)",
    )
    parser.add_argument(
        "--from-dt",
        required=True,
        help="Start date in YYYYMMDD format",
    )
    parser.add_argument(
        "--to-dt",
        required=True,
        help="End date in YYYYMMDD format",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=1000,
        help="Aggregation interval in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=60,
        help="Maximum lag to test in steps (default: 60)",
    )

    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
