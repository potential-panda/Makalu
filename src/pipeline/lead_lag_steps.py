"""
Lead-lag analysis steps for computing cross-correlation between two markets.

This module provides steps to analyze the lead-lag relationship between
price movements on different exchanges/markets.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
from influxdb_client import Point, WritePrecision
from qate.core.model import Market
from qate.util.dt_range import DtRange

from lead_lag.lead_lag import align_series, compute_cross_correlation

from .framework import PipelineContext, StageContext, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constant
LEAD_LAG_BUCKET = "lead_lag"


@dataclass
class Context(StageContext):
    market_lead: Market
    market_lag: Market
    execution_size_str: str


class LeadLagStep(Step):
    def __init__(
        self,
        window_size_sec: int,
        window_step_sec: int,
        # NOTE Below 3 args are based on data frequency of the actual market,
        # the default values are good for the markets we are trading
        max_lag: int = 10,
        interval_ms: int = 500,
        resample_interval: str = "100ms",
    ):
        super().__init__()
        self.window_size_sec = window_size_sec
        self.window_step_sec = window_step_sec
        self.max_lag = max_lag
        self.interval_ms = interval_ms
        self.resample_interval = resample_interval

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        market_lead: Market = stage_context.market_lead
        market_lag: Market = stage_context.market_lag

        LOG.info(f"LeadLagStep: Computing lead-lag between {market_lead.id} and {market_lag.id}")
        LOG.info(f"LeadLagStep: Window size={self.window_size_sec}s, step={self.window_step_sec}s")

        influx_writer = pipeline_context.influx_store.create_writer(LEAD_LAG_BUCKET, no_retention=True)
        read_store_lead = pipeline_context.parquet_store.create_execution_price(
            market_lead,
            stage_context.execution_size_str,
        )
        read_store_lag = pipeline_context.parquet_store.create_execution_price(
            market_lag,
            stage_context.execution_size_str,
        )
        write_store_lead_lag = pipeline_context.parquet_store.create_lead_lag(market_lead, market_lag)

        populate_dt_range = stage_context.populate_dt_range
        compute_dt_range = DtRange(
            populate_dt_range.start_dt - timedelta(seconds=self.window_size_sec * 1.5),
            populate_dt_range.end_dt,
        )

        df_lead = read_store_lead.read(compute_dt_range)
        df_lag = read_store_lag.read(compute_dt_range)
        LOG.info(f"LeadLagStep: Loaded {len(df_lead)} prices for {market_lead.id}, {len(df_lag)} for {market_lag.id}")

        if df_lead.empty or df_lag.empty:
            LOG.warning("LeadLagStep: No data available for one or both markets")
            return

        # Convert ts column to DatetimeIndex for resampling
        df_lead.index = pd.to_datetime(df_lead["ts"], unit="s")
        df_lag.index = pd.to_datetime(df_lag["ts"], unit="s")

        # Drop duplicate timestamps, keeping the last value (most recent state)
        df_lead = df_lead[~df_lead.index.duplicated(keep="last")]
        df_lag = df_lag[~df_lag.index.duplicated(keep="last")]

        # Resample to fixed intervals using last value, then forward-fill gaps
        df_lead_resampled = df_lead.resample(self.resample_interval).last().ffill().dropna()
        df_lag_resampled = df_lag.resample(self.resample_interval).last().ffill().dropna()

        LOG.info(
            f"LeadLagStep: After resampling: {len(df_lead_resampled)} for {market_lead.id},"
            f" {len(df_lag_resampled)} for {market_lag.id}"
        )

        # Align series with forward-fill (like Study/lead_lag.py)
        timestamps, aligned_lead, aligned_lag = align_series(df_lead_resampled, df_lag_resampled)

        if len(aligned_lead) == 0:
            LOG.warning("LeadLagStep: No overlapping timestamps between markets")
            return

        LOG.info(f"LeadLagStep: Aligned series length: {len(aligned_lead)}")

        # Compute lead-lag for each rolling window
        window_size_steps = self.window_size_sec * 1000 // self.interval_ms
        window_step_steps = self.window_step_sec * 1000 // self.interval_ms

        progress = StepProgressLogger(self.__class__.__name__, len(aligned_lead) // window_size_steps)
        start_idx = 0
        while start_idx + window_size_steps <= len(aligned_lead):
            end_idx = start_idx + window_size_steps

            # Extract window data
            window_lead = aligned_lead[start_idx:end_idx]
            window_lag = aligned_lag[start_idx:end_idx]
            window_end_ts = timestamps[end_idx - 1].timestamp()  # Convert to Unix timestamp

            if not populate_dt_range.is_within(window_end_ts):
                start_idx += window_step_steps
                continue

            # Compute cross-correlation for this window
            best_lag, best_corr, corr_by_lag = compute_cross_correlation(window_lead, window_lag, self.max_lag)
            LOG.info(f"{best_lag}, {best_corr}, {corr_by_lag}")
            if best_lag is None:
                start_idx += window_step_steps
                continue

            best_lag_ms = int(best_lag * self.interval_ms)
            fields = {
                "best_lag": best_lag_ms,
                "best_corr": best_corr,
                "lag_0": corr_by_lag[0],
                f"lag_{int(self.interval_ms * 1)}": corr_by_lag[1],
                f"lag_{int(self.interval_ms * 2)}": corr_by_lag[2],
            }

            point = (
                Point("LEAD_LAG")
                .tag("market_lead", market_lead.id)
                .tag("market_lag", market_lag.id)
                .time(int(window_end_ts * 1000), WritePrecision.NS)
            )
            for k, v in fields.items():
                point.field(k, v)

            influx_writer.add(point)
            write_store_lead_lag.add(window_end_ts, fields)
            progress.update()
            start_idx += window_step_steps

        influx_writer.close()
        write_store_lead_lag.close()
        LOG.info(f"LeadLagStep: Processed {progress.cnt} objects")
