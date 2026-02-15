"""
Triple-Barrier Labeling Pipeline Steps.

This module implements pipeline steps for generating triple-barrier labels
using the volatility-based dynamic barrier method.

Usage:
    1. Use barrier_analysis.ipynb to find appropriate parameters
    2. Configure TripleBarrierConfig with chosen parameters
    3. Add TripleBarrierStage to pipeline with QueryDataStep and GenerateLabelsStep
    4. Run pipeline to populate labels in Parquet store
    5. Read labels in ML notebooks using parquet_store.create_label()
"""

import logging
from dataclasses import dataclass

import pandas as pd
from influxdb_client import Point
from qate.core.model import Market

from labeling.triple_barrier import (
    TripleBarrierConfig,
    calculate_volatility_and_spread,
    compute_labels,
    filter_by_vol_spread_ratio,
    ts_index,
)

from .framework import PipelineContext, StageContext, Step

LOG = logging.getLogger(__name__)

# InfluxDB bucket for label data
LABEL_DATA_BUCKET = "label_data"
EPS = 1e-9


@dataclass
class Context(StageContext):
    market: Market
    execution_size_str: str
    execution_price_df: pd.DataFrame = None


class GenerateLabelsStep(Step):
    """Generates triple-barrier labels and saves to Parquet and InfluxDB.

    Uses the execution price data loaded by QueryDataStep and the configuration
    from the stage context to generate labels. Labels are saved to:
    - Parquet store: For ML training data loading
    - InfluxDB: For visualization in Grafana
    """

    def __init__(self, config: TripleBarrierConfig):
        super().__init__()
        self.config = config

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        df = self._populate(pipeline_context, stage_context)
        if df is None or df.empty:
            return
        parquet_store = pipeline_context.parquet_store.create_label(
            "TripleBarrierLabel",
            stage_context.market,
            self.config.param_id,
        )
        parquet_store.add_dataframe(df.copy(), ts_column=None)
        parquet_store.close()
        LOG.info(f"GenerateLabelsStep: Saved {len(df)} labels to Parquet (param_id={self.config.param_id})")

    def _populate(self, pipeline_context: PipelineContext, stage_context: Context):
        df = stage_context.execution_price_df

        LOG.info(
            f"GenerateLabelsStep: Generating labels with config: "
            f"barrier={self.config.barrier_multiplier}x, timeout={self.config.timeout_sec}s, "
            f"return_shift={self.config.return_shift_sec}s"
        )

        # Pre-process data to get volatility and spread
        df = ts_index(df)
        df = calculate_volatility_and_spread(df, self.config)
        LOG.info(f"GenerateLabelsStep: {len(df)} points after volatility calculation")

        # Compute labels
        result = compute_labels(df, self.config)

        # Exclude edge effect (last timeout_sec of data)
        labels_df = result.labels_df.copy()
        df["label"] = labels_df["label"].reindex(df.index, method="ffill")
        df["pnl"] = labels_df["pnl"].reindex(df.index, method="ffill")

        # Filter to populate range
        df = df[
            (df.index >= stage_context.populate_dt_range.start_dt_utc)
            & (df.index <= stage_context.populate_dt_range.end_dt_utc)
        ]

        LOG.info(
            f"GenerateLabelsStep: Generated {len(df)} labels "
            f"(Win: {(df['label'] == 1).sum()}, "
            f"Loss: {(df['label'] == -1).sum()}, "
            f"Timeout: {(df['label'] == 0).sum()}, "
            f"Ambiguous: {result.ambiguous_count})"
        )

        return df

    def populate_influxdb(self, pipeline_context: PipelineContext, stage_context: Context):
        df = self._populate(pipeline_context, stage_context)
        if df is None or df.empty:
            return

        influx_writer = pipeline_context.influx_store.create_writer(LABEL_DATA_BUCKET, no_retention=True)

        exchange_name = stage_context.market.exchange_name.name
        symbol = stage_context.market.symbol.name

        point_count = 0
        for ts, row in df.iterrows():
            point = (
                Point("TRIPLE_BARRIER_LABEL")
                .tag("exchange_name", exchange_name)
                .tag("symbol", symbol)
                .tag("exec_size", stage_context.execution_size_str)
                .tag("param_id", self.config.param_id)
                .field("label", int(row["label"]))
                .field("volatility_ema", float(row["volatility_ema"]))
                .field("spread_pct_ema", float(row["spread_pct_ema"]))
                .time(ts)
            )
            influx_writer.add(point)
            point_count += 1
        LOG.info(f"GenerateLabelsStep: Wrote {point_count} label points to InfluxDB")

        df = filter_by_vol_spread_ratio(df, self.config)
        point_count = 0
        for ts, row in df.iterrows():
            point = (
                Point("TRIPLE_BARRIER_PNL")
                .tag("exchange_name", exchange_name)
                .tag("symbol", symbol)
                .tag("exec_size", stage_context.execution_size_str)
                .tag("param_id", self.config.param_id)
                .tag("label", int(row["label"]))
                .field("pnl", float(row["pnl"]))
                .time(ts)
            )
            influx_writer.add(point)
            point_count += 1
        LOG.info(f"GenerateLabelsStep: Wrote {point_count} PnL points to InfluxDB")

        influx_writer.close()
