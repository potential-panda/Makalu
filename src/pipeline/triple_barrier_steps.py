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
from qate.core.model import Market

from triple_barrier.triple_barrier import TripleBarrierConfig, generate_labels

from .converter import DataPointConverter
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
        df = stage_context.execution_price_df

        LOG.info(
            f"GenerateLabelsStep: Generating labels with config: "
            f"barrier={self.config.barrier_multiplier}x, timeout={self.config.timeout_sec}s, "
            f"return_shift={self.config.return_shift_sec}s"
        )

        result = generate_labels(df, self.config, exclude_edge=True)
        labels_df = result.labels_df[
            (result.labels_df.index >= stage_context.populate_dt_range.start_dt_utc)
            & (result.labels_df.index <= stage_context.populate_dt_range.end_dt_utc)
        ]

        LOG.info(
            f"GenerateLabelsStep: Generated {len(labels_df)} labels "
            f"(Win: {(labels_df['label'] == 1).sum()}, "
            f"Loss: {(labels_df['label'] == -1).sum()}, "
            f"Timeout: {(labels_df['label'] == 0).sum()}, "
            f"Ambiguous: {result.ambiguous_count})"
        )

        self._save_to_parquet(pipeline_context, stage_context, labels_df)
        self._save_to_influxdb(pipeline_context, stage_context, labels_df)

    def _save_to_parquet(self, pipeline_context: PipelineContext, stage_context: Context, labels_df: pd.DataFrame):
        parquet_store = pipeline_context.parquet_store.create_label(
            "TripleBarrierLabel",
            stage_context.market,
            self.config.param_id,
        )
        parquet_store.add_dataframe(labels_df.copy(), ts_column=None)
        parquet_store.close()
        LOG.info(f"GenerateLabelsStep: Saved {len(labels_df)} labels to Parquet " f"(param_id={self.config.param_id})")

    def _save_to_influxdb(self, pipeline_context: PipelineContext, stage_context: Context, labels_df: pd.DataFrame):
        influx_writer = pipeline_context.influx_store.create_writer(LABEL_DATA_BUCKET, no_retention=True)

        custom_tags = {
            "barrier_multiplier": str(self.config.barrier_multiplier),
            "timeout_sec": str(self.config.timeout_sec),
            "return_shift_sec": str(self.config.return_shift_sec),
            "exec_size": stage_context.execution_size_str,
            "param_id": self.config.param_id,
        }

        points = DataPointConverter.convert_dataframe_to_influx_points(
            labels_df,
            measurement="TRIPLE_BARRIER_LABEL",
            exchange_name=stage_context.market.exchange_name.name,
            symbol=stage_context.market.symbol.name,
            tags=custom_tags,
        )

        for point in points:
            influx_writer.add(point)
        influx_writer.close()

        LOG.info(f"GenerateLabelsStep: Wrote {len(points)} label points to InfluxDB")
