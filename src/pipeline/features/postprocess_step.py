import logging

import numpy as np
import pandas as pd

from pipeline.framework import PipelineContext, Step

from .context import Context

LOG = logging.getLogger(__name__)


class PostprocessAndSaveStep(Step):
    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        if stage_context.source_df is None or not stage_context.features:
            LOG.warning("PostprocessAndSaveStep: No source data or features available")
            return

        # Create DataFrame from features dict in one operation (avoids fragmentation)
        feature_df = pd.DataFrame(stage_context.features, index=stage_context.source_df.index)

        LOG.info(f"PostprocessAndSaveStep: Built {len(feature_df.columns)} features, " f"{len(feature_df)} rows")

        # Post-processing: replace inf with NaN
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        # Drop rows with too many NaN values (keep rows with at least 30% valid data)
        threshold = int(len(feature_df.columns) * 0.3)
        feature_df = feature_df.dropna(thresh=threshold)

        # Forward-fill remaining NaN values (with limit)
        with pd.option_context("future.no_silent_downcasting", True):
            feature_df = feature_df.ffill(limit=3)
        feature_df = feature_df.infer_objects(copy=False)

        LOG.info(
            f"PostprocessAndSaveStep: After post-processing: "
            f"{len(feature_df)} rows, {len(feature_df.columns)} columns"
        )

        # Filter to populate range
        feature_df = feature_df[
            (feature_df.index >= stage_context.populate_dt_range.start_dt_utc)
            & (feature_df.index <= stage_context.populate_dt_range.end_dt_utc)
        ]

        if feature_df.empty:
            LOG.warning("PostprocessAndSaveStep: No data after filtering to populate range")
            return

        LOG.info(f"PostprocessAndSaveStep: Feature columns: {sorted(feature_df.columns.tolist())}")

        # Save to parquet store
        parquet_store = pipeline_context.parquet_store.create_feature(stage_context.market, stage_context.target)
        parquet_store.add_dataframe(feature_df, ts_column=None)
        parquet_store.close()
        LOG.info(
            f"PostprocessAndSaveStep: Saved {len(feature_df)} rows "
            f"with {len(feature_df.columns)} columns to Parquet store."
        )

    def populate_influxdb(self, pipeline_context: PipelineContext, stage_context: Context):
        pass
        pass
