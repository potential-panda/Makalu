import logging

import numpy as np
import pandas as pd

from pipeline.framework import PipelineContext, Step

from .context import Context

LOG = logging.getLogger(__name__)


def _is_binary(series: pd.Series) -> bool:
    vals = series.dropna()
    if vals.empty:
        return False
    uniq = set(vals.unique().tolist())
    return uniq.issubset({0, 1, 0.0, 1.0, False, True})


def _normalize_for_ml(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, leakage-free normalization for ML-ready feature ranges.

    Policy:
    - Binary columns: keep as-is.
    - Already bounded [0, 1] or [-1, 1]: clip to range.
    - Other numeric columns: monotonic signed-log + tanh squash to [-1, 1].
    """
    out = feature_df.copy()
    transformed_cols = 0

    for col in out.columns:
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if _is_binary(s):
            continue

        vals = s.dropna()
        if vals.empty:
            continue

        col_min = vals.min()
        col_max = vals.max()

        if col_min >= 0.0 and col_max <= 1.0:
            out[col] = s.clip(0.0, 1.0)
            continue
        if col_min >= -1.0 and col_max <= 1.0:
            out[col] = s.clip(-1.0, 1.0)
            continue

        # Monotonic compression for unbounded values.
        out[col] = np.tanh(np.sign(s) * np.log1p(np.abs(s)))
        transformed_cols += 1

    LOG.info(f"PostprocessAndSaveStep: Normalized {transformed_cols} unbounded numeric features")
    return out


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
        feature_df = _normalize_for_ml(feature_df)

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
