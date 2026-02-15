"""
Feature engineering pipeline steps.

This module implements comprehensive feature engineering as documented in doc/features.md.
All features are designed to be stationary (normalized) for ML model compatibility.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from qate.core.model import Market

from feature_engineering.book_change import build_book_change_features
from feature_engineering.execution_price import build_execution_price_features
from feature_engineering.interactions import build_cross_feature_interactions
from feature_engineering.order_book_state import build_order_book_state_features
from feature_engineering.trade_aggression import build_trade_aggression_features
from feature_engineering.trend import build_trend_features
from feature_engineering.volatility import build_volatility_features

from .converter import DataPointConverter
from .framework import PipelineContext, StageContext, Step

LOG = logging.getLogger(__name__)

# Bucket name constant
FEATURE_DATA_BUCKET = "feature_data"

# =============================================================================
# Context and Step Definitions
# =============================================================================


class FeatureTarget:
    NEAR_HFT = "NearHFT"  # hold positions for 20-30 seconds
    MID = "Mid"  # hold positions for 120-300 seconds
    FAR = "Far"
    ORION = "Orion"  # special config for Orion strategy


@dataclass
class Context(StageContext):
    market: Market
    execution_price_df: pd.DataFrame = None
    trade_aggression_df: pd.DataFrame = None
    order_book_state_df: pd.DataFrame = None
    book_change_df: pd.DataFrame = None


class AlignAndBuildFeaturesStep(Step):
    """
    Aligns data from different sources and builds a comprehensive feature DataFrame.

    Implements all features documented in doc/features.md:
    1. Trade Aggression Features (Section 1)
    2. Order Book State Features (Section 2)
    3. Book Change Features (Section 3)
    4. Execution Price Features (Section 4)
    5. Cross-Feature Interactions (Section 5)
    6. Trend & Momentum Features (Section 6)
    7. Volatility Features (Section 7)

    All features are normalized/stationary for ML model compatibility.

    Architecture: source_df holds raw aligned data (read-only). Features are
    collected in a dictionary and converted to DataFrame at the end to avoid
    fragmentation warnings.
    """

    def __init__(
        self,
        target: str,
        trade_aggression_windows: List[str] = None,
        order_book_depth_levels: List[int] = None,
        order_book_state_windows: List[str] = None,
        trend_windows: List[int] = None,
        lead_market: Market = None,
    ):
        super().__init__()
        self.target = target
        self.trade_aggression_windows = trade_aggression_windows
        self.order_book_depth_levels = order_book_depth_levels
        self.order_book_state_windows = order_book_state_windows
        self.trend_windows = trend_windows
        self.lead_market = lead_market

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        """Main entry point for feature computation - saves to Parquet only."""
        feature_df = self._populate(pipeline_context, stage_context)
        if feature_df is None or feature_df.empty:
            return

        # Save to parquet store
        parquet_store = pipeline_context.parquet_store.create_feature(stage_context.market)
        parquet_store.add_dataframe(feature_df, ts_column=None)
        parquet_store.close()
        LOG.info(
            f"AlignAndBuildFeaturesStep: Saved {len(feature_df)} rows "
            f"with {len(feature_df.columns)} columns to Parquet store."
        )

    def _populate(self, pipeline_context: PipelineContext, stage_context: Context):
        """Shared logic for feature computation."""
        # Align all data sources into a single read-only source DataFrame
        source_df = self._align_data_sources(stage_context)
        if source_df is None or source_df.empty:
            LOG.warning("AlignAndBuildFeaturesStep: No data sources available")
            return None

        LOG.info(
            f"AlignAndBuildFeaturesStep: Starting with {len(source_df)} rows, "
            f"{len(source_df.columns)} source columns"
        )
        LOG.info(f"AlignAndBuildFeaturesStep: Source columns: {sorted(source_df.columns.tolist())}")

        # Collect all features in a dictionary to avoid DataFrame fragmentation
        features: Dict[str, pd.Series] = {}

        # Build features by section
        before = len(features)
        build_trade_aggression_features(source_df, features, self.trade_aggression_windows)
        LOG.info(f"  Trade Aggression: +{len(features) - before} features")

        before = len(features)
        build_order_book_state_features(source_df, features, self.order_book_depth_levels)
        LOG.info(f"  Order Book State: +{len(features) - before} features")

        before = len(features)
        build_book_change_features(source_df, features, self.order_book_depth_levels)
        LOG.info(f"  Book Change: +{len(features) - before} features")

        before = len(features)
        build_execution_price_features(source_df, features)
        LOG.info(f"  Execution Price: +{len(features) - before} features")

        before = len(features)
        build_cross_feature_interactions(
            source_df, features, self.trade_aggression_windows, self.order_book_depth_levels
        )
        LOG.info(f"  Cross-Feature Interactions: +{len(features) - before} features")

        if self.trend_windows:
            before = len(features)
            build_trend_features(source_df, features, self.trend_windows)
            LOG.info(f"  Trend: +{len(features) - before} features")

        before = len(features)
        build_volatility_features(source_df, features)
        LOG.info(f"  Volatility: +{len(features) - before} features")

        LOG.info(
            f"AlignAndBuildFeaturesStep: Built {len(features)} computed features " f"(source columns not included)"
        )

        # Create DataFrame from features dict in one operation (avoids fragmentation)
        feature_df = pd.DataFrame(features, index=source_df.index)

        # source_df no longer needed
        del source_df

        # Post-processing: handle NaN/Inf and filter to populate range
        feature_df = self._postprocess_features(feature_df, stage_context)
        # Filter to populate range (using UTC for comparison with UTC index)
        feature_df = feature_df[
            (feature_df.index >= stage_context.populate_dt_range.start_dt_utc)
            & (feature_df.index <= stage_context.populate_dt_range.end_dt_utc)
        ]
        LOG.info(f"AlignAndBuildFeaturesStep: Feature columns: {sorted(feature_df.columns.tolist())}")

        return feature_df

    def populate_influxdb(self, pipeline_context: PipelineContext, stage_context: Context):
        """Saves features to InfluxDB only."""
        feature_df = self._populate(pipeline_context, stage_context)
        if feature_df is None or feature_df.empty:
            return

        influx_writer = pipeline_context.influx_store.create_writer(FEATURE_DATA_BUCKET, no_retention=True)
        points = DataPointConverter.convert_dataframe_to_influx_points(
            feature_df,
            measurement="FEATURE",
            exchange_name=stage_context.market.exchange_name.name,
            symbol=stage_context.market.symbol.name,
        )
        for point in points:
            influx_writer.add(point)
        influx_writer.close()
        LOG.info(f"AlignAndBuildFeaturesStep: Wrote {len(points)} points to InfluxDB.")

    # =========================================================================
    # Data alignment
    # =========================================================================

    def _align_data_sources(self, stage_context: Context) -> Optional[pd.DataFrame]:
        """Align all data sources into a single DataFrame with DatetimeIndex."""
        dfs_to_merge = []

        # Process each data source
        for df, name in [
            (stage_context.trade_aggression_df, "trade_aggression"),
            (stage_context.order_book_state_df, "order_book_state"),
            (stage_context.book_change_df, "book_change"),
            (stage_context.execution_price_df, "execution_price"),
        ]:
            if df is not None and not df.empty:
                df = df.copy()
                # Convert ts column to DatetimeIndex if present
                if "ts" in df.columns:
                    df.index = pd.to_datetime(df["ts"], unit="s")
                    df = df.drop(columns=["ts"])
                # Ensure sorted
                if not df.index.is_monotonic_increasing:
                    df = df.sort_index()

                dfs_to_merge.append((df, name))
                LOG.info(f"AlignAndBuildFeaturesStep: Loaded {name} with {len(df)} rows")

        if not dfs_to_merge:
            return None

        # Start with the first DataFrame
        result = dfs_to_merge[0][0].copy()

        # Merge remaining DataFrames using outer join
        for df, name in dfs_to_merge[1:]:
            result = result.join(df, how="outer", rsuffix="_dup")
            # Drop duplicate columns
            dup_cols = [col for col in result.columns if col.endswith("_dup")]
            if dup_cols:
                result = result.drop(columns=dup_cols)

        # Sort by index
        result = result.sort_index()

        # Forward-fill all columns to carry forward last known values.
        with pd.option_context("future.no_silent_downcasting", True):
            result = result.ffill()
        # Explicitly infer better dtypes after filling
        result = result.infer_objects(copy=False)
        LOG.info(f"AlignAndBuildFeaturesStep: Forward-filled {len(result.columns)} columns across all sources")

        return result

    # =========================================================================
    # Post-processing
    # =========================================================================

    def _postprocess_features(self, df: pd.DataFrame, stage_context: Context) -> pd.DataFrame:
        """Post-process features: handle NaN/Inf, filter to populate range."""
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        # Drop rows with too many NaN values (keep rows with at least 30% valid data)
        threshold = int(len(df.columns) * 0.3)
        df = df.dropna(thresh=threshold)

        # Forward-fill remaining NaN values (with limit)
        with pd.option_context("future.no_silent_downcasting", True):
            df = df.ffill(limit=3)
        df = df.infer_objects(copy=False)

        LOG.info(f"AlignAndBuildFeaturesStep: After post-processing: {len(df)} rows, {len(df.columns)} columns")

        return df
