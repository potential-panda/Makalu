import logging
from dataclasses import dataclass


import pandas as pd
from qate.core.model import Market
from qate.util.dt_range import DtRange

from .converter import DataPointConverter
from .framework import PipelineContext, StageContext, Step

LOG = logging.getLogger(__name__)

# Bucket name constant
FEATURE_DATA_BUCKET = "feature_data"


@dataclass
class Context(StageContext):
    compute_dt_range: DtRange
    market: Market
    trade_aggression_df: pd.DataFrame = None
    order_book_state_df: pd.DataFrame = None
    book_change_df: pd.DataFrame = None


class QueryDataStep(Step):
    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        parquet_store = pipeline_context.parquet_store.create_single_market("TradeAggression", stage_context.market)
        df = parquet_store.read(stage_context.compute_dt_range)
        stage_context.trade_aggression_df = df

        parquet_store = pipeline_context.parquet_store.create_single_market("BookState", stage_context.market)
        df = parquet_store.read(stage_context.compute_dt_range)
        stage_context.order_book_state_df = df

        parquet_store = pipeline_context.parquet_store.create_single_market("BookChange", stage_context.market)
        df = parquet_store.read(stage_context.compute_dt_range)
        stage_context.book_change_df = df


class AlignAndBuildFeaturesStep(Step):
    """
    Aligns data from different sources and builds a comprehensive feature DataFrame.

    This step:
    1. Aligns orderbook state, trade aggression, and book change data
    2. Computes derived features (microprice_offset_bps, etc.)
    3. Computes rolling/EMA aggregates for orderbook features
    4. Computes cross-window features for trade aggression
    5. Computes interaction features
    """

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        """Aligns data from different sources and builds a single feature DataFrame."""
        # Ensure all DataFrames are sorted by index (time)
        dfs = [stage_context.trade_aggression_df, stage_context.order_book_state_df, stage_context.book_change_df]
        for df in dfs:
            if df is not None and not df.empty and not df.index.is_monotonic_increasing:
                df.sort_index(inplace=True)

        # Filter out empty dataframes
        dfs = [df for df in dfs if df is not None and not df.empty]

        if not dfs:
            LOG.warning("AlignAndBuildFeaturesStep: No data available to build features.")
            return

        # Start with the first DataFrame
        feature_df = dfs[0].copy()

        # Merge the remaining DataFrames using join (since they're already time-indexed)
        for df in dfs[1:]:
            feature_df = feature_df.join(df, how="outer", rsuffix="_dup")
            # Drop duplicate columns if any
            feature_df = feature_df.loc[:, ~feature_df.columns.str.endswith("_dup")]

        # Sort by index after merging
        feature_df.sort_index(inplace=True)

        # 1. Compute microprice_offset_bps for each microprice_k
        feature_df = self._compute_microprice_offset_bps(feature_df)

        # 2. Compute VWAP offset in bps for trade aggression windows
        feature_df = self._compute_vwap_offset_bps(feature_df)

        # 3. Compute rolling/EMA aggregates for orderbook features (1s, 3s, 5s)
        feature_df = self._compute_rolling_aggregates(feature_df)

        # 4. Compute cross-window features for trade aggression
        feature_df = self._compute_cross_window_features(feature_df)

        # 5. Compute interaction features
        feature_df = self._compute_interaction_features(feature_df)

        # Filter the DataFrame to the populate_dt_range
        feature_df = feature_df[
            (feature_df.index >= stage_context.populate_dt_range.start_dt)
            & (feature_df.index <= stage_context.populate_dt_range.end_dt)
        ]

        # Drop rows with too many NaN values (keep rows with at least 50% valid data)
        threshold = len(feature_df.columns) * 0.5
        feature_df = feature_df.dropna(thresh=threshold)

        # Save to parquet store (partitioned by date)
        parquet_store = pipeline_context.parquet_store.create_feature(stage_context.market)
        parquet_store.add_dataframe(feature_df, ts_column=None)  # DataFrame has DatetimeIndex
        parquet_store.close()
        LOG.info(
            f"AlignAndBuildFeaturesStep: Built and saved {len(feature_df)} features "
            f"with {len(feature_df.columns)} columns to parquet store."
        )

        # Save to InfluxDB
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

    def _compute_microprice_offset_bps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes microprice_offset_bps = (microprice - mid) / mid * 1e4
        for each microprice_k column.
        """
        if "mid" not in df.columns:
            LOG.warning("'mid' column not found, skipping microprice_offset_bps computation.")
            return df

        microprice_cols = [col for col in df.columns if col.startswith("microprice_k")]
        for col in microprice_cols:
            offset_col = col.replace("microprice_k", "microprice_offset_bps_k")
            df[offset_col] = (df[col] - df["mid"]) / df["mid"] * 1e4

        return df

    def _compute_vwap_offset_bps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes ta_vwap_offset_bps = (vwap - mid) / mid * 1e4
        for each trade aggression window.
        """
        if "mid" not in df.columns:
            LOG.warning("'mid' column not found, skipping vwap_offset_bps computation.")
            return df

        vwap_cols = [col for col in df.columns if col == "vwap" or "_vwap" in col]
        for col in vwap_cols:
            # Create offset column name
            if col == "vwap":
                offset_col = "vwap_offset_bps"
            else:
                offset_col = col.replace("vwap", "vwap_offset_bps")

            df[offset_col] = (df[col] - df["mid"]) / df["mid"] * 1e4

        return df

    def _compute_rolling_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling mean and std for orderbook features over 1s, 3s, 5s windows.
        """
        windows = {"1s": "1S", "3s": "3S", "5s": "5S"}

        # Select orderbook features to aggregate
        # Include: spread_bps, volume imbalance, weighted imbalance, microprice_offset_bps
        feature_patterns = [
            "spread_bps",
            "vol_imb_k",
            "weighted_imb_k",
            "microprice_offset_bps_k",
            "depth_imb_bps",
        ]

        features_to_aggregate = []
        for pattern in feature_patterns:
            features_to_aggregate.extend([col for col in df.columns if pattern in col])

        # Remove duplicates
        features_to_aggregate = list(set(features_to_aggregate))

        for window_name, window_size in windows.items():
            for feature in features_to_aggregate:
                # Rolling mean
                df[f"{feature}_mean_{window_name}"] = df[feature].rolling(window_size).mean()
                # Rolling std
                df[f"{feature}_std_{window_name}"] = df[feature].rolling(window_size).std()

        return df

    def _compute_cross_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes cross-window features for trade aggression.
        E.g., imbalance_qty difference between 1s and 3s windows (impulse vs persistence).
        """
        # Define window pairs for comparison
        window_pairs = [
            ("1s", "3s"),
            ("1s", "5s"),
            ("3s", "5s"),
        ]

        # Features to compute cross-window comparisons for
        features = ["imbalance_qty", "imbalance_count", "qty_per_sec", "trades_per_sec"]

        for feature in features:
            for w1, w2 in window_pairs:
                col1 = f"{feature}_{w1}"
                col2 = f"{feature}_{w2}"

                if col1 in df.columns and col2 in df.columns:
                    # Difference (impulse vs persistence)
                    df[f"{feature}_diff_{w1}_{w2}"] = df[col1] - df[col2]

                    # Sign agreement (both positive or both negative)
                    df[f"{feature}_sign_agree_{w1}_{w2}"] = (((df[col1] > 0) == (df[col2] > 0))).astype(int)

        return df

    def _compute_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes interaction features:
        1. Sign agreement between trade imbalance and microprice offset (per window)
        2. Strong book but weak trades flags
        3. Spread-conditioned strength
        """
        # 1. Sign agreement: sign(ta_imbalance_qty) == sign(microprice_offset_bps_k1)
        #    Compute for each trade aggression window
        windows = ["1s", "3s", "5s"]
        for window in windows:
            imb_col = f"imbalance_qty_{window}"
            if imb_col in df.columns and "microprice_offset_bps_k1" in df.columns:
                df[f"sign_agreement_trade_book_{window}"] = (
                    (df[imb_col] > 0) == (df["microprice_offset_bps_k1"] > 0)
                ).astype(int)

        # 2. Strong book but weak trades: large book imbalance + low trades_per_sec
        for window in windows:
            trades_col = f"trades_per_sec_{window}"
            if "vol_imb_k1" in df.columns and trades_col in df.columns:
                # Define thresholds
                strong_book_threshold = 0.3  # absolute imbalance > 0.3
                weak_trades_threshold = 10  # trades_per_sec < 10

                df[f"strong_book_weak_trades_{window}"] = (
                    (df["vol_imb_k1"].abs() > strong_book_threshold) & (df[trades_col] < weak_trades_threshold)
                ).astype(int)

        # 3. Spread-conditioned strength: only consider signals when spread is reasonable
        if "spread_bps" in df.columns:
            spread_threshold = 5  # 5 bps
            df["tight_spread"] = (df["spread_bps"] < spread_threshold).astype(int)

            # Create spread-conditioned imbalance signals
            for window in windows:
                imb_col = f"imbalance_qty_{window}"
                if imb_col in df.columns:
                    df[f"imbalance_qty_tight_spread_{window}"] = df[imb_col] * df["tight_spread"]

        return df
