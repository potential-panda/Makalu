"""
Feature engineering pipeline steps.

This module implements comprehensive feature engineering as documented in doc/features.md.
All features are designed to be stationary (normalized) for ML model compatibility.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from qate.core.model import Market
from qate.util.dt_range import DtRange

from .converter import DataPointConverter
from .framework import PipelineContext, StageContext, Step

LOG = logging.getLogger(__name__)

# Bucket name constant
FEATURE_DATA_BUCKET = "feature_data"

# Default rolling window for z-score and percentile calculations
DEFAULT_ROLLING_WINDOW = "5min"

# Small epsilon to avoid division by zero
EPS = 1e-9


# =============================================================================
# Utility Functions for Feature Computation
# =============================================================================


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safe division with epsilon to avoid division by zero."""
    return numerator / (denominator + EPS)


def compute_imbalance(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Compute normalized imbalance: (A - B) / (A + B).
    Returns values in range [-1, 1].
    """
    return (a - b) / (a + b + EPS)


def compute_imbalance_abs(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Compute normalized imbalance with absolute denominator: (A - B) / (|A| + |B|).
    Useful when A and B can be negative.
    Returns values in range [-1, 1].
    """
    return (a - b) / (a.abs() + b.abs() + EPS)


def compute_zscore(series: pd.Series, window: str = DEFAULT_ROLLING_WINDOW) -> pd.Series:
    """
    Compute rolling z-score: (x - mean) / std.
    Returns values typically in range [-3, 3].
    """
    rolling_mean = series.rolling(window, closed="left").mean()
    rolling_std = series.rolling(window, closed="left").std()
    return (series - rolling_mean) / (rolling_std + EPS)


def compute_percentile(series: pd.Series, window: str = DEFAULT_ROLLING_WINDOW) -> pd.Series:
    """
    Compute rolling percentile rank.
    Returns values in range [0, 1].
    """
    return series.rolling(window, closed="left").rank(pct=True)


def compute_bps_offset(price: pd.Series, reference: pd.Series) -> pd.Series:
    """
    Compute basis points offset: (price - reference) / reference * 10000.
    """
    return (price - reference) / (reference + EPS) * 10000


def get_column_if_exists(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Safely get a column from DataFrame, returning None if not exists."""
    return df[col] if col in df.columns else None


def has_columns(df: pd.DataFrame, cols: List[str]) -> bool:
    """Check if all specified columns exist in DataFrame."""
    return all(col in df.columns for col in cols)


# =============================================================================
# Context and Step Definitions
# =============================================================================


@dataclass
class Context(StageContext):
    compute_dt_range: DtRange
    market: Market
    execution_price_df: pd.DataFrame = None
    trade_aggression_df: pd.DataFrame = None
    order_book_state_df: pd.DataFrame = None
    book_change_df: pd.DataFrame = None


class QueryDataStep(Step):
    def __init__(self, execution_size_str: str):
        super().__init__()
        self.execution_size_str = execution_size_str

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        # Query execution price data
        parquet_store = pipeline_context.parquet_store.create_execution_price(
            stage_context.market, self.execution_size_str
        )
        df = parquet_store.read(stage_context.compute_dt_range)
        stage_context.execution_price_df = df

        # Query trade aggression data
        parquet_store = pipeline_context.parquet_store.create_single_market("TradeAggression", stage_context.market)
        df = parquet_store.read(stage_context.compute_dt_range)
        stage_context.trade_aggression_df = df

        # Query order book state data
        parquet_store = pipeline_context.parquet_store.create_single_market("BookState", stage_context.market)
        df = parquet_store.read(stage_context.compute_dt_range)
        stage_context.order_book_state_df = df

        # Query book change data
        parquet_store = pipeline_context.parquet_store.create_single_market("BookChange", stage_context.market)
        df = parquet_store.read(stage_context.compute_dt_range)
        stage_context.book_change_df = df


class AlignAndBuildFeaturesStep(Step):
    """
    Aligns data from different sources and builds a comprehensive feature DataFrame.

    Implements all features documented in doc/features.md:
    1. Trade Aggression Features (Section 1)
    2. Order Book State Features (Section 2)
    3. Book Change Features (Section 3)
    4. Execution Price Features (Section 4)
    5. Cross-Feature Interactions (Section 5)

    All features are normalized/stationary for ML model compatibility.
    """

    # Time windows for trade aggression features
    TRADE_WINDOWS = ["1s", "3s", "5s"]

    # Depth levels for order book features
    DEPTH_LEVELS = [1, 3, 5, 10]

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        """Main entry point for feature computation."""
        # Align all data sources
        feature_df = self._align_data_sources(stage_context)

        if feature_df is None or feature_df.empty:
            LOG.warning("AlignAndBuildFeaturesStep: No data available to build features.")
            return

        LOG.info(f"AlignAndBuildFeaturesStep: Starting with {len(feature_df)} rows, {len(feature_df.columns)} columns")

        # Build features by section
        feature_df = self._build_trade_aggression_features(feature_df)
        feature_df = self._build_order_book_state_features(feature_df)
        feature_df = self._build_book_change_features(feature_df)
        feature_df = self._build_execution_price_features(feature_df)
        feature_df = self._build_cross_feature_interactions(feature_df)

        # Post-processing: handle NaN/Inf and filter to populate range
        feature_df = self._postprocess_features(feature_df, stage_context)
        feature_df = feature_df[
            (feature_df.index >= stage_context.populate_dt_range.start_dt)
            & (feature_df.index <= stage_context.populate_dt_range.end_dt)
        ]

        if feature_df.empty:
            LOG.warning("AlignAndBuildFeaturesStep: No data after post-processing.")
            return

        # Save to parquet store
        parquet_store = pipeline_context.parquet_store.create_feature(stage_context.market)
        parquet_store.add_dataframe(feature_df, ts_column=None)
        parquet_store.close()
        LOG.info(
            f"AlignAndBuildFeaturesStep: Built and saved {len(feature_df)} rows "
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
        feature_df = dfs_to_merge[0][0].copy()

        # Merge remaining DataFrames using outer join
        for df, name in dfs_to_merge[1:]:
            feature_df = feature_df.join(df, how="outer", rsuffix="_dup")
            # Drop duplicate columns
            dup_cols = [col for col in feature_df.columns if col.endswith("_dup")]
            if dup_cols:
                feature_df = feature_df.drop(columns=dup_cols)

        # Sort by index
        feature_df = feature_df.sort_index()

        return feature_df

    # =========================================================================
    # Section 1: Trade Aggression Features
    # =========================================================================

    def _build_trade_aggression_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all trade aggression features (Section 1 of features.md).

        Features include:
        - Normalized imbalances (vol_imbalance_ratio, count_imbalance_ratio)
        - VWAP features (vwap_buy_sell_spread_bps)
        - Trade size features (size_asymmetry_norm, avg_size_zscore, avg_size_pct)
        - Activity level features (qty_per_sec_zscore, trades_per_sec_pct, etc.)
        - Activity concentration (buy_qty_share, qty_count_divergence)
        - Cross-window momentum features
        """
        LOG.info("Building trade aggression features...")

        for window in self.TRADE_WINDOWS:
            df = self._build_trade_features_for_window(df, window)

        # Cross-window features (momentum, sign agreement, magnitude dynamics)
        df = self._build_cross_window_features(df)

        return df

    def _build_trade_features_for_window(self, df: pd.DataFrame, window: str) -> pd.DataFrame:
        """Build trade aggression features for a specific time window."""
        suffix = f"_{window}"

        # Get raw columns for this window
        buy_qty = get_column_if_exists(df, f"buy_qty{suffix}")
        sell_qty = get_column_if_exists(df, f"sell_qty{suffix}")
        buy_count = get_column_if_exists(df, f"buy_count{suffix}")
        sell_count = get_column_if_exists(df, f"sell_count{suffix}")
        total_qty = get_column_if_exists(df, f"total_qty{suffix}")
        total_count = get_column_if_exists(df, f"total_count{suffix}")
        vwap = get_column_if_exists(df, f"vwap{suffix}")
        vwap_buy = get_column_if_exists(df, f"vwap_buy{suffix}")
        vwap_sell = get_column_if_exists(df, f"vwap_sell{suffix}")
        qty_per_sec = get_column_if_exists(df, f"qty_per_sec{suffix}")
        trades_per_sec = get_column_if_exists(df, f"trades_per_sec{suffix}")

        # 1. Normalized Imbalances
        if buy_qty is not None and sell_qty is not None:
            df[f"vol_imbalance_ratio{suffix}"] = compute_imbalance(buy_qty, sell_qty)

        if buy_count is not None and sell_count is not None:
            df[f"count_imbalance_ratio{suffix}"] = compute_imbalance(buy_count, sell_count)

        # 2. VWAP Features (Trade-Only)
        if vwap_buy is not None and vwap_sell is not None and vwap is not None:
            df[f"vwap_buy_sell_spread_bps{suffix}"] = (vwap_buy - vwap_sell) / (vwap + EPS) * 10000

        # 3. Trade Size Features
        if buy_qty is not None and buy_count is not None and sell_qty is not None and sell_count is not None:
            avg_buy_size = buy_qty / (buy_count + EPS)
            avg_sell_size = sell_qty / (sell_count + EPS)
            avg_size = (buy_qty + sell_qty) / (buy_count + sell_count + EPS)

            # Size asymmetry (which side trades bigger)
            df[f"size_asymmetry_norm{suffix}"] = compute_imbalance(avg_buy_size, avg_sell_size)

            # Avg size z-score and percentile
            df[f"avg_size_zscore{suffix}"] = compute_zscore(avg_size)
            df[f"avg_size_pct{suffix}"] = compute_percentile(avg_size)

        # 4. Activity Level Features (normalized)
        if qty_per_sec is not None:
            df[f"qty_per_sec_zscore{suffix}"] = compute_zscore(qty_per_sec)
            df[f"qty_per_sec_pct{suffix}"] = compute_percentile(qty_per_sec)

        if trades_per_sec is not None:
            df[f"trades_per_sec_zscore{suffix}"] = compute_zscore(trades_per_sec)
            df[f"trades_per_sec_pct{suffix}"] = compute_percentile(trades_per_sec)

        # 5. Activity Concentration
        if buy_qty is not None and total_qty is not None:
            df[f"buy_qty_share{suffix}"] = buy_qty / (total_qty + EPS)

        if buy_count is not None and total_count is not None:
            df[f"buy_count_share{suffix}"] = buy_count / (total_count + EPS)

        # Qty-count divergence
        buy_qty_share = get_column_if_exists(df, f"buy_qty_share{suffix}")
        buy_count_share = get_column_if_exists(df, f"buy_count_share{suffix}")
        if buy_qty_share is not None and buy_count_share is not None:
            df[f"qty_count_divergence{suffix}"] = buy_qty_share - buy_count_share

        return df

    def _build_cross_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build cross-window features for trade aggression (momentum, sign agreement, etc.)."""
        window_pairs = [("1s", "3s"), ("1s", "5s"), ("3s", "5s")]

        # Get vol_imbalance_ratio for each window
        ratio_1s = get_column_if_exists(df, "vol_imbalance_ratio_1s")
        ratio_3s = get_column_if_exists(df, "vol_imbalance_ratio_3s")
        ratio_5s = get_column_if_exists(df, "vol_imbalance_ratio_5s")

        if ratio_1s is None or ratio_3s is None or ratio_5s is None:
            LOG.warning("Missing vol_imbalance_ratio columns for cross-window features")
            return df

        # Momentum features (difference between windows)
        df["imbalance_momentum_1s_3s"] = ratio_1s - ratio_3s
        df["imbalance_momentum_1s_5s"] = ratio_1s - ratio_5s
        df["imbalance_momentum_3s_5s"] = ratio_3s - ratio_5s

        # Direction consistency (sign agreement)
        df["imbalance_sign_agree_1s_3s"] = (np.sign(ratio_1s) == np.sign(ratio_3s)).astype(int)
        df["imbalance_sign_agree_1s_5s"] = (np.sign(ratio_1s) == np.sign(ratio_5s)).astype(int)
        df["imbalance_sign_agree_3s_5s"] = (np.sign(ratio_3s) == np.sign(ratio_5s)).astype(int)
        df["imbalance_sign_agree_all"] = (
            (df["imbalance_sign_agree_1s_3s"] == 1) & (df["imbalance_sign_agree_1s_5s"] == 1)
        ).astype(int)

        # Magnitude dynamics (strengthening/weakening)
        df["imbalance_strengthening_1s_3s"] = (ratio_1s.abs() > ratio_3s.abs()).astype(int)
        df["imbalance_strengthening_1s_5s"] = (ratio_1s.abs() > ratio_5s.abs()).astype(int)
        df["imbalance_weakening_1s_5s"] = (ratio_1s.abs() < ratio_5s.abs()).astype(int)

        # Signed flow indicators (boolean)
        df["strong_buy_signal"] = (
            (ratio_1s > 0.3) & (df["imbalance_sign_agree_all"] == 1) & (df["imbalance_strengthening_1s_5s"] == 1)
        ).astype(int)

        df["strong_sell_signal"] = (
            (ratio_1s < -0.3) & (df["imbalance_sign_agree_all"] == 1) & (df["imbalance_strengthening_1s_5s"] == 1)
        ).astype(int)

        df["exhaustion_signal"] = (
            (ratio_5s.abs() > 0.3) & (df["imbalance_weakening_1s_5s"] == 1) & (df["imbalance_sign_agree_1s_5s"] == 1)
        ).astype(int)

        df["reversal_signal"] = ((ratio_5s.abs() > 0.3) & (df["imbalance_sign_agree_1s_5s"] == 0)).astype(int)

        return df

    # =========================================================================
    # Section 2: Order Book State Features
    # =========================================================================

    def _build_order_book_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all order book state features (Section 2 of features.md).

        Features include:
        - Microprice offset (microprice_offset_bps_k)
        - Imbalance gradient and divergence
        - Rolling aggregates for key features
        """
        LOG.info("Building order book state features...")

        best_mid = get_column_if_exists(df, "best_mid")
        if best_mid is None:
            best_mid = get_column_if_exists(df, "mid")
        if best_mid is None:
            LOG.warning("'best_mid' column not found, skipping order book state features")
            return df

        # Microprice offset in bps for each k
        for k in self.DEPTH_LEVELS:
            microprice_col = f"microprice_k{k}"
            microprice = get_column_if_exists(df, microprice_col)
            if microprice is not None:
                df[f"microprice_offset_bps_k{k}"] = compute_bps_offset(microprice, best_mid)

        # Imbalance gradient and divergence
        vol_imb_1 = get_column_if_exists(df, "vol_imb_k1")
        vol_imb_5 = get_column_if_exists(df, "vol_imb_k5")
        vol_imb_10 = get_column_if_exists(df, "vol_imb_k10")

        if vol_imb_1 is not None and vol_imb_5 is not None:
            df["imbalance_gradient"] = vol_imb_1 - vol_imb_5

        if vol_imb_1 is not None and vol_imb_10 is not None:
            df["imbalance_divergence"] = (vol_imb_1 - vol_imb_10).abs()

        # Spread-normalized microprice offset
        spread_bps = get_column_if_exists(df, "best_spread_bps")
        if spread_bps is None:
            spread_bps = get_column_if_exists(df, "spread_bps")
        microprice_offset_bps_1 = get_column_if_exists(df, "microprice_offset_bps_k1")
        if spread_bps is not None and microprice_offset_bps_1 is not None:
            df["microprice_offset_norm"] = microprice_offset_bps_1 / (spread_bps + EPS)

        # Rolling aggregates for key order book features
        df = self._build_order_book_rolling_aggregates(df)

        return df

    def _build_order_book_rolling_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build rolling mean/std for order book features."""
        windows = {"1s": "1s", "3s": "3s", "5s": "5s"}

        # Features to compute rolling stats for
        book_spread_bps_name = "best_spread_bps" if "best_spread_bps" in df.columns else "spread_bps"
        feature_patterns = [
            book_spread_bps_name,
            "vol_imb_k1",
            "vol_imb_k5",
            "microprice_offset_bps_k1",
        ]

        for feature in feature_patterns:
            if feature not in df.columns:
                continue

            for window_name, window_size in windows.items():
                df[f"{feature}_mean_{window_name}"] = df[feature].rolling(window_size).mean()
                df[f"{feature}_std_{window_name}"] = df[feature].rolling(window_size).std()

        return df

    # =========================================================================
    # Section 3: Book Change Features
    # =========================================================================

    def _build_book_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all book change features (Section 3 of features.md).

        Features include:
        - Delta depth imbalance (normalized)
        - Pull/refill imbalance
        - Activity level features (z-score, percentile)
        - Cross-k features (gradient)
        """
        LOG.info("Building book change features...")

        for k in self.DEPTH_LEVELS:
            df = self._build_book_change_features_for_k(df, k)

        # Cross-k features (gradient between shallow and deep)
        df = self._build_book_change_cross_k_features(df)

        return df

    def _build_book_change_features_for_k(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """Build book change features for a specific depth level k."""
        suffix = f"_k{k}"

        # Get raw columns
        delta_depth_bid = get_column_if_exists(df, f"delta_depth_bid{suffix}")
        delta_depth_ask = get_column_if_exists(df, f"delta_depth_ask{suffix}")
        pull_bid_rate = get_column_if_exists(df, f"pull_bid_rate{suffix}")
        pull_ask_rate = get_column_if_exists(df, f"pull_ask_rate{suffix}")
        refill_bid_rate = get_column_if_exists(df, f"refill_bid_rate{suffix}")
        refill_ask_rate = get_column_if_exists(df, f"refill_ask_rate{suffix}")

        # 1. Delta depth imbalance (normalized)
        if delta_depth_bid is not None and delta_depth_ask is not None:
            df[f"delta_depth_imbalance_norm{suffix}"] = compute_imbalance_abs(delta_depth_bid, delta_depth_ask)

        # 2. Pull/refill imbalance
        if pull_bid_rate is not None and pull_ask_rate is not None:
            df[f"pull_imbalance{suffix}"] = compute_imbalance(pull_bid_rate, pull_ask_rate)

        if refill_bid_rate is not None and refill_ask_rate is not None:
            df[f"refill_imbalance{suffix}"] = compute_imbalance(refill_bid_rate, refill_ask_rate)

        # 3. Activity level features (normalized)
        if pull_bid_rate is not None and pull_ask_rate is not None:
            total_pull_rate = pull_bid_rate + pull_ask_rate
            df[f"total_pull_zscore{suffix}"] = compute_zscore(total_pull_rate)

        if refill_bid_rate is not None and refill_ask_rate is not None:
            total_refill_rate = refill_bid_rate + refill_ask_rate

            # Pull/refill ratio
            if pull_bid_rate is not None and pull_ask_rate is not None:
                total_pull_rate = pull_bid_rate + pull_ask_rate
                df[f"pull_refill_ratio{suffix}"] = total_pull_rate / (total_refill_rate + EPS)

        # Total activity (pull + refill)
        if (
            pull_bid_rate is not None
            and pull_ask_rate is not None
            and refill_bid_rate is not None
            and refill_ask_rate is not None
        ):
            total_activity = pull_bid_rate + pull_ask_rate + refill_bid_rate + refill_ask_rate
            df[f"total_activity_zscore{suffix}"] = compute_zscore(total_activity)
            df[f"total_activity_pct{suffix}"] = compute_percentile(total_activity)

        return df

    def _build_book_change_cross_k_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build cross-k features comparing shallow vs deep book changes."""
        # Pull imbalance gradient (k1 vs k5)
        pull_imb_k1 = get_column_if_exists(df, "pull_imbalance_k1")
        pull_imb_k5 = get_column_if_exists(df, "pull_imbalance_k5")
        if pull_imb_k1 is not None and pull_imb_k5 is not None:
            df["pull_imbalance_gradient"] = pull_imb_k1 - pull_imb_k5

        # Refill imbalance gradient (k1 vs k5)
        refill_imb_k1 = get_column_if_exists(df, "refill_imbalance_k1")
        refill_imb_k5 = get_column_if_exists(df, "refill_imbalance_k5")
        if refill_imb_k1 is not None and refill_imb_k5 is not None:
            df["refill_imbalance_gradient"] = refill_imb_k1 - refill_imb_k5

        return df

    # =========================================================================
    # Section 4: Execution Price Features
    # =========================================================================

    def _build_execution_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all execution price features (Section 4 of features.md).

        Features include:
        - Slippage metrics (bid, ask, total in bps)
        - Slippage magnitude features (z-score, percentile, asymmetry)
        - Spread expansion features
        - Depth quality score
        """
        LOG.info("Building execution price features...")

        # Get required columns
        best_bid = get_column_if_exists(df, "best_bid")
        best_ask = get_column_if_exists(df, "best_ask")
        exec_bid = get_column_if_exists(df, "bid")  # Execution price for selling
        exec_ask = get_column_if_exists(df, "ask")  # Execution price for buying
        mid = get_column_if_exists(df, "mid")
        spread_bps = get_column_if_exists(df, "best_spread_bps")
        if spread_bps is None:
            spread_bps = get_column_if_exists(df, "spread_bps")

        # Slippage metrics in bps
        if best_bid is not None and exec_bid is not None:
            df["slippage_bid_bps"] = (best_bid - exec_bid) / (best_bid + EPS) * 10000

        if best_ask is not None and exec_ask is not None:
            df["slippage_ask_bps"] = (exec_ask - best_ask) / (best_ask + EPS) * 10000

        slippage_bid_bps = get_column_if_exists(df, "slippage_bid_bps")
        slippage_ask_bps = get_column_if_exists(df, "slippage_ask_bps")

        if slippage_bid_bps is not None and slippage_ask_bps is not None:
            slippage_total_bps = slippage_bid_bps + slippage_ask_bps
            df["slippage_total_bps"] = slippage_total_bps

            # Slippage magnitude features (normalized)
            df["slippage_zscore"] = compute_zscore(slippage_total_bps)
            df["slippage_pct"] = compute_percentile(slippage_total_bps)

            # Slippage asymmetry
            df["slippage_asymmetry"] = (slippage_ask_bps - slippage_bid_bps) / (slippage_total_bps + EPS)

            # Depth quality score
            df["depth_quality"] = 1 / (slippage_total_bps + 1)

        # Spread expansion features
        if exec_bid is not None and exec_ask is not None and mid is not None and spread_bps is not None:
            exec_spread_bps = (exec_ask - exec_bid) / (mid + EPS) * 10000
            df["exec_spread_bps"] = exec_spread_bps
            df["spread_expansion_bps"] = exec_spread_bps - spread_bps
            df["spread_expansion_ratio"] = exec_spread_bps / (spread_bps + EPS)

        return df

    # =========================================================================
    # Section 5: Cross-Feature Interactions
    # =========================================================================

    def _build_cross_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all cross-feature interaction features (Section 5 of features.md).

        Includes:
        - Trade Aggression × Order Book State
        - Trade Aggression × Book Change
        - Order Book State × Book Change
        - Order Book State × Execution Price
        - Trade Aggression × Execution Price
        - Multi-way interactions
        """
        LOG.info("Building cross-feature interactions...")

        df = self._build_trade_book_interactions(df)
        df = self._build_trade_change_interactions(df)
        df = self._build_book_change_interactions(df)
        df = self._build_book_exec_interactions(df)
        df = self._build_trade_exec_interactions(df)
        df = self._build_multiway_interactions(df)

        return df

    def _build_trade_book_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build Trade Aggression × Order Book State interactions."""
        best_mid = get_column_if_exists(df, "best_mid")
        if best_mid is None:
            best_mid = get_column_if_exists(df, "mid")
        vol_imb_1 = get_column_if_exists(df, "vol_imb_k1")
        vol_imbalance_ratio_1s = get_column_if_exists(df, "vol_imbalance_ratio_1s")

        # VWAP offset features (requires mid from order book)
        for window in self.TRADE_WINDOWS:
            suffix = f"_{window}"
            vwap = get_column_if_exists(df, f"vwap{suffix}")
            vwap_buy = get_column_if_exists(df, f"vwap_buy{suffix}")
            vwap_sell = get_column_if_exists(df, f"vwap_sell{suffix}")

            if best_mid is not None:
                if vwap is not None:
                    df[f"vwap_offset_bps{suffix}"] = compute_bps_offset(vwap, best_mid)
                if vwap_buy is not None:
                    df[f"vwap_buy_offset_bps{suffix}"] = compute_bps_offset(vwap_buy, best_mid)
                if vwap_sell is not None:
                    df[f"vwap_sell_offset_bps{suffix}"] = compute_bps_offset(vwap_sell, best_mid)

        # Sign agreement between trade and book
        if vol_imbalance_ratio_1s is not None and vol_imb_1 is not None:
            df["sign_agree_trade_book"] = ((vol_imbalance_ratio_1s > 0) == (vol_imb_1 > 0)).astype(int)

            # Strong agreement
            df["strong_agreement"] = (
                (vol_imbalance_ratio_1s.abs() > 0.3) & (vol_imb_1.abs() > 0.3) & (df["sign_agree_trade_book"] == 1)
            ).astype(int)

            # Divergence
            df["trade_book_divergence"] = (
                ((vol_imbalance_ratio_1s > 0.1) & (vol_imb_1 < -0.2))
                | ((vol_imbalance_ratio_1s < -0.1) & (vol_imb_1 > 0.2))
            ).astype(int)

            # Divergence magnitude
            df["divergence_magnitude"] = vol_imbalance_ratio_1s * (-vol_imb_1)

        # VWAP vs microprice comparison
        vwap_offset_bps_1s = get_column_if_exists(df, "vwap_offset_bps_1s")
        microprice_offset_bps_1 = get_column_if_exists(df, "microprice_offset_bps_k1")

        if vwap_offset_bps_1s is not None and microprice_offset_bps_1 is not None:
            df["vwap_microprice_agree"] = ((vwap_offset_bps_1s > 0) == (microprice_offset_bps_1 > 0)).astype(int)
            df["vwap_microprice_spread"] = vwap_offset_bps_1s - microprice_offset_bps_1

        return df

    def _build_trade_change_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build Trade Aggression × Book Change interactions."""
        vol_imbalance_ratio_1s = get_column_if_exists(df, "vol_imbalance_ratio_1s")
        pull_imbalance_1 = get_column_if_exists(df, "pull_imbalance_k1")
        delta_depth_imbalance_norm_1 = get_column_if_exists(df, "delta_depth_imbalance_norm_k1")

        if vol_imbalance_ratio_1s is not None and pull_imbalance_1 is not None:
            # Trade consumption detection
            df["trade_consuming_asks"] = ((vol_imbalance_ratio_1s > 0.3) & (pull_imbalance_1 < -0.3)).astype(int)

            df["trade_consuming_bids"] = ((vol_imbalance_ratio_1s < -0.3) & (pull_imbalance_1 > 0.3)).astype(int)

            # Consumption alignment
            df["consumption_alignment"] = vol_imbalance_ratio_1s * (-pull_imbalance_1)

        if vol_imbalance_ratio_1s is not None and delta_depth_imbalance_norm_1 is not None:
            # Absorption detection
            df["absorption_bid"] = ((vol_imbalance_ratio_1s < -0.3) & (delta_depth_imbalance_norm_1 >= 0)).astype(int)

            df["absorption_ask"] = ((vol_imbalance_ratio_1s > 0.3) & (delta_depth_imbalance_norm_1 <= 0)).astype(int)

            # Absorption strength
            df["absorption_strength"] = vol_imbalance_ratio_1s * (-delta_depth_imbalance_norm_1)

        return df

    def _build_book_change_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build Order Book State × Book Change interactions."""
        vol_imb_1 = get_column_if_exists(df, "vol_imb_k1")
        delta_depth_imbalance_norm_1 = get_column_if_exists(df, "delta_depth_imbalance_norm_k1")
        refill_imbalance_1 = get_column_if_exists(df, "refill_imbalance_k1")
        pull_imbalance_1 = get_column_if_exists(df, "pull_imbalance_k1")
        best_bid_reset = get_column_if_exists(df, "best_bid_reset_k1")
        best_ask_reset = get_column_if_exists(df, "best_ask_reset_k1")

        # Imbalance momentum
        if vol_imb_1 is not None and delta_depth_imbalance_norm_1 is not None:
            df["imbalance_momentum"] = vol_imb_1 * delta_depth_imbalance_norm_1

        # Imbalance building/fading
        if vol_imb_1 is not None and refill_imbalance_1 is not None:
            df["imbalance_building"] = ((vol_imb_1 > 0) & (refill_imbalance_1 > 0.2)).astype(int)

        if vol_imb_1 is not None and pull_imbalance_1 is not None:
            df["imbalance_fading"] = ((vol_imb_1 > 0) & (pull_imbalance_1 > 0.2)).astype(int)

        # Best level stability vs imbalance
        if vol_imb_1 is not None and best_bid_reset is not None:
            df["stable_wall_bid"] = ((vol_imb_1 > 0.5) & (best_bid_reset == 0)).astype(int)

        if vol_imb_1 is not None and best_ask_reset is not None:
            df["stable_wall_ask"] = ((vol_imb_1 < -0.5) & (best_ask_reset == 0)).astype(int)

        if vol_imb_1 is not None and best_bid_reset is not None and best_ask_reset is not None:
            df["unstable_book"] = ((vol_imb_1.abs() < 0.2) & ((best_bid_reset == 1) | (best_ask_reset == 1))).astype(
                int
            )

        return df

    def _build_book_exec_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build Order Book State × Execution Price interactions."""
        vol_imb_1 = get_column_if_exists(df, "vol_imb_k1")
        slippage_bid_bps = get_column_if_exists(df, "slippage_bid_bps")
        slippage_ask_bps = get_column_if_exists(df, "slippage_ask_bps")
        slippage_total_bps = get_column_if_exists(df, "slippage_total_bps")
        microprice_k1 = get_column_if_exists(df, "microprice_k1")
        exec_bid = get_column_if_exists(df, "bid")
        exec_ask = get_column_if_exists(df, "ask")
        mid = get_column_if_exists(df, "mid")
        spread_bps = get_column_if_exists(df, "best_spread_bps")
        if spread_bps is None:
            spread_bps = get_column_if_exists(df, "spread_bps")
        depth_quality = get_column_if_exists(df, "depth_quality")

        # Slippage vs imbalance alignment
        if vol_imb_1 is not None and slippage_bid_bps is not None and slippage_ask_bps is not None:
            df["slippage_imbalance_align"] = (
                ((vol_imb_1 > 0) & (slippage_ask_bps > slippage_bid_bps))
                | ((vol_imb_1 < 0) & (slippage_bid_bps > slippage_ask_bps))
            ).astype(int)

            # Slippage ratio vs imbalance correlation
            slippage_ratio = slippage_ask_bps / (slippage_bid_bps + EPS)
            df["slippage_imbalance_corr"] = slippage_ratio * vol_imb_1

        # Microprice vs execution mid gap
        if microprice_k1 is not None and exec_bid is not None and exec_ask is not None and mid is not None:
            exec_mid = (exec_bid + exec_ask) / 2
            df["microprice_exec_gap"] = (microprice_k1 - exec_mid) / (mid + EPS) * 10000

        # Depth quality per spread
        if depth_quality is not None and spread_bps is not None:
            df["depth_quality_per_spread"] = depth_quality / (spread_bps + EPS)

        # Spread-depth ratio
        if spread_bps is not None and slippage_total_bps is not None:
            df["spread_depth_ratio"] = spread_bps / (slippage_total_bps + EPS)

        return df

    def _build_trade_exec_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build Trade Aggression × Execution Price interactions."""
        trades_per_sec_pct_1s = get_column_if_exists(df, "trades_per_sec_pct_1s")
        slippage_pct = get_column_if_exists(df, "slippage_pct")
        vwap_1s = get_column_if_exists(df, "vwap_1s")
        exec_bid = get_column_if_exists(df, "bid")
        exec_ask = get_column_if_exists(df, "ask")
        mid = get_column_if_exists(df, "mid")

        # Trade intensity vs slippage
        if trades_per_sec_pct_1s is not None and slippage_pct is not None:
            df["efficient_trading"] = ((trades_per_sec_pct_1s > 0.7) & (slippage_pct < 0.3)).astype(int)
            df["stressed_trading"] = ((trades_per_sec_pct_1s > 0.7) & (slippage_pct > 0.7)).astype(int)
            df["activity_slippage_divergence"] = trades_per_sec_pct_1s - slippage_pct

        # VWAP vs execution price
        if vwap_1s is not None and exec_bid is not None and exec_ask is not None and mid is not None:
            exec_mid = (exec_bid + exec_ask) / 2
            df["trade_execution_edge"] = (vwap_1s - exec_mid) / (mid + EPS) * 10000

        return df

    def _build_multiway_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build multi-way interaction features (confirmation score, regime detection, etc.)."""
        vol_imbalance_ratio_1s = get_column_if_exists(df, "vol_imbalance_ratio_1s")
        vol_imb_1 = get_column_if_exists(df, "vol_imb_k1")
        microprice_offset_bps_1 = get_column_if_exists(df, "microprice_offset_bps_k1")
        delta_depth_imbalance_norm_1 = get_column_if_exists(df, "delta_depth_imbalance_norm_k1")
        sign_agree_trade_book = get_column_if_exists(df, "sign_agree_trade_book")
        trade_consuming_asks = get_column_if_exists(df, "trade_consuming_asks")
        trade_consuming_bids = get_column_if_exists(df, "trade_consuming_bids")
        absorption_bid = get_column_if_exists(df, "absorption_bid")
        absorption_ask = get_column_if_exists(df, "absorption_ask")
        trades_per_sec_pct_1s = get_column_if_exists(df, "trades_per_sec_pct_1s")
        total_activity_pct_1 = get_column_if_exists(df, "total_activity_pct_k1")
        spread_bps = get_column_if_exists(df, "best_spread_bps")
        if spread_bps is None:
            spread_bps = get_column_if_exists(df, "spread_bps")
        depth_quality = get_column_if_exists(df, "depth_quality")

        # Confirmation score: count agreeing signals
        signals = []
        if vol_imbalance_ratio_1s is not None:
            signals.append(np.sign(vol_imbalance_ratio_1s))
        if vol_imb_1 is not None:
            signals.append(np.sign(vol_imb_1))
        if microprice_offset_bps_1 is not None:
            signals.append(np.sign(microprice_offset_bps_1))
        if delta_depth_imbalance_norm_1 is not None:
            signals.append(np.sign(delta_depth_imbalance_norm_1))

        if len(signals) >= 2:
            signals_sum = sum(signals)
            df["confirmation_score"] = signals_sum / len(signals)
            df["signal_agreement"] = (signals_sum / len(signals)).abs()

        # Regime detection
        signal_agreement = get_column_if_exists(df, "signal_agreement")

        if signal_agreement is not None:
            # Trending regime
            if (
                sign_agree_trade_book is not None
                and trade_consuming_asks is not None
                and trade_consuming_bids is not None
            ):
                df["regime_trending"] = (
                    (sign_agree_trade_book == 1)
                    & ((trade_consuming_asks == 1) | (trade_consuming_bids == 1))
                    & (signal_agreement > 0.5)
                ).astype(int)

            # Mean-reverting regime
            if sign_agree_trade_book is not None and absorption_bid is not None and absorption_ask is not None:
                df["regime_mean_reverting"] = (
                    (sign_agree_trade_book == 0)
                    & ((absorption_bid == 1) | (absorption_ask == 1))
                    & (signal_agreement < 0.3)
                ).astype(int)

            # Volatile regime
            if trades_per_sec_pct_1s is not None and total_activity_pct_1 is not None:
                df["regime_volatile"] = (
                    (trades_per_sec_pct_1s > 0.8) & (total_activity_pct_1 > 0.8) & (signal_agreement < 0.3)
                ).astype(int)

        # Quality-weighted signal
        confirmation_score = get_column_if_exists(df, "confirmation_score")
        if (
            confirmation_score is not None
            and signal_agreement is not None
            and depth_quality is not None
            and spread_bps is not None
        ):
            spread_factor = 1 / (spread_bps + 1)
            df["quality_weighted_signal"] = confirmation_score * signal_agreement * depth_quality * spread_factor

        return df

    # =========================================================================
    # Post-processing
    # =========================================================================

    def _drop_non_stationary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop raw non-stationary inputs that should not be ML features."""
        trade_raw_cols = []
        for window in self.TRADE_WINDOWS:
            trade_raw_cols.extend(
                [
                    f"buy_qty_{window}",
                    f"sell_qty_{window}",
                    f"buy_count_{window}",
                    f"sell_count_{window}",
                    f"total_qty_{window}",
                    f"total_count_{window}",
                    f"vwap_{window}",
                    f"vwap_buy_{window}",
                    f"vwap_sell_{window}",
                    f"qty_per_sec_{window}",
                    f"trades_per_sec_{window}",
                ]
            )

        book_raw_cols = ["best_bid", "best_ask", "best_mid", "best_spread"]
        exec_raw_cols = ["bid", "ask", "mid", "spread"]

        microprice_cols = []
        for k in self.DEPTH_LEVELS:
            microprice_cols.extend([f"microprice_k{k}", f"microprice_{k}"])

        book_change_raw_cols = ["delta_sec"]
        for k in self.DEPTH_LEVELS:
            for prefix in [
                "delta_depth_bid",
                "delta_depth_ask",
                "delta_depth_bid_rate",
                "delta_depth_ask_rate",
                "pull_bid_rate",
                "pull_ask_rate",
                "refill_bid_rate",
                "refill_ask_rate",
            ]:
                book_change_raw_cols.extend([f"{prefix}_k{k}", f"{prefix}_{k}"])

        drop_cols = set(trade_raw_cols + book_raw_cols + exec_raw_cols + microprice_cols + book_change_raw_cols)
        existing = [col for col in drop_cols if col in df.columns]
        if existing:
            df = df.drop(columns=existing)
        return df

    def _postprocess_features(self, df: pd.DataFrame, stage_context: Context) -> pd.DataFrame:
        """Post-process features: handle NaN/Inf, filter to populate range."""
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        df = self._drop_non_stationary_columns(df)

        # Filter to populate_dt_range
        df = df[
            (df.index >= stage_context.populate_dt_range.start_dt)
            & (df.index <= stage_context.populate_dt_range.end_dt)
        ]

        # Drop rows with too many NaN values (keep rows with at least 30% valid data)
        threshold = int(len(df.columns) * 0.3)
        df = df.dropna(thresh=threshold)

        # Forward-fill remaining NaN values (with limit)
        df = df.ffill(limit=3)

        LOG.info(f"AlignAndBuildFeaturesStep: After post-processing: {len(df)} rows, {len(df.columns)} columns")

        return df
