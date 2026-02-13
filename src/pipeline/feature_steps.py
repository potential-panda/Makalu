"""
Feature engineering pipeline steps.

This module implements comprehensive feature engineering as documented in doc/features.md.
All features are designed to be stationary (normalized) for ML model compatibility.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from qate.core.model import Market

from .converter import DataPointConverter
from .framework import PipelineContext, StageContext, Step

LOG = logging.getLogger(__name__)

# Bucket name constant
FEATURE_DATA_BUCKET = "feature_data"

# Rolling windows for z-score and percentile calculations
# Multiple windows capture different timescales (short-term reactivity vs long-term stability)
ROLLING_WINDOWS = ["1min", "3min", "5min"]
DEFAULT_ROLLING_WINDOW = "1min"  # Default for single-window features

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


# =============================================================================
# Context and Step Definitions
# =============================================================================


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

    All features are normalized/stationary for ML model compatibility.

    Architecture: source_df holds raw aligned data (read-only). Features are
    collected in a dictionary and converted to DataFrame at the end to avoid
    fragmentation warnings.
    """

    # Time windows for trade aggression features
    TRADE_WINDOWS = ["1s", "3s", "5s"]

    # Depth levels for order book features
    DEPTH_LEVELS = [1, 3, 5, 10]

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
        sections = [
            ("Trade Aggression", self._build_trade_aggression_features),
            ("Order Book State", self._build_order_book_state_features),
            ("Book Change", self._build_book_change_features),
            ("Execution Price", self._build_execution_price_features),
            ("Cross-Feature Interactions", self._build_cross_feature_interactions),
        ]
        for section_name, build_fn in sections:
            before = len(features)
            build_fn(source_df, features)
            after = len(features)
            LOG.info(f"  {section_name}: +{after - before} features (total: {after})")

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
        """Align all data sources into a single DataFrame with DatetimeIndex.

        Sparse data sources are forward-filled to carry forward the last known
        values to timestamps where data is missing:
        - Trade aggression: trades are sparse events; ffill carries last known metrics
        - Execution price: some order books lack depth for the execution size;
          ffill uses last known execution price as best estimate
        """
        dfs_to_merge = []
        ffill_columns = []

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

                # Track columns that need forward-fill after outer join
                if name in ("trade_aggression", "execution_price"):
                    ffill_columns.extend(df.columns.tolist())

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

        # Forward-fill sparse data source columns to carry forward last known values
        if ffill_columns:
            cols_to_ffill = [c for c in ffill_columns if c in result.columns]
            if cols_to_ffill:
                result[cols_to_ffill] = result[cols_to_ffill].ffill()
                LOG.info(f"AlignAndBuildFeaturesStep: Forward-filled {len(cols_to_ffill)} columns (trade_aggression + execution_price)")

        return result

    # =========================================================================
    # Section 1: Trade Aggression Features
    # =========================================================================

    def _build_trade_aggression_features(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build all trade aggression features (Section 1 of features.md)."""
        LOG.info("Building trade aggression features...")

        for window in self.TRADE_WINDOWS:
            self._build_trade_features_for_window(source_df, features, window)

        # Cross-window features (momentum, sign agreement, magnitude dynamics)
        self._build_cross_window_features(source_df, features)

    def _build_trade_features_for_window(
        self, source_df: pd.DataFrame, features: Dict[str, pd.Series], window: str
    ) -> None:
        """Build trade aggression features for a specific time window."""
        suffix = f"_{window}"

        # Get raw columns for this window
        buy_qty = source_df[f"buy_qty{suffix}"]
        sell_qty = source_df[f"sell_qty{suffix}"]
        buy_count = source_df[f"buy_count{suffix}"]
        sell_count = source_df[f"sell_count{suffix}"]
        total_qty = source_df[f"total_qty{suffix}"]
        total_count = source_df[f"total_count{suffix}"]
        vwap = source_df[f"vwap{suffix}"]
        vwap_buy = source_df[f"vwap_buy{suffix}"]
        vwap_sell = source_df[f"vwap_sell{suffix}"]
        qty_per_sec = source_df[f"qty_per_sec{suffix}"]
        trades_per_sec = source_df[f"trades_per_sec{suffix}"]

        features[f"ta_vol_imbalance_ratio{suffix}"] = compute_imbalance(buy_qty, sell_qty)
        features[f"ta_count_imbalance_ratio{suffix}"] = compute_imbalance(buy_count, sell_count)
        features[f"ta_vwap_buy_sell_spread_bps{suffix}"] = (vwap_buy - vwap_sell) / (vwap + EPS) * 10000

        avg_buy_size = buy_qty / (buy_count + EPS)
        avg_sell_size = sell_qty / (sell_count + EPS)
        avg_size = (buy_qty + sell_qty) / (buy_count + sell_count + EPS)

        # Size asymmetry (which side trades bigger)
        features[f"ta_size_asymmetry_norm{suffix}"] = compute_imbalance(avg_buy_size, avg_sell_size)

        # Avg size z-score and percentile (multiple rolling windows)
        for roll_window in ROLLING_WINDOWS:
            roll_suffix = f"_{roll_window.replace('min', 'm')}"
            features[f"ta_avg_size_zscore{suffix}{roll_suffix}"] = compute_zscore(avg_size, roll_window)
            features[f"ta_avg_size_pct{suffix}{roll_suffix}"] = compute_percentile(avg_size, roll_window)

            features[f"ta_qty_per_sec_zscore{suffix}{roll_suffix}"] = compute_zscore(qty_per_sec, roll_window)
            features[f"ta_qty_per_sec_pct{suffix}{roll_suffix}"] = compute_percentile(qty_per_sec, roll_window)

            features[f"ta_trades_per_sec_zscore{suffix}{roll_suffix}"] = compute_zscore(trades_per_sec, roll_window)
            features[f"ta_trades_per_sec_pct{suffix}{roll_suffix}"] = compute_percentile(trades_per_sec, roll_window)

        features[f"ta_buy_qty_share{suffix}"] = buy_qty / (total_qty + EPS)
        features[f"ta_buy_count_share{suffix}"] = buy_count / (total_count + EPS)

        buy_qty_share = features[f"ta_buy_qty_share{suffix}"]
        buy_count_share = features[f"ta_buy_count_share{suffix}"]
        features[f"ta_qty_count_divergence{suffix}"] = buy_qty_share - buy_count_share

    def _build_cross_window_features(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build cross-window features for trade aggression (momentum, sign agreement, etc.)."""
        # Get vol_imbalance_ratio for each window (computed by _build_trade_features_for_window)
        ratio_1s = features["ta_vol_imbalance_ratio_1s"]
        ratio_3s = features["ta_vol_imbalance_ratio_3s"]
        ratio_5s = features["ta_vol_imbalance_ratio_5s"]

        # Momentum features (difference between windows)
        features["ta_imbalance_momentum_1s_3s"] = ratio_1s - ratio_3s
        features["ta_imbalance_momentum_1s_5s"] = ratio_1s - ratio_5s
        features["ta_imbalance_momentum_3s_5s"] = ratio_3s - ratio_5s

        # Direction consistency (sign agreement)
        features["ta_imbalance_sign_agree_1s_3s"] = (np.sign(ratio_1s) == np.sign(ratio_3s)).astype(int)
        features["ta_imbalance_sign_agree_1s_5s"] = (np.sign(ratio_1s) == np.sign(ratio_5s)).astype(int)
        features["ta_imbalance_sign_agree_3s_5s"] = (np.sign(ratio_3s) == np.sign(ratio_5s)).astype(int)
        features["ta_imbalance_sign_agree_all"] = (
            (features["ta_imbalance_sign_agree_1s_3s"] == 1) & (features["ta_imbalance_sign_agree_1s_5s"] == 1)
        ).astype(int)

        # Magnitude dynamics (strengthening/weakening)
        features["ta_imbalance_strengthening_1s_3s"] = (ratio_1s.abs() > ratio_3s.abs()).astype(int)
        features["ta_imbalance_strengthening_1s_5s"] = (ratio_1s.abs() > ratio_5s.abs()).astype(int)
        features["ta_imbalance_weakening_1s_5s"] = (ratio_1s.abs() < ratio_5s.abs()).astype(int)

        # Signed flow indicators (boolean)
        features["ta_strong_buy_signal"] = (
            (ratio_1s > 0.3)
            & (features["ta_imbalance_sign_agree_all"] == 1)
            & (features["ta_imbalance_strengthening_1s_5s"] == 1)
        ).astype(int)

        features["ta_strong_sell_signal"] = (
            (ratio_1s < -0.3)
            & (features["ta_imbalance_sign_agree_all"] == 1)
            & (features["ta_imbalance_strengthening_1s_5s"] == 1)
        ).astype(int)

        features["ta_exhaustion_signal"] = (
            (ratio_5s.abs() > 0.3)
            & (features["ta_imbalance_weakening_1s_5s"] == 1)
            & (features["ta_imbalance_sign_agree_1s_5s"] == 1)
        ).astype(int)

        features["ta_reversal_signal"] = (
            (ratio_5s.abs() > 0.3) & (features["ta_imbalance_sign_agree_1s_5s"] == 0)
        ).astype(int)

    # =========================================================================
    # Section 2: Order Book State Features
    # =========================================================================

    def _build_order_book_state_features(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build all order book state features (Section 2 of features.md)."""
        LOG.info("Building order book state features...")

        best_mid = source_df["best_mid"]
        for k in self.DEPTH_LEVELS:
            microprice = source_df[f"microprice_k{k}"]
            features[f"bs_microprice_offset_bps_k{k}"] = compute_bps_offset(microprice, best_mid)

        vol_imb_1 = source_df["vol_imb_k1"]
        vol_imb_5 = source_df["vol_imb_k5"]
        vol_imb_10 = source_df["vol_imb_k10"]

        features["bs_imbalance_gradient"] = vol_imb_1 - vol_imb_5
        features["bs_imbalance_divergence"] = (vol_imb_1 - vol_imb_10).abs()

        spread_bps = source_df["spread_bps"]
        microprice_offset_bps_1 = features["bs_microprice_offset_bps_k1"]
        features["bs_microprice_offset_norm"] = microprice_offset_bps_1 / (spread_bps + EPS)

        self._build_order_book_rolling_aggregates(source_df, features)

    def _build_order_book_rolling_aggregates(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build rolling mean/std for order book features."""
        windows = {"1s": "1s", "3s": "3s", "5s": "5s"}

        # Features to compute rolling stats for
        features_to_roll = {
            "spread_bps": source_df["spread_bps"],
            "vol_imb_k1": source_df["vol_imb_k1"],
            "vol_imb_k5": source_df["vol_imb_k5"],
            "microprice_offset_bps_k1": features["bs_microprice_offset_bps_k1"],
        }
        for name, series in features_to_roll.items():
            for window_name, window_size in windows.items():
                rolling_op = series.rolling(window_size, closed="left")
                features[f"bs_{name}_mean_{window_name}"] = rolling_op.mean()
                features[f"bs_{name}_std_{window_name}"] = rolling_op.std()

    # =========================================================================
    # Section 3: Book Change Features
    # =========================================================================

    def _build_book_change_features(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build all book change features (Section 3 of features.md)."""
        LOG.info("Building book change features...")

        for k in self.DEPTH_LEVELS:
            self._build_book_change_features_for_k(source_df, features, k)

        # Cross-k features (gradient between shallow and deep)
        self._build_book_change_cross_k_features(source_df, features)

    def _build_book_change_features_for_k(
        self, source_df: pd.DataFrame, features: Dict[str, pd.Series], k: int
    ) -> None:
        """Build book change features for a specific depth level k."""
        suffix = f"_k{k}"

        # Get raw columns
        delta_depth_bid = source_df[f"delta_depth_bid{suffix}"]
        delta_depth_ask = source_df[f"delta_depth_ask{suffix}"]
        pull_bid_rate = source_df[f"pull_bid_rate{suffix}"]
        pull_ask_rate = source_df[f"pull_ask_rate{suffix}"]
        refill_bid_rate = source_df[f"refill_bid_rate{suffix}"]
        refill_ask_rate = source_df[f"refill_ask_rate{suffix}"]

        features[f"bc_delta_depth_imbalance_norm{suffix}"] = compute_imbalance_abs(delta_depth_bid, delta_depth_ask)
        features[f"bc_pull_imbalance{suffix}"] = compute_imbalance(pull_bid_rate, pull_ask_rate)
        features[f"bc_refill_imbalance{suffix}"] = compute_imbalance(refill_bid_rate, refill_ask_rate)

        total_pull_rate = pull_bid_rate + pull_ask_rate
        total_refill_rate = refill_bid_rate + refill_ask_rate
        features[f"bc_pull_refill_ratio{suffix}"] = total_pull_rate / (total_refill_rate + EPS)

        total_activity = pull_bid_rate + pull_ask_rate + refill_bid_rate + refill_ask_rate

        # Z-score and percentile with multiple rolling windows
        for roll_window in ROLLING_WINDOWS:
            roll_suffix = f"_{roll_window.replace('min', 'm')}"
            features[f"bc_total_pull_zscore{suffix}{roll_suffix}"] = compute_zscore(total_pull_rate, roll_window)
            features[f"bc_total_activity_zscore{suffix}{roll_suffix}"] = compute_zscore(total_activity, roll_window)
            features[f"bc_total_activity_pct{suffix}{roll_suffix}"] = compute_percentile(total_activity, roll_window)

    def _build_book_change_cross_k_features(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build cross-k features comparing shallow vs deep book changes."""
        # Pull imbalance gradient (k1 vs k5)
        pull_imb_k1 = features["bc_pull_imbalance_k1"]
        pull_imb_k5 = features["bc_pull_imbalance_k5"]
        features["bc_pull_imbalance_gradient"] = pull_imb_k1 - pull_imb_k5

        # Refill imbalance gradient (k1 vs k5)
        refill_imb_k1 = features["bc_refill_imbalance_k1"]
        refill_imb_k5 = features["bc_refill_imbalance_k5"]
        features["bc_refill_imbalance_gradient"] = refill_imb_k1 - refill_imb_k5

    # =========================================================================
    # Section 4: Execution Price Features
    # =========================================================================

    def _build_execution_price_features(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build all execution price features (Section 4 of features.md)."""
        LOG.info("Building execution price features...")

        # Get required columns
        best_bid = source_df["best_bid"]
        best_ask = source_df["best_ask"]
        exec_bid = source_df["bid"]
        exec_ask = source_df["ask"]
        mid = source_df["mid"]
        spread_bps = source_df["spread_bps"]

        # Slippage metrics in bps
        features["ep_slippage_bid_bps"] = (best_bid - exec_bid) / (best_bid + EPS) * 10000
        features["ep_slippage_ask_bps"] = (exec_ask - best_ask) / (best_ask + EPS) * 10000

        slippage_bid_bps = features["ep_slippage_bid_bps"]
        slippage_ask_bps = features["ep_slippage_ask_bps"]

        slippage_total_bps = slippage_bid_bps + slippage_ask_bps
        features["ep_slippage_total_bps"] = slippage_total_bps

        # Slippage magnitude features (normalized) with multiple rolling windows
        for roll_window in ROLLING_WINDOWS:
            roll_suffix = f"_{roll_window.replace('min', 'm')}"
            features[f"ep_slippage_zscore{roll_suffix}"] = compute_zscore(slippage_total_bps, roll_window)
            features[f"ep_slippage_pct{roll_suffix}"] = compute_percentile(slippage_total_bps, roll_window)

        # Slippage asymmetry
        features["ep_slippage_asymmetry"] = (slippage_ask_bps - slippage_bid_bps) / (slippage_total_bps + EPS)

        # Depth quality score
        features["ep_depth_quality"] = 1 / (slippage_total_bps + 1)

        exec_spread_bps = (exec_ask - exec_bid) / (mid + EPS) * 10000
        features["ep_exec_spread_bps"] = exec_spread_bps
        features["ep_spread_expansion_bps"] = exec_spread_bps - spread_bps
        features["ep_spread_expansion_ratio"] = exec_spread_bps / (spread_bps + EPS)

    # =========================================================================
    # Section 5: Cross-Feature Interactions
    # =========================================================================

    def _build_cross_feature_interactions(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build all cross-feature interaction features (Section 5 of features.md)."""
        LOG.info("Building cross-feature interactions...")

        self._build_trade_book_interactions(source_df, features)
        self._build_trade_change_interactions(source_df, features)
        self._build_book_change_interactions(source_df, features)
        self._build_book_exec_interactions(source_df, features)
        self._build_trade_exec_interactions(source_df, features)
        self._build_multiway_interactions(source_df, features)

    def _build_trade_book_interactions(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build Trade Aggression x Order Book State interactions."""
        mid = source_df["mid"]
        vol_imb_1 = source_df["vol_imb_k1"]
        vol_imbalance_ratio_1s = features["ta_vol_imbalance_ratio_1s"]

        # VWAP offset features (requires mid from order book)
        for window in self.TRADE_WINDOWS:
            suffix = f"_{window}"
            vwap = source_df[f"vwap{suffix}"]
            vwap_buy = source_df[f"vwap_buy{suffix}"]
            vwap_sell = source_df[f"vwap_sell{suffix}"]
            features[f"cf_vwap_offset_bps{suffix}"] = compute_bps_offset(vwap, mid)
            features[f"cf_vwap_buy_offset_bps{suffix}"] = compute_bps_offset(vwap_buy, mid)
            features[f"cf_vwap_sell_offset_bps{suffix}"] = compute_bps_offset(vwap_sell, mid)

        # Sign agreement between trade and book
        features["cf_sign_agree_trade_book"] = ((vol_imbalance_ratio_1s > 0) == (vol_imb_1 > 0)).astype(int)

        # Strong agreement
        features["cf_strong_agreement"] = (
            (vol_imbalance_ratio_1s.abs() > 0.3)
            & (vol_imb_1.abs() > 0.3)
            & (features["cf_sign_agree_trade_book"] == 1)
        ).astype(int)

        # Divergence
        features["cf_trade_book_divergence"] = (
            ((vol_imbalance_ratio_1s > 0.1) & (vol_imb_1 < -0.2))
            | ((vol_imbalance_ratio_1s < -0.1) & (vol_imb_1 > 0.2))
        ).astype(int)

        # Divergence magnitude
        features["cf_divergence_magnitude"] = vol_imbalance_ratio_1s * (-vol_imb_1)

        # VWAP vs microprice comparison
        vwap_offset_bps_1s = features["cf_vwap_offset_bps_1s"]
        microprice_offset_bps_1 = features["bs_microprice_offset_bps_k1"]

        features["cf_vwap_microprice_agree"] = ((vwap_offset_bps_1s > 0) == (microprice_offset_bps_1 > 0)).astype(int)
        features["cf_vwap_microprice_spread"] = vwap_offset_bps_1s - microprice_offset_bps_1

    def _build_trade_change_interactions(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build Trade Aggression x Book Change interactions."""
        vol_imbalance_ratio_1s = features["ta_vol_imbalance_ratio_1s"]
        pull_imbalance_1 = features["bc_pull_imbalance_k1"]
        delta_depth_imbalance_norm_1 = features["bc_delta_depth_imbalance_norm_k1"]

        features["cf_trade_consuming_asks"] = ((vol_imbalance_ratio_1s > 0.3) & (pull_imbalance_1 < -0.3)).astype(int)

        features["cf_trade_consuming_bids"] = ((vol_imbalance_ratio_1s < -0.3) & (pull_imbalance_1 > 0.3)).astype(int)

        features["cf_consumption_alignment"] = vol_imbalance_ratio_1s * (-pull_imbalance_1)

        features["cf_absorption_bid"] = ((vol_imbalance_ratio_1s < -0.3) & (delta_depth_imbalance_norm_1 >= 0)).astype(
            int
        )

        features["cf_absorption_ask"] = ((vol_imbalance_ratio_1s > 0.3) & (delta_depth_imbalance_norm_1 <= 0)).astype(
            int
        )

        features["cf_absorption_strength"] = vol_imbalance_ratio_1s * (-delta_depth_imbalance_norm_1)

    def _build_book_change_interactions(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build Order Book State x Book Change interactions."""
        vol_imb_1 = source_df["vol_imb_k1"]
        best_bid_reset = source_df["best_bid_reset_k1"]
        best_ask_reset = source_df["best_ask_reset_k1"]
        refill_imbalance_1 = features["bc_refill_imbalance_k1"]
        pull_imbalance_1 = features["bc_pull_imbalance_k1"]
        delta_depth_imbalance_norm_1 = features["bc_delta_depth_imbalance_norm_k1"]

        features["cf_imbalance_momentum"] = vol_imb_1 * delta_depth_imbalance_norm_1
        features["cf_imbalance_building"] = ((vol_imb_1 > 0) & (refill_imbalance_1 > 0.2)).astype(int)
        features["cf_imbalance_fading"] = ((vol_imb_1 > 0) & (pull_imbalance_1 > 0.2)).astype(int)
        features["cf_stable_wall_bid"] = ((vol_imb_1 > 0.5) & (best_bid_reset == 0)).astype(int)
        features["cf_stable_wall_ask"] = ((vol_imb_1 < -0.5) & (best_ask_reset == 0)).astype(int)
        features["cf_unstable_book"] = (
            (vol_imb_1.abs() < 0.2) & ((best_bid_reset == 1) | (best_ask_reset == 1))
        ).astype(int)

    def _build_book_exec_interactions(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build Order Book State x Execution Price interactions."""
        mid = source_df["mid"]
        vol_imb_1 = source_df["vol_imb_k1"]
        microprice_k1 = source_df["microprice_k1"]
        spread_bps = source_df["spread_bps"]
        slippage_bid_bps = features["ep_slippage_bid_bps"]
        slippage_ask_bps = features["ep_slippage_ask_bps"]
        slippage_total_bps = features["ep_slippage_total_bps"]
        depth_quality = features["ep_depth_quality"]

        features["cf_slippage_imbalance_align"] = (
            ((vol_imb_1 > 0) & (slippage_ask_bps > slippage_bid_bps))
            | ((vol_imb_1 < 0) & (slippage_bid_bps > slippage_ask_bps))
        ).astype(int)

        slippage_ratio = slippage_ask_bps / (slippage_bid_bps + EPS)
        features["cf_slippage_imbalance_corr"] = slippage_ratio * vol_imb_1
        features["cf_microprice_exec_gap"] = (microprice_k1 - mid) / (mid + EPS) * 10000
        features["cf_depth_quality_per_spread"] = depth_quality / (spread_bps + EPS)
        features["cf_spread_depth_ratio"] = spread_bps / (slippage_total_bps + EPS)

    def _build_trade_exec_interactions(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build Trade Aggression x Execution Price interactions."""
        vwap_1s = source_df["vwap_1s"]
        mid = source_df["mid"]
        # Use 1-minute rolling window (most responsive for short-term prediction)
        trades_per_sec_pct_1s = features["ta_trades_per_sec_pct_1s_1m"]
        slippage_pct = features["ep_slippage_pct_1m"]

        features["cf_efficient_trading"] = ((trades_per_sec_pct_1s > 0.7) & (slippage_pct < 0.3)).astype(int)
        features["cf_stressed_trading"] = ((trades_per_sec_pct_1s > 0.7) & (slippage_pct > 0.7)).astype(int)
        features["cf_activity_slippage_divergence"] = trades_per_sec_pct_1s - slippage_pct
        features["cf_trade_execution_edge"] = (vwap_1s - mid) / (mid + EPS) * 10000

    def _build_multiway_interactions(self, source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
        """Build multi-way interaction features (confirmation score, regime detection, etc.)."""
        vol_imb_1 = source_df["vol_imb_k1"]
        vol_imbalance_ratio_1s = features["ta_vol_imbalance_ratio_1s"]
        microprice_offset_bps_1 = features["bs_microprice_offset_bps_k1"]
        delta_depth_imbalance_norm_1 = features["bc_delta_depth_imbalance_norm_k1"]
        sign_agree_trade_book = features["cf_sign_agree_trade_book"]
        trade_consuming_asks = features["cf_trade_consuming_asks"]
        trade_consuming_bids = features["cf_trade_consuming_bids"]
        absorption_bid = features["cf_absorption_bid"]
        absorption_ask = features["cf_absorption_ask"]
        # Use 1-minute rolling window (most responsive for short-term prediction)
        trades_per_sec_pct_1s = features["ta_trades_per_sec_pct_1s_1m"]
        total_activity_pct_1 = features["bc_total_activity_pct_k1_1m"]
        spread_bps = source_df["spread_bps"]
        depth_quality = features["ep_depth_quality"]

        # Confirmation score: count agreeing signals
        signals = [
            np.sign(vol_imbalance_ratio_1s),
            np.sign(vol_imb_1),
            np.sign(microprice_offset_bps_1),
            np.sign(delta_depth_imbalance_norm_1),
        ]

        signals_sum = sum(signals)
        features["cf_confirmation_score"] = signals_sum / len(signals)
        features["cf_signal_agreement"] = (signals_sum / len(signals)).abs()

        # Regime detection
        signal_agreement = features["cf_signal_agreement"]
        features["cf_regime_trending"] = (
            (sign_agree_trade_book == 1)
            & ((trade_consuming_asks == 1) | (trade_consuming_bids == 1))
            & (signal_agreement > 0.5)
        ).astype(int)

        features["cf_regime_mean_reverting"] = (
            (sign_agree_trade_book == 0) & ((absorption_bid == 1) | (absorption_ask == 1)) & (signal_agreement < 0.3)
        ).astype(int)

        features["cf_regime_volatile"] = (
            (trades_per_sec_pct_1s > 0.8) & (total_activity_pct_1 > 0.8) & (signal_agreement < 0.3)
        ).astype(int)

        confirmation_score = features["cf_confirmation_score"]
        spread_factor = 1 / (spread_bps + 1)
        features["cf_quality_weighted_signal"] = confirmation_score * signal_agreement * depth_quality * spread_factor

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
        df = df.ffill(limit=3)

        LOG.info(f"AlignAndBuildFeaturesStep: After post-processing: {len(df)} rows, {len(df.columns)} columns")

        return df
