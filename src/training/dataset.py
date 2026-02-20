"""
Dataset module for loading and aligning features and labels.
"""

import gc
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _downcast_float64(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 to halve memory usage.

    LightGBM bins features into 256 buckets, so float64 precision is wasted.
    """
    f64_cols = df.select_dtypes(include=["float64"]).columns
    if len(f64_cols) > 0:
        before_mb = df.memory_usage(deep=True).sum() / 1e6
        df[f64_cols] = df[f64_cols].astype(np.float32)
        after_mb = df.memory_usage(deep=True).sum() / 1e6
        LOG.info(f"Downcast float64->float32: {before_mb:.0f} MB -> {after_mb:.0f} MB ({len(f64_cols)} columns)")
    return df


@dataclass
class Dataset:
    """
    Dataset class for managing ML training data.

    Handles loading, aligning, and splitting features and labels
    for LightGBM training.
    """

    features_df: pd.DataFrame
    labels_df: pd.DataFrame
    target_column: str
    feature_columns: List[str] = field(default_factory=list)

    # Train/test split data
    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None

    # Aligned data
    _aligned_features: Optional[pd.DataFrame] = None
    _aligned_labels: Optional[pd.DataFrame] = None  # all label columns as DataFrame

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        """Convert ts column to DatetimeIndex and sort by time."""
        if "ts" in df.columns:
            df = df.copy()
            df.index = pd.to_datetime(df["ts"], unit="s")
            df = df.drop(columns=["ts"])
        return df.sort_index()

    def __post_init__(self):
        """Prepare DataFrames (convert ts -> DatetimeIndex), initialise feature columns, and align."""
        self.features_df = Dataset._prepare_df(self.features_df)
        self.labels_df = Dataset._prepare_df(self.labels_df)
        if not self.feature_columns:
            self.feature_columns = list(self.features_df.columns)
        self._align()

    def _align(
        self,
        tolerance: str = "1s",
        direction: str = "backward",
    ) -> "Dataset":
        """
        Align labels to features by finding the nearest label for each feature timestamp.

        Uses merge_asof to match each feature row with its closest label row.
        _aligned_features holds all feature columns; _aligned_labels holds all label columns.

        Args:
            tolerance: Maximum time difference for matching (e.g., '1s', '100ms', '5s').
            direction: Direction for matching - 'nearest', 'backward', 'forward'.

        Returns:
            Self for method chaining
        """
        LOG.info(f"Aligning labels to features using merge_asof (tolerance={tolerance}, direction={direction})")

        feature_col_names = list(self.features_df.columns)
        label_col_names = list(self.labels_df.columns)

        features_reset = self.features_df.reset_index(names="_time")
        labels_reset = self.labels_df.reset_index(names="_time")

        self.features_df = None
        self.labels_df = None
        gc.collect()

        merged = pd.merge_asof(
            features_reset,
            labels_reset,
            on="_time",
            tolerance=pd.Timedelta(tolerance),
            direction=direction,
        )
        del features_reset, labels_reset
        gc.collect()

        merged = merged.set_index("_time")
        merged.index.name = None

        LOG.info(
            f"Merge result: {len(merged)} rows, "
            f"{merged[self.target_column].notna().sum()} with matched labels, "
            f"{merged[self.target_column].isna().sum()} without labels"
        )

        self._aligned_features = merged[feature_col_names].copy()
        self._aligned_labels = merged[label_col_names].copy()
        del merged
        gc.collect()

        # Drop rows with NaN in target (no matching label found)
        valid_mask = ~self._aligned_labels[self.target_column].isna()
        self._aligned_features = self._aligned_features.loc[valid_mask]
        self._aligned_labels = self._aligned_labels.loc[valid_mask]

        self._aligned_features = _downcast_float64(self._aligned_features)

        LOG.info(f"Aligned data: {len(self._aligned_features)} rows (after dropping unmatched)")

        return self

    def filter_by_vol_spread(self, min_ratio: float) -> "Dataset":
        """
        Filter aligned data to keep only rows where volatility exceeds spread by the given ratio.

        Requires volatility_ema and spread_pct_ema to be present in _aligned_labels
        (they are included automatically via align() since all label columns are kept).

        Args:
            min_ratio: Minimum ratio of volatility to spread (e.g., 1.2 means vol must be 1.2x spread)

        Returns:
            Self for method chaining
        """
        before_count = len(self._aligned_features)

        filter_mask = self._aligned_labels["volatility_ema"] > self._aligned_labels["spread_pct_ema"] * min_ratio
        self._aligned_features = self._aligned_features.loc[filter_mask]
        self._aligned_labels = self._aligned_labels.loc[filter_mask]

        after_count = len(self._aligned_features)
        removed = before_count - after_count

        LOG.info(
            f"Vol/Spread filter (ratio={min_ratio}): "
            f"{before_count:,} -> {after_count:,} rows (removed {removed:,}, {removed/before_count*100:.1f}%)"
        )

        return self

    def convert_to_binary(self, drop_timeout: bool = True) -> "Dataset":
        """
        Convert triple-barrier labels to binary classification.

        Triple-barrier labels: 0=Short, 1=Long, 2=Timeout
        Binary labels: 1=Long, 0=Short (Timeout dropped by default)

        Args:
            drop_timeout: If True, drop rows where label == 2 (Timeout).
                         If False, treat Timeout as Short (0).

        Returns:
            Self for method chaining
        """
        before_count = len(self._aligned_labels)

        if drop_timeout:
            # Drop timeout labels (2)
            mask = self._aligned_labels[self.target_column] != 2
            self._aligned_features = self._aligned_features.loc[mask]
            self._aligned_labels = self._aligned_labels.loc[mask]
            dropped = before_count - len(self._aligned_labels)
            LOG.info(f"Dropped {dropped:,} timeout labels (label=2)")

        # Convert: 1 (Long) -> 1, 0 (Short) -> 0, 2 (Timeout if kept) -> 0
        self._aligned_labels[self.target_column] = (self._aligned_labels[self.target_column] == 1).astype(int)

        after_count = len(self._aligned_labels)
        win_count = self._aligned_labels[self.target_column].sum()
        loss_count = after_count - win_count

        LOG.info(
            f"Converted to binary: {after_count:,} samples "
            f"(Win={win_count:,}, Loss={loss_count:,}, Win rate={win_count/after_count:.2%})"
        )

        return self

    def get_feature_columns(self) -> List[str]:
        """Get the list of feature columns to use for training."""
        return self.feature_columns

    def get_aligned_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get aligned features and target labels.

        Returns:
            Tuple of (features DataFrame, target Series)

        Raises:
            ValueError: If aligned data has been freed (e.g. after leak_safe_preprocess).
        """
        if self._aligned_features is None:
            raise ValueError(
                "Aligned data has been freed to save memory. "
                "Use get_train_data() / get_test_data() instead, "
                "or call align() again before get_aligned_data()."
            )

        return self._aligned_features, self._aligned_labels[self.target_column]

    def train_test_split(
        self,
        test_size: float = 0.2,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ) -> "Dataset":
        """
        Split aligned data into training and test sets.

        For time-series data, shuffle=False is recommended to maintain
        temporal ordering (train on past, test on future).

        Args:
            test_size: Fraction of data to use for testing (0.0 to 1.0)
            shuffle: Whether to shuffle before splitting. Default False for time-series.
            random_state: Random seed for shuffling

        Returns:
            Self for method chaining
        """
        X, y = self.get_aligned_data()

        if shuffle:
            # Shuffle indices
            rng = np.random.default_rng(random_state)
            indices = rng.permutation(len(X))
            X = X.iloc[indices]
            y = y.iloc[indices]

        # Calculate split point
        split_idx = int(len(X) * (1 - test_size))

        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]

        LOG.info(f"Train set: {len(self.X_train)} rows, Test set: {len(self.X_test)} rows")

        return self

    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training data."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call train_test_split() first")
        return self.X_train, self.y_train

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get test data."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Call train_test_split() first")
        return self.X_test, self.y_test

    def leak_safe_preprocess(
        self,
        test_size: float = 0.2,
        na_threshold: float = 0.5,
        fill_method: str = "ffill",
    ) -> "Dataset":
        """
        Execute split and cleaning in a leak-safe manner.
        1. Split Train/Test.
        2. Identify NA columns using Train set only.
        3. Drop those columns from both.
        4. Fill NA in Train and Test independently.
        """
        # 1. Split
        self.train_test_split(test_size=test_size, shuffle=False)

        # --- Fix #3: Free aligned data; train/test now own the only references ---
        self._aligned_features = None
        self._aligned_labels = None
        gc.collect()

        # 2. Identify columns to drop (Train stats only)
        na_fractions = self.X_train.isna().mean()
        cols_to_drop = na_fractions[na_fractions > na_threshold].index.tolist()

        if cols_to_drop:
            LOG.info(f"Leak-safe drop: Removing {len(cols_to_drop)} columns based on Train stats")
            self.X_train = self.X_train.drop(columns=cols_to_drop)
            self.X_test = self.X_test.drop(columns=cols_to_drop)
            if self.feature_columns:
                self.feature_columns = [c for c in self.feature_columns if c not in cols_to_drop]

        # 3. Fill NA independently
        def _fill(df):
            if fill_method == "ffill":
                return df.ffill()
            elif fill_method == "bfill":
                return df.bfill()
            elif fill_method == "zero":
                return df.fillna(0)
            elif fill_method == "mean":
                # Mean from same set to avoid leakage
                return df.fillna(df.mean())
            return df

        self.X_train = _fill(self.X_train)
        self.X_test = _fill(self.X_test)

        # 4. Drop remaining NaN rows (usually at the start of each set)
        train_valid = ~self.X_train.isna().any(axis=1)
        test_valid = ~self.X_test.isna().any(axis=1)

        self.X_train = self.X_train.loc[train_valid]
        self.y_train = self.y_train.loc[train_valid]
        self.X_test = self.X_test.loc[test_valid]
        self.y_test = self.y_test.loc[test_valid]

        LOG.info(f"Preprocess complete. Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        return self

    def drop_na_features(self, threshold: float = 0.5) -> "Dataset":
        """
        Drop feature columns with too many NaN values.

        Args:
            threshold: Maximum fraction of NaN values allowed (0.0 to 1.0)

        Returns:
            Self for method chaining
        """
        na_fractions = self._aligned_features.isna().mean()
        cols_to_drop = na_fractions[na_fractions > threshold].index.tolist()

        if cols_to_drop:
            LOG.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% NaN values")
            self._aligned_features = self._aligned_features.drop(columns=cols_to_drop)

            # Update feature columns
            if self.feature_columns:
                self.feature_columns = [c for c in self.feature_columns if c not in cols_to_drop]

        return self

    def fill_na(self, method: str = "ffill") -> "Dataset":
        """
        Fill NaN values in features.

        Args:
            method: Fill method - 'ffill', 'bfill', 'zero', 'mean'

        Returns:
            Self for method chaining
        """
        if method == "ffill":
            self._aligned_features = self._aligned_features.ffill()
        elif method == "bfill":
            self._aligned_features = self._aligned_features.bfill()
        elif method == "zero":
            self._aligned_features = self._aligned_features.fillna(0)
        elif method == "mean":
            self._aligned_features = self._aligned_features.fillna(self._aligned_features.mean())
        else:
            raise ValueError(f"Unknown fill method: {method}")

        # Drop any remaining NaN rows
        valid_mask = ~self._aligned_features.isna().any(axis=1)
        self._aligned_features = self._aligned_features.loc[valid_mask]
        self._aligned_labels = self._aligned_labels.loc[valid_mask]

        LOG.info(f"After filling NaN: {len(self._aligned_features)} rows")

        return self

    def summary(self) -> dict:
        """Get a summary of the dataset."""
        X, y = self.get_aligned_data()
        return {
            "total_rows": len(X),
            "num_features": len(X.columns),
            "feature_columns": list(X.columns),
            "target_column": self.target_column,
            "target_distribution": (
                y.value_counts().to_dict()
                if y.dtype in ["int64", "int32", "bool"]
                else {
                    "mean": y.mean(),
                    "std": y.std(),
                    "min": y.min(),
                    "max": y.max(),
                }
            ),
            "na_counts": X.isna().sum().to_dict(),
            "train_size": len(self.X_train) if self.X_train is not None else None,
            "test_size": len(self.X_test) if self.X_test is not None else None,
        }
