"""
Dataset module for loading and aligning features and labels.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

LOG = logging.getLogger(__name__)


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
    _aligned_labels: Optional[pd.Series] = None

    def __post_init__(self):
        """Initialize feature columns if not provided."""
        if not self.feature_columns:
            self.feature_columns = list(self.features_df.columns)

    @classmethod
    def load(
        cls,
        feature_market,
        label_market,
        parquet_store: ParquetStore,
        target_column: str,
        dt_range: DtRange,
        label_param_id: Optional[str] = None,
    ) -> "Dataset":
        feature_store = parquet_store.create_feature(feature_market)
        if label_param_id:
            label_store = parquet_store.create_label("TripleBarrierLabel", label_market, label_param_id)
        else:
            label_store = parquet_store.create_label("TripleBarrierLabel", label_market)
        features_df = feature_store.read(dt_range)
        labels_df = label_store.read(dt_range)

        LOG.info(f"Loaded {len(features_df)} feature rows with {len(features_df.columns)} columns")
        LOG.info(f"Loaded {len(labels_df)} label rows with {len(labels_df.columns)} columns")

        dataset = cls(
            features_df=features_df,
            labels_df=labels_df,
            target_column=target_column,
        )

        # Auto-align on load
        dataset.align()

        return dataset

    def align(
        self,
        tolerance: str = "1s",
        direction: str = "backward",
        include_label_columns: Optional[List[str]] = None,
    ) -> "Dataset":
        """
        Align labels to features by finding the nearest label for each feature timestamp.

        Uses merge_asof to match each feature row with its closest label row,
        keeping all feature rows. This handles cases where features and labels
        are generated at different intervals or slightly different timestamps.

        Args:
            tolerance: Maximum time difference for matching (e.g., '1s', '100ms', '5s').
                      Labels further than this from a feature timestamp won't match.
            direction: Direction for matching - 'nearest', 'backward', 'forward'.
                      'nearest' finds closest label, 'backward' avoids look-ahead bias.
            include_label_columns: Additional columns from labels_df to include in aligned data.
                      Useful for post-alignment filtering (e.g., volatility_ema, spread_pct_ema).

        Returns:
            Self for method chaining
        """
        LOG.info(f"Aligning labels to features using merge_asof (tolerance={tolerance}, direction={direction})")

        # Determine which columns to include from labels_df
        label_columns = [self.target_column]
        if include_label_columns:
            label_columns.extend([c for c in include_label_columns if c in self.labels_df.columns])

        # Prepare DataFrames for merge_asof (needs 'on' column, not index)
        # Use a consistent column name '_time' for the merge key
        features_reset = self.features_df.reset_index(names="_time")
        labels_reset = self.labels_df[label_columns].reset_index(names="_time")

        # Use merge_asof to find nearest label for each feature timestamp
        merged = pd.merge_asof(
            features_reset,
            labels_reset,
            on="_time",
            tolerance=pd.Timedelta(tolerance),
            direction=direction,
        )

        # Set time back as index
        merged = merged.set_index("_time")
        merged.index.name = None  # Remove the temporary name

        LOG.info(
            f"Merge result: {len(merged)} rows, "
            f"{merged[self.target_column].notna().sum()} with matched labels, "
            f"{merged[self.target_column].isna().sum()} without labels"
        )

        # Extract aligned features and labels
        # Keep additional label columns in features for post-alignment filtering
        columns_to_drop = [self.target_column]
        self._aligned_features = merged.drop(columns=columns_to_drop).copy()
        self._aligned_labels = merged[self.target_column].copy()

        # Drop rows with NaN in target (no matching label found)
        valid_mask = ~self._aligned_labels.isna()
        self._aligned_features = self._aligned_features.loc[valid_mask]
        self._aligned_labels = self._aligned_labels.loc[valid_mask]

        LOG.info(f"Aligned data: {len(self._aligned_features)} rows (after dropping unmatched)")

        return self

    def filter_by_vol_spread(self, min_ratio: float = 1.2) -> "Dataset":
        """
        Filter aligned data to keep only rows where volatility exceeds spread by the given ratio.

        This ensures we only train on periods with favorable structural conditions for trading.
        Must be called after align() with include_label_columns=['volatility_ema', 'spread_pct_ema'].

        Args:
            min_ratio: Minimum ratio of volatility to spread (e.g., 1.2 means vol must be 1.2x spread)

        Returns:
            Self for method chaining
        """
        before_count = len(self._aligned_features)

        # Apply filter
        filter_mask = self._aligned_features["volatility_ema"] > self._aligned_features["spread_pct_ema"] * min_ratio
        self._aligned_features = self._aligned_features.loc[filter_mask]
        self._aligned_labels = self._aligned_labels.loc[filter_mask]

        after_count = len(self._aligned_features)
        removed = before_count - after_count

        LOG.info(
            f"Vol/Spread filter (ratio={min_ratio}): "
            f"{before_count:,} -> {after_count:,} rows (removed {removed:,}, {removed/before_count*100:.1f}%)"
        )

        # Remove the filter columns from features (not needed for training)
        self._aligned_features = self._aligned_features.drop(
            columns=["volatility_ema", "spread_pct_ema"], errors="ignore"
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
            mask = self._aligned_labels != 2
            self._aligned_features = self._aligned_features.loc[mask]
            self._aligned_labels = self._aligned_labels.loc[mask]
            dropped = before_count - len(self._aligned_labels)
            LOG.info(f"Dropped {dropped:,} timeout labels (label=2)")

        # Convert: 1 (Long) -> 1, 0 (Short) -> 0, 2 (Timeout if kept) -> 0
        self._aligned_labels = (self._aligned_labels == 1).astype(int)

        after_count = len(self._aligned_labels)
        win_count = self._aligned_labels.sum()
        loss_count = after_count - win_count

        LOG.info(
            f"Converted to binary: {after_count:,} samples "
            f"(Win={win_count:,}, Loss={loss_count:,}, Win rate={win_count/after_count:.2%})"
        )

        return self

    def convert_to_multiclass(self) -> "Dataset":
        """
        Convert triple-barrier labels to multiclass format for LightGBM.

        Triple-barrier labels: 0=Short, 1=Long, 2=Timeout
        Multiclass labels (for training and simulation):
            0 = Short (was 0)
            1 = NoEntry (was 2, Timeout)
            2 = Long (was 1)

        Returns:
            Self for method chaining
        """
        # Map: 0(Short)->0, 1(Long)->2, 2(Timeout)->1(NoEntry)
        label_map = {0: 0, 1: 2, 2: 1}
        self._aligned_labels = self._aligned_labels.map(label_map).astype(int)

        counts = self._aligned_labels.value_counts().sort_index()
        LOG.info(
            f"Converted to multiclass: Short(0)={counts.get(0, 0):,}, "
            f"NoEntry(1)={counts.get(1, 0):,}, Long(2)={counts.get(2, 0):,}"
        )

        return self

    def get_feature_columns(self) -> List[str]:
        """Get the list of feature columns to use for training."""
        if self.feature_columns:
            return self.feature_columns
        return list(self.features_df.columns)

    def get_aligned_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get aligned features and labels.

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        feature_cols = self.get_feature_columns()
        available_cols = [c for c in feature_cols if c in self._aligned_features.columns]

        if len(available_cols) < len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            LOG.warning(f"Some feature columns not found: {missing}")

        return self._aligned_features[available_cols], self._aligned_labels

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

        self.X_train = X.iloc[:split_idx].copy()
        self.X_test = X.iloc[split_idx:].copy()
        self.y_train = y.iloc[:split_idx].copy()
        self.y_test = y.iloc[split_idx:].copy()

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
        if self._aligned_features is None:
            self.align()

        # 1. Split
        self.train_test_split(test_size=test_size, shuffle=False)

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
        if self._aligned_features is None:
            self.align()

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
        if self._aligned_features is None:
            self.align()

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
