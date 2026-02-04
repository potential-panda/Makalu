"""
Triple-Barrier Labeling Library.

This module provides core functions for computing triple-barrier labels
using dynamic volatility-based barriers. The implementation is optimized
with numba JIT compilation for performance.

The triple-barrier method labels each potential trade entry as:
- Win (1): Price hit profit-take barrier before timeout
- Loss (-1): Price hit stop-loss barrier before timeout
- Timeout (0): Neither barrier was hit within the timeout period

Reference: "Advances in Financial Machine Learning" by Marcos López de Prado
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from numba import jit

LOG = logging.getLogger(__name__)
TIMEZONE = "Asia/Tokyo"  # Define the local timezone for analysis


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling.

    Attributes:
        barrier_multiplier: Multiplier for volatility to set barrier levels (e.g., 2.0 = 2x volatility)
        timeout_sec: Maximum time in seconds to wait for barrier hit
        return_shift_sec: Time shift for return calculation (used in volatility calculation)
        volatility_lookback_sec: Lookback window for volatility calculation in seconds
        ema_span_sec: EMA span for smoothing volatility and spread
        vol_resample_interval: Resample interval for volatility calculation (e.g., '1s')
        label_resample_interval: Resample interval for label generation (e.g., '100ms')
    """

    barrier_multiplier: float = 2.0
    return_shift_sec: int = 180
    timeout_sec: int = 180
    volatility_lookback_sec: int = 3600
    ema_span_sec: int = 600
    vol_resample_interval: str = "1s"
    label_resample_interval: str = "100ms"

    @property
    def param_id(self) -> str:
        """Generate a unique parameter ID string for storage."""
        barrier_bps = int(self.barrier_multiplier * 100)
        return f"tb_{barrier_bps}bps_{self.timeout_sec}s_{self.return_shift_sec}s"


@dataclass
class LabelingResult:
    """Result container for triple-barrier labeling.

    Attributes:
        labels_df: DataFrame with 'label' column indexed by timestamp
        ambiguous_count: Number of cases where both barriers were hit in the same bar
        total_count: Total number of labels generated
        win_count: Number of Win labels (1)
        loss_count: Number of Loss labels (-1)
        timeout_count: Number of Timeout labels (0)
    """

    labels_df: pd.DataFrame
    ambiguous_count: int
    total_count: int
    win_count: int
    loss_count: int
    timeout_count: int

    @property
    def win_rate(self) -> float:
        """Win rate excluding timeouts."""
        non_timeout = self.win_count + self.loss_count
        return self.win_count / non_timeout if non_timeout > 0 else 0.0

    @property
    def timeout_rate(self) -> float:
        """Timeout rate."""
        return self.timeout_count / self.total_count if self.total_count > 0 else 0.0


@jit(nopython=True)
def _compute_labels_numba(
    timestamps: np.ndarray,
    ask_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    pt_pct: np.ndarray,
    sl_pct: np.ndarray,
    timeout_ns: int,
) -> Tuple[np.ndarray, int]:
    """
    Numba-optimized inner loop for triple-barrier label computation.

    For a LONG position:
    - Entry: Buy at ASK price
    - Exit: Sell at BID price (check against high_prices/low_prices which are bid max/min)

    Args:
        timestamps: Array of timestamps as int64 nanoseconds
        ask_prices: Entry prices (ask_first in each bar)
        high_prices: Maximum bid prices in each bar (for profit-take check)
        low_prices: Minimum bid prices in each bar (for stop-loss check)
        pt_pct: Profit-take percentage for each entry
        sl_pct: Stop-loss percentage for each entry
        timeout_ns: Timeout in nanoseconds

    Returns:
        labels: Array of labels (1=Win, -1=Loss, 0=Timeout)
        ambiguous_count: Number of cases where both barriers were hit in the same bar
    """
    n = len(timestamps)
    labels = np.zeros(n, dtype=np.int8)
    ambiguous_count = 0

    for i in range(n):
        entry_price = ask_prices[i]
        entry_ts = timestamps[i]
        timeout_ts = entry_ts + timeout_ns

        pt_level = pt_pct[i]
        sl_level = sl_pct[i]

        if np.isnan(entry_price) or np.isnan(pt_level) or np.isnan(sl_level):
            continue

        upper_barrier = entry_price * (1 + pt_level)
        lower_barrier = entry_price * (1 - sl_level)

        # Search future bars for a barrier hit
        for j in range(i + 1, n):
            if timestamps[j] > timeout_ts:
                break  # Timeout reached, label stays 0

            hit_upper = high_prices[j] >= upper_barrier
            hit_lower = low_prices[j] <= lower_barrier

            if hit_upper and hit_lower:
                # Both barriers hit in same bar - ambiguous case
                # Conservatively mark as Loss (avoid upward bias)
                labels[i] = -1
                ambiguous_count += 1
                break
            elif hit_upper:
                labels[i] = 1  # Profit Take (Win)
                break
            elif hit_lower:
                labels[i] = -1  # Stop Loss (Loss)
                break

    return labels, ambiguous_count


def compute_labels(df: pd.DataFrame, config: TripleBarrierConfig) -> LabelingResult:
    """
    Compute triple-barrier labels using OHLC resampling for accuracy.

    The DataFrame must have columns: 'ask', 'bid', 'pt_pct', 'sl_pct'
    and be indexed by timestamp.

    Args:
        df: DataFrame with price and barrier data
        config: TripleBarrierConfig with labeling parameters

    Returns:
        LabelingResult with labels DataFrame and statistics
    """
    timeout_delta = pd.Timedelta(seconds=config.timeout_sec)

    # Aggregate to resampled bars
    # - ask_first: Entry price for long position
    # - bid_max/bid_min: Exit price range for checking barrier hits
    aggregations = {
        "ask": "first",
        "bid": ["max", "min"],
        "pt_pct": "first",
        "sl_pct": "first",
    }

    df_resampled = df.resample(config.label_resample_interval).agg(aggregations).dropna()
    df_resampled.columns = ["_".join(col).strip() for col in df_resampled.columns.values]

    n = len(df_resampled)
    if n == 0:
        return LabelingResult(
            labels_df=pd.DataFrame({"label": []}, index=pd.DatetimeIndex([])),
            ambiguous_count=0,
            total_count=0,
            win_count=0,
            loss_count=0,
            timeout_count=0,
        )

    # Convert to numpy arrays for numba
    timestamps = df_resampled.index.values.astype("datetime64[ns]").astype(np.int64)
    ask_prices = df_resampled["ask_first"].values.astype(np.float64)
    high_prices = df_resampled["bid_max"].values.astype(np.float64)
    low_prices = df_resampled["bid_min"].values.astype(np.float64)
    pt_pct = df_resampled["pt_pct_first"].values.astype(np.float64)
    sl_pct = df_resampled["sl_pct_first"].values.astype(np.float64)
    timeout_ns = int(timeout_delta.total_seconds() * 1e9)

    # Run numba-optimized computation
    labels, ambiguous_count = _compute_labels_numba(
        timestamps, ask_prices, high_prices, low_prices, pt_pct, sl_pct, timeout_ns
    )

    labels_df = pd.DataFrame({"label": labels}, index=df_resampled.index)

    # Calculate statistics
    label_counts = pd.Series(labels).value_counts()

    return LabelingResult(
        labels_df=labels_df,
        ambiguous_count=ambiguous_count,
        total_count=len(labels),
        win_count=int(label_counts.get(1, 0)),
        loss_count=int(label_counts.get(-1, 0)),
        timeout_count=int(label_counts.get(0, 0)),
    )


def ts_index(df: pd.DataFrame) -> pd.DataFrame:
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df.dropna(subset=["ts"], inplace=True)
    df.index = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(TIMEZONE)
    df.drop(columns=["ts"])
    LOG.info(f"generate_labels: Processing {len(df)} data points")
    return df


def calculate_volatility_and_spread(df: pd.DataFrame, config: TripleBarrierConfig) -> pd.DataFrame:
    df = df.sort_index()

    # Calculate time-based parameters
    dt = pd.Timedelta(config.vol_resample_interval).total_seconds()
    shift_periods = int(round(config.return_shift_sec / dt))
    lookback_periods = int(round(config.volatility_lookback_sec / dt))

    # Resample to fixed interval
    mid_resampled = df["mid"].resample(config.vol_resample_interval).last().ffill()
    spread_resampled = df["spread"].resample(config.vol_resample_interval).last().ffill()

    # Calculate returns and volatility
    returns = mid_resampled.pct_change(periods=shift_periods)
    volatility = returns.rolling(window=lookback_periods, min_periods=1).std()
    spread_pct = spread_resampled / mid_resampled

    # Apply EMA smoothing
    volatility_ema = volatility.ewm(span=config.ema_span_sec, adjust=False).mean()
    spread_pct_ema = spread_pct.ewm(span=config.ema_span_sec, adjust=False).mean()

    # Reindex back to original tick-level index
    df["volatility_ema"] = volatility_ema.reindex(df.index, method="ffill")
    df["spread_pct_ema"] = spread_pct_ema.reindex(df.index, method="ffill")

    # Calculate barrier levels
    df["pt_pct"] = config.barrier_multiplier * df["volatility_ema"]
    df["sl_pct"] = config.barrier_multiplier * df["volatility_ema"]

    # Drop rows with NaN values
    df.dropna(inplace=True)

    return df


def generate_labels(df: pd.DataFrame, config: TripleBarrierConfig, exclude_edge: bool = True) -> LabelingResult:
    df = ts_index(df)

    # Calculate volatility and barrier levels
    df = calculate_volatility_and_spread(df, config)
    LOG.info(f"generate_labels: {len(df)} points after volatility calculation")

    # Compute labels
    result = compute_labels(df, config)

    # Exclude edge effect (last timeout_sec of data)
    if exclude_edge and not result.labels_df.empty:
        cutoff_time = result.labels_df.index.max() - pd.Timedelta(seconds=config.timeout_sec)
        result.labels_df = result.labels_df.loc[result.labels_df.index <= cutoff_time]

        # Recalculate statistics
        labels = result.labels_df["label"].values
        label_counts = pd.Series(labels).value_counts()
        result = LabelingResult(
            labels_df=result.labels_df,
            ambiguous_count=result.ambiguous_count,
            total_count=len(labels),
            win_count=int(label_counts.get(1, 0)),
            loss_count=int(label_counts.get(-1, 0)),
            timeout_count=int(label_counts.get(0, 0)),
        )

    LOG.info(
        f"generate_labels: Generated {result.total_count} labels "
        f"(Win: {result.win_count}, Loss: {result.loss_count}, Timeout: {result.timeout_count}, "
        f"Ambiguous: {result.ambiguous_count})"
    )

    return result
