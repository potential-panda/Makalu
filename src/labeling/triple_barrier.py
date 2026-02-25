"""
Triple-Barrier Labeling Library.

This module provides core functions for computing triple-barrier labels
using dynamic volatility-based barriers. The implementation is optimized
with numba JIT compilation for performance.

The triple-barrier method labels each potential trade entry as:
- Short (0): Price hit stop-loss (lower) barrier before timeout
- Long (1): Price hit profit-take (upper) barrier before timeout
- Timeout (2): Neither barrier was hit within the timeout period

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
    """

    barrier_multiplier: float = 1.0
    return_shift_sec: int = 30
    timeout_sec: int = 30
    volatility_lookback_sec: int = 900
    ema_span_sec: int = 60
    vol_resample_interval: str = "1s"

    @property
    def param_id(self) -> str:
        """Generate a unique parameter ID string for storage."""
        params = [
            int(self.barrier_multiplier * 100),
            self.timeout_sec,
            self.return_shift_sec,
            self.volatility_lookback_sec,
        ]
        return f"tb_{'_'.join(map(str, params))}"


@dataclass
class LabelingResult:
    """Result container for triple-barrier labeling.

    Attributes:
        labels_df: DataFrame with 'label', 'long_pnl', and 'short_pnl' columns indexed by timestamp
        ambiguous_count: Number of cases where both barriers were hit in the same bar
        total_count: Total number of labels generated
        long_count: Number of Long labels (1)
        short_count: Number of Short labels (0)
        timeout_count: Number of Timeout labels (2)
    """

    labels_df: pd.DataFrame
    ambiguous_count: int
    total_count: int
    long_count: int
    short_count: int
    timeout_count: int

    @property
    def long_rate(self) -> float:
        """Long rate excluding timeouts."""
        non_timeout = self.long_count + self.short_count
        return self.long_count / non_timeout if non_timeout > 0 else 0.0

    @property
    def timeout_rate(self) -> float:
        """Timeout rate."""
        return self.timeout_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def total_long_pnl(self) -> float:
        """Total long PnL across all trades."""
        if "long_pnl" in self.labels_df.columns:
            return self.labels_df["long_pnl"].sum()
        return 0.0

    @property
    def avg_long_pnl(self) -> float:
        """Average long PnL per trade."""
        if "long_pnl" in self.labels_df.columns and self.total_count > 0:
            return self.labels_df["long_pnl"].mean()
        return 0.0


@jit(nopython=True)
def _compute_labels_numba(
    timestamps: np.ndarray,
    mid_prices: np.ndarray,
    ask_prices: np.ndarray,
    bid_prices: np.ndarray,
    pt_pct: np.ndarray,
    sl_pct: np.ndarray,
    timeout_ns: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized inner loop for tick-level triple-barrier label computation with PnL.

    Labels are determined using mid price for barrier checking.
    PnL is calculated for both long and short positions at the exit tick:
    - long_pnl  = bid(t+n) - ask(t)   (buy at ask, sell at bid)
    - short_pnl = bid(t) - ask(t+n)   (sell at bid, buy at ask)

    For barrier hits (label 0/1), exit tick is the tick where the barrier was hit.
    For timeout (label 2), exit tick is the last tick within the timeout window.

    Args:
        timestamps: Array of timestamps as int64 nanoseconds
        mid_prices: Mid prices at each tick
        ask_prices: ASK prices at each tick
        bid_prices: BID prices at each tick
        pt_pct: Profit-take percentage for each entry
        sl_pct: Stop-loss percentage for each entry
        timeout_ns: Timeout in nanoseconds

    Returns:
        labels: Array of labels (0=Short, 1=Long, 2=Timeout)
        long_pnl: Array of PnL for long position
        short_pnl: Array of PnL for short position
    """
    n = len(timestamps)
    labels = np.full(n, 2, dtype=np.int8)  # Default to timeout (2)
    long_pnl = np.zeros(n, dtype=np.float64)
    short_pnl = np.zeros(n, dtype=np.float64)

    for i in range(n):
        entry_mid = mid_prices[i]
        entry_ts = timestamps[i]
        timeout_ts = entry_ts + timeout_ns

        pt_level = pt_pct[i]
        sl_level = sl_pct[i]

        if np.isnan(entry_mid) or np.isnan(pt_level) or np.isnan(sl_level):
            continue

        upper_barrier = entry_mid * (1 + pt_level)
        lower_barrier = entry_mid * (1 - sl_level)

        exit_j = -1  # Track exit tick

        # Search future ticks for a barrier hit
        for j in range(i + 1, n):
            if timestamps[j] > timeout_ts:
                break

            exit_j = j

            if mid_prices[j] >= upper_barrier:
                labels[i] = 1  # Long - upper barrier hit
                break
            elif mid_prices[j] <= lower_barrier:
                labels[i] = 0  # Short - lower barrier hit
                break

        # Calculate long/short PnL at exit tick
        if exit_j > i:
            long_pnl[i] = bid_prices[exit_j] - ask_prices[i]
            short_pnl[i] = bid_prices[i] - ask_prices[exit_j]

    return labels, long_pnl, short_pnl


def compute_labels(df: pd.DataFrame, config: TripleBarrierConfig) -> LabelingResult:
    """
    Compute triple-barrier labels and PnL at tick level.

    Labels are determined using mid price for barrier checking.
    PnL is calculated for both long and short positions at the exit tick:
    - Label 1 (Long): upper barrier hit.
    - Label 0 (Short): lower barrier hit.
    - Label 2 (Timeout): PnL calculated at the last tick within the timeout window.

    The DataFrame must have columns: 'mid', 'ask', 'bid', 'pt_pct', 'sl_pct'
    and be indexed by timestamp.

    Args:
        df: DataFrame with price and barrier data
        config: TripleBarrierConfig with labeling parameters

    Returns:
        LabelingResult with labels DataFrame (including 'label', 'long_pnl', 'short_pnl') and statistics
    """
    timeout_ns = int(pd.Timedelta(seconds=config.timeout_sec).total_seconds() * 1e9)

    # Convert to numpy arrays for numba
    timestamps = df.index.values.astype("datetime64[ns]").astype(np.int64)
    mid_prices = df["mid"].values.astype(np.float64)
    ask_prices = df["ask"].values.astype(np.float64)
    bid_prices = df["bid"].values.astype(np.float64)
    pt_pct = df["pt_pct"].values.astype(np.float64)
    sl_pct = df["sl_pct"].values.astype(np.float64)

    # Run numba-optimized computation
    labels, long_pnl, short_pnl = _compute_labels_numba(
        timestamps,
        mid_prices,
        ask_prices,
        bid_prices,
        pt_pct,
        sl_pct,
        timeout_ns,
    )

    labels_df = pd.DataFrame(
        {"label": labels, "long_pnl": long_pnl, "short_pnl": short_pnl},
        index=df.index,
    )

    # Calculate statistics
    label_counts = pd.Series(labels).value_counts()

    return LabelingResult(
        labels_df=labels_df,
        ambiguous_count=0,
        total_count=len(labels),
        long_count=int(label_counts.get(1, 0)),
        short_count=int(label_counts.get(0, 0)),
        timeout_count=int(label_counts.get(2, 0)),
    )


def ts_index(df: pd.DataFrame) -> pd.DataFrame:
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df.dropna(subset=["ts"], inplace=True)
    df.index = pd.to_datetime(df["ts"], unit="s")
    df = df.drop(columns=["ts"])
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
    volatility_ema = volatility
    # volatility_ema = volatility.ewm(span=config.ema_span_sec, adjust=False).mean()
    spread_pct_ema = spread_pct.ewm(span=config.ema_span_sec, adjust=False).mean()

    # Reindex back to original tick-level index
    df["volatility_ema"] = volatility_ema.reindex(df.index, method="ffill")
    df["spread_pct_ema"] = spread_pct_ema.reindex(df.index, method="ffill")

    # Calculate barrier levels based on spread (transaction cost)
    # This ensures barriers have direct economic meaning: profit target = k * spread
    # Volatility then determines the probability of hitting these barriers
    df["pt_pct"] = config.barrier_multiplier * df["spread_pct_ema"]
    df["sl_pct"] = config.barrier_multiplier * df["spread_pct_ema"]

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
            long_count=int(label_counts.get(1, 0)),
            short_count=int(label_counts.get(0, 0)),
            timeout_count=int(label_counts.get(2, 0)),
        )

    LOG.info(
        f"generate_labels: Generated {result.total_count} labels "
        f"(Long: {result.long_count}, Short: {result.short_count}, Timeout: {result.timeout_count}, "
        f"Ambiguous: {result.ambiguous_count}) "
        f"| Total Long PnL: {result.total_long_pnl:.4f}, Avg Long PnL: {result.avg_long_pnl:.6f}"
    )

    return result


def filter_by_vol_spread_ratio(df: pd.DataFrame, min_vol_spread_ratio: float) -> pd.DataFrame:
    """
    Filter DataFrame to keep only rows where volatility exceeds spread by the configured ratio.

    This ensures we only generate labels for periods with favorable structural conditions,
    where the potential profit margin exceeds transaction cost.

    Args:
        df: DataFrame with 'volatility_ema' and 'spread_pct_ema' columns
        config: TripleBarrierConfig with min_vol_spread_ratio parameter

    Returns:
        Filtered DataFrame
    """
    original_count = len(df)
    df = df[df["volatility_ema"] > df["spread_pct_ema"] * min_vol_spread_ratio].copy()
    filtered_count = len(df)

    LOG.info(
        f"filter_by_vol_spread_ratio: {original_count} -> {filtered_count} rows "
        f"(removed {original_count - filtered_count} rows, ratio={min_vol_spread_ratio})"
    )

    return df
