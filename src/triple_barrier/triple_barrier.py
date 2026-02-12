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
        label_resample_interval: Resample interval for label generation (e.g., '100ms')
        min_vol_spread_ratio: Minimum ratio of volatility to spread for structural profitability filter
            (e.g., 1.2 means volatility must be at least 1.2x the spread)
    """

    barrier_multiplier: float = 0.7
    return_shift_sec: int = 30
    timeout_sec: int = 30
    min_vol_spread_ratio: float = 1.2
    volatility_lookback_sec: int = 1800
    ema_span_sec: int = 600
    vol_resample_interval: str = "1s"
    label_resample_interval: str = "100ms"

    @property
    def param_id(self) -> str:
        """Generate a unique parameter ID string for storage."""
        barrier_bps = int(self.barrier_multiplier * 100)
        min_vol_spread_ratio_str = int(self.min_vol_spread_ratio * 10)
        return f"tb_{barrier_bps}_{self.timeout_sec}_{self.return_shift_sec}_{min_vol_spread_ratio_str}"


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
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    ask_entry_prices: np.ndarray,
    bid_entry_prices: np.ndarray,
    ask_exit_prices: np.ndarray,
    bid_exit_prices: np.ndarray,
    pt_pct: np.ndarray,
    sl_pct: np.ndarray,
    timeout_ns: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Numba-optimized inner loop for triple-barrier label computation with PnL.

    Labels are determined using mid price for barrier checking.
    PnL is calculated for both long and short positions at the exit bar:
    - long_pnl  = bid(t+n) - ask(t)   (buy at ask, sell at bid)
    - short_pnl = bid(t) - ask(t+n)   (sell at bid, buy at ask)

    For barrier hits (label 0/1), exit bar is the bar where the barrier was hit.
    For timeout (label 2), exit bar is the last bar within the timeout window.

    Args:
        timestamps: Array of timestamps as int64 nanoseconds
        mid_prices: Entry prices for barrier calculation (mid_first in each bar)
        high_prices: Maximum mid prices in each bar (for profit-take check)
        low_prices: Minimum mid prices in each bar (for stop-loss check)
        ask_entry_prices: ASK prices at entry (ask_first)
        bid_entry_prices: BID prices at entry (bid_first)
        ask_exit_prices: ASK prices at exit (ask_last)
        bid_exit_prices: BID prices at exit (bid_last)
        pt_pct: Profit-take percentage for each entry
        sl_pct: Stop-loss percentage for each entry
        timeout_ns: Timeout in nanoseconds

    Returns:
        labels: Array of labels (0=Short, 1=Long, 2=Timeout)
        long_pnl: Array of PnL for long position
        short_pnl: Array of PnL for short position
        ambiguous_count: Number of cases where both barriers were hit in the same bar
    """
    n = len(timestamps)
    labels = np.full(n, 2, dtype=np.int8)  # Default to timeout (2)
    long_pnl = np.zeros(n, dtype=np.float64)
    short_pnl = np.zeros(n, dtype=np.float64)
    ambiguous_count = 0

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

        exit_j = -1  # Track exit bar

        # Search future bars for a barrier hit
        for j in range(i + 1, n):
            if timestamps[j] > timeout_ts:
                break

            exit_j = j

            hit_upper = high_prices[j] >= upper_barrier
            hit_lower = low_prices[j] <= lower_barrier

            if hit_upper and hit_lower:
                # Both barriers hit in same bar - ambiguous case
                # Conservatively mark as Short (lower barrier)
                labels[i] = 0
                ambiguous_count += 1
                break
            elif hit_upper:
                labels[i] = 1  # Long - upper barrier hit
                break
            elif hit_lower:
                labels[i] = 0  # Short - lower barrier hit
                break

        # Calculate long/short PnL at exit bar
        if exit_j > i:
            long_pnl[i] = bid_exit_prices[exit_j] - ask_entry_prices[i]
            short_pnl[i] = bid_entry_prices[i] - ask_exit_prices[exit_j]

    return labels, long_pnl, short_pnl, ambiguous_count


def compute_labels(df: pd.DataFrame, config: TripleBarrierConfig) -> LabelingResult:
    """
    Compute triple-barrier labels and PnL using OHLC resampling.

    Labels are determined using mid price for barrier checking.
    PnL is calculated for both long and short positions at the exit bar:
    - Label 1 (Long): upper barrier hit.
    - Label 0 (Short): lower barrier hit.
    - Label 2 (Timeout): PnL calculated at the last bar within the timeout window.

    The DataFrame must have columns: 'mid', 'ask', 'bid', 'pt_pct', 'sl_pct'
    and be indexed by timestamp.

    Args:
        df: DataFrame with price and barrier data
        config: TripleBarrierConfig with labeling parameters

    Returns:
        LabelingResult with labels DataFrame (including 'label', 'long_pnl', 'short_pnl') and statistics
    """
    timeout_delta = pd.Timedelta(seconds=config.timeout_sec)

    # Aggregate to resampled bars
    # - mid: first/max/min for barrier checking
    # - ask/bid first: Entry prices
    # - ask/bid last: Exit prices
    aggregations = {
        "mid": ["first", "max", "min"],
        "ask": ["first", "last"],
        "bid": ["first", "last"],
        "pt_pct": "first",
        "sl_pct": "first",
    }

    df_resampled = df.resample(config.label_resample_interval).agg(aggregations).dropna()
    df_resampled.columns = ["_".join(col).strip() for col in df_resampled.columns.values]

    # Convert to numpy arrays for numba
    timestamps = df_resampled.index.values.astype("datetime64[ns]").astype(np.int64)
    mid_prices = df_resampled["mid_first"].values.astype(np.float64)
    high_prices = df_resampled["mid_max"].values.astype(np.float64)
    low_prices = df_resampled["mid_min"].values.astype(np.float64)
    ask_entry_prices = df_resampled["ask_first"].values.astype(np.float64)
    bid_entry_prices = df_resampled["bid_first"].values.astype(np.float64)
    ask_exit_prices = df_resampled["ask_last"].values.astype(np.float64)
    bid_exit_prices = df_resampled["bid_last"].values.astype(np.float64)
    pt_pct = df_resampled["pt_pct_first"].values.astype(np.float64)
    sl_pct = df_resampled["sl_pct_first"].values.astype(np.float64)
    timeout_ns = int(timeout_delta.total_seconds() * 1e9)

    # Run numba-optimized computation
    labels, long_pnl, short_pnl, ambiguous_count = _compute_labels_numba(
        timestamps,
        mid_prices,
        high_prices,
        low_prices,
        ask_entry_prices,
        bid_entry_prices,
        ask_exit_prices,
        bid_exit_prices,
        pt_pct,
        sl_pct,
        timeout_ns,
    )

    labels_df = pd.DataFrame(
        {"label": labels, "long_pnl": long_pnl, "short_pnl": short_pnl},
        index=df_resampled.index,
    )

    # Calculate statistics
    label_counts = pd.Series(labels).value_counts()

    return LabelingResult(
        labels_df=labels_df,
        ambiguous_count=ambiguous_count,
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
    volatility_ema = volatility.ewm(span=config.ema_span_sec, adjust=False).mean()
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


def filter_by_vol_spread_ratio(df: pd.DataFrame, config: TripleBarrierConfig) -> pd.DataFrame:
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
    df = df[df["volatility_ema"] > df["spread_pct_ema"] * config.min_vol_spread_ratio].copy()
    filtered_count = len(df)

    LOG.info(
        f"filter_by_vol_spread_ratio: {original_count} -> {filtered_count} rows "
        f"(removed {original_count - filtered_count} rows, ratio={config.min_vol_spread_ratio})"
    )

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
