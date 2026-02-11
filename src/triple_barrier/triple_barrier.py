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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
        labels_df: DataFrame with 'label' and 'pnl' columns indexed by timestamp
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

    @property
    def total_pnl(self) -> float:
        """Total PnL across all trades."""
        if "pnl" in self.labels_df.columns:
            return self.labels_df["pnl"].sum()
        return 0.0

    @property
    def avg_pnl(self) -> float:
        """Average PnL per trade."""
        if "pnl" in self.labels_df.columns and self.total_count > 0:
            return self.labels_df["pnl"].mean()
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
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Numba-optimized inner loop for triple-barrier label computation with PnL.

    Labels are determined using mid price for barrier checking.
    PnL is calculated based on the label (trading direction):
    - Label 1 (PT hit): LONG position - buy at ask(t), sell at bid(t+n)
      PnL = bid(t+n) - ask(t)
    - Label -1 (SL hit): SHORT position - sell at bid(t), buy at ask(t+n)
      PnL = bid(t) - ask(t+n)
    - Label 0 (Timeout): No trade, PnL = 0

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
        labels: Array of labels (1=Win, -1=Loss, 0=Timeout)
        pnl: Array of PnL values
        ambiguous_count: Number of cases where both barriers were hit in the same bar
    """
    n = len(timestamps)
    labels = np.zeros(n, dtype=np.int8)
    pnl = np.zeros(n, dtype=np.float64)
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

        # Search future bars for a barrier hit
        for j in range(i + 1, n):
            if timestamps[j] > timeout_ts:
                # Timeout reached - label stays 0, pnl stays 0 (no trade)
                break

            hit_upper = high_prices[j] >= upper_barrier
            hit_lower = low_prices[j] <= lower_barrier

            if hit_upper and hit_lower:
                # Both barriers hit in same bar - ambiguous case
                # Conservatively mark as Loss (short position)
                labels[i] = -1
                # SHORT: sell at bid(t), buy at ask(t+n)
                pnl[i] = bid_entry_prices[i] - ask_exit_prices[j]
                ambiguous_count += 1
                break
            elif hit_upper:
                labels[i] = 1  # Profit Take (Win) - LONG position
                # LONG: buy at ask(t), sell at bid(t+n)
                pnl[i] = bid_exit_prices[j] - ask_entry_prices[i]
                break
            elif hit_lower:
                labels[i] = -1  # Stop Loss (Loss) - SHORT position
                # SHORT: sell at bid(t), buy at ask(t+n)
                pnl[i] = bid_entry_prices[i] - ask_exit_prices[j]
                break

    return labels, pnl, ambiguous_count


def compute_labels(df: pd.DataFrame, config: TripleBarrierConfig) -> LabelingResult:
    """
    Compute triple-barrier labels and estimated PnL using OHLC resampling.

    Labels are determined using mid price for barrier checking.
    PnL is calculated based on trade direction:
    - Label 1 (PT hit): LONG - buy at ask(t), sell at bid(t+n). PnL = bid(t+n) - ask(t)
    - Label -1 (SL hit): SHORT - sell at bid(t), buy at ask(t+n). PnL = bid(t) - ask(t+n)
    - Label 0 (Timeout): No trade, PnL = 0

    The DataFrame must have columns: 'mid', 'ask', 'bid', 'pt_pct', 'sl_pct'
    and be indexed by timestamp.

    Args:
        df: DataFrame with price and barrier data
        config: TripleBarrierConfig with labeling parameters

    Returns:
        LabelingResult with labels DataFrame (including 'label' and 'pnl' columns) and statistics
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
    labels, pnl, ambiguous_count = _compute_labels_numba(
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

    labels_df = pd.DataFrame({"label": labels, "pnl": pnl}, index=df_resampled.index)

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

    # Calculate barrier levels
    df["pt_pct"] = config.barrier_multiplier * df["volatility_ema"]
    df["sl_pct"] = config.barrier_multiplier * df["volatility_ema"]  # Use same multiplier for both

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
            win_count=int(label_counts.get(1, 0)),
            loss_count=int(label_counts.get(-1, 0)),
            timeout_count=int(label_counts.get(0, 0)),
        )

    LOG.info(
        f"generate_labels: Generated {result.total_count} labels "
        f"(Win: {result.win_count}, Loss: {result.loss_count}, Timeout: {result.timeout_count}, "
        f"Ambiguous: {result.ambiguous_count}) "
        f"| Total PnL: {result.total_pnl:.4f}, Avg PnL: {result.avg_pnl:.6f}"
    )

    return result


@dataclass
class SimulationResult:
    """Result container for trading simulation.

    Attributes:
        simulation_df: DataFrame with simulation details including predictions, labels, and PnL
        total_samples: Total number of samples in the simulation
        trades_taken: Number of trades executed (predictions != NoEntry)
        trades_skipped: Number of trades skipped (predictions == NoEntry)
        total_pnl: Total PnL from the model strategy
        avg_pnl_per_trade: Average PnL per trade taken
        win_rate: Win rate for trades taken (PnL > 0)
        baseline_pnl: Total PnL from always-long baseline
        random_pnl: Total PnL from random strategy baseline
    """

    simulation_df: pd.DataFrame
    total_samples: int
    trades_taken: int
    trades_skipped: int
    total_pnl: float
    avg_pnl_per_trade: float
    win_rate: float
    baseline_pnl: float
    random_pnl: float

    def summary(self) -> Dict:
        """Return a summary dictionary of the simulation results."""
        return {
            "total_samples": self.total_samples,
            "trades_taken": self.trades_taken,
            "trades_skipped": self.trades_skipped,
            "trade_rate": self.trades_taken / self.total_samples if self.total_samples > 0 else 0,
            "total_pnl": self.total_pnl,
            "avg_pnl_per_trade": self.avg_pnl_per_trade,
            "win_rate": self.win_rate,
            "baseline_pnl": self.baseline_pnl,
            "random_pnl": self.random_pnl,
        }

    def print_summary(self) -> None:
        """Print a formatted summary of simulation results."""
        print("=" * 60)
        print("Trading Simulation Summary")
        print("=" * 60)
        print(f"\nTotal test samples: {self.total_samples:,}")
        print(f"Trades taken (pred != NoEntry): {self.trades_taken:,} ({self.trades_taken/self.total_samples:.1%})")
        print(f"Trades skipped (pred = NoEntry): {self.trades_skipped:,} ({self.trades_skipped/self.total_samples:.1%})")

        print(f"\nModel Strategy:")
        print(f"  Total PnL: {self.total_pnl:.6f}")
        print(f"  Avg PnL per trade: {self.avg_pnl_per_trade:.8f}")
        print(f"  Win rate: {self.win_rate:.2%}")

        print(f"\nAlways-Long Baseline:")
        print(f"  Total PnL: {self.baseline_pnl:.6f}")

        print(f"\nRandom Strategy Baseline:")
        print(f"  Total PnL: {self.random_pnl:.6f}")


def simulate_trading(
    predictions: np.ndarray,
    labels_df: pd.DataFrame,
    test_indices: pd.DatetimeIndex,
    random_seed: int = 42,
) -> SimulationResult:
    """
    Simulate trading using model predictions and compare against baselines.

    This function simulates trading based on multiclass predictions:
    - Predicted Long (2): Enter long position
    - Predicted Short (0): Enter short position
    - Predicted NoEntry (1): Skip, no trade

    PnL is calculated based on whether the prediction matches the actual label:
    - Correct direction: Use actual PnL from labels
    - Wrong direction: Inverse of actual PnL
    - Trade on NoEntry period: Lose approximately half the spread

    Args:
        predictions: Array of predictions (0=Short, 1=NoEntry, 2=Long)
        labels_df: DataFrame with columns ['label', 'pnl', 'volatility_ema', 'spread_pct_ema']
                   where label uses original encoding (-1=Short, 0=NoEntry, 1=Long)
        test_indices: DatetimeIndex of test samples to match with labels
        random_seed: Random seed for reproducible random baseline

    Returns:
        SimulationResult with simulation DataFrame and summary statistics
    """
    # Create lookup DataFrame for test indices
    test_lookup = pd.DataFrame({"time": test_indices}).reset_index(drop=True)
    test_lookup["time"] = pd.to_datetime(test_lookup["time"])

    # Prepare labels for merge
    labels_lookup = labels_df[["label", "pnl", "volatility_ema", "spread_pct_ema"]].reset_index()
    labels_lookup.columns = ["time", "original_label", "pnl", "volatility_ema", "spread_pct_ema"]

    # Merge to get original labels and pnl for each test sample
    merged = pd.merge_asof(
        test_lookup,
        labels_lookup,
        on="time",
        tolerance=pd.Timedelta("1s"),
        direction="nearest",
    )

    # Add predictions
    merged["y_pred"] = predictions

    # Calculate simulated PnL
    merged["sim_pnl"] = merged.apply(_calculate_sim_pnl, axis=1)
    merged["cumulative_pnl"] = merged["sim_pnl"].cumsum()

    # Calculate always-long baseline
    merged["baseline_pnl"] = merged.apply(_calculate_baseline_pnl, axis=1)
    merged["cumulative_baseline"] = merged["baseline_pnl"].cumsum()

    # Calculate random strategy baseline
    np.random.seed(random_seed)
    random_preds = np.random.choice([0, 1, 2], size=len(merged), p=[1 / 3, 1 / 3, 1 / 3])
    merged["y_pred_random"] = random_preds
    merged["random_pnl"] = merged.apply(
        lambda row: _calculate_sim_pnl(
            pd.Series(
                {
                    "y_pred": row["y_pred_random"],
                    "original_label": row["original_label"],
                    "pnl": row["pnl"],
                    "spread_pct_ema": row["spread_pct_ema"],
                }
            )
        ),
        axis=1,
    )
    merged["cumulative_random"] = merged["random_pnl"].cumsum()

    # Calculate summary statistics
    trade_mask = merged["y_pred"] != 1
    trades_taken = trade_mask.sum()
    trades_skipped = (~trade_mask).sum()

    return SimulationResult(
        simulation_df=merged,
        total_samples=len(merged),
        trades_taken=int(trades_taken),
        trades_skipped=int(trades_skipped),
        total_pnl=float(merged["sim_pnl"].sum()),
        avg_pnl_per_trade=float(merged.loc[trade_mask, "sim_pnl"].mean()) if trades_taken > 0 else 0.0,
        win_rate=float((merged.loc[trade_mask, "sim_pnl"] > 0).mean()) if trades_taken > 0 else 0.0,
        baseline_pnl=float(merged["baseline_pnl"].sum()),
        random_pnl=float(merged["random_pnl"].sum()),
    )


def _calculate_sim_pnl(row: pd.Series) -> float:
    """
    Calculate PnL for a simulated trade based on prediction.

    Prediction mapping: 0=Short, 1=NoEntry, 2=Long
    Original label: -1=Short, 0=NoEntry, 1=Long

    PnL in labels_df:
    - Label 1 (Long win): PnL = bid(exit) - ask(entry) > 0
    - Label -1 (Short win): PnL = bid(entry) - ask(exit) > 0
    - Label 0 (Timeout): PnL = 0
    """
    pred = row["y_pred"]
    original_label = row["original_label"]
    original_pnl = row["pnl"] if not pd.isna(row["pnl"]) else 0
    spread_cost = row["spread_pct_ema"] * 0.5 if not pd.isna(row["spread_pct_ema"]) else 0.0001

    # NoEntry prediction - no trade, no pnl
    if pred == 1:
        return 0.0

    # Predicted Long (pred=2)
    if pred == 2:
        if original_label == 1:  # Actual was Long win - correct prediction
            return original_pnl
        elif original_label == -1:  # Actual was Short win - wrong direction
            return -abs(original_pnl)  # We lose what the short would have gained
        else:  # original_label == 0 (Timeout/NoEntry)
            return -spread_cost  # Trade during low vol, lose spread

    # Predicted Short (pred=0)
    if pred == 0:
        if original_label == -1:  # Actual was Short win - correct prediction
            return original_pnl
        elif original_label == 1:  # Actual was Long win - wrong direction
            return -abs(original_pnl)  # We lose what the long would have gained
        else:  # original_label == 0 (Timeout/NoEntry)
            return -spread_cost  # Trade during low vol, lose spread

    return 0.0


def _calculate_baseline_pnl(row: pd.Series) -> float:
    """Calculate PnL for always-long baseline strategy."""
    original_label = row["original_label"]
    original_pnl = row["pnl"] if not pd.isna(row["pnl"]) else 0
    spread_cost = row["spread_pct_ema"] * 0.5 if not pd.isna(row["spread_pct_ema"]) else 0.0001

    if original_label == 1:
        return original_pnl
    elif original_label == -1:
        return -abs(original_pnl)
    else:
        return -spread_cost


def plot_simulation_results(result: SimulationResult, window: int = 500) -> None:
    """
    Plot trading simulation results.

    Creates two figures:
    1. Cumulative PnL over time with rolling metrics
    2. PnL distribution and breakdown by prediction type

    Args:
        result: SimulationResult from simulate_trading()
        window: Rolling window size for metrics
    """
    import matplotlib.pyplot as plt

    merged = result.simulation_df

    # Figure 1: Cumulative PnL and rolling metrics
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Cumulative PnL comparison
    ax1 = axes[0]
    ax1.plot(merged["time"], merged["cumulative_pnl"], "b-", label="Model Strategy", linewidth=1.5)
    ax1.plot(merged["time"], merged["cumulative_baseline"], "gray", label="Always Long", alpha=0.5, linewidth=1)
    ax1.plot(merged["time"], merged["cumulative_random"], "orange", label="Random", alpha=0.5, linewidth=1)
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.set_ylabel("Cumulative PnL")
    ax1.set_title("Trading Simulation: Cumulative PnL Over Time")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rolling win rate and trade frequency
    ax2 = axes[1]

    # Rolling win rate
    trade_pnl = merged["sim_pnl"].copy()
    trade_pnl[merged["y_pred"] == 1] = np.nan  # Mask NoEntry predictions
    rolling_winrate = (trade_pnl > 0).rolling(window, min_periods=100).mean()
    ax2.plot(merged["time"], rolling_winrate, "g-", label="Win Rate (rolling)", linewidth=1)
    ax2.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="50% baseline")

    # Trade frequency
    trade_taken = (merged["y_pred"] != 1).astype(float)
    rolling_trade_freq = trade_taken.rolling(window, min_periods=100).mean()
    ax2_twin = ax2.twinx()
    ax2_twin.plot(merged["time"], rolling_trade_freq, "purple", label="Trade Frequency", alpha=0.5)
    ax2_twin.set_ylabel("Trade Frequency", color="purple")

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Win Rate", color="green")
    ax2.set_title(f"Rolling Win Rate & Trade Frequency (window={window})")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Figure 2: PnL distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PnL distribution
    trade_pnl_values = merged.loc[merged["y_pred"] != 1, "sim_pnl"]
    axes[0].hist(trade_pnl_values, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0, color="r", linestyle="--", alpha=0.5)
    axes[0].axvline(x=trade_pnl_values.mean(), color="g", linestyle="-", label=f"Mean: {trade_pnl_values.mean():.6f}")
    axes[0].set_xlabel("PnL per Trade")
    axes[0].set_ylabel("Count")
    axes[0].set_title("PnL Distribution (Trades Only)")
    axes[0].legend()

    # PnL by prediction type
    pred_pnl = merged.groupby("y_pred")["sim_pnl"].agg(["sum", "mean", "count"])
    pred_pnl.index = ["Short", "NoEntry", "Long"]
    pred_pnl["sum"].plot(kind="bar", ax=axes[1], color=["red", "gray", "green"], edgecolor="black")
    axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[1].set_xlabel("Prediction Type")
    axes[1].set_ylabel("Total PnL")
    axes[1].set_title("Total PnL by Prediction Type")
    axes[1].set_xticklabels(["Short", "NoEntry", "Long"], rotation=0)

    plt.tight_layout()
    plt.show()

    # Print breakdown
    print("\nPnL Breakdown by Prediction Type:")
    print(pred_pnl.to_string())
