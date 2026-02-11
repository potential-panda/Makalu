import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from training.simulation import _calculate_baseline_pnl, _calculate_sim_pnl

LOG = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result container for trading simulation (strict version)."""

    simulation_df: pd.DataFrame
    total_samples: int
    trades_taken: int
    trades_skipped: int
    trades_blocked: int
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
            "trades_blocked": self.trades_blocked,
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
        print("Trading Simulation Summary (Strict)")
        print("=" * 60)
        print(f"\nTotal test samples: {self.total_samples:,}")
        print(f"Trades taken (pred != NoEntry): {self.trades_taken:,} ({self.trades_taken/self.total_samples:.1%})")
        print(
            f"Trades skipped (pred = NoEntry): {self.trades_skipped:,} ({self.trades_skipped/self.total_samples:.1%})"
        )
        if self.trades_blocked > 0:
            print(f"Trades blocked (overlap): {self.trades_blocked:,}")

        print("\nModel Strategy:")
        print(f"  Total PnL: {self.total_pnl:.6f}")
        print(f"  Avg PnL per trade: {self.avg_pnl_per_trade:.8f}")
        print(f"  Win rate: {self.win_rate:.2%}")

        print("\nAlways-Long Baseline:")
        print(f"  Total PnL: {self.baseline_pnl:.6f}")

        print("\nRandom Strategy Baseline:")
        print(f"  Total PnL: {self.random_pnl:.6f}")


def simulate_trading_strict(
    predictions: np.ndarray,
    labels_df: pd.DataFrame,
    test_indices: pd.DatetimeIndex,
    timeout_sec: int = 30,
    allow_overlap: bool = False,
    tolerance: str = "1s",
    direction: str = "backward",
    random_seed: int = 42,
) -> SimulationResult:
    """
    Simulate trading with leak-safe label alignment and optional non-overlapping trades.

    Differences vs training.simulation.simulate_trading:
    - Uses merge_asof direction='backward' by default (no look-ahead)
    - Optionally blocks overlapping trades with a fixed hold period (timeout_sec)
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
        tolerance=pd.Timedelta(tolerance),
        direction=direction,
    )

    # Add predictions
    merged["y_pred"] = predictions

    # Sort to ensure chronological processing
    merged = merged.sort_values("time").reset_index(drop=True)

    # Simulated PnL with optional non-overlap
    sim_pnl = np.zeros(len(merged), dtype=float)
    trade_taken = np.zeros(len(merged), dtype=bool)
    trade_blocked = np.zeros(len(merged), dtype=bool)

    next_allowed_time: Optional[pd.Timestamp] = None
    hold_delta = pd.Timedelta(seconds=timeout_sec)

    for i, row in merged.iterrows():
        pred = row["y_pred"]
        if pred == 1:
            continue

        ts = row["time"]
        if not allow_overlap and next_allowed_time is not None and ts < next_allowed_time:
            trade_blocked[i] = True
            continue

        trade_taken[i] = True
        sim_pnl[i] = _calculate_sim_pnl(row)

        if not allow_overlap:
            next_allowed_time = ts + hold_delta

    merged["sim_pnl"] = sim_pnl
    merged["trade_taken"] = trade_taken
    merged["trade_blocked"] = trade_blocked
    merged["cumulative_pnl"] = merged["sim_pnl"].cumsum()

    # Always-long baseline (same overlap rules)
    baseline_pnl = np.zeros(len(merged), dtype=float)
    next_allowed_time = None
    for i, row in merged.iterrows():
        ts = row["time"]
        if not allow_overlap and next_allowed_time is not None and ts < next_allowed_time:
            continue
        baseline_pnl[i] = _calculate_baseline_pnl(row)
        if not allow_overlap:
            next_allowed_time = ts + hold_delta

    merged["baseline_pnl"] = baseline_pnl
    merged["cumulative_baseline"] = merged["baseline_pnl"].cumsum()

    # Random strategy baseline (same overlap rules)
    np.random.seed(random_seed)
    random_preds = np.random.choice([0, 1, 2], size=len(merged), p=[1 / 3, 1 / 3, 1 / 3])
    merged["y_pred_random"] = random_preds

    random_pnl = np.zeros(len(merged), dtype=float)
    next_allowed_time = None
    for i, row in merged.iterrows():
        ts = row["time"]
        pred = row["y_pred_random"]
        if pred == 1:
            continue
        if not allow_overlap and next_allowed_time is not None and ts < next_allowed_time:
            continue
        random_pnl[i] = _calculate_sim_pnl(
            pd.Series(
                {
                    "y_pred": pred,
                    "original_label": row["original_label"],
                    "pnl": row["pnl"],
                    "spread_pct_ema": row["spread_pct_ema"],
                }
            )
        )
        if not allow_overlap:
            next_allowed_time = ts + hold_delta

    merged["random_pnl"] = random_pnl
    merged["cumulative_random"] = merged["random_pnl"].cumsum()

    # Calculate summary statistics
    trades_taken = int(trade_taken.sum())
    trades_blocked = int(trade_blocked.sum())
    trades_skipped = int((merged["y_pred"] == 1).sum())

    return SimulationResult(
        simulation_df=merged,
        total_samples=len(merged),
        trades_taken=trades_taken,
        trades_skipped=trades_skipped,
        trades_blocked=trades_blocked,
        total_pnl=float(merged["sim_pnl"].sum()),
        avg_pnl_per_trade=float(merged.loc[trade_taken, "sim_pnl"].mean()) if trades_taken > 0 else 0.0,
        win_rate=float((merged.loc[trade_taken, "sim_pnl"] > 0).mean()) if trades_taken > 0 else 0.0,
        baseline_pnl=float(merged["baseline_pnl"].sum()),
        random_pnl=float(merged["random_pnl"].sum()),
    )


def plot_simulation_results(result: SimulationResult, window: int = 500) -> None:
    """Plot trading simulation results (strict version)."""
    import matplotlib.pyplot as plt

    merged = result.simulation_df

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(merged["time"], merged["cumulative_pnl"], "b-", label="Model Strategy", linewidth=1.5)
    ax1.plot(merged["time"], merged["cumulative_baseline"], "gray", label="Always Long", alpha=0.5, linewidth=1)
    ax1.plot(merged["time"], merged["cumulative_random"], "orange", label="Random", alpha=0.5, linewidth=1)
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.set_ylabel("Cumulative PnL")
    ax1.set_title("Trading Simulation (Strict): Cumulative PnL Over Time")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    trade_pnl = merged["sim_pnl"].copy()
    trade_pnl[~merged["trade_taken"]] = np.nan
    rolling_winrate = (trade_pnl > 0).rolling(window, min_periods=100).mean()
    ax2.plot(merged["time"], rolling_winrate, "g-", label="Win Rate (rolling)", linewidth=1)
    ax2.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="50% baseline")

    trade_taken = merged["trade_taken"].astype(float)
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

    trade_pnl_values = merged.loc[merged["trade_taken"], "sim_pnl"]
    axes[0].hist(trade_pnl_values, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0, color="r", linestyle="--", alpha=0.5)
    if len(trade_pnl_values) > 0:
        axes[0].axvline(
            x=trade_pnl_values.mean(),
            color="g",
            linestyle="-",
            label=f"Mean: {trade_pnl_values.mean():.6f}",
        )
    axes[0].set_xlabel("PnL per Trade")
    axes[0].set_ylabel("Count")
    axes[0].set_title("PnL Distribution (Trades Only)")
    axes[0].legend()

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

    print("\nPnL Breakdown by Prediction Type:")
    print(pred_pnl.to_string())
