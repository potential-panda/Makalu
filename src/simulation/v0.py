import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result container for trading simulation.

    Attributes:
        simulation_df: DataFrame with simulation details including predictions, labels, and PnL
        total_samples: Total number of samples in the simulation
        trades_taken: Number of trades executed (predictions != Timeout)
        trades_skipped: Number of trades skipped (predictions == Timeout)
        total_pnl: Total PnL from the model strategy
        avg_pnl_per_trade: Average PnL per trade taken
        win_rate: Win rate for trades taken (PnL > 0)
        baseline_pnl: Total PnL from always-long baseline
        random_pnl: Total PnL from random strategy baseline
        confusion_pnl: 3x3 DataFrame of PnL sum grouped by (actual_label, predicted)
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
    confusion_pnl: pd.DataFrame

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
            "confusion_pnl": self.confusion_pnl.to_dict(),
        }

    def print_summary(self) -> None:
        """Print a formatted summary of simulation results."""
        print("=" * 60)
        print("Trading Simulation Summary")
        print("=" * 60)
        print(f"\nTotal test samples: {self.total_samples:,}")
        print(f"Trades taken (pred != Timeout): {self.trades_taken:,} ({self.trades_taken/self.total_samples:.1%})")
        print(
            f"Trades skipped (pred = Timeout): {self.trades_skipped:,} ({self.trades_skipped/self.total_samples:.1%})"
        )

        print("\nModel Strategy:")
        print(f"  Total PnL: {self.total_pnl:.6f}")
        print(f"  Avg PnL per trade: {self.avg_pnl_per_trade:.8f}")
        print(f"  Win rate: {self.win_rate:.2%}")

        print("\nAlways-Long Baseline:")
        print(f"  Total PnL: {self.baseline_pnl:.6f}")

        print("\nRandom Strategy Baseline:")
        print(f"  Total PnL: {self.random_pnl:.6f}")

        print("\nConfusion Matrix PnL (rows=actual, cols=predicted):")
        display_pnl = self.confusion_pnl.copy()
        display_pnl.index = ["Actual Short", "Actual Long", "Actual Timeout"]
        display_pnl.columns = ["Pred Short", "Pred Long", "Pred Timeout"]
        print(display_pnl.to_string(float_format=lambda x: f"{x:.6f}"))


def simulate_trading(
    predictions: np.ndarray,
    labels_df: pd.DataFrame,
    test_indices: pd.DatetimeIndex,
    probabilities: np.ndarray = None,
    threshold: float = None,
    random_seed: int = 42,
) -> SimulationResult:
    """
    Simulate trading using model predictions and compare against baselines.

    This function simulates trading based on multiclass predictions:
    - Predicted Long (1): Enter long position
    - Predicted Short (0): Enter short position
    - Predicted Timeout (2): Skip, no trade

    PnL is calculated using pre-computed long_pnl/short_pnl from triple barrier:
    - Predicted Long: Use long_pnl (actual PnL of a long position at exit bar)
    - Predicted Short: Use short_pnl (actual PnL of a short position at exit bar)
    - Predicted Timeout: 0

    Args:
        predictions: Array of predictions (0=Short, 1=Long, 2=Timeout)
        labels_df: DataFrame with columns ['label', 'long_pnl', 'short_pnl', 'volatility_ema', 'spread_pct_ema']
                   where label uses encoding (0=Short, 1=Long, 2=Timeout)
        test_indices: DatetimeIndex of test samples to match with labels
        probabilities: Optional (n_samples, 3) array of class probabilities from predict_proba.
                       When provided with threshold, low-confidence trade predictions become Timeout.
        threshold: Minimum probability to follow the predicted direction.
                   Predictions below this are overridden to Timeout (2).
        random_seed: Random seed for reproducible random baseline

    Returns:
        SimulationResult with simulation DataFrame and summary statistics
    """
    # Create lookup DataFrame for test indices
    test_lookup = pd.DataFrame({"time": test_indices}).reset_index(drop=True)
    test_lookup["time"] = pd.to_datetime(test_lookup["time"])

    # Prepare labels for merge
    labels_lookup = labels_df[["label", "long_pnl", "short_pnl", "volatility_ema", "spread_pct_ema"]].reset_index()
    labels_lookup.columns = ["time", "original_label", "long_pnl", "short_pnl", "volatility_ema", "spread_pct_ema"]

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

    # Apply probability threshold: override low-confidence trade predictions to Timeout
    if threshold is not None and probabilities is None:
        raise ValueError("threshold requires probabilities (e.g. probabilities=model.predict_proba(X))")
    if probabilities is not None and threshold is not None:
        pred_probs = probabilities[np.arange(len(predictions)), predictions]
        merged["pred_prob"] = pred_probs
        low_conf_trade = (pred_probs < threshold) & (merged["y_pred"] != 2)
        merged.loc[low_conf_trade, "y_pred"] = 2

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
                    "long_pnl": row["long_pnl"],
                    "short_pnl": row["short_pnl"],
                }
            )
        ),
        axis=1,
    )
    merged["cumulative_random"] = merged["random_pnl"].cumsum()

    # Calculate summary statistics
    trade_mask = merged["y_pred"] != 2
    trades_taken = trade_mask.sum()
    trades_skipped = (~trade_mask).sum()

    # Confusion matrix PnL: rows=actual label, cols=predicted
    label_order = [0, 1, 2]
    confusion_pnl = merged.pivot_table(
        values="sim_pnl",
        index="original_label",
        columns="y_pred",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=label_order, columns=label_order, fill_value=0.0)

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
        confusion_pnl=confusion_pnl,
    )


def _calculate_sim_pnl(row: pd.Series) -> float:
    """
    Calculate PnL for a simulated trade based on prediction.

    Prediction mapping: 0=Short, 1=Long, 2=Timeout

    Uses pre-computed long_pnl/short_pnl from triple barrier which already
    account for bid-ask spread and actual price movement at the exit bar.
    """
    pred = row["y_pred"]

    if pred == 2:  # Timeout - no trade
        return 0.0
    if pred == 1:  # Long
        pnl = row["long_pnl"]
        return pnl if not pd.isna(pnl) else 0.0
    if pred == 0:  # Short
        pnl = row["short_pnl"]
        return pnl if not pd.isna(pnl) else 0.0

    return 0.0


def _calculate_baseline_pnl(row: pd.Series) -> float:
    """Calculate PnL for always-long baseline strategy."""
    pnl = row["long_pnl"]
    return pnl if not pd.isna(pnl) else 0.0


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
    trade_pnl[merged["y_pred"] == 2] = np.nan  # Mask Timeout predictions
    rolling_winrate = (trade_pnl > 0).rolling(window, min_periods=100).mean()
    ax2.plot(merged["time"], rolling_winrate, "g-", label="Win Rate (rolling)", linewidth=1)
    ax2.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="50% baseline")

    # Trade frequency
    trade_taken = (merged["y_pred"] != 2).astype(float)
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
    trade_pnl_values = merged.loc[merged["y_pred"] != 2, "sim_pnl"]
    axes[0].hist(trade_pnl_values, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0, color="r", linestyle="--", alpha=0.5)
    axes[0].axvline(x=trade_pnl_values.mean(), color="g", linestyle="-", label=f"Mean: {trade_pnl_values.mean():.6f}")
    axes[0].set_xlabel("PnL per Trade")
    axes[0].set_ylabel("Count")
    axes[0].set_title("PnL Distribution (Trades Only)")
    axes[0].legend()

    # PnL by prediction type
    pred_pnl = merged.groupby("y_pred")["sim_pnl"].agg(["sum", "mean", "count"])
    pred_pnl.index = ["Short", "Long", "Timeout"]
    pred_pnl["sum"].plot(kind="bar", ax=axes[1], color=["red", "gray", "green"], edgecolor="black")
    axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[1].set_xlabel("Prediction Type")
    axes[1].set_ylabel("Total PnL")
    axes[1].set_title("Total PnL by Prediction Type")
    axes[1].set_xticklabels(["Short", "Long", "Timeout"], rotation=0)

    plt.tight_layout()
    plt.show()

    # Print breakdown
    print("\nPnL Breakdown by Prediction Type:")
    print(pred_pnl.to_string())

    # Figure 3: Confusion Matrix PnL heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    pnl_data = result.confusion_pnl.values

    count_data = merged.pivot_table(
        values="sim_pnl",
        index="original_label",
        columns="y_pred",
        aggfunc="count",
        fill_value=0,
    ).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).values.astype(int)

    vmax = max(abs(pnl_data.min()), abs(pnl_data.max())) or 1e-9
    im = ax.imshow(pnl_data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    actual_names = ["Short(0)", "Long(1)", "Timeout(2)"]
    pred_names = ["Short(0)", "Long(1)", "Timeout(2)"]
    for i in range(3):
        for j in range(3):
            text = f"n={count_data[i, j]}\n{pnl_data[i, j]:.4f}"
            color = "white" if abs(pnl_data[i, j]) > vmax * 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(pred_names)
    ax.set_yticklabels(actual_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix PnL (sum)")
    plt.colorbar(im, ax=ax, label="Total PnL")
    plt.tight_layout()
    plt.show()
