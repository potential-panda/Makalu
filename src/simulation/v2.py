"""
Trading simulation v2: tick-by-tick forward simulation with explicit barriers.

Unlike v0/v1 which use pre-computed long_pnl/short_pnl from triple-barrier labels,
this version walks forward through execution price data to determine trade exits.
This allows varying pt/sl barriers and vol/spread filters without re-running the
labeling pipeline.

Prediction encoding: 0=Short, 1=Long, 2=Timeout (skip)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result container for v2 tick-by-tick simulation."""

    simulation_df: pd.DataFrame
    total_samples: int
    trades_taken: int
    trades_skipped: int
    trades_blocked: int
    trades_filtered: int
    total_pnl: float
    avg_pnl_per_trade: float
    win_rate: float
    baseline_pnl: float
    random_pnl: float
    exit_reason_counts: Dict[str, int]

    def summary(self) -> Dict:
        """Return a summary dictionary of the simulation results."""
        return {
            "total_samples": self.total_samples,
            "trades_taken": self.trades_taken,
            "trades_skipped": self.trades_skipped,
            "trades_blocked": self.trades_blocked,
            "trades_filtered": self.trades_filtered,
            "trade_rate": self.trades_taken / self.total_samples if self.total_samples > 0 else 0,
            "total_pnl": self.total_pnl,
            "avg_pnl_per_trade": self.avg_pnl_per_trade,
            "win_rate": self.win_rate,
            "baseline_pnl": self.baseline_pnl,
            "random_pnl": self.random_pnl,
            "exit_reason_counts": self.exit_reason_counts,
        }

    def print_summary(self) -> None:
        """Print a formatted summary of simulation results."""
        print("=" * 60)
        print("Trading Simulation Summary (V2 - Tick-by-Tick)")
        print("=" * 60)
        print(f"\nTotal test samples: {self.total_samples:,}")
        print(f"Trades taken: {self.trades_taken:,} ({self.trades_taken / self.total_samples:.1%})")
        print(
            f"Trades skipped (pred = Timeout): {self.trades_skipped:,} ({self.trades_skipped / self.total_samples:.1%})"
        )
        if self.trades_blocked > 0:
            print(f"Trades blocked (overlap): {self.trades_blocked:,}")
        if self.trades_filtered > 0:
            print(f"Trades filtered (vol/spread): {self.trades_filtered:,}")

        if self.trades_taken > 0:
            print("\nExit Reasons:")
            for reason in ("pt", "sl", "timeout"):
                count = self.exit_reason_counts.get(reason, 0)
                pct = count / self.trades_taken * 100
                print(f"  {reason}: {count:,} ({pct:.1f}%)")

        print("\nModel Strategy:")
        print(f"  Total PnL: {self.total_pnl:.6f}")
        print(f"  Avg PnL per trade: {self.avg_pnl_per_trade:.8f}")
        print(f"  Win rate: {self.win_rate:.2%}")

        print("\nAlways-Long Baseline:")
        print(f"  Total PnL: {self.baseline_pnl:.6f}")

        print("\nRandom Strategy Baseline:")
        print(f"  Total PnL: {self.random_pnl:.6f}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _simulate_one_trade(
    direction: int,
    entry_pos: int,
    bids: np.ndarray,
    asks: np.ndarray,
    mids: np.ndarray,
    price_times_ns: np.ndarray,
    timeout_ns: int,
    pt_pct: float,
    sl_pct: float,
) -> tuple:
    """
    Simulate a single trade by walking forward through price data.

    Args:
        direction: 0=Short, 1=Long
        entry_pos: index in price arrays for trade entry
        bids/asks/mids: price arrays
        price_times_ns: timestamps as int64 nanoseconds
        timeout_ns: timeout in nanoseconds
        pt_pct: profit-take threshold as fraction of entry mid
        sl_pct: stop-loss threshold as fraction of entry mid

    Returns:
        (pnl, exit_reason) where exit_reason is "pt", "sl", or "timeout"
    """
    entry_mid = mids[entry_pos]
    deadline_ns = price_times_ns[entry_pos] + timeout_ns

    # Scan range: ticks from entry+1 up to and including deadline
    end_pos = min(
        int(np.searchsorted(price_times_ns, deadline_ns, side="right")),
        len(mids),
    )
    start = entry_pos + 1

    if start >= end_pos:
        # No forward ticks; immediate exit at spread cost
        return bids[entry_pos] - asks[entry_pos], "timeout"

    window = mids[start:end_pos]

    if direction == 1:  # Long: buy at ask, exit at bid
        entry_price = asks[entry_pos]
        pt_level = entry_mid * (1 + pt_pct)
        sl_level = entry_mid * (1 - sl_pct)
        pt_mask = window >= pt_level
        sl_mask = window <= sl_level
    else:  # Short (0): sell at bid, exit at ask
        entry_price = bids[entry_pos]
        pt_level = entry_mid * (1 - pt_pct)
        sl_level = entry_mid * (1 + sl_pct)
        pt_mask = window <= pt_level
        sl_mask = window >= sl_level

    # Find first barrier hit (argmax on bool gives first True index)
    pt_idx = int(np.argmax(pt_mask)) if pt_mask.any() else len(window)
    sl_idx = int(np.argmax(sl_mask)) if sl_mask.any() else len(window)

    if pt_mask.any() and (not sl_mask.any() or pt_idx <= sl_idx):
        exit_pos = start + pt_idx
        reason = "pt"
    elif sl_mask.any():
        exit_pos = start + sl_idx
        reason = "sl"
    else:
        exit_pos = end_pos - 1
        reason = "timeout"

    if direction == 1:
        pnl = bids[exit_pos] - entry_price
    else:
        pnl = entry_price - asks[exit_pos]

    return pnl, reason


def _run_trades(
    directions: np.ndarray,
    entry_positions: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    mids: np.ndarray,
    price_times_ns: np.ndarray,
    timeout_ns: int,
    hold_ns: int,
    pt_pct: float,
    sl_pct: float,
    allow_overlap: bool,
) -> tuple:
    """
    Run simulation for an array of trade entries (chronologically sorted).

    Returns:
        (pnl, exit_reasons, trade_taken, trade_blocked)
    """
    n = len(directions)
    pnl = np.zeros(n, dtype=np.float64)
    exit_reasons = np.empty(n, dtype=object)
    exit_reasons[:] = ""
    taken = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)

    next_allowed_ns = np.int64(-1)
    n_prices = len(mids)

    for i in range(n):
        d = int(directions[i])
        if d == 2:  # Timeout / skip
            continue

        pos = int(entry_positions[i])
        if pos < 0 or pos >= n_prices:
            continue

        t_ns = price_times_ns[pos]
        if not allow_overlap and t_ns < next_allowed_ns:
            blocked[i] = True
            continue

        taken[i] = True
        p, r = _simulate_one_trade(
            d, pos, bids, asks, mids, price_times_ns,
            timeout_ns, pt_pct, sl_pct,
        )
        pnl[i] = p
        exit_reasons[i] = r

        if not allow_overlap:
            next_allowed_ns = t_ns + hold_ns

    return pnl, exit_reasons, taken, blocked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_trading(
    predictions: np.ndarray,
    price_df: pd.DataFrame,
    test_indices: pd.DatetimeIndex,
    pt_pct: float,
    sl_pct: float,
    timeout_sec: int = 30,
    allow_overlap: bool = False,
    min_vol_spread_ratio: float = None,
    probabilities: np.ndarray = None,
    threshold: float = None,
    random_seed: int = 42,
) -> SimulationResult:
    """
    Simulate trading by walking forward through execution price data.

    For each trade prediction the simulation enters at the current bid/ask,
    then scans forward tick-by-tick until a profit-take barrier, stop-loss
    barrier, or timeout is hit.  PnL is computed from actual entry/exit
    prices (Long: buy@ask sell@bid, Short: sell@bid buy@ask).

    Prediction encoding: 0=Short, 1=Long, 2=Timeout (skip)

    Args:
        predictions: Array of predictions (0=Short, 1=Long, 2=Timeout).
        price_df: Execution prices with DatetimeIndex and columns: bid, ask, mid.
                  Optionally volatility_ema, spread_pct_ema for vol/spread filter.
        test_indices: Timestamps of each prediction.
        pt_pct: Profit-take barrier as fraction of mid (e.g. 0.0005 for 5 bps).
        sl_pct: Stop-loss barrier as fraction of mid.
        timeout_sec: Maximum hold time in seconds.
        allow_overlap: If False, block trades that overlap with active positions.
        min_vol_spread_ratio: Skip trades where volatility_ema / spread_pct_ema
                              is below this ratio.
        probabilities: Optional (n, 3) class probabilities for threshold filter.
        threshold: Minimum predicted-class probability to take a trade.
        random_seed: Seed for reproducible random baseline.

    Returns:
        SimulationResult with simulation DataFrame and summary statistics.
    """
    # --- Validate ---
    if threshold is not None and probabilities is None:
        raise ValueError(
            "threshold requires probabilities (e.g. probabilities=model.predict_proba(X))"
        )
    for col in ("bid", "ask", "mid"):
        if col not in price_df.columns:
            raise ValueError(f"price_df must have '{col}' column")
    if min_vol_spread_ratio is not None:
        for col in ("volatility_ema", "spread_pct_ema"):
            if col not in price_df.columns:
                raise ValueError(
                    f"min_vol_spread_ratio requires '{col}' column in price_df"
                )

    # --- Pre-extract numpy arrays for fast access ---
    if not price_df.index.is_monotonic_increasing:
        price_df = price_df.sort_index()

    price_times_ns = price_df.index.astype(np.int64)
    bids = price_df["bid"].values.astype(np.float64)
    asks = price_df["ask"].values.astype(np.float64)
    mids = price_df["mid"].values.astype(np.float64)

    timeout_ns = int(timeout_sec * 1_000_000_000)
    hold_ns = timeout_ns

    # --- Map test_indices → positions in price_df ---
    test_times = pd.DatetimeIndex(test_indices)
    test_times_ns = test_times.astype(np.int64)
    # Last price entry at or before each test time
    entry_positions = np.searchsorted(price_times_ns, test_times_ns, side="right") - 1

    n = len(predictions)

    # --- Build working DataFrame ---
    merged = pd.DataFrame({
        "time": test_times,
        "y_pred": predictions.copy(),
        "entry_pos": entry_positions,
        "trade_filtered": False,
    })

    # --- Probability threshold ---
    if probabilities is not None and threshold is not None:
        pred_probs = probabilities[np.arange(n), predictions]
        merged["pred_prob"] = pred_probs
        low_conf = (pred_probs < threshold) & (merged["y_pred"] != 2)
        merged.loc[low_conf, "y_pred"] = 2

    # --- Vol / spread filter ---
    if min_vol_spread_ratio is not None:
        vol_ema = price_df["volatility_ema"].values
        spread_ema = price_df["spread_pct_ema"].values
        valid_pos = np.clip(entry_positions, 0, len(vol_ema) - 1)
        ratio = vol_ema[valid_pos] / (spread_ema[valid_pos] + 1e-12)
        below = (ratio < min_vol_spread_ratio) & (merged["y_pred"].values != 2)
        merged.loc[below, "trade_filtered"] = True
        merged.loc[below, "y_pred"] = 2

    # --- Sort chronologically ---
    merged = merged.sort_values("time").reset_index(drop=True)
    sorted_positions = merged["entry_pos"].values
    sorted_directions = merged["y_pred"].values.astype(int)

    # --- Model strategy ---
    sim_pnl, exit_reasons, trade_taken, trade_blocked = _run_trades(
        sorted_directions, sorted_positions,
        bids, asks, mids, price_times_ns,
        timeout_ns, hold_ns, pt_pct, sl_pct, allow_overlap,
    )
    merged["sim_pnl"] = sim_pnl
    merged["exit_reason"] = exit_reasons
    merged["trade_taken"] = trade_taken
    merged["trade_blocked"] = trade_blocked
    merged["cumulative_pnl"] = sim_pnl.cumsum()

    # --- Always-long baseline (same overlap rules) ---
    baseline_dirs = np.full(n, 1, dtype=int)
    baseline_pnl_arr, _, _, _ = _run_trades(
        baseline_dirs, sorted_positions,
        bids, asks, mids, price_times_ns,
        timeout_ns, hold_ns, pt_pct, sl_pct, allow_overlap,
    )
    merged["baseline_pnl"] = baseline_pnl_arr
    merged["cumulative_baseline"] = baseline_pnl_arr.cumsum()

    # --- Random strategy baseline (same overlap rules) ---
    np.random.seed(random_seed)
    random_dirs = np.random.choice([0, 1, 2], size=n, p=[1 / 3, 1 / 3, 1 / 3])
    merged["y_pred_random"] = random_dirs
    random_pnl_arr, _, _, _ = _run_trades(
        random_dirs, sorted_positions,
        bids, asks, mids, price_times_ns,
        timeout_ns, hold_ns, pt_pct, sl_pct, allow_overlap,
    )
    merged["random_pnl"] = random_pnl_arr
    merged["cumulative_random"] = random_pnl_arr.cumsum()

    # --- Summary statistics ---
    trades_taken_count = int(trade_taken.sum())
    trades_blocked_count = int(trade_blocked.sum())
    trades_skipped_count = int((sorted_directions == 2).sum())
    trades_filtered_count = int(merged["trade_filtered"].sum())

    exit_reason_counts: Dict[str, int] = {}
    if trades_taken_count > 0:
        for r in exit_reasons[trade_taken]:
            exit_reason_counts[r] = exit_reason_counts.get(r, 0) + 1

    taken_pnl = sim_pnl[trade_taken]

    return SimulationResult(
        simulation_df=merged,
        total_samples=n,
        trades_taken=trades_taken_count,
        trades_skipped=trades_skipped_count,
        trades_blocked=trades_blocked_count,
        trades_filtered=trades_filtered_count,
        total_pnl=float(sim_pnl.sum()),
        avg_pnl_per_trade=float(taken_pnl.mean()) if trades_taken_count > 0 else 0.0,
        win_rate=float((taken_pnl > 0).mean()) if trades_taken_count > 0 else 0.0,
        baseline_pnl=float(baseline_pnl_arr.sum()),
        random_pnl=float(random_pnl_arr.sum()),
        exit_reason_counts=exit_reason_counts,
    )


def plot_simulation_results(result: SimulationResult, window: int = 500) -> None:
    """Plot trading simulation results for v2."""
    import matplotlib.pyplot as plt

    merged = result.simulation_df

    # Figure 1: Cumulative PnL + rolling metrics
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1 = axes[0]
    ax1.plot(merged["time"], merged["cumulative_pnl"], "b-", label="Model Strategy", linewidth=1.5)
    ax1.plot(merged["time"], merged["cumulative_baseline"], "gray", label="Always Long", alpha=0.5, linewidth=1)
    ax1.plot(merged["time"], merged["cumulative_random"], "orange", label="Random", alpha=0.5, linewidth=1)
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.set_ylabel("Cumulative PnL")
    ax1.set_title("Trading Simulation (V2 Tick-by-Tick): Cumulative PnL Over Time")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    trade_pnl = merged["sim_pnl"].copy()
    trade_pnl[~merged["trade_taken"]] = np.nan
    rolling_winrate = (trade_pnl > 0).rolling(window, min_periods=100).mean()
    ax2.plot(merged["time"], rolling_winrate, "g-", label="Win Rate (rolling)", linewidth=1)
    ax2.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="50% baseline")

    trade_taken_float = merged["trade_taken"].astype(float)
    rolling_trade_freq = trade_taken_float.rolling(window, min_periods=100).mean()
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

    # Figure 2: PnL distribution + by prediction type
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    trade_pnl_values = merged.loc[merged["trade_taken"], "sim_pnl"]
    axes[0].hist(trade_pnl_values, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0, color="r", linestyle="--", alpha=0.5)
    if len(trade_pnl_values) > 0:
        axes[0].axvline(
            x=trade_pnl_values.mean(), color="g", linestyle="-",
            label=f"Mean: {trade_pnl_values.mean():.6f}",
        )
    axes[0].set_xlabel("PnL per Trade")
    axes[0].set_ylabel("Count")
    axes[0].set_title("PnL Distribution (Trades Only)")
    axes[0].legend()

    pred_pnl = (
        merged.groupby("y_pred")["sim_pnl"]
        .agg(["sum", "mean", "count"])
        .reindex([0, 1, 2], fill_value=0)
    )
    pred_pnl.index = ["Short", "Long", "Timeout"]
    pred_pnl["sum"].plot(kind="bar", ax=axes[1], color=["red", "green", "gray"], edgecolor="black")
    axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
    axes[1].set_xlabel("Prediction Type")
    axes[1].set_ylabel("Total PnL")
    axes[1].set_title("Total PnL by Prediction Type")
    axes[1].set_xticklabels(["Short", "Long", "Timeout"], rotation=0)

    plt.tight_layout()
    plt.show()

    print("\nPnL Breakdown by Prediction Type:")
    print(pred_pnl.to_string())

    # Figure 3: Exit reason breakdown
    taken_df = merged[merged["trade_taken"]].copy()
    if len(taken_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        reason_stats = (
            taken_df.groupby("exit_reason")["sim_pnl"]
            .agg(["count", "sum", "mean"])
            .reindex(["pt", "sl", "timeout"], fill_value=0)
        )

        reason_stats["count"].plot(
            kind="bar", ax=axes[0],
            color=["green", "red", "orange"], edgecolor="black",
        )
        axes[0].set_xlabel("Exit Reason")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Exit Reason Distribution")
        axes[0].set_xticklabels(["PT", "SL", "Timeout"], rotation=0)

        reason_stats["mean"].plot(
            kind="bar", ax=axes[1],
            color=["green", "red", "orange"], edgecolor="black",
        )
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[1].set_xlabel("Exit Reason")
        axes[1].set_ylabel("Average PnL")
        axes[1].set_title("Average PnL by Exit Reason")
        axes[1].set_xticklabels(["PT", "SL", "Timeout"], rotation=0)

        plt.tight_layout()
        plt.show()

        print("\nPnL by Exit Reason:")
        print(reason_stats.to_string())
