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
from typing import Dict

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

        print("\nModel Strategy (PnL in BPS):")
        print(f"  Total PnL: {self.total_pnl:.2f} bps")
        print(f"  Avg PnL per trade: {self.avg_pnl_per_trade:.2f} bps")
        print(f"  Win rate: {self.win_rate:.2%}")

        print("\nAlways-Long Baseline:")
        print(f"  Total PnL: {self.baseline_pnl:.2f} bps")

        print("\nRandom Strategy Baseline:")
        print(f"  Total PnL: {self.random_pnl:.2f} bps")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _simulate_one_trade(
    direction: int,
    entry_pos: int,
    bids: np.ndarray,
    asks: np.ndarray,
    mids: np.ndarray,
    spread_pct_ema: np.ndarray,
    price_times_ns: np.ndarray,
    timeout_ns: int,
    pt_multiplier: float,
    sl_multiplier: float,
    fee_bps: float,
) -> tuple:
    """
    Simulate a single trade by walking forward through price data.

    Returns:
        (pnl_bps, exit_reason, exit_pos)
    """
    entry_mid = mids[entry_pos]
    entry_spread_pct = spread_pct_ema[entry_pos]
    deadline_ns = price_times_ns[entry_pos] + timeout_ns

    # Dynamic barriers based on current spread
    pt_pct = pt_multiplier * entry_spread_pct
    sl_pct = sl_multiplier * entry_spread_pct

    # Scan range: ticks from entry+1 up to and including deadline
    end_pos = min(
        int(np.searchsorted(price_times_ns, deadline_ns, side="right")),
        len(mids),
    )
    start = entry_pos + 1

    if start >= end_pos:
        # No forward ticks; immediate exit at spread cost
        pnl_abs = bids[entry_pos] - asks[entry_pos]
        pnl_bps = (pnl_abs / (entry_mid + 1e-12)) * 10000 - (2 * fee_bps)
        return pnl_bps, "timeout", entry_pos

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

    # Find first barrier hit
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
        pnl_abs = bids[exit_pos] - entry_price
    else:
        pnl_abs = entry_price - asks[exit_pos]

    pnl_bps = (pnl_abs / (entry_mid + 1e-12)) * 10000 - (2 * fee_bps)
    return pnl_bps, reason, exit_pos


def _run_trades(
    directions: np.ndarray,
    entry_positions: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    mids: np.ndarray,
    spread_pct_ema: np.ndarray,
    price_times_ns: np.ndarray,
    timeout_ns: int,
    pt_multiplier: float,
    sl_multiplier: float,
    fee_bps: float,
    entry_latency_ticks: int,
    allow_overlap: bool,
) -> tuple:
    """
    Run simulation for an array of trade entries.

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
        if d == 2:
            continue

        pos = int(entry_positions[i]) + entry_latency_ticks
        if pos < 0 or pos >= n_prices:
            continue

        t_ns = price_times_ns[pos]
        if not allow_overlap and t_ns < next_allowed_ns:
            blocked[i] = True
            continue

        taken[i] = True
        p, r, ex_pos = _simulate_one_trade(
            d,
            pos,
            bids,
            asks,
            mids,
            spread_pct_ema,
            price_times_ns,
            timeout_ns,
            pt_multiplier,
            sl_multiplier,
            fee_bps,
        )
        pnl[i] = p
        exit_reasons[i] = r

        if not allow_overlap:
            next_allowed_ns = price_times_ns[ex_pos]

    return pnl, exit_reasons, taken, blocked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_trading(
    predictions: np.ndarray,
    price_df: pd.DataFrame,
    pt_multiplier: float,
    sl_multiplier: float,
    timeout_sec: int,
    probabilities: np.ndarray,
    threshold: float,
    fee_bps: float = 0.0,
    entry_latency_ticks: int = 1,
    allow_overlap: bool = False,
    min_vol_spread_ratio: float = None,
    random_seed: int = 42,
) -> SimulationResult:
    """
    Simulate trading by walking forward through execution price data.

    Prediction encoding: 0=Short, 1=Long, 2=Timeout (skip)
    PnL is reported in Basis Points (BPS).
    Barriers are dynamic: pt_pct = pt_multiplier * spread_pct_ema

    Args:
        predictions: Array of predictions (0=Short, 1=Long, 2=Timeout).
        price_df: Execution prices with DatetimeIndex and columns: bid, ask, mid, spread_pct_ema.
                  Must be pre-aligned to predictions (same index, same order).
                  If it contains a 'label' column, it will be included as 'original_label' in the result.
        pt_multiplier: Multiplier for spread_pct_ema to set profit-take barrier.
        sl_multiplier: Multiplier for spread_pct_ema to set stop-loss barrier.
        timeout_sec: Maximum hold time in seconds.
        fee_bps: Exchange commission per side in basis points (e.g. 2.0).
        entry_latency_ticks: Number of ticks to delay entry after signal.
        allow_overlap: If False, block trades that overlap with active positions.
        min_vol_spread_ratio: Volatility-to-spread filter.
        probabilities: Optional class probabilities for threshold filter.
        threshold: Minimum predicted-class probability to take a trade.
        random_seed: Seed for reproducible random baseline.
    """
    # --- Validate ---
    if threshold is not None and probabilities is None:
        raise ValueError("threshold requires probabilities")
    for col in ("bid", "ask", "mid", "spread_pct_ema"):
        if col not in price_df.columns:
            raise ValueError(f"price_df must have '{col}' column")

    # --- Pre-extract numpy arrays ---
    if not price_df.index.is_monotonic_increasing:
        price_df = price_df.sort_index()

    price_times_ns = price_df.index.astype(np.int64)
    bids = price_df["bid"].values.astype(np.float64)
    asks = price_df["ask"].values.astype(np.float64)
    mids = price_df["mid"].values.astype(np.float64)
    spread_pct_ema = price_df["spread_pct_ema"].values.astype(np.float64)

    timeout_ns = int(timeout_sec * 1_000_000_000)
    test_times = price_df.index
    entry_positions = np.arange(len(test_times), dtype=np.int64)

    n = len(predictions)
    merged = pd.DataFrame(
        {
            "time": test_times,
            "y_pred": predictions.copy(),
            "entry_pos": entry_positions,
            "trade_filtered": False,
            "original_label": price_df["label"].values,
        }
    )

    # --- Threshold / Vol Filter ---
    if probabilities is not None and threshold is not None:
        pred_probs = probabilities[np.arange(n), predictions]
        low_conf = (pred_probs < threshold) & (merged["y_pred"] != 2)
        merged.loc[low_conf, "y_pred"] = 2

    if min_vol_spread_ratio is not None:
        volatility = price_df["volatility"].values
        # spread_ema is already checked/used
        ratio = volatility[np.clip(entry_positions, 0, len(volatility) - 1)] / (
            spread_pct_ema[np.clip(entry_positions, 0, len(spread_pct_ema) - 1)] + 1e-12
        )
        below = (ratio < min_vol_spread_ratio) & (merged["y_pred"].values != 2)
        merged.loc[below, "trade_filtered"] = True
        merged.loc[below, "y_pred"] = 2

    # --- Sort chronologically ---
    merged = merged.sort_values("time").reset_index(drop=True)
    sorted_positions = merged["entry_pos"].values
    sorted_directions = merged["y_pred"].values.astype(int)

    # --- Model strategy ---
    sim_pnl, exit_reasons, trade_taken, trade_blocked = _run_trades(
        sorted_directions,
        sorted_positions,
        bids,
        asks,
        mids,
        spread_pct_ema,
        price_times_ns,
        timeout_ns,
        pt_multiplier,
        sl_multiplier,
        fee_bps,
        entry_latency_ticks,
        allow_overlap,
    )
    merged["sim_pnl"] = sim_pnl
    merged["exit_reason"] = exit_reasons
    merged["trade_taken"] = trade_taken
    merged["trade_blocked"] = trade_blocked
    merged["cumulative_pnl"] = sim_pnl.cumsum()

    # --- Always-long baseline ---
    baseline_dirs = np.full(n, 1, dtype=int)
    baseline_pnl_arr, _, _, _ = _run_trades(
        baseline_dirs,
        sorted_positions,
        bids,
        asks,
        mids,
        spread_pct_ema,
        price_times_ns,
        timeout_ns,
        pt_multiplier,
        sl_multiplier,
        fee_bps,
        entry_latency_ticks,
        allow_overlap,
    )
    merged["baseline_pnl"] = baseline_pnl_arr
    merged["cumulative_baseline"] = baseline_pnl_arr.cumsum()

    # --- Random strategy baseline ---
    np.random.seed(random_seed)
    random_dirs = np.random.choice([0, 1, 2], size=n, p=[1 / 3, 1 / 3, 1 / 3])
    random_pnl_arr, _, _, _ = _run_trades(
        random_dirs,
        sorted_positions,
        bids,
        asks,
        mids,
        spread_pct_ema,
        price_times_ns,
        timeout_ns,
        pt_multiplier,
        sl_multiplier,
        fee_bps,
        entry_latency_ticks,
        allow_overlap,
    )
    merged["random_pnl"] = random_pnl_arr
    merged["cumulative_random"] = random_pnl_arr.cumsum()

    # --- Summary statistics ---
    trades_taken_count = int(trade_taken.sum())
    taken_pnl = sim_pnl[trade_taken]

    exit_reason_counts: Dict[str, int] = {}
    if trades_taken_count > 0:
        for r in exit_reasons[trade_taken]:
            exit_reason_counts[r] = exit_reason_counts.get(r, 0) + 1

    return SimulationResult(
        simulation_df=merged,
        total_samples=n,
        trades_taken=trades_taken_count,
        trades_skipped=int((sorted_directions == 2).sum()),
        trades_blocked=int(trade_blocked.sum()),
        trades_filtered=int(merged["trade_filtered"].sum()),
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
            x=trade_pnl_values.mean(),
            color="g",
            linestyle="-",
            label=f"Mean: {trade_pnl_values.mean():.6f}",
        )
    axes[0].set_xlabel("PnL per Trade")
    axes[0].set_ylabel("Count")
    axes[0].set_title("PnL Distribution (Trades Only)")
    axes[0].legend()

    pred_pnl = merged.groupby("y_pred")["sim_pnl"].agg(["sum", "mean", "count"]).reindex([0, 1, 2], fill_value=0)
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
            kind="bar",
            ax=axes[0],
            color=["green", "red", "orange"],
            edgecolor="black",
        )
        axes[0].set_xlabel("Exit Reason")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Exit Reason Distribution")
        axes[0].set_xticklabels(["PT", "SL", "Timeout"], rotation=0)

        reason_stats["mean"].plot(
            kind="bar",
            ax=axes[1],
            color=["green", "red", "orange"],
            edgecolor="black",
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
