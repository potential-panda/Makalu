"""
Simulation result container and analysis utilities.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

LOG = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade."""

    entry_ts: float
    exit_ts: float
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str  # 'pt', 'sl', 'timeout', 'force'


@dataclass
class SimulationResult:
    """Container for simulation results and analysis methods."""

    trades: list[TradeRecord] = field(default_factory=list)
    pnl_history: list[tuple[float, float]] = field(default_factory=list)  # (ts, cumulative_pnl)

    @property
    def total_pnl(self) -> float:
        """Total PnL across all trades."""
        return sum(t.pnl for t in self.trades)

    @property
    def trade_count(self) -> int:
        """Total number of trades."""
        return len(self.trades)

    @property
    def win_count(self) -> int:
        """Number of winning trades (PnL > 0)."""
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def loss_count(self) -> int:
        """Number of losing trades (PnL < 0)."""
        return sum(1 for t in self.trades if t.pnl < 0)

    @property
    def win_rate(self) -> float:
        """Win rate (winning trades / total trades)."""
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count

    @property
    def avg_pnl_per_trade(self) -> float:
        """Average PnL per trade."""
        if self.trade_count == 0:
            return 0.0
        return self.total_pnl / self.trade_count

    @property
    def avg_win_pnl(self) -> float:
        """Average PnL for winning trades."""
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss_pnl(self) -> float:
        """Average PnL for losing trades."""
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        """Profit factor (gross profit / gross loss)."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from peak."""
        if not self.pnl_history:
            return 0.0
        peak = 0.0
        max_dd = 0.0
        for _, pnl in self.pnl_history:
            if pnl > peak:
                peak = pnl
            dd = peak - pnl
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "entry_time": datetime.fromtimestamp(t.entry_ts),
                "exit_time": datetime.fromtimestamp(t.exit_ts),
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "hold_time_sec": t.exit_ts - t.entry_ts,
            })

        return pd.DataFrame(records)

    def get_pnl_by_exit_reason(self) -> dict[str, dict]:
        """Get PnL statistics grouped by exit reason."""
        reasons = {}
        for t in self.trades:
            reason = t.exit_reason
            if reason not in reasons:
                reasons[reason] = {"count": 0, "total_pnl": 0.0, "trades": []}
            reasons[reason]["count"] += 1
            reasons[reason]["total_pnl"] += t.pnl
            reasons[reason]["trades"].append(t.pnl)

        for reason, data in reasons.items():
            data["avg_pnl"] = data["total_pnl"] / data["count"] if data["count"] > 0 else 0.0
            data["win_rate"] = sum(1 for p in data["trades"] if p > 0) / data["count"] if data["count"] > 0 else 0.0
            del data["trades"]

        return reasons

    def get_pnl_by_direction(self) -> dict[str, dict]:
        """Get PnL statistics grouped by direction."""
        directions = {"LONG": {"count": 0, "total_pnl": 0.0, "wins": 0}, "SHORT": {"count": 0, "total_pnl": 0.0, "wins": 0}}
        for t in self.trades:
            key = "LONG" if t.direction == 1 else "SHORT"
            directions[key]["count"] += 1
            directions[key]["total_pnl"] += t.pnl
            if t.pnl > 0:
                directions[key]["wins"] += 1

        for key, data in directions.items():
            data["avg_pnl"] = data["total_pnl"] / data["count"] if data["count"] > 0 else 0.0
            data["win_rate"] = data["wins"] / data["count"] if data["count"] > 0 else 0.0

        return directions

    def plot_pnl(self, figsize: tuple = (12, 6)) -> plt.Figure:
        """Plot cumulative PnL over time."""
        if not self.pnl_history:
            LOG.warning("No PnL history to plot")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        times = [datetime.fromtimestamp(ts) for ts, _ in self.pnl_history]
        pnls = [pnl for _, pnl in self.pnl_history]

        ax.plot(times, pnls, linewidth=1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative PnL")
        ax.set_title(f"Cumulative PnL (Total: {self.total_pnl:.6f}, Trades: {self.trade_count})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_trades_by_hour(self, figsize: tuple = (12, 4)) -> plt.Figure:
        """Plot trade count and PnL by hour of day."""
        if not self.trades:
            LOG.warning("No trades to plot")
            return None

        df = self.to_dataframe()
        df["hour"] = df["entry_time"].dt.hour

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Trade count by hour
        hourly_count = df.groupby("hour").size()
        ax1.bar(hourly_count.index, hourly_count.values)
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Trade Count")
        ax1.set_title("Trades by Hour")
        ax1.set_xticks(range(24))

        # PnL by hour
        hourly_pnl = df.groupby("hour")["pnl"].sum()
        colors = ["green" if p > 0 else "red" for p in hourly_pnl.values]
        ax2.bar(hourly_pnl.index, hourly_pnl.values, color=colors)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Total PnL")
        ax2.set_title("PnL by Hour")
        ax2.set_xticks(range(24))

        plt.tight_layout()
        return fig

    def plot_pnl_distribution(self, figsize: tuple = (10, 4)) -> plt.Figure:
        """Plot PnL distribution histogram."""
        if not self.trades:
            LOG.warning("No trades to plot")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        pnls = [t.pnl for t in self.trades]

        ax.hist(pnls, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=self.avg_pnl_per_trade, color='blue', linestyle='--', alpha=0.7, label=f'Mean: {self.avg_pnl_per_trade:.6f}')

        ax.set_xlabel("PnL")
        ax.set_ylabel("Frequency")
        ax.set_title("PnL Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self) -> dict:
        """Get summary statistics as a dictionary."""
        return {
            "trade_count": self.trade_count,
            "total_pnl": self.total_pnl,
            "avg_pnl_per_trade": self.avg_pnl_per_trade,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "avg_win_pnl": self.avg_win_pnl,
            "avg_loss_pnl": self.avg_loss_pnl,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "pnl_by_exit_reason": self.get_pnl_by_exit_reason(),
            "pnl_by_direction": self.get_pnl_by_direction(),
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the simulation results."""
        print("=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Total Trades:      {self.trade_count}")
        print(f"Total PnL:         {self.total_pnl:.6f}")
        print(f"Avg PnL/Trade:     {self.avg_pnl_per_trade:.6f}")
        print(f"Win Rate:          {self.win_rate:.2%}")
        print(f"Profit Factor:     {self.profit_factor:.2f}")
        print(f"Max Drawdown:      {self.max_drawdown:.6f}")
        print()
        print("By Exit Reason:")
        for reason, stats in self.get_pnl_by_exit_reason().items():
            print(f"  {reason:10s}: {stats['count']:4d} trades, PnL={stats['total_pnl']:.6f}, WR={stats['win_rate']:.2%}")
        print()
        print("By Direction:")
        for direction, stats in self.get_pnl_by_direction().items():
            print(f"  {direction:10s}: {stats['count']:4d} trades, PnL={stats['total_pnl']:.6f}, WR={stats['win_rate']:.2%}")
        print("=" * 60)
