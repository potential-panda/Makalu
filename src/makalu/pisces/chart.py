"""Pisces strategy chart: taker slippage, time-to-fill, cumulative PnL."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


_SIDE_COLORS = {"BUY": "#4c9be8", "SELL": "#e87b4c"}


def _fmt_axes(ax, ylabel: str):
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def plot_pisces_metrics(
    order_df: pd.DataFrame,
    pnl_df: pd.DataFrame,
    title: str = "Pisces Metrics",
) -> plt.Figure:
    """Three-panel figure:
    1. Taker slippage (JPY) over time
    2. Taker time-to-fill (seconds) over time  [≈ maker-fill → taker-fill]
    3. Cumulative PnL and net (after fees)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    taker = pd.DataFrame()
    if not order_df.empty:
        taker = order_df[order_df["order_type"] == "DEFAULT"].copy()

    # --- Panel 1: Taker slippage ---
    ax = axes[0]
    if not taker.empty:
        for side, grp in taker.groupby("side"):
            ax.scatter(
                grp["ts"], grp["slippage"],
                label=side, color=_SIDE_COLORS.get(side, "gray"),
                s=20, alpha=0.7,
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No taker orders", transform=ax.transAxes, ha="center")
    _fmt_axes(ax, "Taker slippage (JPY)")

    # --- Panel 2: Taker time-to-fill (= maker fill → taker fill) ---
    ax = axes[1]
    if not taker.empty:
        valid = taker[taker["time_to_fill"] > 0]
        for side, grp in valid.groupby("side"):
            ax.scatter(
                grp["ts"], grp["time_to_fill"],
                label=side, color=_SIDE_COLORS.get(side, "gray"),
                s=20, alpha=0.7,
            )
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No taker orders", transform=ax.transAxes, ha="center")
    _fmt_axes(ax, "Taker time-to-fill (s)\n[≈ maker→taker latency]")

    # --- Panel 3: Cumulative PnL ---
    ax = axes[2]
    if not pnl_df.empty:
        ax.step(pnl_df["ts"], pnl_df["cumulative_pnl"], where="post",
                label="gross PnL", color="#4c9be8", linewidth=1.2)
        ax.step(pnl_df["ts"], pnl_df["cumulative_net"], where="post",
                label="net (after fees)", color="#e87b4c", linewidth=1.2, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=8, loc="upper left")
        total_net = pnl_df["cumulative_net"].iloc[-1]
        n_trades = len(pnl_df)
        ax.set_title(f"Net: ¥{total_net:.2f}  ({n_trades} round-trips)", fontsize=8, loc="right")
    else:
        ax.text(0.5, 0.5, "No PnL data", transform=ax.transAxes, ha="center")
    _fmt_axes(ax, "Cumulative PnL (JPY)")

    fig.tight_layout()
    return fig
