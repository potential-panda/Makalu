import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from qate.core.model import Market
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

from active_score.curve_builder import build_curve
from feature_engineering.volatility import VolSpreadConfig

LOG = logging.getLogger(__name__)
ENV_NAME = EnvName.MARIO_MARIO

# VolSpread configs kept in sync with the ExecutionPriceDerived stage.
VOL_CONFIGS = [
    VolSpreadConfig(60, 300, 120),
    VolSpreadConfig(60, 900, 300),
    VolSpreadConfig(180, 900, 300),
    VolSpreadConfig(300, 900, 300),
]

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def plot_curve(
    scores: list[list[float]],
    std: list[list[float]],
    bin_size_sec: int,
    thresholds: list[tuple[str, float]],
    output_path: Path,
) -> None:
    n_bins = len(scores[0])
    xs = np.arange(7 * n_bins)
    mean = np.array(scores).flatten()
    sigma = np.array(std).flatten()

    fig, ax = plt.subplots(figsize=(18, 4))

    # Std bands (widest first so narrower ones draw on top)
    band_alphas = [0.12, 0.18, 0.25]
    for n, alpha in zip([3, 2, 1], band_alphas):
        ax.fill_between(xs, mean - n * sigma, mean + n * sigma, alpha=alpha, color="steelblue", label=f"±{n}σ")

    # Mean curve
    ax.plot(xs, mean, linewidth=1.0, color="steelblue", label="mean")

    # Horizontal threshold lines
    hline_colors = ["tomato", "orange", "green", "purple"]
    hline_styles = ["--", "-.", ":", "-"]
    for (label, y), color, ls in zip(thresholds, hline_colors, hline_styles):
        ax.axhline(y, color=color, linewidth=0.9, linestyle=ls, label=f"{label} ({y:.3f})")

    # Vertical lines and day labels at each day boundary
    for d in range(7):
        x = d * n_bins
        ax.axvline(x, color="gray", linewidth=0.6, linestyle="--")
        ax.text(
            x + n_bins / 2,
            1.02,
            DAY_NAMES[d],
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
        )

    # X-axis: hour labels every 6 h within each day
    bins_per_hour = 3600 // bin_size_sec
    tick_positions = []
    tick_labels = []
    for d in range(7):
        for h in range(0, 24, 6):
            pos = d * n_bins + h * bins_per_hour
            tick_positions.append(pos)
            tick_labels.append(f"{h:02d}h" if h > 0 else "")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlim(0, 7 * n_bins)
    y_max = min((mean + 3 * sigma).max() * 1.05, (mean + 3 * sigma).max() + 0.1)
    ax.set_ylim(min((mean - 3 * sigma).min() - 0.05, -0.05), y_max)
    ax.set_ylabel("Activity Score")
    ax.set_title(f"Weekly Activity Curve  (bin={bin_size_sec}s)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plot_path = output_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    LOG.info(f"Plot saved to {plot_path}")


def main(market: Market, execution_size: str, dt_range: DtRange, bin_size_sec: int, output_path: Path) -> None:
    parquet_store = ParquetStore(sys_env.get_parquet_root_dir())
    param_ids = [c.param_id for c in VOL_CONFIGS]

    LOG.info(f"Building active score curve for {market} / {execution_size}")
    LOG.info(f"  param_ids ({len(param_ids)}): {param_ids}")
    LOG.info(f"  dt_range: {dt_range.start_str} → {dt_range.end_str}")
    LOG.info(f"  bin_size_sec: {bin_size_sec}")

    build_curve(
        parquet_store=parquet_store,
        market=market,
        execution_size_str=execution_size,
        param_ids=param_ids,
        dt_range=dt_range,
        output_path=output_path,
        bin_size_sec=bin_size_sec,
    )

    # Load the saved curve, compute coverage stats, re-save, then plot
    import json

    with open(output_path) as f:
        payload = json.load(f)

    flat = np.array(payload["scores"]).flatten()
    mu = float(np.mean(flat))
    sigma = float(np.std(flat))
    total = len(flat)

    thresholds = [
        ("std", mu),
        ("std1dev", mu + sigma),
        ("std2dev", mu + 2 * sigma),
        ("std3dev", mu + 3 * sigma),
    ]

    coverage = {}
    print("\n── Active-time coverage ─────────────────────────────")
    print(f"  total bins : {total}  (bin_size={payload['bin_size_sec']}s)")
    print(f"  mean score : {mu:.4f}   std: {sigma:.4f}")
    print(f"  {'threshold':<12}  {'value':>7}  {'bins above':>10}  {'% of week':>10}")
    print(f"  {'-'*50}")
    for label, thr in thresholds:
        above = int(np.sum(flat > thr))
        pct = above / total * 100
        coverage[label] = {"threshold": round(thr, 6), "bins_above": above, "pct_of_week": round(pct, 2)}
        print(f"  {label:<12}  {thr:>7.4f}  {above:>10d}  {pct:>9.2f}%")
    print("─────────────────────────────────────────────────────\n")

    payload["thresholds"] = coverage
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    plot_curve(payload["scores"], payload["std"], payload["bin_size_sec"], thresholds, output_path)

    print(f"Done. Curve → {output_path}")
    print(f"Plot  → {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ActiveScore weekly curve from VolSpread parquet data.")
    parser.add_argument("--market", required=True, help="Market in format EXCHANGE:SYMBOL (e.g. GMO:BTC_JPY)")
    parser.add_argument("--execution-size", required=True, help="Execution size string (e.g. 10bps)")
    parser.add_argument("--from-dt", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--to-dt", help="End date YYYYMMDD (default: today)")
    parser.add_argument(
        "--bin-size", type=int, default=300, help="Curve bin size in seconds (e.g. 60, 300, 900; default: 300)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: models/active_score_curve_{SYMBOL}.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    env = Env.load_env(sys_env.ENV_DIR, ENV_NAME)
    env.enable_multiprocess_logging = True
    env.visit()

    market = Market.from_str(args.market)
    output_path = Path(args.output) if args.output else Path(f"models/active_score_curve_{market.symbol.name}.json")

    main(
        market=market,
        execution_size=args.execution_size,
        dt_range=DtRange.from_strings(args.from_dt, args.to_dt),
        bin_size_sec=args.bin_size,
        output_path=output_path,
    )
