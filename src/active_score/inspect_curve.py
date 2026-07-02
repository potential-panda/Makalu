"""
Diagnostic tool: recompute the weekly ActivityCurve straight from VolSpread
and dump it as data + a heatmap PNG for eye confirmation.

Unlike `cli.py populate ActivityCurve` (the production path, which writes a
gold-layer snapshot per as-of date), this never touches gold parquet -- it's
meant to be run ad hoc while iterating on a market/date range, to sanity
check whether a real day-of-week pattern exists before trusting the curve.

Shares its aggregation core (active_score.weekly_curve) with the production
pipeline (pipeline/curve_steps.py), so both compute the curve the same way.

Also derives a simple weekday active/inactive UTC+JST time window from the
curve -- see doc/vol_spread.md's "Active/Inactive Window Simplification" for
why a coarse contiguous block is preferred over trusting raw bin values.

Usage:
    python src/active_score/inspect_curve.py \
        --exchange GMO --symbol ETH_JPY --execution-size 0.1 \
        --start 2025-07-01 --end 2026-06-20
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from qate.core.model import Market
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

from active_score.weekly_curve import aggregate_weekly, load_volspread_series
from feature_engineering.volatility import VolSpreadConfig

LOG = logging.getLogger(__name__)
ENV_NAME = EnvName.MARIO_MARIO

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Kept in sync with populate.py's _volspread_configs() / silver.py's own
# VolSpreadStep list -- the 4 configs actually populated for VolSpread.
VOL_CONFIGS = [
    VolSpreadConfig(60, 300, 120),
    VolSpreadConfig(60, 900, 300),
    VolSpreadConfig(180, 900, 300),
    VolSpreadConfig(300, 900, 300),
]


def find_active_window(hourly_scores: np.ndarray) -> tuple[int, int, float, float] | None:
    """Longest contiguous UTC-hour run above the day's own mean score -- the
    simplification described in doc/vol_spread.md's "Active/Inactive Window
    Simplification" (raw 5-min bins are too noisy to schedule against; the
    active/inactive *ratio* of a coarse contiguous block is what actually
    holds up across independent halves of the training data).

    Rotates to start scanning at the quietest hour so the active run (assumed
    not to wrap past the daily minimum) is captured as one simple slice.
    Returns (start_hour, end_hour, inside_mean, outside_mean), UTC, half-open
    [start, end); None if every hour is on the same side of the threshold.
    """
    n = len(hourly_scores)
    quiet_hour = int(np.argmin(hourly_scores))
    rotated = np.roll(hourly_scores, -quiet_hour)
    threshold = hourly_scores.mean()
    above = rotated > threshold

    best_start, best_len, cur_start, cur_len = None, 0, None, 0
    for i, v in enumerate(above):
        if v:
            if cur_start is None:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_start, cur_len = None, 0
    if best_start is None:
        return None

    start = (quiet_hour + best_start) % n
    end = (start + best_len) % n
    idx = [(start + i) % n for i in range(best_len)]
    outside_idx = [h for h in range(n) if h not in idx]
    return start, end, float(hourly_scores[idx].mean()), float(hourly_scores[outside_idx].mean())


def _fmt_utc_jst(hour: int) -> str:
    return f"UTC {hour:02d}:00 (JST {(hour + 9) % 24:02d}:00)"


def _print_active_window(scores: np.ndarray, bin_size_sec: int) -> dict:
    """Weekday-only (Mon-Fri) active/inactive window; weekend is always
    reported as fully inactive -- see doc/vol_spread.md for why (weekend
    activity correlates poorly with the average day, not just around
    exchange maintenance reopen windows)."""
    bins_per_hour = 3600 // bin_size_sec
    weekday_avg = scores[0:5].mean(axis=0)
    n_hours = len(weekday_avg) // bins_per_hour
    hourly = weekday_avg[: n_hours * bins_per_hour].reshape(n_hours, bins_per_hour).mean(axis=1)

    window = find_active_window(hourly)
    print("\n── Active/inactive window (weekday, see doc/vol_spread.md) ──")
    if window is None:
        print("  No contiguous active window found (curve too flat).")
        return {}
    start, end, inside_mean, outside_mean = window
    print(f"  Active:   {_fmt_utc_jst(start)} -> {_fmt_utc_jst(end)}  (mean={inside_mean:.4f})")
    print(f"  Inactive: {_fmt_utc_jst(end)} -> {_fmt_utc_jst(start)}  (mean={outside_mean:.4f})")
    print(f"  Weekend:  inactive (all)")
    print(f"  active/inactive ratio: {inside_mean / outside_mean:.2f}x")
    print("───────────────────────────────────────────────────────────\n")
    return {
        "weekday_active_utc": [f"{start:02d}:00", f"{end:02d}:00"],
        "weekday_active_jst": [f"{(start + 9) % 24:02d}:00", f"{(end + 9) % 24:02d}:00"],
        "weekend": "inactive",
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
    }


def _print_weekday_summary(scores: np.ndarray) -> None:
    """Per-weekday mean/std/min/max, plus each day's correlation with the
    across-week-average day curve -- a quick numeric eye-check for whether
    the pattern is genuinely weekly or just the same day repeated 7 times
    (correlations near 1.0 for every day => no real day-of-week structure)."""
    avg_day = scores.mean(axis=0)
    print("\n── Per-weekday summary ──────────────────────────────────────")
    print(f"  {'day':<5}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}  {'corr_vs_avg_day':>16}")
    print(f"  {'-'*58}")
    for d in range(7):
        row = scores[d]
        corr = np.corrcoef(row, avg_day)[0, 1] if row.std() > 0 and avg_day.std() > 0 else float("nan")
        print(f"  {DAY_NAMES[d]:<5}  {row.mean():>7.4f}  {row.std():>7.4f}  {row.min():>7.4f}  {row.max():>7.4f}  {corr:>16.4f}")
    print("────────────────────────────────────────────────────────────\n")


def _plot_heatmap(scores: np.ndarray, bin_size_sec: int, title: str, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_bins = scores.shape[1]
    bins_per_hour = 3600 // bin_size_sec

    fig, ax = plt.subplots(figsize=(18, 4))
    im = ax.imshow(scores, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")

    ax.set_yticks(range(7))
    ax.set_yticklabels(DAY_NAMES)

    tick_positions = list(range(0, n_bins, 6 * bins_per_hour))
    tick_labels = [f"{(p // bins_per_hour):02d}h" for p in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)

    ax.set_xlabel("Time of day (UTC)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Activity score (0-1)", fraction=0.02, pad=0.01)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOG.info(f"Heatmap saved to {output_path}")


def main(
    market: Market,
    execution_size: str,
    dt_range: DtRange,
    bin_size_sec: int,
    output_dir: Path,
) -> None:
    parquet_store = ParquetStore(sys_env.get_parquet_root_dir())
    param_ids = [c.param_id for c in VOL_CONFIGS]

    LOG.info(f"Recomputing ActivityCurve from VolSpread for {market} / execution_size={execution_size}")
    LOG.info(f"  param_ids ({len(param_ids)}): {param_ids}")
    LOG.info(f"  dt_range: {dt_range.start_str} → {dt_range.end_str}")
    LOG.info(f"  bin_size_sec: {bin_size_sec}")

    series_list = load_volspread_series(parquet_store, market, execution_size, param_ids, dt_range)
    if not series_list:
        raise ValueError("No valid VolSpread data found for any of the given param_ids -- nothing to plot.")
    LOG.info(f"Combining {len(series_list)} normalized (volatility_ema/spread_pct_ema) series.")

    scores, std_scores = aggregate_weekly(series_list, bin_size_sec)

    _print_weekday_summary(scores)
    active_window = _print_active_window(scores, bin_size_sec)

    stem = f"activity_curve_{market.exchange_name.name}_{market.symbol.name}_es{execution_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    png_path = output_dir / f"{stem}.png"

    payload = {
        "bin_size_sec": bin_size_sec,
        "param_ids": param_ids,
        "market": f"{market.exchange_name.name}:{market.symbol.name}",
        "execution_size": execution_size,
        "trained_on": f"{dt_range.start_str} to {dt_range.end_str}",
        "scores": scores.tolist(),
        "std": std_scores.tolist(),
        "active_window": active_window,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    LOG.info(f"Curve data saved to {json_path}")

    title = f"ActivityCurve (recomputed) — {market.exchange_name.name}:{market.symbol.name} es={execution_size}\n{dt_range.start_str} → {dt_range.end_str}  (bin={bin_size_sec}s)"
    _plot_heatmap(scores, bin_size_sec, title, png_path)

    print(f"Done.\n  data → {json_path}\n  plot → {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute weekly ActivityCurve from VolSpread and plot for eye confirmation.")
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--execution-size", required=True, help="Execution size partition (e.g. 0.1)")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYYMMDD or YYYY-MM-DD (default: today)")
    parser.add_argument("--bin-size", type=int, default=300, help="Curve bin size in seconds (default: 300)")
    parser.add_argument("--output-dir", default="models", help="Output directory for JSON + PNG (default: models/)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    env = Env.load_env(sys_env.get_env_root_dir(), ENV_NAME)
    env.visit()

    main(
        market=Market.from_str(f"{args.exchange}:{args.symbol}"),
        execution_size=args.execution_size,
        dt_range=DtRange.from_strings(args.start, args.end),
        bin_size_sec=args.bin_size,
        output_dir=Path(args.output_dir),
    )
