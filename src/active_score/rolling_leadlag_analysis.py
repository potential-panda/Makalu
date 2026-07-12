"""
Rolling N-month LeadLagCurve analysis: recompute the weekly curve over a
sliding window of calendar months and track how its three headline numbers
-- chronological stability corr, interleaved stability corr, and the
active/inactive ratio -- evolve over time. Built to test whether lead-lag
structure itself drifts (and whether that drift lines up with anything else,
e.g. strategy performance).

Reuses the same aggregation core (active_score.weekly_curve) and diagnostics
(active_score.curve_diagnostics) as inspect_leadlag_curve.py -- this script
just repeats that computation over a sliding window of calendar months
instead of a single fixed range, and adds a trend chart across windows.

Usage:
    python src/active_score/rolling_leadlag_analysis.py \
        --pair eth --configs mid \
        --start-month 2025-07 --end-window-start-month 2026-04 --window-months 3
"""

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

from qate.core.model import Market
from qate.env import sys_env
from qate.env.env import Env, EnvName
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

from active_score.curve_diagnostics import plot_heatmap, print_active_window, run_stability_check
from active_score.weekly_curve import aggregate_weekly, load_leadlag_series
from lead_lag.lead_lag import LEAD_LAG_CONFIG_MID, LEAD_LAG_CONFIG_NEARHFT
from pipeline.market_pair import LEAD_LAG_PAIRS

LOG = logging.getLogger(__name__)
ENV_NAME = EnvName("MARIO_MARIO")

LEADLAG_CONFIGS = {"nearhft": LEAD_LAG_CONFIG_NEARHFT, "mid": LEAD_LAG_CONFIG_MID}


def _resolve_pair(pair_key: str) -> tuple[Market, Market]:
    pair = LEAD_LAG_PAIRS.get(pair_key)
    if pair is None:
        raise ValueError(f"Unknown pair '{pair_key}'. Known: {list(LEAD_LAG_PAIRS)}")
    return pair


def _parse_ym(s: str) -> tuple[int, int]:
    y, m = s.split("-")
    return int(y), int(m)


def _add_months(ym: tuple[int, int], n: int) -> tuple[int, int]:
    y, m = ym
    total = y * 12 + (m - 1) + n
    return total // 12, total % 12 + 1


def _month_window_dates(start_ym: tuple[int, int], window_months: int) -> tuple[str, str]:
    """Calendar-month window [start_ym, start_ym + window_months) as inclusive date strings."""
    start_date = date(start_ym[0], start_ym[1], 1)
    end_ym = _add_months(start_ym, window_months)
    end_date = date(end_ym[0], end_ym[1], 1) - timedelta(days=1)
    return start_date.isoformat(), end_date.isoformat()


def raw_strength_stats(
    parquet_store: ParquetStore,
    market_lead: Market,
    market_lag: Market,
    param_ids: list[str],
    dt_range: DtRange,
) -> dict | None:
    """Mean/median |ll_max_corr| in *raw* units (no z-score, no min-max
    normalization) -- the actual real-world correlation magnitude, as opposed
    to the shape-only [0,1] scores used for the heatmap/stability check.
    Weekday-only excludes Sat/Sun so the maintenance-closure dead zone
    doesn't pull the average down."""
    import numpy as np
    import pandas as pd

    all_vals = []
    weekday_vals = []
    for param_id in param_ids:
        store = parquet_store.create_lead_lag(market_lead, market_lag, param_id)
        df = store.read(dt_range)
        if df.empty:
            continue
        corr_cols = [c for c in df.columns if c.startswith("ll_max_corr")]
        if not corr_cols:
            continue
        index = pd.to_datetime(df["ts"], unit="s", utc=True)
        is_weekday = index.dt.dayofweek < 5
        for col in corr_cols:
            vals = df[col].abs().to_numpy()
            all_vals.append(vals)
            weekday_vals.append(vals[is_weekday.to_numpy()])

    if not all_vals:
        return None
    all_vals = np.concatenate(all_vals)
    weekday_vals = np.concatenate(weekday_vals)
    return {
        "mean_abs_corr_all": float(np.mean(all_vals)),
        "median_abs_corr_all": float(np.median(all_vals)),
        "mean_abs_corr_weekday": float(np.mean(weekday_vals)),
        "median_abs_corr_weekday": float(np.median(weekday_vals)),
    }


def run_one_window(
    market_lead: Market,
    market_lag: Market,
    param_ids: list[str],
    start_ym: tuple[int, int],
    window_months: int,
    bin_size_sec: int,
    output_dir: Path,
    label: str,
) -> dict | None:
    start_str, end_str = _month_window_dates(start_ym, window_months)
    dt_range = DtRange.from_strings(start_str, end_str)
    parquet_store = ParquetStore(sys_env.get_parquet_root_dir())

    LOG.info(f"[{label}] window {start_str} -> {end_str}")
    series_list = load_leadlag_series(parquet_store, market_lead, market_lag, param_ids, dt_range)
    if not series_list:
        LOG.warning(f"[{label}] no data, skipping.")
        return None

    scores, std_scores = aggregate_weekly(series_list, bin_size_sec)
    active_window = print_active_window(scores, bin_size_sec)
    stability = run_stability_check(series_list, bin_size_sec, active_window)
    raw_strength = raw_strength_stats(parquet_store, market_lead, market_lag, param_ids, dt_range)

    stem = (
        f"leadlag_curve_{market_lead.exchange_name.name}_{market_lead.symbol.name}"
        f"_{market_lag.exchange_name.name}_{market_lag.symbol.name}_{label}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    png_path = output_dir / f"{stem}.png"

    payload = {
        "bin_size_sec": bin_size_sec,
        "param_ids": param_ids,
        "market_lead": f"{market_lead.exchange_name.name}:{market_lead.symbol.name}",
        "market_lag": f"{market_lag.exchange_name.name}:{market_lag.symbol.name}",
        "trained_on": f"{start_str} to {end_str}",
        "scores": scores.tolist(),
        "std": std_scores.tolist(),
        "active_window": active_window,
        "stability": stability,
        "raw_strength": raw_strength,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    title = (
        f"LeadLagCurve — {market_lead.exchange_name.name}:{market_lead.symbol.name} -> "
        f"{market_lag.exchange_name.name}:{market_lag.symbol.name}\n{start_str} → {end_str}  (bin={bin_size_sec}s)"
    )
    plot_heatmap(scores, bin_size_sec, title, png_path, value_label="|lead-lag corr| (z-scored, 0-1)")

    chrono_corr = stability.get("chronological", {}).get("overall_corr")
    interleaved_corr = stability.get("interleaved", {}).get("overall_corr")
    ratio = (
        active_window["inside_mean"] / active_window["outside_mean"]
        if active_window and active_window.get("outside_mean")
        else None
    )
    return {
        "window": label,
        "start": start_str,
        "end": end_str,
        "chronological_corr": chrono_corr,
        "interleaved_corr": interleaved_corr,
        "active_inactive_ratio": ratio,
        "mean_abs_corr_weekday": raw_strength["mean_abs_corr_weekday"] if raw_strength else None,
        "json_path": str(json_path),
        "png_path": str(png_path),
    }


def plot_trend(results: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["window"] for r in results]
    chrono = [r["chronological_corr"] for r in results]
    interleaved = [r["interleaved_corr"] for r in results]
    ratio = [r["active_inactive_ratio"] for r in results]
    raw_strength = [r["mean_abs_corr_weekday"] for r in results]
    x = range(len(results))

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(x, chrono, marker="o", color="#e74c3c", label="chronological corr")
    ax1.plot(x, interleaved, marker="o", color="#3498db", label="interleaved corr")
    ax1.set_ylabel("stability correlation")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(x, ratio, marker="s", color="#2ecc71", label="active/inactive ratio")
    ax2.set_ylabel("active/inactive ratio")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title("Rolling 3-month LeadLagCurve: shape stability (top) vs raw strength (bottom)")

    ax3.plot(x, raw_strength, marker="d", color="#9b59b6", label="raw mean |ll_max_corr| (weekday, unnormalized)")
    ax3.set_ylabel("raw |correlation|")
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax3.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOG.info(f"Trend chart saved to {output_path}")


def main(
    market_lead: Market,
    market_lag: Market,
    configs: list[str],
    start_month: str,
    end_window_start_month: str,
    window_months: int,
    step_months: int,
    bin_size_sec: int,
    output_dir: Path,
) -> None:
    param_ids = [LEADLAG_CONFIGS[c].param_id() for c in configs]
    start_ym = _parse_ym(start_month)
    end_ym = _parse_ym(end_window_start_month)

    windows = []
    cur = start_ym
    while cur <= end_ym:
        windows.append(cur)
        cur = _add_months(cur, step_months)

    results = []
    for start in windows:
        window_end = _add_months(start, window_months - 1)
        label = f"{start[0]:04d}{start[1]:02d}-{window_end[0]:04d}{window_end[1]:02d}"
        r = run_one_window(market_lead, market_lag, param_ids, start, window_months, bin_size_sec, output_dir, label)
        if r is not None:
            results.append(r)

    print("\n── Rolling window summary ─────────────────────────────────────")
    print(
        f"  {'window':<14} {'chrono_corr':>12} {'interleaved_corr':>17} {'active/inactive':>16} "
        f"{'raw |corr| (weekday)':>21}"
    )
    for r in results:
        raw = r["mean_abs_corr_weekday"]
        raw_str = f"{raw:.4f}" if raw is not None else "n/a"
        print(
            f"  {r['window']:<14} {r['chronological_corr']:>12.4f} {r['interleaved_corr']:>17.4f} "
            f"{r['active_inactive_ratio']:>15.2f}x {raw_str:>21}"
        )
    print("────────────────────────────────────────────────────────────────\n")

    stem = f"leadlag_rolling_{market_lead.exchange_name.name}_{market_lead.symbol.name}_{market_lag.exchange_name.name}_{market_lag.symbol.name}"
    summary_path = output_dir / f"{stem}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary data → {summary_path}")

    trend_path = output_dir / f"{stem}_trend.png"
    plot_trend(results, trend_path)
    print(f"Trend chart → {trend_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling N-month LeadLagCurve stability/strength analysis.")
    parser.add_argument("--pair", help="Known pair shorthand, e.g. btc, eth, xrp, sol")
    parser.add_argument("--lead-market", help="e.g. BINANCE:ETH_USDT -- alternative to --pair")
    parser.add_argument("--lag-market", help="e.g. GMO:ETH_JPY -- alternative to --pair")
    parser.add_argument("--configs", default="mid", help="Comma-separated subset of {nearhft,mid} (default: mid)")
    parser.add_argument("--start-month", required=True, help="First window's start month, YYYY-MM")
    parser.add_argument("--end-window-start-month", required=True, help="Last window's start month, YYYY-MM")
    parser.add_argument("--window-months", type=int, default=3, help="Window length in calendar months (default: 3)")
    parser.add_argument("--step-months", type=int, default=1, help="Step between windows in months (default: 1)")
    parser.add_argument("--bin-size", type=int, default=300, help="Curve bin size in seconds (default: 300)")
    parser.add_argument("--output-dir", default="models", help="Output directory (default: models/)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.pair and (args.lead_market or args.lag_market):
        parser.error("--pair is mutually exclusive with --lead-market/--lag-market.")
    if args.pair:
        market_lead, market_lag = _resolve_pair(args.pair)
    elif args.lead_market and args.lag_market:
        market_lead = Market.from_str(args.lead_market)
        market_lag = Market.from_str(args.lag_market)
    else:
        parser.error("Pass either --pair, or both --lead-market and --lag-market.")

    env = Env.load_env(sys_env.get_env_root_dir(), ENV_NAME)
    env.visit()

    main(
        market_lead=market_lead,
        market_lag=market_lag,
        configs=args.configs.split(","),
        start_month=args.start_month,
        end_window_start_month=args.end_window_start_month,
        window_months=args.window_months,
        step_months=args.step_months,
        bin_size_sec=args.bin_size,
        output_dir=Path(args.output_dir),
    )
