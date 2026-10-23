"""
Diagnostic tool: recompute the weekly ActivityCurve straight from VolSpread
and dump it as data + a heatmap PNG for eye confirmation.

Unlike `cli.py populate ActivityCurve` (the production path, which writes a
gold-layer snapshot per as-of date), this never touches gold parquet -- it's
meant to be run ad hoc while iterating on a market/date range, to sanity
check whether a real day-of-week pattern exists before trusting the curve.

Shares its aggregation core (active_score.weekly_curve) with the production
pipeline (pipeline/curve_steps.py), so both compute the curve the same way.
Shares its presentation layer (heatmap, weekday summary, active window,
stability check) with active_score/inspect_leadlag_curve.py via
active_score.curve_diagnostics, since none of that is VolSpread-specific.

Also derives a simple weekday active/inactive UTC+JST time window from the
curve -- see doc/vol_spread.md's "Active/Inactive Window Simplification" for
why a coarse contiguous block is preferred over trusting raw bin values.

Runs a cross-period stability check by default: the already-loaded series are
split two ways (chronological first-half/second-half, and interleaved
odd/even ISO week) and each half's curve is compared to the other, both
overall and per weekday. This is what "Active/Inactive Window Simplification"
in doc/vol_spread.md means by "holds up across independent halves of the
training data" -- no extra VolSpread reads, just re-aggregating the same
in-memory series. Skip with --skip-stability-check for a faster run.

Usage:
    python src/active_score/inspect_curve.py \
        --exchange GMO --symbol ETH_JPY --execution-size 0.1 \
        --start 2025-07-01 --end 2026-06-20
"""

import argparse
import json
import logging
from pathlib import Path

from qate.core.model import Market
from makalu import settings as sys_env
from qate_env.env import Env
from qate_env.env_name import EnvName
from makalu.storage.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

from active_score.curve_diagnostics import (
    plot_heatmap,
    print_active_window,
    print_weekday_summary,
    run_stability_check,
)
from active_score.weekly_curve import aggregate_weekly, load_volspread_series
from feature_engineering.volatility import VolSpreadConfig

LOG = logging.getLogger(__name__)
ENV_NAME = EnvName("MARIO_MARIO")

# Kept in sync with populate.py's _volspread_configs() / silver.py's own
# VolSpreadStep list -- the 4 configs actually populated for VolSpread.
VOL_CONFIGS = [
    VolSpreadConfig(60, 300, 120),
    VolSpreadConfig(60, 900, 300),
    VolSpreadConfig(180, 900, 300),
    VolSpreadConfig(300, 900, 300),
]


def main(
    market: Market,
    execution_size: str,
    dt_range: DtRange,
    bin_size_sec: int,
    output_dir: Path,
    skip_stability_check: bool = False,
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

    print_weekday_summary(scores)
    active_window = print_active_window(scores, bin_size_sec)

    stability = {}
    if not skip_stability_check:
        stability = run_stability_check(series_list, bin_size_sec, active_window)

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
        "stability": stability,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    LOG.info(f"Curve data saved to {json_path}")

    title = f"ActivityCurve (recomputed) — {market.exchange_name.name}:{market.symbol.name} es={execution_size}\n{dt_range.start_str} → {dt_range.end_str}  (bin={bin_size_sec}s)"
    plot_heatmap(scores, bin_size_sec, title, png_path, value_label="Activity score (0-1)")

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
    parser.add_argument(
        "--skip-stability-check",
        action="store_true",
        help="Skip the cross-period (chronological + interleaved) stability check for a faster run",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    env = Env(sys_env.get_env_root_dir(), ENV_NAME)
    env.visit()

    main(
        market=Market.from_str(f"{args.exchange}:{args.symbol}"),
        execution_size=args.execution_size,
        dt_range=DtRange.from_strings(args.start, args.end),
        bin_size_sec=args.bin_size,
        output_dir=Path(args.output_dir),
        skip_stability_check=args.skip_stability_check,
    )
