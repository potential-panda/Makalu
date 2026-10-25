"""
Diagnostic tool: recompute the weekly LeadLagCurve straight from LeadLag
and dump it as data + a heatmap PNG for eye confirmation.

Unlike `cli.py populate LeadLagCurve` (the production path, which writes a
gold-layer snapshot per as-of date), this never touches gold parquet -- it's
meant to be run ad hoc while iterating on a market pair/date range, to sanity
check whether a real day-of-week pattern exists in lead-lag strength before
trusting the curve.

Shares its aggregation core (active_score.weekly_curve) with the production
pipeline (pipeline/curve_steps.py), and its presentation layer (heatmap,
weekday summary, active window, stability check) with
active_score/inspect_curve.py via active_score.curve_diagnostics, since none
of that is dataset-specific.

Only uses the two LeadLagConfig param sets actually populated in production
(NEARHFT + MID, see lead_lag/lead_lag.py) -- LEAD_LAG_CONFIG_FAR exists in
code but isn't wired into populate.py, so it's excluded here too.

Usage:
    python src/active_score/inspect_leadlag_curve.py \
        --pair eth --start 2026-04-20 --end 2026-06-20
"""

import argparse
import json
import logging
from pathlib import Path

from qate.core.model import Market
from qate.util.dt_range import DtRange
from qate_env.env import Env
from qate_env.env_name import EnvName

from active_score.curve_diagnostics import (
    plot_heatmap,
    print_active_window,
    print_weekday_summary,
    run_stability_check,
)
from active_score.weekly_curve import aggregate_weekly, load_leadlag_series
from lead_lag.lead_lag import LEAD_LAG_CONFIG_MID, LEAD_LAG_CONFIG_NEARHFT
from makalu import settings as sys_env
from makalu.storage.parquet_store import ParquetStore
from pipeline.market_pair import LEAD_LAG_PAIRS

LOG = logging.getLogger(__name__)
ENV_NAME = EnvName("MARIO_MARIO")

LEADLAG_CONFIGS = {"nearhft": LEAD_LAG_CONFIG_NEARHFT, "mid": LEAD_LAG_CONFIG_MID}


def _resolve_pair(pair_key: str) -> tuple[Market, Market]:
    pair = LEAD_LAG_PAIRS.get(pair_key)
    if pair is None:
        raise ValueError(f"Unknown pair '{pair_key}'. Known: {list(LEAD_LAG_PAIRS)}")
    return pair


def main(
    market_lead: Market,
    market_lag: Market,
    dt_range: DtRange,
    bin_size_sec: int,
    output_dir: Path,
    configs: list[str] = ("nearhft", "mid"),
    skip_stability_check: bool = False,
) -> None:
    parquet_store = ParquetStore(sys_env.get_parquet_root_dir())
    param_ids = [LEADLAG_CONFIGS[c].param_id() for c in configs]

    LOG.info(f"Recomputing LeadLagCurve from LeadLag for {market_lead} -> {market_lag}")
    LOG.info(f"  param_ids ({len(param_ids)}): {param_ids}")
    LOG.info(f"  dt_range: {dt_range.start_str} → {dt_range.end_str}")
    LOG.info(f"  bin_size_sec: {bin_size_sec}")

    series_list = load_leadlag_series(
        parquet_store, market_lead, market_lag, param_ids, dt_range
    )
    if not series_list:
        raise ValueError(
            "No valid LeadLag data found for any of the given param_ids -- nothing to plot."
        )
    LOG.info(f"Combining {len(series_list)} normalized |ll_max_corr| series.")

    scores, std_scores = aggregate_weekly(series_list, bin_size_sec)

    print_weekday_summary(scores)
    active_window = print_active_window(scores, bin_size_sec)

    stability = {}
    if not skip_stability_check:
        stability = run_stability_check(series_list, bin_size_sec, active_window)

    stem = (
        f"leadlag_curve_{market_lead.exchange_name.name}_{market_lead.symbol.name}"
        f"_{market_lag.exchange_name.name}_{market_lag.symbol.name}_{'-'.join(configs)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    png_path = output_dir / f"{stem}.png"

    payload = {
        "bin_size_sec": bin_size_sec,
        "param_ids": param_ids,
        "market_lead": f"{market_lead.exchange_name.name}:{market_lead.symbol.name}",
        "market_lag": f"{market_lag.exchange_name.name}:{market_lag.symbol.name}",
        "trained_on": f"{dt_range.start_str} to {dt_range.end_str}",
        "scores": scores.tolist(),
        "std": std_scores.tolist(),
        "active_window": active_window,
        "stability": stability,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    LOG.info(f"Curve data saved to {json_path}")

    title = (
        f"LeadLagCurve (recomputed) — {market_lead.exchange_name.name}:{market_lead.symbol.name} -> "
        f"{market_lag.exchange_name.name}:{market_lag.symbol.name}\n"
        f"{dt_range.start_str} → {dt_range.end_str}  (bin={bin_size_sec}s)"
    )
    plot_heatmap(
        scores,
        bin_size_sec,
        title,
        png_path,
        value_label="|lead-lag corr| (z-scored, 0-1)",
    )

    print(f"Done.\n  data → {json_path}\n  plot → {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recompute weekly LeadLagCurve from LeadLag and plot for eye confirmation."
    )
    parser.add_argument(
        "--pair",
        help="Known pair shorthand, e.g. btc, eth, xrp, sol (see pipeline/market_pair.py)",
    )
    parser.add_argument(
        "--lead-market", help="e.g. BINANCE:ETH_USDT -- alternative to --pair"
    )
    parser.add_argument(
        "--lag-market", help="e.g. GMO:ETH_JPY -- alternative to --pair"
    )
    parser.add_argument(
        "--start", required=True, help="Start date YYYYMMDD or YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", help="End date YYYYMMDD or YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=300,
        help="Curve bin size in seconds (default: 300)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Output directory for JSON + PNG (default: models/)",
    )
    parser.add_argument(
        "--configs",
        default="nearhft,mid",
        help="Comma-separated subset of {nearhft,mid} to include -- useful when one config isn't fully "
        "populated yet for the requested range (default: both)",
    )
    parser.add_argument(
        "--skip-stability-check",
        action="store_true",
        help="Skip the cross-period (chronological + interleaved) stability check for a faster run",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.pair and (args.lead_market or args.lag_market):
        parser.error("--pair is mutually exclusive with --lead-market/--lag-market.")
    if args.pair:
        market_lead, market_lag = _resolve_pair(args.pair)
    elif args.lead_market and args.lag_market:
        market_lead = Market.from_str(args.lead_market)
        market_lag = Market.from_str(args.lag_market)
    else:
        parser.error("Pass either --pair, or both --lead-market and --lag-market.")

    env = Env(sys_env.get_env_root_dir(), ENV_NAME)
    env.visit()

    main(
        market_lead=market_lead,
        market_lag=market_lag,
        dt_range=DtRange.from_strings(args.start, args.end),
        bin_size_sec=args.bin_size,
        output_dir=Path(args.output_dir),
        configs=args.configs.split(","),
        skip_stability_check=args.skip_stability_check,
    )
