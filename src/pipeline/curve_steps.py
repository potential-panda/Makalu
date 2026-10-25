"""Builds ActivityCurve and LeadLagCurve rows: a (day_of_week x time-of-day
bin) mean+std curve over a trailing training window, read from already-
populated Volatility/Spread/LeadLag gold data -- never recomputed from bronze.

ActivityCurve combines Hase's Volatility (vol_bps_ema) and Spread
(spread_bps_ema), paired per window in makalu/activity_windows.py, via
active_score/weekly_curve.py's load_activity_series --
shared with active_score/inspect_curve.py, the diagnostic tool that recomputes
the same curve for eyeballing without writing gold parquet. LeadLagCurve is
the lead-lag-strength equivalent, built from LeadLag's ll_max_corr* columns
(whichever window/span suffixes are actually present for the populated
config, discovered from the data rather than hardcoded, since the suffix set
depends on which LeadLagConfig was populated).
"""

import logging

from qate.core.model import Market
from qate.util.dt_range import DtRange

from active_score.weekly_curve import (
    aggregate_weekly,
    load_activity_series,
    load_leadlag_series,
)
from makalu.storage.parquet_store import ParquetStore

LOG = logging.getLogger(__name__)


def _curve_rows(scores, std_scores, mean_col: str, std_col: str) -> list[dict]:
    n_bins = scores.shape[1]
    rows = []
    for d in range(7):
        for b in range(n_bins):
            rows.append(
                {
                    "day_of_week": d,
                    "bucket_index": b,
                    mean_col: float(scores[d, b]),
                    std_col: float(std_scores[d, b]),
                }
            )
    return rows


def build_activity_curve_rows(
    parquet_store: ParquetStore,
    market: Market,
    execution_size_str: str,
    series: list[tuple[str, str, str]],
    dt_range: DtRange,
    bin_size_sec: int = 300,
) -> list[dict]:
    series_list = load_activity_series(
        parquet_store, market, execution_size_str, series, dt_range
    )

    if not series_list:
        raise ValueError("No valid Volatility or Spread data found for the windows.")
    LOG.info(
        f"Combining {len(series_list)} normalized (volatility/spread) series into weekly ActivityCurve."
    )

    scores, std_scores = aggregate_weekly(series_list, bin_size_sec)
    return _curve_rows(scores, std_scores, "score_mean", "score_std")


def build_leadlag_curve_rows(
    parquet_store: ParquetStore,
    market_lead: Market,
    market_lag: Market,
    leadlag_param_ids: list[str],
    dt_range: DtRange,
    bin_size_sec: int = 300,
) -> list[dict]:
    series_list = load_leadlag_series(
        parquet_store, market_lead, market_lag, leadlag_param_ids, dt_range
    )

    if not series_list:
        raise ValueError("No valid LeadLag data found for any of the given param_ids.")
    LOG.info(
        f"Combining {len(series_list)} normalized correlation series into weekly LeadLagCurve."
    )

    scores, std_scores = aggregate_weekly(series_list, bin_size_sec)
    return _curve_rows(scores, std_scores, "corr_mean", "corr_std")
