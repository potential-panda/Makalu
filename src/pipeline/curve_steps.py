"""Builds ActivityCurve and LeadLagCurve rows: a (day_of_week x time-of-day
bin) mean+std curve over a trailing training window, read from already-
populated VolSpread/LeadLag gold data -- never recomputed from bronze.

ActivityCurve combines BOTH of VolSpread's columns (volatility_ema and
spread_pct_ema), via active_score/weekly_curve.py's load_volspread_series --
shared with active_score/inspect_curve.py, the diagnostic tool that recomputes
the same curve for eyeballing without writing gold parquet. LeadLagCurve is
the lead-lag-strength equivalent, built from LeadLag's ll_max_corr* columns
(whichever window/span suffixes are actually present for the populated
config, discovered from the data rather than hardcoded, since the suffix set
depends on which LeadLagConfig was populated).
"""

import logging

import pandas as pd
from qate.core.model import Market
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

from active_score.weekly_curve import aggregate_weekly, load_volspread_series, zscore_normalize

LOG = logging.getLogger(__name__)


def _curve_rows(scores, std_scores, mean_col: str, std_col: str) -> list[dict]:
    n_bins = scores.shape[1]
    rows = []
    for d in range(7):
        for b in range(n_bins):
            rows.append({
                "day_of_week": d,
                "bucket_index": b,
                mean_col: float(scores[d, b]),
                std_col: float(std_scores[d, b]),
            })
    return rows


def build_activity_curve_rows(
    parquet_store: ParquetStore,
    market: Market,
    execution_size_str: str,
    vol_param_ids: list[str],
    dt_range: DtRange,
    bin_size_sec: int = 300,
) -> list[dict]:
    series_list = load_volspread_series(parquet_store, market, execution_size_str, vol_param_ids, dt_range)

    if not series_list:
        raise ValueError("No valid VolSpread data found for any of the given param_ids.")
    LOG.info(f"Combining {len(series_list)} normalized (volatility_ema/spread) series into weekly ActivityCurve.")

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
    series_list = []
    for param_id in leadlag_param_ids:
        store = parquet_store.create_lead_lag(market_lead, market_lag, param_id)
        df = store.read(dt_range)
        if df.empty:
            LOG.warning(f"No LeadLag data for param_id={param_id}, skipping.")
            continue
        index = pd.to_datetime(df["ts"], unit="s", utc=True)
        corr_cols = [c for c in df.columns if c.startswith("ll_max_corr")]
        if not corr_cols:
            LOG.warning(f"param_id={param_id} has no ll_max_corr* columns, skipping.")
            continue
        for col in corr_cols:
            # abs(): a strong relationship is informative regardless of sign;
            # the sign itself (who leads whom) isn't what "is this a good
            # time to trade the relationship" should depend on.
            z = zscore_normalize(pd.Series(df[col].abs().values, index=index, name=f"{param_id}:{col}"))
            if z is not None:
                series_list.append(z)
            else:
                LOG.warning(f"param_id={param_id} column={col} has no variance after resampling, skipping.")

    if not series_list:
        raise ValueError("No valid LeadLag data found for any of the given param_ids.")
    LOG.info(f"Combining {len(series_list)} normalized correlation series into weekly LeadLagCurve.")

    scores, std_scores = aggregate_weekly(series_list, bin_size_sec)
    return _curve_rows(scores, std_scores, "corr_mean", "corr_std")
