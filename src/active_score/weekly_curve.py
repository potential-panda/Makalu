"""Shared (day_of_week x time-of-day bin) aggregation core for any
weekly-pattern dataset built from VolSpread (or similar per-param_id series).

Used by makalu/cli/populate.py's ActivityCurve (gold pipeline, volatility +
spread_pct_ema combined) and LeadLagCurve (correlation strength) commands, and
by active_score/inspect_curve.py (diagnostic tool that recomputes the same
curve straight from VolSpread for eyeballing, without writing gold parquet).
"""

import logging

import numpy as np
import pandas as pd
from qate.core.model import Market
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

LOG = logging.getLogger(__name__)

ALIGN_INTERVAL = "1min"


def zscore_normalize(series: pd.Series) -> pd.Series | None:
    """Resample to 1-min, drop NaN, z-score normalize -- None if there's no
    variance left to normalize against (flat or empty series)."""
    series = series.resample(ALIGN_INTERVAL).mean().dropna()
    if series.empty or series.std() == 0:
        return None
    return (series - series.mean()) / series.std()


def load_volspread_series(
    parquet_store: ParquetStore,
    market: Market,
    execution_size_str: str,
    param_ids: list[str],
    dt_range: DtRange,
) -> list[pd.Series]:
    """Load volatility_ema and spread_pct_ema for each param_id, z-score
    normalize each column independently (so configs and columns of different
    scale contribute equally), and return the list of normalized series ready
    for aggregate_weekly. Columns/param_ids with no data or no variance are
    skipped with a warning.
    """
    series_list = []
    for param_id in param_ids:
        store = parquet_store.create_vol_spread(market, execution_size_str, param_id)
        df = store.read(dt_range)
        if df.empty:
            LOG.warning(f"No VolSpread data for param_id={param_id}, skipping.")
            continue
        index = pd.to_datetime(df["ts"], unit="s", utc=True)
        for col in ("volatility_ema", "spread_pct_ema"):
            if col not in df.columns:
                continue
            z = zscore_normalize(pd.Series(df[col].values, index=index, name=f"{param_id}:{col}"))
            if z is not None:
                series_list.append(z)
            else:
                LOG.warning(f"param_id={param_id} column={col} has no variance after resampling, skipping.")
    return series_list


def load_leadlag_series(
    parquet_store: ParquetStore,
    market_lead: Market,
    market_lag: Market,
    param_ids: list[str],
    dt_range: DtRange,
) -> list[pd.Series]:
    """Load every ll_max_corr* column for each param_id, take abs() (a strong
    relationship is informative regardless of sign -- who leads whom isn't
    what "is this a good time to trade the relationship" depends on),
    z-score normalize each column independently, and return the list of
    normalized series ready for aggregate_weekly. The set of ll_max_corr*
    suffixes is discovered from the data rather than hardcoded, since it
    depends on which LeadLagConfig.windows_sec/decay_spans_sec were used.
    """
    series_list = []
    for param_id in param_ids:
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
            z = zscore_normalize(pd.Series(df[col].abs().values, index=index, name=f"{param_id}:{col}"))
            if z is not None:
                series_list.append(z)
            else:
                LOG.warning(f"param_id={param_id} column={col} has no variance after resampling, skipping.")
    return series_list


def aggregate_weekly(series_list: list[pd.Series], bin_size_sec: int) -> tuple[np.ndarray, np.ndarray]:
    """Combine already-normalized series into one (mean, std) pair of
    [7, n_bins] arrays, binned by (day_of_week, time_of_day // bin_size_sec).
    mean is min-max normalized to [0, 1]; std is scaled by the same factor.
    """
    if 86400 % bin_size_sec != 0:
        raise ValueError(f"bin_size_sec={bin_size_sec} must evenly divide 86400")
    if not series_list:
        raise ValueError("series_list is empty -- nothing to aggregate")

    combined = pd.concat(series_list)
    dow = combined.index.dayofweek
    seconds = combined.index.hour * 3600 + combined.index.minute * 60 + combined.index.second
    bin_idx = seconds // bin_size_sec

    n_bins = 86400 // bin_size_sec
    sums = np.zeros((7, n_bins), dtype=np.float64)
    sum_sq = np.zeros((7, n_bins), dtype=np.float64)
    counts = np.zeros((7, n_bins), dtype=np.int64)

    dow_arr = dow.to_numpy()
    bin_arr = bin_idx.to_numpy()
    val_arr = combined.to_numpy(dtype=np.float64)

    valid = np.isfinite(val_arr)
    np.add.at(sums, (dow_arr[valid], bin_arr[valid]), val_arr[valid])
    np.add.at(sum_sq, (dow_arr[valid], bin_arr[valid]), val_arr[valid] ** 2)
    np.add.at(counts, (dow_arr[valid], bin_arr[valid]), 1)

    mean_curve = np.where(counts > 0, sums / counts, np.nan)
    variance = np.where(counts > 0, sum_sq / counts - (sums / counts) ** 2, np.nan)
    std_curve = np.sqrt(np.maximum(variance, 0.0))

    col_means = np.nanmean(mean_curve, axis=0)
    col_stds = np.nanmean(std_curve, axis=0)
    for d in range(7):
        for b in range(n_bins):
            if np.isnan(mean_curve[d, b]):
                mean_curve[d, b] = col_means[b]
                std_curve[d, b] = col_stds[b]

    v_min, v_max = mean_curve.min(), mean_curve.max()
    if v_max > v_min:
        scale = v_max - v_min
        scores = (mean_curve - v_min) / scale
        std_scores = std_curve / scale
    else:
        scores = np.zeros_like(mean_curve)
        std_scores = np.zeros_like(std_curve)

    return scores, std_scores
