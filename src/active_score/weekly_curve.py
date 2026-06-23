"""Shared (day_of_week x time-of-day bin) aggregation core, extracted from
active_score/curve_builder.py's build_curve so it can be reused for any
weekly-pattern dataset, not just the original VolSpread-only ActiveScore.

Used by makalu/cli/populate.py's ActivityCurve (volatility + spread_pct_ema
combined, fixing the original curve_builder's volatility-only gap) and
LeadLagCurve (correlation strength) commands.
"""

import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

ALIGN_INTERVAL = "1min"


def zscore_normalize(series: pd.Series) -> pd.Series | None:
    """Resample to 1-min, drop NaN, z-score normalize -- None if there's no
    variance left to normalize against (flat or empty series)."""
    series = series.resample(ALIGN_INTERVAL).mean().dropna()
    if series.empty or series.std() == 0:
        return None
    return (series - series.mean()) / series.std()


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
