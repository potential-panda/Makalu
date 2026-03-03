"""
Offline curve builder for ActiveScore.

Loads historical volatility_ema data from VolSpreadStore across multiple
param_ids, aligns them to 1-min resolution, z-score normalizes each series
independently (to remove scale differences between short/long lookback configs),
then averages to produce a single weekly activity curve.

The resulting curve is min-max normalized to [0, 1] and saved as JSON.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from qate.core.model import Market
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

LOG = logging.getLogger(__name__)

ALIGN_INTERVAL = "1min"   # all param series are resampled to this before combining
DEFAULT_BIN_SIZE_SEC = 300  # 5-min bins → 288/day × 7 = 2016 slots


def _load_and_normalize(
    parquet_store: ParquetStore,
    market: Market,
    execution_size_str: str,
    param_id: str,
    dt_range: DtRange,
) -> pd.Series | None:
    """
    Load volatility_ema for one param_id, resample to 1-min, z-score normalize.
    Returns a Series indexed by UTC DatetimeIndex, or None if no data found.
    """
    store = parquet_store.create_vol_spread(market, execution_size_str, param_id)
    df = store.read(dt_range)

    if df.empty:
        LOG.warning(f"No data for param_id={param_id}, skipping.")
        return None

    # Build a DatetimeIndex from Unix-second timestamps (UTC)
    index = pd.to_datetime(df["ts"], unit="s", utc=True)
    series = pd.Series(df["volatility_ema"].values, index=index, name=param_id)

    # Resample to 1-min for alignment across all configs
    series = series.resample(ALIGN_INTERVAL).mean()

    # Drop NaN and flat series (can't normalize)
    series = series.dropna()
    if series.empty or series.std() == 0:
        LOG.warning(f"param_id={param_id} has no variance after resampling, skipping.")
        return None

    # Z-score normalize so all configs contribute equally regardless of scale
    series = (series - series.mean()) / series.std()

    LOG.info(f"Loaded param_id={param_id}: {len(series)} 1-min rows after resampling.")
    return series


def build_curve(
    parquet_store: ParquetStore,
    market: Market,
    execution_size_str: str,
    param_ids: list[str],
    dt_range: DtRange,
    output_path: str | Path,
    bin_size_sec: int = DEFAULT_BIN_SIZE_SEC,
) -> None:
    """
    Build and save a weekly activity score curve from historical volatility data.

    Args:
        parquet_store: ParquetStore instance pointing at the data root.
        market: The market whose VolSpread data to use.
        execution_size_str: Execution size string (partition key in VolSpreadStore).
        param_ids: List of VolSpreadConfig param_ids to combine (e.g. all 15 configs).
        dt_range: Date range of historical data to train on.
        output_path: Where to write the resulting JSON curve file.
        bin_size_sec: Output curve time bin width in seconds (e.g. 60, 300, 900).
    """
    if 86400 % bin_size_sec != 0:
        raise ValueError(f"bin_size_sec={bin_size_sec} must evenly divide 86400")

    # Load and normalize each param_id series
    series_list = []
    used_param_ids = []
    for param_id in param_ids:
        s = _load_and_normalize(parquet_store, market, execution_size_str, param_id, dt_range)
        if s is not None:
            series_list.append(s)
            used_param_ids.append(param_id)

    if not series_list:
        raise ValueError("No valid VolSpread data found for any of the given param_ids.")

    LOG.info(f"Combining {len(series_list)} param series into weekly curve.")

    # Stack all normalized series into one long series, then bin by (dow, bin_idx)
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
    # variance = E[x²] - E[x]²; guard against floating-point negatives
    variance = np.where(counts > 0, sum_sq / counts - (sums / counts) ** 2, np.nan)
    std_curve = np.sqrt(np.maximum(variance, 0.0))

    # Fill any empty bins (e.g. no weekend data in some configs) with column mean
    col_means = np.nanmean(mean_curve, axis=0)
    col_stds = np.nanmean(std_curve, axis=0)
    for d in range(7):
        for b in range(n_bins):
            if np.isnan(mean_curve[d, b]):
                mean_curve[d, b] = col_means[b]
                std_curve[d, b] = col_stds[b]

    # Min-max normalize mean and std by the same scale factor
    v_min, v_max = mean_curve.min(), mean_curve.max()
    if v_max > v_min:
        scale = v_max - v_min
        scores = (mean_curve - v_min) / scale
        std_scores = std_curve / scale
    else:
        scores = np.zeros_like(mean_curve)
        std_scores = np.zeros_like(std_curve)

    payload = {
        "bin_size_sec": bin_size_sec,
        "param_ids": used_param_ids,
        "market": f"{market.exchange_name.name}:{market.symbol.name}",
        "trained_on": f"{dt_range.start_str} to {dt_range.end_str}",
        "scores": scores.tolist(),      # shape: [7][n_bins], mean normalized to [0, 1]
        "std": std_scores.tolist(),     # shape: [7][n_bins], std in same normalized scale
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    LOG.info(
        f"Saved curve to {output_path} — "
        f"{len(used_param_ids)} param_ids, {n_bins} bins/day, "
        f"score range [{scores.min():.3f}, {scores.max():.3f}], "
        f"std range [{std_scores.min():.3f}, {std_scores.max():.3f}]"
    )
