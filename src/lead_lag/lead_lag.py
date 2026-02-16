"""
Lead-lag feature computation.

Prepares aligned return series from two price DataFrames and computes
EWLL features over sliding windows. Pipeline-independent — operates
on raw DataFrames and returns results as a list of (timestamp, fields) tuples.

Reference: doc/lead_lag_spec.md
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Generator

import numpy as np
import pandas as pd

from .ewll import compute_lead_lag_features

LOG = logging.getLogger(__name__)


@dataclass
class LeadLagConfig:
    search_lags_ms: range
    decay_spans_sec: list[int]
    # Sub-window lengths (in seconds) for multi-scale EWLL feature computation.
    # Each value produces a separate set of features (e.g. 5s, 15s, 30s, 60s)
    # using the tail of the sliding window.
    windows_sec: list[int]
    # Size of the outer sliding window (in seconds) that moves over the full
    # return series. Must be >= max(windows_sec). Determines how much history
    # is available for sub-window extraction at each position.
    window_size_sec: int
    # Stride (in seconds) between consecutive sliding window positions.
    # Controls output frequency — one row of features is emitted per stride.
    window_step_sec: int
    resample_interval: str = "100ms"

    def __post_init__(self):
        # Keep here for memo, the caller must provide the values
        # if self.search_lags_ms is None:
        #     self.search_lags_ms = range(0, 5100, 100)
        # if self.decay_spans_sec is None:
        #     self.decay_spans_sec = [10, 30, 60]
        # if self.windows_sec is None:
        #     self.windows_sec = [5, 15, 30, 60]

        self.resample_ms = pd.Timedelta(self.resample_interval).total_seconds() * 1000
        self.lag_steps = np.array(
            [int(lag_ms / self.resample_ms) for lag_ms in self.search_lags_ms],
            dtype=np.int64,
        )
        self.max_lag_ms = max(self.search_lags_ms)

    def param_id(self) -> str:
        lags_str = f"{self.search_lags_ms.start}_{self.search_lags_ms.stop}_{self.search_lags_ms.step}"
        spans_str = "_".join(str(s) for s in self.decay_spans_sec)
        windows_str = "_".join(str(w) for w in self.windows_sec)
        return f"l{lags_str}_s{spans_str}_w{windows_str}_wsz{self.window_size_sec}_wst{self.window_step_sec}"


# target to hold positions for 20s-30s
LEAD_LAG_CONFIG_NEARHFT = LeadLagConfig(
    resample_interval="100ms",
    search_lags_ms=range(0, 5100, 100),
    decay_spans_sec=[5, 15, 30],
    windows_sec=[10, 20, 30, 60],
    window_size_sec=600,
    window_step_sec=1,
)

# target to hold positions for 120s-180s
LEAD_LAG_CONFIG_MID = LeadLagConfig(
    search_lags_ms=range(0, 5100, 100),
    decay_spans_sec=[30, 60, 120],
    windows_sec=[60, 120, 180, 300],
    window_size_sec=3600,
    window_step_sec=30,
)


def prepare_returns(
    df_lead: pd.DataFrame,
    df_lag: pd.DataFrame,
    resample_interval: str,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray] | None:
    """Resample, align, and compute log-returns from two execution-price DataFrames.

    Args:
        df_lead: Leader market execution prices (must have 'ts' and 'mid' or 'bid'/'ask' columns).
        df_lag: Target/lag market execution prices.
        resample_interval: Pandas-compatible resample string (e.g. '100ms').

    Returns:
        (timestamps, lead_returns, lag_returns) or None if insufficient data.
    """
    if df_lead.empty or df_lag.empty:
        LOG.warning("prepare_returns: empty input DataFrame(s)")
        return None

    # Convert ts → DatetimeIndex, deduplicate, resample, ffill
    df_lead = df_lead.copy()
    df_lag = df_lag.copy()
    df_lead.index = pd.to_datetime(df_lead["ts"], unit="s")
    df_lag.index = pd.to_datetime(df_lag["ts"], unit="s")
    df_lead = df_lead[~df_lead.index.duplicated(keep="last")]
    df_lag = df_lag[~df_lag.index.duplicated(keep="last")]

    df_lead = df_lead.resample(resample_interval).last().ffill().dropna()
    df_lag = df_lag.resample(resample_interval).last().ffill().dropna()

    LOG.info(f"prepare_returns: after resampling: {len(df_lead)} lead, {len(df_lag)} lag")

    # Intersect time range
    start_ts = max(df_lead.index[0], df_lag.index[0])
    end_ts = min(df_lead.index[-1], df_lag.index[-1])
    if start_ts >= end_ts:
        LOG.warning("prepare_returns: no overlapping timestamps")
        return None

    df_lead = df_lead.loc[start_ts:end_ts]

    # Extract mid prices
    def _mid(df):
        if "mid" in df.columns:
            return df["mid"]
        return (df["bid"] + df["ask"]) / 2

    lead_mid = _mid(df_lead).values
    lag_mid_aligned = _mid(df_lag).reindex(df_lead.index, method="ffill").values

    # Remove NaN
    valid = ~(np.isnan(lead_mid) | np.isnan(lag_mid_aligned))
    timestamps = df_lead.index[valid]
    lead_prices = lead_mid[valid]
    lag_prices = lag_mid_aligned[valid]

    if len(lead_prices) < 2:
        LOG.warning("prepare_returns: not enough aligned data points")
        return None

    # Log-returns
    lead_returns = np.log(lead_prices[1:] / lead_prices[:-1])
    lag_returns = np.log(lag_prices[1:] / lag_prices[:-1])
    ret_timestamps = timestamps[1:]

    LOG.info(f"prepare_returns: aligned returns length: {len(lead_returns)}")
    return ret_timestamps, lead_returns, lag_returns


def compute_sliding_windows(
    ret_timestamps: pd.DatetimeIndex,
    lead_returns: np.ndarray,
    lag_returns: np.ndarray,
    config: LeadLagConfig,
    populate_start_ts: float | None = None,
    populate_end_ts: float | None = None,
) -> Generator[tuple[float, dict], None, None]:
    """Slide windows over aligned returns and yield (unix_ts, feature_fields) per window.

    Args:
        ret_timestamps: DatetimeIndex aligned with return arrays.
        lead_returns: Log-returns of leader market.
        lag_returns: Log-returns of target/lag market.
        config: LeadLagConfig with all parameter settings.
        populate_start_ts: If set, skip windows ending before this unix timestamp.
        populate_end_ts: If set, skip windows ending after this unix timestamp.

    Yields:
        (window_end_unix_ts, fields_dict) for each valid window position.
    """
    resample_sec = config.resample_ms / 1000.0
    window_steps = int(config.window_size_sec / resample_sec)
    step_steps = int(config.window_step_sec / resample_sec)

    # Track opt_lag history per (window_sec, span) for stability/drift.
    # Capped at 100 entries to prevent memory growth.
    opt_lag_history: dict[tuple[int, int], deque[float]] = {}

    start_idx = 0
    while start_idx + window_steps <= len(lead_returns):
        end_idx = start_idx + window_steps
        window_end_ts = ret_timestamps[end_idx - 1].timestamp()

        # Filter to populate range
        if populate_start_ts is not None and window_end_ts < populate_start_ts:
            start_idx += step_steps
            continue
        if populate_end_ts is not None and window_end_ts > populate_end_ts:
            start_idx += step_steps
            continue

        fields = {}

        for window_sec in config.windows_sec:
            w_steps = int(window_sec / resample_sec)
            w_start = max(start_idx, end_idx - w_steps)
            if end_idx - w_start < 2:
                continue

            w_lead = lead_returns[w_start:end_idx]
            w_lag = lag_returns[w_start:end_idx]

            # Spec §7: >50% missing → skip
            if len(w_lead) < w_steps * 0.5:
                continue

            for span_sec in config.decay_spans_sec:
                span_steps = span_sec / resample_sec
                alpha = 2.0 / (span_steps + 1.0)

                result = compute_lead_lag_features(w_lag, w_lead, config.lag_steps, alpha)

                suffix = f"_{window_sec}s_{span_sec}s"
                max_corr = result["ll_max_corr"]
                opt_lag_idx = result["ll_opt_lag"]
                opt_lag_ms = float(opt_lag_idx * config.resample_ms)
                corr_at_zero = result["ll_corr_at_zero"]

                # Alpha gain: How much better is the optimal lag vs zero lag?
                # Clipped at 100.0 to prevent division-by-zero noise from dominating ML training.
                if abs(corr_at_zero) > 1e-6:
                    alpha_gain = float(np.clip(abs(max_corr) / abs(corr_at_zero), 1.0, 100.0))
                else:
                    # If corr_at_zero is basically zero, but max_corr exists, it's high gain
                    alpha_gain = 100.0 if abs(max_corr) > 1e-6 else 1.0

                # Track history for stability / drift
                hist_key = (window_sec, span_sec)
                if hist_key not in opt_lag_history:
                    opt_lag_history[hist_key] = deque(maxlen=100)
                opt_lag_history[hist_key].append(opt_lag_ms)
                history = opt_lag_history[hist_key]

                # Stability: rolling std (10-period)
                if len(history) >= 2:
                    # slice of deque is a list
                    recent_hist = list(history)[-10:]
                    stability = float(np.std(recent_hist))
                else:
                    stability = 0.0

                # Drift: delta from previous or recent history
                drift_lookback = max(1, int(5.0 / (config.window_step_sec or 1)))
                if len(history) > drift_lookback:
                    drift = opt_lag_ms - history[-drift_lookback - 1]
                else:
                    drift = 0.0

                # Normalization (spec §6)
                norm_lag = opt_lag_ms / config.max_lag_ms if config.max_lag_ms > 0 else 0.0
                norm_stability = float(np.log1p(stability))
                # Keep units consistent with ll_opt_lag_ms (normalized [0, 1]).
                norm_drift = drift / config.max_lag_ms if config.max_lag_ms > 0 else 0.0

                fields[f"ll_corr_at_zero{suffix}"] = float(np.clip(corr_at_zero, -1.0, 1.0))
                fields[f"ll_max_corr{suffix}"] = float(np.clip(max_corr, -1.0, 1.0))
                fields[f"ll_opt_lag_ms{suffix}"] = norm_lag
                fields[f"ll_alpha_gain{suffix}"] = alpha_gain
                fields[f"ll_alpha_gain_log{suffix}"] = float(np.log1p(alpha_gain))
                fields[f"ll_lag_stability{suffix}"] = norm_stability
                fields[f"ll_drift{suffix}"] = float(np.clip(norm_drift, -1.0, 1.0))

        if fields:
            yield window_end_ts, fields

        start_idx += step_steps
