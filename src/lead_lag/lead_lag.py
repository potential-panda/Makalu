"""
Lead-lag analysis steps for computing cross-correlation between two markets.

This module provides steps to analyze the lead-lag relationship between
price movements on different exchanges/markets.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


@dataclass
class LeadLagResult:
    """Container for lead-lag analysis results."""

    best_lag: Optional[int]
    best_corr: float
    corr_by_lag: Dict[int, float]
    lag_seconds: float
    leader_market: Optional[str]
    follower_market: Optional[str]
    aligned_length: int
    interval_ms: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_lag": self.best_lag,
            "best_corr": self.best_corr,
            "lag_seconds": self.lag_seconds,
            "leader_market": self.leader_market,
            "follower_market": self.follower_market,
            "aligned_length": self.aligned_length,
            "interval_ms": self.interval_ms,
        }


def compute_returns(prices: np.ndarray, step: int = 1) -> np.ndarray:
    """Compute log returns over given step size."""
    if len(prices) <= step:
        return np.array([])
    return np.log(prices[step:] / prices[:-step])


def align_series(
    df_a: pd.DataFrame, df_b: pd.DataFrame, price_col: str = "mid"
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    if df_a.empty or df_b.empty:
        return pd.DatetimeIndex([]), np.array([]), np.array([])

    # Find overlapping time range
    start_ts = max(df_a.index[0], df_b.index[0])
    end_ts = min(df_a.index[-1], df_b.index[-1])

    if start_ts >= end_ts:
        return pd.DatetimeIndex([]), np.array([]), np.array([])

    # Filter market A to overlapping range
    mask_a = (df_a.index >= start_ts) & (df_a.index <= end_ts)
    df_a_overlap = df_a.loc[mask_a]

    # Use market A's timestamps as reference, forward-fill market B
    # Reindex B to A's timestamps with forward-fill
    df_b_aligned = df_b[price_col].reindex(df_a_overlap.index, method="ffill")

    # Get aligned arrays
    timestamps = df_a_overlap.index
    aligned_a = df_a_overlap[price_col].values
    aligned_b = df_b_aligned.values

    # Remove any NaN values
    valid_mask = ~(np.isnan(aligned_a) | np.isnan(aligned_b))
    return timestamps[valid_mask], aligned_a[valid_mask], aligned_b[valid_mask]


def compute_cross_correlation(
    series_a: np.ndarray, series_b: np.ndarray, max_lag: int
) -> Tuple[Optional[int], float, Dict[int, float]]:
    """
    Compute cross-correlation at various lags.

    Args:
        series_a: Price series for market A
        series_b: Price series for market B
        max_lag: Maximum lag to test in steps

    Returns:
        (best_lag, best_corr, corr_by_lag)
        - best_lag: Lag that maximizes correlation (negative = A leads B)
        - best_corr: Maximum correlation value
        - corr_by_lag: Dict mapping lag -> correlation
    """
    # Compute returns
    returns_a = compute_returns(series_a)
    returns_b = compute_returns(series_b)

    if len(returns_a) == 0 or len(returns_b) == 0:
        return None, 0.0, {}

    # Convert to directional changes
    dir_a = np.sign(returns_a)
    dir_b = np.sign(returns_b)

    results = {}
    best_lag = None
    best_corr = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # A leads B: align A[0:] with B[-lag:]
            x = dir_a[:lag]
            y = dir_b[-lag:]
        elif lag > 0:
            # B leads A: align A[lag:] with B[0:]
            x = dir_a[lag:]
            y = dir_b[:-lag]
        else:
            x = dir_a
            y = dir_b

        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            continue

        corr = np.corrcoef(x, y)[0, 1]
        results[lag] = corr

        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag, best_corr, results


def resample_and_forward_fill(df: pd.DataFrame, interval: str, ts_col: str) -> pd.DataFrame:
    df.index = pd.to_datetime(df[ts_col], unit="s")
    # Drop duplicate timestamps, keeping the last value (most recent state)
    df = df[~df.index.duplicated(keep="last")]
    # Resample to fixed intervals using last value, then forward-fill gaps
    df = df.resample(interval).last().ffill().dropna()
    return df


class LeadLag:
    def __init__(
        self,
        window_size_sec: int,
        window_step_sec: int,
        # NOTE Below 3 args are based on data frequency of the actual market,
        # the default values are good for the markets we are trading
        max_lag: int = 3,
        interval_ms: int = 500,
        resample_interval: str = "100ms",
    ):
        super().__init__()
        self.window_size_sec = window_size_sec
        self.window_step_sec = window_step_sec
        self.max_lag = max_lag
        self.interval_ms = interval_ms
        self.resample_interval = resample_interval

    def calculate(
        self,
        df_lead: pd.DataFrame,
        df_lag: pd.DataFrame,
        ts_col: str = "ts",
    ) -> LeadLagResult:
        df_lead_resampled = resample_and_forward_fill(df_lead, self.resample_interval, ts_col)
        df_lag_resampled = resample_and_forward_fill(df_lag, self.resample_interval, ts_col)
        LOG.info(f"Resampled lead: {len(df_lead_resampled)} lag: {len(df_lag_resampled)}")
        timestamps, aligned_lead, aligned_lag = align_series(df_lead_resampled, df_lag_resampled)
        LOG.info(f"Aligned series length: {len(aligned_lead)}")

        # Compute lead-lag for each rolling window
        window_size_steps = self.window_size_sec * 1000 // self.interval_ms
        window_step_steps = self.window_step_sec * 1000 // self.interval_ms

        start_idx = 0
        while start_idx + window_size_steps <= len(aligned_lead):
            end_idx = start_idx + window_size_steps

            # Extract window data
            window_lead = aligned_lead[start_idx:end_idx]
            window_lag = aligned_lag[start_idx:end_idx]
            # window_end_ts = timestamps[end_idx - 1].timestamp()  # Convert to Unix timestamp

            # Compute cross-correlation for this window
            best_lag, best_corr, corr_by_lag = compute_cross_correlation(window_lead, window_lag, self.max_lag)
            LOG.info(f"{best_lag}, {best_corr}, {corr_by_lag}")
            if best_lag is None:
                start_idx += window_step_steps
                continue

            # best_lag_ms = int(best_lag * self.interval_ms)
            # fields = {
            #     "best_lag": best_lag_ms,
            #     "best_corr": best_corr,
            #     "lag_0": corr_by_lag[0],
            #     f"lag_{int(self.interval_ms * 1)}": corr_by_lag[1],
            #     f"lag_{int(self.interval_ms * 2)}": corr_by_lag[2],
            # }

            start_idx += window_step_steps
