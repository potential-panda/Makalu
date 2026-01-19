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
from influxdb_client import Point, WritePrecision
from qate.core.model import Market
from qate.store.influx_reader import InfluxReader
from qate.store.influx_writer import InfluxWriter

from .framework import Context, Step
from .market_data_steps import MARKET_DATA_BUCKET

LOG = logging.getLogger(__name__)

# Bucket name constant
LEAD_LAG_BUCKET = "lead_lag"


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
    """
    Align two price series using market A's timestamps as reference.

    Uses forward-fill to interpolate market B's prices to match market A's timestamps,
    matching the behavior of Study/lead_lag.py.

    Args:
        df_a: DataFrame with time index and price column for market A
        df_b: DataFrame with time index and price column for market B
        price_col: Column name to use for price (default: 'mid')

    Returns:
        (timestamps, aligned_prices_a, aligned_prices_b) as numpy arrays
    """
    if df_a.empty or df_b.empty:
        return pd.DatetimeIndex([]), np.array([]), np.array([])

    # Calculate mid price if not present
    if price_col == "mid" and "mid" not in df_a.columns:
        df_a = df_a.copy()
        df_a["mid"] = (df_a["bid"] + df_a["ask"]) / 2

    if price_col == "mid" and "mid" not in df_b.columns:
        df_b = df_b.copy()
        df_b["mid"] = (df_b["bid"] + df_b["ask"]) / 2

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


class LeadLagStep(Step):
    """
    Step for computing lead-lag metrics between two markets using rolling windows.

    Queries market price data for both markets from InfluxDB,
    aligns the series, and computes cross-correlation at various lags
    for each rolling window.
    """

    def __init__(
        self,
        window_size_sec: int,
        window_step_sec: int,
        # NOTE Below 3 args are based on data frequency of the actual market,
        # the default values are good for the markets we are trading
        max_lag: int = 10,
        interval_ms: int = 500,
        resample_interval: str = "100ms",
    ):
        """
        Initialize the LeadLagStep.

        Args:
            interval_ms: Aggregation interval in milliseconds (for output)
            max_lag: Maximum lag to test in steps
            resample_interval: Pandas resample interval string (e.g., '1s', '100ms')
            window_size_sec: Rolling window size in seconds (default: 300 = 5 minutes)
            window_step_sec: Step between windows in seconds (default: 60 = 1 minute)
        """
        super().__init__()
        self.interval_ms = interval_ms
        self.max_lag = max_lag
        self.resample_interval = resample_interval
        self.window_size_sec = window_size_sec
        self.window_step_sec = window_step_sec

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """
        Compute lead-lag metrics using rolling windows and write to InfluxDB.

        Expects context to have:
        - market_a: First market (potential leader)
        - market_b: Second market (potential follower)
        - influx_reader: InfluxReader instance
        - dt_range: Date range to process
        """
        market_lead: Market = context.market_lead
        market_lag: Market = context.market_lag
        influx_reader: InfluxReader = context.influx_reader

        LOG.info(f"LeadLagStep: Computing lead-lag between {market_lead.id} and {market_lag.id}")
        LOG.info(f"LeadLagStep: Window size={self.window_size_sec}s, step={self.window_step_sec}s")

        # Query market price data for both markets
        df_lead = influx_reader.query_market_price(MARKET_DATA_BUCKET, context.dt_range, market_lead)
        df_lag = influx_reader.query_market_price(MARKET_DATA_BUCKET, context.dt_range, market_lag)

        if df_lead.empty:
            LOG.warning(f"LeadLagStep: No market price data found for {market_lead.id}")
            return

        if df_lag.empty:
            LOG.warning(f"LeadLagStep: No market price data found for {market_lag.id}")
            return

        LOG.info(f"LeadLagStep: Loaded {len(df_lead)} prices for {market_lead.id}, {len(df_lag)} for {market_lag.id}")

        # Resample to fixed intervals using forward-fill
        df_lead_resampled = df_lead.resample(self.resample_interval).ffill().dropna()
        df_lag_resampled = df_lag.resample(self.resample_interval).ffill().dropna()

        LOG.info(
            f"LeadLagStep: After resampling: {len(df_lead_resampled)} for {market_lead.id},"
            f" {len(df_lag_resampled)} for {market_lag.id}"
        )

        # Align series with forward-fill (like Study/lead_lag.py)
        timestamps, aligned_lead, aligned_lag = align_series(df_lead_resampled, df_lag_resampled)

        if len(aligned_lead) == 0:
            LOG.warning("LeadLagStep: No overlapping timestamps between markets")
            return

        LOG.info(f"LeadLagStep: Aligned series length: {len(aligned_lead)}")

        # Compute lead-lag for each rolling window
        window_size_steps = self.window_size_sec * 1000 // self.interval_ms
        window_step_steps = self.window_step_sec * 1000 // self.interval_ms

        points_written = 0
        start_idx = 0

        while start_idx + window_size_steps <= len(aligned_lead):
            end_idx = start_idx + window_size_steps

            # Extract window data
            window_lead = aligned_lead[start_idx:end_idx]
            window_lag = aligned_lag[start_idx:end_idx]
            window_end_ts = timestamps[end_idx - 1]

            # Compute cross-correlation for this window
            best_lag, best_corr, corr_by_lag = compute_cross_correlation(window_lead, window_lag, self.max_lag)
            LOG.info(f"{best_lag}, {best_corr}, {corr_by_lag}")
            if best_lag is not None:
                # Convert best_lag from steps to milliseconds
                best_lag_ms = best_lag * self.interval_ms

                # Create and write point
                point = self._create_lead_lag_point(market_lead, market_lag, best_lag_ms, corr_by_lag, window_end_ts)
                influx_writer.add(point)
                points_written += 1

            # Move to next window
            start_idx += window_step_steps

        LOG.info(f"LeadLagStep: Wrote {points_written} points to InfluxDB")

    def _create_lead_lag_point(
        self,
        market_a: Market,
        market_b: Market,
        best_lag_ms: int,
        corr_by_lag: Dict[int, float],
        timestamp,
    ) -> Point:
        """
        Create a combined InfluxDB point for lead-lag metrics.

        Args:
            market_a: First market
            market_b: Second market
            best_lag_ms: Best lag in milliseconds
            corr_by_lag: Dict mapping lag steps to correlation values
            timestamp: Timestamp for the point

        Returns:
            InfluxDB Point with tags (market_a, market_b) and fields (best_lag, lag_0, lag_500, lag_1000)
        """
        point = (
            Point("LEAD_LAG")
            .tag("market_a", market_a.id)
            .tag("market_b", market_b.id)
            .field("best_lag", best_lag_ms)
            .time(timestamp, WritePrecision.NS)
        )

        # Add correlations at specific lag values (0ms, 500ms, 1000ms)
        target_lags_ms = [0, 500, 1000]
        for lag_ms in target_lags_ms:
            # Convert lag_ms to step
            step = lag_ms // self.interval_ms
            if step in corr_by_lag:
                point = point.field(f"lag_{lag_ms}", corr_by_lag[step])

        return point
