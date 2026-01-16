"""
Lead-lag analysis steps for computing cross-correlation between two markets.

This module provides steps to analyze the lead-lag relationship between
price movements on different exchanges/markets.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from influxdb_client import Point, WritePrecision
from qate.core.model import Market
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two price series by their time index.

    Args:
        df_a: DataFrame with time index and price column for market A
        df_b: DataFrame with time index and price column for market B
        price_col: Column name to use for price (default: 'mid')

    Returns:
        (aligned_prices_a, aligned_prices_b) as numpy arrays
    """
    if df_a.empty or df_b.empty:
        return np.array([]), np.array([])

    # Calculate mid price if not present
    if price_col == "mid" and "mid" not in df_a.columns:
        df_a = df_a.copy()
        df_a["mid"] = (df_a["bid"] + df_a["ask"]) / 2

    if price_col == "mid" and "mid" not in df_b.columns:
        df_b = df_b.copy()
        df_b["mid"] = (df_b["bid"] + df_b["ask"]) / 2

    # Find common time index
    common_index = df_a.index.intersection(df_b.index)

    if len(common_index) == 0:
        return np.array([]), np.array([])

    aligned_a = df_a.loc[common_index, price_col].values
    aligned_b = df_b.loc[common_index, price_col].values

    # Remove any NaN values
    valid_mask = ~(np.isnan(aligned_a) | np.isnan(aligned_b))
    return aligned_a[valid_mask], aligned_b[valid_mask]


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
    Step for computing lead-lag metrics between two markets.

    Queries market price data for both markets from InfluxDB,
    aligns the series, and computes cross-correlation at various lags.
    """

    def __init__(
        self,
        interval_ms: int = 1000,
        max_lag: int = 60,
        resample_interval: str = "1s",
    ):
        """
        Initialize the LeadLagStep.

        Args:
            interval_ms: Aggregation interval in milliseconds (for output)
            max_lag: Maximum lag to test in steps
            resample_interval: Pandas resample interval string (e.g., '1s', '100ms')
        """
        super().__init__()
        self.interval_ms = interval_ms
        self.max_lag = max_lag
        self.resample_interval = resample_interval

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """
        Compute lead-lag metrics and write to InfluxDB.

        Expects context to have:
        - market_a: First market (potential leader)
        - market_b: Second market (potential follower)
        - influx_reader: InfluxReader instance
        - populate_dt_range: Date range to process
        """
        market_a: Market = context.market_a
        market_b: Market = context.market_b
        influx_reader = context.influx_reader
        dt_range = context.populate_dt_range

        LOG.info(f"LeadLagStep: Computing lead-lag between {market_a.id} and {market_b.id}")

        # Query market price data for both markets
        df_a = influx_reader.query_market_price(MARKET_DATA_BUCKET, dt_range, market_a)
        df_b = influx_reader.query_market_price(MARKET_DATA_BUCKET, dt_range, market_b)

        if df_a.empty:
            LOG.warning(f"LeadLagStep: No market price data found for {market_a.id}")
            return

        if df_b.empty:
            LOG.warning(f"LeadLagStep: No market price data found for {market_b.id}")
            return

        LOG.info(f"LeadLagStep: Loaded {len(df_a)} prices for {market_a.id}, {len(df_b)} for {market_b.id}")

        # Resample to fixed intervals using forward-fill
        df_a_resampled = df_a.resample(self.resample_interval).ffill().dropna()
        df_b_resampled = df_b.resample(self.resample_interval).ffill().dropna()

        LOG.info(f"LeadLagStep: After resampling: {len(df_a_resampled)} for {market_a.id}, {len(df_b_resampled)} for {market_b.id}")

        # Align series
        aligned_a, aligned_b = align_series(df_a_resampled, df_b_resampled)

        if len(aligned_a) == 0:
            LOG.warning("LeadLagStep: No overlapping timestamps between markets")
            return

        LOG.info(f"LeadLagStep: Aligned series length: {len(aligned_a)}")

        # Compute cross-correlation
        best_lag, best_corr, corr_by_lag = compute_cross_correlation(aligned_a, aligned_b, self.max_lag)

        if best_lag is None:
            LOG.warning("LeadLagStep: No significant correlation found")
            return

        # Convert best_lag from steps to milliseconds
        best_lag_ms = best_lag * self.interval_ms

        LOG.info(f"LeadLagStep: Best lag={best_lag_ms}ms, correlation={best_corr:.4f}")

        # Create combined point with best_lag and correlations at specific lags
        point = self._create_lead_lag_point(market_a, market_b, best_lag_ms, corr_by_lag, dt_range)
        influx_writer.add(point)

        # Store result in context for potential downstream use
        context.lead_lag_result = LeadLagResult(
            best_lag=best_lag,
            best_corr=best_corr,
            corr_by_lag=corr_by_lag,
            lag_seconds=best_lag_ms / 1000.0,
            leader_market=market_a.id if best_lag < 0 else (market_b.id if best_lag > 0 else None),
            follower_market=market_b.id if best_lag < 0 else (market_a.id if best_lag > 0 else None),
            aligned_length=len(aligned_a),
            interval_ms=self.interval_ms,
        )

        LOG.info("LeadLagStep: Wrote 1 point to InfluxDB")

    def _create_lead_lag_point(
        self,
        market_a: Market,
        market_b: Market,
        best_lag_ms: int,
        corr_by_lag: Dict[int, float],
        dt_range,
    ) -> Point:
        """
        Create a combined InfluxDB point for lead-lag metrics.

        Args:
            market_a: First market
            market_b: Second market
            best_lag_ms: Best lag in milliseconds
            corr_by_lag: Dict mapping lag steps to correlation values
            dt_range: Date range for timestamp

        Returns:
            InfluxDB Point with tags (market_a, market_b) and fields (best_lag, lag_0, lag_500, lag_1000)
        """
        point = (
            Point("LEAD_LAG")
            .tag("market_a", market_a.id)
            .tag("market_b", market_b.id)
            .field("best_lag", best_lag_ms)
            .time(dt_range.end_dt, WritePrecision.S)
        )

        # Add correlations at specific lag values (0ms, 500ms, 1000ms)
        target_lags_ms = [0, 500, 1000]
        for lag_ms in target_lags_ms:
            # Convert lag_ms to step
            step = lag_ms // self.interval_ms
            if step in corr_by_lag:
                point = point.field(f"lag_{lag_ms}", corr_by_lag[step])

        return point
