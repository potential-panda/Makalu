"""
TradeAggression: Microstructure features from taker-initiated trades.

Transforms trade flow into aggregated metrics for scalping and market analysis.

Optimized version using a single shared numpy buffer across all windows.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        """Run the same calculations in Python when numba is unavailable."""
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda func: func


from qate.core.model import Side, Trade


@dataclass(slots=True)
class TradeAggression:
    """
    Aggregated trade aggression metrics for a time window.

    Computed from taker-initiated trades to measure market pressure and flow.
    """

    # Timestamp and window
    ts: float  # Timestamp (seconds, matching Trade.exchange_ts)
    window_sec: int  # Window size in seconds

    # Core volume metrics
    buy_qty: float = 0.0  # Total aggressive buy volume
    sell_qty: float = 0.0  # Total aggressive sell volume
    buy_count: int = 0  # Number of aggressive buy trades
    sell_count: int = 0  # Number of aggressive sell trades

    # Derived metrics
    imbalance_qty: float = 0.0  # Quantity imbalance in [-1, 1]
    imbalance_count: float = 0.0  # Count imbalance in [-1, 1]
    qty_per_sec: float = 0.0  # Total volume per second
    trades_per_sec: float = 0.0  # Trades per second

    # Side-specific rates
    buy_qty_per_sec: float = 0.0  # Buy volume per second
    sell_qty_per_sec: float = 0.0  # Sell volume per second

    # VWAP (Volume Weighted Average Price)
    vwap: Optional[float] = None  # VWAP across all trades in window
    vwap_buy: Optional[float] = None  # VWAP for buy trades only
    vwap_sell: Optional[float] = None  # VWAP for sell trades only

    # Optional: size clustering (only populated if track_buckets=True)
    dominant_size_bucket: Optional[float] = None  # Most common trade size bucket
    dominant_bucket_count: int = 0  # Count of trades in dominant bucket
    dominant_bucket_share: float = 0.0  # Share of total trades in dominant bucket

    # Optional: burst detection
    burst_flag: bool = False  # True if burst detected in window
    max_trade_size: float = 0.0  # Largest single trade in window

    @property
    def total_qty(self) -> float:
        """Total aggressive volume (buy + sell)."""
        return self.buy_qty + self.sell_qty

    @property
    def total_count(self) -> int:
        """Total trade count (buy + sell)."""
        return self.buy_count + self.sell_count

    @property
    def avg_trade_size(self) -> float:
        """Average trade size in window."""
        if self.total_count == 0:
            return 0.0
        return self.total_qty / self.total_count

    @property
    def net_flow(self) -> float:
        """Net flow: buy_qty - sell_qty."""
        return self.buy_qty - self.sell_qty

    @property
    def buy_sell_ratio(self) -> float:
        """Ratio of buy to sell volume. Returns 0 if sell_qty is 0."""
        if self.sell_qty == 0:
            return float("inf") if self.buy_qty > 0 else 0.0
        return self.buy_qty / self.sell_qty

    @property
    def is_buy_dominant(self) -> bool:
        """Returns True if aggressive buys dominate."""
        return self.imbalance_qty > 0

    @property
    def is_sell_dominant(self) -> bool:
        """Returns True if aggressive sells dominate."""
        return self.imbalance_qty < 0


# Numba-optimized aggregation functions
@jit(nopython=True, cache=True)
def _compute_aggregates(
    timestamps: np.ndarray,
    is_buy: np.ndarray,
    sizes: np.ndarray,
    pq_sums: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> tuple:
    """
    Compute aggregates for trades in range [start_idx, end_idx).

    Returns: (buy_qty, sell_qty, buy_count, sell_count, buy_pq_sum, sell_pq_sum)
    """
    buy_qty = 0.0
    sell_qty = 0.0
    buy_count = 0
    sell_count = 0
    buy_pq_sum = 0.0
    sell_pq_sum = 0.0

    for i in range(start_idx, end_idx):
        size = sizes[i]
        pq = pq_sums[i]
        if is_buy[i]:
            buy_qty += size
            buy_count += 1
            buy_pq_sum += pq
        else:
            sell_qty += size
            sell_count += 1
            sell_pq_sum += pq

    return buy_qty, sell_qty, buy_count, sell_count, buy_pq_sum, sell_pq_sum


@jit(nopython=True, cache=True)
def _find_cutoff_index(timestamps: np.ndarray, cutoff_ts: float, start_idx: int, end_idx: int) -> int:
    """
    Find first index where timestamp >= cutoff_ts using binary search.

    Returns index in range [start_idx, end_idx].
    """
    lo = start_idx
    hi = end_idx

    while lo < hi:
        mid = (lo + hi) // 2
        if timestamps[mid] < cutoff_ts:
            lo = mid + 1
        else:
            hi = mid

    return lo


@jit(nopython=True, cache=True)
def _find_max_size(sizes: np.ndarray, start_idx: int, end_idx: int) -> float:
    """Find maximum size in range [start_idx, end_idx)."""
    if start_idx >= end_idx:
        return 0.0

    max_size = sizes[start_idx]
    for i in range(start_idx + 1, end_idx):
        if sizes[i] > max_size:
            max_size = sizes[i]
    return max_size


class _WindowAggregates:
    """
    Lightweight per-window aggregates that reference a shared trade buffer.

    Only stores the start_idx and running aggregates — no trade data arrays.
    """

    __slots__ = (
        "window_sec",
        "start_idx",
        "buy_qty",
        "sell_qty",
        "buy_count",
        "sell_count",
        "buy_pq_sum",
        "sell_pq_sum",
        "max_size",
        "max_size_valid",
        "track_buckets",
        "bucket_unit",
        "bucket_counts_total",
        "bucket_counts_buy",
        "bucket_counts_sell",
    )

    def __init__(self, window_sec: int, track_buckets: bool = False, bucket_unit: float = 0.001):
        self.window_sec = window_sec
        self.start_idx = 0
        self.buy_qty = 0.0
        self.sell_qty = 0.0
        self.buy_count = 0
        self.sell_count = 0
        self.buy_pq_sum = 0.0
        self.sell_pq_sum = 0.0
        self.max_size = 0.0
        self.max_size_valid = True
        self.track_buckets = track_buckets
        self.bucket_unit = bucket_unit
        if track_buckets:
            self.bucket_counts_total: dict[int, int] = {}
            self.bucket_counts_buy: dict[int, int] = {}
            self.bucket_counts_sell: dict[int, int] = {}
        else:
            self.bucket_counts_total = None
            self.bucket_counts_buy = None
            self.bucket_counts_sell = None

    def clear(self):
        self.start_idx = 0
        self.buy_qty = 0.0
        self.sell_qty = 0.0
        self.buy_count = 0
        self.sell_count = 0
        self.buy_pq_sum = 0.0
        self.sell_pq_sum = 0.0
        self.max_size = 0.0
        self.max_size_valid = True
        if self.track_buckets:
            self.bucket_counts_total.clear()
            self.bucket_counts_buy.clear()
            self.bucket_counts_sell.clear()


class TradeAggressionBuilder:
    """
    Builds TradeAggression metrics from streaming trades.

    Uses a single shared numpy buffer for all windows. Each window maintains
    only its own start_idx and running aggregates, avoiding redundant data
    storage and array writes.

    Usage:
        builder = TradeAggressionBuilder(
            windows_sec=[1, 3, 5],
            track_buckets=False,  # Disable for better performance
        )

        for trade in trades:
            builder.on_trade(trade)

        # Get current metrics
        aggression_metrics = builder.get_trade_aggression(current_ts)
        for metric in aggression_metrics:
            print(f"{metric.window_sec}s: imbalance={metric.imbalance_qty:.3f}")
    """

    def __init__(
        self,
        windows_sec: list[int],
        bucket_unit: float = 0.001,
        burst_qty_threshold: Optional[float] = None,
        eps: float = 1e-9,
        track_buckets: bool = False,
        track_max_size: bool = True,
        initial_capacity: int = 1024,
    ):
        self.windows_sec = windows_sec
        self.bucket_unit = bucket_unit
        self.burst_qty_threshold = burst_qty_threshold
        self.eps = eps
        self.track_buckets = track_buckets
        self.track_max_size = track_max_size

        # Single shared trade buffer
        self._capacity = initial_capacity
        self._timestamps = np.zeros(initial_capacity, dtype=np.float64)
        self._is_buy = np.zeros(initial_capacity, dtype=np.bool_)
        self._sizes = np.zeros(initial_capacity, dtype=np.float64)
        self._pq_sums = np.zeros(initial_capacity, dtype=np.float64)
        self._end_idx = 0

        # Per-window aggregates (lightweight — no data arrays)
        self._window_aggs: dict[int, _WindowAggregates] = {
            w: _WindowAggregates(w, track_buckets, bucket_unit) for w in self.windows_sec
        }
        self._max_window_sec = max(windows_sec)

        self.last_ts: Optional[float] = None

    def _grow(self):
        """Double the capacity of the shared buffer."""
        new_capacity = self._capacity * 2
        new_timestamps = np.zeros(new_capacity, dtype=np.float64)
        new_is_buy = np.zeros(new_capacity, dtype=np.bool_)
        new_sizes = np.zeros(new_capacity, dtype=np.float64)
        new_pq_sums = np.zeros(new_capacity, dtype=np.float64)

        n = self._end_idx
        new_timestamps[:n] = self._timestamps[:n]
        new_is_buy[:n] = self._is_buy[:n]
        new_sizes[:n] = self._sizes[:n]
        new_pq_sums[:n] = self._pq_sums[:n]

        self._timestamps = new_timestamps
        self._is_buy = new_is_buy
        self._sizes = new_sizes
        self._pq_sums = new_pq_sums
        self._capacity = new_capacity

    def _compact(self):
        """Compact by shifting data to remove globally evicted trades."""
        # The largest window has the smallest start_idx (keeps the most trades)
        global_start = self._window_aggs[self._max_window_sec].start_idx
        if global_start == 0:
            return

        n = self._end_idx - global_start
        if n > 0:
            env_idx = self._end_idx
            self._timestamps[:n] = self._timestamps[global_start:env_idx]
            self._is_buy[:n] = self._is_buy[global_start:env_idx]
            self._sizes[:n] = self._sizes[global_start:env_idx]
            self._pq_sums[:n] = self._pq_sums[global_start:env_idx]

        # Adjust all window start_idxs
        for agg in self._window_aggs.values():
            agg.start_idx -= global_start

        self._end_idx = n

    def _evict_window(self, agg: _WindowAggregates, cutoff_ts: float):
        """Evict trades older than cutoff_ts from a window's aggregates."""
        if agg.start_idx >= self._end_idx:
            return

        new_start = _find_cutoff_index(self._timestamps, cutoff_ts, agg.start_idx, self._end_idx)

        if new_start > agg.start_idx:
            for i in range(agg.start_idx, new_start):
                size = self._sizes[i]
                pq = self._pq_sums[i]
                if self._is_buy[i]:
                    agg.buy_qty -= size
                    agg.buy_count -= 1
                    agg.buy_pq_sum -= pq
                else:
                    agg.sell_qty -= size
                    agg.sell_count -= 1
                    agg.sell_pq_sum -= pq

                if agg.track_buckets:
                    bucket = int((size / agg.bucket_unit) + 0.5)
                    agg.bucket_counts_total[bucket] -= 1
                    if agg.bucket_counts_total[bucket] <= 0:
                        del agg.bucket_counts_total[bucket]

                    if self._is_buy[i]:
                        agg.bucket_counts_buy[bucket] -= 1
                        if agg.bucket_counts_buy[bucket] <= 0:
                            del agg.bucket_counts_buy[bucket]
                    else:
                        agg.bucket_counts_sell[bucket] -= 1
                        if agg.bucket_counts_sell[bucket] <= 0:
                            del agg.bucket_counts_sell[bucket]

            agg.max_size_valid = False
            agg.start_idx = new_start

    def on_trade(self, trade: Trade):
        """
        Process a new trade.

        Args:
            trade: Trade object with exchange_ts, side, size fields
        """
        ts = trade.get_ts()

        # Validate monotonic timestamps
        if self.last_ts is not None and ts < self.last_ts:
            # Drop out-of-order trades
            return

        self.last_ts = ts

        # Pre-compute values
        is_buy = trade.side == Side.BUY
        size = trade.size
        pq = trade.price * size

        # Ensure capacity in shared buffer
        if self._end_idx >= self._capacity:
            global_start = self._window_aggs[self._max_window_sec].start_idx
            if global_start > self._capacity // 4:
                self._compact()
            else:
                self._grow()

        # Append to shared buffer (once, not per-window)
        idx = self._end_idx
        self._timestamps[idx] = ts
        self._is_buy[idx] = is_buy
        self._sizes[idx] = size
        self._pq_sums[idx] = pq
        self._end_idx += 1

        # Update all window aggregates
        for agg in self._window_aggs.values():
            # Evict old trades from this window. Timestamps are in SECONDS
            # (Trade.exchange_ts), so the window is window_sec seconds — NOT
            # window_sec*1000. Using *1000 made every window 1000x too long.
            cutoff_ts = ts - agg.window_sec
            self._evict_window(agg, cutoff_ts)

            # Add new trade to aggregates
            if is_buy:
                agg.buy_qty += size
                agg.buy_count += 1
                agg.buy_pq_sum += pq
            else:
                agg.sell_qty += size
                agg.sell_count += 1
                agg.sell_pq_sum += pq

            # Update max size
            if size > agg.max_size:
                agg.max_size = size
                agg.max_size_valid = True

            # Update buckets if tracking
            if agg.track_buckets:
                bucket = int((size / agg.bucket_unit) + 0.5)
                agg.bucket_counts_total[bucket] = agg.bucket_counts_total.get(bucket, 0) + 1
                if is_buy:
                    agg.bucket_counts_buy[bucket] = agg.bucket_counts_buy.get(bucket, 0) + 1
                else:
                    agg.bucket_counts_sell[bucket] = agg.bucket_counts_sell.get(bucket, 0) + 1

    def get_trade_aggression(self, now_ts: float) -> list[TradeAggression]:
        """
        Get current TradeAggression metrics for all windows.

        Args:
            now_ts: Current timestamp (seconds, matching Trade.exchange_ts)

        Returns:
            List of TradeAggression objects, one per window
        """
        results = []

        for window_sec, agg in self._window_aggs.items():
            # Evict old trades first (timestamps in SECONDS — see on_trade)
            cutoff_ts = now_ts - window_sec
            self._evict_window(agg, cutoff_ts)

            # Get aggregates from state
            total_qty = agg.buy_qty + agg.sell_qty
            total_count = agg.buy_count + agg.sell_count

            # Compute imbalances
            if total_qty > self.eps:
                imbalance_qty = (agg.buy_qty - agg.sell_qty) / total_qty
            else:
                imbalance_qty = 0.0

            if total_count > 0:
                imbalance_count = (agg.buy_count - agg.sell_count) / (total_count + self.eps)
            else:
                imbalance_count = 0.0

            # Compute rates
            qty_per_sec = total_qty / float(window_sec)
            trades_per_sec = total_count / float(window_sec)
            buy_qty_per_sec = agg.buy_qty / float(window_sec)
            sell_qty_per_sec = agg.sell_qty / float(window_sec)

            # Compute VWAP
            vwap = None
            vwap_buy = None
            vwap_sell = None

            total_pq = agg.buy_pq_sum + agg.sell_pq_sum
            if total_qty > self.eps:
                vwap = total_pq / total_qty

            if agg.buy_qty > self.eps:
                vwap_buy = agg.buy_pq_sum / agg.buy_qty

            if agg.sell_qty > self.eps:
                vwap_sell = agg.sell_pq_sum / agg.sell_qty

            # Find dominant bucket (only if tracking)
            dominant_bucket = None
            dominant_count = 0
            dominant_share = 0.0

            if self.track_buckets and agg.bucket_counts_total:
                dominant_bucket_idx = max(agg.bucket_counts_total, key=agg.bucket_counts_total.get)
                dominant_count = agg.bucket_counts_total[dominant_bucket_idx]
                dominant_share = dominant_count / total_count if total_count > 0 else 0.0
                dominant_bucket = dominant_bucket_idx * self.bucket_unit

            # Burst detection
            burst_flag = False
            if self.burst_qty_threshold is not None:
                burst_flag = qty_per_sec >= self.burst_qty_threshold

            # Max size (lazy computation from shared buffer)
            max_size = 0.0
            if self.track_max_size:
                if not agg.max_size_valid:
                    agg.max_size = _find_max_size(self._sizes, agg.start_idx, self._end_idx)
                    agg.max_size_valid = True
                max_size = agg.max_size

            # Create TradeAggression
            aggression = TradeAggression(
                ts=now_ts,
                window_sec=window_sec,
                buy_qty=agg.buy_qty,
                sell_qty=agg.sell_qty,
                buy_count=agg.buy_count,
                sell_count=agg.sell_count,
                imbalance_qty=imbalance_qty,
                imbalance_count=imbalance_count,
                qty_per_sec=qty_per_sec,
                trades_per_sec=trades_per_sec,
                buy_qty_per_sec=buy_qty_per_sec,
                sell_qty_per_sec=sell_qty_per_sec,
                vwap=vwap,
                vwap_buy=vwap_buy,
                vwap_sell=vwap_sell,
                dominant_size_bucket=dominant_bucket,
                dominant_bucket_count=dominant_count,
                dominant_bucket_share=dominant_share,
                burst_flag=burst_flag,
                max_trade_size=max_size,
            )

            results.append(aggression)

        return results

    def clear(self):
        """Clear all windows and the shared buffer."""
        self._end_idx = 0
        for agg in self._window_aggs.values():
            agg.clear()


# Legacy compatibility
_FastWindowState = _WindowAggregates
_WindowState = _WindowAggregates
