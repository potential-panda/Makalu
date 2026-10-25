"""
Volatility tracker for GMO market data.

Tracks volatility and spread in real-time for trade entry filtering.
"""

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
from qate.core.model import OrderBook

LOG = logging.getLogger(__name__)


@dataclass
class VolatilityConfig:
    """Configuration for volatility tracking."""

    return_shift_sec: int = 30
    volatility_lookback_sec: int = 1800
    ema_span_sec: int = 600
    min_vol_spread_ratio: float = 1.2
    sample_interval_sec: float = 1.0  # Resample interval for volatility calculation


class VolatilityTracker:
    """
    Tracks volatility and spread from GMO market data for trade entry filtering.

    Calculates:
    - Rolling volatility using return standard deviation
    - Rolling spread percentage
    - EMA smoothing of both metrics
    """

    def __init__(self, config: VolatilityConfig):
        self.config = config

        # Calculate number of samples needed
        self._shift_samples = int(config.return_shift_sec / config.sample_interval_sec)
        self._lookback_samples = int(
            config.volatility_lookback_sec / config.sample_interval_sec
        )

        # EMA decay factor
        self._ema_alpha = 2.0 / (config.ema_span_sec / config.sample_interval_sec + 1)

        # Price history for return calculation (stores mid prices)
        self._price_history: deque[float] = deque(
            maxlen=self._lookback_samples + self._shift_samples
        )

        # Return history for volatility calculation
        self._return_history: deque[float] = deque(maxlen=self._lookback_samples)

        # Spread history
        self._spread_pct_history: deque[float] = deque(maxlen=self._lookback_samples)

        # EMA values
        self._volatility_ema: float | None = None
        self._spread_pct_ema: float | None = None

        # Sampling state
        self._last_sample_ts: float | None = None
        self._last_orderbook: OrderBook | None = None

    def on_orderbook(self, orderbook: OrderBook) -> None:
        """Process an order book update."""
        ts = orderbook.get_ts()
        self._last_orderbook = orderbook

        # Check if we should sample
        if self._last_sample_ts is None:
            self._last_sample_ts = ts
            self._sample(orderbook)
            return

        # Sample at fixed intervals
        if ts - self._last_sample_ts >= self.config.sample_interval_sec:
            self._sample(orderbook)
            self._last_sample_ts = ts

    def _sample(self, orderbook: OrderBook) -> None:
        """Take a sample of the current state."""
        mid = (orderbook.best_bid + orderbook.best_ask) / 2
        spread = orderbook.best_ask - orderbook.best_bid
        spread_pct = spread / mid if mid > 0 else 0

        # Store price
        self._price_history.append(mid)

        # Calculate return if we have enough history
        if len(self._price_history) > self._shift_samples:
            old_price = self._price_history[-self._shift_samples - 1]
            if old_price > 0:
                ret = (mid - old_price) / old_price
                self._return_history.append(ret)

        # Store spread
        self._spread_pct_history.append(spread_pct)

        # Update volatility EMA
        if len(self._return_history) >= 10:  # Minimum samples for volatility
            volatility = np.std(list(self._return_history))
            if self._volatility_ema is None:
                self._volatility_ema = volatility
            else:
                self._volatility_ema = (
                    self._ema_alpha * volatility
                    + (1 - self._ema_alpha) * self._volatility_ema
                )

        # Update spread EMA
        if len(self._spread_pct_history) >= 1:
            current_spread_pct = self._spread_pct_history[-1]
            if self._spread_pct_ema is None:
                self._spread_pct_ema = current_spread_pct
            else:
                self._spread_pct_ema = (
                    self._ema_alpha * current_spread_pct
                    + (1 - self._ema_alpha) * self._spread_pct_ema
                )

    def get_volatility_ema(self) -> float | None:
        """Get the current EMA-smoothed volatility."""
        return self._volatility_ema

    def get_spread_pct_ema(self) -> float | None:
        """Get the current EMA-smoothed spread percentage."""
        return self._spread_pct_ema

    def is_tradeable(self) -> bool:
        """
        Check if current market conditions are favorable for trading.

        Returns True if volatility exceeds spread by the configured ratio.
        """
        if self._volatility_ema is None or self._spread_pct_ema is None:
            return False

        if self._spread_pct_ema <= 0:
            return False

        return (
            self._volatility_ema
            > self._spread_pct_ema * self.config.min_vol_spread_ratio
        )

    def get_barrier_pct(self, multiplier: float) -> float | None:
        """
        Get the barrier percentage based on current volatility.

        Args:
            multiplier: Barrier multiplier (e.g., 0.7)

        Returns:
            Barrier percentage, or None if volatility not available
        """
        if self._volatility_ema is None:
            return None
        return self._volatility_ema * multiplier

    def get_last_orderbook(self) -> OrderBook | None:
        """Get the last processed order book."""
        return self._last_orderbook

    def reset(self) -> None:
        """Reset all state."""
        self._price_history.clear()
        self._return_history.clear()
        self._spread_pct_history.clear()
        self._volatility_ema = None
        self._spread_pct_ema = None
        self._last_sample_ts = None
        self._last_orderbook = None
