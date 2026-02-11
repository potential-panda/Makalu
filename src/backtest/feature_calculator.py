"""
Feature calculator for real-time feature extraction during backtesting.

Calculates features from BINANCE market data (order books and trades) in real-time.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from qate.core.model import OrderBook, Side, Trade
from qate.trading.aggression import TradeAggressionBuilder
from qate.trading.book_change import BookChange, BookChangeBuilder
from qate.trading.intent import (
    compute_depth_imbalance,
    compute_depth_k,
    compute_microprice,
    compute_volume_imbalance,
    compute_weighted_imbalance,
)

LOG = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature calculation."""

    # Order book features
    book_levels: int = 5

    # Trade aggression features (window sizes in seconds)
    aggression_windows_sec: list[int] = field(default_factory=lambda: [1, 3, 5])

    # Book change features (depth levels to track)
    book_change_depths: list[int] = field(default_factory=lambda: [5, 10, 20])

    # Depth imbalance distance rates
    depth_imbalance_rates: list[float] = field(default_factory=lambda: [0.001, 0.005, 0.01])

    # Minimum number of order books required before returning features
    min_orderbooks: int = 10


class FeatureCalculator:
    """
    Calculates features from BINANCE market data in real-time.

    Maintains rolling state for:
    - Order book features (microprice, imbalance, spread, depth)
    - Trade aggression features (buy/sell volume ratios over time windows)
    - Book change features (order book delta analysis)
    """

    def __init__(self, config: FeatureConfig):
        self.config = config

        # Trade aggression builder
        self._aggression_builder = TradeAggressionBuilder(
            windows_sec=config.aggression_windows_sec,
            track_buckets=False,
            track_max_size=True,
        )

        # Book change builder
        self._book_change_builder = BookChangeBuilder(
            depths=config.book_change_depths,
        )

        # Latest book change (for each depth level)
        self._last_book_changes: Optional[list[BookChange]] = None

        # Latest order book state
        self._last_orderbook: Optional[OrderBook] = None
        self._orderbook_count = 0

    def on_orderbook(self, orderbook: OrderBook) -> None:
        """Process an order book update."""
        book_changes = self._book_change_builder.on_snapshot(orderbook)
        if book_changes is not None:
            self._last_book_changes = book_changes
        self._last_orderbook = orderbook
        self._orderbook_count += 1

    def on_trade(self, trade: Trade) -> None:
        """Process a trade."""
        self._aggression_builder.on_trade(trade)

    def get_features(self) -> Optional[dict[str, float]]:
        """
        Get current features based on accumulated state.

        Returns:
            Dictionary of feature name -> value, or None if insufficient data
        """
        if self._last_orderbook is None:
            return None

        if self._orderbook_count < self.config.min_orderbooks:
            return None

        features = {}

        # Order book features
        ob = self._last_orderbook
        ts = ob.get_ts()

        # Basic prices and spread
        mid = (ob.best_bid + ob.best_ask) / 2
        spread = ob.best_ask - ob.best_bid
        spread_bps = (spread / mid) * 10000 if mid > 0 else 0

        features["mid"] = mid
        features["spread_bps"] = spread_bps

        # Microprice
        microprice = compute_microprice(ob, k=1)
        if microprice is not None:
            features["microprice"] = microprice
            features["microprice_vs_mid"] = (microprice - mid) / mid * 10000 if mid > 0 else 0

        # Volume imbalance at different levels
        for k in [1, 3, 5]:
            features[f"volume_imbalance_{k}"] = compute_volume_imbalance(ob, k=k)

        # Weighted imbalance
        features["weighted_imbalance"] = compute_weighted_imbalance(ob, k=self.config.book_levels)

        # Depth imbalance at different distance rates
        for rate in self.config.depth_imbalance_rates:
            rate_name = str(int(rate * 1000))  # e.g., 0.001 -> "1"
            features[f"depth_imbalance_{rate_name}bps"] = compute_depth_imbalance(ob, distance_rate=rate)

        # Depth at different levels
        for k in [1, 3, 5]:
            bid_depth, ask_depth = compute_depth_k(ob, k=k)
            features[f"bid_depth_{k}"] = bid_depth
            features[f"ask_depth_{k}"] = ask_depth
            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                features[f"depth_ratio_{k}"] = bid_depth / total_depth
            else:
                features[f"depth_ratio_{k}"] = 0.5

        # Trade aggression features
        aggression_list = self._aggression_builder.get_trade_aggression(ts)
        for aggression in aggression_list:
            window = aggression.window_sec
            features[f"buy_qty_{window}s"] = aggression.buy_qty
            features[f"sell_qty_{window}s"] = aggression.sell_qty
            features[f"aggression_imbalance_{window}s"] = aggression.imbalance_qty
            features[f"trades_per_sec_{window}s"] = aggression.trades_per_sec
            features[f"qty_per_sec_{window}s"] = aggression.qty_per_sec

        # Book change features
        if self._last_book_changes is not None:
            for bc in self._last_book_changes:
                k = bc.k
                features[f"delta_depth_bid_{k}"] = bc.delta_depth_bid
                features[f"delta_depth_ask_{k}"] = bc.delta_depth_ask
                features[f"pull_bid_rate_{k}"] = bc.pull_bid_rate
                features[f"pull_ask_rate_{k}"] = bc.pull_ask_rate
                features[f"refill_bid_rate_{k}"] = bc.refill_bid_rate
                features[f"refill_ask_rate_{k}"] = bc.refill_ask_rate

                # Derived features
                total_pull = bc.pull_bid_rate + bc.pull_ask_rate
                total_refill = bc.refill_bid_rate + bc.refill_ask_rate

                if total_pull > 0:
                    features[f"pull_imbalance_{k}"] = (bc.pull_bid_rate - bc.pull_ask_rate) / total_pull
                else:
                    features[f"pull_imbalance_{k}"] = 0.0

                if total_refill > 0:
                    features[f"refill_imbalance_{k}"] = (bc.refill_bid_rate - bc.refill_ask_rate) / total_refill
                else:
                    features[f"refill_imbalance_{k}"] = 0.0

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of all possible feature names."""
        names = [
            "mid",
            "spread_bps",
            "microprice",
            "microprice_vs_mid",
            "weighted_imbalance",
        ]

        # Volume imbalance
        for k in [1, 3, 5]:
            names.append(f"volume_imbalance_{k}")

        # Depth imbalance
        for rate in self.config.depth_imbalance_rates:
            rate_name = str(int(rate * 1000))
            names.append(f"depth_imbalance_{rate_name}bps")

        # Depth
        for k in [1, 3, 5]:
            names.extend([f"bid_depth_{k}", f"ask_depth_{k}", f"depth_ratio_{k}"])

        # Trade aggression
        for window in self.config.aggression_windows_sec:
            names.extend([
                f"buy_qty_{window}s",
                f"sell_qty_{window}s",
                f"aggression_imbalance_{window}s",
                f"trades_per_sec_{window}s",
                f"qty_per_sec_{window}s",
            ])

        # Book change
        for k in self.config.book_change_depths:
            names.extend([
                f"delta_depth_bid_{k}",
                f"delta_depth_ask_{k}",
                f"pull_bid_rate_{k}",
                f"pull_ask_rate_{k}",
                f"refill_bid_rate_{k}",
                f"refill_ask_rate_{k}",
                f"pull_imbalance_{k}",
                f"refill_imbalance_{k}",
            ])

        return names

    def reset(self) -> None:
        """Reset all state."""
        self._aggression_builder.clear()
        self._book_change_builder = BookChangeBuilder(
            depths=self.config.book_change_depths,
        )
        self._last_book_changes = None
        self._last_orderbook = None
        self._orderbook_count = 0
