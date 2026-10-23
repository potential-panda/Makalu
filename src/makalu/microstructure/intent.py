from typing import Optional

from qate.core.model import OrderBook


def compute_volume_imbalance(order_book: OrderBook, k: int = 1) -> float:
    """
    Computes the volume imbalance at the top k levels of the order book.

    Args:
        order_book: The order book snapshot.
        k: The number of levels to consider.

    Returns:
        The volume imbalance, a value between -1 and 1.
    """
    if not order_book.bids or not order_book.asks:
        return 0.0

    bid_volume = sum(level.amount for i, level in enumerate(order_book.bids) if i < k)
    ask_volume = sum(level.amount for i, level in enumerate(order_book.asks) if i < k)

    total_volume = bid_volume + ask_volume
    if total_volume == 0:
        return 0.0

    return (bid_volume - ask_volume) / total_volume


def compute_weighted_imbalance(order_book: OrderBook, k: int = 5) -> float:
    """
    Computes the distance-weighted imbalance of the order book.

    Args:
        order_book: The order book snapshot.
        k: The number of levels to consider.

    Returns:
        The weighted imbalance, a value between -1 and 1.
    """
    if not order_book.bids or not order_book.asks:
        return 0.0

    mid_price = (order_book.best_bid + order_book.best_ask) / 2.0

    weighted_bid_volume = 0.0
    for i, level in enumerate(order_book.bids):
        if i >= k:
            break
        distance = abs(mid_price - level.price)
        weight = 1.0 / (1.0 + distance) if distance > 0 else 1.0
        weighted_bid_volume += level.amount * weight

    weighted_ask_volume = 0.0
    for i, level in enumerate(order_book.asks):
        if i >= k:
            break
        distance = abs(level.price - mid_price)
        weight = 1.0 / (1.0 + distance) if distance > 0 else 1.0
        weighted_ask_volume += level.amount * weight

    total_weighted_volume = weighted_bid_volume + weighted_ask_volume
    if total_weighted_volume == 0:
        return 0.0

    return (weighted_bid_volume - weighted_ask_volume) / total_weighted_volume


def compute_depth_imbalance(order_book: OrderBook, distance_rate: float = 0.1) -> float:
    """
    Computes the depth imbalance within a certain distance from the mid-price.

    Args:
        order_book: The order book snapshot.
        distance_rate: The distance from the mid-price as a rate of the mid-price.

    Returns:
        The depth imbalance, a value between -1 and 1.
    """
    if not order_book.bids or not order_book.asks:
        return 0.0

    mid_price = (order_book.best_bid + order_book.best_ask) / 2.0
    price_distance = mid_price * distance_rate

    bid_volume = sum(level.amount for level in order_book.bids if mid_price - level.price <= price_distance)
    ask_volume = sum(level.amount for level in order_book.asks if level.price - mid_price <= price_distance)

    total_volume = bid_volume + ask_volume
    if total_volume == 0:
        return 0.0

    return (bid_volume - ask_volume) / total_volume


def compute_microprice(order_book: OrderBook, k: int = 1) -> Optional[float]:
    """
    Computes the microprice of the order book.

    Args:
        order_book: The order book snapshot.
        k: The number of levels to consider.

    Returns:
        The microprice, or None if it cannot be computed.
    """
    if not order_book.bids or not order_book.asks:
        return None

    bid_volume = sum(level.amount for i, level in enumerate(order_book.bids) if i < k)
    ask_volume = sum(level.amount for i, level in enumerate(order_book.asks) if i < k)

    if bid_volume == 0 or ask_volume == 0:
        return None

    best_bid = order_book.best_bid
    best_ask = order_book.best_ask

    return (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)


def compute_depth_k(order_book: OrderBook, k: int) -> tuple[float, float]:
    """
    Compute top-k depth for bid and ask sides.

    Args:
        order_book: OrderBook snapshot
        k: Number of top levels to sum

    Returns:
        Tuple of (bid_depth, ask_depth)
    """
    bid_depth = sum(level.amount for i, level in enumerate(order_book.bids) if i < k)
    ask_depth = sum(level.amount for i, level in enumerate(order_book.asks) if i < k)
    return bid_depth, ask_depth


class OrderBookMetricsCache:
    """
    Precomputes prefix sums for an order book up to max_k levels,
    enabling O(1) lookups of volume_imbalance, weighted_imbalance,
    microprice, and depth_k for any k <= max_k.
    """

    def __init__(self, order_book: OrderBook, max_k: int):
        self._order_book = order_book
        self._max_k = max_k
        self._valid = bool(order_book.bids and order_book.asks)

        if not self._valid:
            return

        self._best_bid = order_book.best_bid
        self._best_ask = order_book.best_ask
        mid_price = (self._best_bid + self._best_ask) / 2.0

        # Build prefix sums for bid volumes and weighted bid volumes
        n_bids = min(len(order_book.bids), max_k)
        # bid_vol_prefix[i] = sum of amounts for bids[0..i-1], bid_vol_prefix[0] = 0
        self._bid_vol_prefix = [0.0] * (max_k + 1)
        self._weighted_bid_prefix = [0.0] * (max_k + 1)
        for i in range(n_bids):
            level = order_book.bids[i]
            self._bid_vol_prefix[i + 1] = self._bid_vol_prefix[i] + level.amount
            distance = abs(mid_price - level.price)
            weight = 1.0 / (1.0 + distance) if distance > 0 else 1.0
            self._weighted_bid_prefix[i + 1] = self._weighted_bid_prefix[i] + level.amount * weight
        # Fill remaining with the last value (if fewer levels than max_k)
        for i in range(n_bids + 1, max_k + 1):
            self._bid_vol_prefix[i] = self._bid_vol_prefix[n_bids]
            self._weighted_bid_prefix[i] = self._weighted_bid_prefix[n_bids]

        # Build prefix sums for ask volumes and weighted ask volumes
        n_asks = min(len(order_book.asks), max_k)
        self._ask_vol_prefix = [0.0] * (max_k + 1)
        self._weighted_ask_prefix = [0.0] * (max_k + 1)
        for i in range(n_asks):
            level = order_book.asks[i]
            self._ask_vol_prefix[i + 1] = self._ask_vol_prefix[i] + level.amount
            distance = abs(level.price - mid_price)
            weight = 1.0 / (1.0 + distance) if distance > 0 else 1.0
            self._weighted_ask_prefix[i + 1] = self._weighted_ask_prefix[i] + level.amount * weight
        for i in range(n_asks + 1, max_k + 1):
            self._ask_vol_prefix[i] = self._ask_vol_prefix[n_asks]
            self._weighted_ask_prefix[i] = self._weighted_ask_prefix[n_asks]

    def volume_imbalance(self, k: int) -> float:
        if not self._valid:
            return 0.0
        bid_vol = self._bid_vol_prefix[k]
        ask_vol = self._ask_vol_prefix[k]
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def weighted_imbalance(self, k: int) -> float:
        if not self._valid:
            return 0.0
        w_bid = self._weighted_bid_prefix[k]
        w_ask = self._weighted_ask_prefix[k]
        total = w_bid + w_ask
        if total == 0:
            return 0.0
        return (w_bid - w_ask) / total

    def microprice(self, k: int) -> Optional[float]:
        if not self._valid:
            return None
        bid_vol = self._bid_vol_prefix[k]
        ask_vol = self._ask_vol_prefix[k]
        if bid_vol == 0 or ask_vol == 0:
            return None
        return (self._best_bid * ask_vol + self._best_ask * bid_vol) / (bid_vol + ask_vol)

    def depth_k(self, k: int) -> tuple[float, float]:
        if not self._valid:
            return 0.0, 0.0
        return self._bid_vol_prefix[k], self._ask_vol_prefix[k]
