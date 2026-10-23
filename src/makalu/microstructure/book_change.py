"""
BookChange: Market microstructure features derived from order book snapshots.

This module implements price-aligned order book diffing to generate BookChange objects
for scalping and microstructure feature generation.
"""

from dataclasses import dataclass
from logging import getLogger

from qate.core.model import OrderBook, OrderLevel

LOG = getLogger(__name__)


@dataclass
class BookChange:
    ts: float  # Current snapshot timestamp
    k: int  # Number of top levels used in diff computation
    delta_sec: float  # Time elapsed since previous snapshot (seconds)

    # Best level change flags
    best_bid_reset: bool  # True if best bid price changed
    best_ask_reset: bool  # True if best ask price changed

    # Aggregated depth changes (top-K levels)
    delta_depth_bid: float  # Total bid depth change
    delta_depth_ask: float  # Total ask depth change
    delta_depth_bid_rate: float  # Bid depth change per second
    delta_depth_ask_rate: float  # Ask depth change per second

    # Pull/refill proxies (derived from depth rates)
    pull_bid_rate: float  # Rate of bid depth removal (positive)
    pull_ask_rate: float  # Rate of ask depth removal (positive)
    refill_bid_rate: float  # Rate of bid depth addition (positive)
    refill_ask_rate: float  # Rate of ask depth addition (positive)


@dataclass
class _Leg:
    order_book: OrderBook
    bid_depths: dict[int, float]
    ask_depths: dict[int, float]


def _extract_top_k_depths(
    depths: list[int], levels: list[OrderLevel], price_multiplier: float
) -> dict[int, float]:
    # price_multiplier kept for API compatibility; no longer needed for depth sums.
    depth_sums: dict[int, float] = {}
    depth_idx = 0
    total = 0.0
    for i, level in enumerate(levels):
        total += level.amount
        if (i + 1) == depths[depth_idx]:
            depth_sums[depths[depth_idx]] = total
            depth_idx += 1
            if depth_idx == len(depths):
                break

    # NOTE Analysis for below block
    # if depth_idx < len(depths):
    #   for i in range(depth_idx, len(depths)):
    #     price_maps[depths[i]] = price_map.copy()

    # If you request depths=[5, 10, 20] but only have 7 levels, this copies the 7-level map into the slots for
    # depth 10 and 20. The map still only contains the 7 prices that actually exist.

    # Why depth loss is NOT masked:

    # Consider this scenario:
    # - Snapshot A: 20 bid levels at prices [100, 99, 98, ..., 81]
    # - Snapshot B: 7 bid levels at prices [100, 99, ..., 94] (truncated)

    # For depth=20 comparison:
    # - prev_map[20] has entries for prices 100–81
    # - curr_map[20] has entries for prices 100–94 only

    # When _compute_delta_by_price runs:
    # all_prices = set(prev_map.keys()) | set(curr_map.keys())  # includes 100-81
    # for price in all_prices:
    # prev_size = prev_map.get(price, 0.0)
    # curr_size = curr_map.get(price, 0.0)  # returns 0.0 for prices 93-81
    # delta = curr_size - prev_size  # negative → depth loss captured!

    # The prices 93–81 will show delta = 0 - prev_size, correctly reflecting the depth loss.

    # The copy is just a fallback to ensure the key exists—it doesn't inject fake depth. The diff logic with set
    # union + dict.get(price, 0.0) correctly handles the asymmetry.

    if depth_idx < len(depths):
        for i in range(depth_idx, len(depths)):
            depth_sums[depths[i]] = total

    return depth_sums


class BookChangeBuilder:
    def __init__(self, depths: list[int], max_delta_sec: float = 10.0, price_multiplier: float = 1000.0):
        self._depths = sorted(depths)
        self._max_delta_sec = max_delta_sec
        self._price_multiplier = price_multiplier
        self._prev_leg = None

    def _is_valid(self, order_book: OrderBook) -> bool:
        if not order_book.bids or not order_book.asks:
            return False
        if order_book.spread is None:
            return False
        return True

    def on_snapshot(self, curr: OrderBook) -> list[BookChange] | None:
        if not self._is_valid(curr):
            return None

        curr_leg = _Leg(
            curr,
            _extract_top_k_depths(self._depths, curr.bids, self._price_multiplier),
            _extract_top_k_depths(self._depths, curr.asks, self._price_multiplier),
        )
        if not self._prev_leg:  # first update
            self._prev_leg = curr_leg
            return None

        prev_leg = self._prev_leg
        self._prev_leg = curr_leg

        delta_sec = curr_leg.order_book.get_ts() - prev_leg.order_book.get_ts()
        if delta_sec > self._max_delta_sec:
            # Discontinuity detected
            return None

        if delta_sec < 0.000001:  # 1us
            curr_str = str(curr_leg.order_book)
            prev_str = str(prev_leg.order_book)
            same = curr_str == prev_str
            if same:
                return None
            LOG.info(f"{delta_sec}\n{curr_str}\n{prev_str}\n")
            delta_sec = 0.000001

        results = []
        for k in self._depths:
            results.append(self._build(prev_leg, curr_leg, delta_sec, k))

        return results

    def _build(self, prev: _Leg, curr: _Leg, delta_sec: float, depth: int) -> BookChange:
        # Aggregate deltas directly without allocating per-price delta maps.
        delta_depth_bid = curr.bid_depths[depth] - prev.bid_depths[depth]
        delta_depth_ask = curr.ask_depths[depth] - prev.ask_depths[depth]

        # Compute rates
        # in practice delta_sec can not be 0, checked in on_snapshot
        delta_depth_bid_rate = delta_depth_bid / delta_sec
        delta_depth_ask_rate = delta_depth_ask / delta_sec

        # Compute pull/refill proxies
        pull_bid_rate = max(-delta_depth_bid_rate, 0.0)
        pull_ask_rate = max(-delta_depth_ask_rate, 0.0)
        refill_bid_rate = max(delta_depth_bid_rate, 0.0)
        refill_ask_rate = max(delta_depth_ask_rate, 0.0)

        # Reset flags
        best_bid_reset = curr.order_book.best_bid != prev.order_book.best_bid
        best_ask_reset = curr.order_book.best_ask != prev.order_book.best_ask

        # Create BookChange
        return BookChange(
            ts=curr.order_book.get_ts(),
            k=depth,
            delta_sec=delta_sec,
            best_bid_reset=best_bid_reset,
            best_ask_reset=best_ask_reset,
            delta_depth_bid=float(delta_depth_bid),
            delta_depth_ask=float(delta_depth_ask),
            delta_depth_bid_rate=float(delta_depth_bid_rate),
            delta_depth_ask_rate=float(delta_depth_ask_rate),
            pull_bid_rate=float(pull_bid_rate),
            pull_ask_rate=float(pull_ask_rate),
            refill_bid_rate=float(refill_bid_rate),
            refill_ask_rate=float(refill_ask_rate),
        )

    # Removed per-price delta computation in favor of direct depth sums.
