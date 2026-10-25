"""
Event merger for combining market data from multiple sources.

Merges events from different exchanges into a single chronologically ordered stream.
"""

import heapq
import logging
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum, auto

from qate.core.model import OrderBook, Trade

LOG = logging.getLogger(__name__)


class EventType(Enum):
    """Type of market event."""

    BINANCE_ORDERBOOK = auto()
    BINANCE_TRADE = auto()
    GMO_ORDERBOOK = auto()


@dataclass
class Event:
    """A market event with timestamp and data."""

    ts: float
    event_type: EventType
    data: OrderBook | Trade

    def __lt__(self, other: "Event") -> bool:
        """Compare events by timestamp for heap ordering."""
        return self.ts < other.ts


class EventMerger:
    """Merges events from multiple sources in chronological order."""

    @staticmethod
    def merge(
        binance_orderbooks: list[OrderBook],
        binance_trades: list[Trade],
        gmo_orderbooks: list[OrderBook],
    ) -> Generator[Event, None, None]:
        """
        Merge events from multiple sources into a single chronologically ordered stream.

        Uses a min-heap to efficiently merge sorted lists while minimizing memory usage.

        Args:
            binance_orderbooks: List of BINANCE order books
            binance_trades: List of BINANCE trades
            gmo_orderbooks: List of GMO order books

        Yields:
            Events in chronological order
        """
        # Create iterators with event types
        sources = [
            (binance_orderbooks, EventType.BINANCE_ORDERBOOK),
            (binance_trades, EventType.BINANCE_TRADE),
            (gmo_orderbooks, EventType.GMO_ORDERBOOK),
        ]

        # Initialize heap with first element from each non-empty source
        heap = []
        iterators = []

        for data_list, event_type in sources:
            if data_list:
                it = iter(data_list)
                try:
                    first = next(it)
                    event = Event(ts=first.get_ts(), event_type=event_type, data=first)
                    heapq.heappush(heap, (event.ts, len(iterators), event))
                    iterators.append((it, event_type))
                except StopIteration:
                    iterators.append((None, event_type))
            else:
                iterators.append((None, event_type))

        LOG.info(
            f"EventMerger: Merging {len(binance_orderbooks)} BINANCE OBs, "
            f"{len(binance_trades)} BINANCE trades, {len(gmo_orderbooks)} GMO OBs"
        )

        count = 0
        while heap:
            _ts, source_idx, event = heapq.heappop(heap)
            yield event
            count += 1

            # Get next element from the same source
            it, event_type = iterators[source_idx]
            if it is not None:
                try:
                    next_item = next(it)
                    next_event = Event(
                        ts=next_item.get_ts(), event_type=event_type, data=next_item
                    )
                    heapq.heappush(heap, (next_event.ts, source_idx, next_event))
                except StopIteration:
                    iterators[source_idx] = (None, event_type)

        LOG.info(f"EventMerger: Yielded {count} events")
