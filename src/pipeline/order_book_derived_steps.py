import logging
from dataclasses import dataclass
from typing import List

from qate.core.model import Market, OrderBook

from makalu.microstructure import intent
from makalu.microstructure.book_change import BookChangeBuilder

from .framework import PipelineContext, StageContext, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constants
ORDER_BOOK_DERIVED_BUCKET = "order_book_derived"


# NOTE Want to make it clear to see what are populated in each step
# Should explicitly write each field even it looks verbose


@dataclass
class Context(StageContext):
    market: Market
    order_books: list[OrderBook]


class PopulateBookStateStep(Step):
    def __init__(
        self,
        volume_imbalance_k: List[int],
        weighted_imbalance_k: List[int],
        depth_imbalance_rate: List[float],
        microprice_k: List[int],
        max_spread_bps: float = 10.0,
        min_valid_levels: int = 3,
    ):
        super().__init__()
        self.volume_imbalance_k = volume_imbalance_k
        self.weighted_imbalance_k = weighted_imbalance_k
        self.depth_imbalance_rate = depth_imbalance_rate if depth_imbalance_rate else []
        self.microprice_k = microprice_k
        self.max_spread_bps = max_spread_bps
        self.min_valid_levels = min_valid_levels
        # Precompute max k across all k-based metrics for prefix sum caching
        self._max_k = max(
            max(volume_imbalance_k, default=0),
            max(weighted_imbalance_k, default=0),
            max(microprice_k, default=0),
        )

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        parquet_store = pipeline_context.parquet_store.create_single_market(
            "BookState", stage_context.market
        )
        count = 0
        for ts, fields in self._populate(pipeline_context, stage_context):
            parquet_store.add(ts, fields)
            count += 1
        parquet_store.close()
        LOG.info(f"PopulateBookStateStep: Wrote {count} records to Parquet store")

    def _populate(self, pipeline_context: PipelineContext, stage_context: Context):
        """Generator that yields (ts, fields) tuples."""
        skip_count = 0
        progress = StepProgressLogger(
            self.__class__.__name__, len(stage_context.order_books)
        )
        for order_book in stage_context.order_books:
            if not stage_context.populate_dt_range.is_within(order_book.get_ts()):
                continue
            if not order_book.best_bid or not order_book.best_ask:
                skip_count += 1
                continue
            if order_book.spread_bps > self.max_spread_bps:
                skip_count += 1
                continue
            # Depth as well as the touch. A partial book from a collector
            # resync often keeps a tight top-of-book with almost nothing
            # behind it, which the spread check above passes but the
            # imbalance and microprice fields below cannot be computed from.
            # Same threshold as PopulateOrderBookStep.MIN_VALID_LEVELS.
            if (
                len(order_book.bids) < self.min_valid_levels
                or len(order_book.asks) < self.min_valid_levels
            ):
                skip_count += 1
                continue

            fields = {
                "best_bid": order_book.best_bid,
                "best_ask": order_book.best_ask,
                "best_mid": order_book.mid,
                "best_spread": order_book.spread,
                "best_spread_bps": order_book.spread_bps,
            }

            cache = intent.OrderBookMetricsCache(order_book, self._max_k)

            for k in self.volume_imbalance_k:
                fields[f"vol_imb_k{k}"] = cache.volume_imbalance(k)

            for k in self.weighted_imbalance_k:
                fields[f"weighted_imb_k{k}"] = cache.weighted_imbalance(k)

            for rate in self.depth_imbalance_rate:
                value = intent.compute_depth_imbalance(order_book, rate)
                rate_bps = int(rate * 10000)
                fields[f"depth_imb_{rate_bps}bps"] = value

            for k in self.microprice_k:
                fields[f"microprice_k{k}"] = cache.microprice(k)

            yield (order_book.get_ts(), fields)
            progress.update()

        LOG.info(
            f"PopulateBookStateStep: Processed {progress.cnt} objects and skipped {skip_count} objects"
        )


class PopulateBookChangeStep(Step):
    def __init__(self, k_levels: List[int], max_spread_bps: float = 10.0):
        super().__init__()
        self.k_levels = k_levels
        self.max_spread_bps = max_spread_bps

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        parquet_store = pipeline_context.parquet_store.create_single_market(
            "BookChange", stage_context.market
        )
        count = 0
        for ts, write_fields in self._populate(pipeline_context, stage_context):
            parquet_store.add(ts, write_fields)
            count += 1
        parquet_store.close()
        LOG.info(f"PopulateBookChangeStep: Wrote {count} records to Parquet store")

    def _populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(
            self.__class__.__name__, len(stage_context.order_books)
        )
        builder = BookChangeBuilder(self.k_levels)
        for order_book in stage_context.order_books:
            book_changes = builder.on_snapshot(order_book)
            if not book_changes:
                continue

            if not stage_context.populate_dt_range.is_within(order_book.get_ts()):
                continue
            if order_book.spread_bps > self.max_spread_bps:
                continue

            write_fields = {
                "delta_sec": book_changes[0].delta_sec,
            }
            for book_change in book_changes:
                fields = {
                    "best_bid_reset": book_change.best_bid_reset,
                    "best_ask_reset": book_change.best_ask_reset,
                    "delta_depth_bid": book_change.delta_depth_bid,
                    "delta_depth_ask": book_change.delta_depth_ask,
                    "delta_depth_bid_rate": book_change.delta_depth_bid_rate,
                    "delta_depth_ask_rate": book_change.delta_depth_ask_rate,
                    "pull_bid_rate": book_change.pull_bid_rate,
                    "pull_ask_rate": book_change.pull_ask_rate,
                    "refill_bid_rate": book_change.refill_bid_rate,
                    "refill_ask_rate": book_change.refill_ask_rate,
                }
                for k, v in fields.items():
                    write_fields[f"{k}_k{book_change.k}"] = v

            yield (order_book.get_ts(), write_fields)
            progress.update()

        LOG.info(f"PopulateBookChangeStep: Processed {progress.cnt} objects")
