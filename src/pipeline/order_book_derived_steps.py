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


class PopulateMarketPriceStep(Step):
    # Degenerate-snapshot guard, NOT the old max_spread_bps=10.0 pre-filter
    # (removed in 24ecb0d because it dropped ~7% of legitimate snapshots at
    # SAR-flip moments). The collector used to emit partial books after every
    # snapshot resync (1-3 levels with stale far-away prices — fixed in
    # potential-panda-core cdc172a, but historical bronze retains them), and
    # pricing those produced +/-200-1100 bps reverting mid spikes. The check
    # is on the WALKED spread (market_price.spread_bps): partial books often
    # keep a tight top-of-book with almost nothing behind it, so the walk
    # lands hundreds of bps deep. 100 bps is ~4-10x any real walked stress
    # spread on the recorded venues (BINANCE p99.9 ~25 bps at size 0.1) yet
    # far below the artifact spreads; on the days checked (2026-03-03,
    # 05-13, 06-10) it removes every reverting spike.
    DEGENERATE_SPREAD_BPS = 100.0

    def __init__(self, execution_size_str: str, degenerate_spread_bps: float = DEGENERATE_SPREAD_BPS):
        super().__init__()
        self.execution_size_str = execution_size_str
        self.degenerate_spread_bps = degenerate_spread_bps

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        parquet_store = pipeline_context.parquet_store.create_execution_price(
            stage_context.market,
            self.execution_size_str,
        )
        count = 0
        for ts, fields in self._populate(pipeline_context, stage_context):
            parquet_store.add(ts, fields)
            count += 1
        parquet_store.close()
        LOG.info(f"PopulateMarketPriceStep: Wrote {count} records to Parquet store")

    def _populate(self, pipeline_context: PipelineContext, stage_context: Context):
        """Generator that yields (ts, fields) tuples."""
        skip_count = 0
        degenerate_count = 0
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.order_books))
        for order_book in stage_context.order_books:
            if not stage_context.populate_dt_range.is_within(order_book.get_ts()):
                continue

            market_price = order_book.market_price(float(self.execution_size_str))
            if not market_price:
                skip_count += 1
                continue

            if market_price.spread_bps > self.degenerate_spread_bps:
                degenerate_count += 1
                continue

            fields = {
                "bid": market_price.bid,
                "ask": market_price.ask,
                "mid": market_price.mid,
                "spread": market_price.spread,
                "spread_bps": market_price.spread_bps,
            }

            yield (order_book.get_ts(), fields)
            progress.update()
        LOG.info(
            f"PopulateMarketPriceStep: Processed {progress.cnt} objects,"
            f" skipped {skip_count} objects,"
            f" dropped {degenerate_count} degenerate snapshots"
        )


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
        parquet_store = pipeline_context.parquet_store.create_single_market("BookState", stage_context.market)
        count = 0
        for ts, fields in self._populate(pipeline_context, stage_context):
            parquet_store.add(ts, fields)
            count += 1
        parquet_store.close()
        LOG.info(f"PopulateBookStateStep: Wrote {count} records to Parquet store")

    def _populate(self, pipeline_context: PipelineContext, stage_context: Context):
        """Generator that yields (ts, fields) tuples."""
        skip_count = 0
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.order_books))
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
            if len(order_book.bids) < self.min_valid_levels or len(order_book.asks) < self.min_valid_levels:
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

        LOG.info(f"PopulateBookStateStep: Processed {progress.cnt} objects and skipped {skip_count} objects")


class PopulateBookChangeStep(Step):
    def __init__(self, k_levels: List[int], max_spread_bps: float = 10.0):
        super().__init__()
        self.k_levels = k_levels
        self.max_spread_bps = max_spread_bps

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        parquet_store = pipeline_context.parquet_store.create_single_market("BookChange", stage_context.market)
        count = 0
        for ts, write_fields in self._populate(pipeline_context, stage_context):
            parquet_store.add(ts, write_fields)
            count += 1
        parquet_store.close()
        LOG.info(f"PopulateBookChangeStep: Wrote {count} records to Parquet store")

    def _populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.order_books))
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
