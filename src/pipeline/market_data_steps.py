import logging
from dataclasses import dataclass

from qate.core.model import Market, OrderBook, Trade

from . import order_book_quality
from .framework import PipelineContext, StageContext, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constants
MARKET_DATA_BUCKET = "market_data"


# NOTE I Want to make it clear to see what are populated in each step
# Should explicitly write each field as much as possible even it looks verbose


@dataclass
class Context(StageContext):
    market: Market
    trades: list[Trade]
    order_books: list[OrderBook]


class PopulateTradeStep(Step):
    def __init__(self):
        super().__init__()

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.trades))
        parquet_store = pipeline_context.parquet_store.create_trade(stage_context.market)
        for trade in stage_context.trades:
            parquet_store.add_trade(trade)
            progress.update()
        parquet_store.close()
        LOG.info(f"PopulateTradeStep: Wrote {progress.cnt} trades to Parquet store")


class PopulateOrderBookStep(Step):
    # Malformed snapshots are dropped rather than stored: see
    # pipeline.order_book_quality for the rule and why it exists. Named here
    # too so a caller can read the thresholds off the step.
    MAX_SPREAD_BPS = order_book_quality.MAX_SPREAD_BPS
    MIN_VALID_LEVELS = order_book_quality.MIN_VALID_LEVELS

    def __init__(self):
        super().__init__()

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.order_books))
        parquet_store = pipeline_context.parquet_store.create_order_book(stage_context.market)
        skip_count = 0
        for order_book in stage_context.order_books:
            progress.update()
            if order_book_quality.is_malformed(order_book):
                skip_count += 1
                continue
            parquet_store.add_order_book(order_book)
        parquet_store.close()
        total = progress.cnt
        written = total - skip_count
        pct = (skip_count / total * 100) if total else 0.0
        LOG.info(f"PopulateOrderBookStep: Wrote {written} order books to Parquet store")
        # Counted and logged, never silent: these rows were present from the
        # start of collection and went unnoticed for months because nothing
        # downstream said how many it was dropping.
        LOG.info(
            f"PopulateOrderBookStep: Skipped {skip_count} malformed snapshots "
            f"of {total} ({pct:.4f}%): {order_book_quality.describe_rule()}"
        )
