import logging
from dataclasses import dataclass

from qate.core.model import Market, OrderBook, Trade

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
    def __init__(self):
        super().__init__()

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.order_books))
        parquet_store = pipeline_context.parquet_store.create_order_book(stage_context.market)
        for order_book in stage_context.order_books:
            parquet_store.add_order_book(order_book)
            progress.update()
        parquet_store.close()
        LOG.info(f"PopulateOrderBookStep: Wrote {progress.cnt} order books to Parquet store")
