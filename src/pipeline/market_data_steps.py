import logging
from dataclasses import dataclass

from influxdb_client import Point, WritePrecision
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

    def populate_influxdb(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.trades))
        influx_writer = pipeline_context.influx_store.create_writer(MARKET_DATA_BUCKET, no_retention=True)
        for trade in stage_context.trades:
            influx_writer.add(
                Point("TRADE")
                .tag("exchange_name", trade.exchange_name.name)
                .tag("symbol", trade.symbol.name)
                .tag("side", trade.side.name)
                .field("price", trade.price)
                .field("size", trade.size)
                .time(int(trade.get_ts() * 1000), WritePrecision.MS)
            )
            progress.update()
        influx_writer.close()
        LOG.info(f"PopulateTradeStep: Wrote {progress.cnt} points to InfluxDB")


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

    def populate_influxdb(self, pipeline_context: PipelineContext, stage_context: Context):
        # Order book data is not written to InfluxDB (too large)
        LOG.info("PopulateOrderBookStep: No InfluxDB output (order book data stored in Parquet only)")
