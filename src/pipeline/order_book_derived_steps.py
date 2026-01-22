import logging
from dataclasses import dataclass
from typing import List

from influxdb_client import Point, WritePrecision
from qate.core.model import Market, OrderBook
from qate.trading.book_change import BookChangeBuilder
from qate.util.dt_range import DtRange

from .framework import PipelineContext, StageContext, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constants
ORDER_BOOK_DERIVED_BUCKET = "order_book_derived"


# NOTE Want to make it clear to see what are populated in each step
# Should explicitly write each field even it looks verbose


@dataclass
class Context(StageContext):
    compute_dt_range: DtRange
    market: Market
    order_books: list[OrderBook]


class PopulateBookChangeStep(Step):
    def __init__(self, k_levels: List[int]):
        super().__init__()
        self.k_levels = k_levels

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.order_books) * len(self.k_levels))

        influx_writer = pipeline_context.influx_store.create_writer(ORDER_BOOK_DERIVED_BUCKET, no_retention=True)
        parquet_store = pipeline_context.parquet_store.create_single_market("BookChange", stage_context.market)
        builder = BookChangeBuilder(self.k_levels)
        for order_book in stage_context.order_books:
            if not stage_context.compute_dt_range.is_within(order_book.get_ts()):
                continue

            book_changes = builder.on_snapshot(order_book)
            if not book_changes:
                continue

            if not stage_context.populate_dt_range.is_within(order_book.get_ts()):
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
                point = (
                    Point("BOOK_CHANGE")
                    .tag("exchange_name", order_book.exchange_name.name)
                    .tag("symbol", order_book.symbol.name)
                    .tag("k", str(book_change.k))
                    .field("delta_sec", book_change.delta_sec)
                    .time(int(order_book.get_ts() * 1000), WritePrecision.MS)
                )
                for k, v in fields.items():
                    point.field(k, v)
                influx_writer.add(point)

                for k, v in fields.items():
                    write_fields[f"{k}_{book_change.k}"] = v

            influx_writer.add(point)
            parquet_store.add(order_book.get_ts(), fields)
            progress.update()

        influx_writer.close()
        parquet_store.close()
        LOG.info(f"PopulateBookChangeStep: Processed {progress.cnt} objects.")
