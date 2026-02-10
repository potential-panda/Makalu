import logging
from dataclasses import dataclass
from typing import List

from influxdb_client import Point, WritePrecision
from qate.core.model import Market, OrderBook
from qate.trading import intent
from qate.trading.book_change import BookChangeBuilder

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
    def __init__(self, execution_size_str: str):
        super().__init__()
        self.execution_size_str = execution_size_str

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.order_books))

        influx_writer = pipeline_context.influx_store.create_writer(ORDER_BOOK_DERIVED_BUCKET, no_retention=True)
        parquet_store = pipeline_context.parquet_store.create_execution_price(
            stage_context.market,
            self.execution_size_str,
        )
        for order_book in stage_context.order_books:
            if not stage_context.populate_dt_range.is_within(order_book.get_ts()):
                continue

            market_price = order_book.market_price(float(self.execution_size_str))
            if not market_price:
                continue

            fields = {
                "bid": market_price.bid,
                "ask": market_price.ask,
                "mid": market_price.mid,
                "spread": market_price.spread,
                "spread_bps": market_price.spread_bps,
            }

            point = (
                Point("MARKET_PRICE")
                .tag("exchange_name", order_book.exchange_name.name)
                .tag("symbol", order_book.symbol.name)
                .tag("execution_size", self.execution_size_str)
                .time(int(order_book.get_ts() * 1000), WritePrecision.MS)
            )
            for k, v in fields.items():
                point.field(k, v)

            influx_writer.add(point)
            parquet_store.add(order_book.get_ts(), fields)
            progress.update()

        influx_writer.close()
        parquet_store.close()
        LOG.info(f"PopulateMarketPriceStep: Processed {progress.cnt} objects")


class PopulateBookStateStep(Step):
    def __init__(
        self,
        volume_imbalance_k: List[int],
        weighted_imbalance_k: List[int],
        depth_imbalance_rate: List[float],
        microprice_k: List[int],
    ):
        super().__init__()
        self.volume_imbalance_k = volume_imbalance_k
        self.weighted_imbalance_k = weighted_imbalance_k
        self.depth_imbalance_rate = depth_imbalance_rate
        self.microprice_k = microprice_k

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.order_books))

        influx_writer = pipeline_context.influx_store.create_writer(ORDER_BOOK_DERIVED_BUCKET, no_retention=True)
        parquet_store = pipeline_context.parquet_store.create_single_market("BookState", stage_context.market)
        for order_book in stage_context.order_books:
            if not stage_context.populate_dt_range.is_within(order_book.get_ts()):
                continue

            fields = {
                "best_bid": order_book.best_bid,
                "best_ask": order_book.best_ask,
                "best_mid": order_book.mid,
                "best_spread": order_book.spread,
                "best_spread_bps": order_book.spread_bps,
            }
            for k in self.volume_imbalance_k:
                value = intent.compute_volume_imbalance(order_book, k)
                fields[f"vol_imb_k{k}"] = value

            for k in self.weighted_imbalance_k:
                value = intent.compute_weighted_imbalance(order_book, k)
                fields[f"weighted_imb_k{k}"] = value

            for rate in self.depth_imbalance_rate:
                value = intent.compute_depth_imbalance(order_book, rate)
                rate_bps = int(rate * 10000)
                fields[f"depth_imb_{rate_bps}bps"] = value

            for k in self.microprice_k:
                value = intent.compute_microprice(order_book, k)
                fields[f"microprice_k{k}"] = value

            point = (
                Point("BOOK_STATE")
                .tag("exchange_name", order_book.exchange_name.name)
                .tag("symbol", order_book.symbol.name)
                .time(int(order_book.get_ts() * 1000), WritePrecision.MS)
            )
            for k, v in fields.items():
                point.field(k, v)

            influx_writer.add(point)
            parquet_store.add(order_book.get_ts(), fields)
            progress.update()

        influx_writer.close()
        parquet_store.close()
        LOG.info(f"PopulateOrderBookStep: Processed {progress.cnt} objects")


class PopulateBookChangeStep(Step):
    def __init__(self, k_levels: List[int]):
        super().__init__()
        self.k_levels = k_levels

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.order_books))

        influx_writer = pipeline_context.influx_store.create_writer(ORDER_BOOK_DERIVED_BUCKET, no_retention=True)
        parquet_store = pipeline_context.parquet_store.create_single_market("BookChange", stage_context.market)
        builder = BookChangeBuilder(self.k_levels)
        for order_book in stage_context.order_books:
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
                    write_fields[f"{k}_k{book_change.k}"] = v

            influx_writer.add(point)
            parquet_store.add(order_book.get_ts(), write_fields)
            progress.update()

        influx_writer.close()
        parquet_store.close()
        LOG.info(f"PopulateBookChangeStep: Processed {progress.cnt} objects.")
