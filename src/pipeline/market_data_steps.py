import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from influxdb_client import Point, WritePrecision
from qate.core.model import Market, OrderBook, Side, Trade
from qate.store.influx_writer import InfluxWriter
from qate.store.parquet_store import ExecutionPriceStore, SingleMarketStore
from qate.trading import intent
from qate.trading.book_change import BookChangeBuilder

from .framework import ContextBase, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constants
MARKET_DATA_BUCKET = "market_data"


# NOTE I Want to make it clear to see what are populated in each step
# Should explicitly write each field as much as possible even it looks verbose


@dataclass
class Context(ContextBase):
    trades: list[Trade]
    order_books: list[OrderBook]
    market: Market


class PopulateTradeStep(Step):
    """
    Populates InfluxDB and optionally parquet files with Trade data.

    Args:
        parquet_store: Optional TradeStore for writing to daily rolling parquet files.
                       If provided, trades will be written to both InfluxDB and parquet.
    """

    def __init__(self):
        super().__init__()

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Converts Trade objects in context to InfluxDB points and writes them."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.trades))

        parquet_store = SingleMarketStore(context.parquet_root_dir, "Trade", context.market)

        for trade in context.trades:
            if not context.populate_dt_range.is_within(trade.get_ts()):
                continue

            influx_writer.add(
                Point("TRADE")
                .tag("exchange_name", trade.exchange_name.name)
                .tag("symbol", trade.symbol.name)
                .tag("side", trade.side.name)
                .field("price", trade.price)
                .field("size", trade.size)
                .time(int(trade.get_ts() * 1000), WritePrecision.MS)
            )

            parquet_store.add(
                trade.get_ts(),
                {
                    "side": 0 if trade.side == Side.BUY else 1,
                    "price": trade.price,
                    "size": trade.size,
                },
            )
            progress.update()

        parquet_store.close()
        LOG.info(f"PopulateTradeStep: Processed {progress.cnt} trades to InfluxDB.")


class PopulateMarketPriceStep(Step):
    """Populates InfluxDB with Market Price data."""

    def __init__(self, target_size: str):
        super().__init__()
        self.target_size = target_size

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Calculates market price from OrderBook objects and writes to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.order_books))
        execution_size_str = self.target_size
        parquet_store = ExecutionPriceStore(context.parquet_root_dir, context.market, execution_size_str)
        for order_book in context.order_books:
            if not context.populate_dt_range.is_within(order_book.get_ts()):
                continue
            market_price = order_book.market_price(float(self.target_size))
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
                .tag("execution_size", execution_size_str)
                .time(int(order_book.get_ts() * 1000), WritePrecision.MS)
            )
            for k, v in fields.items():
                point.field(k, v)
            influx_writer.add(point)

            parquet_store.add(order_book.get_ts(), fields)

            progress.update()

        parquet_store.close()
        LOG.info(f"PopulateMarketPriceStep: Processed {progress.cnt} market prices.")


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

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Extracts basic metrics from OrderBook objects and writes them to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.order_books))
        parquet_store = SingleMarketStore(context.parquet_root_dir, "BookState", context.market)
        for order_book in context.order_books:
            if not context.populate_dt_range.is_within(order_book.get_ts()):
                continue

            fields = {
                "best_bid": order_book.best_bid,
                "best_ask": order_book.best_ask,
                "mid": order_book.mid,
                "spread": order_book.spread,
                "spread_bps": order_book.spread_bps,
            }
            for k in self.volume_imbalance_k:
                value = intent.compute_volume_imbalance(order_book, k)
                fields[f"vol_imb_{k}"] = value

            for k in self.weighted_imbalance_k:
                value = intent.compute_weighted_imbalance(order_book, k)
                fields[f"weighted_imb_{k}"] = value

            for rate in self.depth_imbalance_rate:
                value = intent.compute_depth_imbalance(order_book, rate)
                rate_bps = int(rate * 10000)
                fields[f"depth_imb_{rate_bps}bps"] = value

            for k in self.microprice_k:
                value = intent.compute_microprice(order_book, k)
                fields[f"microprice_{k}"] = value

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

        parquet_store.close()
        LOG.info(f"PopulateOrderBookStep: Processed {progress.cnt} order books.")


class PopulateBookChangeStep(Step):
    """Populates InfluxDB with OrderBook Book Change data for various 'k' levels."""

    def __init__(self, k_levels: List[int]):
        super().__init__()
        self.k_levels = k_levels

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Calculates book changes from OrderBook objects for different k-levels and writes to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.order_books) * len(self.k_levels))
        parquet_store = SingleMarketStore(context.parquet_root_dir, "BookChange", context.market)

        builder = BookChangeBuilder(self.k_levels)
        for order_book in context.order_books:
            if not context.compute_dt_range.is_within(order_book.get_ts()):
                continue

            book_changes = builder.on_snapshot(order_book)
            if not book_changes:
                continue

            if not context.populate_dt_range.is_within(order_book.get_ts()):
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

            parquet_store.add(order_book.get_ts(), fields)
            progress.update()

        parquet_store.close()
        LOG.info(f"PopulateBookChangeStep: Processed {progress.cnt} book changes.")
