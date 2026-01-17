import logging
from typing import Any, Dict, List

from influxdb_client import Point, WritePrecision
from qate.core.symbol import get_symbol_def
from qate.store.influx_writer import InfluxWriter
from qate.trading import intent
from qate.trading.book_change import BookChangeBuilder

from .framework import Context, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constants
MARKET_DATA_BUCKET = "market_data"


# NOTE Want to make it clear to see what are populated in each step
# Should explicitly write each field even it looks verbose


class PopulateTradeStep(Step):
    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Converts Trade objects in context to InfluxDB points and writes them."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.trades))
        for trade in context.trades:
            if not context.populate_dt_range.is_within(trade.get_ts()):
                continue

            point = (
                Point("TRADE")
                .tag("exchange_name", trade.exchange_name.name)
                .tag("symbol", trade.symbol.name)
                .tag("side", trade.side.name)
                .field("price", trade.price)
                .field("size", trade.size)
                .time(int(trade.get_ts() * 1000), WritePrecision.MS)
            )
            influx_writer.add(point)
            progress.update()
        LOG.info(f"PopulateTradeStep: Processed {progress.cnt} trades.")


class PopulateMarketPriceStep(Step):
    """Populates InfluxDB with Market Price data."""

    def __init__(self, target_size: float = 0.01):
        super().__init__()
        self.target_size = target_size

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Calculates market price from OrderBook objects and writes to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.order_books))
        symbol_def = get_symbol_def(context.order_books[0].symbol)
        execution_size_str = symbol_def.fmt_size(self.target_size)
        for orderbook in context.order_books:
            if not context.populate_dt_range.is_within(orderbook.get_ts()):
                continue
            market_price = orderbook.market_price(self.target_size)
            if not market_price:
                continue
            point = (
                Point("MARKET_PRICE")
                .tag("exchange_name", orderbook.exchange_name.name)
                .tag("symbol", orderbook.symbol.name)
                .tag("execution_size", execution_size_str)
                .field("bid", market_price.bid)
                .field("ask", market_price.ask)
                .field("spread", market_price.spread)
                .field("spread_bps", market_price.spread_bps)
                .time(int(orderbook.get_ts() * 1000), WritePrecision.MS)
            )
            influx_writer.add(point)
            progress.update()
        LOG.info(f"PopulateMarketPriceStep: Processed {progress.cnt} market prices.")


class PopulateOrderBookStep(Step):
    """Populates InfluxDB with basic OrderBook data."""

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Extracts basic metrics from OrderBook objects and writes them to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.order_books))
        for orderbook in context.order_books:
            if not context.populate_dt_range.is_within(orderbook.get_ts()):
                continue
            point = (
                Point("ORDER_BOOK")
                .tag("exchange_name", orderbook.exchange_name.name)
                .tag("symbol", orderbook.symbol.name)
                .field("best_ask", orderbook.best_ask)
                .field("best_bid", orderbook.best_bid)
                .field("mid", orderbook.mid)
                .field("spread", orderbook.spread)
                .field("spread_bps", orderbook.spread_bps)
                .time(int(orderbook.get_ts() * 1000), WritePrecision.MS)
            )
            influx_writer.add(point)
            progress.update()
        LOG.info(f"PopulateOrderBookStep: Processed {progress.cnt} order books.")


class PopulateOrderImbalanceStep(Step):
    """Populates InfluxDB with OrderBook Imbalance data."""

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
        """Calculates and writes order book imbalance metrics to InfluxDB."""
        progress = StepProgressLogger(
            self.__class__.__name__,
            len(context.order_books)
            * (
                len(self.volume_imbalance_k)
                + len(self.weighted_imbalance_k)
                + len(self.depth_imbalance_rate)
                + len(self.microprice_k)
            ),
        )
        for orderbook in context.order_books:
            if not context.populate_dt_range.is_within(orderbook.get_ts()):
                continue
            for k in self.volume_imbalance_k:
                value = intent.compute_volume_imbalance(orderbook, k)
                point = (
                    Point("VOLUME_IMBALANCE")
                    .tag("exchange_name", orderbook.exchange_name.name)
                    .tag("symbol", orderbook.symbol.name)
                    .tag("k", k)
                    .field("value", value)
                    .time(int(orderbook.get_ts() * 1000), WritePrecision.MS)
                )
                influx_writer.add(point)
                progress.update()

            for k in self.weighted_imbalance_k:
                value = intent.compute_weighted_imbalance(orderbook, k)
                point = (
                    Point("WEIGHTED_IMBALANCE")
                    .tag("exchange_name", orderbook.exchange_name.name)
                    .tag("symbol", orderbook.symbol.name)
                    .tag("k", k)
                    .field("value", value)
                    .time(int(orderbook.get_ts() * 1000), WritePrecision.MS)
                )
                influx_writer.add(point)
                progress.update()

            for rate in self.depth_imbalance_rate:
                value = intent.compute_depth_imbalance(orderbook, rate)
                point = (
                    Point("DEPTH_IMBALANCE")
                    .tag("exchange_name", orderbook.exchange_name.name)
                    .tag("symbol", orderbook.symbol.name)
                    .tag("bps", int(rate * 10000))
                    .field("value", value)
                    .time(int(orderbook.get_ts() * 1000), WritePrecision.MS)
                )
                influx_writer.add(point)
                progress.update()

            for k in self.microprice_k:
                value = intent.compute_microprice(orderbook, k)
                if value is not None:
                    point = (
                        Point("MICROPRICE")
                        .tag("exchange_name", orderbook.exchange_name.name)
                        .tag("symbol", orderbook.symbol.name)
                        .tag("k", k)
                        .field("value", value)
                        .time(int(orderbook.get_ts() * 1000), WritePrecision.MS)
                    )
                    influx_writer.add(point)
                    progress.update()
        LOG.info(f"PopulateOrderImbalanceStep: Processed {progress.cnt} order book imbalances.")


class PopulateBookChangeStep(Step):
    """Populates InfluxDB with OrderBook Book Change data for various 'k' levels."""

    def __init__(self, k_levels: List[int]):
        super().__init__()
        self.k_levels = k_levels
        self.builders = [BookChangeBuilder(k) for k in self.k_levels]

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Calculates book changes from OrderBook objects for different k-levels and writes to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.order_books) * len(self.k_levels))
        for orderbook in context.order_books:
            if not context.compute_dt_range.is_within(orderbook.get_ts()):
                continue
            for builder in self.builders:
                book_change = builder.on_snapshot(orderbook)
                if not book_change:
                    continue
                if not context.populate_dt_range.is_within(orderbook.get_ts()):
                    continue
                point = (
                    Point("BOOK_CHANGE")
                    .tag("exchange_name", orderbook.exchange_name.name)
                    .tag("symbol", orderbook.symbol.name)
                    .tag("k", str(book_change.k))
                    .field("delta_sec", book_change.delta_sec)
                    .field("best_bid_reset", book_change.best_bid_reset)
                    .field("best_ask_reset", book_change.best_ask_reset)
                    .field("delta_depth_bid", book_change.delta_depth_bid)
                    .field("delta_depth_ask", book_change.delta_depth_ask)
                    .field("delta_depth_bid_rate", book_change.delta_depth_bid_rate)
                    .field("delta_depth_ask_rate", book_change.delta_depth_ask_rate)
                    .field("pull_bid_rate", book_change.pull_bid_rate)
                    .field("pull_ask_rate", book_change.pull_ask_rate)
                    .field("refill_bid_rate", book_change.refill_bid_rate)
                    .field("refill_ask_rate", book_change.refill_ask_rate)
                    .time(int(orderbook.get_ts() * 1000), WritePrecision.MS)
                )
                influx_writer.add(point)
                progress.update()
        LOG.info(f"PopulateBookChangeStep: Processed {progress.cnt} book changes.")
