import logging
from typing import Any, Dict, List

from qate.core.symbol import get_symbol_def
from qate.store.influx_writer import InfluxWriter
from qate.trading import intent
from qate.trading.book_change import BookChangeBuilder

from .converter import DataPointConverter
from .framework import Context, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constants
MARKET_DATA_BUCKET = "market_data"


class PopulateTradeStep(Step):
    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Converts Trade objects in context to InfluxDB points and writes them."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.trades))
        for trade in context.trades:
            if not context.populate_dt_range.is_within(trade.get_ts()):
                continue
            point = DataPointConverter.convert_trade_to_influx_point(trade)
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
            point = DataPointConverter.convert_market_price_to_influx_point(
                orderbook,
                market_price,
                execution_size_str,
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
            point = DataPointConverter.convert_order_book_to_influx_point(orderbook)
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
                point = DataPointConverter.convert_volume_imbalance_to_influx_point(orderbook, k, value)
                influx_writer.add(point)
                progress.update()

            for k in self.weighted_imbalance_k:
                value = intent.compute_weighted_imbalance(orderbook, k)
                point = DataPointConverter.convert_weighted_imbalance_to_influx_point(orderbook, k, value)
                influx_writer.add(point)
                progress.update()

            for rate in self.depth_imbalance_rate:
                value = intent.compute_depth_imbalance(orderbook, rate)
                point = DataPointConverter.convert_depth_imbalance_to_influx_point(orderbook, rate, value)
                influx_writer.add(point)
                progress.update()

            for k in self.microprice_k:
                value = intent.compute_microprice(orderbook, k)
                if value is not None:
                    point = DataPointConverter.convert_microprice_to_influx_point(orderbook, k, value)
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
                point = DataPointConverter.convert_book_change_to_influx_point(orderbook, book_change)
                influx_writer.add(point)
                progress.update()
        LOG.info(f"PopulateBookChangeStep: Processed {progress.cnt} book changes.")
