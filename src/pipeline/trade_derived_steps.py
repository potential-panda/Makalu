import logging
from abc import abstractmethod
from typing import Any, Dict, List

from influxdb_client import Point, WritePrecision
from qate.store.influx_writer import InfluxWriter
from qate.trading.aggression import TradeAggressionBuilder
from qate.trading.chart import (
    BaseBarBuilder,
    CandleBarBuilder,
    HeikinAshiBarBuilder,
    RangeBarBuilder,
)

from .framework import Context, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constants
TRADE_DERIVED_BUCKET = "trade_derived"


# NOTE Want to make it clear to see what are populated in each step
# Should explicitly write each field even it looks verbose


class PopulateAggressionStep(Step):
    """Populates InfluxDB with Trade Aggression data."""

    def __init__(self, windows_sec: List[int]):
        super().__init__()
        self.windows_sec = windows_sec
        self.builder = TradeAggressionBuilder(windows_sec=self.windows_sec)

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Calculates and writes trade aggression metrics to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.trades) * len(self.windows_sec))
        for trade in context.trades:
            if not context.compute_dt_range.is_within(trade.get_ts()):
                continue
            self.builder.on_trade(trade)
            if not context.populate_dt_range.is_within(trade.get_ts()):
                continue
            aggressions = self.builder.get_trade_aggression(trade.get_ts())
            for aggression in aggressions:
                point = (
                    Point("TRADE_AGGRESSION")
                    .tag("exchange_name", trade.market.exchange_name.name)
                    .tag("symbol", trade.market.symbol.name)
                    .tag("window_sec", aggression.window_sec)
                    .field("buy_qty", aggression.buy_qty)
                    .field("sell_qty", aggression.sell_qty)
                    .field("buy_count", aggression.buy_count)
                    .field("sell_count", aggression.sell_count)
                    .field("imbalance_qty", aggression.imbalance_qty)
                    .field("imbalance_count", aggression.imbalance_count)
                    .field("qty_per_sec", aggression.qty_per_sec)
                    .field("trades_per_sec", aggression.trades_per_sec)
                    .field("buy_qty_per_sec", aggression.buy_qty_per_sec)
                    .field("sell_qty_per_sec", aggression.sell_qty_per_sec)
                    .field("vwap", aggression.vwap)
                    .field("vwap_buy", aggression.vwap_buy)
                    .field("vwap_sell", aggression.vwap_sell)
                    .field("dominant_size_bucket", aggression.dominant_size_bucket)
                    .field("dominant_bucket_count", aggression.dominant_bucket_count)
                    .field("dominant_bucket_share", aggression.dominant_bucket_share)
                    .field("burst_flag", aggression.burst_flag)
                    .field("max_trade_size", aggression.max_trade_size)
                    .field("total_qty", aggression.total_qty)
                    .field("total_count", aggression.total_count)
                    .field("avg_trade_size", aggression.avg_trade_size)
                    .field("net_flow", aggression.net_flow)
                    .field("buy_sell_ratio", aggression.buy_sell_ratio)
                    .time(int(aggression.ts * 1000), WritePrecision.MS)
                )
                influx_writer.add(point)
                progress.update()
        LOG.info(f"PopulateAggressionStep: Processed {progress.cnt} trade aggressions.")


class AbstractPopulateChartStep(Step):
    """
    Abstract base class for steps that populate chart data.
    """

    @abstractmethod
    def create_bar_builder(self) -> BaseBarBuilder:
        """Create a specific bar builder instance."""
        pass

    @abstractmethod
    def get_bar_type_tag(self) -> str:
        """Return the tag for the bar type."""
        pass

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """
        Builds bars from trades in context and writes them to InfluxDB.
        """
        builder = self.create_bar_builder()
        progress = StepProgressLogger(self.__class__.__name__, len(context.trades))
        for trade in context.trades:
            if not context.populate_dt_range.is_within(trade.get_ts()):
                continue
            bar = builder.on_update(trade)
            if bar:
                point = (
                    Point("CHART")
                    .tag("exchange_name", trade.market.exchange_name.name)
                    .tag("symbol", trade.market.symbol.name)
                    .tag("interval", self.get_interval_tag())
                    .tag("bar_type", self.get_bar_type_tag())
                    .field("open", bar.open)
                    .field("high", bar.high)
                    .field("low", bar.low)
                    .field("close", bar.close)
                    .field("volume", bar.volume)
                    .field("tick_count", bar.tick_count)
                    .time(int(bar.get_ts() * 1000), WritePrecision.MS)
                )
                influx_writer.add(point)
                progress.update()

        LOG.info(f"{self.__class__.__name__}: Created {progress.cnt} bars.")

    def get_interval_tag(self) -> str:
        """
        Returns a string representation of the interval for InfluxDB tagging.
        This can be overridden by subclasses if a different format is needed.
        """
        return "trade_based"


class PopulateCandleBarStep(AbstractPopulateChartStep):
    def __init__(self, interval_seconds: int):
        super().__init__()
        self.interval_seconds = interval_seconds

    def create_bar_builder(self) -> BaseBarBuilder:
        return CandleBarBuilder(period=self.interval_seconds)

    def get_bar_type_tag(self) -> str:
        return "CANDLE"

    def get_interval_tag(self) -> str:
        return f"{self.interval_seconds}s"


class PopulateHeikinAshiBarStep(AbstractPopulateChartStep):
    def __init__(self, interval_seconds: int):
        super().__init__()
        self.interval_seconds = interval_seconds

    def create_bar_builder(self) -> BaseBarBuilder:
        return HeikinAshiBarBuilder(period=self.interval_seconds)

    def get_bar_type_tag(self) -> str:
        return "HEIKIN_ASHI"

    def get_interval_tag(self) -> str:
        return f"{self.interval_seconds}s"


class PopulateRangeBarStep(AbstractPopulateChartStep):
    def __init__(self, box_size_rate: float):
        super().__init__()
        self.box_size_rate = box_size_rate

    def create_bar_builder(self) -> BaseBarBuilder:
        return RangeBarBuilder(box_size_rate=self.box_size_rate)

    def get_bar_type_tag(self) -> str:
        return "RANGE"

    def get_interval_tag(self) -> str:
        return f"rate_{int(100_000 * self.box_size_rate)}"
