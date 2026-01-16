import logging
from abc import abstractmethod
from typing import Any, Dict, List

from qate.store.influx_writer import InfluxWriter
from qate.trading.aggression import TradeAggressionBuilder
from qate.trading.chart import (
    BaseBarBuilder,
    CandleBarBuilder,
    HeikinAshiBarBuilder,
    RangeBarBuilder,
)

from .converter import DataPointConverter
from .framework import Context, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constants
TRADE_DERIVED_BUCKET = "trade_derived"


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
                point = DataPointConverter.convert_trade_aggression_to_influx_point(trade.market, aggression)
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
                point = DataPointConverter.convert_bar_to_influx_point(
                    bar,
                    symbol=trade.market.symbol.name,
                    exchange=trade.market.exchange_name.name,
                    interval=self.get_interval_tag(),
                    bar_type=self.get_bar_type_tag(),
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
