import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
from influxdb_client import Point, WritePrecision
from qate.core.model import Market, Side, Trade
from qate.store.influx_writer import InfluxWriter
from qate.store.parquet_store import ChartStore, SingleMarketStore
from qate.trading.aggression import TradeAggressionBuilder
from qate.trading.chart import (
    BaseBarBuilder,
    CandleBarBuilder,
    HeikinAshiBarBuilder,
    RangeBarBuilder,
)

from .framework import ContextBase, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constants
TRADE_DERIVED_BUCKET = "trade_derived"


# NOTE Want to make it clear to see what are populated in each step
# Should explicitly write each field even it looks verbose


@dataclass
class Context(ContextBase):
    trades_df: pd.DataFrame
    market: Market


def _convert_to_trade(trade_row) -> Trade:
    return Trade(
        "",
        Side.BUY if trade_row.side == 0 else Side.SELL,
        trade_row.price,
        trade_row.size,
        None,
        None,
        trade_row.ts,
    )


class PopulateAggressionStep(Step):
    """Populates InfluxDB with Trade Aggression data."""

    def __init__(self, windows_sec: List[int]):
        super().__init__()
        self.windows_sec = windows_sec

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Calculates and writes trade aggression metrics to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.trades_df) * len(self.windows_sec))
        parquet_store = SingleMarketStore(context.parquet_root_dir, "TradeAggression", context.market)
        builder = TradeAggressionBuilder(windows_sec=self.windows_sec)

        for trade_row in context.trades_df.itertuples(index=False):
            if not context.compute_dt_range.is_within(trade_row.ts):
                continue
            trade = _convert_to_trade(trade_row)
            builder.on_trade(trade)
            if not context.populate_dt_range.is_within(trade.get_ts()):
                continue

            write_fields = {}
            aggressions = builder.get_trade_aggression(trade.get_ts())
            for aggression in aggressions:
                fields = {
                    "buy_qty": aggression.buy_qty,
                    "sell_qty": aggression.sell_qty,
                    "buy_count": aggression.buy_count,
                    "sell_count": aggression.sell_count,
                    "imbalance_qty": aggression.imbalance_qty,
                    "imbalance_count": aggression.imbalance_count,
                    "qty_per_sec": aggression.qty_per_sec,
                    "trades_per_sec": aggression.trades_per_sec,
                    "buy_qty_per_sec": aggression.buy_qty_per_sec,
                    "sell_qty_per_sec": aggression.sell_qty_per_sec,
                    "vwap": aggression.vwap,
                    "vwap_buy": aggression.vwap_buy,
                    "vwap_sell": aggression.vwap_sell,
                    "dominant_size_bucket": aggression.dominant_size_bucket,
                    "dominant_bucket_count": aggression.dominant_bucket_count,
                    "dominant_bucket_share": aggression.dominant_bucket_share,
                    "burst_flag": aggression.burst_flag,
                    "max_trade_size": aggression.max_trade_size,
                    "total_qty": aggression.total_qty,
                    "total_count": aggression.total_count,
                    "avg_trade_size": aggression.avg_trade_size,
                    "net_flow": aggression.net_flow,
                    "buy_sell_ratio": aggression.buy_sell_ratio,
                }
                point = (
                    Point("TRADE_AGGRESSION")
                    .tag("exchange_name", context.market.exchange_name.name)
                    .tag("symbol", context.market.symbol.name)
                    .tag("window_sec", aggression.window_sec)
                    .time(int(aggression.ts * 1000), WritePrecision.MS)
                )
                for k, v in fields.items():
                    point.field(k, v)
                influx_writer.add(point)

                for k, v in fields.items():
                    write_fields[f"{k}_{aggression.window_sec}"] = v

            parquet_store.add(trade.get_ts(), write_fields)
            progress.update()

        parquet_store.close()
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
    def get_bar_type(self) -> str:
        """Return the tag for the bar type."""
        pass

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """
        Builds bars from trades in context and writes them to InfluxDB.
        """
        progress = StepProgressLogger(self.__class__.__name__, len(context.trades_df))
        parquet_store = ChartStore(context.parquet_root_dir, context.market, self.get_bar_type())
        builder = self.create_bar_builder()
        for trade_row in context.trades_df.itertuples(index=False):
            if not context.populate_dt_range.is_within(trade_row.ts):
                continue
            trade = _convert_to_trade(trade_row)
            bar = builder.on_update(trade)
            if not bar:
                continue
            if not context.populate_dt_range.is_within(trade.get_ts()):
                continue

            fields = {
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "tick_count": bar.tick_count,
            }
            point = (
                Point("CHART")
                .tag("exchange_name", context.market.exchange_name.name)
                .tag("symbol", context.market.symbol.name)
                .tag("interval", self.get_threshold_str())
                .tag("bar_type", self.get_bar_type())
                .time(int(bar.get_ts() * 1000), WritePrecision.MS)
            )
            for k, v in fields.items():
                point.field(k, v)
            influx_writer.add(point)

            parquet_store.add(trade.get_ts(), fields)
            progress.update()

        parquet_store.close()
        LOG.info(f"{self.__class__.__name__}: Created {progress.cnt} bars.")

    @abstractmethod
    def get_threshold_str(self) -> str:
        pass


class PopulateCandleBarStep(AbstractPopulateChartStep):
    def __init__(self, interval_seconds: int):
        super().__init__()
        self.interval_seconds = interval_seconds

    def create_bar_builder(self) -> BaseBarBuilder:
        return CandleBarBuilder(period=self.interval_seconds)

    def get_bar_type(self) -> str:
        return "CANDLE"

    def get_threshold_str(self) -> str:
        return f"{self.interval_seconds}s"


class PopulateHeikinAshiBarStep(AbstractPopulateChartStep):
    def __init__(self, interval_seconds: int):
        super().__init__()
        self.interval_seconds = interval_seconds

    def create_bar_builder(self) -> BaseBarBuilder:
        return HeikinAshiBarBuilder(period=self.interval_seconds)

    def get_bar_type(self) -> str:
        return "HEIKIN_ASHI"

    def get_threshold_str(self) -> str:
        return f"{self.interval_seconds}s"


class PopulateRangeBarStep(AbstractPopulateChartStep):
    def __init__(self, box_size_rate: float):
        super().__init__()
        self.box_size_rate = box_size_rate

    def create_bar_builder(self) -> BaseBarBuilder:
        return RangeBarBuilder(box_size_rate=self.box_size_rate)

    def get_bar_type(self) -> str:
        return "RANGE"

    def get_threshold_str(self) -> str:
        return f"rate_{int(100_000 * self.box_size_rate)}"
