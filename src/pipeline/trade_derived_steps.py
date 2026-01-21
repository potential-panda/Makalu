import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

from influxdb_client import Point, WritePrecision
from qate.core.model import Market, OrderBook, Trade
from qate.trading.aggression import TradeAggressionBuilder
from qate.trading.chart import (
    BaseBarBuilder,
    CandleBarBuilder,
    HeikinAshiBarBuilder,
    RangeBarBuilder,
)

from .converter import convert_to_trade
from .framework import PipelineContext, StageContext, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constants
TRADE_DERIVED_BUCKET = "trade_derived"


# NOTE Want to make it clear to see what are populated in each step
# Should explicitly write each field even it looks verbose


@dataclass
class Context(StageContext):
    market: Market
    trades: list[Trade]
    order_books: list[OrderBook]


class PopulateAggressionStep(Step):
    def __init__(self, windows_sec: List[int]):
        super().__init__()
        self.windows_sec = windows_sec

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.trades_df) * len(self.windows_sec))

        influx_writer = pipeline_context.influx_store.create_writer(TRADE_DERIVED_BUCKET, no_retention=True)
        parquet_store = pipeline_context.parquet_store.create_single_market("TradeAggression", stage_context.market)
        builder = TradeAggressionBuilder(windows_sec=self.windows_sec)
        for trade_row in stage_context.trades_df.itertuples(index=False):
            if not stage_context.compute_dt_range.is_within(trade_row.ts):
                continue
            trade = convert_to_trade(trade_row, stage_context.market)
            builder.on_trade(trade)
            if not stage_context.populate_dt_range.is_within(trade.get_ts()):
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
                    .tag("exchange_name", stage_context.market.exchange_name.name)
                    .tag("symbol", stage_context.market.symbol.name)
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

        influx_writer.close()
        parquet_store.close()
        LOG.info(f"PopulateAggressionStep: Processed {progress.cnt} objects")


class AbstractPopulateChartStep(Step):
    @abstractmethod
    def create_bar_builder(self) -> BaseBarBuilder:
        pass

    @abstractmethod
    def get_bar_type(self) -> str:
        pass

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.trades_df))

        influx_writer = pipeline_context.influx_store.create_writer(TRADE_DERIVED_BUCKET, no_retention=True)
        parquet_store = pipeline_context.parquet_store.create_chart(stage_context.market, self.get_bar_type())
        builder = self.create_bar_builder()
        for trade_row in stage_context.trades_df.itertuples(index=False):
            if not stage_context.populate_dt_range.is_within(trade_row.ts):
                continue
            trade = convert_to_trade(trade_row, stage_context.market)
            bar = builder.on_update(trade)
            if not bar:
                continue
            if not stage_context.populate_dt_range.is_within(trade.get_ts()):
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
                .tag("exchange_name", stage_context.market.exchange_name.name)
                .tag("symbol", stage_context.market.symbol.name)
                .tag("interval", self.get_threshold_str())
                .tag("bar_type", self.get_bar_type())
                .time(int(bar.get_ts() * 1000), WritePrecision.MS)
            )
            for k, v in fields.items():
                point.field(k, v)

            influx_writer.add(point)
            parquet_store.add(trade.get_ts(), fields)
            progress.update()

        influx_writer.close()
        parquet_store.close()
        LOG.info(f"{self.__class__.__name__}: Created {progress.cnt} bars")

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
