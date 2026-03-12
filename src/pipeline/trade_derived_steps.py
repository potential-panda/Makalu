import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

from qate.core.model import Market, Trade
from qate.trading.aggression import TradeAggressionBuilder
from qate.trading.chart import (
    BaseBarBuilder,
    CandleBarBuilder,
    HeikinAshiBarBuilder,
    RangeBarBuilder,
)

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


class PopulateAggressionStep(Step):
    def __init__(self, windows_sec: List[int]):
        super().__init__()
        self.windows_sec = windows_sec

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        parquet_store = pipeline_context.parquet_store.create_single_market(
            "TradeAggression", stage_context.market, layer="silver"
        )
        count = 0
        for ts, write_fields in self._populate(pipeline_context, stage_context):
            parquet_store.add(ts, write_fields)
            count += 1
        parquet_store.close()
        LOG.info(f"PopulateAggressionStep: Wrote {count} records to Parquet store")

    def _populate(self, pipeline_context: PipelineContext, stage_context: Context):
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.trades))
        builder = TradeAggressionBuilder(windows_sec=self.windows_sec)
        for trade in stage_context.trades:
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

                for k, v in fields.items():
                    write_fields[f"{k}_{aggression.window_sec}s"] = v

            yield (trade.get_ts(), write_fields)
            progress.update()

        LOG.info(f"PopulateAggressionStep: Processed {progress.cnt} objects")


class AbstractPopulateChartStep(Step):
    @abstractmethod
    def create_bar_builder(self) -> BaseBarBuilder:
        pass

    @abstractmethod
    def get_bar_type(self) -> str:
        pass

    @abstractmethod
    def get_threshold_str(self) -> str:
        pass

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        parquet_store = pipeline_context.parquet_store.create_chart(
            self.get_bar_type(), stage_context.market, self.get_threshold_str()
        )
        count = 0
        for ts, fields in self._populate(pipeline_context, stage_context):
            parquet_store.add(ts, fields)
            count += 1
        parquet_store.close()
        LOG.info(f"{self.__class__.__name__}: Wrote {count} bars to Parquet store")

    def _populate(self, pipeline_context: PipelineContext, stage_context: Context):
        """Generator that yields (ts, fields) tuples."""
        progress = StepProgressLogger(self.__class__.__name__, len(stage_context.trades))
        builder = self.create_bar_builder()
        for trade in stage_context.trades:
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
            yield (bar.get_ts(), fields)
            progress.update()

        LOG.info(f"{self.__class__.__name__}: Created {progress.cnt} bars")


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
