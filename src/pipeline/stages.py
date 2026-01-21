import time
from datetime import timedelta
from logging import getLogger
from typing import Any, Dict

from qate.core.ev_type import EventType
from qate.core.model import Market
from qate.store.market_data_store import partition_market_events
from qate.util.dt_range import DtRange

from .framework import PipelineContext, Stage, StageContext
from .market_data_steps import Context as MarketDataContext
from .trade_derived_steps import Context as TradeDerivedContext

LOG = getLogger(__name__)


class MarketDataStage(Stage):
    def __init__(self, market: Market):
        super().__init__()
        self.market = market

    def before(self, dt_range, pipeline_context: PipelineContext) -> StageContext:
        start_time = time.time()

        loader = pipeline_context.market_data_store.create_loader(
            self.market,
            [EventType.MARKET_ORDER_BOOK, EventType.MARKET_TRADE],
        )
        warmup_dt_range = DtRange(dt_range.start_dt - timedelta(hours=1), dt_range.end_dt)
        all_data = loader.batch_load_mt(warmup_dt_range, max_workers=8)
        order_books, trades = partition_market_events(all_data)
        end_time = time.time()
        LOG.info(
            f"MarketDataStage: Loaded {len(order_books)} order books and {len(trades)} trades"
            f" in {end_time - start_time:.2f} seconds."
        )
        return MarketDataContext(
            warmup_dt_range,
            dt_range,
            self.market,
            trades,
            order_books,
        )


class TradeDerivedStage(Stage):
    def __init__(self, market: Market):
        super().__init__()
        self.market = market

    def before(self, dt_range, pipeline_context: PipelineContext) -> StageContext:
        start_time = time.time()
        warmup_dt_range = DtRange(dt_range.start_dt - timedelta(hours=8), dt_range.end_dt)
        LOG.info(
            f"TradeDerivedStage: Querying trades for {self.market.id}"
            f"from {warmup_dt_range.start_str} to {warmup_dt_range.end_str}"
        )
        store = pipeline_context.parquet_store.create_trade(self.market)
        trades = store.read_trades(warmup_dt_range)

        end_time = time.time()
        LOG.info(f"TradeDerivedStage: Found {len(trades)} trades in {end_time - start_time:.2f} seconds.")
        return TradeDerivedContext(
            warmup_dt_range,
            dt_range,
            self.market,
            trades,
            None,
        )


class LeadLagStage(Stage):
    def __init__(self, market_lead: Market, market_lag: Market):
        super().__init__()
        self.market_lead = market_lead
        self.market_lag = market_lag

    def before(self, dt_range: DtRange, args: Dict[str, Any]):
        self.context.dt_range = dt_range
        self.context.influx_reader = self.influx_reader
        self.context.market_lead = self.market_lead
        self.context.market_lag = self.market_lag


class FeatureStage(Stage):
    """
    Stage for building features from InfluxDB data.
    """

    def __init__(self, market: Market):
        super().__init__()
        self.market = market

    def before(self, dt_range: DtRange, args: Dict[str, Any]):
        self.context.compute_dt_range = DtRange(dt_range.start_dt - timedelta(hours=8), dt_range.end_dt)
        self.context.populate_dt_range = dt_range
        self.context.influx_reader = self.influx_reader
        self.context.market = self.market


class LabelStage(Stage):
    """
    Stage for building labels from PNL data in InfluxDB.

    The PNL data is expected to be populated by a backtest program
    with measurement=PNL and appropriate tags.
    """

    def __init__(self, market: Market, pnl_bucket: str):
        super().__init__()
        self.market = market
        self.pnl_bucket = pnl_bucket

    def before(self, dt_range: DtRange, args: Dict[str, Any]):
        self.context.compute_dt_range = DtRange(dt_range.start_dt - timedelta(hours=8), dt_range.end_dt)
        self.context.populate_dt_range = dt_range
        self.context.market = self.market
        self.context.pnl_bucket = self.pnl_bucket
