import time
from datetime import timedelta
from logging import getLogger
from typing import Any, Dict

from qate.core.ev_type import EventType
from qate.core.model import Market
from qate.store.market_data_store import partition_market_events
from qate.util.dt_range import DtRange

from .feature_steps import Context as FeatureContext
from .framework import PipelineContext, Stage, StageContext
from .lead_lag_steps import Context as LeadLagContext
from .market_data_steps import Context as MarketDataContext
from .order_book_derived_steps import Context as OrderBookDerivedContext
from .trade_derived_steps import Context as TradeDerivedContext

LOG = getLogger(__name__)


class MarketDataStage(Stage):
    def __init__(self, market: Market):
        super().__init__()
        self.market = market

    def before(self, dt_range: DtRange, pipeline_context: PipelineContext) -> StageContext:
        start_time = time.time()

        loader = pipeline_context.market_data_store.create_loader(
            self.market,
            [EventType.MARKET_ORDER_BOOK, EventType.MARKET_TRADE],
        )
        all_data = loader.batch_load_mt(dt_range, max_workers=8)
        order_books, trades = partition_market_events(all_data)
        end_time = time.time()
        LOG.info(
            f"MarketDataStage: Loaded {len(order_books)} order books and {len(trades)} trades"
            f" in {end_time - start_time:.2f} seconds."
        )
        return MarketDataContext(
            dt_range,
            self.market,
            trades,
            order_books,
        )


class OrderBookDerivedStage(Stage):
    def __init__(self, market: Market):
        super().__init__()
        self.market = market

    def before(self, dt_range: DtRange, pipeline_context: PipelineContext) -> StageContext:
        start_time = time.time()
        compute_dt_range = DtRange(dt_range.start_dt - timedelta(minutes=5), dt_range.end_dt)
        LOG.info(
            f"OrderBookDerivedStage: Querying data for {self.market.id}"
            f"from {compute_dt_range.start_str} to {compute_dt_range.end_str}"
        )
        store = pipeline_context.parquet_store.create_order_book(self.market)
        order_books = store.read_order_books(compute_dt_range)

        end_time = time.time()
        LOG.info(f"OrderBookDerivedStage: Found {len(order_books)} objects in {end_time - start_time:.2f} seconds.")
        return OrderBookDerivedContext(
            compute_dt_range,
            dt_range,
            self.market,
            order_books,
        )


class TradeDerivedStage(Stage):
    def __init__(self, market: Market):
        super().__init__()
        self.market = market

    def before(self, dt_range: DtRange, pipeline_context: PipelineContext) -> StageContext:
        start_time = time.time()
        compute_dt_range = DtRange(dt_range.start_dt - timedelta(hours=1), dt_range.end_dt)
        LOG.info(
            f"TradeDerivedStage: Querying data for {self.market.id}"
            f"from {compute_dt_range.start_str} to {compute_dt_range.end_str}"
        )
        store = pipeline_context.parquet_store.create_trade(self.market)
        trades = store.read_trades(compute_dt_range)

        end_time = time.time()
        LOG.info(f"TradeDerivedStage: Found {len(trades)} objects in {end_time - start_time:.2f} seconds.")
        return TradeDerivedContext(
            compute_dt_range,
            dt_range,
            self.market,
            trades,
        )


class LeadLagStage(Stage):
    def __init__(self, market_lead: Market, market_lag: Market, execution_size_str: str):
        super().__init__()
        self.market_lead = market_lead
        self.market_lag = market_lag
        self.execution_size_str = execution_size_str

    def before(self, dt_range: DtRange, pipeline_context: PipelineContext) -> StageContext:
        return LeadLagContext(
            dt_range,
            self.market_lead,
            self.market_lag,
            self.execution_size_str,
        )


class FeatureStage(Stage):
    """
    Stage for building features from InfluxDB data.
    """

    def __init__(self, market: Market):
        super().__init__()
        self.market = market

    def before(self, dt_range: DtRange, pipeline_context: PipelineContext) -> StageContext:
        return FeatureContext(
            dt_range,
            DtRange(dt_range.start_dt - timedelta(hours=1), dt_range.end_dt),
            self.market,
        )


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
