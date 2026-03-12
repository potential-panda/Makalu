import time
from datetime import timedelta
from logging import getLogger

from qate.core.ev_type import EventType
from qate.core.model import Market
from qate.store.market_data_store import partition_market_events
from qate.util.dt_range import DtRange, TimeFormat

from labeling.triple_barrier import ts_index
from lead_lag.lead_lag import prepare_returns

from .execution_price_derived_steps import Context as ExecutionPriceDerivedContext
from .framework import PipelineContext, Stage, StageContext
from .lead_lag_steps import Context as LeadLagContext
from .market_data_steps import Context as MarketDataContext
from .order_book_derived_steps import Context as OrderBookDerivedContext
from .trade_derived_steps import Context as TradeDerivedContext
from .triple_barrier_steps import Context as TripleBarrierContext

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
        # TODO
        # remove this after fully migrate the date format in market data files
        dt_range = DtRange(dt_range._start_dt, dt_range._end_dt, TimeFormat.YYYYMMDD)

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
        compute_dt_range = DtRange(dt_range.start_dt - timedelta(hours=1), dt_range.end_dt)
        LOG.info(
            f"OrderBookDerivedStage: Querying data for {self.market.id}"
            f"from {compute_dt_range.start_str} to {compute_dt_range.end_str}"
        )
        store = pipeline_context.parquet_store.create_order_book(self.market)
        order_books = store.read_order_books(compute_dt_range)

        end_time = time.time()
        LOG.info(f"OrderBookDerivedStage: Found {len(order_books)} objects in {end_time - start_time:.2f} seconds.")
        return OrderBookDerivedContext(
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
            dt_range,
            self.market,
            trades,
        )


class LabelingStage(Stage):
    def __init__(self, market: Market, execution_size_str: str):
        super().__init__()
        self.market = market
        self.execution_size_str = execution_size_str

    def before(self, dt_range: DtRange, pipeline_context: PipelineContext) -> StageContext:
        parquet_store = pipeline_context.parquet_store.create_execution_price(
            self.market,
            self.execution_size_str,
        )
        read_range = DtRange(dt_range.start_dt - timedelta(hours=8), dt_range.end_dt + timedelta(hours=8))
        df = parquet_store.read(read_range)

        return TripleBarrierContext(
            dt_range,
            self.market,
            self.execution_size_str,
            df,
        )


class LeadLagStage(Stage):
    def __init__(self, market_lead: Market, market_lag: Market, execution_size_str: str):
        super().__init__()
        self.market_lead = market_lead
        self.market_lag = market_lag
        self.execution_size_str = execution_size_str

    def before(self, dt_range: DtRange, pipeline_context: PipelineContext) -> StageContext:
        read_range = DtRange(dt_range.start_dt - timedelta(hours=8), dt_range.end_dt)

        df_lead = pipeline_context.parquet_store.create_execution_price(
            self.market_lead,
            self.execution_size_str,
        ).read(read_range)
        df_lag = pipeline_context.parquet_store.create_execution_price(
            self.market_lag,
            self.execution_size_str,
        ).read(read_range)

        LOG.info(
            f"LeadLagStep: Loaded {len(df_lead)} prices for {self.market_lead.id},"
            f" {len(df_lag)} for {self.market_lag.id}"
        )

        ret_timestamps, lead_returns, lag_returns = prepare_returns(df_lead, df_lag, resample_interval="100ms")

        return LeadLagContext(
            dt_range,
            self.market_lead,
            self.market_lag,
            ret_timestamps,
            lead_returns,
            lag_returns,
        )


class ExecutionPriceDerivedStage(Stage):
    def __init__(self, market: Market, execution_size_str: str):
        super().__init__()
        self.market = market
        self.execution_size_str = execution_size_str

    def before(self, dt_range: DtRange, pipeline_context: PipelineContext) -> StageContext:
        parquet_store = pipeline_context.parquet_store.create_execution_price(
            self.market,
            self.execution_size_str,
        )
        read_range = DtRange(dt_range.start_dt - timedelta(hours=8), dt_range.end_dt)
        df = parquet_store.read(read_range)
        df = df.sort_index()
        df = ts_index(df)

        return ExecutionPriceDerivedContext(
            dt_range,
            self.market,
            self.execution_size_str,
            df,
        )
