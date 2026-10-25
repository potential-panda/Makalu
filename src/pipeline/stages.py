import time
from datetime import timedelta
from logging import getLogger

from qate.core.model import Market
from qate.util.dt_range import DtRange

from labeling.triple_barrier import ts_index
from lead_lag.lead_lag import prepare_returns

from .framework import NoDataForChunk, PipelineContext, Stage, StageContext
from .lead_lag_steps import Context as LeadLagContext
from .order_book_derived_steps import Context as OrderBookDerivedContext
from .return_steps import Context as ReturnContext
from .trade_derived_steps import Context as TradeDerivedContext
from .triple_barrier_steps import Context as TripleBarrierContext

LOG = getLogger(__name__)


def _safe_read(read_fn, *args, label: str = "", **kwargs):
    """Confirmed for real: when no parquet partition exists for a date
    range (an upstream raw-data gap -- e.g. BINANCE/ETH_USDT bronze is
    missing all of 2026-03-29..2026-04-30), qate's read() returns a fully
    empty, columnless DataFrame and then crashes with KeyError('ts')
    trying to filter it by timestamp, rather than returning an empty
    result cleanly. Converts that into NoDataForChunk so
    Pipeline.process() can skip this chunk and continue to the next one,
    instead of the whole populate command crashing on the first gap and
    never reaching any later, available data."""
    try:
        return read_fn(*args, **kwargs)
    except KeyError as e:
        raise NoDataForChunk(f"{label}: no data for this range ({e})") from e


class OrderBookDerivedStage(Stage):
    def __init__(self, market: Market):
        super().__init__()
        self.market = market

    def before(
        self, dt_range: DtRange, pipeline_context: PipelineContext
    ) -> StageContext:
        start_time = time.time()
        compute_dt_range = DtRange(
            dt_range.start_dt - timedelta(hours=1), dt_range.end_dt
        )
        LOG.info(
            f"OrderBookDerivedStage: Querying data for {self.market.id}"
            f"from {compute_dt_range.start_str} to {compute_dt_range.end_str}"
        )
        store = pipeline_context.parquet_store.create_order_book(self.market)
        order_books = _safe_read(
            store.read_order_books,
            compute_dt_range,
            label=f"OrderBook {self.market.id}",
        )

        end_time = time.time()
        LOG.info(
            f"OrderBookDerivedStage: Found {len(order_books)} objects in {end_time - start_time:.2f} seconds."
        )
        return OrderBookDerivedContext(
            dt_range,
            self.market,
            order_books,
        )


class TradeDerivedStage(Stage):
    def __init__(self, market: Market):
        super().__init__()
        self.market = market

    def before(
        self, dt_range: DtRange, pipeline_context: PipelineContext
    ) -> StageContext:
        start_time = time.time()
        compute_dt_range = DtRange(
            dt_range.start_dt - timedelta(hours=1), dt_range.end_dt
        )
        LOG.info(
            f"TradeDerivedStage: Querying data for {self.market.id}"
            f"from {compute_dt_range.start_str} to {compute_dt_range.end_str}"
        )
        store = pipeline_context.parquet_store.create_trade(self.market)
        trades = _safe_read(
            store.read_trades, compute_dt_range, label=f"Trade {self.market.id}"
        )

        end_time = time.time()
        LOG.info(
            f"TradeDerivedStage: Found {len(trades)} objects in {end_time - start_time:.2f} seconds."
        )
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

    def before(
        self, dt_range: DtRange, pipeline_context: PipelineContext
    ) -> StageContext:
        parquet_store = pipeline_context.parquet_store.create_execution_price(
            self.market,
            self.execution_size_str,
        )
        read_range = DtRange(
            dt_range.start_dt - timedelta(hours=8), dt_range.end_dt + timedelta(hours=8)
        )
        df = _safe_read(
            parquet_store.read, read_range, label=f"ExecutionPrice {self.market.id}"
        )

        return TripleBarrierContext(
            dt_range,
            self.market,
            self.execution_size_str,
            df,
        )


class LeadLagStage(Stage):
    def __init__(
        self, market_lead: Market, market_lag: Market, execution_size_str: str
    ):
        super().__init__()
        self.market_lead = market_lead
        self.market_lag = market_lag
        self.execution_size_str = execution_size_str

    def before(
        self, dt_range: DtRange, pipeline_context: PipelineContext
    ) -> StageContext:
        read_range = DtRange(dt_range.start_dt - timedelta(hours=8), dt_range.end_dt)

        df_lead = _safe_read(
            pipeline_context.parquet_store.create_execution_price(
                self.market_lead, self.execution_size_str
            ).read,
            read_range,
            label=f"ExecutionPrice {self.market_lead.id}",
        )
        df_lag = _safe_read(
            pipeline_context.parquet_store.create_execution_price(
                self.market_lag, self.execution_size_str
            ).read,
            read_range,
            label=f"ExecutionPrice {self.market_lag.id}",
        )

        LOG.info(
            f"LeadLagStep: Loaded {len(df_lead)} prices for {self.market_lead.id},"
            f" {len(df_lag)} for {self.market_lag.id}"
        )

        ret_timestamps, lead_returns, lag_returns = prepare_returns(
            df_lead, df_lag, resample_interval="100ms"
        )

        return LeadLagContext(
            dt_range,
            self.market_lead,
            self.market_lag,
            ret_timestamps,
            lead_returns,
            lag_returns,
        )


class ReturnDerivedStage(Stage):
    """Forward returns need FUTURE data beyond dt_range.end -- unlike the
    backward-only pad Hase's Spread/Volatility read (the 8 hours before), this
    pads the read range forward by max_horizon_sec too. Without this, the
    last max_horizon_sec of every populate call would have a permanently
    missing return: ReturnStep's "idempotent, later dates do not affect
    earlier ones" populate model means a later populate call never goes
    back to fill in an earlier gap."""

    def __init__(self, market: Market, execution_size_str: str, max_horizon_sec: int):
        super().__init__()
        self.market = market
        self.execution_size_str = execution_size_str
        self.max_horizon_sec = max_horizon_sec

    def before(
        self, dt_range: DtRange, pipeline_context: PipelineContext
    ) -> StageContext:
        parquet_store = pipeline_context.parquet_store.create_execution_price(
            self.market,
            self.execution_size_str,
        )
        read_range = DtRange(
            dt_range.start_dt - timedelta(hours=1),
            dt_range.end_dt + timedelta(seconds=self.max_horizon_sec),
        )
        df = _safe_read(
            parquet_store.read, read_range, label=f"ExecutionPrice {self.market.id}"
        )
        df = df.sort_index()
        df = ts_index(df)

        return ReturnContext(
            dt_range,
            self.market,
            self.execution_size_str,
            df,
        )
