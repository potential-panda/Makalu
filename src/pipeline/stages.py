import time
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List

from qate.core.ev_type import EventType
from qate.core.model import Market, OrderBook, Trade
from qate.store.influx_reader import InfluxReader
from qate.store.influx_writer import InfluxWriter
from qate.store.msgpack_loader import MarketDataLoader
from qate.util.dt_range import DtRange

from .framework import Stage
from .market_data_steps import MARKET_DATA_BUCKET

LOG = getLogger(__name__)


class MarketDataStage(Stage):
    """
    Stage for processing raw market data files, deserializing them,
    and populating InfluxDB with basic market data metrics.
    """

    def __init__(self, influx_writer: InfluxWriter, market: Market, data_dir: str):
        super().__init__(influx_writer)
        self.market = market
        self.data_dir = Path(data_dir)

    def before(self, dt_range: DtRange, args: Dict[str, Any]):
        """
        Loads market data files based on the date range and populates the context.
        """
        start_time = time.time()
        warmup_dt_range = DtRange(dt_range.start_dt - timedelta(days=1), dt_range.end_dt)

        file_paths = self._get_files_for_range(warmup_dt_range)
        if not file_paths:
            LOG.info(f"MarketDataStage: No data files found for {self.market.id} in range {warmup_dt_range}")
            return

        loader = MarketDataLoader(self.market, [EventType.MARKET_ORDER_BOOK, EventType.MARKET_TRADE])
        all_data = loader.batch_load_mt(file_paths, max_workers=args.get("max_workers", 8))

        order_books, trades = self._parse_data(all_data)

        self.context.order_books = order_books
        self.context.trades = trades
        self.context.compute_dt_range = warmup_dt_range
        self.context.populate_dt_range = warmup_dt_range
        end_time = time.time()
        LOG.info(
            f"MarketDataStage: Loaded {len(order_books)} order books and {len(trades)} trades"
            f" in {end_time - start_time:.2f} seconds."
        )

    def _get_files_for_range(self, dt_range: DtRange) -> List[str]:
        """
        Generates a list of file paths for the given date range.
        """
        file_paths = []
        for date_str in dt_range.days:
            filename = f"{self.market.exchange_name.name}_{self.market.symbol.name}_{date_str}.msgpack"
            filepath = self.data_dir / filename
            if filepath.exists():
                file_paths.append(str(filepath))
            else:
                LOG.warning(f"Data file not found: {filepath}, skipping")
        return file_paths

    def _parse_data(self, data: List[tuple[str, Any]]) -> tuple[List[OrderBook], List[Trade]]:
        """
        Parses the flat list of data objects into separate lists of OrderBook and Trade objects.
        """
        order_books = []
        trades = []
        for event_type, obj in data:
            if event_type == EventType.MARKET_ORDER_BOOK:
                order_books.append(obj)
            elif event_type == EventType.MARKET_TRADE:
                trades.append(obj)

        return order_books, trades


class TradeDerivedStage(Stage):
    """
    Stage for calculating derived metrics from trades stored in InfluxDB.
    """

    def __init__(self, influx_writer: InfluxWriter, market: Market, influx_reader: InfluxReader):
        super().__init__(influx_writer)
        self.market = market
        self.influx_reader = influx_reader
        self.bucket_name = MARKET_DATA_BUCKET

    def before(self, dt_range: DtRange, args: Dict[str, Any]):
        start_time = time.time()
        warmup_dt_range = DtRange(dt_range.start_dt - timedelta(hours=12), dt_range.end_dt)
        LOG.info(
            f"TradeDerivedStage: Querying trades for {self.market.id}"
            f"from {warmup_dt_range.start_str} to {warmup_dt_range.end_str}"
        )
        trades = self.influx_reader.query_trades(self.bucket_name, warmup_dt_range, self.market)
        self.context.trades = trades
        self.context.compute_dt_range = warmup_dt_range
        self.context.populate_dt_range = dt_range
        end_time = time.time()
        LOG.info(f"TradeDerivedStage: Found {len(trades)} trades in {end_time - start_time:.2f} seconds.")


class FeatureStage(Stage):
    """
    Stage for building features from InfluxDB data.
    """

    def __init__(self, influx_writer: InfluxWriter, market: Market, influx_reader: InfluxReader):
        super().__init__(influx_writer)
        self.market = market
        self.influx_reader = influx_reader

    def before(self, dt_range: DtRange, args: Dict[str, Any]):
        self.context.populate_dt_range = dt_range
        self.context.compute_dt_range = dt_range
        self.context.influx_reader = self.influx_reader
        self.context.market = self.market


class LabelStage(Stage):
    """
    Stage for building labels from PNL data in InfluxDB.

    The PNL data is expected to be populated by a backtest program
    with measurement=PNL and appropriate tags.
    """

    def __init__(self, influx_writer: InfluxWriter, market: Market, influx_reader: InfluxReader, pnl_bucket: str):
        super().__init__(influx_writer)
        self.market = market
        self.influx_reader = influx_reader
        self.pnl_bucket = pnl_bucket

    def before(self, dt_range: DtRange, args: Dict[str, Any]):
        self.context.populate_dt_range = dt_range
        self.context.compute_dt_range = dt_range
        self.context.influx_reader = self.influx_reader
        self.context.market = self.market
        self.context.pnl_bucket = self.pnl_bucket
