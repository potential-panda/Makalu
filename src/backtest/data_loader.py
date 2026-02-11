"""
Data loading utilities for backtesting.

Loads market data (order books, trades) from parquet stores.
"""

import logging
from dataclasses import dataclass

from qate.core.model import Market, OrderBook, Trade
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

LOG = logging.getLogger(__name__)


@dataclass
class BacktestDataConfig:
    """Configuration for loading backtest data."""

    dt_range: DtRange
    market_binance: Market
    market_gmo: Market
    parquet_root: str


class BacktestDataLoader:
    """Loads market data from parquet stores for backtesting."""

    def __init__(self, config: BacktestDataConfig):
        self.config = config
        self.parquet_store = ParquetStore(config.parquet_root)

    def load_binance_orderbooks(self) -> list[OrderBook]:
        """Load BINANCE order books for the configured date range."""
        store = self.parquet_store.create_order_book(self.config.market_binance)
        orderbooks = store.read_order_books(self.config.dt_range)
        LOG.info(f"Loaded {len(orderbooks)} BINANCE order books")
        return orderbooks

    def load_binance_trades(self) -> list[Trade]:
        """Load BINANCE trades for the configured date range."""
        store = self.parquet_store.create_trade(self.config.market_binance)
        trades = store.read_trades(self.config.dt_range)
        LOG.info(f"Loaded {len(trades)} BINANCE trades")
        return trades

    def load_gmo_orderbooks(self) -> list[OrderBook]:
        """Load GMO order books for the configured date range."""
        store = self.parquet_store.create_order_book(self.config.market_gmo)
        orderbooks = store.read_order_books(self.config.dt_range)
        LOG.info(f"Loaded {len(orderbooks)} GMO order books")
        return orderbooks
