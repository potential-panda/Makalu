from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from logging import getLogger
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from komachi import bronze as komachi_bronze
from komachi.layout import data_path as komachi_data_path
from qate.core.ev_type import EventType
from qate.core.model import Market, OrderBook, OrderLevel, Side, Trade
from qate.util.dt_range import DtRange, TimeFormat

from .influx_store import InfluxdbStore
from .manifest import Manifest

LOG = getLogger(__name__)


class TimeSeriesStore(ABC):
    """
    Abstract base class for storing timeseries data in parquet format.

    Data is organized in Hive-style partitioning:
    /root_dir/dataset=$dataset/key=$value.../date=$YYYY-MM-DD/data.parquet

    This is natively understood by PyArrow, DuckDB, Spark, and AWS Athena/Glue.

    Subclasses must implement:
    - get_partition_keys(): Returns dict mapping partition key name -> value
    - get_time_partition_key(ts): Returns time partition key for a given timestamp
    """

    def __init__(self, root_dir: str, dataset: str):
        self.root_dir = root_dir
        self.dataset = dataset
        self._buffer: list[dict] = []
        self._cols: list[str] = None
        self._current_time_key: str | None = None
        self.manifest: Manifest | None = None  # injected by ParquetStore

    @abstractmethod
    def get_partition_keys(self) -> dict[str, str]:
        pass

    def get_time_partition_key(self, ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime(TimeFormat.ISO_DATE)

    def _get_data_dir(self, time_key: str) -> Path:
        path = Path(self.root_dir) / f"dataset={self.dataset}"
        for key, val in self.get_partition_keys().items():
            path = path / f"{key}={val}"
        path = path / f"date={time_key}"
        return path

    def _get_parquet_path(self, time_key: str) -> Path:
        return self._get_data_dir(time_key) / "data.parquet"

    def add(self, ts: float, data: dict) -> None:
        time_key = self.get_time_partition_key(ts)

        if self._current_time_key is None:
            self._current_time_key = time_key

        if time_key != self._current_time_key:
            self.flush()
            self._current_time_key = time_key

        record = {"ts": ts, **data}
        self._buffer.append(record)

    def flush(self) -> None:
        if not self._buffer:
            return
        parquet_path = self._get_parquet_path(self._current_time_key)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        if self._cols:
            df = pd.DataFrame(self._buffer, columns=self._cols)
        else:
            df = pd.DataFrame(self._buffer)
        table = pa.Table.from_pandas(df, preserve_index=False)
        # if parquet_path.exists():
        #     existing_table = pq.read_table(parquet_path)
        #     table = pa.concat_tables([existing_table, table])
        pq.write_table(table, parquet_path, compression="zstd")
        LOG.info(
            f"Flushed {len(self._buffer)} records to {parquet_path} (total: {table.num_rows} records)"
        )
        if self.manifest:
            self.manifest.record_write(
                str(parquet_path), table.num_rows, parquet_path.stat().st_size
            )
        self._buffer.clear()

    def read_a_day(self, date_str) -> pd.DataFrame:
        parquet_path = self._get_parquet_path(date_str)
        LOG.info(f"Reading from {parquet_path}")
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                LOG.info(f"Loaded {len(df)} records from {parquet_path}")
                if self.manifest:
                    self.manifest.record_read(str(parquet_path))
                return df
            except Exception as e:
                LOG.warning(f"Failed to read {parquet_path}: {e}")
        return pd.DataFrame()

    def read(self, date_arg) -> pd.DataFrame:
        # TODO The dt_range.days works because for now we partition the time by day
        # Should get the list of files based on the partition method
        result = None

        if isinstance(date_arg, DtRange):
            dt_range = date_arg
            result = self.read_days(dt_range.days)
            # Filter by exact timestamp range
            if result.empty:
                return result
            result = result[
                (result["ts"] >= dt_range.start_ts) & (result["ts"] <= dt_range.end_ts)
            ]
            LOG.info(
                f"Read {len(result)} records for range{dt_range.start_str} to {dt_range.end_str}"
            )
        elif isinstance(date_arg, list):
            days = date_arg
            result = self.read_days(days)
            LOG.info(f"Read {len(result)} records for days {days}")

        if result is not None:
            result = result.sort_values("ts").reset_index(drop=True)
            return result

        raise ValueError("date_arg must be DtRange or list of date strings")

    def read_days(self, days: list[str]) -> pd.DataFrame:
        # explicitly read multiple specified days (e.g. [5 mondays], [2mondays, 2 tuesdays], etc.)
        result = None
        for day in days:
            df = self.read_a_day(day)
            if df.empty:
                continue
            if result is None:
                result = df
            else:
                import warnings

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", FutureWarning)
                    # Concatenate incrementally to avoid doubling memory usage
                    result = pd.concat([result, df], ignore_index=True)
                    for w in caught:
                        if (
                            "DataFrame concatenation with empty or all-NA entries"
                            in str(w.message)
                        ):
                            all_na_cols = df.columns[df.isna().all()].tolist()
                            message = (
                                f"FutureWarning during concat: day={day}, days={days}, "
                                f"df_rows={len(df)}, df_cols={df.shape[1]}, all_na_cols={all_na_cols}, "
                                f"dtypes={{k: str(v) for k, v in df.dtypes.items()}}"
                            )
                            LOG.warning(message)
        if result is None:
            return pd.DataFrame()
        return result

    def add_dataframe(self, df: pd.DataFrame, ts_column: str = "ts") -> None:
        """
        Add all rows from a DataFrame to the store.

        Args:
            df: DataFrame with a timestamp column and data columns
            ts_column: Name of the timestamp column (default "ts").
                       If the DataFrame has a DatetimeIndex, set ts_column to None
                       and the index will be converted to timestamps.
        """
        if df.empty:
            return

        # Handle DatetimeIndex
        if ts_column is None:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    "DataFrame must have DatetimeIndex when ts_column is None"
                )
            for idx, row in df.iterrows():
                ts = idx.timestamp()
                data = row.to_dict()
                # Remove NaN values
                data = {k: v for k, v in data.items() if pd.notna(v)}
                self.add(ts, data)
        else:
            if ts_column not in df.columns:
                raise ValueError(f"Column '{ts_column}' not found in DataFrame")
            for _, row in df.iterrows():
                ts = row[ts_column]
                data = row.drop(ts_column).to_dict()
                # Remove NaN values
                data = {k: v for k, v in data.items() if pd.notna(v)}
                self.add(ts, data)

    def close(self) -> None:
        self.flush()

    def populate_influxdb(self, influx_store: InfluxdbStore, dt_range: DtRange) -> None:
        from influxdb_client import Point, WritePrecision

        bucket_name = "makalu_" + self.dataset
        df = self.read(dt_range)
        if df.empty:
            LOG.info(f"No data found in parquet for bucket '{bucket_name}'")
            return

        tags = self.get_partition_keys()
        field_cols = [c for c in df.columns if c != "ts"]
        measurement = self.dataset

        BATCH_SIZE = 2048
        writer = influx_store.create_writer(bucket_name, buffer_size=BATCH_SIZE)
        points = []

        for row in df.itertuples(index=False):
            point = Point(measurement).time(int(row.ts * 1000), WritePrecision.MS)
            for name, val in tags.items():
                point.tag(name, val)
            for col in field_cols:
                v = getattr(row, col)
                if v == v:  # skip NaN
                    point.field(col, v)
            points.append(point)
            if len(points) >= BATCH_SIZE:
                writer.write(points)
                points.clear()

        if points:
            writer.write(points)
        writer.close()
        LOG.info(f"Written {len(df)} records to influxdb bucket '{bucket_name}'")

    def data_check(self, dt_range: DtRange) -> dict:
        days = dt_range.days
        missing = []
        sizes = []

        for day in days:
            path = self._get_parquet_path(day)
            if path.exists():
                sizes.append(path.stat().st_size)
            else:
                missing.append(day)

        return {
            "num_days": len(days),
            "num_files": len(sizes),
            "missing_files": missing,
            "avg_size": int(sum(sizes) / len(sizes)) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "min_size": min(sizes) if sizes else 0,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class SingleMarketStore(TimeSeriesStore):
    """
    Directory structure:
    /root_dir/dataset=$dataset/exchange=$exchange/symbol=$symbol/date=$YYYY-MM-DD/data.parquet
    """

    def __init__(
        self, root_dir: str, dataset: str, market: Market, param_id: str | None = None
    ):
        super().__init__(root_dir, dataset)
        self.market = market
        # Named distinctly from subclasses' own `self.param_id` (MeasuredStore,
        # LeadLagStore, ChartStore, ...) so this generic, base-class-level
        # partition key can never collide with -- and silently get reordered
        # by -- a subclass attribute of the same name. Confirmed for real:
        # an earlier version used `self.param_id` here, and VolSpreadStore's
        # own `self.param_id = param_id` (set *after* calling super().__init__)
        # clobbered it, moving param_id earlier in the partition path than
        # what was actually written to disk -- reads then 404'd entirely.
        self._extra_partition_param_id = param_id

    def get_partition_keys(self) -> dict[str, str]:
        keys = {
            "exchange": self.market.exchange_name.name,
            "symbol": self.market.symbol.name,
        }
        if self._extra_partition_param_id is not None:
            keys["param_id"] = self._extra_partition_param_id
        return keys


class BronzeStore(SingleMarketStore):
    """Read-only bronze. Kitakamakura writes it; Komachi says where a market-day is.

    `root_dir` is the datalake's bronze directory, so the datalake root handed
    to Komachi is its parent. A day Komachi does not hold reads as empty, the
    same as a missing file for every other layer.
    """

    def __init__(self, root_dir: str, dataset: str, market: Market):
        super().__init__(root_dir, dataset, market)
        self._datalake_root = Path(root_dir).parent

    def _get_parquet_path(self, time_key: str) -> Path:
        return komachi_data_path(
            self._datalake_root, self.market.id, self.dataset, time_key
        )

    def read_a_day(self, date_str) -> pd.DataFrame:
        try:
            parquet_path = komachi_bronze.file(
                self.market.id, self.dataset, str(date_str), root=self._datalake_root
            )
        except komachi_bronze.MissingData:
            LOG.info(f"No bronze {self.dataset} for {self.market.id} on {date_str}")
            return pd.DataFrame()
        LOG.info(f"Reading from {parquet_path}")
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            LOG.warning(f"Failed to read {parquet_path}: {e}")
            return pd.DataFrame()

    def add(self, ts: float, data: dict) -> None:
        raise TypeError("Bronze is written by Kitakamakura, not Makalu")


class TradeStore(BronzeStore):
    def __init__(self, root_dir: str, market: Market):
        super().__init__(root_dir, "Trade", market)

    def convert_to_trade(self, row) -> Trade:
        return Trade(
            "",
            Side.BUY if row.side == 0 else Side.SELL,
            row.price,
            row.size,
            self.market.symbol,
            self.market.exchange_name,
            row.ts,
        )

    def read_trades(self, dt_range: DtRange) -> list[Trade]:
        trades = []
        # TODO
        df = self.read(dt_range)
        for row in df.itertuples(index=False):
            trades.append(self.convert_to_trade(row))
        return trades

    def load(self, date_str: str):  # MarketDataLoader interface
        results = []
        df = self.read_a_day(date_str)
        for row in df.itertuples(index=False):
            results.append((EventType.MARKET_TRADE, self.convert_to_trade(row)))
        return results


class OrderBookStore(BronzeStore):
    def __init__(self, root_dir: str, market: Market, depth: int = 20):
        super().__init__(root_dir, "OrderBook", market)
        self._depth = depth

    def convert_to_order_book(self, row) -> OrderBook:
        depth = self._depth
        bid_start = 1
        ask_start = 1 + 2 * depth
        bids = [
            OrderLevel(row[bid_start + 2 * i], row[bid_start + 2 * i + 1])
            for i in range(depth)
            if row[bid_start + 2 * i] == row[bid_start + 2 * i]  # NaN check
        ]
        asks = [
            OrderLevel(row[ask_start + 2 * i], row[ask_start + 2 * i + 1])
            for i in range(self._depth)
            if row[ask_start + 2 * i] == row[ask_start + 2 * i]
        ]
        ts = row[0]
        return OrderBook(
            bids,
            asks,
            self.market.symbol,
            self.market.exchange_name,
            ts,
        )

    def read_order_books(self, dt_range: DtRange) -> list[OrderBook]:
        order_books = []
        df = self.read(dt_range)
        for row in df.itertuples(index=False, name=None):
            order_books.append(self.convert_to_order_book(row))
        return order_books

    def load(self, date_str: str):  # MarketDataLoader interface
        results = []
        df = self.read_a_day(date_str)
        for row in df.itertuples(index=False):
            results.append(
                (EventType.MARKET_ORDER_BOOK, self.convert_to_order_book(row))
            )
        return results


class ExecutionPriceStore(SingleMarketStore):
    """
    Directory structure:
    /root_dir/dataset=MarketPrice/exchange=BINANCE/symbol=BTC_USDT/execution_size=10bps/date=$YYYY-MM-DD/data.parquet
    """

    def __init__(self, root_dir: str, market: Market, execution_size_str: str):
        super().__init__(root_dir, "MarketPrice", market)
        self.execution_size_str = execution_size_str

    def get_partition_keys(self) -> dict[str, str]:
        return {
            **super().get_partition_keys(),
            "execution_size": self.execution_size_str,
        }


class LabelStore(SingleMarketStore):
    """
    Directory structure:
    /root_dir/dataset=Label_$label/exchange=BINANCE/symbol=BTC_USDT/param_id=5_5_10/date=$YYYY-MM-DD/data.parquet
    """

    def __init__(self, root_dir: str, label_name: str, market: Market, param_id: str):
        super().__init__(root_dir, "Label_" + label_name, market)
        self.param_id = param_id

    def get_partition_keys(self) -> dict[str, str]:
        return {**super().get_partition_keys(), "param_id": self.param_id}


class ChartStore(SingleMarketStore):
    def __init__(self, root_dir: str, bar_type: str, market: Market, param_id: str):
        super().__init__(root_dir, "Chart_" + bar_type, market)
        self.param_id = param_id

    def get_partition_keys(self) -> dict[str, str]:
        return {**super().get_partition_keys(), "param_id": self.param_id}


class CrossMarketStore(TimeSeriesStore):
    def __init__(
        self,
        root_dir: str,
        dataset: str,
        market_a: Market,
        market_b: Market,
        param_id: str | None = None,
    ):
        super().__init__(root_dir, dataset)
        self.market_a = market_a
        self.market_b = market_b
        # See SingleMarketStore's _extra_partition_param_id comment -- same
        # name-collision reasoning applies here for LeadLagStore.
        self._extra_partition_param_id = param_id

    def get_partition_keys(self) -> dict[str, str]:
        keys = {"market_a": self.market_a.id, "market_b": self.market_b.id}
        if self._extra_partition_param_id is not None:
            keys["param_id"] = self._extra_partition_param_id
        return keys


class LeadLagStore(CrossMarketStore):
    """
    Directory structure:
    /root_dir/dataset=LeadLag/market_a=BINANCE:BTC_USDT/market_b=GMO:BTC_JPY/param_id=$param_id/date=20990101/data.parquet
    """

    def __init__(
        self, root_dir: str, market_lead: Market, market_lag: Market, param_id: str
    ):
        super().__init__(root_dir, "LeadLag", market_lead, market_lag)
        self.param_id = param_id

    def get_partition_keys(self) -> dict[str, str]:
        return {**super().get_partition_keys(), "param_id": self.param_id}


class MeasuredStore(SingleMarketStore):
    """Hase's Spread and Volatility: measured at an execution size, then a window set.

    /root_dir/dataset=$dataset/exchange=$exchange/symbol=$symbol/execution_size=$size/param_id=$param_id/date=$YYYY-MM-DD/data.parquet
    """

    def __init__(
        self,
        root_dir: str,
        dataset: str,
        market: Market,
        execution_size_str: str,
        param_id: str,
    ):
        super().__init__(root_dir, dataset, market)
        self.execution_size_str = execution_size_str
        self.param_id = param_id

    def get_partition_keys(self) -> dict[str, str]:
        return {
            **super().get_partition_keys(),
            "execution_size": self.execution_size_str,
            "param_id": self.param_id,
        }


class SingleMarketStrategyStore(SingleMarketStore):
    """
    `prefix` is the env_name for produciton pnl data (e.g. QASIM_TARIQ)
    `dataset` can be "pnl", "order" (include slippage, latency, etc)
    """

    def __init__(
        self,
        root_dir: str,
        dataset: str,
        market: Market,
        strategy_name: str,
        param_id: str,
    ):
        super().__init__(root_dir, dataset, market)
        self.strategy_name = strategy_name
        self.param_id = param_id

    def get_partition_keys(self) -> dict[str, str]:
        return {
            "strategy": self.strategy_name,
            **super().get_partition_keys(),
            "param_id": self.param_id,
        }


# TODO should keep this file as a parquet store util
# shoudl not include the layer here
# Please move below to somewhere
class Layer(Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    FEATURE = "feature"
    RESAERCH = "research"


class ParquetStore:
    def __init__(self, root_dir: str, manifest: Manifest | None = None):
        from pathlib import Path

        self.root_dir = Path(root_dir)
        self.manifest = manifest
        self.layer_dir = {
            Layer.BRONZE: self.root_dir / Layer.BRONZE.value,
            Layer.SILVER: self.root_dir / Layer.SILVER.value,
            Layer.GOLD: self.root_dir / Layer.GOLD.value,
            Layer.FEATURE: self.root_dir / Layer.FEATURE.value,
            Layer.RESAERCH: self.root_dir / Layer.RESAERCH.value,
        }

    def _m(self, store: TimeSeriesStore) -> TimeSeriesStore:
        """Inject manifest into a store instance."""
        store.manifest = self.manifest
        return store

    # Bronze - only 2
    def create_trade(self, market: Market):
        return self._m(TradeStore(self.layer_dir[Layer.BRONZE], market))

    def create_order_book(self, market: Market):
        return self._m(OrderBookStore(self.layer_dir[Layer.BRONZE], market))

    # SILVER
    def create_single_market(
        self,
        dataset: str,
        market: Market,
        layer: Layer = Layer.SILVER,
        param_id: str | None = None,
    ):
        return self._m(
            SingleMarketStore(self.layer_dir[layer], dataset, market, param_id)
        )

    def create_cross_market(
        self,
        dataset: str,
        market_a: Market,
        market_b: Market,
        layer: Layer = Layer.GOLD,
        param_id: str | None = None,
    ):
        return self._m(
            CrossMarketStore(
                self.layer_dir[layer], dataset, market_a, market_b, param_id
            )
        )

    def create_execution_price(self, market: Market, execution_size_str: str):
        return self._m(
            ExecutionPriceStore(
                self.layer_dir[Layer.SILVER], market, execution_size_str
            )
        )

    # GOLD
    def create_chart(self, bar_type: str, market: Market, param_id: str):
        return self._m(
            ChartStore(self.layer_dir[Layer.GOLD], bar_type, market, param_id)
        )

    def create_lead_lag(self, market_lead: Market, market_lag: Market, param_id: str):
        return self._m(
            LeadLagStore(self.layer_dir[Layer.GOLD], market_lead, market_lag, param_id)
        )

    def create_measured(
        self, dataset: str, market: Market, execution_size_str: str, param_id: str
    ):
        """Hase's Spread or Volatility, read only: Hase writes them."""
        return self._m(
            MeasuredStore(
                self.layer_dir[Layer.GOLD],
                dataset,
                market,
                execution_size_str,
                param_id,
            )
        )

    def create_feature(self, market: Market, target: str):
        return self._m(
            SingleMarketStore(
                self.layer_dir[Layer.FEATURE], f"Feature_{target}", market
            )
        )

    # Ephemeral
    def create_label(self, label_name: str, market: Market, param_id: str):
        return self._m(
            LabelStore(self.layer_dir[Layer.RESAERCH], label_name, market, param_id)
        )

    def create_pnl(
        self,
        market: Market,
        strategy_name: str,
        param_id: str,
        prefix: str = "Backtest",
    ):
        return self._m(
            SingleMarketStrategyStore(
                self.layer_dir[Layer.RESAERCH],
                f"{prefix}_Pnl",
                market,
                strategy_name,
                param_id,
            )
        )

    def create_backtest_bar(self, market: Market, strategy_name: str, param_id: str):
        return self._m(
            SingleMarketStrategyStore(
                self.layer_dir[Layer.RESAERCH],
                "Backtest_Bar",
                market,
                strategy_name,
                param_id,
            )
        )
