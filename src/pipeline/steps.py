import logging
from abc import abstractmethod
from typing import Any, Dict, List

import pandas as pd
from qate.core.symbol import get_symbol_def
from qate.store.influx_writer import InfluxWriter
from qate.trading import intent
from qate.trading.aggression import TradeAggressionBuilder
from qate.trading.book_change import BookChangeBuilder
from qate.trading.chart import (
    BaseBarBuilder,
    CandleBarBuilder,
    HeikinAshiBarBuilder,
    RangeBarBuilder,
)

from .converter import DataPointConverter
from .framework import Context, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

# Bucket name constants
MARKET_DATA_BUCKET = "market_data"
TRADE_DERIVED_BUCKET = "trade_derived"

# #####
# MarketDataStage Steps


class PopulateTradeStep(Step):
    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Converts Trade objects in context to InfluxDB points and writes them."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.trades))
        for trade in context.trades:
            if not context.populate_dt_range.is_within(trade.get_ts()):
                continue
            point = DataPointConverter.convert_trade_to_influx_point(trade)
            influx_writer.add(point)
            progress.update()
        LOG.info(f"PopulateTradeStep: Processed {progress.cnt} trades.")


class PopulateMarketPriceStep(Step):
    """Populates InfluxDB with Market Price data."""

    def __init__(self, target_size: float = 0.01):
        super().__init__()
        self.target_size = target_size

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Calculates market price from OrderBook objects and writes to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.order_books))
        symbol_def = get_symbol_def(context.order_books[0].symbol)
        execution_size_str = symbol_def.fmt_size(self.target_size)
        for orderbook in context.order_books:
            if not context.populate_dt_range.is_within(orderbook.get_ts()):
                continue
            market_price = orderbook.market_price(self.target_size)
            if not market_price:
                continue
            point = DataPointConverter.convert_market_price_to_influx_point(
                orderbook,
                market_price,
                execution_size_str,
            )
            influx_writer.add(point)
            progress.update()
        LOG.info(f"PopulateMarketPriceStep: Processed {progress.cnt} market prices.")


class PopulateOrderBookStep(Step):
    """Populates InfluxDB with basic OrderBook data."""

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Extracts basic metrics from OrderBook objects and writes them to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.order_books))
        for orderbook in context.order_books:
            if not context.populate_dt_range.is_within(orderbook.get_ts()):
                continue
            point = DataPointConverter.convert_order_book_to_influx_point(orderbook)
            influx_writer.add(point)
            progress.update()
        LOG.info(f"PopulateOrderBookStep: Processed {progress.cnt} order books.")


class PopulateOrderImbalanceStep(Step):
    """Populates InfluxDB with OrderBook Imbalance data."""

    def __init__(
        self,
        volume_imbalance_k: List[int],
        weighted_imbalance_k: List[int],
        depth_imbalance_rate: List[float],
        microprice_k: List[int],
    ):
        super().__init__()
        self.volume_imbalance_k = volume_imbalance_k
        self.weighted_imbalance_k = weighted_imbalance_k
        self.depth_imbalance_rate = depth_imbalance_rate
        self.microprice_k = microprice_k

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Calculates and writes order book imbalance metrics to InfluxDB."""
        progress = StepProgressLogger(
            self.__class__.__name__,
            len(context.order_books)
            * (
                len(self.volume_imbalance_k)
                + len(self.weighted_imbalance_k)
                + len(self.depth_imbalance_rate)
                + len(self.microprice_k)
            ),
        )
        for orderbook in context.order_books:
            if not context.populate_dt_range.is_within(orderbook.get_ts()):
                continue
            for k in self.volume_imbalance_k:
                value = intent.compute_volume_imbalance(orderbook, k)
                point = DataPointConverter.convert_volume_imbalance_to_influx_point(orderbook, k, value)
                influx_writer.add(point)
                progress.update()

            for k in self.weighted_imbalance_k:
                value = intent.compute_weighted_imbalance(orderbook, k)
                point = DataPointConverter.convert_weighted_imbalance_to_influx_point(orderbook, k, value)
                influx_writer.add(point)
                progress.update()

            for rate in self.depth_imbalance_rate:
                value = intent.compute_depth_imbalance(orderbook, rate)
                point = DataPointConverter.convert_depth_imbalance_to_influx_point(orderbook, rate, value)
                influx_writer.add(point)
                progress.update()

            for k in self.microprice_k:
                value = intent.compute_microprice(orderbook, k)
                if value is not None:
                    point = DataPointConverter.convert_microprice_to_influx_point(orderbook, k, value)
                    influx_writer.add(point)
                    progress.update()
        LOG.info(f"PopulateOrderImbalanceStep: Processed {progress.cnt} order book imbalances.")


class PopulateBookChangeStep(Step):
    """Populates InfluxDB with OrderBook Book Change data for various 'k' levels."""

    def __init__(self, k_levels: List[int]):
        super().__init__()
        self.k_levels = k_levels
        self.builders = [BookChangeBuilder(k) for k in self.k_levels]

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Calculates book changes from OrderBook objects for different k-levels and writes to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.order_books) * len(self.k_levels))
        for orderbook in context.order_books:
            if not context.compute_dt_range.is_within(orderbook.get_ts()):
                continue
            for builder in self.builders:
                book_change = builder.on_snapshot(orderbook)
                if not book_change:
                    continue
                if not context.populate_dt_range.is_within(orderbook.get_ts()):
                    continue
                point = DataPointConverter.convert_book_change_to_influx_point(orderbook, book_change)
                influx_writer.add(point)
                progress.update()
        LOG.info(f"PopulateBookChangeStep: Processed {progress.cnt} book changes.")


# #####
# TradeDerivedStage Steps


class PopulateAggressionStep(Step):
    """Populates InfluxDB with Trade Aggression data."""

    def __init__(self, windows_sec: List[int]):
        super().__init__()
        self.windows_sec = windows_sec
        self.builder = TradeAggressionBuilder(windows_sec=self.windows_sec)

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Calculates and writes trade aggression metrics to InfluxDB."""
        progress = StepProgressLogger(self.__class__.__name__, len(context.trades) * len(self.windows_sec))
        for trade in context.trades:
            if not context.compute_dt_range.is_within(trade.get_ts()):
                continue
            self.builder.on_trade(trade)
            if not context.populate_dt_range.is_within(trade.get_ts()):
                continue
            aggressions = self.builder.get_trade_aggression(trade.get_ts())
            for aggression in aggressions:
                point = DataPointConverter.convert_trade_aggression_to_influx_point(trade.market, aggression)
                influx_writer.add(point)
                progress.update()
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
    def get_bar_type_tag(self) -> str:
        """Return the tag for the bar type."""
        pass

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """
        Builds bars from trades in context and writes them to InfluxDB.
        """
        builder = self.create_bar_builder()
        progress = StepProgressLogger(self.__class__.__name__, len(context.trades))
        for trade in context.trades:
            if not context.populate_dt_range.is_within(trade.get_ts()):
                continue
            bar = builder.on_update(trade)
            if bar:
                point = DataPointConverter.convert_bar_to_influx_point(
                    bar,
                    symbol=trade.market.symbol.name,
                    exchange=trade.market.exchange_name.name,
                    interval=self.get_interval_tag(),
                    bar_type=self.get_bar_type_tag(),
                )
                influx_writer.add(point)
                progress.update()

        LOG.info(f"{self.__class__.__name__}: Created {progress.cnt} bars.")

    def get_interval_tag(self) -> str:
        """
        Returns a string representation of the interval for InfluxDB tagging.
        This can be overridden by subclasses if a different format is needed.
        """
        return "trade_based"


class PopulateCandleBarStep(AbstractPopulateChartStep):
    def __init__(self, interval_seconds: int):
        super().__init__()
        self.interval_seconds = interval_seconds

    def create_bar_builder(self) -> BaseBarBuilder:
        return CandleBarBuilder(period=self.interval_seconds)

    def get_bar_type_tag(self) -> str:
        return "CANDLE"

    def get_interval_tag(self) -> str:
        return f"{self.interval_seconds}s"


class PopulateHeikinAshiBarStep(AbstractPopulateChartStep):
    def __init__(self, interval_seconds: int):
        super().__init__()
        self.interval_seconds = interval_seconds

    def create_bar_builder(self) -> BaseBarBuilder:
        return HeikinAshiBarBuilder(period=self.interval_seconds)

    def get_bar_type_tag(self) -> str:
        return "HEIKIN_ASHI"

    def get_interval_tag(self) -> str:
        return f"{self.interval_seconds}s"


class PopulateRangeBarStep(AbstractPopulateChartStep):
    def __init__(self, box_size_rate: float):
        super().__init__()
        self.box_size_rate = box_size_rate

    def create_bar_builder(self) -> BaseBarBuilder:
        return RangeBarBuilder(box_size_rate=self.box_size_rate)

    def get_bar_type_tag(self) -> str:
        return "RANGE"

    def get_interval_tag(self) -> str:
        return f"rate_{int(100_000 * self.box_size_rate)}"


# #####
# FeatureStage Steps


class QueryTradeAggressionStep(Step):
    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Queries trade aggression data from InfluxDB and stores it in context."""
        ta_df = context.influx_reader.query_trade_aggression(
            TRADE_DERIVED_BUCKET, context.compute_dt_range, context.market
        )

        if ta_df.empty:
            context.trade_aggression_df = ta_df
            LOG.warning("QueryTradeAggressionStep: No trade aggression data found.")
            return

        # Pivot by window_sec tag to get separate columns for each window
        # Reset index to access 'time' as a column
        ta_df_reset = ta_df.reset_index()

        # Check if window_sec column exists
        if "window_sec" in ta_df_reset.columns:
            # Get list of field columns (exclude time, window_sec, and any other tags)
            exclude_cols = {"time", "window_sec"}
            field_cols = [col for col in ta_df_reset.columns if col not in exclude_cols]

            # Create separate dataframes for each field, pivoted by window_sec
            pivoted_dfs = []
            for field in field_cols:
                pivoted = ta_df_reset.pivot(index="time", columns="window_sec", values=field)
                # Rename columns to include window suffix (e.g., imbalance_qty_1, imbalance_qty_3)
                pivoted.columns = [f"{field}_{int(col)}s" for col in pivoted.columns]
                pivoted_dfs.append(pivoted)

            # Join all pivoted dataframes
            if pivoted_dfs:
                context.trade_aggression_df = pivoted_dfs[0]
                for df in pivoted_dfs[1:]:
                    context.trade_aggression_df = context.trade_aggression_df.join(df)
            else:
                context.trade_aggression_df = ta_df
        else:
            # No window_sec tag, use as is
            context.trade_aggression_df = ta_df.set_index("time") if "time" in ta_df_reset.columns else ta_df

        LOG.info(f"QueryTradeAggressionStep: Queried {len(context.trade_aggression_df)} trade aggression records.")


class QueryOrderBookStateStep(Step):
    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Queries order book state data from InfluxDB and stores it in context."""
        # Query base orderbook data (mid, spread, spread_bps, best_bid, best_ask)
        ob_base_df = context.influx_reader.query_order_book_state(
            MARKET_DATA_BUCKET, context.compute_dt_range, context.market
        )

        # Query volume imbalance
        vol_imb_df = context.influx_reader.query_volume_imbalance(
            MARKET_DATA_BUCKET, context.compute_dt_range, context.market
        )
        vol_imb_df = self._pivot_by_tag(vol_imb_df, "k", "vol_imb_k")

        # Query weighted imbalance
        weighted_imb_df = context.influx_reader.query_weighted_imbalance(
            MARKET_DATA_BUCKET, context.compute_dt_range, context.market
        )
        weighted_imb_df = self._pivot_by_tag(weighted_imb_df, "k", "weighted_imb_k")

        # Query depth imbalance
        depth_imb_df = context.influx_reader.query_depth_imbalance(
            MARKET_DATA_BUCKET, context.compute_dt_range, context.market
        )
        depth_imb_df = self._pivot_by_tag(depth_imb_df, "bps", "depth_imb_bps")

        # Query microprice
        microprice_df = context.influx_reader.query_microprice(
            MARKET_DATA_BUCKET, context.compute_dt_range, context.market
        )
        microprice_df = self._pivot_by_tag(microprice_df, "k", "microprice_k")

        # Merge all dataframes
        dfs = [ob_base_df, vol_imb_df, weighted_imb_df, depth_imb_df, microprice_df]
        dfs = [df for df in dfs if not df.empty]

        if dfs:
            context.order_book_state_df = dfs[0]
            for df in dfs[1:]:
                context.order_book_state_df = context.order_book_state_df.join(df, how="outer")
        else:
            context.order_book_state_df = pd.DataFrame()

        LOG.info(f"QueryOrderBookStateStep: Queried {len(context.order_book_state_df)} order book state records.")

    def _pivot_by_tag(self, df: pd.DataFrame, tag_col: str, prefix: str) -> pd.DataFrame:
        """Pivots a dataframe by tag column to create separate columns for each tag value."""
        if df.empty:
            return df

        # Reset index to make 'time' a column for pivoting
        df_reset = df.reset_index()

        # Pivot: index=time, columns=tag_col, values='value'
        pivoted = df_reset.pivot(index="time", columns=tag_col, values="value")

        # Rename columns with prefix
        pivoted.columns = [f"{prefix}{col}" for col in pivoted.columns]

        return pivoted


class QueryBookChangeStep(Step):
    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Queries book change data from InfluxDB and stores it in context."""
        bc_df = context.influx_reader.query_book_change(MARKET_DATA_BUCKET, context.compute_dt_range, context.market)

        if bc_df.empty:
            context.book_change_df = bc_df
            LOG.warning("QueryBookChangeStep: No book change data found.")
            return

        # Pivot by k tag to get separate columns for each k level
        # Reset index to access 'time' as a column
        bc_df_reset = bc_df.reset_index()

        # Check if k column exists
        if "k" in bc_df_reset.columns:
            # Get list of field columns (exclude time, k, and any other tags)
            exclude_cols = {"time", "k"}
            field_cols = [col for col in bc_df_reset.columns if col not in exclude_cols]

            # Create separate dataframes for each field, pivoted by k
            pivoted_dfs = []
            for field in field_cols:
                pivoted = bc_df_reset.pivot(index="time", columns="k", values=field)
                # Rename columns to include k suffix (e.g., net_depth_bid_k3, net_depth_bid_k5)
                pivoted.columns = [f"{field}_k{int(col)}" for col in pivoted.columns]
                pivoted_dfs.append(pivoted)

            # Join all pivoted dataframes
            if pivoted_dfs:
                context.book_change_df = pivoted_dfs[0]
                for df in pivoted_dfs[1:]:
                    context.book_change_df = context.book_change_df.join(df)
            else:
                context.book_change_df = bc_df
        else:
            # No k tag, use as is
            context.book_change_df = bc_df.set_index("time") if "time" in bc_df_reset.columns else bc_df

        LOG.info(f"QueryBookChangeStep: Queried {len(context.book_change_df)} book change records.")


class AlignAndBuildFeaturesStep(Step):
    """
    Aligns data from different sources and builds a comprehensive feature DataFrame.

    This step:
    1. Aligns orderbook state, trade aggression, and book change data
    2. Computes derived features (microprice_offset_bps, etc.)
    3. Computes rolling/EMA aggregates for orderbook features
    4. Computes cross-window features for trade aggression
    5. Computes interaction features
    """

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Aligns data from different sources and builds a single feature DataFrame."""
        # Ensure all DataFrames are sorted by index (time)
        dfs = [context.trade_aggression_df, context.order_book_state_df, context.book_change_df]
        for df in dfs:
            if df is not None and not df.empty and not df.index.is_monotonic_increasing:
                df.sort_index(inplace=True)

        # Filter out empty dataframes
        dfs = [df for df in dfs if df is not None and not df.empty]

        if not dfs:
            LOG.warning("AlignAndBuildFeaturesStep: No data available to build features.")
            return

        # Start with the first DataFrame
        feature_df = dfs[0].copy()

        # Merge the remaining DataFrames using join (since they're already time-indexed)
        for df in dfs[1:]:
            feature_df = feature_df.join(df, how="outer", rsuffix="_dup")
            # Drop duplicate columns if any
            feature_df = feature_df.loc[:, ~feature_df.columns.str.endswith("_dup")]

        # Sort by index after merging
        feature_df.sort_index(inplace=True)

        # 1. Compute microprice_offset_bps for each microprice_k
        feature_df = self._compute_microprice_offset_bps(feature_df)

        # 2. Compute VWAP offset in bps for trade aggression windows
        feature_df = self._compute_vwap_offset_bps(feature_df)

        # 3. Compute rolling/EMA aggregates for orderbook features (1s, 3s, 5s)
        feature_df = self._compute_rolling_aggregates(feature_df)

        # 4. Compute cross-window features for trade aggression
        feature_df = self._compute_cross_window_features(feature_df)

        # 5. Compute interaction features
        feature_df = self._compute_interaction_features(feature_df)

        # Filter the DataFrame to the populate_dt_range
        feature_df = feature_df[
            (feature_df.index >= context.populate_dt_range.start_dt)
            & (feature_df.index <= context.populate_dt_range.end_dt)
        ]

        # Drop rows with too many NaN values (keep rows with at least 50% valid data)
        threshold = len(feature_df.columns) * 0.5
        feature_df = feature_df.dropna(thresh=threshold)

        # Save to parquet
        output_dir = args.get("feature_output_dir", "/tmp")
        file_path = f"{output_dir}/{context.market.id}_features.parquet"
        DataPointConverter.convert_dataframe_to_parquet(feature_df, file_path)
        LOG.info(
            f"AlignAndBuildFeaturesStep: Built and saved {len(feature_df)} features "
            f"with {len(feature_df.columns)} columns to {file_path}."
        )

    def _compute_microprice_offset_bps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes microprice_offset_bps = (microprice - mid) / mid * 1e4
        for each microprice_k column.
        """
        if "mid" not in df.columns:
            LOG.warning("'mid' column not found, skipping microprice_offset_bps computation.")
            return df

        microprice_cols = [col for col in df.columns if col.startswith("microprice_k")]
        for col in microprice_cols:
            offset_col = col.replace("microprice_k", "microprice_offset_bps_k")
            df[offset_col] = (df[col] - df["mid"]) / df["mid"] * 1e4

        return df

    def _compute_vwap_offset_bps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes ta_vwap_offset_bps = (vwap - mid) / mid * 1e4
        for each trade aggression window.
        """
        if "mid" not in df.columns:
            LOG.warning("'mid' column not found, skipping vwap_offset_bps computation.")
            return df

        vwap_cols = [col for col in df.columns if col == "vwap" or "_vwap" in col]
        for col in vwap_cols:
            # Create offset column name
            if col == "vwap":
                offset_col = "vwap_offset_bps"
            else:
                offset_col = col.replace("vwap", "vwap_offset_bps")

            df[offset_col] = (df[col] - df["mid"]) / df["mid"] * 1e4

        return df

    def _compute_rolling_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling mean and std for orderbook features over 1s, 3s, 5s windows.
        """
        windows = {"1s": "1S", "3s": "3S", "5s": "5S"}

        # Select orderbook features to aggregate
        # Include: spread_bps, volume imbalance, weighted imbalance, microprice_offset_bps
        feature_patterns = [
            "spread_bps",
            "vol_imb_k",
            "weighted_imb_k",
            "microprice_offset_bps_k",
            "depth_imb_bps",
        ]

        features_to_aggregate = []
        for pattern in feature_patterns:
            features_to_aggregate.extend([col for col in df.columns if pattern in col])

        # Remove duplicates
        features_to_aggregate = list(set(features_to_aggregate))

        for window_name, window_size in windows.items():
            for feature in features_to_aggregate:
                # Rolling mean
                df[f"{feature}_mean_{window_name}"] = df[feature].rolling(window_size).mean()
                # Rolling std
                df[f"{feature}_std_{window_name}"] = df[feature].rolling(window_size).std()

        return df

    def _compute_cross_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes cross-window features for trade aggression.
        E.g., imbalance_qty difference between 1s and 3s windows (impulse vs persistence).
        """
        # Define window pairs for comparison
        window_pairs = [
            ("1s", "3s"),
            ("1s", "5s"),
            ("3s", "5s"),
        ]

        # Features to compute cross-window comparisons for
        features = ["imbalance_qty", "imbalance_count", "qty_per_sec", "trades_per_sec"]

        for feature in features:
            for w1, w2 in window_pairs:
                col1 = f"{feature}_{w1}"
                col2 = f"{feature}_{w2}"

                if col1 in df.columns and col2 in df.columns:
                    # Difference (impulse vs persistence)
                    df[f"{feature}_diff_{w1}_{w2}"] = df[col1] - df[col2]

                    # Sign agreement (both positive or both negative)
                    df[f"{feature}_sign_agree_{w1}_{w2}"] = (((df[col1] > 0) == (df[col2] > 0))).astype(int)

        return df

    def _compute_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes interaction features:
        1. Sign agreement between trade imbalance and microprice offset (per window)
        2. Strong book but weak trades flags
        3. Spread-conditioned strength
        """
        # 1. Sign agreement: sign(ta_imbalance_qty) == sign(microprice_offset_bps_k1)
        #    Compute for each trade aggression window
        windows = ["1s", "3s", "5s"]
        for window in windows:
            imb_col = f"imbalance_qty_{window}"
            if imb_col in df.columns and "microprice_offset_bps_k1" in df.columns:
                df[f"sign_agreement_trade_book_{window}"] = (
                    (df[imb_col] > 0) == (df["microprice_offset_bps_k1"] > 0)
                ).astype(int)

        # 2. Strong book but weak trades: large book imbalance + low trades_per_sec
        for window in windows:
            trades_col = f"trades_per_sec_{window}"
            if "vol_imb_k1" in df.columns and trades_col in df.columns:
                # Define thresholds
                strong_book_threshold = 0.3  # absolute imbalance > 0.3
                weak_trades_threshold = 10  # trades_per_sec < 10

                df[f"strong_book_weak_trades_{window}"] = (
                    (df["vol_imb_k1"].abs() > strong_book_threshold) & (df[trades_col] < weak_trades_threshold)
                ).astype(int)

        # 3. Spread-conditioned strength: only consider signals when spread is reasonable
        if "spread_bps" in df.columns:
            spread_threshold = 5  # 5 bps
            df["tight_spread"] = (df["spread_bps"] < spread_threshold).astype(int)

            # Create spread-conditioned imbalance signals
            for window in windows:
                imb_col = f"imbalance_qty_{window}"
                if imb_col in df.columns:
                    df[f"imbalance_qty_tight_spread_{window}"] = df[imb_col] * df["tight_spread"]

        return df
