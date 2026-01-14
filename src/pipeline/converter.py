import pandas as pd
from influxdb_client import Point, WritePrecision
from qate.core.model import Market, MarketPrice, OrderBook, Trade
from qate.trading.aggression import TradeAggression
from qate.trading.book_change import BookChange
from qate.trading.chart import Bar


class DataPointConverter:
    """
    Utility class to convert pipeline objects into InfluxDB point format.
    """

    @staticmethod
    def convert_trade_to_influx_point(trade: Trade) -> Point:
        """Converts a Trade object into an InfluxDB Point."""
        point = (
            Point("TRADE")
            .tag("exchange_name", trade.exchange_name.name)
            .tag("symbol", trade.symbol.name)
            .tag("side", trade.side.name)
            .field("price", trade.price)
            .field("size", trade.size)
            .time(int(trade.get_ts() * 1000), WritePrecision.MS)
        )
        return point

    @staticmethod
    def convert_market_price_to_influx_point(
        order_book: OrderBook, market_price: MarketPrice, execution_size_str: str
    ) -> Point:
        """Converts MarketPrice and target_size into an InfluxDB point."""
        return (
            Point("MARKET_PRICE")
            .tag("exchange_name", order_book.exchange_name.name)
            .tag("symbol", order_book.symbol.name)
            .tag("execution_size", execution_size_str)
            .field("bid", market_price.bid)
            .field("ask", market_price.ask)
            .field("spread", market_price.spread)
            .field("spread_bps", market_price.spread_bps)
            .time(int(order_book.get_ts() * 1000), WritePrecision.MS)
        )

    @staticmethod
    def convert_order_book_to_influx_point(order_book: OrderBook) -> Point:
        """Converts an OrderBook object into an InfluxDB Point."""
        return (
            Point("ORDER_BOOK")
            .tag("exchange_name", order_book.exchange_name.name)
            .tag("symbol", order_book.symbol.name)
            .field("best_ask", order_book.best_ask)
            .field("best_bid", order_book.best_bid)
            .field("mid", order_book.mid)
            .field("spread", order_book.spread)
            .field("spread_bps", order_book.spread_bps)
            .time(int(order_book.get_ts() * 1000), WritePrecision.MS)
        )

    @staticmethod
    def convert_volume_imbalance_to_influx_point(order_book: OrderBook, k: int, value: float) -> Point:
        """Converts a volume imbalance value into an InfluxDB Point."""
        return (
            Point("VOLUME_IMBALANCE")
            .tag("exchange_name", order_book.exchange_name.name)
            .tag("symbol", order_book.symbol.name)
            .tag("k", k)
            .field("value", value)
            .time(int(order_book.get_ts() * 1000), WritePrecision.MS)
        )

    @staticmethod
    def convert_weighted_imbalance_to_influx_point(order_book: OrderBook, k: int, value: float) -> Point:
        """Converts a weighted imbalance value into an InfluxDB Point."""
        return (
            Point("WEIGHTED_IMBALANCE")
            .tag("exchange_name", order_book.exchange_name.name)
            .tag("symbol", order_book.symbol.name)
            .tag("k", k)
            .field("value", value)
            .time(int(order_book.get_ts() * 1000), WritePrecision.MS)
        )

    @staticmethod
    def convert_depth_imbalance_to_influx_point(order_book: OrderBook, bps: float, value: float) -> Point:
        """Converts a depth imbalance value into an InfluxDB Point."""
        return (
            Point("DEPTH_IMBALANCE")
            .tag("exchange_name", order_book.exchange_name.name)
            .tag("symbol", order_book.symbol.name)
            .tag("bps", int(bps * 10000))
            .field("value", value)
            .time(int(order_book.get_ts() * 1000), WritePrecision.MS)
        )

    @staticmethod
    def convert_microprice_to_influx_point(order_book: OrderBook, k: int, value: float) -> Point:
        """Converts a microprice value into an InfluxDB Point."""
        return (
            Point("MICROPRICE")
            .tag("exchange_name", order_book.exchange_name.name)
            .tag("symbol", order_book.symbol.name)
            .tag("k", k)
            .field("value", value)
            .time(int(order_book.get_ts() * 1000), WritePrecision.MS)
        )

    @staticmethod
    def convert_book_change_to_influx_point(order_book: OrderBook, book_change: BookChange) -> Point:
        """Converts OrderBook, BookChange, and k into an InfluxDB point."""
        point = (
            Point("BOOK_CHANGE")
            .tag("exchange_name", order_book.exchange_name.name)
            .tag("symbol", order_book.symbol.name)
            .tag("k", str(book_change.k))
            .time(int(order_book.get_ts() * 1000), WritePrecision.MS)
        )
        for k, v in book_change.to_dict().items():
            if v is not None:
                point = point.field(k, v)
        return point

    @staticmethod
    def convert_trade_aggression_to_influx_point(market: Market, aggression: TradeAggression) -> Point:
        """Converts a TradeAggression object into an InfluxDB Point."""
        point = (
            Point("TRADE_AGGRESSION")
            .tag("exchange_name", market.exchange_name.name)
            .tag("symbol", market.symbol.name)
            .tag("window_sec", aggression.window_sec)
            .time(int(aggression.ts * 1000), WritePrecision.MS)
        )
        for k, v in aggression.to_dict().items():
            if v is not None:
                point = point.field(k, v)
        return point

    @staticmethod
    def convert_bar_to_influx_point(bar: Bar, symbol: str, exchange: str, interval: str, bar_type: str) -> Point:
        """Converts a Bar object into an InfluxDB Point for CHART measurement."""
        point = (
            Point("CHART")
            .tag("symbol", symbol)
            .tag("exchange_name", exchange)
            .tag("interval", interval)
            .tag("bar_type", bar_type)
            .field("open", bar.open)
            .field("high", bar.high)
            .field("low", bar.low)
            .field("close", bar.close)
            .field("volume", bar.volume)
            .field("tick_count", bar.tick_count)
            .time(int(bar.get_ts() * 1000), WritePrecision.MS)
        )
        return point

    @staticmethod
    def convert_dataframe_to_parquet(df: pd.DataFrame, file_path: str):
        """Saves a DataFrame to a Parquet file."""
        df.to_parquet(file_path)
