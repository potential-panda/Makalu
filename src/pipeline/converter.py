import pandas as pd
from influxdb_client import Point, WritePrecision


class DataPointConverter:
    """
    Utility class to convert pipeline objects into InfluxDB point format.
    """

    @staticmethod
    def convert_dataframe_to_parquet(df: pd.DataFrame, file_path: str):
        """Saves a DataFrame to a Parquet file."""
        df.to_parquet(file_path)

    @staticmethod
    def convert_dataframe_to_influx_points(
        df: pd.DataFrame,
        measurement: str,
        exchange_name: str,
        symbol: str,
    ) -> list[Point]:
        """
        Converts a DataFrame to a list of InfluxDB Points.

        Args:
            df: DataFrame with time index and numeric columns as fields
            measurement: InfluxDB measurement name
            exchange_name: Exchange name tag
            symbol: Symbol tag

        Returns:
            List of InfluxDB Point objects
        """
        points = []
        for timestamp, row in df.iterrows():
            point = (
                Point(measurement)
                .tag("exchange_name", exchange_name)
                .tag("symbol", symbol)
                .time(timestamp, WritePrecision.NS)
            )
            for col, value in row.items():
                if pd.notna(value):
                    # Convert numpy types to Python native types
                    if hasattr(value, "item"):
                        value = value.item()
                    point = point.field(str(col), float(value))
            points.append(point)
        return points
