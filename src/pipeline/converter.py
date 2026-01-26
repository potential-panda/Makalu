import numpy as np
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
        exchange_name: str = None,
        symbol: str = None,
        tags: dict = None,
    ) -> list[Point]:
        """
        Converts a DataFrame to a list of InfluxDB Points.

        Args:
            df: DataFrame with time index and numeric columns as fields
            measurement: InfluxDB measurement name
            exchange_name: (Optional) Exchange name tag
            symbol: (Optional) Symbol tag
            tags: (Optional) A dictionary of additional custom tags

        Returns:
            List of InfluxDB Point objects
        """
        points = []
        for timestamp, row in df.iterrows():
            point = Point(measurement).time(timestamp, WritePrecision.NS)

            # Add standard and custom tags
            if exchange_name:
                point = point.tag("exchange_name", exchange_name)
            if symbol:
                point = point.tag("symbol", symbol)
            if tags:
                for tag_key, tag_value in tags.items():
                    point = point.tag(tag_key, tag_value)

            for col, value in row.items():
                if pd.notna(value):
                    # Convert numpy types to Python native types
                    if hasattr(value, "item"):
                        value = value.item()

                    # Ensure integer fields are written as integers
                    if isinstance(value, (int, np.integer)):
                        point = point.field(str(col), int(value))
                    else:
                        point = point.field(str(col), float(value))
            points.append(point)
        return points
