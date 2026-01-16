import logging
from typing import Any, Dict

import pandas as pd
from qate.store.influx_writer import InfluxWriter

from .converter import DataPointConverter
from .framework import Context, Step

LOG = logging.getLogger(__name__)


class PnlLabelStep(Step):
    """
    Queries PNL data from InfluxDB and computes label features.

    Expected PNL data structure in InfluxDB:
    - measurement: PNL
    - tags: exchange_name, symbol, side, settlement, time_align
    - fields: pnl, pnl_return

    Filters applied:
    - exchange_name and symbol from context.market
    - settlement=close, time_align=open

    Output columns:
    - pnl: absolute PnL change
    - pnl_return: pnl / cost
    - ema_pnl_return_{window}: EMA of pnl_return
    - cumulative_pnl: running sum of pnl
    - rolling_sharpe_{window}: rolling Sharpe ratio (mean/std of pnl_return)
    - rolling_volatility_{window}: rolling std of pnl_return
    - pnl_return_zscore_{window}: z-score of pnl_return
    - direction_label: 1 if pnl_return > 0, else 0
    """

    def __init__(self, ema_windows: list[int] = None, rolling_windows: list[int] = None):
        """
        Args:
            ema_windows: List of window sizes for EMA calculation (default: [5, 10, 20])
            rolling_windows: List of window sizes for rolling metrics (default: [10, 20, 50])
        """
        super().__init__()
        self.ema_windows = ema_windows or [5, 10, 20]
        self.rolling_windows = rolling_windows or [10, 20, 50]

    def populate(self, influx_writer: InfluxWriter, context: Context, args: Dict[str, Any]):
        """Queries PNL data from InfluxDB and computes label features."""
        pnl_bucket = context.pnl_bucket
        if not pnl_bucket:
            LOG.error("PnlLabelStep: pnl_bucket not specified in context")
            return

        # Query PNL data
        pnl_df = context.influx_reader.query_pnl(
            bucket=pnl_bucket,
            dt_range=context.compute_dt_range,
            market=context.market,
            settlement="close",
            time_align="open",
        )

        if pnl_df.empty:
            LOG.warning("PnlLabelStep: No PNL data found.")
            context.label_df = pd.DataFrame()
            return

        # Reset index if time is the index
        if pnl_df.index.name == "time" or pnl_df.index.name == "_time":
            pnl_df = pnl_df.reset_index()

        # Aggregate by time if there are multiple sides (net PnL)
        if "side" in pnl_df.columns:
            # Group by time and sum pnl, weighted average for pnl_return
            pnl_df = self._aggregate_by_time(pnl_df)

        # Ensure time index
        if "time" in pnl_df.columns:
            pnl_df = pnl_df.set_index("time")
        elif "_time" in pnl_df.columns:
            pnl_df = pnl_df.set_index("_time")

        # Sort by time
        pnl_df = pnl_df.sort_index()

        # Build label DataFrame
        label_df = pd.DataFrame(index=pnl_df.index)

        # Base columns
        label_df["pnl"] = pnl_df["pnl"]
        label_df["pnl_return"] = pnl_df["pnl_return"]

        # Direction label (1 = profit, 0 = loss)
        label_df["direction_label"] = (pnl_df["pnl_return"] > 0).astype(int)

        # Cumulative PnL
        label_df["cumulative_pnl"] = pnl_df["pnl"].cumsum()

        # EMA of pnl_return for various windows
        for window in self.ema_windows:
            label_df[f"ema_pnl_return_{window}"] = pnl_df["pnl_return"].ewm(span=window, adjust=False).mean()

        # Rolling metrics
        for window in self.rolling_windows:
            # Rolling mean and std for Sharpe and z-score
            rolling_mean = pnl_df["pnl_return"].rolling(window=window, min_periods=1).mean()
            rolling_std = pnl_df["pnl_return"].rolling(window=window, min_periods=1).std()

            # Rolling Sharpe ratio (annualization factor not applied - raw ratio)
            label_df[f"rolling_sharpe_{window}"] = rolling_mean / rolling_std.replace(0, float("nan"))

            # Rolling volatility
            label_df[f"rolling_volatility_{window}"] = rolling_std

            # Z-score of pnl_return
            label_df[f"pnl_return_zscore_{window}"] = (pnl_df["pnl_return"] - rolling_mean) / rolling_std.replace(
                0, float("nan")
            )

        # Filter to populate_dt_range
        label_df = label_df[
            (label_df.index >= context.populate_dt_range.start_dt)
            & (label_df.index <= context.populate_dt_range.end_dt)
        ]

        context.label_df = label_df
        LOG.info(
            f"PnlLabelStep: Computed {len(label_df)} label records with {len(label_df.columns)} columns "
            f"from bucket '{pnl_bucket}'."
        )

    def _aggregate_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates PNL data by time when multiple sides exist.
        - pnl: sum across sides
        - pnl_return: weighted average by absolute pnl (or simple average if pnl is zero)
        """
        time_col = "time" if "time" in df.columns else "_time"

        # Sum pnl by time
        pnl_sum = df.groupby(time_col)["pnl"].sum()

        # Weighted average of pnl_return (weight by absolute pnl)
        def weighted_avg_return(group):
            weights = group["pnl"].abs()
            if weights.sum() == 0:
                return group["pnl_return"].mean()
            return (group["pnl_return"] * weights).sum() / weights.sum()

        pnl_return_avg = df.groupby(time_col).apply(weighted_avg_return, include_groups=False)

        result = pd.DataFrame({"pnl": pnl_sum, "pnl_return": pnl_return_avg})
        result.index.name = time_col
        return result.reset_index()
