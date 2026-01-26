"""
Label generation pipeline steps.

This module implements label generation for ML models, primarily focusing on the
Triple-Barrier Method to create actionable buy/sell/hold labels. This version
uses execution bid/ask prices for realistic P&L calculation.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from qate.core.model import Market
from qate.util.dt_range import DtRange

from .converter import DataPointConverter
from .framework import PipelineContext, StageContext, Step

LOG = logging.getLogger(__name__)

# Bucket name constant
LABEL_DATA_BUCKET = "label_data"
EPS = 1e-9


@dataclass
class Context(StageContext):
    compute_dt_range: DtRange
    market: Market
    execution_price_df: pd.DataFrame = None


def _compute_labels_for_long_scenario(
    entry_ask_prices: pd.Series,
    future_bid_prices: pd.Series,
    pt_pct: float,
    sl_pct: float,
    timeout_delta: pd.Timedelta,
) -> pd.DataFrame:
    """Computes triple-barrier labels for a long position."""
    n = len(entry_ask_prices)
    results = []

    for i in range(n):
        entry_ts = entry_ask_prices.index[i]
        entry_price = entry_ask_prices.iloc[i]
        timeout_ts = entry_ts + timeout_delta

        if pd.isna(entry_price):
            continue

        upper_barrier = entry_price * (1 + pt_pct)
        lower_barrier = entry_price * (1 - sl_pct)

        future_path = future_bid_prices.loc[future_bid_prices.index > entry_ts]

        pt_hits = future_path[future_path >= upper_barrier]
        sl_hits = future_path[future_path <= lower_barrier]

        first_pt_hit_ts = pt_hits.index[0] if not pt_hits.empty else None
        first_sl_hit_ts = sl_hits.index[0] if not sl_hits.empty else None

        first_hit_ts = (
            min(ts for ts in [first_pt_hit_ts, first_sl_hit_ts] if ts is not None)
            if any([first_pt_hit_ts, first_sl_hit_ts])
            else None
        )

        if first_hit_ts and first_hit_ts <= timeout_ts:
            if future_path[first_hit_ts] >= upper_barrier:
                results.append({"label_long": 1, "return_long": pt_pct, "end_dt_long": first_hit_ts})
            else:
                results.append({"label_long": -1, "return_long": -sl_pct, "end_dt_long": first_hit_ts})
        else:
            exit_price = (
                future_path.loc[future_path.index <= timeout_ts].iloc[-1]
                if not future_path.loc[future_path.index <= timeout_ts].empty
                else entry_price
            )
            ret = (exit_price - entry_price) / (entry_price + EPS)
            results.append({"label_long": 0, "return_long": ret, "end_dt_long": timeout_ts})

    return pd.DataFrame(results, index=entry_ask_prices.index[: len(results)])


def _compute_labels_for_short_scenario(
    entry_bid_prices: pd.Series,
    future_ask_prices: pd.Series,
    pt_pct: float,
    sl_pct: float,
    timeout_delta: pd.Timedelta,
) -> pd.DataFrame:
    """Computes triple-barrier labels for a short position."""
    n = len(entry_bid_prices)
    results = []

    for i in range(n):
        entry_ts = entry_bid_prices.index[i]
        entry_price = entry_bid_prices.iloc[i]
        timeout_ts = entry_ts + timeout_delta

        if pd.isna(entry_price):
            continue

        # Note: For short, profit-take is on the downside.
        lower_barrier = entry_price * (1 - pt_pct)
        upper_barrier = entry_price * (1 + sl_pct)

        future_path = future_ask_prices.loc[future_ask_prices.index > entry_ts]

        pt_hits = future_path[future_path <= lower_barrier]
        sl_hits = future_path[future_path >= upper_barrier]

        first_pt_hit_ts = pt_hits.index[0] if not pt_hits.empty else None
        first_sl_hit_ts = sl_hits.index[0] if not sl_hits.empty else None

        first_hit_ts = (
            min(ts for ts in [first_pt_hit_ts, first_sl_hit_ts] if ts is not None)
            if any([first_pt_hit_ts, first_sl_hit_ts])
            else None
        )

        if first_hit_ts and first_hit_ts <= timeout_ts:
            if future_path[first_hit_ts] <= lower_barrier:
                results.append({"label_short": 1, "return_short": pt_pct, "end_dt_short": first_hit_ts})
            else:
                results.append({"label_short": -1, "return_short": -sl_pct, "end_dt_short": first_hit_ts})
        else:
            exit_price = (
                future_path.loc[future_path.index <= timeout_ts].iloc[-1]
                if not future_path.loc[future_path.index <= timeout_ts].empty
                else entry_price
            )
            ret = (entry_price - exit_price) / (entry_price + EPS)
            results.append({"label_short": 0, "return_short": ret, "end_dt_short": timeout_ts})

    return pd.DataFrame(results, index=entry_bid_prices.index[: len(results)])


class QueryExecutionPriceStep(Step):
    """Queries the execution price data needed for labeling."""

    def __init__(self, execution_size_str: str):
        super().__init__()
        self.execution_size_str = execution_size_str

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        parquet_store = pipeline_context.parquet_store.create_execution_price(
            stage_context.market, self.execution_size_str
        )
        read_range = DtRange(
            stage_context.compute_dt_range.start_dt,
            stage_context.compute_dt_range.end_dt + pd.Timedelta(seconds=120),  # Read extra for future path
        )
        df = parquet_store.read(read_range)
        stage_context.execution_price_df = df


class GenerateLabelsStep(Step):
    """
    Generates labels using the Triple-Barrier Method on realistic execution prices.
    """

    def __init__(self, profit_take_pct: float, stop_loss_pct: float, timeout_sec: int, execution_size_str: str):
        super().__init__()
        self.pt_pct = profit_take_pct
        self.sl_pct = stop_loss_pct
        self.timeout_sec = timeout_sec
        self.execution_size_str = execution_size_str

        self.pt_bps = int(self.pt_pct * 10000)
        self.sl_bps = int(self.sl_pct * 10000)
        self.param_id = f"{self.pt_bps}_{self.sl_bps}_{self.timeout_sec}_{self.execution_size_str}"

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        df = stage_context.execution_price_df
        if df is None or df.empty or not all(c in df.columns for c in ["bid", "ask"]):
            LOG.warning("GenerateLabelsStep: No execution price data with 'bid' and 'ask' available.")
            return

        LOG.info(f"GenerateLabelsStep: Starting label generation for param_id={self.param_id} with {len(df)} rows.")

        if "ts" in df.columns:
            df.index = pd.to_datetime(df["ts"], unit="s")
            df = df.drop(columns=["ts"])
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        bid_prices = df["bid"].dropna()
        ask_prices = df["ask"].dropna()
        timeout_delta = pd.Timedelta(seconds=self.timeout_sec)

        # Compute labels for both long and short scenarios
        long_labels_df = _compute_labels_for_long_scenario(
            ask_prices, bid_prices, self.pt_pct, self.sl_pct, timeout_delta
        )
        short_labels_df = _compute_labels_for_short_scenario(
            bid_prices, ask_prices, self.pt_pct, self.sl_pct, timeout_delta
        )

        # Join the results
        # The resulting labels_df will contain the following columns:
        # - label_long (int): Triple-Barrier label for a hypothetical long position (1: Profit Take, -1: Stop Loss, 0: Timeout)
        # - return_long (float): Realized return for the long position (pt_pct, -sl_pct, or actual return at timeout)
        # - end_dt_long (pd.Timestamp): Timestamp when the barrier was hit or timeout occurred for the long position
        # - label_short (int): Triple-Barrier label for a hypothetical short position (1: Profit Take, -1: Stop Loss, 0: Timeout)
        # - return_short (float): Realized return for the short position (pt_pct, -sl_pct, or actual return at timeout)
        # - end_dt_short (pd.Timestamp): Timestamp when the barrier was hit or timeout occurred for the short position
        labels_df = long_labels_df.join(short_labels_df, how="outer")

        # Filter to original populate range
        labels_df = labels_df[
            (labels_df.index >= stage_context.populate_dt_range.start_dt)
            & (labels_df.index <= stage_context.populate_dt_range.end_dt)
        ]

        if labels_df.empty:
            LOG.warning("GenerateLabelsStep: No labels generated after filtering.")
            return

        LOG.info(f"GenerateLabelsStep: Generated {len(labels_df)} labels.")
        self._save_to_parquet(pipeline_context, stage_context, labels_df)
        self._save_to_influxdb(pipeline_context, stage_context, labels_df)

    def _save_to_parquet(self, pipeline_context: PipelineContext, stage_context: Context, labels_df: pd.DataFrame):
        parquet_store = pipeline_context.parquet_store.create_label(
            "TripleBarrierLabel",
            stage_context.market,
            self.param_id,
        )
        parquet_store.add_dataframe(labels_df.copy(), ts_column=None)
        parquet_store.close()
        LOG.info(f"GenerateLabelsStep: Saved {len(labels_df)} labels to Parquet with partition {self.param_id}.")

    def _save_to_influxdb(self, pipeline_context: PipelineContext, stage_context: Context, labels_df: pd.DataFrame):
        influx_writer = pipeline_context.influx_store.create_writer(LABEL_DATA_BUCKET, no_retention=True)

        custom_tags = {
            "profit_take_bps": self.pt_bps,
            "stop_loss_bps": self.sl_bps,
            "timeout_sec": self.timeout_sec,
            "exec_size": self.execution_size_str,
        }

        # Select only numeric columns for InfluxDB to avoid errors with datetime fields
        numeric_labels_df = labels_df.select_dtypes(include=[np.number])

        points = DataPointConverter.convert_dataframe_to_influx_points(
            numeric_labels_df,
            measurement="TRIPLE_BARRIER_LABEL",
            exchange_name=stage_context.market.exchange_name.name,
            symbol=stage_context.market.symbol.name,
            tags=custom_tags,
        )
        for point in points:
            influx_writer.add(point)
        influx_writer.close()
        LOG.info(f"GenerateLabelsStep: Wrote {len(points)} label points to InfluxDB.")
