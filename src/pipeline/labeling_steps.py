"""
Label generation pipeline steps.

This module implements label generation for ML models, primarily focusing on the
Triple-Barrier Method to create actionable buy/sell/hold labels.
"""

import logging
from dataclasses import dataclass

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
    book_state_df: pd.DataFrame = None


def _compute_triple_barrier_labels_pandas(
    prices: pd.Series,
    pt_pct: float,
    sl_pct: float,
    timeout_delta: pd.Timedelta,
) -> pd.DataFrame:
    """
    Computes labels using the Triple-Barrier Method with a pure Pandas approach.

    Args:
        prices: Series of mid prices with a DatetimeIndex.
        pt_pct: Profit-take percentage.
        sl_pct: Stop-loss percentage.
        timeout_delta: Timedelta for the vertical barrier.

    Returns:
        A DataFrame with columns: [label, end_dt, return].
    """
    n = len(prices)
    results = []

    for i in range(n):
        entry_ts = prices.index[i]
        entry_price = prices.iloc[i]

        if pd.isna(entry_price):
            continue

        upper_barrier = entry_price * (1 + pt_pct)
        lower_barrier = entry_price * (1 - sl_pct)
        timeout_ts = entry_ts + timeout_delta

        # Slice future prices for the current entry
        start = i + 1
        future_prices = prices.iloc[start:]

        # Find times when barriers are hit
        pt_hits = future_prices[future_prices >= upper_barrier]
        sl_hits = future_prices[future_prices <= lower_barrier]

        # Find the first hit for each barrier
        first_pt_hit_ts = pt_hits.index[0] if not pt_hits.empty else None
        first_sl_hit_ts = sl_hits.index[0] if not sl_hits.empty else None

        # Determine which event happened first
        first_hit_ts = None
        label = 0
        ret = 0.0

        if first_pt_hit_ts and first_sl_hit_ts:
            first_hit_ts = min(first_pt_hit_ts, first_sl_hit_ts)
        elif first_pt_hit_ts:
            first_hit_ts = first_pt_hit_ts
        elif first_sl_hit_ts:
            first_hit_ts = first_sl_hit_ts

        if first_hit_ts and first_hit_ts <= timeout_ts:
            if prices[first_hit_ts] >= upper_barrier:
                label = 1
                ret = pt_pct
            else:
                label = -1
                ret = -sl_pct
            end_dt = first_hit_ts
        else:
            # Timeout case
            label = 0
            # Get the price at or just before timeout
            last_price_before_timeout = future_prices[future_prices.index <= timeout_ts]
            if not last_price_before_timeout.empty:
                ret = (last_price_before_timeout.iloc[-1] - entry_price) / (entry_price + EPS)
            else:  # If timeout is before the next tick
                ret = 0.0
            end_dt = timeout_ts

        results.append({"label": label, "end_dt": end_dt, "return": ret})

    return pd.DataFrame(results, index=prices.index[: len(results)])


class QueryBookStateStep(Step):
    """Queries the raw order book data needed for labeling."""

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        parquet_store = pipeline_context.parquet_store.create_single_market("BookState", stage_context.market)
        # We need to read a larger range to have future data for labeling
        read_range = DtRange(
            stage_context.compute_dt_range.start_dt,
            stage_context.compute_dt_range.end_dt + pd.Timedelta(seconds=120),  # Read 2 extra minutes for future path
        )
        df = parquet_store.read(read_range)
        stage_context.book_state_df = df


class GenerateLabelsStep(Step):
    """
    Generates labels using the Triple-Barrier Method.

    This step processes raw price data to create forward-looking labels that
    indicate whether a price is likely to hit a profit-take, stop-loss, or
    timeout barrier first.
    """

    def __init__(self, profit_take_pct: float, stop_loss_pct: float, timeout_sec: int):
        super().__init__()
        self.pt_pct = profit_take_pct
        self.sl_pct = stop_loss_pct
        self.timeout_sec = timeout_sec

        # For tagging and partitioning
        self.pt_bps = int(self.pt_pct * 10000)
        self.sl_bps = int(self.sl_pct * 10000)
        self.param_id = f"{self.pt_bps}_{self.sl_bps}_{self.timeout_sec}"

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        """Main entry point for label generation."""
        df = stage_context.book_state_df
        if df is None or df.empty or "mid" not in df.columns:
            LOG.warning("GenerateLabelsStep: No book state data with 'mid' price available.")
            return

        LOG.info(f"GenerateLabelsStep: Starting label generation for param_id={self.param_id} with {len(df)} rows.")

        # Ensure dataframe is sorted by time and has a DatetimeIndex
        if "ts" in df.columns:
            df.index = pd.to_datetime(df["ts"], unit="s")
            df = df.drop(columns=["ts"])
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # Extract mid-price series
        mid_prices = df["mid"].dropna()
        timeout_delta = pd.Timedelta(seconds=self.timeout_sec)

        # Compute labels
        labels_df = _compute_triple_barrier_labels_pandas(mid_prices, self.pt_pct, self.sl_pct, timeout_delta)

        # Filter to original populate range (to drop the extra data we read)
        labels_df = labels_df[
            (labels_df.index >= stage_context.populate_dt_range.start_dt)
            & (labels_df.index <= stage_context.populate_dt_range.end_dt)
        ]

        if labels_df.empty:
            LOG.warning("GenerateLabelsStep: No labels generated after filtering to populate range.")
            return

        LOG.info(f"GenerateLabelsStep: Generated {len(labels_df)} labels.")

        # --- Save to Parquet Store ---
        self._save_to_parquet(pipeline_context, stage_context, labels_df)

        # --- Save to InfluxDB ---
        self._save_to_influxdb(pipeline_context, stage_context, labels_df)

    def _save_to_parquet(self, pipeline_context: PipelineContext, stage_context: Context, labels_df: pd.DataFrame):
        """Saves the generated labels to a partitioned Parquet store."""
        parquet_store = pipeline_context.parquet_store.create_label(stage_context.market, param_id=self.param_id)

        # Parquet store's add_dataframe expects ts column, not index
        df_to_save = labels_df.copy()

        parquet_store.add_dataframe(df_to_save, ts_column=None)  # Pass ts_column=None to use the DatetimeIndex
        parquet_store.close()
        LOG.info(f"GenerateLabelsStep: Saved {len(labels_df)} labels to Parquet with partition {self.param_id}.")

    def _save_to_influxdb(self, pipeline_context: PipelineContext, stage_context: Context, labels_df: pd.DataFrame):
        """Saves the generated labels to InfluxDB with specified tags."""
        influx_writer = pipeline_context.influx_store.create_writer(LABEL_DATA_BUCKET, no_retention=True)

        # Add required tags for InfluxDB
        custom_tags = {"profit_take_bps": self.pt_bps, "stop_loss_bps": self.sl_bps, "timeout_sec": self.timeout_sec}

        points = DataPointConverter.convert_dataframe_to_influx_points(
            labels_df,
            measurement="LABEL",
            exchange_name=stage_context.market.exchange_name.name,
            symbol=stage_context.market.symbol.name,
            tags=custom_tags,
        )
        for point in points:
            influx_writer.add(point)
        influx_writer.close()
        LOG.info(f"GenerateLabelsStep: Wrote {len(points)} label points to InfluxDB.")
