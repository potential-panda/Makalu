"""
Forward Return Pipeline Step.

Computes simple forward returns of mid price at fixed horizons
(return_30s, return_120s, return_300s, ...) from execution-price data.

This is deliberately separate from Hase's Volatility (backward-looking)
and TripleBarrierLabel (a research-layer, ephemeral, strategy-specific
label) -- forward return at a fixed horizon is a stable, widely-reusable
quantity on its own, used by many different signal-scanning purposes that
would otherwise each recompute the exact same ASOF-join-style forward
join independently.

Stored via the generic create_label() store (same mechanism as
TripleBarrierLabel) under label_name="Return", with param_id encoding the
execution_size, resample interval, and the full horizon set.
"""

import logging
from dataclasses import dataclass

import pandas as pd
from qate.core.model import Market

from .framework import PipelineContext, StageContext, Step

LOG = logging.getLogger(__name__)


@dataclass
class ReturnConfig:
    """horizons_sec: the forward horizons to compute, e.g. [30, 120, 300].
    resample_interval: mid price is resampled to a regular grid first
    (the convention Hase's quote grid uses), e.g. '1s'."""

    horizons_sec: tuple[int, ...]
    execution_size_str: str
    resample_interval: str = "1s"

    @property
    def param_id(self) -> str:
        horizons = "_".join(str(h) for h in self.horizons_sec)
        return f"ret_{self.execution_size_str}_{horizons}_{self.resample_interval}"


@dataclass
class Context(StageContext):
    market: Market
    execution_size_str: str
    price_df: pd.DataFrame


class ReturnStep(Step):
    def __init__(self, config: ReturnConfig):
        super().__init__()
        self.config = config

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        mid_resampled = df["mid"].resample(self.config.resample_interval).last().ffill()
        dt_sec = pd.Timedelta(self.config.resample_interval).total_seconds()

        df = df.copy()
        for horizon_sec in self.config.horizons_sec:
            shift_periods = int(round(horizon_sec / dt_sec))
            # Forward return: needs FUTURE data beyond populate_dt_range.end --
            # the stage reading this context must pad the read range forward by
            # at least max(horizons_sec), or the last horizon_sec of every
            # populate call would have a permanently-missing return (later
            # populate calls for subsequent dates do not retroactively fill it
            # in, per the "idempotent, later dates do not affect earlier ones"
            # contract -- see pipeline/stages.py's ReturnDerivedStage).
            return_resampled = mid_resampled.shift(-shift_periods) / mid_resampled - 1.0
            # reindex back onto df's original tick-level index (mirrors
            # the labellers' pattern exactly) -- a plain .join() on the
            # mismatched resampled-vs-tick indices left ~99.9% of rows null,
            # confirmed against real data before this fix.
            df[f"return_{horizon_sec}s"] = return_resampled.reindex(
                df.index, method="ffill"
            )

        return df

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        df = self._calculate(stage_context.price_df)
        df = df[
            (df.index >= stage_context.populate_dt_range.start_dt_utc)
            & (df.index <= stage_context.populate_dt_range.end_dt_utc)
        ]

        store = pipeline_context.parquet_store.create_label(
            "Return",
            stage_context.market,
            self.config.param_id,
        )
        store.add_dataframe(df, ts_column=None)
        store.close()
        LOG.info(
            f"ReturnStep: Wrote {len(df)} records to Parquet store (param_id={self.config.param_id})"
        )
