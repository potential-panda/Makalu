import logging
from dataclasses import dataclass

import pandas as pd
from qate.core.model import Market

from feature_engineering.volatility import VolSpreadConfig

from .framework import PipelineContext, StageContext, Step

LOG = logging.getLogger(__name__)

# Bucket name constants
EXECUTION_PRICE_DERIVED_BUCKET = "execution_price_derived"


@dataclass
class Context(StageContext):
    market: Market
    execution_size_str: str
    price_df: pd.DataFrame


class VolSpreadStep(Step):
    def __init__(self, config: VolSpreadConfig):
        super().__init__()
        self.config = config

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        dt_sec = pd.Timedelta(self.config.resample_interval).total_seconds()
        shift_periods = int(round(self.config.return_shift_sec / dt_sec))
        lookback_periods = int(round(self.config.lookback_sec / dt_sec))

        mid_resampled = df["mid"].resample(self.config.resample_interval).last().ffill()
        spread_resampled = df["spread"].resample(self.config.resample_interval).last().ffill()

        returns = mid_resampled.pct_change(periods=shift_periods)
        volatility = returns.rolling(window=lookback_periods, min_periods=1).std()
        spread_pct = spread_resampled / mid_resampled
        spread_pct_ema = spread_pct.ewm(span=self.config.ema_span_sec, adjust=False).mean()

        df["volatility"] = volatility.reindex(df.index, method="ffill")
        df["spread_pct_ema"] = spread_pct_ema.reindex(df.index, method="ffill")
        return df

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        df = self._calculate(stage_context.price_df)
        df = df[
            (df.index >= stage_context.populate_dt_range.start_dt_utc)
            & (df.index <= stage_context.populate_dt_range.end_dt_utc)
        ]
        # df["ts"] = df.index.astype("int64") / 1e9  # nanoseconds → float seconds

        store = pipeline_context.parquet_store.create_vol_spread(
            stage_context.market,
            stage_context.execution_size_str,
            self.config.param_id,
        )
        # store.add_dataframe(df, ts_column="ts")
        store.add_dataframe(df, ts_column=None)
        store.close()
        LOG.info(f"PopulateVolSpreadStep: Wrote {len(df)} records to Parquet store")
