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
        sample_sec = pd.Timedelta(self.config.sample_interval).total_seconds()
        emit_sec = pd.Timedelta(self.config.emit_interval).total_seconds()
        if sample_sec <= 0 or emit_sec <= 0:
            raise ValueError("sample_interval and emit_interval must be positive")

        horizon_periods = max(1, int(round(self.config.return_horizon_sec / sample_sec)))
        lookback_periods = max(1, int(round(self.config.lookback_sec / sample_sec)))
        smooth_periods = max(1, int(round(self.config.smooth_sec / sample_sec)))

        bid = df["bid"].resample(self.config.sample_interval).last().ffill()
        ask = df["ask"].resample(self.config.sample_interval).last().ffill()
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_pct = spread / mid
        spread_bps = spread_pct * 10000

        returns = mid.pct_change(periods=horizon_periods)
        volatility_raw = returns.rolling(window=lookback_periods, min_periods=1).std()
        volatility_ema = volatility_raw.ewm(span=smooth_periods, adjust=False).mean()
        spread_pct_ema = spread_pct.ewm(span=lookback_periods, adjust=False).mean()
        vol_spread_ratio = volatility_ema / spread_pct_ema.where(spread_pct_ema > 0)

        out = pd.DataFrame(
            {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": spread,
                "spread_pct": spread_pct,
                "spread_bps": spread_bps,
                "volatility_raw": volatility_raw,
                "volatility_ema": volatility_ema,
                "spread_pct_ema": spread_pct_ema,
                "vol_spread_ratio": vol_spread_ratio,
            }
        )

        return out.resample(self.config.emit_interval).last().ffill()

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
