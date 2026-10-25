import logging

import pandas as pd
from qate.core.model import Market

from lead_lag.lead_lag import LeadLagConfig
from makalu import settings as sys_env
from makalu.storage.parquet_store import ParquetStore
from pipeline.framework import PipelineContext, Step

from .context import Context

LOG = logging.getLogger(__name__)


class LoadLeadLagStep(Step):
    def __init__(self, market_lag: Market, config: LeadLagConfig):
        self.market_lag = market_lag
        self.config = config

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        store = ParquetStore(sys_env.get_parquet_root_dir())
        market_lead = stage_context.market
        ll_store = store.create_lead_lag(
            market_lead, self.market_lag, self.config.param_id()
        )
        df = ll_store.read(stage_context.populate_dt_range)

        if df.empty:
            LOG.warning(
                f"LoadLeadLagStep: No data found for {market_lead.id}-{self.market_lag.id}"
            )
            return

        if "ts" in df.columns:
            df.index = pd.to_datetime(df["ts"], unit="s")
            df = df.drop(columns=["ts"])

        # Align with source_df index (ffill to cover tick-level frequency)
        df = df.reindex(stage_context.source_df.index, method="ffill")

        before = len(stage_context.features)
        for col in df.columns:
            stage_context.features[col] = df[col]

        LOG.info(f"  Load Lead-Lag: +{len(stage_context.features) - before} features")
