import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from qate.core.model import Market

from lead_lag.lead_lag import LeadLagConfig, compute_sliding_windows

from .framework import PipelineContext, StageContext, Step, StepProgressLogger

LOG = logging.getLogger(__name__)

LEAD_LAG_BUCKET = "lead_lag"


@dataclass
class Context(StageContext):
    market_lead: Market
    market_lag: Market
    ret_timestamps: pd.DatetimeIndex
    lead_returns: np.ndarray
    lag_returns: np.ndarray


class LeadLagStep(Step):
    def __init__(self, config: LeadLagConfig):
        super().__init__()
        self.config = config
        resample_sec = self.config.resample_ms / 1000.0
        self.window_steps = int(self.config.window_size_sec / resample_sec)
        self.step_steps = int(self.config.window_step_sec / resample_sec)

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        write_store = pipeline_context.parquet_store.create_lead_lag(
            stage_context.market_lead, stage_context.market_lag, self.config.param_id()
        )
        n_returns = len(stage_context.lead_returns)
        n_windows = max(0, (n_returns - self.window_steps) // self.step_steps + 1)
        progress = StepProgressLogger(self.__class__.__name__, n_windows)

        for window_end_ts, fields in compute_sliding_windows(
            stage_context.ret_timestamps,
            stage_context.lead_returns,
            stage_context.lag_returns,
            self.config,
            populate_start_ts=stage_context.populate_dt_range.start_ts,
            populate_end_ts=stage_context.populate_dt_range.end_ts,
        ):
            write_store.add(window_end_ts, fields)
            progress.update()

        write_store.close()
        LOG.info(f"LeadLagStep: Processed {progress.cnt} windows")
