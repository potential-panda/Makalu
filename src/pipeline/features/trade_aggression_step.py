import logging
from typing import List

from feature_engineering.trade_aggression import build_trade_aggression_features
from pipeline.framework import PipelineContext, Step

from .context import Context

LOG = logging.getLogger(__name__)


class TradeAggressionFeatureStep(Step):
    def __init__(self, windows: List[str], rolling_windows: List[str]):
        self.windows = windows
        self.rolling_windows = rolling_windows

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        before = len(stage_context.features)
        build_trade_aggression_features(
            stage_context.source_df,
            stage_context.features,
            self.windows,
            self.rolling_windows,
        )
        LOG.info(
            f"  Trade Aggression: +{len(stage_context.features) - before} features"
        )
