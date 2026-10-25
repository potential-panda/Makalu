import logging
from typing import List

from feature_engineering.volatility import build_volatility_features
from pipeline.framework import PipelineContext, Step

from .context import Context

LOG = logging.getLogger(__name__)


class VolatilityFeatureStep(Step):
    def __init__(self, windows: List[str]):
        self.windows = windows

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        before = len(stage_context.features)
        build_volatility_features(
            stage_context.source_df, stage_context.features, self.windows
        )
        LOG.info(f"  Volatility: +{len(stage_context.features) - before} features")
