import logging
from typing import List

from feature_engineering.execution_price import build_execution_price_features
from pipeline.framework import PipelineContext, Step

from .context import Context

LOG = logging.getLogger(__name__)


class ExecutionPriceFeatureStep(Step):
    def __init__(self, rolling_windows: List[str]):
        self.rolling_windows = rolling_windows

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        before = len(stage_context.features)
        build_execution_price_features(
            stage_context.source_df, stage_context.features, self.rolling_windows
        )
        LOG.info(f"  Execution Price: +{len(stage_context.features) - before} features")
