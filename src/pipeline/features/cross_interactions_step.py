import logging
from typing import List

from feature_engineering.interactions import build_cross_feature_interactions
from pipeline.framework import PipelineContext, Step

from .context import Context

LOG = logging.getLogger(__name__)


class CrossInteractionsFeatureStep(Step):
    def __init__(self, windows: List[str], depth_levels: List[int], base_roll_window: str):
        self.windows = windows
        self.depth_levels = depth_levels
        self.base_roll_window = base_roll_window

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        before = len(stage_context.features)
        build_cross_feature_interactions(
            stage_context.source_df,
            stage_context.features,
            self.windows,
            self.depth_levels,
            self.base_roll_window,
        )
        LOG.info(f"  Cross-Feature Interactions: +{len(stage_context.features) - before} features")
