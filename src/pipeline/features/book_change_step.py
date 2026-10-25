import logging
from typing import List

from feature_engineering.book_change import build_book_change_features
from pipeline.framework import PipelineContext, Step

from .context import Context

LOG = logging.getLogger(__name__)


class BookChangeFeatureStep(Step):
    def __init__(self, depth_levels: List[int], rolling_windows: List[str]):
        self.depth_levels = depth_levels
        self.rolling_windows = rolling_windows

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        before = len(stage_context.features)
        build_book_change_features(
            stage_context.source_df,
            stage_context.features,
            self.depth_levels,
            self.rolling_windows,
        )
        LOG.info(f"  Book Change: +{len(stage_context.features) - before} features")
