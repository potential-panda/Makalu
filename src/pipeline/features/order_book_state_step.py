import logging
from typing import List

from feature_engineering.order_book_state import build_order_book_state_features
from pipeline.framework import PipelineContext, Step

from .context import Context

LOG = logging.getLogger(__name__)


class OrderBookStateFeatureStep(Step):
    def __init__(self, depth_levels: List[int]):
        self.depth_levels = depth_levels

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        before = len(stage_context.features)
        build_order_book_state_features(stage_context.source_df, stage_context.features, self.depth_levels)
        LOG.info(f"  Order Book State: +{len(stage_context.features) - before} features")

    def populate_influxdb(self, pipeline_context: PipelineContext, stage_context: Context):
        pass
