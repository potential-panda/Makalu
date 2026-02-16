import logging
from typing import List

from context import Context

from feature_engineering.trend import build_trend_features
from pipeline.framework import PipelineContext, Step

LOG = logging.getLogger(__name__)


class TrendFeatureStep(Step):
    def __init__(self, windows: List[str]):
        self.windows = windows

    def populate(self, pipeline_context: PipelineContext, stage_context: Context):
        before = len(stage_context.features)
        build_trend_features(stage_context.source_df, stage_context.features, self.windows)
        LOG.info(f"  Trend: +{len(stage_context.features) - before} features")

    def populate_influxdb(self, pipeline_context: PipelineContext, stage_context: Context):
        pass
