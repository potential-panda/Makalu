from datetime import timedelta
from logging import getLogger
from typing import Optional

import pandas as pd
from qate.core.model import Market
from qate.util.dt_range import DtRange

from pipeline.framework import PipelineContext, Stage, StageContext

from .context import Context

LOG = getLogger(__name__)


class FeatureStage(Stage):
    def __init__(self, market: Market, execution_size_str: str, target: str):
        super().__init__()
        self.market = market
        self.execution_size_str = execution_size_str
        self.target = target

    def before(
        self, dt_range: DtRange, pipeline_context: PipelineContext
    ) -> StageContext:
        compute_dt_range = DtRange(
            dt_range.start_dt - timedelta(hours=1), dt_range.end_dt
        )

        # Load all 4 data sources
        execution_price_df = pipeline_context.parquet_store.create_execution_price(
            self.market, self.execution_size_str
        ).read(compute_dt_range)
        trade_aggression_df = pipeline_context.parquet_store.create_single_market(
            "TradeAggression", self.market
        ).read(compute_dt_range)
        order_book_state_df = pipeline_context.parquet_store.create_single_market(
            "BookState", self.market
        ).read(compute_dt_range)
        book_change_df = pipeline_context.parquet_store.create_single_market(
            "BookChange", self.market
        ).read(compute_dt_range)

        # Align into a single source_df
        source_df = self._align_data_sources(
            trade_aggression_df,
            order_book_state_df,
            book_change_df,
            execution_price_df,
        )

        return Context(
            populate_dt_range=dt_range,
            market=self.market,
            target=self.target,
            source_df=source_df,
            features={},
        )

    @staticmethod
    def _align_data_sources(
        trade_aggression_df: pd.DataFrame,
        order_book_state_df: pd.DataFrame,
        book_change_df: pd.DataFrame,
        execution_price_df: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """Align all data sources into a single DataFrame with DatetimeIndex."""
        dfs_to_merge = []

        for df, name in [
            (trade_aggression_df, "trade_aggression"),
            (order_book_state_df, "order_book_state"),
            (book_change_df, "book_change"),
            (execution_price_df, "execution_price"),
        ]:
            if df is not None and not df.empty:
                df = df.copy()
                if "ts" in df.columns:
                    df.index = pd.to_datetime(df["ts"], unit="s")
                    df = df.drop(columns=["ts"])
                if not df.index.is_monotonic_increasing:
                    df = df.sort_index()
                dfs_to_merge.append((df, name))
                LOG.info(f"FeatureStage: Loaded {name} with {len(df)} rows")

        if not dfs_to_merge:
            return None

        result = dfs_to_merge[0][0].copy()
        for df, name in dfs_to_merge[1:]:
            result = result.join(df, how="outer", rsuffix="_dup")
            dup_cols = [col for col in result.columns if col.endswith("_dup")]
            if dup_cols:
                result = result.drop(columns=dup_cols)

        result = result.sort_index()
        with pd.option_context("future.no_silent_downcasting", True):
            result = result.ffill()
        result = result.infer_objects(copy=False)
        LOG.info(
            f"FeatureStage: Aligned {len(result)} rows with {len(result.columns)} columns"
        )

        return result
