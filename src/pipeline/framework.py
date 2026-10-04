import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import List

from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

from qate_market_data.store import MarketDataStore

CHUNK_SIZE = 3
LOG = getLogger(__name__)


class NoDataForChunk(Exception):
    """Raise from a Stage.before() when the underlying read genuinely has no
    data for this chunk (e.g. a real gap in upstream raw data collection,
    confirmed for real: BINANCE/ETH_USDT bronze is missing an entire month,
    2026-03-29..2026-04-30, and several other markets have scattered
    week-long gaps). Pipeline.process() catches this per-chunk and skips
    forward, rather than the whole populate command crashing on the first
    missing day and never reaching any later, available data -- confirmed
    for real: a single populate call spanning ~90 days with one missing
    week in the middle never got anywhere near the later dates before this
    fix, regardless of how long it ran."""


@dataclass
class StageContext:
    populate_dt_range: DtRange


@dataclass
class PipelineContext:
    market_data_store: MarketDataStore
    parquet_store: ParquetStore


class StepProgressLogger:
    def __init__(self, name: str, total: int):
        self.name = name
        self.total = total
        self.last_log_time = time.time()
        self.cnt = 0

    def update(self):
        self.cnt += 1
        now = time.time()
        if now - self.last_log_time > 180:
            self.last_log_time = now
            progress = (self.cnt / self.total) * 100
            LOG.info(
                f"{self.name}: {self.cnt}/{self.total} processed ({progress:.2f}%)"
            )


class Step(ABC):
    @abstractmethod
    def populate(self, pipeline_context: PipelineContext, stage_context: StageContext):
        pass


class Stage(ABC):
    def __init__(self):
        self._steps: list[Step] = []

    def step(self, step: Step):
        self._steps.append(step)
        return self

    def before(
        self, dt_range: DtRange, pipeline_context: PipelineContext
    ) -> StageContext:
        pass

    def after(self):
        pass

    def run(self, dt_range: DtRange, pipeline_context: PipelineContext):
        stage_start_time = time.time()
        LOG.info(f"Running stage {self.__class__.__name__}...")
        stage_context = self.before(dt_range, pipeline_context)

        for s in self._steps:
            step_start_time = time.time()
            LOG.info(f"Running step {s.__class__.__name__}...")
            s.populate(pipeline_context, stage_context)
            step_end_time = time.time()
            LOG.info(
                f"Step {s.__class__.__name__} finished in {step_end_time - step_start_time:.2f} seconds."
            )

        self.after()
        stage_end_time = time.time()
        LOG.info(
            f"Stage {self.__class__.__name__} finished in {stage_end_time - stage_start_time:.2f} seconds."
        )


class Pipeline:
    def __init__(self, market_data_store: MarketDataStore, parquet_store: ParquetStore):
        self._context = PipelineContext(market_data_store, parquet_store)
        self._stages: List[Stage] = []

    def stage(self, stage: Stage):
        self._stages.append(stage)
        return self

    def process(self, dt_range: DtRange):
        chunks = dt_range.chunks(CHUNK_SIZE)
        total_chunks = len(chunks)
        skipped_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            LOG.info(
                f"Starting pipeline process for chunk {i + 1}/{total_chunks} ({chunk.start_str} to {chunk.end_str})"
            )
            try:
                for s in self._stages:
                    s.run(chunk, self._context)
            except NoDataForChunk as e:
                LOG.warning(
                    f"Chunk {i + 1}/{total_chunks} ({chunk.start_str} to {chunk.end_str}) skipped: {e}"
                )
                skipped_chunks.append((chunk.start_str, chunk.end_str))
                continue
            chunk_end_time = time.time()
            LOG.info(
                f"Chunk {i + 1}/{total_chunks} processed in {chunk_end_time - chunk_start_time:.2f} seconds."
            )

        if skipped_chunks:
            LOG.warning(
                f"Skipped {len(skipped_chunks)}/{total_chunks} chunk(s) with no data: {skipped_chunks}"
            )
