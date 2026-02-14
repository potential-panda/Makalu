import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import List

from qate.store.influx_writer import InfluxdbStore
from qate.store.market_data_store import MarketDataStore
from qate.store.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

CHUNK_SIZE = 7
LOG = getLogger(__name__)


@dataclass
class StageContext:
    populate_dt_range: DtRange


@dataclass
class PipelineContext:
    market_data_store: MarketDataStore
    parquet_store: ParquetStore
    influx_store: InfluxdbStore


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
            LOG.info(f"{self.name}: {self.cnt}/{self.total} processed ({progress:.2f}%)")


class Step(ABC):
    @abstractmethod
    def populate(self, pipeline_context: PipelineContext, stage_context: StageContext):
        pass

    @abstractmethod
    def populate_influxdb(self, pipeline_context: PipelineContext, stage_context: StageContext):
        pass


class Stage(ABC):
    def __init__(self):
        self._steps: list[Step] = []

    def step(self, step: Step):
        self._steps.append(step)
        return self

    def before(self, dt_range: DtRange, pipeline_context: PipelineContext) -> StageContext:
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
            LOG.info(f"Step {s.__class__.__name__} finished in {step_end_time - step_start_time:.2f} seconds.")

        self.after()
        stage_end_time = time.time()
        LOG.info(f"Stage {self.__class__.__name__} finished in {stage_end_time - stage_start_time:.2f} seconds.")


class Pipeline:
    def __init__(
        self, market_data_store: MarketDataStore, parquet_store: ParquetStore, influx_store: InfluxdbStore = None
    ):
        self._context = PipelineContext(market_data_store, parquet_store, influx_store)
        self._stages: List[Stage] = []

    def stage(self, stage: Stage):
        self._stages.append(stage)
        return self

    def process(self, dt_range: DtRange):
        chunks = dt_range.chunks(CHUNK_SIZE)
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            LOG.info(
                f"Starting pipeline process for chunk {i + 1}/{total_chunks} ({chunk.start_str} to {chunk.end_str})"
            )
            for s in self._stages:
                s.run(chunk, self._context)
            chunk_end_time = time.time()
            LOG.info(f"Chunk {i + 1}/{total_chunks} processed in {chunk_end_time - chunk_start_time:.2f} seconds.")
