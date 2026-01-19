import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List

from qate.store.influx_writer import InfluxWriter
from qate.util.dt_range import DtRange

CHUNK_SIZE = 7
LOG = getLogger(__name__)


@dataclass
class ContextBase:
    compute_dt_range: DtRange
    populate_dt_range: DtRange
    parquet_root_dir: str


class StepProgressLogger:
    """Utility class to log progress of steps."""

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
    def populate(self, influx_writer: InfluxWriter, context: ContextBase, args: Dict[str, Any]):
        """
        Executes the logic of the step, populating the context.
        Args:
            influx_writer: The InfluxDB writer instance.
            context: The shared context object for the stage.
            args: Additional arguments for the step (e.g., date range, specific configs).
        """
        pass


class Stage(ABC):
    """Abstract base class for a pipeline stage."""

    def __init__(self, influx_writer: InfluxWriter):
        self.influx_writer = influx_writer
        self.steps: list[Step] = []

    def step(self, step: Step):
        """Adds a step to the stage."""
        self.steps.append(step)
        return self

    @abstractmethod
    def before(self, dt_range: DtRange, args: Dict[str, Any]) -> ContextBase:
        """
        Hook executed before running steps, used for data loading or context setup.
        Args:
            dt_range: A DtRange object representing the start and end datetime for processing.
            args: Additional arguments for the stage.
        """
        pass

    def after(self, args: Dict[str, Any]):
        """
        Hook executed after running steps, used for cleanup or finalization.
        Args:
            args: Additional arguments for the stage.
        """
        self.influx_writer.close()

    def run(self, dt_range: DtRange, args: Dict[str, Any]):
        """
        Runs all steps within the stage.
        Args:
            dt_range: A DtRange object representing the start and end datetime for processing.
            args: Additional arguments for the stage.
        """
        stage_start_time = time.time()
        LOG.info(f"Running stage {self.__class__.__name__}...")
        context = self.before(dt_range, args)

        for s in self.steps:
            step_start_time = time.time()
            LOG.info(f"Running step {s.__class__.__name__}...")
            s.populate(self.influx_writer, context, args)
            step_end_time = time.time()
            LOG.info(f"Step {s.__class__.__name__} finished in {step_end_time - step_start_time:.2f} seconds.")

        self.after(args)
        stage_end_time = time.time()
        LOG.info(f"Stage {self.__class__.__name__} finished in {stage_end_time - stage_start_time:.2f} seconds.")


class Pipeline:
    """
    Orchestrates the execution of different stages of the data pipeline.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config
        self.stages: List[Stage] = []

    def add_stage(self, stage: Stage):
        self.stages.append(stage)
        return self

    def process(self, dt_range: DtRange, args: Dict[str, Any] = None):
        """
        Executes all stages of the pipeline for a given date range.
        Args:
            dt_range: A DtRange object representing the start and end datetime for processing.
            args: Additional arguments for the pipeline execution.
        """
        if args is None:
            args = {}

        chunks = dt_range.chunks(CHUNK_SIZE)
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            LOG.info(
                f"Starting pipeline process for chunk {i + 1}/{total_chunks} ({chunk.start_str} to {chunk.end_str})"
            )
            for s in self.stages:
                s.run(chunk, args)
            chunk_end_time = time.time()
            LOG.info(f"Chunk {i + 1}/{total_chunks} processed in {chunk_end_time - chunk_start_time:.2f} seconds.")
