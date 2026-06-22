"""Known structural data gaps.

The bronze collectors crashed from 2026-04-02 through 2026-04-09 inclusive.
We intentionally model that window as available-but-empty for bronze Trade and
OrderBook: availability checks should not ask callers to backfill it, and reads
should return empty results so caller behavior stays consistent.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd

BRONZE_OUTAGE_START = date(2026, 4, 2)
BRONZE_OUTAGE_END = date(2026, 4, 9)
BRONZE_OUTAGE_DATASETS = {"Trade", "OrderBook"}


def _parse_date(value: str) -> date:
    if "-" in value:
        return date.fromisoformat(value)
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def is_bronze_outage_date(value: str) -> bool:
    day = _parse_date(value)
    return BRONZE_OUTAGE_START <= day <= BRONZE_OUTAGE_END


def is_bronze_outage_dataset(dataset: str) -> bool:
    return dataset in BRONZE_OUTAGE_DATASETS


def is_bronze_outage_store(store: Any) -> bool:
    dataset = getattr(store, "dataset", None)
    root_dir = getattr(store, "root_dir", "")
    return dataset in BRONZE_OUTAGE_DATASETS and Path(root_dir).name == "bronze"


def bronze_outage_dates_between(start: str | None, end: str | None) -> list[str]:
    start_date = _parse_date(start) if start else BRONZE_OUTAGE_START
    end_date = _parse_date(end) if end else BRONZE_OUTAGE_END
    cur = max(start_date, BRONZE_OUTAGE_START)
    last = min(end_date, BRONZE_OUTAGE_END)
    dates = []
    while cur <= last:
        dates.append(cur.isoformat())
        cur += timedelta(days=1)
    return dates


def filter_bronze_outage_missing_dates(dataset: str, missing: list[str]) -> list[str]:
    if not is_bronze_outage_dataset(dataset):
        return missing
    return [day for day in missing if not is_bronze_outage_date(day)]


def install_qate_bronze_outage_patch() -> None:
    """Patch qate parquet reads to return empty data for the known bronze gap.

    qate itself only sees missing files. Makalu knows that this exact bronze
    outage is expected and should behave like an available empty partition.
    """
    from qate.store.parquet_store import TimeSeriesStore

    if getattr(TimeSeriesStore, "_makalu_bronze_outage_patch", False):
        return

    original_read_a_day = TimeSeriesStore.read_a_day
    original_read = TimeSeriesStore.read

    @wraps(original_read_a_day)
    def read_a_day_with_bronze_outage(self, date_str):
        if is_bronze_outage_store(self) and is_bronze_outage_date(str(date_str)):
            # Known missing bronze window. Return an empty frame instead of
            # treating the missing data.parquet file as an unexpected absence.
            return pd.DataFrame(columns=["ts"])
        return original_read_a_day(self, date_str)

    @wraps(original_read)
    def read_with_bronze_outage(self, date_arg):
        try:
            return original_read(self, date_arg)
        except KeyError as exc:
            days = getattr(date_arg, "days", date_arg if isinstance(date_arg, list) else None)
            if (
                days
                and is_bronze_outage_store(self)
                and all(is_bronze_outage_date(str(day)) for day in days)
            ):
                # All requested days are the known bronze outage. Return an
                # empty dataframe with ts so downstream list conversions become
                # [] and dataframe callers can filter/sort consistently.
                return pd.DataFrame(columns=["ts"])
            raise exc

    TimeSeriesStore.read_a_day = read_a_day_with_bronze_outage
    TimeSeriesStore.read = read_with_bronze_outage
    TimeSeriesStore._makalu_bronze_outage_patch = True
