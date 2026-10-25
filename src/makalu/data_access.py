"""
Shared data-access helpers for the Makalu data warehouse.

Used by the CLI (data-check) to query dataset availability.
"""

from __future__ import annotations

from pathlib import Path

from makalu.cli.common import DATASETS, get_dataset_root
from makalu.data_outage import filter_bronze_outage_missing_dates


# ── Per-date scanning ─────────────────────────────────────────────────────────


def _list_partition_values(path: Path, key: str) -> list[str]:
    prefix = f"{key}="
    start = len(prefix)
    return sorted(
        d.name[start:]
        for d in path.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    )


def _collect_date_sizes(path: Path) -> dict[str, int]:
    prefix = "date="
    sizes: dict[str, int] = {}
    for d in path.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            date_str = d.name[len(prefix):]
            total = 0
            direct = d / "data.parquet"
            if direct.exists():
                total += direct.stat().st_size
            else:
                for parquet in d.rglob("data.parquet"):
                    total += parquet.stat().st_size
            if total > 0:
                sizes[date_str] = total
    return sizes


def _missing_dates(dates: list[str], start: str | None, end: str | None) -> list[str]:
    from datetime import date, timedelta

    if not dates:
        return []
    d_start = date.fromisoformat(start) if start else date.fromisoformat(dates[0])
    d_end = date.fromisoformat(end) if end else date.fromisoformat(dates[-1])
    existing = set(dates)
    missing = []
    cur = d_start
    while cur <= d_end:
        s = cur.isoformat()
        if s not in existing:
            missing.append(s)
        cur += timedelta(days=1)
    return missing


def _to_ranges(dates: list[str]) -> list[str]:
    from datetime import date

    if not dates:
        return []
    ranges = []
    start = prev = date.fromisoformat(dates[0])
    for d in dates[1:]:
        cur = date.fromisoformat(d)
        if (cur - prev).days == 1:
            prev = cur
        else:
            ranges.append(_range_str(start, prev))
            start = prev = cur
    ranges.append(_range_str(start, prev))
    return ranges


def _range_str(start, end) -> str:
    if start == end:
        return str(start)
    return f"{start} → {end}"


# ── Availability ──────────────────────────────────────────────────────────────


def dataset_availability(
    dataset: str,
    partitions: dict[str, str | None] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Return progressive-disclosure availability for one dataset."""
    if dataset not in DATASETS:
        return {"error": f"Unknown dataset '{dataset}'", "state": "unknown_dataset"}

    layer, keys = DATASETS[dataset]
    base = get_dataset_root(layer, dataset)
    provided = {key: (partitions or {}).get(key) for key in keys}

    result = {
        "dataset": dataset,
        "layer": layer,
        "keys": keys,
        "base_path": str(base),
        "exists": base.exists(),
        "requested_partitions": provided,
        "resolved_partitions": {},
        "state": None,
    }
    if not base.exists():
        result["state"] = "missing_dataset"
        return result

    path = base
    for key in keys:
        value = provided.get(key)
        if value is None:
            result.update({
                "state": "needs_partition",
                "next_key": key,
                "available_values": _list_partition_values(path, key),
                "path": str(path),
                "resolved_partitions": {k: v for k, v in provided.items() if v is not None},
            })
            return result

        next_path = path / f"{key}={value}"
        if not next_path.exists():
            result.update({
                "state": "missing_partition",
                "missing_key": key,
                "missing_value": value,
                "available_values": _list_partition_values(path, key),
                "path": str(next_path),
                "resolved_partitions": {k: v for k, v in provided.items() if v is not None},
            })
            return result
        path = next_path

    date_sizes = _collect_date_sizes(path)
    dates = sorted(date_sizes)
    missing = _missing_dates(dates, start, end)
    if dataset in {"Trade", "OrderBook"}:
        missing = filter_bronze_outage_missing_dates(dataset, missing)
    window_start = start or (dates[0] if dates else None)
    window_end = end or (dates[-1] if dates else None)
    date_map: dict[str, dict[str, int | bool]] = {}
    if window_start and window_end:
        from datetime import date, timedelta

        cur = date.fromisoformat(window_start)
        end_dt = date.fromisoformat(window_end)
        while cur <= end_dt:
          iso = cur.isoformat()
          date_map[iso] = {
              "present": iso in date_sizes,
              "size": date_sizes.get(iso, 0),
          }
          cur += timedelta(days=1)
    result.update({
        "state": "summary",
        "path": str(path),
        "resolved_partitions": provided,
        "min_date": dates[0] if dates else None,
        "max_date": dates[-1] if dates else None,
        "total_days": len(dates),
        "window_start": window_start,
        "window_end": window_end,
        "missing_days": len(missing),
        "missing_ranges": _to_ranges(missing),
        "has_data": bool(dates),
        "available_dates": dates,
        "date_map": date_map,
    })
    return result
