"""
Shared data-access helpers for the Makalu data warehouse.

Used by both the CLI (data-check) and the HTTP API to query availability,
list markets, and compute storage stats without duplicating filesystem logic.
"""

from __future__ import annotations

import re
from pathlib import Path

import makalu.settings as sys_env

from makalu.cli.common import DATASETS, get_dataset_root
from makalu.data_outage import bronze_outage_dates_between, filter_bronze_outage_missing_dates

KNOWN_EXCHANGES = ["BINANCE", "GMO", "BITBANK", "COINCHECK"]

# Datasets grouped by layer for availability scanning
_LAYER_DATASETS: dict[str, list[str]] = {
    "bronze": ["Trade", "OrderBook"],
    "silver": ["BookChange", "BookState", "TradeAggression", "MarketPrice"],
    "gold":   ["Chart_CANDLE", "Chart_HEIKIN_ASHI", "Chart_RANGE", "VolSpread", "LeadLag"],
}


def raw_dir() -> Path:
    return Path(sys_env.MARKET_DATA_DIR)


def parquet_root() -> Path:
    return Path(sys_env.PARQUET_ROOT_DIR)


# ── Markets ───────────────────────────────────────────────────────────────────

def list_markets() -> list[dict[str, str]]:
    """Return all markets found in the raw msgpack directory.

    Each entry: {"market": "BINANCE_BTC_USDT", "exchange": "BINANCE", "symbol": "BTC_USDT"}
    """
    seen: dict[str, dict] = {}
    pattern = re.compile(r'^(.+)_\d{8}\.msgpack$')
    for f in sorted(raw_dir().glob("*.msgpack")):
        m = pattern.match(f.name)
        if not m:
            continue
        market_key = m.group(1)
        if market_key in seen:
            continue
        for ex in KNOWN_EXCHANGES:
            if market_key.startswith(ex + "_"):
                seen[market_key] = {
                    "market": market_key,
                    "exchange": ex,
                    "symbol": market_key[len(ex) + 1:],
                }
                break
    return list(seen.values())


# ── Per-date scanning ─────────────────────────────────────────────────────────

def _scan_layer_dates(layer: str, dataset: str, exchange: str, symbol: str) -> dict[str, int]:
    """Return {date_iso: total_bytes} for one dataset/exchange/symbol partition."""
    base = get_dataset_root(layer, dataset) / f"exchange={exchange}" / f"symbol={symbol}"
    result: dict[str, int] = {}
    if not base.exists():
        return result
    for date_dir in base.iterdir():
        if not date_dir.is_dir() or not date_dir.name.startswith("date="):
            continue
        date_str = date_dir.name[5:]
        parquet = date_dir / "data.parquet"
        if parquet.exists():
            result[date_str] = result.get(date_str, 0) + parquet.stat().st_size
        else:
            # Sub-partitioned (execution_size, param_id, …) — aggregate all parquet files
            for sub in date_dir.rglob("data.parquet"):
                result[date_str] = result.get(date_str, 0) + sub.stat().st_size
    return result


def _merge(acc: dict[str, int], delta: dict[str, int]) -> None:
    for date, size in delta.items():
        acc[date] = acc.get(date, 0) + size


def _list_partition_values(path: Path, key: str) -> list[str]:
    prefix = f"{key}="
    start = len(prefix)
    return sorted(
        d.name[start:]
        for d in path.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    )


def _collect_dates(path: Path) -> list[str]:
    prefix = "date="
    dates: list[str] = []
    for d in path.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            parquet = d / "data.parquet"
            if parquet.exists():
                dates.append(d.name[len(prefix):])
    return sorted(dates)


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

def availability(exchange: str, symbol: str) -> dict:
    """Return per-date layer availability and summary stats for one market.

    Returns::

        {
            "dates": {
                "2026-01-01": {"raw": int, "bronze": int, "silver": int, "gold": int}
            },
            "stats": {
                "raw_total": int, "bronze_total": int, ...
                "raw_dates": int, "bronze_dates": int, ...
            }
        }
    """
    market_key = f"{exchange}_{symbol}"

    # Raw msgpack files
    raw: dict[str, int] = {}
    pat = re.compile(rf'^{re.escape(market_key)}_(\d{{8}})\.msgpack$')
    for f in raw_dir().glob(f"{market_key}_*.msgpack"):
        m = pat.match(f.name)
        if m:
            ds = m.group(1)
            iso = f"{ds[:4]}-{ds[4:6]}-{ds[6:]}"
            raw[iso] = f.stat().st_size

    # Parquet layers
    layers: dict[str, dict[str, int]] = {"bronze": {}, "silver": {}, "gold": {}}
    for layer, datasets in _LAYER_DATASETS.items():
        for ds in datasets:
            _merge(layers[layer], _scan_layer_dates(layer, ds, exchange, symbol))
    if raw or layers["bronze"] or layers["silver"] or layers["gold"]:
        for day in bronze_outage_dates_between(None, None):
            # Known missing bronze collector outage. Represent it as truthy,
            # available empty bronze data so API/UI callers do not try to
            # backfill it, while reads return [] via makalu.data_outage.
            layers["bronze"].setdefault(day, 1)

    all_dates = sorted(set(raw) | set(layers["bronze"]) | set(layers["silver"]) | set(layers["gold"]))
    dates = {
        d: {
            "raw":    raw.get(d, 0),
            "bronze": layers["bronze"].get(d, 0),
            "silver": layers["silver"].get(d, 0),
            "gold":   layers["gold"].get(d, 0),
        }
        for d in all_dates
    }

    return {
        "dates": dates,
        "stats": {
            "raw_total":    sum(raw.values()),
            "bronze_total": sum(layers["bronze"].values()),
            "silver_total": sum(layers["silver"].values()),
            "gold_total":   sum(layers["gold"].values()),
            "raw_dates":    len(raw),
            "bronze_dates": len(layers["bronze"]),
            "silver_dates": len(layers["silver"]),
            "gold_dates":   len(layers["gold"]),
        },
    }


def dataset_catalog() -> list[dict]:
    """Return static dataset metadata from the warehouse catalog."""
    rows = []
    for dataset, (layer, keys) in DATASETS.items():
        rows.append({
            "dataset": dataset,
            "layer": layer,
            "keys": keys,
            "path": str(get_dataset_root(layer, dataset)),
        })
    return rows


def dataset_overview() -> list[dict]:
    """Return a cheap inventory summary for each dataset."""
    rows = []
    for dataset, (layer, keys) in DATASETS.items():
        base = get_dataset_root(layer, dataset)
        if not base.exists():
            rows.append({
                "dataset": dataset,
                "layer": layer,
                "keys": keys,
                "exists": False,
                "file_count": 0,
                "date_count": 0,
                "earliest_date": None,
                "latest_date": None,
                "total_bytes": 0,
                "path": str(base),
            })
            continue

        file_count = 0
        total_bytes = 0
        dates: set[str] = set()
        for parquet in base.rglob("data.parquet"):
            file_count += 1
            try:
                total_bytes += parquet.stat().st_size
            except FileNotFoundError:
                continue
            parent = parquet.parent
            while parent != base and not parent.name.startswith("date="):
                parent = parent.parent
            if parent.name.startswith("date="):
                dates.add(parent.name[5:])

        rows.append({
            "dataset": dataset,
            "layer": layer,
            "keys": keys,
            "exists": True,
            "file_count": file_count,
            "date_count": len(dates),
            "earliest_date": min(dates) if dates else None,
            "latest_date": max(dates) if dates else None,
            "total_bytes": total_bytes,
            "path": str(base),
        })
    return rows


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


# ── Global stats ──────────────────────────────────────────────────────────────

def global_stats() -> dict:
    """Return total byte counts across all markets and a raw file count.

    Walks the full parquet tree — cache the result externally if needed.
    """
    raw_files = list(raw_dir().glob("*.msgpack"))

    def _layer_total(layer: str) -> int:
        d = parquet_root() / layer
        return sum(f.stat().st_size for f in d.rglob("data.parquet")) if d.exists() else 0

    return {
        "raw":            sum(f.stat().st_size for f in raw_files),
        "raw_file_count": len(raw_files),
        "bronze":         _layer_total("bronze"),
        "silver":         _layer_total("silver"),
        "gold":           _layer_total("gold"),
    }
