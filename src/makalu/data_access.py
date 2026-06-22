"""
Shared data-access helpers for the Makalu data warehouse.

Used by both the CLI (data-check) and the HTTP API to query availability,
list markets, and compute storage stats without duplicating filesystem logic.
"""

from __future__ import annotations

import re
from pathlib import Path

import qate.env.sys_env as sys_env

from makalu.cli.common import DATASETS, get_dataset_root
from makalu.data_outage import bronze_outage_dates_between

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
