#!/usr/bin/env python3
"""Makalu data warehouse HTTP API — port 9776."""

import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

RAW_DIR = Path("/panda/market-data")
PARQUET_ROOT = Path("/panda/makalu-data")
WEB_DIR = Path(__file__).parent.parent / "web"

BRONZE_DATASETS = ["Trade", "OrderBook"]
SILVER_DATASETS = ["BookChange", "BookState", "TradeAggression", "MarketPrice"]
GOLD_DATASETS = ["Chart_CANDLE", "Chart_HEIKIN_ASHI", "Chart_RANGE", "VolSpread", "LeadLag"]
KNOWN_EXCHANGES = ["BINANCE", "GMO", "BITBANK", "COINCHECK"]

POPULATE_DATASETS = [
    ("bronze", "bronze"),
    ("silver", "BookChange"),
    ("silver", "BookState"),
    ("silver", "TradeAggression"),
    ("silver", "MarketPrice"),
]

app = Flask(__name__, static_folder=str(WEB_DIR))

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Stats cache
_stats_cache: dict = {}
_stats_cache_ts: float = 0
STATS_CACHE_TTL = 300  # 5 minutes


def _scan_parquet_dates(layer: str, dataset: str, exchange: str, symbol: str) -> dict[str, int]:
    """Return {date: total_bytes} for a given dataset/exchange/symbol."""
    base = PARQUET_ROOT / layer / f"dataset={dataset}" / f"exchange={exchange}" / f"symbol={symbol}"
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
            for sub in date_dir.rglob("data.parquet"):
                result[date_str] = result.get(date_str, 0) + sub.stat().st_size
    return result


def _layer_total(layer: str) -> int:
    layer_dir = PARQUET_ROOT / layer
    if not layer_dir.exists():
        return 0
    return sum(f.stat().st_size for f in layer_dir.rglob("data.parquet"))


@app.route("/api/markets")
def api_markets():
    markets: dict[str, dict] = {}
    pattern = re.compile(r'^(.+)_(\d{8})\.msgpack$')
    for f in sorted(RAW_DIR.glob("*.msgpack")):
        m = pattern.match(f.name)
        if not m:
            continue
        market_key = m.group(1)
        if market_key in markets:
            continue
        for ex in KNOWN_EXCHANGES:
            if market_key.startswith(ex + "_"):
                sym = market_key[len(ex) + 1:]
                markets[market_key] = {"market": market_key, "exchange": ex, "symbol": sym}
                break
    return jsonify(sorted(markets.values(), key=lambda x: x["market"]))


@app.route("/api/availability")
def api_availability():
    exchange = request.args.get("exchange", "")
    symbol = request.args.get("symbol", "")
    if not exchange or not symbol:
        return jsonify({"error": "exchange and symbol required"}), 400

    market_key = f"{exchange}_{symbol}"
    raw: dict[str, int] = {}
    pat = re.compile(rf'^{re.escape(market_key)}_(\d{{8}})\.msgpack$')
    for f in RAW_DIR.glob(f"{market_key}_*.msgpack"):
        m = pat.match(f.name)
        if m:
            ds = m.group(1)
            iso = f"{ds[:4]}-{ds[4:6]}-{ds[6:]}"
            raw[iso] = f.stat().st_size

    bronze: dict[str, int] = {}
    for ds in BRONZE_DATASETS:
        for d, sz in _scan_parquet_dates("bronze", ds, exchange, symbol).items():
            bronze[d] = bronze.get(d, 0) + sz

    silver: dict[str, int] = {}
    for ds in SILVER_DATASETS:
        for d, sz in _scan_parquet_dates("silver", ds, exchange, symbol).items():
            silver[d] = silver.get(d, 0) + sz

    gold: dict[str, int] = {}
    for ds in GOLD_DATASETS:
        for d, sz in _scan_parquet_dates("gold", ds, exchange, symbol).items():
            gold[d] = gold.get(d, 0) + sz

    all_dates = sorted(set(raw) | set(bronze) | set(silver) | set(gold))
    dates = {
        d: {
            "raw": raw.get(d, 0),
            "bronze": bronze.get(d, 0),
            "silver": silver.get(d, 0),
            "gold": gold.get(d, 0),
        }
        for d in all_dates
    }

    return jsonify({
        "dates": dates,
        "stats": {
            "raw_total": sum(raw.values()),
            "bronze_total": sum(bronze.values()),
            "silver_total": sum(silver.values()),
            "gold_total": sum(gold.values()),
            "raw_dates": len(raw),
            "bronze_dates": len(bronze),
            "silver_dates": len(silver),
            "gold_dates": len(gold),
        },
    })


@app.route("/api/stats")
def api_stats():
    global _stats_cache, _stats_cache_ts
    if time.time() - _stats_cache_ts < STATS_CACHE_TTL and _stats_cache:
        return jsonify(_stats_cache)

    raw_files = list(RAW_DIR.glob("*.msgpack"))
    _stats_cache = {
        "raw": sum(f.stat().st_size for f in raw_files),
        "raw_file_count": len(raw_files),
        "bronze": _layer_total("bronze"),
        "silver": _layer_total("silver"),
        "gold": _layer_total("gold"),
    }
    _stats_cache_ts = time.time()
    return jsonify(_stats_cache)


def _run_job(job_id: str, cmd: list[str], cwd: str) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = time.time()

    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        with _jobs_lock:
            _jobs[job_id]["log"] = lines[-200:]

    proc.wait()
    with _jobs_lock:
        _jobs[job_id]["returncode"] = proc.returncode
        _jobs[job_id]["status"] = "done" if proc.returncode == 0 else "failed"
        _jobs[job_id]["finished_at"] = time.time()


@app.route("/api/ops/download", methods=["POST"])
def api_ops_download():
    data = request.json or {}
    market = data.get("market", "")
    start = data.get("start", "")
    end = data.get("end", "")
    if not start or not end:
        return jsonify({"error": "start and end required"}), 400

    from datetime import date
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    cmd = ["python", "cli.py", "market-data", "--days", str(days)]
    if market:
        cmd += ["--market", market]

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "type": "download",
            "label": f"download {market or 'all'} {start}→{end}",
            "cmd": " ".join(cmd),
            "status": "pending",
            "log": [],
            "created_at": time.time(),
        }

    threading.Thread(
        target=_run_job,
        args=(job_id, cmd, str(Path(__file__).parent.parent)),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/ops/populate", methods=["POST"])
def api_ops_populate():
    data = request.json or {}
    dataset = data.get("dataset", "")
    exchange = data.get("exchange", "")
    symbol = data.get("symbol", "")
    start = data.get("start", "")
    end = data.get("end", "")
    if not all([dataset, exchange, symbol, start, end]):
        return jsonify({"error": "dataset, exchange, symbol, start, end required"}), 400

    cmd = [
        "python", "cli.py", "populate", dataset,
        "--exchange", exchange,
        "--symbol", symbol,
        "--start", start,
        "--end", end,
    ]

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "type": "populate",
            "label": f"populate {dataset} {exchange}:{symbol} {start}→{end}",
            "cmd": " ".join(cmd),
            "status": "pending",
            "log": [],
            "created_at": time.time(),
        }

    threading.Thread(
        target=_run_job,
        args=(job_id, cmd, str(Path(__file__).parent.parent)),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/jobs")
def api_jobs():
    with _jobs_lock:
        return jsonify(list(reversed(list(_jobs.values()))))


@app.route("/api/jobs/<job_id>")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(job)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_web(path: str):
    if path.startswith("api/"):
        return jsonify({"error": "not found"}), 404
    fp = WEB_DIR / path
    if path and fp.is_file():
        return send_from_directory(str(WEB_DIR), path)
    return send_from_directory(str(WEB_DIR), "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9776, debug=False)
