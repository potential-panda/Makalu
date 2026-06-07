#!/usr/bin/env python3
"""Makalu data warehouse HTTP API — port 9776."""

import subprocess
import sys
import threading
import time
import uuid
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from makalu.data_access import availability, global_stats, list_markets

WEB_DIR = Path(__file__).parent.parent / "web"
CLI = Path(__file__).parent / "cli.py"

app = Flask(__name__, static_folder=str(WEB_DIR))

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

_stats_cache: dict = {}
_stats_cache_ts: float = 0
STATS_CACHE_TTL = 300


# ── Data endpoints ────────────────────────────────────────────────────────────

@app.route("/api/markets")
def api_markets():
    return jsonify(list_markets())


@app.route("/api/availability")
def api_availability():
    exchange = request.args.get("exchange", "")
    symbol = request.args.get("symbol", "")
    if not exchange or not symbol:
        return jsonify({"error": "exchange and symbol required"}), 400
    return jsonify(availability(exchange, symbol))


@app.route("/api/stats")
def api_stats():
    global _stats_cache, _stats_cache_ts
    if time.time() - _stats_cache_ts < STATS_CACHE_TTL and _stats_cache:
        return jsonify(_stats_cache)
    _stats_cache = global_stats()
    _stats_cache_ts = time.time()
    return jsonify(_stats_cache)


# ── Job runner ────────────────────────────────────────────────────────────────

def _run_job(job_id: str, cmd: list[str]) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = time.time()

    proc = subprocess.Popen(
        cmd, cwd=str(CLI.parent.parent),
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


def _submit(job_id: str, job_meta: dict, cmd: list[str]) -> None:
    with _jobs_lock:
        _jobs[job_id] = {**job_meta, "cmd": " ".join(cmd), "status": "pending", "log": [], "created_at": time.time()}
    threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True).start()


# ── Operation endpoints ───────────────────────────────────────────────────────

@app.route("/api/ops/download", methods=["POST"])
def api_ops_download():
    data = request.json or {}
    market = data.get("market", "")
    start = data.get("start", "")
    end = data.get("end", "")
    if not start or not end:
        return jsonify({"error": "start and end required"}), 400

    days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    cmd = [sys.executable, str(CLI), "market-data", "--days", str(days)]
    if market:
        cmd += ["--market", market]

    job_id = str(uuid.uuid4())[:8]
    _submit(job_id, {"id": job_id, "type": "download", "label": f"download {market or 'all'} {start}→{end}"}, cmd)
    return jsonify({"job_id": job_id})


@app.route("/api/ops/populate", methods=["POST"])
def api_ops_populate():
    data = request.json or {}
    dataset  = data.get("dataset", "")
    exchange = data.get("exchange", "")
    symbol   = data.get("symbol", "")
    start    = data.get("start", "")
    end      = data.get("end", "")
    if not all([dataset, exchange, symbol, start, end]):
        return jsonify({"error": "dataset, exchange, symbol, start, end required"}), 400

    cmd = [
        sys.executable, str(CLI), "populate", dataset,
        "--exchange", exchange, "--symbol", symbol,
        "--start", start, "--end", end,
    ]
    job_id = str(uuid.uuid4())[:8]
    _submit(job_id, {"id": job_id, "type": "populate", "label": f"populate {dataset} {exchange}:{symbol} {start}→{end}"}, cmd)
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


# ── Static / SPA ──────────────────────────────────────────────────────────────

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
