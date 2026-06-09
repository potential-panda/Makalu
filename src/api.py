#!/usr/bin/env python3
"""Makalu data warehouse HTTP API — port 9776."""

import json
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
HIMALAYA_START_SH = Path("/home/lulurun/workspace/Himalaya/start.sh")
BACKTEST_ENV_ROOT = Path.home() / "env" / "backtest"

app = Flask(__name__, static_folder=str(WEB_DIR))

JOBS_DIR = Path("/panda-infra/makalu/jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _save_job(job: dict) -> None:
    try:
        (JOBS_DIR / f"{job['id']}.json").write_text(json.dumps(job, indent=2))
    except OSError:
        pass


def _load_jobs() -> None:
    for f in sorted(JOBS_DIR.glob("*.json")):
        try:
            job = json.loads(f.read_text())
            _jobs[job["id"]] = job
        except (json.JSONDecodeError, KeyError, OSError):
            pass


_load_jobs()

CACHE_TTL = 300  # 5 min

# Stats cache
_stats_cache: dict = {}
_stats_cache_ts: float = 0
_stats_refreshing = False

# Availability cache: key = "EXCHANGE_SYMBOL"
_avail_cache: dict[str, dict] = {}
_avail_cache_ts: dict[str, float] = {}
_avail_refreshing: set[str] = set()


def _refresh_stats() -> None:
    global _stats_cache, _stats_cache_ts, _stats_refreshing
    _stats_refreshing = True
    try:
        _stats_cache = global_stats()
        _stats_cache_ts = time.time()
    except Exception:
        pass
    finally:
        _stats_refreshing = False


def _refresh_avail(exchange: str, symbol: str) -> None:
    key = f"{exchange}_{symbol}"
    _avail_refreshing.add(key)
    try:
        _avail_cache[key] = availability(exchange, symbol)
        _avail_cache_ts[key] = time.time()
    except Exception:
        pass
    finally:
        _avail_refreshing.discard(key)


# Pre-warm stats cache in background on startup
threading.Thread(target=_refresh_stats, daemon=True).start()


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
    key = f"{exchange}_{symbol}"
    if time.time() - _avail_cache_ts.get(key, 0) > CACHE_TTL and key not in _avail_refreshing:
        threading.Thread(target=_refresh_avail, args=(exchange, symbol), daemon=True).start()
    if key in _avail_cache:
        return jsonify(_avail_cache[key])
    return jsonify({"loading": True})


@app.route("/api/stats")
def api_stats():
    if time.time() - _stats_cache_ts > CACHE_TTL and not _stats_refreshing:
        threading.Thread(target=_refresh_stats, daemon=True).start()
    if _stats_cache:
        return jsonify(_stats_cache)
    return jsonify({"loading": True})


# ── Job runner ────────────────────────────────────────────────────────────────


def _run_job(job_id: str, cmd: list[str]) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = time.time()

    proc = subprocess.Popen(
        cmd,
        cwd=str(CLI.parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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
        _save_job(_jobs[job_id])


def _submit(job_id: str, job_meta: dict, cmd: list[str]) -> None:
    with _jobs_lock:
        job = {**job_meta, "cmd": " ".join(cmd), "status": "pending", "log": [], "created_at": time.time()}
        _jobs[job_id] = job
        _save_job(job)
    threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True).start()


# ── Backtest endpoints ────────────────────────────────────────────────────────


@app.route("/api/backtest/envs")
def api_backtest_envs():
    envs = []
    if BACKTEST_ENV_ROOT.exists():
        for d in sorted(BACKTEST_ENV_ROOT.iterdir()):
            if d.is_dir() and (d / "trading.json").exists():
                envs.append(d.name)
    return jsonify(envs)


@app.route("/api/ops/backtest", methods=["POST"])
def api_ops_backtest():
    data = request.json or {}
    env = data.get("env", "")
    start = data.get("start", "")
    end = data.get("end", "")
    optimize = data.get("optimize", True)
    if not all([env, start, end]):
        return jsonify({"error": "env, start, end required"}), 400
    if not HIMALAYA_START_SH.exists():
        return jsonify({"error": f"start.sh not found: {HIMALAYA_START_SH}"}), 500

    cmd = [str(HIMALAYA_START_SH), "--env", env, "--from-dt", start, "--to-dt", end]
    if optimize:
        cmd.append("--optimize")

    job_id = str(uuid.uuid4())[:8]
    _submit(job_id, {"id": job_id, "type": "backtest", "label": f"backtest {env} {start}→{end}"}, cmd)
    return jsonify({"job_id": job_id})


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
    dataset = data.get("dataset", "")
    exchange = data.get("exchange", "")
    symbol = data.get("symbol", "")
    start = data.get("start", "")
    end = data.get("end", "")
    if not all([dataset, exchange, symbol, start, end]):
        return jsonify({"error": "dataset, exchange, symbol, start, end required"}), 400

    cmd = [
        sys.executable,
        str(CLI),
        "populate",
        dataset,
        "--exchange",
        exchange,
        "--symbol",
        symbol,
        "--start",
        start,
        "--end",
        end,
    ]
    job_id = str(uuid.uuid4())[:8]
    _submit(
        job_id,
        {"id": job_id, "type": "populate", "label": f"populate {dataset} {exchange}:{symbol} {start}→{end}"},
        cmd,
    )
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
    app.run(host="0.0.0.0", port=9776, debug=False, threaded=True)
