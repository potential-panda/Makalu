#!/usr/bin/env python3
"""Makalu data warehouse HTTP API — port 9776."""

import datetime
import json
import os
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
_executor_running = False
_executor_lock = threading.Lock()


# Job file extensions encode state:
#   {id}.json             → pending
#   {id}.json.processing  → running  (contains pid)
#   {id}.done             → finished (done or failed)
#   {id}.log              → subprocess stdout (written while running)
#   {id}.rc               → exit code (written by executor before finalising)

def _job_pending_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"

def _job_processing_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json.processing"

def _job_done_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.done"

def _job_log_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.log"

def _job_rc_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.rc"


def _read_log_tail(job_id: str) -> list[str]:
    try:
        return _job_log_path(job_id).read_text().splitlines()[-200:]
    except OSError:
        return []


def _atomic_write(path: Path, content: str) -> None:
    """Write content atomically: write to .tmp then rename, so the file is never half-written."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.rename(path)


def _load_jobs() -> None:
    """Load all jobs from disk into the in-memory cache at startup."""
    seen: set[str] = set()
    for pattern in ("*.json.processing", "*.json", "*.done"):
        for f in sorted(JOBS_DIR.glob(pattern)):
            try:
                job = json.loads(f.read_text())
                jid = job["id"]
                if jid not in seen:
                    seen.add(jid)
                    _jobs[jid] = job
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


# ── Job queue + executor ──────────────────────────────────────────────────────


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _finalize_job(job_id: str, rc: int) -> None:
    """Rename .json.processing → .done and update in-memory state."""
    proc_file = _job_processing_path(job_id)
    done_file = _job_done_path(job_id)
    try:
        job = json.loads(proc_file.read_text())
    except (OSError, ValueError):
        with _jobs_lock:
            job = dict(_jobs.get(job_id, {"id": job_id}))
    job["returncode"] = rc
    job["status"] = "done" if rc == 0 else "failed"
    job["finished_at"] = time.time()
    job["log"] = _read_log_tail(job_id)
    done_file.write_text(json.dumps(job, indent=2))
    proc_file.unlink(missing_ok=True)
    with _jobs_lock:
        _jobs[job_id] = job


def _monitor_job(job_id: str, pid: int, proc: subprocess.Popen | None = None) -> None:
    """Block until the job process exits, keeping the .processing file up to date."""
    proc_file = _job_processing_path(job_id)
    rc_path = _job_rc_path(job_id)

    while _is_pid_alive(pid) and not rc_path.exists():
        log = _read_log_tail(job_id)
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["log"] = log
        try:
            data = json.loads(proc_file.read_text())
            data["log"] = log
            _atomic_write(proc_file, json.dumps(data, indent=2))
        except (OSError, ValueError):
            pass
        time.sleep(1)

    if proc is not None:
        proc.wait()
        rc = proc.returncode
        try:
            rc_path.write_text(str(rc))
        except OSError:
            pass
    else:
        try:
            rc = int(rc_path.read_text().strip())
        except (OSError, ValueError):
            rc = -1

    _finalize_job(job_id, rc)


def _cleanup_old_jobs() -> None:
    cutoff = time.time() - 86400  # 24 hours
    with _jobs_lock:
        stale = [
            jid for jid, j in list(_jobs.items())
            if j.get("status") in ("done", "failed")
            and j.get("finished_at", j.get("created_at", 0)) < cutoff
        ]
        for jid in stale:
            del _jobs[jid]
            for path in (_job_done_path(jid), _job_log_path(jid), _job_rc_path(jid)):
                path.unlink(missing_ok=True)


def _pending_created_at(f: Path) -> float:
    try:
        return json.loads(f.read_text()).get("created_at", 0.0)
    except (json.JSONDecodeError, OSError):
        return 0.0


def _executor_loop() -> None:
    global _executor_running
    _cleanup_old_jobs()
    try:
        # On startup: resume any job whose .processing file survived a restart.
        for proc_file in JOBS_DIR.glob("*.json.processing"):
            try:
                job = json.loads(proc_file.read_text())
                job_id = job["id"]
            except (json.JSONDecodeError, KeyError, OSError):
                # Corrupted file (e.g. 0-byte from interrupted write) — extract
                # the job ID from the filename and mark failed so the queue unblocks.
                job_id = proc_file.name.split(".")[0]
                _finalize_job(job_id, rc=-1)
                continue
            pid = job.get("pid")
            if pid and _is_pid_alive(pid):
                _monitor_job(job_id, pid)          # reattach: process still running
            else:
                _finalize_job(job_id, rc=-1)       # process died while we were down

        # Pick up pending jobs one at a time until the queue is empty.
        while True:
            pending = sorted(JOBS_DIR.glob("*.json"), key=_pending_created_at)
            if not pending:
                break

            job_file = pending[0]
            try:
                job = json.loads(job_file.read_text())
            except (json.JSONDecodeError, OSError):
                job_file.unlink(missing_ok=True)
                continue

            job_id = job["id"]
            proc_file = _job_processing_path(job_id)
            job_file.rename(proc_file)             # atomic: pending → processing

            job["status"] = "running"
            job["started_at"] = time.time()

            with open(_job_log_path(job_id), "w") as log_f:
                proc = subprocess.Popen(
                    job["cmd_args"],
                    cwd=str(CLI.parent.parent),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )

            job["pid"] = proc.pid
            _atomic_write(proc_file, json.dumps(job, indent=2))
            with _jobs_lock:
                _jobs[job_id] = job

            _monitor_job(job_id, proc.pid, proc=proc)

    finally:
        with _executor_lock:
            _executor_running = False
        if any(JOBS_DIR.glob("*.json")):
            _ensure_executor()


def _ensure_executor() -> None:
    global _executor_running
    with _executor_lock:
        if _executor_running:
            return
        _executor_running = True
    threading.Thread(target=_executor_loop, daemon=True).start()


def _submit(job_id: str, job_meta: dict, cmd: list[str]) -> None:
    job = {
        **job_meta,
        "cmd": " ".join(cmd),
        "cmd_args": cmd,
        "status": "pending",
        "log": [],
        "created_at": time.time(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _job_pending_path(job_id).write_text(json.dumps(job, indent=2))
    _ensure_executor()


_ensure_executor()


# ── Daily scheduler ───────────────────────────────────────────────────────────

DAILY_SYNC_HOUR = 4  # 04:00 local time — matches former cron job

SCHEDULED_MARKETS = [
    "BINANCE:BTC_USDT",
    "BINANCE:ETH_USDT",
    "BINANCE:XRP_USDT",
    "GMO:BTC_JPY",
    "GMO:ETH_JPY",
    "GMO:XRP_JPY",
]

_next_scheduled_run: float = 0.0


def _next_scheduled_time() -> float:
    now = datetime.datetime.now()
    candidate = now.replace(hour=DAILY_SYNC_HOUR, minute=0, second=0, microsecond=0)
    if candidate <= now:
        candidate += datetime.timedelta(days=1)
    return candidate.timestamp()


def _submit_daily_market_data() -> None:
    for market in SCHEDULED_MARKETS:
        job_id = str(uuid.uuid4())[:8]
        cmd = [sys.executable, str(CLI), "market-data", "--market", market, "--days", "3"]
        _submit(
            job_id,
            {"id": job_id, "type": "download", "label": f"daily sync {market}"},
            cmd,
        )


def _scheduler_loop() -> None:
    global _next_scheduled_run
    while True:
        _next_scheduled_run = _next_scheduled_time()
        sleep_secs = _next_scheduled_run - time.time()
        if sleep_secs > 0:
            time.sleep(sleep_secs)
        _submit_daily_market_data()


threading.Thread(target=_scheduler_loop, daemon=True).start()


@app.route("/api/scheduler")
def api_scheduler():
    next_ts = _next_scheduled_run
    return jsonify({
        "next_run": next_ts,
        "next_run_iso": datetime.datetime.fromtimestamp(next_ts).isoformat() if next_ts else None,
        "daily_sync_hour": DAILY_SYNC_HOUR,
        "markets": SCHEDULED_MARKETS,
    })


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

    cmd = [str(HIMALAYA_START_SH), "--env", env, "--from-dt", start, "--to-dt", end, "--retest"]
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
        jobs = sorted(_jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)
    return jsonify(jobs)


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
