#!/usr/bin/env python3
"""Makalu data warehouse HTTP API — port 9776."""

import datetime
import shlex
import sys
import threading
import time
import uuid
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from makalu.data_access import availability, dataset_availability, dataset_overview, global_stats, list_markets
from makalu.data_outage import install_bronze_outage_patch
from makalu.job_executor import JobExecutor

install_bronze_outage_patch()

WEB_DIR = Path(__file__).parent.parent / "web"
CLI = Path(__file__).parent / "cli.py"
HIMALAYA_START_SH = Path("/home/lulurun/workspace/Himalaya/start.sh")
BACKTEST_ENV_ROOT = Path.home() / "env" / "backtest"

app = Flask(__name__, static_folder=str(WEB_DIR))

executor = JobExecutor()

CACHE_TTL = 300

# Stats cache
_stats_cache: dict = {}
_stats_cache_ts: float = 0
_stats_refreshing = False

# Dataset inventory cache
_datasets_cache: list[dict] = []
_datasets_cache_ts: float = 0
_datasets_refreshing = False

# Availability cache: key = "EXCHANGE_SYMBOL"
_avail_cache: dict[str, dict] = {}
_avail_cache_ts: dict[str, float] = {}
_avail_refreshing: set[str] = set()


@app.after_request
def _no_cache_web_assets(resp):
    if not request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        resp.headers["Pragma"] = "no-cache"
    elif request.path in {"/api/availability", "/api/markets", "/api/datasets", "/api/dataset-availability", "/api/stats"}:
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        resp.headers["Pragma"] = "no-cache"
    return resp


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


def _refresh_datasets() -> None:
    global _datasets_cache, _datasets_cache_ts, _datasets_refreshing
    _datasets_refreshing = True
    try:
        _datasets_cache = dataset_overview()
        _datasets_cache_ts = time.time()
    except Exception:
        pass
    finally:
        _datasets_refreshing = False


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
threading.Thread(target=_refresh_datasets, daemon=True).start()


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
    stale = time.time() - _avail_cache_ts.get(key, 0) > CACHE_TTL
    if stale and key not in _avail_refreshing:
        threading.Thread(target=_refresh_avail, args=(exchange, symbol), daemon=True).start()
    if key in _avail_cache:
        payload = dict(_avail_cache[key])
        if stale:
            payload["loading"] = True
            payload["stale"] = True
        return jsonify(payload)
    return jsonify({"loading": True})


@app.route("/api/datasets")
def api_datasets():
    if time.time() - _datasets_cache_ts > CACHE_TTL and not _datasets_refreshing:
        threading.Thread(target=_refresh_datasets, daemon=True).start()
    if _datasets_cache:
        return jsonify(_datasets_cache)
    return jsonify({"loading": True})


@app.route("/api/dataset-availability")
def api_dataset_availability():
    dataset = request.args.get("dataset", "")
    start = request.args.get("start", "") or None
    end = request.args.get("end", "") or None
    partitions = {
        k: request.args.get(k) or None
        for k in ("exchange", "symbol", "execution_size", "param_id", "market_a", "market_b")
        if request.args.get(k) is not None
    }
    if not dataset:
        return jsonify({"error": "dataset required"}), 400
    return jsonify(dataset_availability(dataset, partitions=partitions, start=start, end=end))


@app.route("/api/stats")
def api_stats():
    if time.time() - _stats_cache_ts > CACHE_TTL and not _stats_refreshing:
        threading.Thread(target=_refresh_stats, daemon=True).start()
    if _stats_cache:
        return jsonify(_stats_cache)
    return jsonify({"loading": True})


def _submit(job_id: str, job_meta: dict, cmd: list[str]) -> None:
    job = {
        **job_meta,
        "cmd": " ".join(cmd),
        "cmd_args": cmd,
        "status": "pending",
        "log": [],
        "created_at": time.time(),
    }
    executor.add(job)


executor.kickoff()


# ── Daily scheduler ───────────────────────────────────────────────────────────

DAILY_SYNC_HOUR = 4  # 04:00 local time
WEEKLY_MAINTENANCE_WEEKDAY = 6  # Sunday, Python weekday numbering: Monday=0
WEEKLY_MAINTENANCE_HOUR = 5

SCHEDULED_MARKETS = [
    "BINANCE:BTC_USDT",
    "BINANCE:ETH_USDT",
    "BINANCE:XRP_USDT",
    "GMO:BTC_JPY",
    "GMO:ETH_JPY",
    "GMO:XRP_JPY",
    "COINCHECK:BTC_SPOT",
    "COINCHECK:ETH_SPOT",
    "COINCHECK:XRP_SPOT",
]

DAILY_SYNC_DAYS = 3  # market-data --days N: downloads today-N .. today-1

# Bronze populate window, in days back from today. The upper bound stays at
# today-2 because the collectors only deliver yesterday's data early morning,
# and the lower bound matches what --days 3 just downloaded. Each date therefore
# gets two populate attempts (first as today-2, again as today-3); populate is
# idempotent, so the second pass skips any data.parquet already written.
BRONZE_POPULATE_START_DAYS_AGO = 3
BRONZE_POPULATE_END_DAYS_AGO = 2

_next_scheduled_run: float = 0.0
_next_weekly_maintenance_run: float = 0.0


def _next_scheduled_time() -> float:
    now = datetime.datetime.now()
    candidate = now.replace(hour=DAILY_SYNC_HOUR, minute=0, second=0, microsecond=0)
    if candidate <= now:
        candidate += datetime.timedelta(days=1)
    return candidate.timestamp()


def _next_weekly_maintenance_time() -> float:
    now = datetime.datetime.now()
    days_ahead = (WEEKLY_MAINTENANCE_WEEKDAY - now.weekday()) % 7
    candidate = (now + datetime.timedelta(days=days_ahead)).replace(
        hour=WEEKLY_MAINTENANCE_HOUR, minute=0, second=0, microsecond=0
    )
    if candidate <= now:
        candidate += datetime.timedelta(days=7)
    return candidate.timestamp()


def _bronze_populate_window() -> tuple[str, str]:
    today = date.today()
    start = today - datetime.timedelta(days=BRONZE_POPULATE_START_DAYS_AGO)
    end = today - datetime.timedelta(days=BRONZE_POPULATE_END_DAYS_AGO)
    return start.isoformat(), end.isoformat()


def _submit_daily_market_data() -> None:
    """Enqueue the daily download jobs, then the bronze populate jobs.

    The executor runs one job at a time in creation order, so every download
    for the day completes before the first populate reads /panda/market-data.
    """
    for market in SCHEDULED_MARKETS:
        job_id = str(uuid.uuid4())[:8]
        cmd = [
            sys.executable, str(CLI), "market-data",
            "--market", market, "--days", str(DAILY_SYNC_DAYS),
        ]
        _submit(
            job_id,
            {"id": job_id, "type": "download", "label": f"daily sync {market}"},
            cmd,
        )

    start, end = _bronze_populate_window()
    for market in SCHEDULED_MARKETS:
        exchange, symbol = market.split(":")
        job_id = str(uuid.uuid4())[:8]
        cmd = [
            sys.executable, str(CLI), "populate", "bronze",
            "--exchange", exchange, "--symbol", symbol,
            "--start", start, "--end", end,
        ]
        _submit(
            job_id,
            {
                "id": job_id,
                "type": "populate",
                "label": f"daily bronze {market} {start}→{end}",
            },
            cmd,
        )


def _submit_weekly_maintenance() -> None:
    job_id = str(uuid.uuid4())[:8]
    scan_cmd = [sys.executable, str(CLI), "scan"]
    evict_cmd = [sys.executable, str(CLI), "evict", "--apply"]
    cmd = ["bash", "-lc", f"{shlex.join(scan_cmd)} && {shlex.join(evict_cmd)}"]
    _submit(
        job_id,
        {"id": job_id, "type": "maintenance", "label": "weekly scan + evict"},
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


def _weekly_maintenance_loop() -> None:
    global _next_weekly_maintenance_run
    while True:
        _next_weekly_maintenance_run = _next_weekly_maintenance_time()
        sleep_secs = _next_weekly_maintenance_run - time.time()
        if sleep_secs > 0:
            time.sleep(sleep_secs)
        _submit_weekly_maintenance()


threading.Thread(target=_scheduler_loop, daemon=True).start()
threading.Thread(target=_weekly_maintenance_loop, daemon=True).start()


@app.route("/api/scheduler")
def api_scheduler():
    next_ts = _next_scheduled_run
    next_maintenance_ts = _next_weekly_maintenance_run
    return jsonify({
        "next_run": next_ts,
        "next_run_iso": datetime.datetime.fromtimestamp(next_ts).isoformat() if next_ts else None,
        "daily_sync_hour": DAILY_SYNC_HOUR,
        "next_weekly_maintenance": next_maintenance_ts,
        "next_weekly_maintenance_iso": (
            datetime.datetime.fromtimestamp(next_maintenance_ts).isoformat() if next_maintenance_ts else None
        ),
        "weekly_maintenance_weekday": WEEKLY_MAINTENANCE_WEEKDAY,
        "weekly_maintenance_hour": WEEKLY_MAINTENANCE_HOUR,
        # Backward-compatible names for any existing dashboard clients.
        "next_weekly_scan": next_maintenance_ts,
        "next_weekly_scan_iso": (
            datetime.datetime.fromtimestamp(next_maintenance_ts).isoformat() if next_maintenance_ts else None
        ),
        "weekly_scan_weekday": WEEKLY_MAINTENANCE_WEEKDAY,
        "weekly_scan_hour": WEEKLY_MAINTENANCE_HOUR,
        "markets": SCHEDULED_MARKETS,
        "daily_sync_days": DAILY_SYNC_DAYS,
        "daily_bronze_populate": True,
        "daily_bronze_window": dict(zip(("start", "end"), _bronze_populate_window())),
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

    cmd = [sys.executable, str(CLI), "market-data", "--start", start, "--end", end]
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
    return jsonify(executor.list_jobs())


@app.route("/api/jobs/<job_id>")
def api_job(job_id: str):
    job = executor.get_job(job_id)
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
