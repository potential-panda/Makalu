import argparse
import json
import os
from datetime import datetime, timedelta
from logging import getLogger
from pathlib import Path
from zoneinfo import ZoneInfo

import msgpack
import qate.env.sys_env as sys_env
from qate.store.influx_store import InfluxdbStore, convert_to_point
from qate.store.s3 import download

LOG = getLogger(__name__)

DOWNLOAD_DIR = "/panda/metrics-data"

HOST_CONFIG = {
    "orion-01": {
        "bucket": "panda-cc-orion-01",
        "envs": ["QIN_TENGFEI"],
    },
    "orion-02": {
        "bucket": "panda-cc-orion-02",
        "envs": ["QASIM_TARIQ"],
    },
}

DEFAULT_HOSTS = list(HOST_CONFIG.keys())
_TZ = ZoneInfo("Asia/Tokyo")


def _intervals_since(start_str: str) -> list[str]:
    """5-minute intervals from start_str (YYYYMMDD_HHMM) to now."""
    start = datetime.strptime(start_str, "%Y%m%d_%H%M").replace(tzinfo=_TZ)
    now = datetime.now(_TZ)
    result = []
    current = start
    while current <= now:
        current += timedelta(minutes=5)
        result.append(current.strftime("%Y%m%d_%H%M"))
    return result


def _last_date_str(env_dir: Path) -> str:
    """Inspect local files to find the most-recent interval string."""
    if not env_dir.exists():
        return ""
    files = sorted(env_dir.iterdir(), key=lambda p: p.stat().st_ctime, reverse=True)
    for f in files:
        if not f.name.endswith((".msgpack", ".msgpack__done")):
            continue
        try:
            parts = f.name.split(".")[0].split("_")
            if len(parts) >= 5:
                return f"{parts[-2]}_{parts[-1]}"
        except Exception:
            continue
    return ""


def _load_file(writer, path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for obj in msgpack.Unpacker(f, raw=False):
            writer.add(convert_to_point(obj))
            count += 1
    os.rename(str(path), str(path) + "__done")
    LOG.info(f"Loaded {path.name}: {count} points → {writer.bucket_name}")
    return count


def _update_env(host: str, env_name: str, base_dir: Path) -> dict:
    stats = {"downloaded": 0, "processed": 0, "skipped": 0}
    s3_bucket = HOST_CONFIG[host]["bucket"]
    env_dir = base_dir / s3_bucket
    env_dir.mkdir(parents=True, exist_ok=True)

    client = sys_env.get_s3_client()

    last_dt = _last_date_str(env_dir)
    lookback = (datetime.today() - timedelta(days=28)).strftime("%Y%m%d") + "_0000"
    if not last_dt or lookback > last_dt:
        last_dt = lookback

    intervals = _intervals_since(last_dt)
    LOG.info(f"{host}/{env_name}: {len(intervals)} intervals from {intervals[0]}")

    # Download all but the last interval (may be incomplete)
    for dt_str in intervals[:-1]:
        s3_name = f"{env_name}_Metrics_{dt_str}.msgpack.bz2"
        local = env_dir / s3_name[:-4]
        done = env_dir / (s3_name[:-4] + "__done")
        if local.exists() or done.exists():
            stats["skipped"] += 1
            continue
        try:
            download(client, s3_bucket, s3_name, env_dir)
            stats["downloaded"] += 1
        except Exception as e:
            LOG.warning(f"Not available {s3_bucket}/{s3_name}: {e}")
            stats["skipped"] += 1

    # Push all unprocessed local files to InfluxDB
    influx = InfluxdbStore(sys_env.get_influxdb())
    writer = influx.create_writer(env_name, 4096, no_retention=True)
    for f in sorted(env_dir.iterdir(), key=lambda p: p.stat().st_ctime, reverse=True):
        if not f.is_file() or not f.name.endswith(".msgpack"):
            continue
        if env_name not in f.name:
            continue
        try:
            stats["processed"] += _load_file(writer, f)
        except Exception as e:
            LOG.error(f"Error loading {f.name}: {e}")
    writer.close()

    return stats


def execute(args) -> None:
    base_dir = Path(DOWNLOAD_DIR)
    hosts = [args.host] if args.host else DEFAULT_HOSTS

    totals = {"downloaded": 0, "processed": 0, "skipped": 0, "errors": 0}

    for host in hosts:
        if host not in HOST_CONFIG:
            LOG.error(f"Unknown host '{host}'. Valid: {DEFAULT_HOSTS}")
            totals["errors"] += 1
            continue
        for env_name in HOST_CONFIG[host]["envs"]:
            try:
                LOG.info(f"Processing {host}/{env_name}")
                stats = _update_env(host, env_name, base_dir)
                totals["downloaded"] += stats["downloaded"]
                totals["processed"] += stats["processed"]
                totals["skipped"] += stats["skipped"]
                LOG.info(f"{host}/{env_name} done: {stats}")
            except Exception as e:
                LOG.error(f"Error on {host}/{env_name}: {e}")
                totals["errors"] += 1

    result = {"status": "done", "hosts": hosts, **totals}
    LOG.info(f"metrics-data complete: {result}")
    print(json.dumps(result))


def register_cli(sub) -> None:
    p = sub.add_parser(
        "metrics-data",
        help="Download trading metrics from S3 and push to InfluxDB",
        description=(
            "Downloads 5-minute trading-metrics files (msgpack.bz2) from S3\n"
            "for each production host and writes them to InfluxDB.\n\n"
            "Looks back up to 28 days for missing intervals.\n"
            "Already-downloaded or already-processed files are skipped.\n\n"
            "Hosts  : orion-01 → QIN_TENGFEI, orion-02 → QASIM_TARIQ\n"
            "Download dir: /panda/metrics-data/"
        ),
        epilog=(
            "Examples:\n"
            "  python cli.py metrics-data                   # all hosts\n"
            "  python cli.py metrics-data --host orion-01   # single host"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--host",
        choices=DEFAULT_HOSTS,
        help="Process a single host only (default: all hosts).",
    )
    p.set_defaults(func=execute)
