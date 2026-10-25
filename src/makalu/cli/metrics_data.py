import argparse
import json
import os
from datetime import datetime, timedelta
from logging import getLogger
from pathlib import Path
from zoneinfo import ZoneInfo

import msgpack
from k2_data.s3 import download, list_files

import makalu.settings as sys_env
from makalu.storage.influx_store import InfluxdbStore, convert_to_point

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
    "pisces-01": {
        "bucket": "panda-cc-pisces-01",
        "envs": ["ASHLEY_XEON", "ATSUSHI_EGAWA"],
    },
}

DEFAULT_HOSTS = list(HOST_CONFIG.keys())
_TZ = ZoneInfo("Asia/Tokyo")


def _current_interval() -> str:
    """Return the in-progress 5-minute interval key (YYYYMMDD_HHMM) — skip this one."""
    now = datetime.now(_TZ)
    floored = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
    return floored.strftime("%Y%m%d_%H%M")


def _load_file(writer, path: Path) -> int:
    count = 0
    with open(path, "rb") as f:
        for obj in msgpack.Unpacker(f, raw=False):
            writer.add(convert_to_point(obj))
            count += 1
    os.rename(str(path), str(path) + "__done")
    LOG.info(f"Loaded {path.name}: {count} points → {writer.bucket_name}")
    return count


def _update_env(host: str, env_name: str, base_dir: Path, client) -> dict:
    stats = {"downloaded": 0, "processed": 0, "skipped": 0}
    s3_bucket = HOST_CONFIG[host]["bucket"]
    env_dir = base_dir / s3_bucket
    env_dir.mkdir(parents=True, exist_ok=True)

    # Build set of already-local filenames (with and without __done suffix)
    local = {
        f.name.replace("__done", "")
        for f in env_dir.iterdir()
        if f.name.startswith(env_name + "_Metrics_")
    }

    # List what actually exists in S3 for this env (one paginated call)
    prefix = f"{env_name}_Metrics_"
    s3_objects = list_files(client, s3_bucket, prefix=prefix)
    current = _current_interval()

    to_download = []
    for _key, objs in s3_objects.items():
        s3_name = objs[0].name  # e.g. ENV_Metrics_YYYYMMDD_HHMM.msgpack.bz2
        local_name = s3_name[:-4] if s3_name.endswith(".bz2") else s3_name
        # skip current (may be incomplete) and already-local files
        if current in s3_name or local_name in local:
            stats["skipped"] += 1
            continue
        to_download.append(s3_name)

    LOG.info(
        f"{host}/{env_name}: {len(to_download)} to download, {stats['skipped']} already local/current"
    )

    for s3_name in sorted(to_download):
        try:
            download(client, s3_bucket, s3_name, env_dir)
            stats["downloaded"] += 1
        except Exception as e:
            LOG.warning(f"Download failed {s3_bucket}/{s3_name}: {e}")

    # Push all unprocessed local files to InfluxDB
    influx = InfluxdbStore(sys_env.get_influxdb())
    writer = influx.create_writer(env_name, 4096, no_retention=True)
    for f in sorted(env_dir.iterdir()):
        if not f.is_file() or not f.name.endswith(".msgpack"):
            continue
        if not f.name.startswith(env_name + "_Metrics_"):
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
    client = sys_env.get_s3_client()

    totals = {"downloaded": 0, "processed": 0, "skipped": 0, "errors": 0}

    for host in hosts:
        if host not in HOST_CONFIG:
            LOG.error(f"Unknown host '{host}'. Valid: {DEFAULT_HOSTS}")
            totals["errors"] += 1
            continue
        for env_name in HOST_CONFIG[host]["envs"]:
            try:
                LOG.info(f"Processing {host}/{env_name}")
                stats = _update_env(host, env_name, base_dir, client)
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
            "Hosts  : orion-01 → QIN_TENGFEI, orion-02 → QASIM_TARIQ, pisces-01 → ASHLEY_XEON ATSUSHI_EGAWA\n"
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
