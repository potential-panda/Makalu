import argparse
import time
from logging import getLogger
from pathlib import Path

from qate.env import sys_env
from qate.store.manifest import Manifest

LOG = getLogger(__name__)
MANIFEST_PATH = "/panda/makalu-data/manifest.db"
ROOT = Path(sys_env.get_parquet_root_dir())

RETENTION_DAYS: dict[str, int] = {
    "bronze": 180,
    "silver": 180,
    "gold": 180,
    "feature": 90,
    "research": 7,
}


def _infer_layer(path: str) -> str | None:
    try:
        return Path(path).relative_to(ROOT).parts[0]
    except ValueError:
        return None


def _last_activity(record: dict) -> float:
    return max(record["write_ts"], record["read_ts"] or 0.0)


def execute(args):
    manifest = Manifest(MANIFEST_PATH)
    records = manifest.get_all()
    now = time.time()
    to_delete = []

    for r in records:
        layer = _infer_layer(r["path"])
        if not layer or layer not in RETENTION_DAYS:
            continue
        cutoff = now - RETENTION_DAYS[layer] * 86_400
        if _last_activity(r) < cutoff:
            to_delete.append(r)

    if not to_delete:
        LOG.info("Nothing to evict.")
        manifest.close()
        return

    total_size = sum(r["file_size"] for r in to_delete)
    LOG.info(f"{'Action':<14} {'Layer':<10} {'Age (days)':>10}  Path")
    LOG.info("-" * 80)
    for r in to_delete:
        layer = _infer_layer(r["path"])
        age_days = int((now - _last_activity(r)) / 86_400)
        action = "DELETE" if args.apply else "WOULD DELETE"
        LOG.info(f"{action:<14} {layer:<10} {age_days:>10}d  {r['path']}")

    LOG.info(f"Total: {len(to_delete)} file(s), {total_size / 1_048_576:.1f} MB")

    if not args.apply:
        LOG.info("Dry-run. Run with --apply to delete.")
        manifest.close()
        return

    deleted = 0
    for r in to_delete:
        path = Path(r["path"])
        try:
            if path.exists():
                path.unlink()
                for parent in path.parents:
                    if parent == ROOT:
                        break
                    if not any(parent.iterdir()):
                        parent.rmdir()
                    else:
                        break
            manifest.delete(r["path"])
            deleted += 1
        except Exception as e:
            LOG.error(f"Error deleting {r['path']}: {e}")

    manifest.close()
    LOG.info(f"Done. Deleted {deleted}/{len(to_delete)} file(s).")


def register_cli(sub):
    p = sub.add_parser(
        "evict",
        help="Delete files past retention policy (DESTRUCTIVE — dry-run by default)",
        description=(
            "Reads the manifest and deletes parquet files whose last activity\n"
            "(write or read timestamp) exceeds the retention limit for their layer.\n\n"
            "Retention policy:\n"
            "  bronze / silver / gold : 180 days\n"
            "  feature                :  90 days\n"
            "  research               :   7 days\n\n"
            "IMPORTANT: Always run without --apply first to review what would be deleted.\n"
            "Prerequisite: run 'scan' first to ensure the manifest is up to date."
        ),
        epilog=(
            "Examples:\n"
            "  python cli.py evict             # dry-run: show what would be deleted\n"
            "  python cli.py evict --apply     # actually delete"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--apply", action="store_true", help="Actually delete files (default: dry-run)")
    p.set_defaults(func=execute)
