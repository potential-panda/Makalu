import argparse
import time
from logging import getLogger
from pathlib import Path

from makalu import settings as sys_env
from makalu.storage.manifest import Manifest

LOG = getLogger(__name__)
MANIFEST_PATH = sys_env.MANIFEST_PATH
ROOT = Path(sys_env.get_parquet_root_dir())

MAX_SIZE_SETTING = "max_size_bytes"
# Bronze is Kitakamakura's: it is neither counted against the cap nor deleted here.
PROTECTED_LAYERS = {"bronze"}
RETENTION_DAYS = 180
SIZE_UNITS = {
    "b": 1,
    "k": 1_000,
    "kb": 1_000,
    "m": 1_000_000,
    "mb": 1_000_000,
    "g": 1_000_000_000,
    "gb": 1_000_000_000,
    "t": 1_000_000_000_000,
    "tb": 1_000_000_000_000,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def _infer_layer(path: str) -> str | None:
    try:
        return Path(path).relative_to(ROOT).parts[0]
    except ValueError:
        return None


def _tracked_age_timestamp(record: dict) -> float:
    return record["write_ts"]


def _ensure_settings_table(manifest: Manifest) -> None:
    manifest._conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    manifest._conn.commit()


def _get_setting(manifest: Manifest, key: str) -> str | None:
    row = manifest._conn.execute(
        "SELECT value FROM settings WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else None


def _set_setting(manifest: Manifest, key: str, value: str) -> None:
    manifest._conn.execute(
        """
        INSERT INTO settings (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    manifest._conn.commit()


def _parse_size(value: str) -> int:
    text = value.strip().lower()
    if not text:
        raise ValueError("empty size")

    number = []
    unit = []
    for ch in text:
        if ch.isdigit() or ch == ".":
            if unit:
                raise ValueError(f"invalid size: {value}")
            number.append(ch)
        elif ch.isspace():
            continue
        else:
            unit.append(ch)

    if not number:
        raise ValueError(f"invalid size: {value}")
    multiplier = SIZE_UNITS.get("".join(unit) or "b")
    if multiplier is None:
        raise ValueError(
            f"unknown size unit in {value!r}; use bytes, GB, GiB, TB, or TiB"
        )
    return int(float("".join(number)) * multiplier)


def _format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


def _actual_root_size() -> int:
    total = 0
    for path in ROOT.rglob("data.parquet"):
        if _infer_layer(str(path)) in PROTECTED_LAYERS:
            continue
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def execute(args):
    manifest = Manifest(MANIFEST_PATH)
    _ensure_settings_table(manifest)

    if args.set_max_size:
        try:
            max_size = _parse_size(args.set_max_size)
        except ValueError as e:
            manifest.close()
            raise SystemExit(str(e)) from e
        _set_setting(manifest, MAX_SIZE_SETTING, str(max_size))
        LOG.info(f"Set Makalu parquet disk cap to {_format_size(max_size)}.")

    max_size_value = _get_setting(manifest, MAX_SIZE_SETTING)
    if not max_size_value:
        LOG.info("No Makalu parquet disk cap is configured. Nothing to evict.")
        LOG.info("Set one with: python cli.py evict --set-max-size 500GiB")
        manifest.close()
        return

    max_size = int(max_size_value)
    current_size = _actual_root_size()
    LOG.info(
        f"Makalu parquet usage: {_format_size(current_size)} / cap {_format_size(max_size)}"
    )
    if current_size <= max_size:
        LOG.info("Usage is below cap. Nothing to evict.")
        manifest.close()
        return

    records = manifest.get_all()
    now = time.time()
    to_delete = []

    for r in records:
        layer = _infer_layer(r["path"])
        if not layer or layer in PROTECTED_LAYERS:
            continue
        cutoff = now - RETENTION_DAYS * 86_400
        if _tracked_age_timestamp(r) < cutoff:
            to_delete.append(r)

    if not to_delete:
        LOG.info(
            "Usage is above cap, but no manifest-tracked files exceed the retention "
            f"threshold of {RETENTION_DAYS} days."
        )
        manifest.close()
        return

    total_size = sum(r["file_size"] for r in to_delete)
    LOG.info(
        f"{'Action':<14} {'Layer':<10} {'Age (days)':>10} {'Keep (days)':>11}  Path"
    )
    LOG.info("-" * 80)
    for r in to_delete:
        layer = _infer_layer(r["path"])
        age_days = int((now - _tracked_age_timestamp(r)) / 86_400)
        action = "DELETE" if args.apply else "WOULD DELETE"
        LOG.info(
            f"{action:<14} {layer:<10} {age_days:>10}d {RETENTION_DAYS:>11}d  {r['path']}"
        )

    LOG.info(f"Total: {len(to_delete)} file(s), {_format_size(total_size)}")

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
        help="Delete stale manifest-tracked files when parquet usage exceeds the configured cap",
        description=(
            "Reads <root>/manifest.db and only considers files registered there.\n"
            "Files that are not in the manifest are never deleted by evict.\n\n"
            "Eviction is disk-cap driven: if Makalu's layers (everything under the datalake\n"
            "root except bronze) are above the configured cap, evict deletes manifest-tracked\n"
            "files older than 6 months (180 days) by write_ts.\n"
            "Bronze is Kitakamakura's: never counted against the cap and never deleted.\n"
            "Makalu does not use OS file-read access time for eviction.\n\n"
            "There is no per-path or per-dataset delete option in this command.\n"
            "IMPORTANT: Always run without --apply first to review what would be deleted.\n"
            "Prerequisite: run 'scan' periodically to keep the manifest up to date."
        ),
        epilog=(
            "Examples:\n"
            "  python cli.py evict --set-max-size 500GiB\n"
            "  python cli.py evict             # dry-run: show what would be deleted\n"
            "  python cli.py evict --apply     # actually delete"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--apply", action="store_true", help="Actually delete files (default: dry-run)"
    )
    p.add_argument(
        "--set-max-size",
        metavar="SIZE",
        help="Persist the Makalu parquet disk cap in manifest.db, e.g. 500GiB or 1TiB",
    )
    p.set_defaults(func=execute)
