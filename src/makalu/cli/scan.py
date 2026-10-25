import argparse
from logging import getLogger
from pathlib import Path

from makalu import settings as sys_env
from makalu.storage.manifest import Manifest

LOG = getLogger(__name__)
MANIFEST_PATH = sys_env.MANIFEST_PATH
ROOT = Path(sys_env.get_parquet_root_dir())


def execute(args):
    import pyarrow.parquet as pq

    manifest = Manifest(MANIFEST_PATH)
    manifest._conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    known = {r["path"] for r in manifest.get_all()}
    # Bronze belongs to Kitakamakura and is never evicted, so it is never tracked.
    files = sorted(
        f
        for f in ROOT.rglob("data.parquet")
        if f.relative_to(ROOT).parts[0] != "bronze"
    )
    new_count = 0

    for f in files:
        path_str = str(f)
        if path_str in known:
            continue
        try:
            meta = pq.read_metadata(f)
            row_count = meta.num_rows
            file_size = f.stat().st_size
            write_ts = f.stat().st_mtime
            manifest._conn.execute(
                """
                INSERT OR IGNORE INTO files (path, row_count, file_size, write_ts)
                VALUES (?, ?, ?, ?)
                """,
                (path_str, row_count, file_size, write_ts),
            )
            new_count += 1
        except Exception as e:
            LOG.warning(f"Could not read {f}: {e}")

    manifest._conn.commit()
    manifest.close()
    LOG.info(
        f"Scan complete. Registered {new_count} new file(s) ({len(known)} already known)."
    )


def register_cli(sub):
    p = sub.add_parser(
        "scan",
        help="Register all parquet files into the manifest database (WRITE, safe to re-run)",
        description=(
            "Walks the datalake root (Komachi's ROOT_PATH) and registers any silver/gold/...\n"
            "parquet files not yet tracked in <root>/manifest.db. Bronze is skipped: it is\n"
            "Kitakamakura's and is never evicted.\n\n"
            "Run this after 'populate' to keep the manifest in sync.\n"
            "Already-known files are skipped — safe to run repeatedly.\n\n"
            "Newly-registered files use the file's last-modified time as write_ts.\n"
            "Makalu does not use OS file-read access time for eviction decisions.\n\n"
            "The manifest is used by 'evict' to determine file age and apply retention policy."
        ),
        epilog="Example:\n  python cli.py scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=execute)
