import argparse
from logging import getLogger
from pathlib import Path

from makalu import settings as sys_env
from makalu.storage.manifest import Manifest

LOG = getLogger(__name__)
MANIFEST_PATH = sys_env.MANIFEST_PATH
ROOT = Path(sys_env.get_parquet_root_dir())


def scan(root: Path, manifest: Manifest) -> tuple[int, int]:
    """Register new parquet files under `root` and drop rows for files gone from it.

    Returns (registered, pruned). A row whose file was deleted -- by hand, or by
    a dataset being replaced -- would otherwise count toward evict's size cap
    forever. Pruning is refused when the walk finds nothing while the manifest
    knows files: that is an unmounted or wrong root, not a datalake emptied.
    """
    import pyarrow.parquet as pq

    if not root.is_dir():
        raise FileNotFoundError(f"Datalake root {root} does not exist")
    known = {r["path"] for r in manifest.get_all()}
    # Bronze belongs to Kitakamakura and is never evicted, so it is never tracked.
    files = sorted(
        f
        for f in root.rglob("data.parquet")
        if f.relative_to(root).parts[0] != "bronze"
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

    walked = {str(f) for f in files}
    gone = [p for p in known - walked if not Path(p).exists()]
    if gone and not walked:
        LOG.warning(
            f"Found no files under {root} but the manifest knows {len(known)}: "
            "not pruning. Is the datalake mounted?"
        )
        gone = []
    manifest._conn.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in gone])
    manifest._conn.commit()
    return new_count, len(gone)


def execute(args):
    manifest = Manifest(MANIFEST_PATH)
    manifest._conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    try:
        new_count, pruned = scan(ROOT, manifest)
    finally:
        manifest.close()
    LOG.info(
        f"Scan complete. Registered {new_count} new file(s), "
        f"pruned {pruned} whose file no longer exists."
    )


def register_cli(sub):
    p = sub.add_parser(
        "scan",
        help="Register new parquet files in the manifest and prune rows for deleted ones (WRITE, safe to re-run)",
        description=(
            "Walks the datalake root (Komachi's ROOT_PATH) and registers any silver/gold/...\n"
            "parquet files not yet tracked in <root>/manifest.db. Bronze is skipped: it is\n"
            "Kitakamakura's and is never evicted.\n\n"
            "Run this after 'populate' to keep the manifest in sync.\n"
            "Already-known files are skipped — safe to run repeatedly.\n"
            "Rows whose file no longer exists are removed, so a deleted file stops counting\n"
            "toward evict's size cap (refused if the walk finds no files at all).\n\n"
            "Newly-registered files use the file's last-modified time as write_ts.\n"
            "Makalu does not use OS file-read access time for eviction decisions.\n\n"
            "The manifest is used by 'evict' to determine file age and apply retention policy."
        ),
        epilog="Example:\n  python cli.py scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(func=execute)
