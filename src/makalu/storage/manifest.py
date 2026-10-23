import sqlite3
import time
from logging import getLogger

LOG = getLogger(__name__)


class Manifest:
    """
    Lightweight SQLite ledger that records every parquet file write and read.
    Layer-agnostic: only knows path, size, row count, and timestamps.
    Eviction policy lives in Makalu (manifest_cli.py), which infers layer from path.
    """

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path       TEXT PRIMARY KEY,
                row_count  INTEGER NOT NULL,
                file_size  INTEGER NOT NULL,
                write_ts   REAL NOT NULL,
                read_ts    REAL
            )
        """
        )
        self._conn.commit()

    def record_write(self, path: str, row_count: int, file_size: int) -> None:
        self._conn.execute(
            """
            INSERT INTO files (path, row_count, file_size, write_ts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                row_count = excluded.row_count,
                file_size = excluded.file_size,
                write_ts  = excluded.write_ts
            """,
            (path, row_count, file_size, time.time()),
        )
        self._conn.commit()

    def record_read(self, path: str) -> None:
        self._conn.execute(
            "UPDATE files SET read_ts = ? WHERE path = ?",
            (time.time(), path),
        )
        self._conn.commit()

    def get_all(self) -> list[dict]:
        cursor = self._conn.execute(
            "SELECT path, row_count, file_size, write_ts, read_ts FROM files ORDER BY write_ts"
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def delete(self, path: str) -> None:
        self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
