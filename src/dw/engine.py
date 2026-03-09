"""
DuckDB query engine for the data warehouse.

Discovers datasets from doc/catalog.yaml and registers them as Hive-partitioned
views. Only datasets that have data on disk are registered.

    engine = Engine()

    # What's available
    engine.list()

    # Pull data — no SQL needed for simple cases
    df = engine.read("VolSpread", exchange="BINANCE", symbol="XRP_USDT",
                     date_from="20250701", date_to="20250731")

    # Full SQL for anything else
    df = engine.query(\"\"\"
        SELECT date_trunc('minute', to_timestamp(ts)) AS minute,
               avg(spread_pct_ema)
        FROM volspread
        WHERE exchange = 'BINANCE' AND symbol = 'XRP_USDT'
        GROUP BY minute ORDER BY minute
    \"\"\")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).parent.parent.parent
_CATALOG_PATH = _REPO_ROOT / "doc" / "catalog.yaml"
_DEFAULT_ROOT = "/panda/makalu-data"


def _view(dataset_name: str) -> str:
    """SQL-safe view name for a dataset."""
    return dataset_name.lower()


class Engine:
    """
    Catalog-driven DuckDB engine for the data warehouse.

    On construction it scans the catalog and registers one DuckDB view per
    dataset whose data directory exists on disk. All views use Hive
    partitioning so DuckDB prunes by exchange, symbol, date, etc.
    automatically when those columns appear in WHERE clauses.
    """

    def __init__(
        self,
        root: str = _DEFAULT_ROOT,
        catalog: str = str(_CATALOG_PATH),
    ):
        self._root = Path(root)
        self._catalog: list[dict[str, Any]] = self._load_catalog(catalog)
        self._conn = duckdb.connect(database=":memory:")
        # dataset_name -> catalog entry, for all registered datasets
        self._datasets: dict[str, dict[str, Any]] = {}
        self._register_all()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_catalog(self, path: str) -> list[dict[str, Any]]:
        with open(path) as f:
            return yaml.safe_load(f).get("datasets", [])

    def _register_all(self) -> None:
        for entry in self._catalog:
            name = entry["dataset_name"]
            layer = entry["layer"]
            data_dir = self._root / layer / f"dataset={name}"
            if not data_dir.is_dir():
                continue
            glob = str(data_dir / "**" / "*.parquet")
            view = _view(name)
            self._conn.execute(f"""
                CREATE OR REPLACE VIEW {view} AS
                SELECT * FROM read_parquet('{glob}', hive_partitioning = true)
            """)
            self._datasets[name] = entry

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list(self) -> pd.DataFrame:
        """Return a summary DataFrame of all registered datasets."""
        rows = [
            {
                "dataset": e["dataset_name"],
                "layer": e["layer"],
                "grain": e["grain"],
                "partitions": ", ".join(e.get("partition_columns", [])),
                "source": (
                    ", ".join(e["source_dataset"])
                    if isinstance(e.get("source_dataset"), list)
                    else e.get("source_dataset") or ""
                ),
            }
            for e in self._datasets.values()
        ]
        return pd.DataFrame(rows).sort_values(["layer", "dataset"]).reset_index(drop=True)

    def describe(self, dataset: str) -> dict[str, Any]:
        """Return catalog metadata for a dataset."""
        entry = self._datasets.get(dataset)
        if entry is None:
            raise KeyError(f"Dataset '{dataset}' is not registered. Call list() to see what's available.")
        return entry

    def schema(self, dataset: str) -> pd.DataFrame:
        """Return column names and DuckDB types for a dataset view."""
        return self._conn.execute(f"DESCRIBE {_view(dataset)}").fetchdf()[["column_name", "column_type"]]

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL and return a DataFrame."""
        return self._conn.execute(sql).fetchdf()

    def read(
        self,
        dataset: str,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int | None = None,
        **partition_filters: str,
    ) -> pd.DataFrame:
        """
        Load a dataset with optional partition filters — no SQL needed.

        Partition column values are passed as keyword arguments matching
        the catalog's partition_columns (e.g. exchange, symbol, param_id).
        Dates are YYYYMMDD strings.

        Examples
        --------
        engine.read("Trade", exchange="BINANCE", symbol="XRP_USDT",
                    date_from="20250701", date_to="20250731")

        engine.read("VolSpread", exchange="BINANCE", symbol="XRP_USDT",
                    execution_size="150", param_id="vs_60s_120s",
                    date_from="20250801", date_to="20250831")
        """
        entry = self.describe(dataset)
        valid_partitions = set(entry.get("partition_columns", []))

        # Validate caller didn't pass unknown filter keys
        unknown = set(partition_filters) - valid_partitions
        if unknown:
            raise ValueError(
                f"Unknown partition filter(s) for '{dataset}': {unknown}. "
                f"Valid columns: {valid_partitions}"
            )

        clauses: list[str] = []
        for col, val in partition_filters.items():
            clauses.append(f"{col} = '{val}'")
        if date_from:
            clauses.append(f"date >= '{date_from}'")
        if date_to:
            clauses.append(f"date <= '{date_to}'")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = f"LIMIT {limit}" if limit else ""

        sql = f"SELECT * FROM {_view(dataset)} {where} {limit_clause}".strip()
        return self._conn.execute(sql).fetchdf()

    def sample(self, dataset: str, n: int = 5) -> pd.DataFrame:
        """Return n rows from a dataset."""
        return self._conn.execute(f"SELECT * FROM {_view(dataset)} LIMIT {n}").fetchdf()
