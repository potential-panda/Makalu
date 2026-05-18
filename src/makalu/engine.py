"""
DuckDB query engine for the data warehouse.

Discovers datasets from Makalu's dataset catalog and registers them as
Hive-partitioned views. Only datasets that have data on disk are registered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

_MAKALU_ROOT = Path(__file__).parents[2]
_CATALOG_PATH = _MAKALU_ROOT / "doc" / "catalog.yaml"
_DEFINITIONS_PATH = _MAKALU_ROOT / "doc" / "dataset_definitions.md"
_DEFAULT_ROOT = "/panda/makalu-data"


def _view(dataset_name: str) -> str:
    return dataset_name.lower()


class Engine:
    def __init__(
        self,
        root: str = _DEFAULT_ROOT,
        catalog: str = str(_CATALOG_PATH),
        definitions: str = str(_DEFINITIONS_PATH),
    ):
        self._root = Path(root)
        self.definitions_path = Path(definitions)
        self._catalog: list[dict[str, Any]] = self._load_catalog(catalog)
        self._conn = duckdb.connect(database=":memory:")
        self._datasets: dict[str, dict[str, Any]] = {}
        self._register_all()

    def _load_catalog(self, path: str) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f).get("datasets", [])

    def _register_all(self) -> None:
        for entry in self._catalog:
            name = entry["dataset_name"]
            layer = entry["layer"]
            data_dir = self._root / layer / f"dataset={name}"
            if not data_dir.is_dir():
                continue

            view = _view(name)
            try:
                self._register_view(view, data_dir, entry)
                self._datasets[name] = entry
            except Exception as e:
                print(f"[Engine] Warning: could not register '{name}': {e}")

    def _register_view(self, view: str, data_dir: Path, entry: dict[str, Any]) -> None:
        glob = str(data_dir / "**" / "*.parquet")
        partition_cols = entry.get("partition_columns", [])

        try:
            self._conn.execute(
                f"""
                CREATE OR REPLACE VIEW {view} AS
                SELECT * FROM read_parquet('{glob}', hive_partitioning = true, union_by_name = true)
                """
            )
        except Exception:
            self._register_view_union(view, data_dir, partition_cols)

    def _register_view_union(self, view: str, data_dir: Path, partition_cols: list[str]) -> None:
        if not partition_cols or partition_cols[-1] == "date":
            raise RuntimeError(f"Cannot build union view for {view}: unexpected partition layout")

        optional_key = partition_cols[-1]
        fixed_keys = partition_cols[:-1]
        with_key_parts = [f"{k}=*" for k in fixed_keys] + [f"{optional_key}=*", "date=*", "*.parquet"]
        glob_with = str(data_dir.joinpath(*with_key_parts))
        without_key_parts = [f"{k}=*" for k in fixed_keys] + ["date=*", "*.parquet"]
        glob_without = str(data_dir.joinpath(*without_key_parts))

        self._conn.execute(
            f"""
            CREATE OR REPLACE VIEW {view} AS
            SELECT * FROM read_parquet('{glob_with}', hive_partitioning = true, union_by_name = true)
            UNION ALL BY NAME
            SELECT *, NULL::VARCHAR AS {optional_key}
            FROM read_parquet('{glob_without}', hive_partitioning = true, union_by_name = true)
            """
        )

    def list(self) -> pd.DataFrame:
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
        entry = self._datasets.get(dataset)
        if entry is None:
            raise KeyError(f"Dataset '{dataset}' is not registered. Call list() to see what's available.")
        return entry

    def schema(self, dataset: str) -> pd.DataFrame:
        return self._conn.execute(f"DESCRIBE {_view(dataset)}").fetchdf()[["column_name", "column_type"]]

    def query(self, sql: str) -> pd.DataFrame:
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
        entry = self.describe(dataset)
        valid_partitions = set(entry.get("partition_columns", []))
        unknown = set(partition_filters) - valid_partitions
        if unknown:
            raise ValueError(
                f"Unknown partition filter(s) for '{dataset}': {unknown}. Valid columns: {valid_partitions}"
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
        return self._conn.execute(f"SELECT * FROM {_view(dataset)} LIMIT {n}").fetchdf()
