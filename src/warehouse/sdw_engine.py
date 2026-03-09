"""
DuckDB query engine for the data warehouse.

Loads the dataset catalog from doc/catalog.yaml and registers a view
for every dataset that has data on disk. Views use Hive partitioning so
DuckDB prunes by exchange, symbol, date, etc. automatically.

Usage:
    engine = SDWEngine()
    engine.list_views()           # what's available
    engine.describe("VolSpread")  # catalog metadata
    df = engine.query(sql)        # run arbitrary SQL
"""

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

# Catalog is in doc/ relative to the repo root (two levels up from this file)
_REPO_ROOT = Path(__file__).parent.parent.parent
_CATALOG_PATH = _REPO_ROOT / "doc" / "catalog.yaml"
_DEFAULT_ROOT = "/panda/makalu-data"


def _view_name(dataset_name: str) -> str:
    """Convert dataset name to a valid SQL view name (lowercase, underscores)."""
    return dataset_name.lower()


class SDWEngine:
    def __init__(self, parquet_root: str = _DEFAULT_ROOT, catalog_path: str = str(_CATALOG_PATH)):
        self.parquet_root = Path(parquet_root)
        self.catalog: list[dict[str, Any]] = self._load_catalog(catalog_path)
        self.conn = duckdb.connect(database=":memory:")
        self._registered: dict[str, str] = {}  # view_name -> dataset_name
        self._register_views()

    def _load_catalog(self, catalog_path: str) -> list[dict[str, Any]]:
        with open(catalog_path, "r") as f:
            data = yaml.safe_load(f)
        return data.get("datasets", [])

    def _register_views(self) -> None:
        for entry in self.catalog:
            dataset = entry["dataset_name"]
            layer = entry["layer"]
            glob_path = self.parquet_root / layer / f"dataset={dataset}" / "**" / "*.parquet"
            glob_str = str(glob_path)

            # Skip datasets with no data on disk
            if not any(self.parquet_root.joinpath(layer, f"dataset={dataset}").glob("**/*.parquet")):
                continue

            view = _view_name(dataset)
            try:
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW {view} AS
                    SELECT * FROM read_parquet('{glob_str}', hive_partitioning = true)
                """)
                self._registered[view] = dataset
            except Exception as e:
                print(f"[SDWEngine] Failed to register view '{view}': {e}")

    def list_views(self) -> list[str]:
        """Return names of all registered dataset views."""
        return sorted(self._registered.keys())

    def describe(self, dataset_name: str) -> dict[str, Any] | None:
        """Return catalog metadata for a dataset (by name or view name)."""
        name_lower = dataset_name.lower()
        for entry in self.catalog:
            if entry["dataset_name"].lower() == name_lower:
                return entry
        return None

    def schema(self, dataset_name: str) -> pd.DataFrame:
        """Return column names and types for a registered view."""
        view = _view_name(dataset_name)
        return self.conn.execute(f"DESCRIBE {view}").fetchdf()

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return a DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def sample(self, dataset_name: str, n: int = 5) -> pd.DataFrame:
        """Return n rows from a dataset view."""
        view = _view_name(dataset_name)
        return self.conn.execute(f"SELECT * FROM {view} LIMIT {n}").fetchdf()


if __name__ == "__main__":
    engine = SDWEngine()
    views = engine.list_views()
    print(f"Registered views ({len(views)}): {views}\n")
    for v in views:
        meta = engine.describe(v)
        print(f"  {meta['dataset_name']:30s}  layer={meta['layer']:8s}  grain={meta['grain']}")
