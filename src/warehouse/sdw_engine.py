import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List

class SDWEngine:
    def __init__(self, parquet_root: str = "/panda/makalu-data"):
        self.parquet_root = Path(parquet_root)
        self.conn = duckdb.connect(database=':memory:')
        self._setup_views()

    def _setup_views(self):
        """Register virtual views for different microstructure categories."""
        categories = [
            "TradeAggression", "BookState", "BookChange", 
            "LeadLag", "VolSpread", "MarketPrice"
        ]
        
        for cat in categories:
            path = self.parquet_root / cat / "*" / "*" / "*" / "*.parquet"
            # DuckDB supports hive partitioning, so exchange/symbol/date are automatically inferred
            try:
                # We use a view to make it accessible via SQL
                # The path globbing might be slow if there are millions of files, 
                # but DuckDB is generally efficient at metadata scanning.
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW {cat.lower()} AS 
                    SELECT * FROM read_parquet('{path}', hive_partitioning = 1)
                """)
                print(f"Registered view: {cat.lower()}")
            except Exception as e:
                print(f"Failed to register {cat}: {e}")

    def query_to_df(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return result as a Pandas DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def get_schema(self, table_name: str) -> str:
        """Get the schema of a specific table/view."""
        df = self.conn.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
        return df[['name', 'type']].to_string(index=False)

    def list_tables(self) -> List[str]:
        """List all available views/tables."""
        df = self.conn.execute("SHOW TABLES").fetchdf()
        return df['name'].tolist()

if __name__ == "__main__":
    engine = SDWEngine()
    tables = engine.list_tables()
    print(f"Available tables: {tables}")
    
    if "tradeaggression" in tables:
        print("\nSchema for tradeaggression:")
        print(engine.get_schema("tradeaggression"))
        
        print("\nRunning sample query: SELECT count(*) FROM tradeaggression")
        count = engine.query_to_df("SELECT count(*) FROM tradeaggression").iloc[0, 0]
        print(f"Total rows in tradeaggression: {count}")
