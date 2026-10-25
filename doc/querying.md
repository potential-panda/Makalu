# Querying the Data Warehouse

Two query methods are available:

```bash
# SQL query (returns JSON by default)
python /home/lulurun/workspace/Makalu/src/cli.py query --sql "SELECT ..."

# Human-readable table output
python /home/lulurun/workspace/Makalu/src/cli.py query --sql "SELECT ..." --format table

# List all registered datasets and their view names
python /home/lulurun/workspace/Makalu/src/cli.py query --list
```

Returns JSON: `{columns, row_count, rows}` — rows capped at 200.

**Always use the CLI for data questions** — this warehouse contains live data beyond any model's training cutoff. Never answer data questions from memory.

---

## 1 SQL Rules

1. **SELECT only** — no INSERT, UPDATE, DELETE, CREATE, DROP.
2. **Always filter** by `exchange`, `symbol`, and `date` range — avoids full scans.
3. **Date column** is a string `'YYYY-MM-DD'`. Use `>=` / `<=`, not timestamp arithmetic.
   ```sql
   WHERE date >= '2026-01-01' AND date <= '2026-01-07'
   ```
4. **View names** are lowercase dataset names: `Trade` → `trade`, `VolSpread` → `volspread`.
5. **Required partition columns** — for datasets with extra partitions, always include them or results are mixed and meaningless:
   - `execution_size` required for `marketprice`, `volspread`
   - `param_id` required for `volspread`, `chart_*`, `backtestpnl`
6. **Discover partition values before querying** if unknown:
   ```sql
   SELECT DISTINCT execution_size FROM marketprice WHERE exchange='GMO' AND symbol='BTC_JPY' LIMIT 20
   SELECT DISTINCT param_id FROM volspread WHERE exchange='GMO' AND symbol='BTC_JPY' LIMIT 20
   ```
7. **Timestamps** (`ts`) are Unix seconds (float). Convert with `to_timestamp(ts)` for display.
8. If a query returns no rows, say so and suggest why (wrong date range, wrong symbol, missing partition value).

---

## 2 Example Queries

### 2.1 Bronze — raw trades
```sql
SELECT ts, side, price, size
FROM trade
WHERE exchange='GMO' AND symbol='BTC_JPY'
  AND date >= '2026-03-01' AND date <= '2026-03-03'
ORDER BY ts
LIMIT 100
```

### 2.2 Bronze — order book snapshot
```sql
SELECT ts, bid0_price, bid0_qty, ask0_price, ask0_qty
FROM orderbook
WHERE exchange='GMO' AND symbol='BTC_JPY'
  AND date = '2026-03-01'
LIMIT 50
```

### 2.3 Silver — execution prices
```sql
SELECT ts, bid, ask, mid, spread_bps
FROM marketprice
WHERE exchange='GMO' AND symbol='BTC_JPY'
  AND execution_size='0.002'
  AND date >= '2026-03-01' AND date <= '2026-03-03'
ORDER BY ts
LIMIT 100
```

### 2.4 Gold — VolSpread (always include param_id and execution_size)
```sql
SELECT ts, mid, volatility_ema, spread_pct_ema
FROM volspread
WHERE exchange='GMO' AND symbol='BTC_JPY'
  AND execution_size='0.002'
  AND param_id='vs_60_300_120_1s'
  AND date >= '2026-03-01' AND date <= '2026-03-07'
ORDER BY ts
LIMIT 200
```

---

## 3 data-check for Availability (Not SQL)

For checking whether data **exists** for a date range, prefer `data-check` over SQL — it reads the filesystem directly and is faster and more reliable:

```bash
python /home/lulurun/workspace/Makalu/src/cli.py data-check \
  --dataset Trade --exchange GMO --symbol BTC_JPY \
  --start 2026-01-01 --end 2026-03-17
```

Output includes missing date ranges. Use this before any populate. See `doc/data_pipeline.md`.

---

## 4 Dataset Reference

- **Column definitions**: `doc/dataset_definitions.md`
- **Full catalog with paths**: `doc/catalog.yaml`
