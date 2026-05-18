
## SQL Query Rules

1. ALWAYS query the data warehouse before answering any factual question. Never use training knowledge for market data facts — this data is beyond your training cutoff.
2. For data availability questions, use the Makalu CLI `data-check` command — it reads the filesystem directly and is more reliable than SQL for availability checks.
3. Use exact column names from the Dataset Definitions section.
4. View names in SQL are lowercase dataset names (e.g. VolSpread → `volspread`).
5. Always filter by exchange, symbol, and date range when possible to avoid large scans.
6. ALWAYS use the `date` partition column (YYYY-MM-DD string) for date filtering — never compute or compare unix timestamps.
   Example: `WHERE date >= '2026-01-01' AND date <= '2026-01-07'`
7. For datasets with extra required partitions (e.g. `execution_size`, `param_id`), ALWAYS
   include them in the WHERE clause — omitting them mixes data across all partitions and
   produces nonsensical results. Discover values first if unknown:
   `SELECT DISTINCT param_id FROM volspread WHERE exchange='GMO' AND symbol='BTC_JPY' LIMIT 20`
   Then filter: `WHERE param_id = '...' AND execution_size = '0.002'`
8. Only SELECT queries are allowed. No DDL or data modification.
9. If a query returns no rows, say so and suggest why (wrong date range, wrong symbol, wrong param_id).
10. After showing results, suggest one or two follow-up analyses.

## Running SQL Queries

```bash
/home/lulurun/workspace/Makalu/openclaw/scripts/query --sql "SELECT ..."
/home/lulurun/workspace/Makalu/openclaw/scripts/query --list   # list registered datasets
```
