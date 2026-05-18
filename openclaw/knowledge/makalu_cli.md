## Data Maintenance (Makalu CLI)

**In Eiger:** use the `run_cli` tool with the args listed below.

**In Claude Code:** use Bash:
```
python /home/lulurun/workspace/Makalu/cli.py <subcommand> [args]
```

### Subcommands

- `data-check`  READ-ONLY. Check what parquet data exists on disk. Always run this first before populate.
  Progressive: omit args to discover available exchanges/symbols/partition values.
  Output includes: min date, max date, total days, missing date ranges.

- `market-data`  WRITE. Download raw market data (msgpack) from S3 to `/panda/market-data/`.
  Idempotent — skips files already on disk.
  Run this **before** `populate bronze` to ensure source files are present.
  Outputs JSON: `{status, markets, dates, downloaded, merged, skipped, errors}`.

- `metrics-data`  WRITE. Download 5-min trading metrics from S3 and push to InfluxDB.
  Idempotent — skips already-downloaded or already-processed files.
  Looks back up to 28 days. Hosts: orion-01, orion-02.
  Outputs JSON: `{status, hosts, downloaded, processed, skipped, errors}`.

- `populate`  WRITE. Convert raw msgpack files (bronze) or derive silver/gold metrics.
  Idempotent — safe to re-run.
  Dataset names: `bronze`, `MarketPrice`, `BookState`, `BookChange`, `TradeAggression`.
  Bronze is always first — all silver datasets depend on it.
  NEVER call populate without first running data-check to identify missing dates.
  Only populate the missing date ranges — do not re-populate dates already on disk.

- `scan`  WRITE. Register newly populated files into the manifest. Run after populate.

- `evict`  DESTRUCTIVE. Shows files past retention (dry-run by default).
  NEVER pass --apply without explicit user confirmation.

### Data flow (must populate in this order)
```
bronze (Trade + OrderBook)
  → MarketPrice               (silver, per market)
      → VolSpread             (gold, per market)
  → LeadLag                   (gold, requires MarketPrice for both markets in the pair)
  → BookState / BookChange / TradeAggression  (silver)
```

### Smart populate — ALWAYS follow this when the user asks to populate data

1. Run `data-check` with `--start` and `--end` to find exactly which dates are missing.
2. From the output, identify:
   (a) internal gaps within the existing range
   (b) the tail: dates after Max date up to the requested end date
3. Build the minimal set of populate calls — one per contiguous missing range.
   Do NOT populate dates that already exist on disk.
4. Show ALL planned populate commands to the user at once before asking for confirmation:
   ```
   I need to run 2 populate commands:
     1. populate bronze ... --start 2026-02-21 --end 2026-02-28  (gap)
     2. populate bronze ... --start 2026-03-13 --end 2026-03-17  (tail)
   Proceed? [y/N]
   ```
5. Only after user confirms, execute them in order.

### Execution size by asset

`MarketPrice`, `VolSpread`, and `LeadLag` are partitioned by `execution_size`.
Always use the correct value for the asset — using the wrong size creates a mismatched partition:

| Asset | execution_size |
|-------|---------------|
| BTC   | 0.002         |
| ETH   | 0.2           |
| XRP   | 150           |

### Full data refresh workflow

When the user asks to refresh market data or check data availability, follow this order:

1. Download raw msgpack files (source data for bronze):
   ```
   python cli.py market-data                         # all markets, last 3 days
   python cli.py market-data --days 7                # last 7 days
   python cli.py market-data --market GMO:BTC_JPY    # single market
   ```

2. Check what bronze/silver data is already on disk:
   ```
   python cli.py data-check --dataset Trade --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-03-17
   ```

3. Populate missing bronze then silver (only for missing date ranges):
   ```
   python cli.py populate bronze --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17
   python cli.py populate MarketPrice --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17 --execution-size 0.002
   python cli.py populate MarketPrice --exchange GMO --symbol XRP_JPY --start 2026-03-13 --end 2026-03-17 --execution-size 150
   ```

4. Update manifest:
   ```
   python cli.py scan
   ```

5. Push prod trading metrics to InfluxDB (Grafana monitoring):
   ```
   python cli.py metrics-data                       # all hosts
   python cli.py metrics-data --host orion-01       # single host
   ```

6. Periodic cleanup (always dry-run first):
   ```
   python cli.py evict
   python cli.py evict --apply   # only after explicit user confirmation
   ```
