# Data Pipeline — Market Data, Populate, Cleanup, Scan, Evict

All commands use the Makalu CLI:
```bash
python /home/lulurun/workspace/Makalu/src/cli.py <subcommand> [args]
```

---

## 1 Data Flow (strict order)

```
S3 raw msgpack files  (market-data download)
        │
        ▼
bronze/  Trade + OrderBook     ← populate bronze
        │
        ├──▶ MarketPrice        ← populate MarketPrice  (silver)
        │         └──▶ VolSpread   (gold)
        │         └──▶ LeadLag     (gold, needs two markets)
        ├──▶ BookState          ← populate BookState    (silver)
        ├──▶ BookChange         ← populate BookChange   (silver)
        └──▶ TradeAggression    ← populate TradeAggression (silver)
```

Always populate in order — a dataset cannot be populated before its source exists.

---

## 1.1 Data Availability Assumptions

Treat **2025-07-01** as the official start date for reliable Makalu market data.
Raw files before 2025-07-01 may exist in S3 or local storage, but they may have
collection or correctness problems. Do not assume pre-2025-07-01 data is valid
for research, features, labels, or backtests unless the user explicitly confirms
that older data is acceptable.

Known global outage:

- **2026-04-02 through 2026-04-09 inclusive**: market data collectors crashed.
  Bronze `Trade` and `OrderBook` reads model this range as available-but-empty,
  so downstream consumers see empty data. Do not try to fill this range by
  re-running `market-data` or `populate`; treat it as a structural data gap.

---

## 2 Subcommands

### 2.1 market-data — download raw msgpack from S3

Downloads raw market data files to `/panda/market-data/`. Run **before** `populate bronze`.
Idempotent — skips files already on disk.

```bash
# All standard markets, last 3 days (default)
python src/cli.py market-data

# Last N days
python src/cli.py market-data --days 7

# Exact date window
python src/cli.py market-data --start 2025-07-01 --end 2025-07-31

# Single market
python src/cli.py market-data --market GMO:BTC_JPY --days 1
python src/cli.py market-data --market BINANCE:BTC_USDT --days 3
```

Standard markets: BINANCE BTC/ETH/XRP USDT, GMO BTC/ETH/XRP JPY.

Output JSON: `{status, markets, dates, downloaded, merged, skipped, errors}`

Exec timeout: 300 s.

### 2.1.1 market-data-s3 — inspect raw S3 availability (READ-ONLY)

Lists available raw market-data markets and date ranges in S3 without
downloading files. A full scan paginates S3 prefixes across all known raw
market-data hosts, so the inventory is cached at:

```bash
/panda/makalu-data/.cache/market_data_s3_inventory.json
```

Use the cache:

```bash
python src/cli.py market-data-s3
python src/cli.py market-data-s3 --market GMO:BTC_SPOT
```

Refresh the cache from S3:

```bash
python src/cli.py market-data-s3 --refresh
```

Check a specific window for a market:

```bash
python src/cli.py market-data-s3 \
  --market GMO:BTC_SPOT --start 2026-04-01 --end 2026-04-15
```

### 2.2 data-check — inspect availability (READ-ONLY)

Always run before populate to find missing date ranges. Never modifies data.

```bash
python src/cli.py data-check \
  --dataset Trade --exchange GMO --symbol BTC_JPY \
  --start 2026-01-01 --end 2026-03-17
```

Output includes: min date, max date, total days, **missing date ranges**.

Omit `--start`/`--end` to see all available dates. Omit `--exchange`/`--symbol` for progressive discovery.

For datasets with extra partitions:
```bash
# MarketPrice / VolSpread require --execution-size
python src/cli.py data-check \
  --dataset MarketPrice --exchange GMO --symbol BTC_JPY \
  --start 2026-01-01 --end 2026-03-17 --execution-size 0.002
```

### 2.3 populate — convert or derive parquet files

Idempotent — each populate writer skips a write when its target `data.parquet`
already exists. Use `--recreate` only when you intentionally want to overwrite
existing target files.

```bash
# Bronze (raw tick data)
python src/cli.py populate bronze \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17

# Overwrite existing target data.parquet files only when explicitly requested
python src/cli.py populate bronze \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17 --recreate

# Silver
python src/cli.py populate MarketPrice \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17 \
  --execution-size 0.002

python src/cli.py populate BookState \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17

python src/cli.py populate BookChange \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17

python src/cli.py populate TradeAggression \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17

# Gold
python src/cli.py populate VolSpread \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17 \
  --execution-size 0.002
```

Exec timeout: 7200 s (populate can be slow for large date ranges).

#### execution_size by asset

`MarketPrice`, `VolSpread`, `LeadLag` are partitioned by `execution_size`.
Using the wrong value creates a mismatched partition.

| Asset | execution_size |
|-------|---------------|
| BTC   | 0.002         |
| ETH   | 0.2           |
| XRP   | 150           |

### 2.4 scan — register new files in the manifest

Register parquet files in `/panda/makalu-data/manifest.db`.

The API schedules weekly maintenance that runs `scan` and then `evict --apply`.
This is acceptable because data not yet scanned is treated as fresh/protected:
`evict` only deletes files that are already registered in the manifest.

For newly-registered files, `scan` initializes `write_ts` to the file's
last-modified time. Makalu does not depend on OS file-read access time; external
DuckDB/Python reads do not refresh eviction age.

```bash
python src/cli.py scan
```

### 2.5 evict — cap-driven parquet cleanup (dry-run by default)

`evict` is destructive only with `--apply`. It has no per-path or per-dataset
delete option.

Policy:

- The parquet disk cap is stored in `manifest.db` in the `settings` table.
- Set it with `python src/cli.py evict --set-max-size 500GiB`.
- If `/panda/makalu-data` is at or below the cap, `evict` deletes nothing.
- If usage is above the cap, `evict` selects only manifest-tracked files past
  retention by `write_ts`: bronze 12 months, all other layers 6 months.
- Files not registered in `manifest.db` are never deleted by `evict`.

```bash
python src/cli.py evict --set-max-size 500GiB
python src/cli.py evict            # dry-run: shows what would be deleted
python src/cli.py evict --apply    # actually deletes — NEVER run without user confirmation
```

### 2.6 metrics-data — push prod trading metrics to InfluxDB

Standalone — downloads 5-min metrics from S3 and pushes to InfluxDB. Looks back 28 days.

```bash
python src/cli.py metrics-data              # all hosts
python src/cli.py metrics-data --host orion-01
```

Requires InfluxDB to be running (see `doc/local_infra.md`).

### 2.7 market-data-cleanup — delete raw msgpack already in bronze

Deletes raw files from `/panda/market-data/` only when the same exchange,
symbol, and date exist in **both** bronze `Trade` and bronze `OrderBook`.
Keeps the most recent 3 calendar days by default.

Dry-run by default:

```bash
python src/cli.py market-data-cleanup
python src/cli.py market-data-cleanup --market GMO:BTC_JPY
```

Apply after reviewing the dry-run output:

```bash
python src/cli.py market-data-cleanup --apply
```

Optional retention override:

```bash
python src/cli.py market-data-cleanup --keep-days 7
```

---

## 3 Smart Populate Pattern

**Always follow this when populating any dataset:**

1. Run `data-check` over the full requested date range to find exactly which dates are missing.
2. From the output, identify:
   - Internal gaps within the existing range
   - The tail: dates after the max available date up to the requested end
3. Build the minimal set of populate calls — one per contiguous missing range.
   Do NOT populate dates that already exist on disk.
4. Show ALL planned populate commands to the user before asking for confirmation.
5. Only after the user confirms, execute them in order.
6. Run `scan` after the last populate.

Example plan to show the user:
```
I need to run 2 populate commands:
  1. populate bronze --start 2026-02-21 --end 2026-02-28  (gap)
  2. populate bronze --start 2026-03-13 --end 2026-03-17  (tail)
Proceed? [y/N]
```

---

## 4 Full Data Refresh Workflow

```bash
# 1. Download raw source files
python src/cli.py market-data --days 7

# 2. Check what's already on disk
python src/cli.py data-check --dataset Trade --exchange GMO --symbol BTC_JPY \
  --start 2026-01-01 --end 2026-03-17

# 3. Populate missing bronze (after confirming with user)
python src/cli.py populate bronze \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17

# 4. Populate silver
python src/cli.py populate MarketPrice \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17 \
  --execution-size 0.002

# 5. Sync manifest
python src/cli.py scan

# 6. Push trading metrics to InfluxDB
python src/cli.py metrics-data
```

---

## 4.1 Scheduled Daily Jobs

The Makalu API (`src/api.py`, systemd user unit `makalu-api.service`) enqueues
the daily pipeline at **04:00 local time**. Jobs land in
`/panda-infra/makalu/jobs/` and the executor runs one at a time in creation
order.

Scheduled markets (`SCHEDULED_MARKETS`):

| Exchange | Symbols |
|----------|---------|
| BINANCE | `BTC_USDT`, `ETH_USDT`, `XRP_USDT` |
| GMO | `BTC_JPY`, `ETH_JPY`, `XRP_JPY` |
| COINCHECK | `BTC_SPOT`, `ETH_SPOT`, `XRP_SPOT` |

Each run enqueues, in this order:

1. 9 download jobs — `market-data --market <EXCHANGE:SYMBOL> --days 3`
   (covers `today-3` → `today-1`).
2. 9 bronze populate jobs — `populate bronze --exchange <E> --symbol <S>
   --start <today-3> --end <today-2>`.

Because the queue is sequential, all downloads finish before the first populate
reads `/panda/market-data`.

The populate window ends at `today-2` per the `to_dt` rule (§5) and starts at
`today-3`, so every date gets two populate attempts on consecutive days. This
is safe: `populate` skips any target `data.parquet` that already exists, so the
second attempt is a no-op for dates already written.

Weekly maintenance (`scan` + `evict --apply`) runs Sunday at 05:00.

To change the schedule, edit `SCHEDULED_MARKETS`, `DAILY_SYNC_DAYS`, or
`BRONZE_POPULATE_{START,END}_DAYS_AGO` in `src/api.py`, then:

```bash
systemctl --user restart makalu-api
curl -s localhost:9776/api/scheduler | python -m json.tool
```

---

## 5 Key Rules

- `market-data` before `populate bronze` — source files must exist first
- Reliable data starts on 2025-07-01; older data may exist but should be treated as suspect
- 2026-04-02 through 2026-04-09 is a known global collector outage; do not try to backfill it
- `data-check` before any populate — never populate blindly
- Only populate **missing** date ranges when planning work. The CLI also skips any existing target `data.parquet` at write time; use `--recreate` only for intentional overwrites.
- `to_dt` for populate: at most today − 2 days (the API market-data scheduler delivers yesterday around early morning)
  Missing today and yesterday is expected — do not attempt to populate them
- Run `scan` after every populate
- `market-data-cleanup` is dry-run by default; only use `--apply` after reviewing the files
- `evict --apply` always requires explicit user confirmation
