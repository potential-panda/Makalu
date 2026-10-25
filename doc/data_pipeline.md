# Data Pipeline — Populate, Scan, Evict

All commands use the Makalu CLI:
```bash
python /home/lulurun/workspace/Makalu/src/cli.py <subcommand> [args]
```

---

## 1 Data Flow (strict order)

```
bronze/  Trade + OrderBook     ← Kitakamakura (not Makalu); located via Komachi
        │
        ├──▶ MarketPrice        ← hase derive MarketPrice  (silver, Hase)
        │         └──▶ VolSpread   (gold)
        │         └──▶ LeadLag     (gold, needs two markets)
        ├──▶ BookState          ← populate BookState    (silver)
        ├──▶ BookChange         ← populate BookChange   (silver)
        └──▶ TradeAggression    ← populate TradeAggression (silver)
```

Always populate in order — a dataset cannot be populated before its source exists.

Makalu does not download, merge, or write bronze. Kitakamakura acquires the raw
msgpack, converts it to bronze parquet, and owns its retention. Makalu reads
bronze through Komachi (`komachi.bronze.file`) and takes the datalake root from
Komachi (`komachi.data_root()`, i.e. `ROOT_PATH` in `~/.kamakuraquantlab.env`,
`/panda/makalu-data` on the warehouse host). A bronze day Komachi does not hold
reads as empty.

Makalu does not write silver `MarketPrice` either. Hase
(`KamakuraQuantLab/Hase`, `hase derive MarketPrice`) walks the book to the
execution size and writes it to the same path Makalu reads,
`silver/dataset=MarketPrice/exchange=…/symbol=…/execution_size=…/date=…`.
Hase leaves out snapshots the book cannot fill at that size and snapshots whose
walked spread is over 100 bps (collector artefacts), and records how many it
left out in the file's pandas metadata (`df.attrs["dropped"]`).

```bash
hase derive MarketPrice --market GMO:BTC_JPY --execution-size 0.002 \
  --start 2026-03-13 --end 2026-03-17            # skips days already on disk
hase derive MarketPrice --market GMO:BTC_JPY --execution-size 0.002 \
  --start 2026-03-13 --end 2026-03-17 --recreate # overwrite
```

---

## 1.1 Data Availability Assumptions

Treat **2025-07-01** as the official start date for reliable Makalu market data.
Data before 2025-07-01 may exist, but they may have
collection or correctness problems. Do not assume pre-2025-07-01 data is valid
for research, features, labels, or backtests unless the user explicitly confirms
that older data is acceptable.

Known global outage:

- **2026-04-02 through 2026-04-09 inclusive**: market data collectors crashed.
  Bronze `Trade` and `OrderBook` reads model this range as available-but-empty,
  so downstream consumers see empty data. Do not try to fill this range by
  re-running `populate`; treat it as a structural data gap.

---

## 2 Subcommands

### 2.1 data-check — inspect availability (READ-ONLY)

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

### 2.2 populate — derive silver/gold parquet files

Idempotent — each populate writer skips a write when its target `data.parquet`
already exists. Use `--recreate` only when you intentionally want to overwrite
existing target files.

```bash
# Silver (MarketPrice comes from Hase, see section 1)
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

# Overwrite existing target data.parquet files only when explicitly requested
python src/cli.py populate VolSpread \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17 \
  --execution-size 0.002 --recreate
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

### 2.3 scan — register new files in the manifest

Register Makalu's parquet files (every layer except bronze) in `<root>/manifest.db`.

Nothing schedules `scan` + `evict` any more (the HTTP service that did is
gone); run them by hand. Data not yet scanned is treated as fresh/protected:
`evict` only deletes files that are already registered in the manifest.

For newly-registered files, `scan` initializes `write_ts` to the file's
last-modified time. Makalu does not depend on OS file-read access time; external
DuckDB/Python reads do not refresh eviction age.

```bash
python src/cli.py scan
```

### 2.4 evict — cap-driven parquet cleanup (dry-run by default)

`evict` is destructive only with `--apply`. It has no per-path or per-dataset
delete option.

Policy:

- The parquet disk cap is stored in `manifest.db` in the `settings` table.
- Set it with `python src/cli.py evict --set-max-size 500GiB`.
- Usage is Makalu's layers only: everything under the datalake root except
  `bronze/`. At or below the cap, `evict` deletes nothing.
- Above the cap, `evict` selects only manifest-tracked files older than
  6 months by `write_ts`.
- Bronze is Kitakamakura's. It is never counted against the cap and never
  deleted by `evict`.
- Files not registered in `manifest.db` are never deleted by `evict`.

```bash
python src/cli.py evict --set-max-size 500GiB
python src/cli.py evict            # dry-run: shows what would be deleted
python src/cli.py evict --apply    # actually deletes — NEVER run without user confirmation
```

### 2.5 metrics-data — push prod trading metrics to InfluxDB

Standalone — downloads 5-min metrics from S3 and pushes to InfluxDB. Looks back 28 days.

```bash
python src/cli.py metrics-data              # all hosts
python src/cli.py metrics-data --host orion-01
```

Requires InfluxDB to be running (see `doc/local_infra.md`).

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
  1. hase derive MarketPrice --market GMO:BTC_JPY --execution-size 0.002 --start 2026-02-21 --end 2026-02-28  (gap)
  2. hase derive MarketPrice --market GMO:BTC_JPY --execution-size 0.002 --start 2026-03-13 --end 2026-03-17  (tail)
Proceed? [y/N]
```

---

## 4 Full Data Refresh Workflow

```bash
# 1. Check bronze is there (Kitakamakura delivers it; Makalu cannot fill gaps)
python src/cli.py data-check --dataset Trade --exchange GMO --symbol BTC_JPY \
  --start 2026-01-01 --end 2026-03-17

# 2. Check what silver is already on disk, then populate the missing ranges
python src/cli.py data-check --dataset MarketPrice --exchange GMO --symbol BTC_JPY \
  --start 2026-01-01 --end 2026-03-17 --execution-size 0.002

# 3. Derive silver MarketPrice with Hase (after confirming with user)
hase derive MarketPrice --market GMO:BTC_JPY --execution-size 0.002 \
  --start 2026-03-13 --end 2026-03-17

# 4. Sync manifest
python src/cli.py scan

# 5. Push trading metrics to InfluxDB
python src/cli.py metrics-data
```

---

## 5 Key Rules

- Bronze comes from Kitakamakura via Komachi; a bronze gap is Kitakamakura's to fill, not Makalu's
- Reliable data starts on 2025-07-01; older data may exist but should be treated as suspect
- 2026-04-02 through 2026-04-09 is a known global collector outage; do not try to backfill it
- `data-check` before any populate — never populate blindly
- Only populate **missing** date ranges when planning work. The CLI also skips any existing target `data.parquet` at write time; use `--recreate` only for intentional overwrites.
- `to_dt` for populate: at most today − 2 days (Kitakamakura's daily run delivers yesterday's bronze)
  Missing today and yesterday is expected — do not attempt to populate them
- Run `scan` after every populate
- `evict --apply` always requires explicit user confirmation
