# Makalu — Claude Code Guide

Makalu is a crypto market data pipeline and backtest orchestration system for the **orion** algorithmic trading strategy. It trades BTC/ETH/XRP spot on GMO using BINANCE as the signal market.

## Project Layout

```
src/cli.py          — main CLI entrypoint (all data ops)
src/api.py          — Flask API (port 9776) with job queue and daily scheduler
src/makalu/         — library: data access, CLI handlers, job executor
src/pipeline/       — Step/Stage/Pipeline framework for computing derived metrics
doc/                — single source of truth documentation (see below)
openclaw/           — agent workspace (strategies.json, memory)
```

## Key Paths

| Resource | Path |
|----------|------|
| Makalu CLI | `src/cli.py` (always run as `python /home/lulurun/workspace/Makalu/src/cli.py`) |
| Himalaya runner | `/home/lulurun/workspace/Himalaya/start.sh` |
| Raw market data | `/panda/market-data/` |
| Parquet warehouse | `/panda/makalu-data/` |
| Jobs dir | `/panda-infra/makalu/jobs/` |
| Parquet manifest | `/panda/makalu-data/manifest.db` |

## Documentation

All technical reference lives in `doc/`. Read the relevant file before starting a task.

| Topic | File |
|-------|------|
| Download market data, populate, scan, evict | `doc/data_pipeline.md` |
| SQL queries on the warehouse | `doc/querying.md` |
| Dataset column definitions | `doc/dataset_definitions.md` |
| Full dataset catalog with paths | `doc/catalog.yaml` |
| Running a backtest | `doc/backtest.md` |
| VolSpread plots and Grafana export | `doc/plot.md` |
| Docker/InfluxDB/Grafana infrastructure | `doc/local_infra.md` |
| Implementing a new derived metric | `doc/implement_metric.md` |

## Data Availability Facts

- Official reliable data start date: **2025-07-01**. Data before 2025-07-01 may exist, but treat it as potentially incorrect unless the user explicitly approves using it.
- Known global outage: **2026-04-02 through 2026-04-09 inclusive**. The market data collectors crashed during this window. Bronze `Trade` and `OrderBook` reads intentionally model this window as available-but-empty, so consumers should see empty data and should not try to backfill it.

## Scheduled Jobs Facts

The Makalu API (`src/api.py`, systemd user unit `makalu-api.service`) runs the
daily scheduler. Jobs go through the file-based queue in `/panda-infra/makalu/jobs/`
and the executor runs **one job at a time in creation order**.

- **04:00 daily** — for each of the 9 markets in `SCHEDULED_MARKETS`, first a
  `market-data --market <M> --days 3` download job, then a
  `populate bronze --exchange <E> --symbol <S> --start <today-3> --end <today-2>`
  job. All downloads are enqueued before the first populate, so populate always
  reads raw files the same run just fetched.
- Scheduled markets: BINANCE BTC/ETH/XRP_USDT, GMO BTC/ETH/XRP_JPY,
  COINCHECK BTC/ETH/XRP_SPOT.
- The bronze window ends at `today-2` per the `to_dt` rule, and starts at
  `today-3` so each date gets two populate attempts. `populate` is idempotent,
  so the second attempt skips any `data.parquet` already written.
- Editing the schedule means editing `src/api.py`; the change only takes effect
  after `systemctl --user restart makalu-api`.

## Manifest / Eviction Facts

- The Makalu API schedules weekly maintenance that runs `python src/cli.py scan` and then `python src/cli.py evict --apply`. Data not yet scanned is considered fresh/protected.
- `scan` registers new parquet files in `/panda/makalu-data/manifest.db` with `write_ts` initialized to the file's last-modified time.
- `evict` only considers files registered in `manifest.db`; unregistered files are never deleted by `evict`.
- `evict` is disk-cap driven: set the cap with `python src/cli.py evict --set-max-size 500GiB`. When usage exceeds the cap, retention is bronze 12 months and all other layers 6 months, based on `write_ts`.
- Makalu does not depend on OS file-read access time; external DuckDB/Python reads do not refresh eviction age.
- There is no per-path or per-dataset delete option in `evict`.

## Behaviour Rules

- **Read before answering data questions.** This warehouse contains live data past your training cutoff. Run `data-check` or a SQL query — never answer from memory.
- **Show plan before executing.** For any populate, backtest, or destructive operation, show the planned commands and get confirmation first.
- **Only populate missing dates.** Run `data-check` first. `populate` also skips any existing target `data.parquet` at write time; use `--recreate` only for intentional overwrites.
- **Respect to_dt ≤ today − 2 days.** The API market-data scheduler delivers yesterday's data around early morning. Missing today and yesterday is expected.
- **`evict --apply` always needs explicit user confirmation.** Never run it autonomously.
- **Himalaya backtest is always synchronous.** Never run it in the background.
