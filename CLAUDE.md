# Makalu — Claude Code Guide

Makalu is the crypto market datalake operator for the **orion** algorithmic trading strategy. It trades BTC/ETH/XRP spot on GMO using BINANCE as the signal market.

## Project Layout

```
src/cli.py          — main CLI entrypoint (all data ops)
src/makalu/         — library: data access, CLI handlers, storage
src/pipeline/       — Step/Stage/Pipeline framework for computing derived metrics
doc/                — single source of truth documentation (see below)
openclaw/           — agent workspace (memory)
```

## Key Paths

| Resource | Path |
|----------|------|
| Makalu CLI | `src/cli.py` (always run as `python /home/lulurun/workspace/Makalu/src/cli.py`) |
| Datalake root | `komachi.data_root()` — `ROOT_PATH` in `~/.kamakuraquantlab.env` (`/panda/makalu-data/`) |
| Parquet manifest | `<root>/manifest.db` |

## Documentation

All technical reference lives in `doc/`. Read the relevant file before starting a task.

| Topic | File |
|-------|------|
| Populate, scan, evict | `doc/data_pipeline.md` |
| SQL queries on the warehouse | `doc/querying.md` |
| Dataset column definitions | `doc/dataset_definitions.md` |
| Full dataset catalog with paths | `doc/catalog.yaml` |
| Volatility / Spread plots and Grafana export | `doc/plot.md` |
| Docker/InfluxDB/Grafana infrastructure | `doc/local_infra.md` |
| Implementing a new derived metric | `doc/implement_metric.md` |
| Static report site (`report`, `serve`) | `doc/report.md` |

## Data Availability Facts

- Official reliable data start date: **2025-07-01**. Data before 2025-07-01 may exist, but treat it as potentially incorrect unless the user explicitly approves using it.
- Known global outage: **2026-04-02 through 2026-04-09 inclusive**. The market data collectors crashed during this window. Bronze `Trade` and `OrderBook` reads intentionally model this window as available-but-empty, so consumers should see empty data and should not try to backfill it.

## Bronze Facts

- Makalu does not download, merge, or write bronze.
  Kitakamakura acquires raw market data and writes bronze `Trade` / `OrderBook`.
- Makalu reads bronze through Komachi (`komachi.bronze.file`) and takes the
  datalake root from `komachi.data_root()`. A missing bronze day reads as empty;
  filling it is Kitakamakura's job.

## MarketPrice Facts

- Makalu does not compute or write silver `MarketPrice`; Hase does:
  `hase derive MarketPrice --market GMO:BTC_JPY --execution-size 0.002 --start … --end …`
  (skips existing days; `--recreate` overwrites). Makalu only reads it.
- Hase drops snapshots that cannot fill the size or whose walked spread is > 100 bps,
  and records the counts in each file (`df.attrs["dropped"]`).

## Report Site Facts

- `python src/cli.py report leadlag ...` builds static pages into `~/env/LUIGI_MARIO/site`; it reads the warehouse and writes nothing to it.
- `makalu-web.service` (systemd --user) runs `python src/cli.py serve`: a read-only static file server on 127.0.0.1:9776, public at `http://157.120.33.185:7700/makalu/` through the eiger tunnel. No API, no jobs, no auth.

## Manifest / Eviction Facts

- Nothing schedules `scan` / `evict`; run them by hand. Data not yet scanned is considered fresh/protected.
- `scan` registers new non-bronze parquet files in `<root>/manifest.db` with `write_ts` initialized to the file's last-modified time.
- `evict` only considers files registered in `manifest.db`; unregistered files are never deleted by `evict`.
- `evict` is disk-cap driven: set the cap with `python src/cli.py evict --set-max-size 500GiB`. Usage counts every layer except bronze; when it exceeds the cap, files older than 6 months by `write_ts` are deleted. Bronze is never counted or deleted.
- Makalu does not depend on OS file-read access time; external DuckDB/Python reads do not refresh eviction age.
- There is no per-path or per-dataset delete option in `evict`.

## Behaviour Rules

- **Read before answering data questions.** This warehouse contains live data past your training cutoff. Run `data-check` or a SQL query — never answer from memory.
- **Show plan before executing.** For any populate or destructive operation, show the planned commands and get confirmation first.
- **Only populate missing dates.** Run `data-check` first. `populate` also skips any existing target `data.parquet` at write time; use `--recreate` only for intentional overwrites.
- **Respect to_dt ≤ today − 2 days.** Kitakamakura's daily run delivers yesterday's bronze. Missing today and yesterday is expected.
- **`evict --apply` always needs explicit user confirmation.** Never run it autonomously.
