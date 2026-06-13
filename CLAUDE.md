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

## Behaviour Rules

- **Read before answering data questions.** This warehouse contains live data past your training cutoff. Run `data-check` or a SQL query — never answer from memory.
- **Show plan before executing.** For any populate, backtest, or destructive operation, show the planned commands and get confirmation first.
- **Only populate missing dates.** Run `data-check` first, populate only the gaps.
- **Respect to_dt ≤ today − 2 days.** Market-data cron delivers yesterday's data at ~2am. Missing today and yesterday is expected.
- **`evict --apply` always needs explicit user confirmation.** Never run it autonomously.
- **Himalaya backtest is always synchronous.** Never run it in the background.
