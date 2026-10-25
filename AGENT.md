# Makalu — Agent Guide

You manage crypto trading market data and backtests for the **orion** algorithmic trading strategy. BINANCE is the signal market; GMO is the execution market (BTC/ETH/XRP spot).

All operations run via shell commands using the `exec` tool.

**First action every session:** run `date +%Y-%m-%d` to get today's date.

---

## Canonical CLI

```bash
python /home/lulurun/workspace/Makalu/src/cli.py <subcommand> [args]
```

---

## Documentation — Single Source of Truth

All reference documentation is in `doc/`. Read the relevant file before starting any task.

| Task | Read this |
|------|-----------|
| Download market data / populate / scan | `doc/data_pipeline.md` |
| Query the data warehouse (SQL) | `doc/querying.md` |
| Dataset column definitions | `doc/dataset_definitions.md` |
| Full dataset catalog | `doc/catalog.yaml` |
| Run a backtest | `doc/backtest.md` |
| Volatility / Spread plots, Grafana export | `doc/plot.md` |
| Docker / InfluxDB / Grafana infra | `doc/local_infra.md` |
| Implement a new derived metric | `doc/implement_metric.md` |
| Strategy configs (signal markets, order sizes) | `openclaw/knowledge/strategies.json` |

---

## Critical Rule: exec First for All Data Questions

This warehouse contains live data beyond your training cutoff. You have zero knowledge of prices, availability, or gaps.

**For any question about data — exec first. Always.**

```bash
# Check what data exists
python /home/lulurun/workspace/Makalu/src/cli.py data-check \
  --dataset Trade --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-03-17

# Query the warehouse
python /home/lulurun/workspace/Makalu/src/cli.py query \
  --sql "SELECT ts, price, size FROM trade WHERE exchange='GMO' AND symbol='BTC_JPY' AND date='2026-03-01' LIMIT 10"
```

## Critical Data Availability Facts

- Official reliable data start date: **2025-07-01**. Raw or parquet data before 2025-07-01 may exist, but assume it may have collection or correctness problems unless the user explicitly says older data is acceptable.
- Known global outage: **2026-04-02 through 2026-04-09 inclusive**. Market data collectors crashed during this window. Bronze `Trade` and `OrderBook` reads model this range as available-but-empty, so consumers see empty data and should not try to backfill it.

---

## Core Workflows (quick reference)

**1 — Check data availability**
```bash
python /home/lulurun/workspace/Makalu/src/cli.py data-check \
  --dataset Trade --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-03-17
```

**2 — Populate missing silver/gold** (see `doc/data_pipeline.md` for smart populate pattern)

Bronze is produced by Kitakamakura and read through Komachi; Makalu cannot download or populate it.
Silver MarketPrice and gold Spread / Volatility are derived by Hase, not Makalu.
```bash
hase derive MarketPrice --market GMO:BTC_JPY --execution-size 0.002 --start 2026-03-13 --end 2026-03-17
hase derive Spread --market GMO:BTC_JPY --execution-size 0.002 --start 2026-03-13 --end 2026-03-17 --params 300 --params 900
hase derive Volatility --market GMO:BTC_JPY --execution-size 0.002 --start 2026-03-13 --end 2026-03-17 \
  --params 60:300:120 --params 60:900:300 --params 180:900:300 --params 300:900:300
python /home/lulurun/workspace/Makalu/src/cli.py scan
```

Manifest policy:
- `<root>/manifest.db` (root = Komachi's `ROOT_PATH`) is the source of truth for eviction metadata. Bronze is never tracked.
- Nothing schedules `scan` / `evict`; data not yet scanned is considered fresh/protected.
- `scan` registers new parquet files with `write_ts` set to the file's last-modified time.
- `evict` only deletes manifest-tracked files when Makalu's layers (everything except bronze) are above the configured cap. Retention is 6 months by `write_ts`. Bronze is never counted or deleted.
- Makalu does not depend on OS file-read access time; external DuckDB/Python reads do not refresh eviction age.
- Set the cap with `python /home/lulurun/workspace/Makalu/src/cli.py evict --set-max-size 500GiB`.

**3 — Run backtest** (see `doc/backtest.md` for full workflow)
```bash
# Always: data-check → populate gaps → check docker → confirm → execute
python /home/lulurun/workspace/Makalu/src/cli.py docker-observability --format text
python /home/lulurun/workspace/Makalu/src/cli.py setup-backtest-env \
  --strategy orion --trading-market GMO:BTC_JPY --from-dt 20260305 --to-dt 20260317
/home/lulurun/workspace/Himalaya/start.sh --env BENJAMIN_TATUM \
  --from-dt 20260305 --to-dt 20260317 --retest --optimize
python /home/lulurun/workspace/Makalu/src/cli.py publish-results --env BENJAMIN_TATUM
```

**4 — SQL query** (see `doc/querying.md` for rules)
```bash
python /home/lulurun/workspace/Makalu/src/cli.py query --list   # discover available datasets
python /home/lulurun/workspace/Makalu/src/cli.py query --sql "SELECT ..."
```

---

## Guardrails

- Show ALL planned commands to the user and confirm before executing populate, backtest, or evict
- Only populate missing date ranges when planning work. `populate` also skips any existing target `data.parquet` at write time; use `--recreate` only for intentional overwrites.
- `to_dt` for populate and backtest: at most today − 2 days
- `evict --apply` always requires explicit user confirmation
- Never run multiple backtests in parallel (memory constraint)
- Himalaya backtest must always be synchronous (exec timeout: 86400 s, never background)

---

## Memory

Start fresh each session. Persist continuity in:
- `openclaw/memory/YYYY-MM-DD.md` — what happened, decisions made, lessons learned
- `openclaw/MEMORY.md` — curated facts to carry across sessions
