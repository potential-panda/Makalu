# Makalu Tools — Exec Command Reference

All data queries and operations use the `exec` tool to run shell commands.

Canonical Makalu CLI:

```bash
python /home/lulurun/workspace/Makalu/cli.py <command> ...
```

---

## Docker Stack Health

```bash
# Check Grafana + InfluxDB status (run before backtest and before grafana-export/metrics-data)
python /home/lulurun/workspace/Makalu/cli.py docker-observability --format text

# JSON output (for programmatic use)
python /home/lulurun/workspace/Makalu/cli.py docker-observability
```

Output fields: `influxdb_up`, `grafana_up`, `healthy`, `warnings`, `docker_ps`.

If stack is down:
```bash
cd /panda-infra/Infra/docker/grafana && docker compose up -d
```

---

## SQL Query

```bash
python /home/lulurun/workspace/Makalu/cli.py query --sql "SELECT ..."
```

- Returns JSON: `{columns, row_count, rows}` (rows capped at 200)
- Add `--format table` for human-readable output
- List available datasets:

```bash
python /home/lulurun/workspace/Makalu/cli.py query --list
```

### SQL Rules

- **SELECT only** — no INSERT, UPDATE, DELETE, CREATE, DROP
- **Always filter** by `exchange`, `symbol`, and `date` range (avoid full scans)
- **Date column** is a string `'YYYY-MM-DD'` — use `>=` / `<=`
- **View names** are lowercase dataset names: `VolSpread` → `volspread`, `Chart_CANDLE` → `chart_candle`
- **Required partitions:**
  - `execution_size` required for `marketprice` and `volspread` (value: `'0.002'` for BTC)
  - `param_id` required for `volspread`, `chart_*`, `backtestpnl`
- Timestamps (`ts`) are Unix seconds (float); convert with `to_timestamp(ts)`

---

## Makalu CLI (data check / populate / scan / evict / market-data / metrics-data)

```bash
# --- S3 data ingestion (run before populate) ---

# Download raw market data from S3 → /panda/market-data/ (idempotent)
# Default: all standard markets (BINANCE BTC/ETH/XRP/SOL, GMO BTC/ETH/XRP/SOL), last 3 days
python /home/lulurun/workspace/Makalu/cli.py market-data
python /home/lulurun/workspace/Makalu/cli.py market-data --days 7
python /home/lulurun/workspace/Makalu/cli.py market-data --market GMO:BTC_JPY --days 1
# Output JSON: {status, markets, dates, downloaded, merged, skipped, errors}

# Download trading metrics from S3 → InfluxDB (idempotent, 28-day lookback)
python /home/lulurun/workspace/Makalu/cli.py metrics-data
python /home/lulurun/workspace/Makalu/cli.py metrics-data --host orion-01
# Output JSON: {status, hosts, downloaded, processed, skipped, errors}

# --- Parquet warehouse ---

# Check data availability (READ-ONLY — always run first)
python /home/lulurun/workspace/Makalu/cli.py data-check \
  --dataset Trade --exchange GMO --symbol BTC_JPY \
  --start 2026-01-01 --end 2026-03-17

# Populate bronze (raw market data → parquet; requires market-data to have run first)
python /home/lulurun/workspace/Makalu/cli.py populate bronze \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17

# Populate derived datasets (after bronze)
python /home/lulurun/workspace/Makalu/cli.py populate MarketPrice \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17 \
  --execution-size 0.002

python /home/lulurun/workspace/Makalu/cli.py populate VolSpread \
  --exchange GMO --symbol BTC_JPY --start 2026-03-13 --end 2026-03-17 \
  --execution-size 0.002

# Register newly populated files
python /home/lulurun/workspace/Makalu/cli.py scan

# Evict (dry-run by default — NEVER add --apply without user approval)
python /home/lulurun/workspace/Makalu/cli.py evict
```

### Makalu CLI Rules

- `market-data` runs before `populate bronze` — it downloads the raw msgpack source files
- `metrics-data` is standalone — pushes prod trading metrics to InfluxDB for Grafana
- `data-check` is read-only — always run before any populate
- `populate` is idempotent; only populate missing date ranges
- **Data flow order (strict):** `market-data` → `populate bronze` → `MarketPrice` / `BookState` / `BookChange` / `TradeAggression` → gold
- **Smart populate pattern:**
  1. `data-check` over the full date range to find gaps
  2. Build one `populate` call per contiguous missing range
  3. Show ALL planned commands to the user at once, confirm once
  4. Execute in order
- Run `scan` after every populate
- `data-check` for VolSpread **requires** `--execution-size 0.002`
- Exec timeouts: data-check/scan/market-data/metrics-data = 300 s; populate = 7200 s (use `timeout: 7200` in exec params)

---

## Backtest

### Strategy: orion

- Default variant: `v601`
- Signal market map: BTC → `BINANCE:BTC_USDT`, ETH → `BINANCE:ETH_USDT`, XRP → `BINANCE:XRP_USDT`
- Order size: BTC=0.002, ETH=0.2, XRP=150; magnifier=5.0
- Data requirements: `Trade` on signal market, `OrderBook` on trading market

### Backtest Workflow (always follow this order)

1. **Resolve dates** — `to_dt` = at most today − 2 days; format: `YYYYMMDD`
2. **Look up signal market** from strategy config above
3. **data-check** for `Trade` (signal market) and `OrderBook` (trading market)
4. **Populate any gaps** (skip today/yesterday) — show plan, confirm, execute
5. **Show full plan** (markets, dates, env) and ask user to confirm
6. **Run three-step exec:**

```bash
# Step 1 — write env config files
python /home/lulurun/workspace/Makalu/cli.py setup-backtest-env \
  --strategy orion \
  --trading-market GMO:BTC_JPY \
  --from-dt 20260305 \
  --to-dt 20260317

# Step 2 — run Himalaya (ALWAYS synchronous — set exec timeout: 86400, NEVER run in background)
# Running in background means the agent loses track of the process and cannot notify the user.
# Block here; when exec returns, proceed immediately to Step 3.
/panda/potential-panda/Himalaya/start.sh \
  --env BENJAMIN_TATUM \
  --from-dt 20260305 \
  --to-dt 20260317 \
  --optimize

# Step 3 — publish results to Kilimanjaro viewer (run immediately after Step 2 completes)
python /home/lulurun/workspace/Makalu/cli.py publish-results --env BENJAMIN_TATUM
```

---

## Plot (VolSpread)

```bash
# Step 1 — list available param_ids (always do this first)
python /home/lulurun/workspace/Makalu/cli.py plot \
  --dataset VolSpread --exchange GMO --symbol BTC_JPY --list-params

# Step 2 — generate PNG
python /home/lulurun/workspace/Makalu/cli.py plot \
  --dataset VolSpread --exchange GMO --symbol BTC_JPY \
  --start 2026-03-01 --end 2026-03-17 \
  --param-id vs_60_300_300_1s \
  --output /tmp/VolSpread_GMO_BTC_JPY.png
```

### Plot Rules

- Always list param_ids first if user hasn't specified one
- Default param_id: `vs_60_300_300_1s`
- If no data → populate flow: bronze → MarketPrice → VolSpread → scan → retry
- VolSpread `data-check` always requires `--execution-size 0.002`
- Populate order: bronze → MarketPrice → VolSpread

---

## Grafana Export

```bash
# Push parquet data to InfluxDB and get a Grafana URL
python /home/lulurun/workspace/Makalu/cli.py grafana-export \
  --dataset VolSpread \
  --exchange GMO --symbol BTC_JPY \
  --start 2026-03-01 --end 2026-03-17 \
  --execution-size 0.002 \
  --param-id vs_60_300_300_1s
# → {"bucket": "makalu_VolSpread", "measurement": "VolSpread", "grafana_url": "http://..."}
```

Optional args by dataset:
- `--execution-size` — required for MarketPrice, VolSpread
- `--param-id` — required for VolSpread, Chart_*, BacktestPnL
- `--strategy` — required for BacktestPnL
- `--exchange-b` / `--symbol-b` — required for LeadLag

---

## Reference Files

```bash
cat /home/lulurun/workspace/Makalu/doc/catalog.yaml           # all datasets
cat /home/lulurun/workspace/Makalu/doc/dataset_definitions.md  # column definitions
cat /home/lulurun/workspace/Makalu/openclaw/knowledge/strategies.json  # strategy configs
```

---

## Key Paths

| Resource | Path |
|---|---|
| Makalu CLI | `/home/lulurun/workspace/Makalu/cli.py` |
| Query script | `/home/lulurun/workspace/Makalu/openclaw/scripts/query` |
| Setup backtest script | `/home/lulurun/workspace/Makalu/openclaw/scripts/setup_backtest_env` |
| Publish results script | `/home/lulurun/workspace/Makalu/openclaw/scripts/publish_results` |
| Himalaya runner | `/panda/potential-panda/Himalaya/start.sh` |
| Dataset catalog | `/home/lulurun/workspace/Makalu/doc/catalog.yaml` |
| Column definitions | `/home/lulurun/workspace/Makalu/doc/dataset_definitions.md` |
| Strategy configs | `/home/lulurun/workspace/Makalu/openclaw/knowledge/strategies.json` |
| Parquet data root | `/panda/makalu-data/` |
| Backtest env dir | `/home/lulurun/env/BENJAMIN_TATUM/` |
| Docker compose dir | `/panda-infra/Infra/docker/grafana/` |
