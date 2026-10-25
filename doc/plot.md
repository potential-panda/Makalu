# Plot and Grafana Export

## 1 VolSpread Plot

Plots `volatility_ema` (red) and `spread_pct_ema` (blue) over a date range.
High volatility relative to spread_pct_ema = favourable trading conditions.

### 1.1 Step 1 — list available param_ids

Always do this first to confirm data exists and discover param_ids:
```bash
python /home/lulurun/workspace/Makalu/src/cli.py plot \
  --dataset VolSpread --exchange GMO --symbol BTC_JPY --list-params
```

Two outcomes:
- **Data exists**: output lists param_ids (e.g. `vs_60_300_120_1s`). Proceed to 1.2.
- **No data**: VolSpread not populated. Follow section 2.

### 1.2 Step 2 — generate PNG

```bash
python /home/lulurun/workspace/Makalu/src/cli.py plot \
  --dataset VolSpread --exchange GMO --symbol BTC_JPY \
  --start 2026-03-01 --end 2026-03-17 \
  --param-id vs_60_300_120_1s \
  --output /tmp/VolSpread_GMO_BTC_JPY.png
```

Default param_id: `vs_60_300_120_1s`

If the plot reports no data, treat as missing data and follow section 2.

### 1.3 param_id naming

`vs_{horizon}_{lookback}_{smooth}_{step}s` -- Hase's spelling (`hase derive VolSpread --params H:L:S`).
The four Makalu reads are in `src/makalu/vol_spread.py`.

| param_id | return horizon | lookback (and spread EMA) | volatility EMA |
|----------|-------|----------|----------|
| `vs_60_300_120_1s` | 60s | 300s | 120s | ← default |
| `vs_60_900_300_1s` | 60s | 900s | 300s |
| `vs_180_900_300_1s` | 180s | 900s | 300s |
| `vs_300_900_300_1s` | 300s | 900s | 300s |

---

## 2 VolSpread Populate Flow

VolSpread depends on MarketPrice, which depends on bronze. Populate in order:

```bash
CLI="python /home/lulurun/workspace/Makalu/src/cli.py"

# 1. Check bronze
$CLI data-check --dataset Trade --exchange GMO --symbol BTC_JPY --start ... --end ...

# 2. If gaps → bronze is Kitakamakura's to fill; Makalu cannot populate it

# 3. Check MarketPrice (execution_size required)
$CLI data-check --dataset MarketPrice --exchange GMO --symbol BTC_JPY \
  --start ... --end ... --execution-size 0.002

# 4. Derive missing MarketPrice (Hase)
hase derive MarketPrice --market GMO:BTC_JPY --execution-size 0.002 \
  --start ... --end ...

# 5. Check VolSpread (execution_size required)
$CLI data-check --dataset VolSpread --exchange GMO --symbol BTC_JPY \
  --start ... --end ... --execution-size 0.002

# 6. Derive missing VolSpread (Hase)
hase derive VolSpread --market GMO:BTC_JPY --execution-size 0.002 --start ... --end ... \
  --params 60:300:120 --params 60:900:300 --params 180:900:300 --params 300:900:300

# 7. Sync manifest
$CLI scan

# 8. Retry --list-params then proceed to section 1
```

`data-check` and `populate` for VolSpread always require `--execution-size`. Without it the command will not find the correct partition.

---

## 3 Grafana Export

Pushes parquet data to InfluxDB and returns a Grafana dashboard URL.
Requires InfluxDB to be running — check first with `docker-observability` (see `doc/local_infra.md`).

```bash
python /home/lulurun/workspace/Makalu/src/cli.py grafana-export \
  --dataset VolSpread \
  --exchange GMO --symbol BTC_JPY \
  --start 2026-03-01 --end 2026-03-17 \
  --execution-size 0.002 \
  --param-id vs_60_300_120_1s
```

Output: `{"bucket": "makalu_VolSpread", "measurement": "VolSpread", "grafana_url": "http://..."}`

### 3.1 Optional args by dataset

| Arg | Required for |
|-----|-------------|
| `--execution-size` | MarketPrice, VolSpread |
| `--param-id` | VolSpread, Chart_*, BacktestPnL |
| `--strategy` | BacktestPnL |
| `--exchange-b` / `--symbol-b` | LeadLag |
