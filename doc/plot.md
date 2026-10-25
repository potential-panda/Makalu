# Plot and Grafana Export

## 1 Volatility and Spread Plot

`plot --dataset Volatility` draws `rv_bps`, `vol_bps` and `vol_bps_ema`; `plot --dataset Spread`
draws `spread_bps_ema`. All in bps. Volatility high relative to spread = favourable trading
conditions.

### 1.1 Step 1 — list available param_ids

Always do this first to confirm data exists and discover param_ids:
```bash
python /home/lulurun/workspace/Makalu/src/cli.py plot \
  --dataset Volatility --exchange GMO --symbol BTC_JPY --list-params
```

Two outcomes:
- **Data exists**: output lists param_ids (e.g. `g1s_h60s_lb300s_ema120s`). Proceed to 1.2.
- **No data**: not derived yet. Follow section 2.

### 1.2 Step 2 — generate PNG

```bash
python /home/lulurun/workspace/Makalu/src/cli.py plot \
  --dataset Volatility --exchange GMO --symbol BTC_JPY \
  --start 2026-03-01 --end 2026-03-17 \
  --param-id g1s_h60s_lb300s_ema120s \
  --output /tmp/Volatility_GMO_BTC_JPY.png
```

Default param_ids: `g1s_h60s_lb300s_ema120s` (Volatility), `g1s_ema300s` (Spread).

If the plot reports no data, treat as missing data and follow section 2.

### 1.3 param_id naming

Hase's spelling. The windows Makalu reads are in `src/makalu/activity_windows.py`.

| Dataset | param_id | Meaning |
|---------|----------|---------|
| Volatility | `g1s_h60s_lb300s_ema120s` | 1 s grid, 60 s return, 300 s lookback, 120 s EMA ← default |
| Volatility | `g1s_h60s_lb900s_ema300s` | 60 s return, 900 s lookback, 300 s EMA |
| Volatility | `g1s_h180s_lb900s_ema300s` | 180 s return, 900 s lookback, 300 s EMA |
| Volatility | `g1s_h300s_lb900s_ema300s` | 300 s return, 900 s lookback, 300 s EMA |
| Spread | `g1s_ema300s` | spread EMA over 300 s ← default |
| Spread | `g1s_ema900s` | spread EMA over 900 s |

---

## 2 Volatility and Spread Derive Flow

Both depend on MarketPrice, which depends on bronze. Derive in order:

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

# 5. Check Volatility / Spread (execution_size required)
$CLI data-check --dataset Volatility --exchange GMO --symbol BTC_JPY \
  --start ... --end ... --execution-size 0.002

# 6. Derive missing Spread and Volatility (Hase; existing days are skipped)
hase derive Spread --market GMO:BTC_JPY --execution-size 0.002 --start ... --end ... \
  --params 300 --params 900
hase derive Volatility --market GMO:BTC_JPY --execution-size 0.002 --start ... --end ... \
  --params 60:300:120 --params 60:900:300 --params 180:900:300 --params 300:900:300

# 7. Sync manifest
$CLI scan

# 8. Retry --list-params then proceed to section 1
```

`data-check` for Spread and Volatility always requires `--execution-size`. Without it the command will not find the correct partition.

---

## 3 Grafana Export

Pushes parquet data to InfluxDB and returns a Grafana dashboard URL.
Requires InfluxDB to be running — check first with `docker-observability` (see `doc/local_infra.md`).

```bash
python /home/lulurun/workspace/Makalu/src/cli.py grafana-export \
  --dataset Volatility \
  --exchange GMO --symbol BTC_JPY \
  --start 2026-03-01 --end 2026-03-17 \
  --execution-size 0.002 \
  --param-id g1s_h60s_lb300s_ema120s
```

Output: `{"bucket": "makalu_Volatility", "measurement": "Volatility", "grafana_url": "http://..."}`

### 3.1 Optional args by dataset

| Arg | Required for |
|-----|-------------|
| `--execution-size` | MarketPrice, Spread, Volatility |
| `--param-id` | Spread, Volatility, Chart_*, BacktestPnL |
| `--strategy` | BacktestPnL |
| `--exchange-b` / `--symbol-b` | LeadLag |
