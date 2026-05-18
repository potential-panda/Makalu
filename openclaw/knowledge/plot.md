## Plot (VolSpread)

### Supported datasets

| Dataset   | Required args                                      | Partition columns                                  |
|-----------|----------------------------------------------------|----------------------------------------------------|
| VolSpread | exchange, symbol, start, end, param-id             | exchange, symbol, execution_size, param_id         |

### Workflow (always follow this)

**Step 1 — List available param_ids:**
```bash
python /home/lulurun/workspace/Makalu/cli.py plot \
  --dataset VolSpread --exchange GMO --symbol BTC_JPY --list-params
```

Two outcomes:

**A) Data exists** — output lists `param_ids` (e.g. `vs_60_300_300_1s`).
If the user has not specified a param_id, use `vs_60_300_300_1s` (default) and tell the user.
Proceed to Step 2.

**B) No data** — no param_ids returned → VolSpread not populated. Follow the populate flow below.

---

### Populate flow when VolSpread is missing

VolSpread depends on MarketPrice. Always check and populate in this order:

1. Check bronze (Trade) availability:
   ```bash
   python /home/lulurun/workspace/Makalu/cli.py data-check \
     --dataset Trade --exchange GMO --symbol BTC_JPY --start ... --end ...
   ```
2. If bronze gaps → populate bronze first.
3. Check MarketPrice (execution_size always 0.002 for BTC):
   ```bash
   python /home/lulurun/workspace/Makalu/cli.py data-check \
     --dataset MarketPrice --exchange GMO --symbol BTC_JPY \
     --start ... --end ... --execution-size 0.002
   ```
4. Populate missing MarketPrice:
   ```bash
   python /home/lulurun/workspace/Makalu/cli.py populate MarketPrice \
     --exchange GMO --symbol BTC_JPY --start ... --end ... --execution-size 0.002
   ```
5. Check VolSpread (always pass --execution-size):
   ```bash
   python /home/lulurun/workspace/Makalu/cli.py data-check \
     --dataset VolSpread --exchange GMO --symbol BTC_JPY \
     --start ... --end ... --execution-size 0.002
   ```
6. Populate missing VolSpread:
   ```bash
   python /home/lulurun/workspace/Makalu/cli.py populate VolSpread \
     --exchange GMO --symbol BTC_JPY --start ... --end ... --execution-size 0.002
   ```
7. Run `scan` after any populate.
8. Retry `--list-params` to confirm, then proceed to Step 2.

**Important:** `data-check` and `populate` for VolSpread REQUIRE `--execution-size 0.002`.
Without it the command will not find the correct partition.

---

**Step 2 — Generate the plot:**
```bash
python /home/lulurun/workspace/Makalu/cli.py plot \
  --dataset VolSpread --exchange GMO --symbol BTC_JPY \
  --start 2026-03-01 --end 2026-03-17 \
  --param-id vs_60_300_300_1s \
  --output /tmp/VolSpread_GMO_BTC_JPY.png
```

If the plot reports no data for the date range, treat as missing data and go to the populate flow above.

---

### param_id naming for VolSpread

`vs_{shift}_{lookback}_{ema_span}_1s`

Examples:
- `vs_60_300_300_1s`  — short-term: 60s shift, 300s lookback, 300s EMA  ← **default**
- `vs_60_900_900_1s`  — medium-term
- `vs_180_900_900_1s` — wider shift
- `vs_300_900_900_1s` — long shift

### Chart content

VolSpread plots two lines:
- `volatility` (red) — EMA of realised volatility
- `spread_pct_ema` (blue) — EMA of bid-ask spread as fraction of mid price

High volatility_ema relative to spread_pct_ema = favourable trading conditions.
