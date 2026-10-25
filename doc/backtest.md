# Backtest Workflow

## 1 Strategy: orion

- Default variant: `v601`
- Signal market: BINANCE (`BTC_USDT`, `ETH_USDT`, `XRP_USDT`)
- Trading market: GMO (`BTC_JPY`, `ETH_JPY`, `XRP_JPY`)
- Data required: `Trade` on signal market, `OrderBook` on trading market
- Order sizes: BTC=0.002, ETH=0.2, XRP=150; magnifier=5.0

Full strategy configs: `openclaw/knowledge/strategies.json`

---

## 2 Date Rules

- Format: `YYYYMMDD`
- `to_dt` ≤ today − 2 days — Kitakamakura's daily run delivers yesterday's bronze.
  Before 2am, even yesterday may be missing. Always use today − 2 as the safe upper bound.
- Missing today and yesterday is **expected and acceptable** — do not attempt to populate them.
- Any other missing dates within the requested range must be populated before running.

---

## 3 Workflow (always follow this order)

**3.1 Resolve dates**
Apply the today/yesterday rule. Convert natural language ("last 2 weeks") relative to today's date.

**3.2 Find signal and trading markets**
Look up the strategy config. For orion: trading_market determines the asset; signal_market is always BINANCE.

**3.3 data-check for required datasets**
```bash
# Signal market
python /home/lulurun/workspace/Makalu/src/cli.py data-check \
  --dataset Trade --exchange BINANCE --symbol BTC_USDT \
  --start 20260305 --end 20260317

# Trading market
python /home/lulurun/workspace/Makalu/src/cli.py data-check \
  --dataset OrderBook --exchange GMO --symbol BTC_JPY \
  --start 20260305 --end 20260317
```

**3.4 Populate gaps**
For any missing dates (excluding today and yesterday), follow the smart populate pattern in `doc/data_pipeline.md`. Show the plan to the user and confirm before executing.

**3.5 Check Docker stack**
```bash
python /home/lulurun/workspace/Makalu/src/cli.py docker-observability --format text
```
- If InfluxDB (8086) is down: warn user — backtest will run but metrics won't be saved. Ask to confirm or start the stack first.
- If Grafana (3000) is down: mention it but don't block.

**3.6 Show full plan and confirm**
Show: trading market, signal market, date range, env name. Get user confirmation.

**3.7 Run the three steps**

Step 1 — write env config files:
```bash
python /home/lulurun/workspace/Makalu/src/cli.py setup-backtest-env \
  --strategy orion \
  --trading-market GMO:BTC_JPY \
  --from-dt 20260305 \
  --to-dt 20260317
```
Output includes the env name (e.g. `BENJAMIN_TATUM`) — use it in step 2.

Step 2 — run Himalaya (ALWAYS synchronous — set exec timeout: 86400, NEVER run in background):
```bash
/home/lulurun/workspace/Himalaya/start.sh \
  --env BENJAMIN_TATUM \
  --from-dt 20260305 \
  --to-dt 20260317 \
  --retest \
  --optimize
```
Block here. When exec returns, proceed immediately to step 3.

Step 3 — publish results to Kilimanjaro viewer:
```bash
python /home/lulurun/workspace/Makalu/src/cli.py publish-results --env BENJAMIN_TATUM
```

---

## 4 Example

User: "run backtest for orion v601 for GMO BTC_JPY for the last 2 weeks"
```
→ trading_market = GMO:BTC_JPY
→ signal_market  = BINANCE:BTC_USDT  (from strategy config)
→ to_dt = today − 2 days, from_dt = to_dt − 14 days
→ data-check Trade BINANCE:BTC_USDT and OrderBook GMO:BTC_JPY
→ populate any gaps (skip today and yesterday)
→ check Docker stack
→ show plan, confirm
→ setup-backtest-env → Himalaya start.sh → publish-results
```

---

## 5 Key Rules

- Never run multiple backtests in parallel (memory constraint)
- Himalaya step must always be synchronous (blocking exec with 86400 s timeout)
- `publish-results` must always run immediately after Himalaya completes
- `to_dt` ≤ today − 2 days, always
