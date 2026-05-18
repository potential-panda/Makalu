## Backtest

### 1 Strategy knowledge

**orion** (default variant: v601)
- Signal market mapping: BTC → `BINANCE:BTC_USDT`, ETH → `BINANCE:ETH_USDT`, XRP → `BINANCE:XRP_USDT`
- order_size by asset: BTC=0.002, ETH=0.2, XRP=150 — magnifier=5.0
- Signal datasets to check: `Trade` (on the signal market exchange/symbol)
- Trading datasets to check: `OrderBook` (on the trading market exchange/symbol)

### 2 Date resolution rules

- Resolve natural language like "last 2 weeks" relative to today's date injected in the system prompt. Note: today's date is set at session start — for long-running sessions (e.g. Discord bot), the date may be stale. If the user says "today is X", use that instead.
- The market data cronjob runs at 2am and can only deliver yesterday's data.
  Before 2am, even yesterday may be missing. **Always set to_dt to at most 2 days before today.**
  Missing today and yesterday is expected and acceptable — do not attempt to populate them.
- Any other missing dates within the requested range must be populated before running.

### 3 Backtest flow (always follow this)

1. Resolve dates — apply the today/yesterday rule. Dates format: YYYYMMDD.
2. Look up the strategy above to find: signal_market, data requirements.
3. Run `data-check` for each required dataset over the full date range:
   - signal market: check `Trade` dataset
   - trading market: check `OrderBook` dataset
4. For any missing dates (excluding today and yesterday), plan and confirm `populate bronze` commands.
5. Show the full backtest plan (markets, dates, env) and ask the user to confirm.
6. Set up the env and run the backtest.

### 4 Running

Step 1 — Set up the env (writes trading.json, config.json, param_grid.json):
```bash
/home/lulurun/workspace/Makalu/openclaw/scripts/setup_backtest_env \
  --strategy orion \
  --trading-market GMO:BTC_JPY \
  --from-dt 20260305 \
  --to-dt 20260317
```

Step 2 — Run:
```bash
/panda/potential-panda/Himalaya/start.sh \
  --env BENJAMIN_TATUM \
  --from-dt 20260305 \
  --to-dt 20260317 \
  --optimize
```

### 5 Examples

User: "run backtest for orion v601 for GMO BTC_JPY for the last 2 weeks"
  → trading_market=GMO:BTC_JPY, signal_market=BINANCE:BTC_USDT
  → to_dt = today − 2 days, from_dt = to_dt − 14 days
  → check Trade for BINANCE BTC_USDT and OrderBook for GMO BTC_JPY
  → populate any gaps (skip today and yesterday)
  → set up env and run backtest

User: "backtest orion on HUOBI BTC_SPOT from 2026-02-01 to 2026-02-28"
  → trading_market=HUOBI:BTC_SPOT, signal_market=BINANCE:BTC_USDT
  → check Trade BINANCE BTC_USDT and OrderBook HUOBI BTC_SPOT
  → set up env and run backtest
