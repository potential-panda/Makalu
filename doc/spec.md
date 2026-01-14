# Microstructure Feature Stack + ML Permission Model (Conversation Summary)

Context: scalping-focused trading system with realistic backtest execution.
Goal: use microstructure features (trades + order book snapshots + order book diffs) mainly as **permission / sizing / execution modulation**, optionally trained via LightGBM to decide **take/skip** (and/or size) for candidate trades produced by a simpler entry/exit signal.

---

## 0) System framing (2-layer decision)

### Layer A — Entry/Exit Signal (direction + candidate)
- Produces “candidate opportunities”: side, timestamp, reference price/level.
- Should be relatively simple and stable (SAR/range/structure/etc.).
- Often weak/near-zero edge after friction if used alone.

### Layer B — Permission / Modulation (microstructure)
- Uses trade aggression + book state + book change features to decide:
  1) trade now or stand down
  2) increase/decrease qty
  3) market vs limit aggressiveness
  4) accelerate exits if flow decays/flips
- Even if not “signal generation”, it **still affects final order timing** because every order passes through this layer.

Key principle:
> Alpha = Price idea × Microstructure permission × Execution quality

---

## 1) Class: `TradeAggression` + `TradeAggressionBuilder` (from realtime trades)

### What it provides
Aggregates taker-initiated trades into rolling-window metrics:
- volume and count by side
- imbalance (qty/count)
- activity rates (qty/sec, trades/sec)
- VWAP overall and per-side
- size clustering / dominant bucket
- burst proxy and max trade size

### Why it’s useful for scalping
Best used as **state / permission / exit timing**, not standalone entry:
- Filters bad trades (avoid fading strong persistent aggression; avoid breakout in churn)
- Sizes up/down depending on aligned persistent flow
- Exits earlier when flow flips/decays

### Bug fixes / improvements
1) **`windows_sec` default bug**
   - `windows_sec=None` currently breaks; set default:
   - `self.windows_sec = windows_sec or [1, 3, 5]`

2) **Type clarity**
   - Use consistent types: `list[float]` or `list[int]`. Avoid `dict[float, ...]` if windows are ints.

3) **Out-of-order trade handling**
   - Dropping out-of-order trades biases metrics.
   - Minimum fix: count and export `dropped_out_of_order`.
   - Better: small reorder buffer (e.g., accept disorder within X ms).

4) **Bucketing stability**
   - `round(size / bucket_unit)` is float-sensitive.
   - Prefer deterministic bucketing:
     - floor-style: `int(size / bucket_unit + 1e-12)`
     - or nearest: `int(size / bucket_unit + 0.5)` (still guard float error)
   - Consider storing bucket index per trade at insert time to avoid recompute on eviction.

5) **Max trade size recompute**
   - Occasional `max()` scan is ok for small windows.
   - If you add longer windows, consider a max-queue structure.

6) **Serialization**
   - `buy_sell_ratio` may be `inf`. For storage (Influx/JSON), map `inf -> None` or large sentinel.

### Window sizes for scalping (practical defaults)
For Binance-like conditions:
- Mandatory: **0.5s–1s**
- Confirmation: **2–5s**
- Optional regime: **8–15s**
Suggested sets:
- minimal: `[1s, 5s]`
- strong: `[0.5s, 3s, 10s]` (Binance 10Hz makes 0.5s especially useful)

Rule of thumb:
> Window ≥ 5–10× decision-to-fill latency

---

## 2) Class: Order book snapshot metrics (functions on `OrderBook`)

Functions provided:
- `compute_volume_imbalance(k)`
- `compute_weighted_imbalance(k)`
- `compute_depth_imbalance(distance_rate)`
- `compute_microprice(k)`
- `compute_depth_k(k)`

### Usefulness
These are core **book-state** features. Best performers for scalping are usually:
- touch imbalance (k=1)
- small-k imbalance (k=3/5)
- microprice and microprice-mid offset (bps)
- spread in bps
- depth imbalance within small price bands (ticks/bps)

### Key improvements / pitfalls
1) **Normalize distance in ticks or bps (critical)**
   - Your `weighted_imbalance` uses absolute price distance: `1/(1+distance)`; this is not scale-invariant.
   - Fix: define distance in **ticks** or **bps**:
     - `distance_ticks = abs(price-mid)/tick_size`
     - or `distance_bps = abs(price-mid)/mid * 1e4`
   - Weighting options:
     - `w = 1/(1 + distance_ticks)`
     - or `w = exp(-alpha * distance_ticks)`

2) **`compute_depth_imbalance(distance_rate=0.1)` not scalping-relevant**
   - 10% of mid is far too wide.
   - Replace with:
     - `distance_ticks: int` (e.g., 5/10/20)
     - or `distance_bps: float` (e.g., 1/2/5)

3) **Microprice**
   - Classic microprice is typically **k=1**.
   - If using k>1 volumes, consider consistent generalized approach (otherwise keep k=1).

4) **Depth raw values need ratios**
   - Use normalized forms:
     - `depth_imb = (bid_depth-ask_depth)/(bid_depth+ask_depth)`
     - `log_depth_diff = log(bid_depth+eps)-log(ask_depth+eps)`

### Binance settings provided by user
- Orderbook snapshots: **100ms (10Hz)**
- Depth: **top 30 levels**

Recommended per-snapshot feature bands for Binance:
- k imbalance: **1, 3, 5**
- depth-band imbalance: within **1, 2, 5 bps** OR **5, 10, 20 ticks**
- compute rolling/EMA aggregates over **0.5s, 1s, 3s** (5/10/30 snapshots)

---

## 3) Class: `BookChange` + `BookChangeBuilder` (diff between snapshots)

### What it provides
Price-aligned diff between consecutive snapshots (top-k) and normalizes by Δt:
- best bid/ask reset flags
- delta depth bid/ask and per-second rates
- pull/refill proxies

### Why it’s useful for scalping
Captures **liquidity behavior**:
- quote pulling vs stacking
- touch movement intensity
- churn regimes (often “don’t trade”)

Pairs strongly with `TradeAggression`:
- trades = what executed
- book change = what liquidity is doing

### Key issues / improvements
1) **Top-k by index causes composition artifacts**
   - When best price moves, the set of top-k prices shifts, so net delta can reflect composition change rather than true add/remove.
   - Fix: compute diffs in a **price band** anchored to best/mid:
     - within N ticks or X bps
   - Keep best-reset features separately.

2) **Use int tick keys instead of float prices**
   - Normalize to `price_tick = round(price / tick_size)` and diff on int keys.

3) **Net delta loses signal; add gross pull/refill**
   - Current pull/refill uses net:
     - `pull = max(-net_rate, 0)`
     - `refill = max(net_rate, 0)`
   - Add gross:
     - `pull_gross_rate = sum(-d for d<0)/dt`
     - `refill_gross_rate = sum(d for d>0)/dt`

4) **Best reset should include magnitude**
   - Add:
     - `best_bid_move_ticks`, `best_ask_move_ticks`

5) **Validity / discontinuities**
   - Consider not nuking state aggressively on transient invalid/cross conditions.
   - Use `max_delta_sec` to detect gaps (keep).

6) **Perf**
   - At 10Hz per symbol, dict/set approach is fine. For scale, consider merge-diff on sorted tick arrays.

### Recommended scalping bands for Binance
Compute BookChange rates in:
- touch band: **0–5 ticks**
- near band: **0–20 ticks** (or 1–5 bps)

Per band, per side:
- net change rate
- pull gross rate
- refill gross rate

---

# 4) Final list: useful ML features (with meaning)

Below is a recommended feature set grouped by source. These are intended for **permission / sizing / execution choice**.

## A) TradeAggression features (per window: 1s/3s/5s as available)
1) `ta_imbalance_qty[w]`
   - (buy_qty - sell_qty) / total_qty
   - captures directional taker pressure

2) `ta_imbalance_count[w]`
   - (buy_count - sell_count) / total_count
   - distinguishes many small trades vs few large trades (with qty metrics)

3) `ta_qty_per_sec[w]`, `ta_trades_per_sec[w]`
   - activity regime; filters dead/churn states

4) `ta_buy_qty_per_sec[w]`, `ta_sell_qty_per_sec[w]`
   - side-specific intensity

5) `ta_vwap_offset_bps[w]`
   - `(vwap - mid)/mid*1e4` at decision time
   - measures whether takers are paying through

6) `ta_dominant_bucket_share[w]` (+ `dominant_size_bucket` optional)
   - size clustering; potential “participant type” indicator

7) `ta_max_trade_size[w]`
   - large actor presence / sweeps

8) Cross-window persistence features
   - `ta_imb_qty[1s]-ta_imb_qty[3s]` (impulse vs persistence)
   - sign agreement across windows (binary)

## B) Order book state features (per snapshot, plus rolling/EMA over 1s/3s/5s)
1) `spread_bps`
   - `(ask1-bid1)/mid*1e4`
   - direct friction + regime indicator

2) `ob_imb_k1`, `ob_imb_k3`, `ob_imb_k5`
   - top-k volume imbalance near touch

3) `microprice_offset_bps`
   - `(microprice - mid)/mid*1e4`
   - strongest single “book pressure” feature

4) Depth-band imbalances (ticks or bps)
   - `ob_imb_1bps`, `ob_imb_2bps`, `ob_imb_5bps`
   - or `ob_imb_5t`, `ob_imb_10t`, `ob_imb_20t`

5) Normalized depths
   - `depth_k_bid`, `depth_k_ask` (raw optional)
   - `depth_imb_k = (bid-ask)/(bid+ask)`
   - `log_depth_diff_k`

6) Volatility/noise proxy from book pressure
   - rolling std of `microprice_offset_bps` over 1s (book instability)

## C) BookChange (diff) features (per band: touch/near; rates are per-sec)
Use tick/bps bands, not top-k index. For each band and side:
1) `bc_net_depth_change_rate_bid`, `bc_net_depth_change_rate_ask`
   - net liquidity change; stacking vs retreat

2) `bc_pull_gross_rate_bid`, `bc_refill_gross_rate_bid`
3) `bc_pull_gross_rate_ask`, `bc_refill_gross_rate_ask`
   - churn vs true pull/refill; more informative than net

4) `best_bid_move_ticks`, `best_ask_move_ticks`
   - touch movement intensity

5) Optional “where did change occur”
   - share of gross change at touch vs deeper within band

## D) Interaction features (high value)
1) `sign_agreement_trade_vs_book[w]`
   - e.g., `sign(ta_imbalance_qty[w]) == sign(microprice_offset_bps)`
   - disagreement can imply spoof/absorption/trap regimes

2) “Strong book but weak trades” flags
   - large book imbalance + low `ta_trades_per_sec` → spoof risk

3) Spread-conditioned strength
   - flow/book signals weighted by spread regime (avoid trading when spread is too wide)

---

# 5) Labels: what to predict (candidates)

Assumption: backtest is realistic (fills at bid/ask, fees, slippage, latency modeled).

Train ML **only on candidate events** produced by your signal (Layer A), not all timestamps.

## A) Binary classification labels (recommended start)
1) **Profitability**
   - `y = 1 if pnl_net_bps >= +X else 0`
   - choose X to exceed friction (e.g., +1 to +3 bps net)

2) **Avoid-loss label**
   - `y = 1 if pnl_net_bps > -X else 0`
   - useful when you want a “loss avoidance / filter” model

## B) Regression labels
1) **Net pnl in bps**
   - `y = clip(pnl_net_bps, -Y, +Y)` (winsorize for stability)

2) **Expected value proxy**
   - use realized pnl but consider robust transforms (Huber loss or quantile regression)

## C) Multi-class labels (often good for gating + sizing)
1) `LOSS / FLAT / WIN`
   - thresholds: `< -X`, `[-X, +X]`, `> +X`

## D) Survival/time-based (optional)
1) **Time-to-target before stop**
   - e.g., hit TP within T seconds?
2) **Max adverse excursion (MAE)**
   - label whether MAE exceeds threshold (risk gating)

## How to compute labels (practical)
For each candidate entry at time `t0`:
- choose horizon `H` (e.g., 3s/5s/10s/15s/30s depending on your holding time)
- simulate using your execution model:
  - entry at ask (for buy) / bid (for sell)
  - exit at bid/ask at `t0+H` OR using your strategy exit logic (choose one and be consistent)
  - include fees/slippage/latency
- compute `pnl_net_bps`

Important: label must match how you actually trade (market vs limit, exit logic).

---

# 6) ML Pipeline Spec (implementation guide for agent)

## 6.1 Objectives
- Build a model that improves trading by:
  - filtering low-quality candidates (take/skip)
  - optionally producing a size multiplier / confidence
- Primary target: LightGBM (classifier first).

## 6.2 Data definitions

### Inputs
1) **OrderBook snapshots** (Binance 10Hz top-30)
2) **Trades stream** (for TradeAggression)
3) **Candidate events** from Signal layer A
   - event structure:
     - `event_id`
     - `ts` (decision time)
     - `side` (BUY/SELL)
     - `signal_type` (optional)
     - `ref_price` (optional)
     - `metadata` (optional)

### Feature row (one row per candidate)
At candidate time `ts`:
- compute latest available features up to `ts` (no future data)
- store:
  - all TradeAggression features (per window)
  - all OrderBook state features (instant + rolling)
  - all BookChange features (latest + rolling optional)
  - interaction features

### Label row (one row per candidate)
- computed from backtest simulation:
  - `pnl_net_bps`
  - optionally `mae_bps`, `mfe_bps`, `hit_tp`, `hit_sl`
  - derived class labels as needed

## 6.3 Leakage rules (must enforce)
- Features at time `ts` must only use:
  - trades with `trade_ts <= ts`
  - snapshots with `snap_ts <= ts`
  - diffs computed only between snapshots up to ts
- Rolling features:
  - use rolling windows that end at `ts` (not centered)
- If candidates are generated between snapshot ticks, use the **latest prior snapshot**.

## 6.4 Feature computation components (recommended modules)

### A) Feature builders
1) `TradeAggressionBuilder(windows=[0.5,1,3,5], bucket_unit=..., burst_threshold=...)`
2) `OrderBookFeatureBuilder` (new)
   - computes:
     - spread_bps, mid
     - imbalance k=1/3/5
     - microprice_offset_bps
     - depth-band imbalances (tick/bps)
     - rolling EMA/mean/std over 0.5s/1s/3s

3) `BookChangeBuilder` (revised to band-based)
   - touch band and near band
   - gross pull/refill + net
   - best move ticks

### B) Feature store / dataset writer
- `DatasetWriter` that writes one row per candidate with:
  - `event_id`, `ts`, `side`
  - feature columns
  - (later) label columns

Output format:
- Parquet recommended (fast, typed)
- Alternatively CSV for initial debugging

## 6.5 Label computation module
- `Labeler` that takes candidate events and runs your backtest execution to produce:
  - pnl_net_bps at horizons H (can store multiple horizons)
- Recommended:
  - start with one horizon aligned to typical holding (e.g., 10s or 15s)
  - later add multiple horizons and either:
    - train separate models per horizon, or
    - train a model to predict best horizon outcome

## 6.6 Train / validation split (walk-forward)
Use time-based splits to reflect non-stationarity:

Example:
- Train: days 1–10
- Validate: days 11–12
- Test: days 13–14
Then roll forward.

No random shuffle.

## 6.7 Model training (LightGBM)
### Baseline model: binary classifier
- target: `y = 1 if pnl_net_bps >= +X else 0`
- class imbalance handling:
  - `scale_pos_weight` or sample weights

### Evaluation metrics (trading-relevant)
In addition to AUC:
- precision/recall at top-k (or above threshold)
- confusion matrix by regime (spread/activity bins)
- simulated PnL using gating:
  - apply model decision to candidate stream
  - compare vs baseline (take all candidates)
  - report:
    - net pnl, sharpe, max drawdown, win-rate, trades/day

### Output artifacts
- model file
- feature importance
- threshold sweep results (p(win) threshold vs PnL/drawdown)

## 6.8 Decision policy integration (live/backtest)
Given candidate event from Layer A:
1) compute features at `ts`
2) model returns score `p`
3) apply policy:
   - gate: trade if `p > p0`
   - sizing: `qty = base_qty * clamp((p - p0)/(1-p0), 0, 1)` (example)
   - execution: if `p` high and spread low, allow aggressive; else passive/probe

## 6.9 Monitoring / debugging outputs
- store per-candidate:
  - features
  - model score
  - decision (take/skip, qty_mult)
  - realized pnl label
- dashboards:
  - score distribution over time
  - PnL vs score bucket
  - feature drift checks:
    - mean/std of key features by day

---

# 7) Suggested implementation order (one-by-one)
1) Fix TradeAggressionBuilder defaults + bucket stability + out-of-order counter
2) Replace book “distance” metrics with tick/bps normalized versions
3) Revise BookChangeBuilder to band-based + gross pull/refill + best move ticks
4) Build candidate dataset writer (features only) and validate no leakage
5) Implement labeler using your backtest execution
6) Train LightGBM take/skip model + walk-forward evaluation with gating PnL
7) Add sizing policy and re-evaluate
