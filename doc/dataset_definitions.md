# Dataset Definitions

Formal definitions for every metric, feature, and label in the data warehouse.
This is the semantic layer the LLM query interface relies on.
All column names used in queries must have an entry here.

**Notation**
- `EPS` = 1e-12 (division guard)
- `mid` = `(best_bid + best_ask) / 2`
- `bps` = basis points = × 10000
- Rolling windows are in seconds unless stated otherwise

---

## 1 Bronze Layer

Raw structured data. No derived analytics.
The bronze layer contains only `Trade` and `OrderBook`.

### dataset=Trade

Source: msgpack trade stream
Grain: per trade

| Column | Definition |
|--------|------------|
| `ts` | Unix timestamp (seconds, float) |
| `side` | 0 = BUY, 1 = SELL |
| `price` | Trade price |
| `size` | Trade quantity |

---

### dataset=OrderBook

Source: msgpack order book snapshots
Grain: per order book update

| Column | Definition |
|--------|------------|
| `ts` | Unix timestamp (seconds, float) |
| `bid{i}_price` | Price of i-th bid level (i = 0..19, 0 = best) |
| `bid{i}_qty` | Quantity at i-th bid level |
| `ask{i}_price` | Price of i-th ask level (i = 0..19, 0 = best) |
| `ask{i}_qty` | Quantity at i-th ask level |

---

## 2 Silver Layer

Derived metrics from bronze data.

### dataset=BookChange

Source: OrderBook (diff between consecutive snapshots)
Grain: per order book update

| Column | Definition |
|--------|------------|
| `ts` | Unix timestamp (seconds, float) |
| `delta_sec` | Time elapsed since previous snapshot |
| `best_bid_reset_k{k}` | 1 if best bid changed within top-k levels, else 0 |
| `best_ask_reset_k{k}` | 1 if best ask changed within top-k levels, else 0 |
| `delta_depth_bid_k{k}` | Change in total bid quantity within top-k levels |
| `delta_depth_ask_k{k}` | Change in total ask quantity within top-k levels |
| `delta_depth_bid_rate_k{k}` | `delta_depth_bid_k{k} / delta_sec` |
| `delta_depth_ask_rate_k{k}` | `delta_depth_ask_k{k} / delta_sec` |
| `pull_bid_rate_k{k}` | Rate of bid-side quantity removed per second within top-k |
| `pull_ask_rate_k{k}` | Rate of ask-side quantity removed per second within top-k |
| `refill_bid_rate_k{k}` | Rate of bid-side quantity added per second within top-k |
| `refill_ask_rate_k{k}` | Rate of ask-side quantity added per second within top-k |

---

### dataset=BookState

Source: OrderBook
Grain: per order book update

| Column | Definition |
|--------|------------|
| `ts` | Unix timestamp (seconds, float) |
| `best_bid` | Best bid price |
| `best_ask` | Best ask price |
| `best_mid` | `(best_bid + best_ask) / 2` |
| `best_spread` | `best_ask - best_bid` |
| `best_spread_bps` | `best_spread / best_mid * 10000` |
| `vol_imb_k{k}` | Volume imbalance within top-k levels: `(bid_qty - ask_qty) / (bid_qty + ask_qty + EPS)` |
| `weighted_imb_k{k}` | Price-weighted imbalance within top-k levels |
| `depth_imb_{rate_bps}bps` | Depth imbalance at price distance `rate_bps` bps from mid |
| `microprice_k{k}` | `(bid_qty * ask_price + ask_qty * bid_price) / (bid_qty + ask_qty)` within top-k levels |

---

### dataset=MarketPrice

Source: OrderBook
Grain: per order book update
Partitioned by: `exchange`, `symbol`, `execution_size`

`execution_size` = notional quantity used to walk the book (e.g. 150 = 150 units).

| Column | Definition |
|--------|------------|
| `ts` | Unix timestamp (seconds, float) |
| `bid` | Effective bid price after consuming `execution_size` on the bid side |
| `ask` | Effective ask price after consuming `execution_size` on the ask side |
| `mid` | `(bid + ask) / 2` |
| `spread` | `ask - bid` |
| `spread_bps` | `spread / mid * 10000` |

---

### dataset=TradeAggression

Source: Trade stream
Grain: per trade, with rolling window aggregations attached
Windows: multiple (e.g. 1s, 5s, 10s — parameterised per run)

| Column | Definition |
|--------|------------|
| `ts` | Unix timestamp (seconds, float) |
| `buy_qty_{w}` | Sum of buy trade quantities over window `w` |
| `sell_qty_{w}` | Sum of sell trade quantities over window `w` |
| `buy_count_{w}` | Count of buy trades over window `w` |
| `sell_count_{w}` | Count of sell trades over window `w` |
| `total_qty_{w}` | `buy_qty_{w} + sell_qty_{w}` |
| `total_count_{w}` | `buy_count_{w} + sell_count_{w}` |
| `imbalance_qty_{w}` | `buy_qty_{w} - sell_qty_{w}` |
| `imbalance_count_{w}` | `buy_count_{w} - sell_count_{w}` |
| `qty_per_sec_{w}` | `total_qty_{w} / w` |
| `trades_per_sec_{w}` | `total_count_{w} / w` |
| `buy_qty_per_sec_{w}` | `buy_qty_{w} / w` |
| `sell_qty_per_sec_{w}` | `sell_qty_{w} / w` |
| `avg_trade_size_{w}` | `total_qty_{w} / (total_count_{w} + EPS)` |
| `vwap_{w}` | Volume-weighted average price over window `w` |
| `vwap_buy_{w}` | VWAP of buy trades over window `w` |
| `vwap_sell_{w}` | VWAP of sell trades over window `w` |
| `net_flow_{w}` | `buy_qty_{w} - sell_qty_{w}` (signed volume) |
| `buy_sell_ratio_{w}` | `buy_qty_{w} / (sell_qty_{w} + EPS)` |
| `max_trade_size_{w}` | Largest single trade size in window `w` |
| `burst_flag_{w}` | 1 if qty_per_sec exceeds burst threshold, else 0 |
| `dominant_size_bucket_{w}` | Most frequent trade size bucket in window `w` |
| `dominant_bucket_count_{w}` | Count of trades in dominant bucket |
| `dominant_bucket_share_{w}` | `dominant_bucket_count_{w} / total_count_{w}` |

---

## 3 Gold Layer

Window-based and statistical signals. Building blocks for features.

### dataset=Chart_{bar_type}

Source: Trade stream
Grain: per bar
Bar types: `CANDLE` (time-based), `HEIKIN_ASHI`, `RANGE`
Partitioned by: `exchange`, `symbol`, `param_id` (bar parameter, e.g. period in seconds)

| Column | Definition |
|--------|------------|
| `ts` | Bar close timestamp |
| `open` | First trade price in bar |
| `high` | Highest trade price in bar |
| `low` | Lowest trade price in bar |
| `close` | Last trade price in bar |
| `volume` | Sum of trade quantities in bar |
| `tick_count` | Number of trades in bar |

For `HEIKIN_ASHI`, OHLC values are smoothed:
- `ha_close = (open + high + low + close) / 4`
- `ha_open = (prev_ha_open + prev_ha_close) / 2`
- `ha_high = max(high, ha_open, ha_close)`
- `ha_low = min(low, ha_open, ha_close)`

---

### dataset=LeadLag

Source: MarketPrice (two markets)
Grain: per sliding window step (stride = 1s–30s depending on config)
Partitioned by: `market_a`, `market_b`, `param_id`

Column naming pattern: `ll_{metric}_{window}s_{span}s`
- `{window}` = cross-correlation window in seconds
- `{span}` = EWM decay span in seconds

| Column | Definition |
|--------|------------|
| `ts` | Window end timestamp |
| `ll_corr_at_zero_{w}_{s}` | Cross-correlation between market_a and market_b returns at zero lag, clipped to [-1, 1] |
| `ll_max_corr_{w}_{s}` | Maximum cross-correlation over all searched lags, clipped to [-1, 1] |
| `ll_opt_lag_ms_{w}_{s}` | Lag in milliseconds at which max correlation occurs, normalised to [0, 1] by max_lag_ms |
| `ll_alpha_gain_{w}_{s}` | `abs(max_corr) / (abs(corr_at_zero) + EPS)`, clipped to [1, 100]. > 1 means market_a leads at a non-zero lag |
| `ll_alpha_gain_log_{w}_{s}` | `log(1 + alpha_gain)` |
| `ll_drift_{w}_{s}` | Change in optimal lag from previous window, normalised by max_lag_ms |
| `ll_lag_stability_lb{n}_{w}_{s}` | `log(1 + std(opt_lag[-n:]))` — rolling stability of the optimal lag over last n windows |

---

### dataset=Spread and dataset=Volatility

Written by Hase (`hase derive Spread --params SPAN`, `hase derive Volatility --params H:L:S`;
formulas `hase.derive.spread.measure` and `hase.derive.volatility.measure`). Makalu only
reads them; the windows its curves read are `src/makalu/activity_windows.py`. Design and
formulas: `doc/vol_spread.md`.
Source: MarketPrice at `execution_size`, put on a 1 s grid (last quote in each second,
carried forward); each day read with the 8 hours before it. `execution_size=touch` is the
top of the book instead.
Grain: 1 second
Partitioned by: `exchange`, `symbol`, `execution_size`, `param_id`
(`g1s_ema300s` for Spread; `g1s_h60s_lb300s_ema120s` for Volatility)

**SQL note:** always include `param_id` in the WHERE clause -- without it the query scans every
window set and returns mixed results. Discover available values first:
`SELECT DISTINCT param_id FROM volatility WHERE exchange='...' AND symbol='...' LIMIT 20`

| Dataset | Column | Definition |
|---------|--------|------------|
| both | `ts` | Unix timestamp (seconds, float) |
| Spread | `spread_bps_ema` | `EMA(spread_bps, span)`, `spread_bps = (ask - bid) / mid * 1e4` |
| Volatility | `rv_bps` | `sqrt(sum(log-return^2))` over `lookback`, `* 1e4` |
| Volatility | `vol_bps` | `rolling_std(mid[t] / mid[t - horizon] - 1, lookback) * 1e4` |
| Volatility | `vol_bps_ema` | `EMA(vol_bps, span=smooth)` |

Bid, ask and mid on the grid are not stored: `hase.derive.grid.quote_grid` returns them.
The tradability ratio is `vol_bps_ema / spread_bps_ema`, computed by the reader.

---

### dataset=ActivityCurve

Source: Volatility and Spread (Hase)
Grain: one row per (day_of_week, bucket_index) — a weekly snapshot, not a time series
Partitioned by: `exchange`, `symbol`, `param_id`

Built by `populate ActivityCurve` (see `src/pipeline/curve_steps.py::build_activity_curve_rows`,
core binning math in `src/active_score/weekly_curve.py::aggregate_weekly`). For each window in
`src/makalu/activity_windows.py`, Volatility's `vol_bps_ema` and the Spread over its lookback (`spread_bps_ema`) are resampled to 1-min,
z-score normalized, then all of those normalized series are combined and binned by
`(day_of_week, time_of_day // bin_size_sec)`. Mean is min-max normalized to [0, 1]; std is scaled
by the same factor. `--start`/`--end` is the *training window* (how much trailing Volatility/Spread
history the curve is built from) — the `date=` partition equals `--end`: the as-of date the
window describes, **not** the day the command happened to run. This matters for backfilling:
running the populate command many times in one sitting with different historical `--start`/`--end`
windows produces one correctly-dated snapshot per window, not a collision on a single partition.
Re-running the *same* window later (e.g. to fix a bug) targets the same `date=` partition as
before — skip-existing/`--recreate` semantics apply meaningfully to "redo this specific
snapshot," not "create an unrelated new one." A consumer wanting the current curve should read
the most recent `date=` partition for a given `param_id`.

`param_id` format: `bin{bin_size_sec}s_z{n_training_days}d_es{execution_size}`, e.g. `bin300s_z19d_es0.002`.

| Column | Definition |
|--------|------------|
| `ts` | Timestamp of the populate run that produced this snapshot (all rows in one snapshot share the same date) |
| `day_of_week` | 0=Monday .. 6=Sunday |
| `bucket_index` | `0 .. (86400/bin_size_sec - 1)`, time-of-day bucket |
| `score_mean` | Mean of the combined normalized vol+spread series in this bin, min-max normalized to [0, 1] over the whole week |
| `score_std` | Std of the combined normalized series in this bin, scaled by the same min-max factor as `score_mean` |

---

### dataset=LeadLagCurve

Source: LeadLag
Grain: one row per (day_of_week, bucket_index) — a weekly snapshot, not a time series
Partitioned by: `market_a`, `market_b`, `param_id`

Same weekly-curve construction as ActivityCurve (`src/pipeline/curve_steps.py::build_leadlag_curve_rows`),
but built from LeadLag's `ll_max_corr_*` columns (absolute value — sign/direction isn't relevant to
"is this a good time to trade the relationship") instead of Volatility/Spread. Whichever `ll_max_corr_*`
suffixes are actually present in the populated LeadLag param_id(s) are discovered from the data and
all combined, rather than hardcoded.

`param_id` format: `bin{bin_size_sec}s_z{n_training_days}d`, e.g. `bin300s_z19d`.

| Column | Definition |
|--------|------------|
| `ts` | Timestamp of the populate run that produced this snapshot |
| `day_of_week` | 0=Monday .. 6=Sunday |
| `bucket_index` | `0 .. (86400/bin_size_sec - 1)`, time-of-day bucket |
| `corr_mean` | Mean of the combined normalized \|ll_max_corr\| series in this bin, min-max normalized to [0, 1] |
| `corr_std` | Std of the combined normalized series in this bin, scaled by the same min-max factor as `corr_mean` |

---

### dataset=Chart_RANGE

Source: Trade
Grain: per bar (variable time -- a new bar closes whenever price moves by `box_size_rate` from the bar's open, not on a fixed interval)
Partitioned by: `exchange`, `symbol`, `param_id`

Built by `pipeline.trade_derived_steps.PopulateRangeBarStep` / `qate.trading.chart.RangeBarBuilder`, the same
range-bar builder `orion`'s `v6`/`v601` live strategy variants use. `param_id` format: `rate_<value>`, e.g.
`--box-size-rate 0.0015` → `rate_150` (`value = box_size_rate * 100_000`).

| Column | Definition |
|--------|------------|
| `ts` | Bar close timestamp (Unix seconds, float) |
| `open` | First trade price in the bar |
| `high` | Highest trade price in the bar |
| `low` | Lowest trade price in the bar |
| `close` | Last trade price in the bar (the price that triggered the bar to close) |
| `volume` | Sum of trade sizes in the bar |
| `tick_count` | Number of trades in the bar |

Sibling datasets `Chart_CANDLE` (fixed time interval) and `Chart_HEIKIN_ASHI` (smoothed candle) share this same
column shape, built by `PopulateCandleBarStep`/`PopulateHeikinAshiBarStep` -- not yet wired into the `populate`
CLI (only `Chart_RANGE` is, as of this writing).

---

## 4 Feature Layer

ML-ready features. All columns are normalised and dimensionless unless stated.
Grain: tick-level (aligned to order book updates).

### dataset=Feature_Mid / dataset=Feature_NearHFT

Source: BookState, BookChange, TradeAggression, MarketPrice, LeadLag
Partitioned by: `exchange`, `symbol`

#### Trade Aggression Features (`ta_`)

| Column | Formula |
|--------|---------|
| `ta_vol_imbalance_ratio_{w}` | `(buy_qty - sell_qty) / (buy_qty + sell_qty + EPS)` |
| `ta_count_imbalance_ratio_{w}` | `(buy_count - sell_count) / (buy_count + sell_count + EPS)` |
| `ta_vwap_buy_sell_spread_bps_{w}` | `(vwap_buy - vwap_sell) / (vwap + EPS) * 10000` |
| `ta_size_asymmetry_norm_{w}` | `(avg_buy_size - avg_sell_size) / (avg_buy_size + avg_sell_size + EPS)` |
| `ta_buy_qty_share_{w}` | `buy_qty / (total_qty + EPS)` |
| `ta_buy_count_share_{w}` | `buy_count / (total_count + EPS)` |
| `ta_qty_count_divergence_{w}` | `ta_buy_qty_share_{w} - ta_buy_count_share_{w}` |
| `ta_avg_size_zscore_{w}_{rw}` | Z-score of `avg_trade_size_{w}` over rolling window `rw` |
| `ta_avg_size_pct_{w}_{rw}` | Percentile rank of `avg_trade_size_{w}` over rolling window `rw` |
| `ta_qty_per_sec_zscore_{w}_{rw}` | Z-score of `qty_per_sec_{w}` over rolling window `rw` |
| `ta_qty_per_sec_pct_{w}_{rw}` | Percentile rank of `qty_per_sec_{w}` over rolling window `rw` |
| `ta_trades_per_sec_zscore_{w}_{rw}` | Z-score of `trades_per_sec_{w}` over rolling window `rw` |
| `ta_trades_per_sec_pct_{w}_{rw}` | Percentile rank of `trades_per_sec_{w}` over rolling window `rw` |
| `ta_imbalance_momentum_{w1}_{w2}` | `ta_vol_imbalance_ratio_{w1} - ta_vol_imbalance_ratio_{w2}` (short minus longer window) |
| `ta_imbalance_sign_agree_{w1}_{w2}` | `1` if `sign(ratio_{w1}) == sign(ratio_{w2})`, else `0` |
| `ta_imbalance_sign_agree_all` | `1` if all three windows agree on sign |
| `ta_imbalance_strengthening_{w1}_{w2}` | `1` if `abs(ratio_{w1}) > abs(ratio_{w2})` (short window stronger) |
| `ta_strong_buy_signal` | `(ratio_w1 > 0.3) & (all windows agree) & (short window strengthening)` |
| `ta_strong_sell_signal` | `(ratio_w1 < -0.3) & (all windows agree) & (short window strengthening)` |
| `ta_exhaustion_signal` | `(abs(ratio_w3) > 0.3) & (short window weakening) & (signs agree w1, w3)` |
| `ta_reversal_signal` | `(abs(ratio_w3) > 0.3) & (sign(w1) != sign(w3))` |

#### Order Book State Features (`bs_`)

| Column | Formula |
|--------|---------|
| `bs_microprice_offset_bps_k{k}` | `(microprice_k{k} - best_mid) / (best_mid + EPS) * 10000` |
| `bs_imbalance_gradient` | `vol_imb_k{shallow} - vol_imb_k{mid}` |
| `bs_imbalance_divergence` | `abs(vol_imb_k{shallow} - vol_imb_k{deep})` |
| `bs_microprice_offset_norm` | `bs_microprice_offset_bps_k1 / (best_spread_bps + EPS)` |
| `bs_spread_bps_mean_{rw}` | Rolling mean of `best_spread_bps` over `rw` |
| `bs_spread_bps_std_{rw}` | Rolling std of `best_spread_bps` over `rw` |
| `bs_vol_imb_k{k}_mean_{rw}` | Rolling mean of `vol_imb_k{k}` over `rw` |
| `bs_vol_imb_k{k}_std_{rw}` | Rolling std of `vol_imb_k{k}` over `rw` |

#### Book Change Features (`bc_`)

| Column | Formula |
|--------|---------|
| `bc_delta_depth_imbalance_norm_k{k}` | `(delta_depth_bid - delta_depth_ask) / (abs(delta_depth_bid) + abs(delta_depth_ask) + EPS)` |
| `bc_pull_imbalance_k{k}` | `(pull_bid_rate - pull_ask_rate) / (pull_bid_rate + pull_ask_rate + EPS)` |
| `bc_refill_imbalance_k{k}` | `(refill_bid_rate - refill_ask_rate) / (refill_bid_rate + refill_ask_rate + EPS)` |
| `bc_pull_refill_ratio_k{k}` | `(pull_bid_rate + pull_ask_rate) / (refill_bid_rate + refill_ask_rate + EPS)` |
| `bc_total_pull_zscore_k{k}_{rw}` | Z-score of `pull_bid_rate + pull_ask_rate` over rolling window `rw` |
| `bc_total_activity_zscore_k{k}_{rw}` | Z-score of `pull_bid_rate + pull_ask_rate + refill_bid_rate + refill_ask_rate` |
| `bc_total_activity_pct_k{k}_{rw}` | Percentile rank of total book activity over `rw` |
| `bc_pull_imbalance_gradient` | `bc_pull_imbalance_k{shallow} - bc_pull_imbalance_k{mid}` |
| `bc_refill_imbalance_gradient` | `bc_refill_imbalance_k{shallow} - bc_refill_imbalance_k{mid}` |

#### Execution Price Features (`ep_`)

| Column | Formula |
|--------|---------|
| `ep_slippage_bid_bps` | `(best_bid - exec_bid) / (best_bid + EPS) * 10000` |
| `ep_slippage_ask_bps` | `(exec_ask - best_ask) / (best_ask + EPS) * 10000` |
| `ep_slippage_total_bps` | `ep_slippage_bid_bps + ep_slippage_ask_bps` |
| `ep_slippage_zscore_{rw}` | Z-score of `ep_slippage_total_bps` over `rw` |
| `ep_slippage_pct_{rw}` | Percentile rank of `ep_slippage_total_bps` over `rw` |
| `ep_slippage_asymmetry` | `(ep_slippage_ask_bps - ep_slippage_bid_bps) / (ep_slippage_total_bps + EPS)` |
| `ep_depth_quality` | `1 / (ep_slippage_total_bps + 1)` — higher = cheaper to execute |
| `ep_exec_spread_bps` | `(exec_ask - exec_bid) / mid * 10000` — effective spread at execution size |
| `ep_spread_expansion_bps` | `ep_exec_spread_bps - best_spread_bps` |
| `ep_spread_expansion_ratio` | `ep_exec_spread_bps / (best_spread_bps + EPS)` |

#### Trend Features (`tr_`)

| Column | Formula |
|--------|---------|
| `tr_roc_{w}` | `mid / mid.shift(lookback) - 1` — rate of change over window `w` |
| `tr_ema_spread_{fast}_{slow}` | `EWM(mid, span=fast) / EWM(mid, span=slow) - 1` |

#### Volatility Features (`vo_`)

| Column | Formula |
|--------|---------|
| `vo_realized_vol_{w}` | `mid.pct_change().rolling(w).std()` |
| `vo_relative_spread_{w}` | `best_spread_bps / (vo_realized_vol_{w} * 10000 + EPS)` |

#### Cross-Feature Interactions (`cf_`)

| Column | Formula |
|--------|---------|
| `cf_vwap_offset_bps_{w}` | `(vwap_{w} - mid) / mid * 10000` |
| `cf_sign_agree_trade_book` | `1` if `sign(ta_vol_imbalance_ratio_w1) == sign(bs_vol_imb_k1)` |
| `cf_strong_agreement` | `(abs(ta_ratio_w1) > 0.3) & (abs(bs_vol_imb_k1) > 0.3) & cf_sign_agree_trade_book` |
| `cf_trade_consuming_asks` | `(ta_ratio_w1 > 0.3) & (bc_pull_imbalance_k1 < -0.3)` — buys consuming ask liquidity |
| `cf_trade_consuming_bids` | `(ta_ratio_w1 < -0.3) & (bc_pull_imbalance_k1 > 0.3)` — sells consuming bid liquidity |
| `cf_consumption_alignment` | `ta_vol_imbalance_ratio_w1 * (-bc_pull_imbalance_k1)` |
| `cf_absorption_bid` | `(ta_ratio_w1 < -0.3) & (bc_delta_depth_imbalance_norm_k1 >= 0)` — sells absorbed at bid |
| `cf_absorption_ask` | `(ta_ratio_w1 > 0.3) & (bc_delta_depth_imbalance_norm_k1 <= 0)` — buys absorbed at ask |
| `cf_absorption_strength` | `ta_vol_imbalance_ratio_w1 * (-bc_delta_depth_imbalance_norm_k1)` |
| `cf_stable_wall_bid` | `(bs_vol_imb_k1 > 0.5) & (bc_best_bid_reset == 0)` |
| `cf_stable_wall_ask` | `(bs_vol_imb_k1 < -0.5) & (bc_best_ask_reset == 0)` |
| `cf_depth_quality_per_spread` | `ep_depth_quality / (best_spread_bps + EPS)` |
| `cf_confirmation_score` | Mean of 4 sign agreement signals, range [-1, 1] |
| `cf_signal_agreement` | `abs(cf_confirmation_score)` |
| `cf_quality_weighted_signal` | `cf_confirmation_score * cf_signal_agreement * ep_depth_quality * (1 / (spread_bps + 1))` |
| `cf_regime_trending` | Multi-factor: strong imbalance + momentum alignment + low spread |
| `cf_regime_mean_reverting` | Multi-factor: high spread + absorption signals |
| `cf_regime_volatile` | Multi-factor: high vol + large spread expansion |

---

## 5 Research Layer

Experiment outputs. Not canonical — may change between runs.

### dataset=TripleBarrierLabel / dataset=TripleBarrierLabel_tmp

Source: MarketPrice
Grain: per entry point (sampled from execution price series)
Partitioned by: `exchange`, `symbol`, `param_id`

Barriers are dynamic, sized as multiples of `volatility_ema`:
- Upper barrier: `entry_mid * (1 + barrier_multiplier * volatility_ema)`
- Lower barrier: `entry_mid * (1 - barrier_multiplier * volatility_ema)`
- Timeout: `timeout_sec`

| Column | Definition |
|--------|------------|
| `ts` | Entry timestamp |
| `label` | `1` = upper barrier hit first (win), `-1` = lower barrier hit first (loss), `0` = timeout |
| `pnl` | `bid_exit - ask_entry` for long; `bid_entry - ask_exit` for short |
| `volatility_ema` | `EMA(mid.pct_change().rolling(w).std(), span=s)` at entry time |
| `spread_pct_ema` | `EMA(spread / mid, span=s)` at entry time |

---

### dataset=BuyLabel_tmp

Source: MarketPrice
Grain: per tick
Partitioned by: `exchange`, `symbol`, `param_id`

Barriers are spread-relative:
- Profit-take: `entry_mid * (1 + pt_multiplier * spread_pct_ema)`
- Stop-loss: `entry_mid * (1 - sl_multiplier * spread_pct_ema)`
- Timeout: `timeout_sec`

| Column | Definition |
|--------|------------|
| `ts` | Entry timestamp |
| `label` | `1` = profit-take hit first (buy signal valid), `0` = stop-loss or timeout |
| `buy_pnl` | `bid_exit - ask_entry` |
| `spread_pct_ema` | `EMA(spread / mid, span=s)` at entry time |

---

### dataset=BacktestPnL

Source: Himalaya backtest engine
Grain: per trade fill
Partitioned by: `strategy`, `exchange`, `symbol`, `variant`, `param_set_id`

| Column | Definition |
|--------|------------|
| `ts` | Fill timestamp |
| `side` | BUY or SELL |
| `price` | Fill price |
| `size` | Fill quantity |
| `pnl` | Realised PnL on close |
| `fee` | Exchange fee paid |
| `settle_type` | OPEN or CLOSE |
