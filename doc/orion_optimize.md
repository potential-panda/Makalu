# Dynamic Parameter Switching for Range-Bar + Parabolic SAR
**Phases 1–4 Implementation Spec (for code agent)**

## Goal
Keep the existing strategy logic (entries/exits driven by Parabolic SAR flips). Replace static parameters with a learned controller that selects one of 8 parameter combinations in realtime based on market-state features.

Parameters:
- `range_threshold_pct ∈ {0.0015, 0.0020, 0.0025, 0.0030}`
- `sar_step ∈ {0.2, 0.4}`
- Total combos: 8

Controller outputs: chosen combo id `k ∈ [0..7]` at each decision time.

---

## Definitions

### Decision Time
A timestamp `t` where a controller decision is allowed.
**Default**: decision happens only at **range-bar close** timestamps.

### Horizon
A forward-looking evaluation window used to create training labels:
- `H` (duration): start with `H = 1h`, also test `30m` and `2h`.

### Score
Per combo `k`, at time `t`:
- `pnl_k(t,H)`: realized pnl from `t` to `t+H` if combo `k` were used.
- `dd_k(t,H)`: maximum drawdown within `[t, t+H]` of that pnl path.

Score function (start simple):
- **V0**: `score_k = pnl_k`
- **V1**: `score_k = pnl_k - λ * dd_k`
  - start with `λ = 1.0` (tune later)

---

## Phase 1 — Build 8 Variants and Produce Comparable Time Series

### 1.1 Parameter Grid
Create a stable mapping from `(range_threshold_pct, sar_step)` to `combo_id`.

Example ordering:
- `range_threshold_pct` outer loop (ascending)
- `sar_step` inner loop (ascending)
Result: 4 * 2 = 8 combos.

### 1.2 Backtest Outputs (per combo)
Run backtests for each `combo_id` over the same historical period(s) and produce aligned artifacts.

Required outputs per `combo_id`:
- `events`: range-bar close timestamps (or unified decision timestamps)
- `pnl_increment`: pnl delta per decision interval (or per bar)
- `equity_curve`: cumulative pnl (starting at 0)
- `drawdown_curve`: drawdown from peak equity
- `position_state` (optional but useful): position size/sign, exposure time
- `turnover` (optional): number of trades or abs position change per interval

**Critical requirement**:
All combos must share the same time index at decision times.
If strategy naturally produces different bar-close times per threshold, choose one:
- **Preferred**: define decision times on a common clock (e.g., every 10s / 60s) and interpolate/align realized pnl changes; OR
- **If staying bar-close**: define decision times by a chosen “reference bar stream” and align other combos by nearest prior time (must be consistent and documented).

Deliverable:
- `perf_by_combo[combo_id]` as a dataframe indexed by `t` with columns:
  - `pnl_inc`, `equity`, `dd`, optionally `turnover`, `exposure`

### 1.3 Sanity Checks
- All 8 combos positive pnl over 1–2 month spans (expected).
- Equity and drawdown curves look consistent with prior observations.
- Ensure fee/slippage assumptions are identical across combos.

---

## Phase 2 — Build Market-State Feature Pipeline

### 2.1 Feature Timestamps
Compute features at each decision time `t` using only information available up to `t` (no leakage).

Output:
- `X[t]` feature vector as a dataframe indexed by decision time.

### 2.2 Feature Set (Initial, Small, Online-Friendly)
Start with 10–30 features, grouped:

**Volatility**
- `rv_5m`, `rv_30m`, `rv_2h` (realized vol of mid or last price)
- `range_5m` (high-low over window / mid)

**Spread**
- `spread_bps_mean_5m`, `spread_bps_p95_5m`
- `spread_bps_mean_30m`

**Lead-Lag (if available)**
- `leadlag_corr_peak` (max corr over lag grid)
- `leadlag_best_lag_ms` (argmax lag)
- `leadlag_strength_5m` (peak corr value)

**Microstructure / Liquidity (optional early)**
- `imbalance_mean_5m` (top-N depth imbalance)
- `trade_aggr_ratio_5m` (aggressive buy vs sell ratio)
- `depth_bid_ask_ratio_5m`

**Trend / Persistence**
- `return_sign_autocorr_5m`
- `trend_duration` (time since last SAR direction flip; or bar trend count)

### 2.3 Normalization
Tree models do not require scaling, but:
- handle NaNs (forward-fill where appropriate; otherwise drop)
- cap extreme outliers (winsorize) if needed

Deliverable:
- `features_df`: dataframe indexed by `t`, columns = feature names.

---

## Phase 3 — Construct Supervised Learning Targets (Scores)

### 3.1 Compute Forward Returns per Combo
For each decision time `t` and each combo `k`:
- compute `pnl_k(t,H)` as sum of `pnl_inc` from `t` (inclusive) to `t+H` (exclusive)

Also compute path drawdown within the horizon window:
- build the local equity path `eq_path = cumsum(pnl_inc)` starting at 0
- `dd_k(t,H) = max(peak - eq_path)` over the window

Then score:
- `score_k(t) = pnl_k(t,H)` (V0)
- later upgrade to `score_k(t) = pnl_k(t,H) - λ*dd_k(t,H)` (V1)

### 3.2 Dataset Shape
For each `t`, you will have:
- Inputs: `X(t)` (row from `features_df`)
- Targets: `[score_0(t), score_1(t), ..., score_7(t)]`

Create one of these two training formats:

**Format A (Recommended): wide targets**
- `dataset_df` with columns:
  - features...
  - `score_0 ... score_7`

**Format B: long targets**
- one row per `(t, k)`:
  - features...
  - `combo_id = k`
  - `score`
(Useful for some model types; not required.)

### 3.3 Leakage Guardrails
- `features_df.loc[t]` must only use data `<= t`.
- `score_k(t)` uses data `(t, t+H]` for labels only (allowed offline).
- When splitting train/test, use time-based splits only.

Deliverable:
- `dataset_df` ready for time-series ML.

---

## Phase 4 — Train and Validate (Walk-Forward)

### 4.1 Model Choice
Use LightGBM regressors (one per combo):

For each combo `k`:
- Train model `M_k` to predict `score_k` from features.

This yields 8 models:
- `M_0 ... M_7`

### 4.2 Training Procedure (Walk-Forward)
Use multiple folds by time:

Example:
- Fold 1: Train = weeks 1–3, Test = week 4
- Fold 2: Train = weeks 2–4, Test = week 5
- etc.

No shuffling.

### 4.3 Evaluation Metrics
Per fold, measure:

**Prediction quality**
- `corr(pred_score_k, true_score_k)` per k
- ability to pick winner:
  - `hit_rate = mean(argmax(pred_scores) == argmax(true_scores))`

**Trading-relevant simulation (offline)**
Simulate the controller using only predictions on test period:
- at each `t`, choose `k* = argmax_k pred_score_k(t)`
- apply chosen combo’s realized `pnl_inc` for the next decision interval (not H)
- accumulate equity curve of the adaptive selector

Compare against baselines:
- Best static combo (in that test period)
- Median static combo
- Your current production combo

Report:
- total pnl
- max drawdown
- drawdown duration
- number of switches (even if Phase 5 not implemented yet, report raw flip count)

### 4.4 Artifacts
Save:
- `models/{k}.txt` or LightGBM binary
- `feature_list.json`
- training config (H, λ, decision clock definition, data ranges)
- fold metrics report

---

## Notes / Non-Goals (for Phases 1–4)
- No realtime switching constraints (hysteresis/min-hold) yet; just measure raw instability.
- Do not change trade execution or SAR entry/exit logic.
- Avoid RL and deep nets; use LightGBM regression.

---

## Deliverables Summary
1) `perf_by_combo`: aligned performance frames for 8 combos
2) `features_df`: market-state features at decision times
3) `dataset_df`: features + 8 score targets per timestamp
4) Walk-forward training code producing 8 LightGBM models + evaluation report
