# Meta-Regime Strategy Filter
**Implementation Spec (for code agent)**

## Goal
Keep the existing base strategy unchanged and add a learned veto layer that suppresses trades when the strategy is entering an unfavorable performance regime.

The model does not predict price. It predicts whether the current market state is compatible with the base strategy continuing to perform well over the next horizon.

This document defines:
- the decision clock
- the labeling target
- the feature/label alignment rules
- the offline evaluation protocol

It is intended to be implementable in the current Makalu pipeline without relying on ambiguous hindsight rules.

---

## 1. Strategy Framing

### 1.1 Base Strategy
Assume the base strategy already exists and emits:
- trade timestamps
- realized `pnl_increment`
- cumulative `equity`
- optional position/exposure fields

The veto model is a second-stage model:
- base strategy proposes trade or continuation
- veto model decides `allow` or `block`

### 1.2 Modeling Objective
We want the model to answer:

> "Given the features observable at time `t`, is the strategy likely to suffer a material drawdown over the next horizon `H`?"

This is a regime-quality problem, not a price-direction problem.

### 1.3 Output Type
Train a binary classifier or probabilistic classifier:
- `label = 1`: favorable regime, allow trading
- `label = 0`: unfavorable regime, veto trading

For deployment, use model probability rather than raw class when possible:
- `p_allow = P(label = 1 | X_t)`

This allows later addition of thresholds, hysteresis, and minimum-hold rules.

---

## 2. Decision Clock

### 2.1 Required Choice
All features, labels, and evaluations must use the same decision timestamps.

**Default decision clock**:
- decision happens at each **range-bar close** timestamp of the base strategy

If this is not operationally stable, fallback is:
- a fixed clock such as every `10s` or `30s`

### 2.2 Decision-Time Contract
At each decision timestamp `t`, the system must be able to:
- compute features using only data with timestamp `<= t`
- assign an offline training label using future data in `(t, t+H]`
- simulate a realtime veto action applied immediately after `t`

### 2.3 Data Index Contract
The modeling dataframe must be indexed by a single monotonic timestamp column:
- `decision_ts`

Every row corresponds to one allowed model decision.

---

## 3. Labeling Objective

### 3.1 Why Not Use Raw Peak/Trough Labels Directly
Simple hindsight segmentation such as:
- trough to peak = `1`
- peak to trough = `0`

is intuitive, but it creates ambiguity near turning points:
- rows immediately before a confirmed peak still get labeled `1`
- those are often exactly the rows where the veto should start activating

Therefore, the target should be based on **future drawdown risk from the current timestamp**, not only retrospective regime state.

### 3.2 Primary Target Definition
For each decision time `t`, define:
- `equity(t)` = cumulative realized PnL of the base strategy up to `t`
- `future_path(t, H)` = equity path over `(t, t+H]`
- `future_peak(t, H)` = max equity over `(t, t+H]`
- `future_trough(t, H)` = min equity over `(t, t+H]`
- `future_drawdown_from_now(t, H)` = `equity(t) - future_trough(t, H)`

Primary binary target:
- `label = 0` if `future_drawdown_from_now(t, H) >= delta`
- `label = 1` otherwise

Interpretation:
- `0` means the strategy is about to enter a sufficiently bad regime within the horizon
- `1` means no material deterioration is observed within the horizon

This target is causal for inference and only uses future data for offline labeling.

### 3.3 Threshold `delta`
Do **not** define `delta` as a fixed raw PnL amount unless the strategy size and volatility are constant.

Use one of these normalized definitions:

1. Volatility-scaled threshold:
- `delta(t) = k * rolling_equity_vol(t, W)`

2. Rolling-PnL-scale threshold:
- `delta(t) = k * rolling_abs_pnl_mean(t, W)`

3. Drawdown-percentile threshold:
- choose `delta` as a percentile of historical forward drawdowns on the training set

**Recommended default**:
- volatility-scaled threshold with rolling window `W = 1d`
- start with `k` chosen so the negative class is roughly `20%` to `40%`

This keeps label density more stable across market conditions.

### 3.4 Horizon `H`
The target must use a fixed forward horizon:
- start with `H = 30m`
- also test `H = 1h`
- optionally test `H = 2h`

Choose `H` to match the expected persistence of strategy deterioration.

### 3.5 Optional Secondary Label
For analysis only, you may also compute:
- `max_runup_from_now(t, H) = future_peak(t, H) - equity(t)`

This supports richer diagnostics such as:
- drawdown dominates run-up
- asymmetric opportunity vs risk

Do not make this part of the first implementation unless needed.

---

## 4. Optional ZigZag Regime Segmentation

### 4.1 Purpose
ZigZag-style segmentation may still be useful:
- for visualization
- for exploratory analysis
- for validating whether detected bad periods align with obvious equity deterioration

### 4.2 Non-Goal
Do **not** use raw ZigZag segments as the primary training label in v1.

They are allowed as:
- analysis overlays
- coarse regime summaries
- sanity-check plots

### 4.3 If Implemented
If you implement ZigZag segmentation for analysis, define all of the following explicitly:
- swing confirmation threshold
- whether threshold is absolute, relative, or volatility-scaled
- minimum regime duration
- minimum swing size
- how unfinished final segments are handled

---

## 5. Turning-Point Ambiguity Guardrails

### 5.1 Buffer Zone
Rows close to regime boundaries are often mislabeled or unstable.

Define a boundary buffer:
- `buffer = 1 to 3 decision intervals`, or
- a time buffer such as `60s` to `300s`

If a row is within the buffer of a confirmed turning point or label flip, either:
- drop it from training, or
- mark it as `ambiguous`

**Recommended default**:
- drop ambiguous rows from training

### 5.2 Minimum Segment Length
If using any segment-based analysis, require:
- minimum favorable segment length
- minimum unfavorable segment length

This avoids one-bar flips being treated as genuine regime changes.

---

## 6. Feature Alignment Rules

### 6.1 Input Rows
For each `decision_ts = t`, compute features `X_t` using only information available at or before `t`.

No feature may use:
- future trades
- future bars
- future PnL
- future volatility estimates that include timestamps after `t`

### 6.2 Initial Feature Groups
Start with a compact online-friendly set, such as:

**Volatility**
- `rv_5m`
- `rv_30m`
- `rv_2h`
- `range_5m`

**Spread / Friction**
- `spread_bps_mean_5m`
- `spread_bps_p95_5m`
- `spread_bps_mean_30m`

**Lead-Lag**
- `leadlag_corr_peak`
- `leadlag_best_lag_ms`
- `leadlag_strength_5m`

**Liquidity / Microstructure**
- `imbalance_mean_5m`
- `trade_aggr_ratio_5m`
- `depth_bid_ask_ratio_5m`

**Strategy State** (Input 1, see §6.4)
- `time_since_last_sar_flip`
- `range_bar_rate_5m`
- `recent_trade_count`
- `recent_win_rate`
- `recent_realized_pnl_5m`
- `recent_realized_pnl_30m`

The last group is important because this is a meta-regime filter on the strategy itself, not only on the market.

### 6.3 Market Feature Implementation

The market feature groups (Volatility, Spread/Friction, Lead-Lag, Liquidity/Microstructure) are implemented as `FeatureTarget.ORION` in `src/silver.py`.

To generate these features, run the pipeline with stage `Feature_Orion`:

```
makalu-pipeline --market <market> --from-dt <YYYYMMDD> --to-dt <YYYYMMDD> --stages Feature_Orion
```

The pipeline step configuration is:

| Step | Windows |
|---|---|
| TradeAggression | 300s, 1800s / rolling 300s, 1800s, 7200s |
| OrderBookState | depth [5, 10, 20] / windows 300s, 1800s |
| BookChange | depth [5, 10, 20] / rolling 300s, 1800s, 7200s |
| ExecutionPrice | rolling 300s, 1800s, 7200s |
| Volatility | 300s, 1800s, 7200s |
| Trend | 300s, 1800s, 7200s |
| LeadLag | `LEAD_LAG_CONFIG_FAR` (decay spans 120–600s, windows 5–20m) |

> **NOTE:** The window parameters above are based on the assumption that Orion operates on regime-level timescales (30m–2h horizon). This assumption is derived from the label horizon `H` in §3.4 and has not been validated against Orion's actual trade holding duration. **Revisit these parameters once the typical hold time is confirmed.**

### 6.4 Strategy State Input (Input 1)

The Strategy State feature group requires a separate input DataFrame produced by the base strategy itself, not the market data pipeline.

**`strategy_df`** — one row per range-bar close, indexed by `decision_ts`:

| Column | Type | Description |
|---|---|---|
| `decision_ts` | datetime (index) | Range-bar close timestamp; defines the decision clock |
| `equity` | float | Cumulative realized PnL up to and including this bar |
| `pnl_increment` | float | Realized PnL from the trade at this bar; `0.0` if no trade executed |
| `sar_direction` | int | SAR indicator direction at bar close: `+1` = long, `-1` = short |
| `trade_executed` | bool | Whether a trade entry or exit occurred at this bar close |

Derived features computed from `strategy_df`:

| Feature | Derivation |
|---|---|
| `time_since_last_sar_flip` | elapsed time since last row where `sar_direction` changed |
| `range_bar_rate_5m` | count of rows (range bars) in a rolling 5m backward window |
| `recent_trade_count` | rolling count of `trade_executed == True` over last 5m |
| `recent_win_rate` | rolling mean of `pnl_increment > 0` over last N executed trades |
| `recent_realized_pnl_5m` | rolling sum of `pnl_increment` over last 5m |
| `recent_realized_pnl_30m` | rolling sum of `pnl_increment` over last 30m |

`equity` is also used directly as input to label construction (§3.2) and is not a model feature.

### 6.5 NaNs and Outliers
- forward-fill only where operationally valid
- otherwise drop rows with missing critical features
- winsorize or cap extreme outliers if necessary

Tree models do not require scaling.

---

## 7. Dataset Schema

### 7.1 Required Columns
The training dataframe should contain at least:
- `decision_ts`
- feature columns `X_*`
- `equity`
- `future_drawdown_from_now`
- `delta`
- `label`

Optional diagnostic columns:
- `future_runup_from_now`
- `is_ambiguous`
- `regime_id`
- `segment_id`

### 7.2 Example Schema

| decision_ts | rv_5m | spread_bps_mean_5m | recent_realized_pnl_5m | equity | future_drawdown_from_now | delta | label |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-03-27 10:00:01 | 0.0041 | 1.8 | 12.4 | 1032.5 | 4.1 | 9.0 | 1 |
| 2026-03-27 10:15:22 | 0.0092 | 4.3 | -8.7 | 1024.0 | 14.2 | 9.0 | 0 |

### 7.3 Label Construction Pseudocode
For each row at time `t`:

1. compute current `equity(t)`
2. collect future equity path on `(t, t+H]`
3. compute `future_drawdown_from_now(t, H)`
4. compute normalized threshold `delta(t)`
5. assign:
   - `0` if drawdown exceeds threshold
   - `1` otherwise
6. optionally drop row if it falls inside an ambiguity buffer

---

## 8. Training Protocol

### 8.1 Model Choice
Use LightGBM binary classification first.

Output:
- probability of favorable regime `p_allow`

Alternative later:
- regression on future drawdown magnitude
- ordinal labels
- multi-task model for drawdown and run-up

These are out of scope for v1.

### 8.2 Split Method
Use time-based walk-forward validation only.

Example:
- Fold 1: train weeks 1 to 3, test week 4
- Fold 2: train weeks 2 to 4, test week 5
- Fold 3: train weeks 3 to 5, test week 6

No shuffling.

### 8.3 Leakage Guardrails
- features at `t` must only use data `<= t`
- labels may use future path `(t, t+H]` offline only
- threshold calibration must be done on train folds only
- if using percentile-based `delta`, compute it only from training data in each fold

---

## 9. Offline Evaluation

### 9.1 Model Metrics
Track standard classification metrics:
- ROC-AUC
- PR-AUC
- recall on negative regime (`label = 0`)
- precision on negative regime
- calibration quality if probabilities are used

These are secondary.

### 9.2 Trading-Relevant Metrics
Primary evaluation must simulate the base strategy with veto applied on the test fold.

At each decision time `t`:
- compute `p_allow(t)`
- if `p_allow(t) >= threshold`, allow base strategy action
- otherwise veto new action or suppress trading according to the execution rule

Report:
- total PnL
- max drawdown
- drawdown duration
- Sharpe-like risk-adjusted metric if already used internally
- trade count
- trade retention rate
- fraction of gross losses removed
- fraction of gross profits mistakenly removed

### 9.3 Baselines
Compare against:
- base strategy with no veto
- random veto at same retention rate
- simple heuristic veto, if available
  - for example spread too wide or volatility too high

The model is only useful if it beats both the no-veto baseline and a simple heuristic.

### 9.4 Threshold Sweep
Evaluate several veto thresholds:
- conservative
- balanced
- aggressive

For example:
- `p_allow >= 0.3`
- `p_allow >= 0.5`
- `p_allow >= 0.7`

This produces a retention-vs-drawdown tradeoff curve.

---

## 10. Deployment Notes

### 10.1 v1 Realtime Logic
Do not deploy with instantaneous class flips only.

Use one of:
- probability threshold plus minimum persistence
- hysteresis with separate on/off thresholds
- minimum-hold time after veto activation

**Recommended default**:
- turn veto on when `p_allow < 0.4`
- turn veto off when `p_allow > 0.6`
- require persistence for `N` consecutive decisions

### 10.2 Execution Semantics
Specify exactly what veto means:
- block only new entries
- or flatten and stop trading
- or pause entries while allowing exits

**Recommended default**:
- veto blocks new entries only
- existing risk management and exits remain unchanged

This is safer and easier to simulate.

---

## 11. Deliverables

### 11.1 Data Artifacts
- `features_df` indexed by `decision_ts`
- `labels_df` indexed by `decision_ts`
- merged `dataset_df`

### 11.2 Model Artifacts
- trained LightGBM model
- feature list
- training config:
  - decision clock
  - horizon `H`
  - threshold rule for `delta`
  - ambiguity buffer
  - walk-forward fold definition

### 11.3 Evaluation Artifacts
- fold metrics report
- vetoed-vs-baseline equity curves
- threshold sweep summary
- optional ZigZag overlay charts for sanity checks

---

## 12. Non-Goals for v1
- no direct price prediction
- no reinforcement learning
- no deep sequence models
- no raw hindsight ZigZag labels as primary target
- no fully automatic position flattening policy unless separately specified

---

## 13. Recommended First Implementation

Implement v1 with these defaults:
- decision clock = range-bar close
- horizon `H = 30m`
- target = future drawdown from now
- `delta(t)` = volatility-scaled threshold
- LightGBM binary classifier
- ambiguous rows near label flips dropped
- evaluation based on vetoed-strategy equity, not only classification metrics
- veto blocks new entries only

This version is simple enough to implement and test, while remaining aligned with realtime use.
