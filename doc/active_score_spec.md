# ActiveScore — Weekly Volatility Seasonality System

## Motivation

Observed from historical `volatility_ema` data:
- **Intra-day pattern**: volatility is higher during US daytime, lower during US night hours.
- **Intra-week pattern**: volatility is lower on US weekends for the full day.
- **Stationarity**: the pattern is stable over time — it is structurally driven by US market hours and human work-life habits, not transient market conditions.

This makes it feasible to model the "activity level" at any moment using a simple statistical average over historical data, with no ML required.

---

## Architecture

Two components:

### 1. Offline Curve Builder (`curve_builder.py`)
- **Input**: historical `VolSpreadStore` parquet data for a chosen market and `param_id`.
- **Process**:
  1. Load `volatility_ema` over the training date range.
  2. Group by `(day_of_week, time_bin)` where `time_bin = floor(seconds_since_utc_midnight / bin_size_sec)`.
  3. Compute mean `volatility_ema` per group → 7 × N_bins matrix.
  4. Min-max normalize the entire matrix to [0.0, 1.0].
- **Output**: a JSON file (`active_score_curve.json`) containing the normalized curve.

### 2. Runtime `ActiveScore` class (`active_score.py`)
- **Input**: the saved JSON curve file.
- **Method**: `get_score(ts: float) -> float`
  - Converts `ts` (Unix seconds, UTC) to `(day_of_week, time_bin)`.
  - Returns the pre-computed normalized score for that slot.
- Score of 1.0 = historically most active period; 0.0 = historically quietest.

---

## Curve File Format

```json
{
  "bin_size_sec": 300,
  "param_id": "vs_60_300_300",
  "market": "GMO:BTC_JPY",
  "trained_on": "20250101-20260101",
  "scores": [
    [0.42, 0.38, ...],
    ...
  ]
}
```

- `scores[dow][bin_idx]` where `dow` ∈ [0=Monday … 6=Sunday], `bin_idx` ∈ [0 … N_bins-1].
- `N_bins = 86400 / bin_size_sec` (e.g. 288 for 5-min bins).

---

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `bin_size_sec` | 300 | 5-minute bins; 288 bins/day × 7 days = 2016 slots |
| `param_id` | `vs_60_300_300` | Which VolSpread config to use as source |

---

## Trading Integration Patterns

### Pattern A — Binary gate
```python
if active_score.get_score(ts) > threshold:
    # place order
```

### Pattern B — Parameter switching
```python
score = active_score.get_score(ts)
params = aggressive_params if score > threshold else conservative_params
```

---

## File Locations

| Item | Path |
|---|---|
| Spec | `doc/active_score_spec.md` |
| Curve builder script | `src/active_score/curve_builder.py` |
| Runtime class | `src/active_score/active_score.py` |
| Output curve file | configurable; e.g. `models/active_score_curve.json` |
