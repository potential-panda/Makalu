# Specification: Lead-Lag Feature Engineering for Near-HFT

## 1. Objective
Implement a high-performance feature engineering module to identify lead-lag relationships between a **Leader** market (e.g., Binance) and a **Target** market (e.g., GMO). The goal is to provide predictive features for a LightGBM model targeting a **20s–30s** holding period.

Please refer to `LEAD_LAG_PAIRS` in /home/lulurun/workspace/Makalu/src/pipeline/market_pair.py for targeted lead-lag markets.
Please implement the lib in src/lead_lag/*.py then integrate it into the pipeline by creating a lead_lag stage
(in pipeline/stages.py) and lead_lag steps in pipeline/*.py.
Please implement data_check/lead_lag.ipynb

## 2. Input Data Requirements
* **Resolution:** 100ms to 500ms OHLCV or Mid-price data.
* **Leader Series ($L$):** High-liquidity reference price.
* **Target Series ($T$):** The local/laggard price being traded.
* **Transformation:** All price inputs must be converted to **Log-Returns** to ensure stationarity.

## 3. Core Algorithm: Exponentially Weighted Lead-Lag (EWLL)
For a given window $W$ and lag $k$, calculate the **Exponentially Weighted Moving Correlation (EWMC)**.

### Mathematical Logic
1.  **Shift:** Lag the Leader series by $k$ milliseconds.
2.  **Weighting:** Apply an exponential decay $\alpha = 2 / (\text{Span} + 1)$.
3.  **Cross-Correlation:** Use the exponentially weighted covariance and variance to derive the correlation $\rho$ at each lag $k$.
4.  **Optimization:** For every timestamp $t$, find $k^*$ that maximizes $|ρ(k)|$ within the search space.

## 4. Parameter Configuration
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Search Lags** | 0 to 5000ms (100ms steps) | Captures microstructure delay. |
| **Windows ($W$)** | 5s, 15s, 30s, 60s | Multi-scale view of leadership. |
| **Decay Spans** | 10s, 30s, 60s | Matches the 20s–30s trade horizon. |

---

## 5. Feature Dictionary & Schema
The implementation must append the following features to the dataset (suffixed by window length, e.g., `_30s`):

### A. Strength & Timing Features
* **`ll_max_corr`**: The peak EW-correlation coefficient. Range: `[-1, 1]`.
* **`ll_opt_lag_ms`**: The discrete lag value (in ms) where `ll_max_corr` occurred.
* **`ll_alpha_gain`**: Ratio of `ll_max_corr / correlation_at_zero_lag`. Measures signal strength added by the lag.

### B. Confidence & Dynamics Features
* **`ll_lag_stability`**: Rolling standard deviation of `ll_opt_lag_ms` (10-period).
* **`ll_drift`**: The 5-second change in `ll_opt_lag_ms`. Indicates if the gap is widening or closing.

---

## 6. Preprocessing & Normalization (ML-Ready)
To ensure optimal LightGBM performance, the following transformations must be applied:

1.  **Stability Scaling:** Apply `log1p` to `ll_lag_stability` to compress the long tail: $f(x) = \ln(1 + x)$.
2.  **Lag Normalization:** Divide `ll_opt_lag_ms` by the maximum search lag (e.g., 5000) to keep it in a `[0, 1]` range.
3.  **Clipping:** Clip correlation values at `[-1, 1]` and stability features at the 99th percentile.

---

## 7. Implementation Constraints for Code Agent
* **Efficiency:** Use **Vectorized NumPy** or **Numba JIT** for the rolling correlation search. Avoid raw Python loops.
* **Memory:** Pre-calculate lagged versions of the Leader series to avoid redundant memory allocations during the grid search.
* **Handling NaNs:** Use forward-filling for missing ticks. If a window has >50% missing data, return 0 for correlation features.

## 8. Numba-Optimized Reference Code

```python
import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def calc_ew_corr_numba(target_returns, leader_returns, alpha):
    """
    Calculates the Exponentially Weighted Correlation between two series.
    Uses a recursive approach for O(N) efficiency.
    """
    n = len(target_returns)
    ew_cov = 0.0
    ew_var_t = 0.0
    ew_var_l = 0.0

    # Initialize with the first value to avoid bias
    # In practice, you might want to 'warm up' the EWMA
    for i in range(n):
        diff_t = target_returns[i]
        diff_l = leader_returns[i]

        # Recursive EWMA updates
        ew_cov = alpha * (diff_t * diff_l) + (1 - alpha) * ew_cov
        ew_var_t = alpha * (diff_t**2) + (1 - alpha) * ew_var_t
        ew_var_l = alpha * (diff_l**2) + (1 - alpha) * ew_var_l

    if ew_var_t > 0 and ew_var_l > 0:
        return ew_cov / np.sqrt(ew_var_t * ew_var_l)
    return 0.0

@njit(parallel=True)
def lead_lag_search_numba(target_resampled, leader_resampled, lags, alpha):
    """
    Parallelized search over multiple lags to find the optimal lead.
    """
    n_lags = len(lags)
    corrs = np.zeros(n_lags)

    # prange parallelizes the lag search across your CPU cores
    for i in prange(n_lags):
        k = lags[i]
        # Align series by shifting leader returns
        # T_t vs L_{t-k}
        t_segment = target_resampled[k:]
        l_segment = leader_resampled[:len(target_resampled)-k]

        corrs[i] = calc_ew_corr_numba(t_segment, l_segment, alpha)

    best_idx = np.argmax(np.abs(corrs))
    return corrs[best_idx], lags[best_idx]
```
