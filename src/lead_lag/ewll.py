"""
Exponentially Weighted Lead-Lag (EWLL) engine.

Core Numba-optimized computation functions for lead-lag correlation analysis
between a Leader market and a Target market.

Reference: doc/lead_lag_spec.md §3, §8
"""

import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def calc_ew_corr_numba(target_returns, leader_returns, alpha):
    """
    Exponentially Weighted Correlation between two return series.
    Recursive O(N) approach using EWMA covariance and variance.
    """
    n = len(target_returns)
    ew_cov = 0.0
    ew_var_t = 0.0
    ew_var_l = 0.0

    for i in range(n):
        diff_t = target_returns[i]
        diff_l = leader_returns[i]

        ew_cov = alpha * (diff_t * diff_l) + (1 - alpha) * ew_cov
        ew_var_t = alpha * (diff_t**2) + (1 - alpha) * ew_var_t
        ew_var_l = alpha * (diff_l**2) + (1 - alpha) * ew_var_l

    if ew_var_t > 0 and ew_var_l > 0:
        return ew_cov / np.sqrt(ew_var_t * ew_var_l)
    return 0.0


@njit(parallel=True)
def lead_lag_search_numba(target_resampled, leader_resampled, lags, alpha):
    """
    Parallelized search over lag grid to find optimal lead-lag.

    Args:
        target_resampled: Log-returns of target (lag) market
        leader_resampled: Log-returns of leader market
        lags: Array of lag indices (in resampled steps), positive, negative, or zero
        alpha: EW decay factor

    Returns:
        (best_corr, best_lag): Peak |ρ(k)| and corresponding lag index
    """
    n = len(target_resampled)
    n_lags = len(lags)
    corrs = np.zeros(n_lags)

    for i in prange(n_lags):
        k = lags[i]
        # Align: target_t vs leader_{t-k}. k>=0: leader_resampled[:n-k]/[:] must NOT
        # use Python/numpy negative-index slicing for k<0 (that means "from the
        # end", not "shift by k") -- so the two directions are split explicitly.
        if k >= 0:
            t_segment = target_resampled[k:]
            l_segment = leader_resampled[: n - k]
        else:
            m = -k
            t_segment = target_resampled[: n - m]
            l_segment = leader_resampled[m:]

        if len(t_segment) > 0 and len(l_segment) > 0:
            corrs[i] = calc_ew_corr_numba(t_segment, l_segment, alpha)

    best_idx = np.argmax(np.abs(corrs))
    return corrs[best_idx], lags[best_idx]


def compute_lead_lag_features(target_returns, leader_returns, lags, alpha):
    """
    Compute EWLL features for a single window.

    Args:
        target_returns: Log-returns of target market (np.ndarray)
        leader_returns: Log-returns of leader market (np.ndarray)
        lags: Array of lag indices (in resampled steps)
        alpha: EW decay factor

    Returns:
        dict with keys: ll_max_corr, ll_opt_lag, ll_corr_at_zero
    """
    if len(target_returns) < 2 or len(leader_returns) < 2:
        return {
            "ll_max_corr": 0.0,
            "ll_opt_lag": 0,
            "ll_corr_at_zero": 0.0,
        }

    best_corr, best_lag = lead_lag_search_numba(
        target_returns, leader_returns, lags, alpha
    )

    # Correlation at zero lag
    corr_at_zero = calc_ew_corr_numba(target_returns, leader_returns, alpha)

    # Clip correlation to [-1, 1]
    best_corr = float(np.clip(best_corr, -1.0, 1.0))
    corr_at_zero = float(np.clip(corr_at_zero, -1.0, 1.0))

    return {
        "ll_max_corr": best_corr,
        "ll_opt_lag": int(best_lag),
        "ll_corr_at_zero": corr_at_zero,
    }
