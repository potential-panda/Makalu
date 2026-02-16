import logging
from typing import Dict, List

import pandas as pd

from .utils import (
    EPS,
    compute_imbalance,
    compute_imbalance_abs,
    compute_percentile,
    compute_zscore,
)

LOG = logging.getLogger(__name__)


def build_book_change_features(
    source_df: pd.DataFrame, features: Dict[str, pd.Series], depth_levels: List[int], rolling_windows: List[str]
):
    """Build all book change features (Section 3)."""
    LOG.info("Building book change features...")

    for k in depth_levels:
        _build_book_change_features_for_k(source_df, features, k, rolling_windows)

    # Cross-k features (gradient between shallow and deep)
    _build_book_change_cross_k_features(features, depth_levels)


def _build_book_change_features_for_k(
    source_df: pd.DataFrame, features: Dict[str, pd.Series], k: int, rolling_windows: List[str]
):
    suffix = f"_k{k}"

    delta_depth_bid = source_df[f"delta_depth_bid{suffix}"]
    delta_depth_ask = source_df[f"delta_depth_ask{suffix}"]
    pull_bid_rate = source_df[f"pull_bid_rate{suffix}"]
    pull_ask_rate = source_df[f"pull_ask_rate{suffix}"]
    refill_bid_rate = source_df[f"refill_bid_rate{suffix}"]
    refill_ask_rate = source_df[f"refill_ask_rate{suffix}"]

    features[f"bc_delta_depth_imbalance_norm{suffix}"] = compute_imbalance_abs(delta_depth_bid, delta_depth_ask)
    features[f"bc_pull_imbalance{suffix}"] = compute_imbalance(pull_bid_rate, pull_ask_rate)
    features[f"bc_refill_imbalance{suffix}"] = compute_imbalance(refill_bid_rate, refill_ask_rate)

    total_pull_rate = pull_bid_rate + pull_ask_rate
    total_refill_rate = refill_bid_rate + refill_ask_rate
    features[f"bc_pull_refill_ratio{suffix}"] = total_pull_rate / (total_refill_rate + EPS)

    total_activity = total_pull_rate + total_refill_rate

    for roll_window in rolling_windows:
        roll_suffix = f"_{roll_window}"
        features[f"bc_total_pull_zscore{suffix}{roll_suffix}"] = compute_zscore(total_pull_rate, roll_window)
        features[f"bc_total_activity_zscore{suffix}{roll_suffix}"] = compute_zscore(total_activity, roll_window)
        features[f"bc_total_activity_pct{suffix}{roll_suffix}"] = compute_percentile(total_activity, roll_window)


def _build_book_change_cross_k_features(features: Dict[str, pd.Series], depth_levels: List[int]):
    k1 = depth_levels[0]
    k_mid = depth_levels[min(1, len(depth_levels) - 1)]

    pull_imb_k1 = features[f"bc_pull_imbalance_k{k1}"]
    pull_imb_kmid = features[f"bc_pull_imbalance_k{k_mid}"]
    features["bc_pull_imbalance_gradient"] = pull_imb_k1 - pull_imb_kmid

    refill_imb_k1 = features[f"bc_refill_imbalance_k{k1}"]
    refill_imb_kmid = features[f"bc_refill_imbalance_k{k_mid}"]
    features["bc_refill_imbalance_gradient"] = refill_imb_k1 - refill_imb_kmid
