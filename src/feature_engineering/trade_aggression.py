import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import (
    EPS,
    ROLLING_WINDOWS,
    compute_imbalance,
    compute_percentile,
    compute_zscore,
)

LOG = logging.getLogger(__name__)


def build_trade_aggression_features(
    source_df: pd.DataFrame, features: Dict[str, pd.Series], trade_windows: List[str]
) -> None:
    """Build all trade aggression features (Section 1)."""
    LOG.info("Building trade aggression features...")

    for window in trade_windows:
        _build_trade_features_for_window(source_df, features, window)

    # Cross-window features
    _build_cross_window_features(features, trade_windows)


def _build_trade_features_for_window(source_df: pd.DataFrame, features: Dict[str, pd.Series], window: str) -> None:
    suffix = f"_{window}"

    buy_qty = source_df[f"buy_qty{suffix}"]
    sell_qty = source_df[f"sell_qty{suffix}"]
    buy_count = source_df[f"buy_count{suffix}"]
    sell_count = source_df[f"sell_count{suffix}"]
    total_qty = source_df[f"total_qty{suffix}"]
    total_count = source_df[f"total_count{suffix}"]
    vwap = source_df[f"vwap{suffix}"]
    vwap_buy = source_df[f"vwap_buy{suffix}"]
    vwap_sell = source_df[f"vwap_sell{suffix}"]
    qty_per_sec = source_df[f"qty_per_sec{suffix}"]
    trades_per_sec = source_df[f"trades_per_sec{suffix}"]

    features[f"ta_vol_imbalance_ratio{suffix}"] = compute_imbalance(buy_qty, sell_qty)
    features[f"ta_count_imbalance_ratio{suffix}"] = compute_imbalance(buy_count, sell_count)
    features[f"ta_vwap_buy_sell_spread_bps{suffix}"] = (vwap_buy - vwap_sell) / (vwap + EPS) * 10000

    avg_buy_size = buy_qty / (buy_count + EPS)
    avg_sell_size = sell_qty / (sell_count + EPS)
    avg_size = (buy_qty + sell_qty) / (buy_count + sell_count + EPS)

    features[f"ta_size_asymmetry_norm{suffix}"] = compute_imbalance(avg_buy_size, avg_sell_size)

    for roll_window in ROLLING_WINDOWS:
        roll_suffix = f"_{roll_window.replace('min', 'm')}"
        features[f"ta_avg_size_zscore{suffix}{roll_suffix}"] = compute_zscore(avg_size, roll_window)
        features[f"ta_avg_size_pct{suffix}{roll_suffix}"] = compute_percentile(avg_size, roll_window)
        features[f"ta_qty_per_sec_zscore{suffix}{roll_suffix}"] = compute_zscore(qty_per_sec, roll_window)
        features[f"ta_qty_per_sec_pct{suffix}{roll_suffix}"] = compute_percentile(qty_per_sec, roll_window)
        features[f"ta_trades_per_sec_zscore{suffix}{roll_suffix}"] = compute_zscore(trades_per_sec, roll_window)
        features[f"ta_trades_per_sec_pct{suffix}{roll_suffix}"] = compute_percentile(trades_per_sec, roll_window)

    features[f"ta_buy_qty_share{suffix}"] = buy_qty / (total_qty + EPS)
    features[f"ta_buy_count_share{suffix}"] = buy_count / (total_count + EPS)
    features[f"ta_qty_count_divergence{suffix}"] = (
        features[f"ta_buy_qty_share{suffix}"] - features[f"ta_buy_count_share{suffix}"]
    )


def _build_cross_window_features(features: Dict[str, pd.Series], trade_windows: List[str]) -> None:
    w1, w2, w3 = trade_windows[:3]
    ratio_w1 = features[f"ta_vol_imbalance_ratio_{w1}"]
    ratio_w2 = features[f"ta_vol_imbalance_ratio_{w2}"]
    ratio_w3 = features[f"ta_vol_imbalance_ratio_{w3}"]

    features[f"ta_imbalance_momentum_{w1}_{w2}"] = ratio_w1 - ratio_w2
    features[f"ta_imbalance_momentum_{w1}_{w3}"] = ratio_w1 - ratio_w3
    features[f"ta_imbalance_momentum_{w2}_{w3}"] = ratio_w2 - ratio_w3

    features[f"ta_imbalance_sign_agree_{w1}_{w2}"] = (np.sign(ratio_w1) == np.sign(ratio_w2)).astype(int)
    features[f"ta_imbalance_sign_agree_{w1}_{w3}"] = (np.sign(ratio_w1) == np.sign(ratio_w3)).astype(int)
    features[f"ta_imbalance_sign_agree_{w2}_{w3}"] = (np.sign(ratio_w2) == np.sign(ratio_w3)).astype(int)
    features["ta_imbalance_sign_agree_all"] = (
        (features[f"ta_imbalance_sign_agree_{w1}_{w2}"] == 1) & (features[f"ta_imbalance_sign_agree_{w1}_{w3}"] == 1)
    ).astype(int)

    features[f"ta_imbalance_strengthening_{w1}_{w2}"] = (ratio_w1.abs() > ratio_w2.abs()).astype(int)
    features[f"ta_imbalance_strengthening_{w1}_{w3}"] = (ratio_w1.abs() > ratio_w3.abs()).astype(int)
    features[f"ta_imbalance_weakening_{w1}_{w3}"] = (ratio_w1.abs() < ratio_w3.abs()).astype(int)

    features["ta_strong_buy_signal"] = (
        (ratio_w1 > 0.3)
        & (features["ta_imbalance_sign_agree_all"] == 1)
        & (features[f"ta_imbalance_strengthening_{w1}_{w3}"] == 1)
    ).astype(int)

    features["ta_strong_sell_signal"] = (
        (ratio_w1 < -0.3)
        & (features["ta_imbalance_sign_agree_all"] == 1)
        & (features[f"ta_imbalance_strengthening_{w1}_{w3}"] == 1)
    ).astype(int)

    features["ta_exhaustion_signal"] = (
        (ratio_w3.abs() > 0.3)
        & (features[f"ta_imbalance_weakening_{w1}_{w3}"] == 1)
        & (features[f"ta_imbalance_sign_agree_{w1}_{w3}"] == 1)
    ).astype(int)

    features["ta_reversal_signal"] = (
        (ratio_w3.abs() > 0.3) & (features[f"ta_imbalance_sign_agree_{w1}_{w3}"] == 0)
    ).astype(int)
