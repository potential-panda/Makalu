import logging
from typing import Dict, List

import pandas as pd

from .utils import EPS, compute_bps_offset

LOG = logging.getLogger(__name__)


def build_order_book_state_features(source_df: pd.DataFrame, features: Dict[str, pd.Series], depth_levels: List[int]):
    """Build all order book state features (Section 2)."""
    LOG.info("Building order book state features...")

    best_mid = source_df["best_mid"]
    for k in depth_levels:
        microprice = source_df[f"microprice_k{k}"]
        features[f"bs_microprice_offset_bps_k{k}"] = compute_bps_offset(microprice, best_mid)

    # Use first, 5th and 10th depth levels for gradient/divergence if available
    k1 = depth_levels[0]
    k_mid = depth_levels[min(2, len(depth_levels) - 1)]  # k5 equivalent
    k_deep = depth_levels[-1]  # k10 equivalent

    vol_imb_k1 = source_df[f"vol_imb_k{k1}"]
    vol_imb_kmid = source_df[f"vol_imb_k{k_mid}"]
    vol_imb_kdeep = source_df[f"vol_imb_k{k_deep}"]

    features["bs_imbalance_gradient"] = vol_imb_k1 - vol_imb_kmid
    features["bs_imbalance_divergence"] = (vol_imb_k1 - vol_imb_kdeep).abs()

    spread_bps = source_df["spread_bps"]
    microprice_offset_bps_k1 = features[f"bs_microprice_offset_bps_k{k1}"]
    features["bs_microprice_offset_norm"] = microprice_offset_bps_k1 / (spread_bps + EPS)

    _build_order_book_rolling_aggregates(source_df, features, depth_levels)


def _build_order_book_rolling_aggregates(
    source_df: pd.DataFrame, features: Dict[str, pd.Series], depth_levels: List[int]
):
    windows = {"1s": "1s", "3s": "3s", "5s": "5s"}
    k1 = depth_levels[0]
    k_mid = depth_levels[min(2, len(depth_levels) - 1)]

    features_to_roll = {
        "spread_bps": source_df["spread_bps"],
        f"vol_imb_k{k1}": source_df[f"vol_imb_k{k1}"],
        f"vol_imb_k{k_mid}": source_df[f"vol_imb_k{k_mid}"],
        f"microprice_offset_bps_k{k1}": features[f"bs_microprice_offset_bps_k{k1}"],
    }
    for name, series in features_to_roll.items():
        for window_name, window_size in windows.items():
            rolling_op = series.rolling(window_size, closed="left")
            features[f"bs_{name}_mean_{window_name}"] = rolling_op.mean()
            features[f"bs_{name}_std_{window_name}"] = rolling_op.std()
