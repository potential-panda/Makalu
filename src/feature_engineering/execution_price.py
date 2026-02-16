import logging
from typing import Dict, List

import pandas as pd

from .utils import EPS, compute_percentile, compute_zscore

LOG = logging.getLogger(__name__)


def build_execution_price_features(
    source_df: pd.DataFrame, features: Dict[str, pd.Series], rolling_windows: List[str]
):
    """Build all execution price features (Section 4)."""
    LOG.info("Building execution price features...")

    best_bid = source_df["best_bid"]
    best_ask = source_df["best_ask"]
    exec_bid = source_df["bid"]
    exec_ask = source_df["ask"]
    mid = source_df["mid"]
    spread_bps = source_df["spread_bps"]

    features["ep_slippage_bid_bps"] = (best_bid - exec_bid) / (best_bid + EPS) * 10000
    features["ep_slippage_ask_bps"] = (exec_ask - best_ask) / (best_ask + EPS) * 10000

    slippage_bid_bps = features["ep_slippage_bid_bps"]
    slippage_ask_bps = features["ep_slippage_ask_bps"]
    slippage_total_bps = slippage_bid_bps + slippage_ask_bps
    features["ep_slippage_total_bps"] = slippage_total_bps

    for roll_window in rolling_windows:
        roll_suffix = f"_{roll_window}"
        features[f"ep_slippage_zscore{roll_suffix}"] = compute_zscore(slippage_total_bps, roll_window)
        features[f"ep_slippage_pct{roll_suffix}"] = compute_percentile(slippage_total_bps, roll_window)

    features["ep_slippage_asymmetry"] = (slippage_ask_bps - slippage_bid_bps) / (slippage_total_bps + EPS)
    features["ep_depth_quality"] = 1 / (slippage_total_bps + 1)

    exec_spread_bps = (exec_ask - exec_bid) / (mid + EPS) * 10000
    features["ep_exec_spread_bps"] = exec_spread_bps
    features["ep_spread_expansion_bps"] = exec_spread_bps - spread_bps
    features["ep_spread_expansion_ratio"] = exec_spread_bps / (spread_bps + EPS)
