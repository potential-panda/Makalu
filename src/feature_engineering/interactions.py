import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import EPS, compute_bps_offset

LOG = logging.getLogger(__name__)


def build_cross_feature_interactions(
    source_df: pd.DataFrame,
    features: Dict[str, pd.Series],
    trade_windows: List[str],
    depth_levels: List[int],
    base_roll_window: str,
):
    """Build all cross-feature interaction features (Section 5)."""
    LOG.info("Building cross-feature interactions...")

    _build_trade_book_interactions(source_df, features, trade_windows, depth_levels)
    _build_trade_change_interactions(features, trade_windows, depth_levels)
    _build_book_change_interactions(source_df, features, depth_levels)
    _build_book_exec_interactions(source_df, features, depth_levels)
    _build_trade_exec_interactions(source_df, features, trade_windows, base_roll_window)
    _build_multiway_interactions(source_df, features, trade_windows, depth_levels, base_roll_window)


def _build_trade_book_interactions(
    source_df: pd.DataFrame, features: Dict[str, pd.Series], trade_windows: List[str], depth_levels: List[int]
):
    mid = source_df["mid"]
    w1 = trade_windows[0]
    k1 = depth_levels[0]

    vol_imb_k1 = source_df[f"vol_imb_k{k1}"]
    vol_imbalance_ratio_w1 = features[f"ta_vol_imbalance_ratio_{w1}"]

    for window in trade_windows:
        suffix = f"_{window}"
        vwap = source_df[f"vwap{suffix}"]
        vwap_buy = source_df[f"vwap_buy{suffix}"]
        vwap_sell = source_df[f"vwap_sell{suffix}"]
        features[f"cf_vwap_offset_bps{suffix}"] = compute_bps_offset(vwap, mid)
        features[f"cf_vwap_buy_offset_bps{suffix}"] = compute_bps_offset(vwap_buy, mid)
        features[f"cf_vwap_sell_offset_bps{suffix}"] = compute_bps_offset(vwap_sell, mid)

    features["cf_sign_agree_trade_book"] = ((vol_imbalance_ratio_w1 > 0) == (vol_imb_k1 > 0)).astype(int)
    features["cf_strong_agreement"] = (
        (vol_imbalance_ratio_w1.abs() > 0.3) & (vol_imb_k1.abs() > 0.3) & (features["cf_sign_agree_trade_book"] == 1)
    ).astype(int)

    features["cf_trade_book_divergence"] = (
        ((vol_imbalance_ratio_w1 > 0.1) & (vol_imb_k1 < -0.2)) | ((vol_imbalance_ratio_w1 < -0.1) & (vol_imb_k1 > 0.2))
    ).astype(int)
    features["cf_divergence_magnitude"] = vol_imbalance_ratio_w1 * (-vol_imb_k1)

    vwap_offset_bps_w1 = features[f"cf_vwap_offset_bps_{w1}"]
    microprice_offset_bps_k1 = features[f"bs_microprice_offset_bps_k{k1}"]
    features["cf_vwap_microprice_agree"] = ((vwap_offset_bps_w1 > 0) == (microprice_offset_bps_k1 > 0)).astype(int)
    features["cf_vwap_microprice_spread"] = vwap_offset_bps_w1 - microprice_offset_bps_k1


def _build_trade_change_interactions(
    features: Dict[str, pd.Series], trade_windows: List[str], depth_levels: List[int]
):
    w1 = trade_windows[0]
    k1 = depth_levels[0]
    vol_imbalance_ratio_w1 = features[f"ta_vol_imbalance_ratio_{w1}"]
    pull_imbalance_k1 = features[f"bc_pull_imbalance_k{k1}"]
    delta_depth_imbalance_norm_k1 = features[f"bc_delta_depth_imbalance_norm_k{k1}"]

    features["cf_trade_consuming_asks"] = ((vol_imbalance_ratio_w1 > 0.3) & (pull_imbalance_k1 < -0.3)).astype(int)
    features["cf_trade_consuming_bids"] = ((vol_imbalance_ratio_w1 < -0.3) & (pull_imbalance_k1 > 0.3)).astype(int)
    features["cf_consumption_alignment"] = vol_imbalance_ratio_w1 * (-pull_imbalance_k1)
    features["cf_absorption_bid"] = ((vol_imbalance_ratio_w1 < -0.3) & (delta_depth_imbalance_norm_k1 >= 0)).astype(
        int
    )
    features["cf_absorption_ask"] = ((vol_imbalance_ratio_w1 > 0.3) & (delta_depth_imbalance_norm_k1 <= 0)).astype(int)
    features["cf_absorption_strength"] = vol_imbalance_ratio_w1 * (-delta_depth_imbalance_norm_k1)


def _build_book_change_interactions(source_df: pd.DataFrame, features: Dict[str, pd.Series], depth_levels: List[int]):
    k1 = depth_levels[0]
    vol_imb_k1 = source_df[f"vol_imb_k{k1}"]
    best_bid_reset = source_df[f"best_bid_reset_k{k1}"]
    best_ask_reset = source_df[f"best_ask_reset_k{k1}"]
    refill_imbalance_k1 = features[f"bc_refill_imbalance_k{k1}"]
    pull_imbalance_k1 = features[f"bc_pull_imbalance_k{k1}"]
    delta_depth_imbalance_norm_k1 = features[f"bc_delta_depth_imbalance_norm_k{k1}"]

    features["cf_imbalance_momentum"] = vol_imb_k1 * delta_depth_imbalance_norm_k1
    features["cf_imbalance_building"] = ((vol_imb_k1 > 0) & (refill_imbalance_k1 > 0.2)).astype(int)
    features["cf_imbalance_fading"] = ((vol_imb_k1 > 0) & (pull_imbalance_k1 > 0.2)).astype(int)
    features["cf_stable_wall_bid"] = ((vol_imb_k1 > 0.5) & (best_bid_reset == 0)).astype(int)
    features["cf_stable_wall_ask"] = ((vol_imb_k1 < -0.5) & (best_ask_reset == 0)).astype(int)
    features["cf_unstable_book"] = ((vol_imb_k1.abs() < 0.2) & ((best_bid_reset == 1) | (best_ask_reset == 1))).astype(
        int
    )


def _build_book_exec_interactions(source_df: pd.DataFrame, features: Dict[str, pd.Series], depth_levels: List[int]):
    mid = source_df["mid"]
    k1 = depth_levels[0]
    vol_imb_k1 = source_df[f"vol_imb_k{k1}"]
    microprice_k1 = source_df[f"microprice_k{k1}"]
    spread_bps = source_df["spread_bps"]
    slippage_bid_bps = features["ep_slippage_bid_bps"]
    slippage_ask_bps = features["ep_slippage_ask_bps"]
    slippage_total_bps = features["ep_slippage_total_bps"]
    depth_quality = features["ep_depth_quality"]

    features["cf_slippage_imbalance_align"] = (
        ((vol_imb_k1 > 0) & (slippage_ask_bps > slippage_bid_bps))
        | ((vol_imb_k1 < 0) & (slippage_bid_bps > slippage_ask_bps))
    ).astype(int)
    slippage_ratio = slippage_ask_bps / (slippage_bid_bps + EPS)
    features["cf_slippage_imbalance_corr"] = slippage_ratio * vol_imb_k1
    features["cf_microprice_exec_gap"] = (microprice_k1 - mid) / (mid + EPS) * 10000
    features["cf_depth_quality_per_spread"] = depth_quality / (spread_bps + EPS)
    features["cf_spread_depth_ratio"] = spread_bps / (slippage_total_bps + EPS)


def _build_trade_exec_interactions(
    source_df: pd.DataFrame, features: Dict[str, pd.Series], trade_windows: List[str], base_roll_window: str
) -> None:
    w1 = trade_windows[0]
    vwap_first = source_df[f"vwap_{w1}"]
    mid = source_df["mid"]

    roll_suffix = f"_{base_roll_window}"
    trades_per_sec_pct_first = features[f"ta_trades_per_sec_pct_{w1}{roll_suffix}"]
    slippage_pct = features[f"ep_slippage_pct{roll_suffix}"]

    features["cf_efficient_trading"] = ((trades_per_sec_pct_first > 0.7) & (slippage_pct < 0.3)).astype(int)
    features["cf_stressed_trading"] = ((trades_per_sec_pct_first > 0.7) & (slippage_pct > 0.7)).astype(int)
    features["cf_activity_slippage_divergence"] = trades_per_sec_pct_first - slippage_pct
    features["cf_trade_execution_edge"] = (vwap_first - mid) / (mid + EPS) * 10000


def _build_multiway_interactions(
    source_df: pd.DataFrame,
    features: Dict[str, pd.Series],
    trade_windows: List[str],
    depth_levels: List[int],
    base_roll_window: str,
) -> None:
    w1 = trade_windows[0]
    k1 = depth_levels[0]
    vol_imb_k1 = source_df[f"vol_imb_k{k1}"]
    vol_imbalance_ratio_w1 = features[f"ta_vol_imbalance_ratio_{w1}"]
    microprice_offset_bps_k1 = features[f"bs_microprice_offset_bps_k{k1}"]
    delta_depth_imbalance_norm_k1 = features[f"bc_delta_depth_imbalance_norm_k{k1}"]
    sign_agree_trade_book = features["cf_sign_agree_trade_book"]
    trade_consuming_asks = features["cf_trade_consuming_asks"]
    trade_consuming_bids = features["cf_trade_consuming_bids"]
    absorption_bid = features["cf_absorption_bid"]
    absorption_ask = features["cf_absorption_ask"]

    roll_suffix = f"_{base_roll_window}"
    trades_per_sec_pct_w1 = features[f"ta_trades_per_sec_pct_{w1}{roll_suffix}"]
    total_activity_pct_k1 = features[f"bc_total_activity_pct_k{k1}{roll_suffix}"]
    spread_bps = source_df["spread_bps"]
    depth_quality = features["ep_depth_quality"]

    signals = [
        np.sign(vol_imbalance_ratio_w1),
        np.sign(vol_imb_k1),
        np.sign(microprice_offset_bps_k1),
        np.sign(delta_depth_imbalance_norm_k1),
    ]
    signals_sum = sum(signals)
    features["cf_confirmation_score"] = signals_sum / len(signals)
    features["cf_signal_agreement"] = (signals_sum / len(signals)).abs()

    signal_agreement = features["cf_signal_agreement"]
    features["cf_regime_trending"] = (
        (sign_agree_trade_book == 1)
        & ((trade_consuming_asks == 1) | (trade_consuming_bids == 1))
        & (signal_agreement > 0.5)
    ).astype(int)
    features["cf_regime_mean_reverting"] = (
        (sign_agree_trade_book == 0) & ((absorption_bid == 1) | (absorption_ask == 1)) & (signal_agreement < 0.3)
    ).astype(int)
    features["cf_regime_volatile"] = (
        (trades_per_sec_pct_w1 > 0.8) & (total_activity_pct_k1 > 0.8) & (signal_agreement < 0.3)
    ).astype(int)

    confirmation_score = features["cf_confirmation_score"]
    spread_factor = 1 / (spread_bps + 1)
    features["cf_quality_weighted_signal"] = confirmation_score * signal_agreement * depth_quality * spread_factor
