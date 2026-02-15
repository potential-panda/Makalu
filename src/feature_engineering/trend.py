import logging
from typing import Dict

import pandas as pd

from .utils import EPS

LOG = logging.getLogger(__name__)


def build_trend_features(source_df: pd.DataFrame, features: Dict[str, pd.Series], trend_windows: list[int]) -> None:
    """Build price trend and momentum features (Section 6)."""
    LOG.info("Building trend features...")
    mid = source_df["mid"]

    for sec in trend_windows:
        lookback = f"{sec}s"
        shifted_mid = mid.shift(1, freq=lookback).reindex(mid.index, method="ffill")
        features[f"tr_roc_{sec}s"] = (mid / (shifted_mid + EPS)) - 1

    ema_fast = mid.ewm(span=trend_windows[0], adjust=False).mean()
    ema_slow = mid.ewm(span=trend_windows[-1], adjust=False).mean()
    features[f"tr_ema_spread_{trend_windows[0]}_{trend_windows[-1]}"] = (ema_fast / (ema_slow + EPS)) - 1
