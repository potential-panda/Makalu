import logging
from typing import Dict

import pandas as pd

from .utils import EPS

LOG = logging.getLogger(__name__)


def build_volatility_features(source_df: pd.DataFrame, features: Dict[str, pd.Series]) -> None:
    """Build volatility and spread dynamics features (Section 7)."""
    LOG.info("Building volatility features...")
    mid = source_df["mid"]
    spread_bps = source_df["spread_bps"]

    returns = mid.pct_change(fill_method=None)
    for sec in [60, 120, 600]:
        features[f"vo_realized_vol_{sec}s"] = returns.rolling(f"{sec}s").std()

    vol_120s = features.get("vo_realized_vol_120s", returns.rolling("120s").std())
    features["vo_relative_spread_120s"] = spread_bps / (vol_120s * 10000 + EPS)
