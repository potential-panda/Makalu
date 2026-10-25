import logging

import pandas as pd

from .utils import EPS

LOG = logging.getLogger(__name__)


def build_volatility_features(
    source_df: pd.DataFrame, features: dict[str, pd.Series], windows: list[str]
) -> None:
    """Build volatility and spread dynamics features (Section 7)."""
    LOG.info("Building volatility features...")
    mid = source_df["mid"]
    spread_bps = source_df["spread_bps"]

    returns = mid.pct_change(fill_method=None)
    for w in windows:
        vol = returns.rolling(f"{w}").std()
        features[f"vo_realized_vol_{w}"] = vol
        features[f"vo_relative_spread_{w}"] = spread_bps / (vol * 10000 + EPS)
