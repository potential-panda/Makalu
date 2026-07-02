import logging
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .utils import EPS

LOG = logging.getLogger(__name__)


@dataclass
class VolSpreadConfig:
    return_horizon_sec: int
    lookback_sec: int
    smooth_sec: int
    emit_interval: str = "1s"
    sample_interval: str = "1s"

    @property
    def param_id(self) -> str:
        params = [
            self.return_horizon_sec,
            self.lookback_sec,
            self.smooth_sec,
            self.emit_interval,
        ]
        if self.sample_interval != "1s":
            params.append(self.sample_interval)
        return f"vs_{'_'.join(map(str, params))}"


def build_volatility_features(source_df: pd.DataFrame, features: Dict[str, pd.Series], windows: List[str]) -> None:
    """Build volatility and spread dynamics features (Section 7)."""
    LOG.info("Building volatility features...")
    mid = source_df["mid"]
    spread_bps = source_df["spread_bps"]

    returns = mid.pct_change(fill_method=None)
    for w in windows:
        vol = returns.rolling(f"{w}").std()
        features[f"vo_realized_vol_{w}"] = vol
        features[f"vo_relative_spread_{w}"] = spread_bps / (vol * 10000 + EPS)
