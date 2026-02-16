import logging
from typing import Dict

import pandas as pd

from .utils import EPS

LOG = logging.getLogger(__name__)


def _window_to_span_steps(index: pd.DatetimeIndex, window: str) -> int:
    """Convert a pandas time window string (e.g. '30s') into EWM span steps."""
    if len(index) < 2:
        return 2

    # Use median spacing to handle irregular ticks robustly.
    step = index.to_series().diff().dropna().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return 2

    window_td = pd.Timedelta(window)
    span_steps = int(round(window_td / step))
    return max(2, span_steps)


def build_trend_features(source_df: pd.DataFrame, features: Dict[str, pd.Series], trend_windows: list[str]) -> None:
    """Build price trend and momentum features (Section 6)."""
    LOG.info("Building trend features...")
    mid = source_df["mid"]

    for w in trend_windows:
        lookback = f"{w}"
        shifted_mid = mid.shift(1, freq=lookback).reindex(mid.index, method="ffill")
        features[f"tr_roc_{w}"] = (mid / (shifted_mid + EPS)) - 1

    span_fast = _window_to_span_steps(mid.index, trend_windows[0])
    span_slow = _window_to_span_steps(mid.index, trend_windows[-1])
    ema_fast = mid.ewm(span=span_fast, adjust=False).mean()
    ema_slow = mid.ewm(span=span_slow, adjust=False).mean()
    features[f"tr_ema_spread_{trend_windows[0]}_{trend_windows[-1]}"] = (ema_fast / (ema_slow + EPS)) - 1
