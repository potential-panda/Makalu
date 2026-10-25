"""
Runtime ActiveScore class.

Loads a pre-built weekly curve JSON file and provides a fast get_score(ts)
lookup and is_active(ts) gate for use in the trading engine.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

THRESHOLD_KEYS = ("std", "std1dev", "std2dev", "std3dev")


class ActiveScore:
    """
    Returns a normalized activity score in [0.0, 1.0] for any UTC timestamp,
    based on a pre-computed weekly volatility curve.

    Score 1.0 = historically most active period of the week.
    Score 0.0 = historically quietest period.

    Args:
        curve_path: Path to the JSON curve file produced by curve_builder.py.
        threshold_key: Which pre-computed threshold to use for is_active().
                       One of: "std", "std1dev", "std2dev", "std3dev".
    """

    def __init__(self, curve_path: str | Path, threshold_key: str = "std"):
        with open(curve_path) as f:
            payload = json.load(f)

        if threshold_key not in payload.get("thresholds", {}):
            raise ValueError(
                f"threshold_key '{threshold_key}' not found in curve file. "
                f"Available: {list(payload.get('thresholds', {}).keys())}"
            )

        self._bin_size_sec: int = payload["bin_size_sec"]
        self._scores: list[list[float]] = payload["scores"]
        self._n_bins: int = 86400 // self._bin_size_sec
        self._threshold: float = payload["thresholds"][threshold_key]["threshold"]

    def get_score(self, ts: float) -> float:
        """Return the activity score in [0.0, 1.0] for the given Unix timestamp (UTC seconds)."""
        dt = datetime.fromtimestamp(ts, tz=UTC)
        dow = dt.weekday()  # 0=Monday, 6=Sunday
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        bin_idx = seconds // self._bin_size_sec
        return self._scores[dow][bin_idx]

    def is_active(self, ts: float) -> bool:
        """Return True if the activity score for the given timestamp exceeds the threshold."""
        return self.get_score(ts) > self._threshold
