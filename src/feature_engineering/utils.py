import pandas as pd

# Small epsilon to avoid division by zero
EPS = 1e-9


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safe division with epsilon to avoid division by zero."""
    return numerator / (denominator + EPS)


def compute_imbalance(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Compute normalized imbalance: (A - B) / (A + B).
    Returns values in range [-1, 1].
    """
    return (a - b) / (a + b + EPS)


def compute_imbalance_abs(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Compute normalized imbalance with absolute denominator: (A - B) / (|A| + |B|).
    Useful when A and B can be negative.
    Returns values in range [-1, 1].
    """
    return (a - b) / (a.abs() + b.abs() + EPS)


def compute_zscore(series: pd.Series, window: str) -> pd.Series:
    """
    Compute rolling z-score: (x - mean) / std.
    Returns values typically in range [-3, 3].
    """
    rolling_mean = series.rolling(window, closed="left").mean()
    rolling_std = series.rolling(window, closed="left").std()
    return (series - rolling_mean) / (rolling_std + EPS)


def compute_percentile(series: pd.Series, window: str) -> pd.Series:
    """
    Compute rolling percentile rank.
    Returns values in range [0, 1].
    """
    return series.rolling(window, closed="left").rank(pct=True)


def compute_bps_offset(price: pd.Series, reference: pd.Series) -> pd.Series:
    """
    Compute basis points offset: (price - reference) / reference * 10000.
    """
    return (price - reference) / (reference + EPS) * 10000
