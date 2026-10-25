"""
Buy/Not-Buy Binary Labeling Library.

Labels each tick as:
  Buy (1)     — A long position entered at this tick would hit the profit-take
                barrier BEFORE the stop-loss barrier or timeout.
  Not-buy (0) — Stop-loss hit first, OR timeout elapsed with no barrier hit.

Barriers are dynamic, sized relative to the current spread:
  PT level = entry_mid * (1 + pt_multiplier * spread_pct_ema)
  SL level = entry_mid * (1 - sl_multiplier * spread_pct_ema)

PnL is computed for the long direction at the exit tick:
  buy_pnl = bid(exit_tick) - ask(entry_tick)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import jit

from .triple_barrier import ts_index

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BuyLabelConfig:
    """
    Configuration for buy/not-buy binary labeling.

    Attributes:
        pt_multiplier:          PT = entry_mid * (1 + pt_multiplier * spread_pct_ema)
        sl_multiplier:          SL = entry_mid * (1 - sl_multiplier * spread_pct_ema)
        timeout_sec:            Max hold time; timeout counts as not-buy (0).
        return_shift_sec:       Period (seconds) for return computation in volatility calc.
        volatility_lookback_sec: Rolling window for realised-volatility calculation.
        ema_span_sec:           EMA span for smoothing spread_pct_ema / volatility_ema.
        vol_resample_interval:  Resample interval for volatility calc (e.g. '1s').
    """

    pt_multiplier: float = 3.0
    sl_multiplier: float = 1.5
    timeout_sec: int = 90
    return_shift_sec: int = 90
    volatility_lookback_sec: int = 300
    ema_span_sec: int = 90
    vol_resample_interval: str = "1s"

    @property
    def param_id(self) -> str:
        """Unique parameter string for Parquet partitioning / model versioning."""
        return (
            f"buy"
            f"_{int(self.pt_multiplier * 100)}"
            f"_{int(self.sl_multiplier * 100)}"
            f"_{self.timeout_sec}"
            f"_{self.volatility_lookback_sec}"
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class BuyLabelResult:
    """
    Container for buy/not-buy labeling output.

    Attributes:
        labels_df:    DataFrame indexed by timestamp with columns
                      ``label`` (0/1) and ``buy_pnl`` (float).
        total_count:  Total labelled ticks.
        buy_count:    Number of Buy (1) labels.
        not_buy_count: Number of Not-buy (0) labels.
    """

    labels_df: pd.DataFrame
    total_count: int
    buy_count: int
    not_buy_count: int

    @property
    def buy_rate(self) -> float:
        return self.buy_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def avg_buy_pnl(self) -> float:
        if "buy_pnl" in self.labels_df.columns and self.total_count > 0:
            return float(self.labels_df["buy_pnl"].mean())
        return 0.0

    @property
    def avg_buy_pnl_when_buy(self) -> float:
        if "buy_pnl" not in self.labels_df.columns or self.buy_count == 0:
            return 0.0
        return float(self.labels_df.loc[self.labels_df["label"] == 1, "buy_pnl"].mean())


# ---------------------------------------------------------------------------
# Numba inner loop
# ---------------------------------------------------------------------------


@jit(nopython=True)
def _compute_buy_labels_numba(
    timestamps: np.ndarray,
    mid_prices: np.ndarray,
    ask_prices: np.ndarray,
    bid_prices: np.ndarray,
    pt_pct: np.ndarray,
    sl_pct: np.ndarray,
    timeout_ns: int,
):
    """
    Numba-optimised tick-level buy/not-buy labeling.

    For each entry tick i, scans forward until the first barrier hit or timeout:
      - mid >= PT level  → label 1 (buy), exit at tick j
      - mid <= SL level  → label 0 (not-buy), exit at tick j
      - timeout elapsed  → label 0 (not-buy), exit at last in-window tick

    buy_pnl = bid[exit] - ask[entry]  (long: buy at ask, exit at bid)

    Returns:
        labels:  int8 array  (0 = not-buy, 1 = buy)
        buy_pnl: float64 array
    """
    n = len(timestamps)
    labels = np.zeros(n, dtype=np.int8)
    buy_pnl = np.zeros(n, dtype=np.float64)

    for i in range(n):
        entry_mid = mid_prices[i]
        timeout_ts = timestamps[i] + timeout_ns

        if np.isnan(entry_mid) or np.isnan(pt_pct[i]) or np.isnan(sl_pct[i]):
            continue

        upper_barrier = entry_mid * (1.0 + pt_pct[i])
        lower_barrier = entry_mid * (1.0 - sl_pct[i])

        exit_j = -1

        for j in range(i + 1, n):
            if timestamps[j] > timeout_ts:
                break

            exit_j = j

            if mid_prices[j] >= upper_barrier:
                labels[i] = 1  # PT hit → buy
                break
            elif mid_prices[j] <= lower_barrier:
                # labels[i] stays 0 (SL hit → not-buy)
                break
        # else: loop exhausted with no barrier → labels[i] stays 0 (not-buy)

        if exit_j > i:
            buy_pnl[i] = bid_prices[exit_j] - ask_prices[i]

    return labels, buy_pnl


# ---------------------------------------------------------------------------
# Spread / barrier calculation
# ---------------------------------------------------------------------------


def calculate_spread_and_barriers(
    df: pd.DataFrame, config: BuyLabelConfig
) -> pd.DataFrame:
    """
    Compute ``spread_pct_ema``, ``volatility_ema``, ``pt_pct``, and ``sl_pct``
    for each tick.

    The input DataFrame must have a DatetimeIndex and columns:
    ``mid`` and ``spread`` (absolute bid–ask spread).

    PT and SL barriers are asymmetric:
      pt_pct = pt_multiplier * spread_pct_ema
      sl_pct = sl_multiplier * spread_pct_ema

    Args:
        df:     Tick-level DataFrame with DatetimeIndex.
        config: BuyLabelConfig.

    Returns:
        df with additional columns: spread_pct_ema, volatility_ema, pt_pct, sl_pct.
        Rows with NaN are dropped.
    """
    df = df.sort_index()

    dt = pd.Timedelta(config.vol_resample_interval).total_seconds()
    shift_periods = round(config.return_shift_sec / dt)
    lookback_periods = round(config.volatility_lookback_sec / dt)

    mid_resampled = df["mid"].resample(config.vol_resample_interval).last().ffill()
    spread_resampled = (
        df["spread"].resample(config.vol_resample_interval).last().ffill()
    )

    returns = mid_resampled.pct_change(periods=shift_periods)
    volatility = returns.rolling(window=lookback_periods, min_periods=1).std()
    spread_pct = spread_resampled / mid_resampled

    volatility_ema = volatility.ewm(span=config.ema_span_sec, adjust=False).mean()
    spread_pct_ema = spread_pct.ewm(span=config.ema_span_sec, adjust=False).mean()

    df["volatility_ema"] = volatility_ema.reindex(df.index, method="ffill")
    df["spread_pct_ema"] = spread_pct_ema.reindex(df.index, method="ffill")

    # Asymmetric barriers
    df["pt_pct"] = config.pt_multiplier * df["spread_pct_ema"]
    df["sl_pct"] = config.sl_multiplier * df["spread_pct_ema"]

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_buy_labels(df: pd.DataFrame, config: BuyLabelConfig) -> BuyLabelResult:
    """
    Compute buy/not-buy labels from a tick-level price DataFrame that already
    contains ``pt_pct`` and ``sl_pct`` columns (i.e. after
    ``calculate_spread_and_barriers``).

    Required columns: ``mid``, ``ask``, ``bid``, ``pt_pct``, ``sl_pct``.

    Args:
        df:     Tick-level DataFrame with DatetimeIndex.
        config: BuyLabelConfig (used for timeout_sec).

    Returns:
        BuyLabelResult with ``labels_df`` (columns: label, buy_pnl) and stats.
    """
    timeout_ns = int(pd.Timedelta(seconds=config.timeout_sec).total_seconds() * 1e9)

    timestamps = df.index.values.astype("datetime64[ns]").astype(np.int64)
    mid_prices = df["mid"].values.astype(np.float64)
    ask_prices = df["ask"].values.astype(np.float64)
    bid_prices = df["bid"].values.astype(np.float64)
    pt_pct = df["pt_pct"].values.astype(np.float64)
    sl_pct = df["sl_pct"].values.astype(np.float64)

    labels, buy_pnl = _compute_buy_labels_numba(
        timestamps, mid_prices, ask_prices, bid_prices, pt_pct, sl_pct, timeout_ns
    )

    labels_df = pd.DataFrame({"label": labels, "buy_pnl": buy_pnl}, index=df.index)

    label_counts = pd.Series(labels).value_counts()
    return BuyLabelResult(
        labels_df=labels_df,
        total_count=len(labels),
        buy_count=int(label_counts.get(1, 0)),
        not_buy_count=int(label_counts.get(0, 0)),
    )


def generate_buy_labels(
    df: pd.DataFrame,
    config: BuyLabelConfig,
    exclude_edge: bool = True,
) -> BuyLabelResult:
    """
    End-to-end labeling: ts conversion → spread/barrier calc → label computation.

    Expects a raw DataFrame with columns:
    ``ts`` (epoch seconds), ``mid``, ``ask``, ``bid``, ``spread``.

    Args:
        df:           Raw DataFrame (ts column will be converted to DatetimeIndex).
        config:       BuyLabelConfig.
        exclude_edge: Drop the last ``timeout_sec`` of ticks to prevent look-ahead
                      edge effects (last entries cannot have complete forward windows).

    Returns:
        BuyLabelResult.
    """
    df = ts_index(df)
    df = calculate_spread_and_barriers(df, config)
    LOG.info(f"generate_buy_labels: {len(df)} ticks after spread/barrier calc")

    result = compute_buy_labels(df, config)

    if exclude_edge and not result.labels_df.empty:
        cutoff = result.labels_df.index.max() - pd.Timedelta(seconds=config.timeout_sec)
        result.labels_df = result.labels_df.loc[result.labels_df.index <= cutoff]
        label_counts = result.labels_df["label"].value_counts()
        result = BuyLabelResult(
            labels_df=result.labels_df,
            total_count=len(result.labels_df),
            buy_count=int(label_counts.get(1, 0)),
            not_buy_count=int(label_counts.get(0, 0)),
        )

    LOG.info(
        f"generate_buy_labels: {result.total_count} labels — "
        f"Buy={result.buy_count} ({result.buy_rate:.1%}), "
        f"Not-buy={result.not_buy_count} ({1 - result.buy_rate:.1%})"
    )
    return result
