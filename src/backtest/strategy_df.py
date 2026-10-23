"""
Loader for Input 1 (strategy_df) used by the Orion meta-regime filter.

Joins two backtest parquet stores:
  - Backtest_Bar:  one row per range-bar close, contains chart/SAR fields
  - Backtest_Pnl:  one row per closed trade, contains pnl_inc and equity

The resulting DataFrame has one row per range-bar close with columns:
  decision_ts   — index (DatetimeIndex, UTC)
  equity        — cumulative realized PnL (forward-filled from trade rows)
  pnl_increment — realized PnL at this bar; 0.0 if no trade was executed
  sar_direction — SAR direction at bar close: +1 (up) or -1 (down)
  trade_executed — True if a trade was closed at this bar close

See spec/orion_optimize_2.md §6.4 for full column definitions.
"""

import logging

import pandas as pd
from qate.core.model import Market
from makalu.storage.parquet_store import ParquetStore
from qate.util.dt_range import DtRange

LOG = logging.getLogger(__name__)

_SAR_UP_COL = "sar:up"


def load_strategy_df(
    parquet_store: ParquetStore,
    market: Market,
    strategy_name: str,
    param_id: str,
    dt_range: DtRange,
) -> pd.DataFrame:
    """Load and join Backtest_Bar and Backtest_Pnl into strategy_df.

    Args:
        parquet_store: ParquetStore instance pointing at the research layer.
        market:        Market the strategy traded on.
        strategy_name: Strategy name used when the backtest was run.
        param_id:      Parameter set ID used when the backtest was run.
        dt_range:      Date range to load.

    Returns:
        DataFrame indexed by decision_ts with columns:
        equity, pnl_increment, sar_direction, trade_executed.
        Rows before the first trade (equity unknown) are dropped.
    """
    bars_df = _load_bars(parquet_store, market, strategy_name, param_id, dt_range)
    pnl_df = _load_pnl(parquet_store, market, strategy_name, param_id, dt_range)

    if bars_df.empty:
        LOG.warning("load_strategy_df: Backtest_Bar is empty for the given range")
        return pd.DataFrame()

    result = bars_df.join(pnl_df, how="left")

    result["trade_executed"] = result["pnl_increment"].notna()
    result["pnl_increment"] = result["pnl_increment"].fillna(0.0)

    # Forward-fill equity from trade rows to all bar rows
    with pd.option_context("future.no_silent_downcasting", True):
        result["equity"] = result["equity"].ffill()
    result = result.infer_objects(copy=False)

    # Drop leading rows before the first trade (equity still NaN)
    result = result.dropna(subset=["equity"])

    LOG.info(
        f"load_strategy_df: {len(result)} bars | "
        f"trades={result['trade_executed'].sum()} | "
        f"equity range [{result['equity'].min():.2f}, {result['equity'].max():.2f}]"
    )
    return result[["equity", "pnl_increment", "sar_direction", "trade_executed"]]


def _load_bars(
    parquet_store: ParquetStore,
    market: Market,
    strategy_name: str,
    param_id: str,
    dt_range: DtRange,
) -> pd.DataFrame:
    store = parquet_store.create_backtest_bar(market, strategy_name, param_id)
    raw = store.read(dt_range)
    if raw.empty:
        return pd.DataFrame()

    raw.index = pd.to_datetime(raw["ts"], unit="s", utc=True)
    raw.index.name = "decision_ts"
    raw = raw.drop(columns=["ts"])
    raw = raw.sort_index()

    if _SAR_UP_COL not in raw.columns:
        LOG.warning(f"load_strategy_df: '{_SAR_UP_COL}' column not found in Backtest_Bar")
        raw["sar_direction"] = float("nan")
    else:
        raw["sar_direction"] = raw[_SAR_UP_COL].map({True: 1, False: -1})

    return raw[["sar_direction"]]


def _load_pnl(
    parquet_store: ParquetStore,
    market: Market,
    strategy_name: str,
    param_id: str,
    dt_range: DtRange,
) -> pd.DataFrame:
    store = parquet_store.create_pnl(market, strategy_name, param_id)
    raw = store.read(dt_range)
    if raw.empty:
        return pd.DataFrame()

    raw.index = pd.to_datetime(raw["ts"], unit="s", utc=True)
    raw.index.name = "decision_ts"
    raw = raw.drop(columns=["ts"])
    raw = raw.sort_index()

    return raw[["pnl_inc", "equity"]].rename(columns={"pnl_inc": "pnl_increment"})
