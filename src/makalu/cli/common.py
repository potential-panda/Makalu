from pathlib import Path

from makalu import settings as sys_env

DATASETS: dict[str, tuple[str, list[str]]] = {
    # Bronze
    "Trade": ("bronze", ["exchange", "symbol"]),
    "OrderBook": ("bronze", ["exchange", "symbol"]),
    # Silver
    "BookChange": ("silver", ["exchange", "symbol"]),
    "BookState": ("silver", ["exchange", "symbol"]),
    "MarketPrice": ("silver", ["exchange", "symbol", "execution_size"]),
    "TradeAggression": ("silver", ["exchange", "symbol"]),
    # Gold
    "Chart_CANDLE": ("gold", ["exchange", "symbol", "param_id"]),
    "Chart_HEIKIN_ASHI": ("gold", ["exchange", "symbol", "param_id"]),
    "Chart_RANGE": ("gold", ["exchange", "symbol", "param_id"]),
    "Spread": ("gold", ["exchange", "symbol", "execution_size", "param_id"]),
    "Volatility": ("gold", ["exchange", "symbol", "execution_size", "param_id"]),
    "LeadLag": ("gold", ["market_a", "market_b", "param_id"]),
}


def get_dataset_root(layer: str, dataset: str) -> Path:
    return Path(sys_env.get_parquet_root_dir()) / layer / f"dataset={dataset}"
