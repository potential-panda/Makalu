from qate.core.model import ExchangeName, Market
from qate.core.symbol import Symbol

MIN_EXECUTION_SIZE = {
    Symbol.BTC_JPY: "0.001",
    Symbol.ETH_JPY: "0.05",
    Symbol.XRP_JPY: "50",
    Symbol.SOL_JPY: "0.3",
    Symbol.BTC_USDT: "0.001",
    Symbol.ETH_USDT: "0.05",
    Symbol.XRP_USDT: "50",
    Symbol.SOL_USDT: "0.3",
    Symbol.BTC_SPOT: "0.001",
    Symbol.ETH_SPOT: "0.05",
    Symbol.XRP_SPOT: "50",
}

EXECUTION_SIZE = {
    Symbol.BTC_JPY: "0.002",
    Symbol.ETH_JPY: "0.1",
    Symbol.XRP_JPY: "150",
    Symbol.SOL_JPY: "1",
    Symbol.BTC_USDT: "0.002",
    Symbol.ETH_USDT: "0.1",
    Symbol.XRP_USDT: "150",
    Symbol.SOL_USDT: "1",
    Symbol.BTC_SPOT: "0.002",
    Symbol.ETH_SPOT: "0.1",
    Symbol.XRP_SPOT: "150",
}

LEAD_LAG_PAIRS = {
    "btc": (Market(ExchangeName.BINANCE, Symbol.BTC_USDT), Market(ExchangeName.GMO, Symbol.BTC_JPY)),
    "eth": (Market(ExchangeName.BINANCE, Symbol.ETH_USDT), Market(ExchangeName.GMO, Symbol.ETH_JPY)),
    "xrp": (Market(ExchangeName.BINANCE, Symbol.XRP_USDT), Market(ExchangeName.GMO, Symbol.XRP_JPY)),
    "sol": (Market(ExchangeName.BINANCE, Symbol.SOL_USDT), Market(ExchangeName.GMO, Symbol.SOL_JPY)),
}


def find_pair_key(market: Market) -> str:
    for pair_key, (lead_market, lag_market) in LEAD_LAG_PAIRS.items():
        if market == lead_market or market == lag_market:
            return pair_key
    raise ValueError(f"Market {market.id} not found in lead-lag pairs.")
