from qate.core.model import ExchangeName, Market
from qate.core.symbol import Symbol

TARGET_SIZE = {
    Symbol.BTC_JPY: "0.002",
    Symbol.ETH_JPY: "0.1",
    Symbol.XRP_JPY: "150",
    Symbol.BTC_USDT: "0.002",
    Symbol.ETH_USDT: "0.1",
    Symbol.XRP_USDT: "150",
}

LEAD_LAG_PAIRS = {
    "btc": (Market(ExchangeName.BINANCE, Symbol.BTC_USDT), Market(ExchangeName.GMO, Symbol.BTC_JPY)),
    "eth": (Market(ExchangeName.BINANCE, Symbol.ETH_USDT), Market(ExchangeName.GMO, Symbol.ETH_JPY)),
    "xrp": (Market(ExchangeName.BINANCE, Symbol.XRP_USDT), Market(ExchangeName.GMO, Symbol.XRP_JPY)),
}


def find_pair_key(market: Market) -> str:
    for pair_key, (lead_market, lag_market) in LEAD_LAG_PAIRS.items():
        if market == lead_market or market == lag_market:
            return pair_key
    raise ValueError(f"Market {market.id} not found in lead-lag pairs.")
