from qate.core.model import ExchangeName, Market, Symbol

from triple_barrier.triple_barrier import TripleBarrierConfig

FEATURE_MARKET = Market(ExchangeName.BINANCE, Symbol.XRP_USDT)
LABEL_MARKET = Market(ExchangeName.GMO, Symbol.XRP_JPY)
EXECUTION_SIZE = "150"
TARGET_COLUMN = "label"


def setup():
    # Markets
    barrier_multiplier: float = 2.0
    return_shift_sec: int = 30
    timeout_sec: int = 30
    min_vol_spread_ratio: float = 2.0
    volatility_lookback_sec: int = 900
    ema_span_sec: int = 60
    vol_resample_interval: str = "1s"
    label_resample_interval: str = "100ms"

    config = TripleBarrierConfig(
        barrier_multiplier=barrier_multiplier,
        return_shift_sec=return_shift_sec,
        timeout_sec=timeout_sec,
        min_vol_spread_ratio=min_vol_spread_ratio,
        volatility_lookback_sec=volatility_lookback_sec,
        ema_span_sec=ema_span_sec,
        vol_resample_interval=vol_resample_interval,
        label_resample_interval=label_resample_interval,
    )
    print(f"Feature market: {FEATURE_MARKET.id}")
    print(f"Label market:   {LABEL_MARKET.id}")
    print(f"Label param_id: {config.param_id}")

    return FEATURE_MARKET, LABEL_MARKET, EXECUTION_SIZE, TARGET_COLUMN, config
