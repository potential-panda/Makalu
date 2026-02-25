from qate.core.model import ExchangeName, Market, Symbol

from labeling.buy_label import BuyLabelConfig
from labeling.triple_barrier import TripleBarrierConfig
from pipeline.market_pair import EXECUTION_SIZE as EXECUTION_SIZE_DICT
from pipeline.market_pair import LEAD_LAG_PAIRS, find_pair_key

market = Market(ExchangeName.BINANCE, Symbol.XRP_USDT)
FEATURE_MARKET, LABEL_MARKET = LEAD_LAG_PAIRS.get(find_pair_key(market))
EXECUTION_SIZE = EXECUTION_SIZE_DICT.get(LABEL_MARKET.symbol)

MIN_VOL_SPREAD_RATIO = 2.0

TARGET_COLUMN = "label"


def setup():
    barrier_multiplier: float = 4.0
    return_shift_sec: int = 60
    timeout_sec: int = 60
    volatility_lookback_sec: int = 1800
    ema_span_sec: int = 1800
    vol_resample_interval: str = "1s"

    config = TripleBarrierConfig(
        barrier_multiplier=barrier_multiplier,
        return_shift_sec=return_shift_sec,
        timeout_sec=timeout_sec,
        volatility_lookback_sec=volatility_lookback_sec,
        ema_span_sec=ema_span_sec,
        vol_resample_interval=vol_resample_interval,
    )
    print(f"Feature market: {FEATURE_MARKET.id}")
    print(f"Label market:   {LABEL_MARKET.id}")
    print(f"Execution size: {EXECUTION_SIZE}")
    print(f"Label param_id: {config.param_id}")

    return (
        FEATURE_MARKET,
        LABEL_MARKET,
        EXECUTION_SIZE,
        TARGET_COLUMN,
        MIN_VOL_SPREAD_RATIO,
        config,
    )


def setup_buy_config():
    pt_multiplier: float = 3.0
    sl_multiplier: float = 1.5
    timeout_sec: int = 1800
    return_shift_sec: int = 1800
    volatility_lookback_sec: int = 600
    ema_span_sec: int = 90
    vol_resample_interval: str = "1s"

    config = BuyLabelConfig(
        pt_multiplier=pt_multiplier,
        sl_multiplier=sl_multiplier,
        timeout_sec=timeout_sec,
        return_shift_sec=return_shift_sec,
        volatility_lookback_sec=volatility_lookback_sec,
        ema_span_sec=ema_span_sec,
        vol_resample_interval=vol_resample_interval,
    )
    print(f"Feature market: {FEATURE_MARKET.id}")
    print(f"Label market:   {LABEL_MARKET.id}")
    print(f"Execution size: {EXECUTION_SIZE}")
    print(f"Label param_id: {config.param_id}")

    return (
        FEATURE_MARKET,
        LABEL_MARKET,
        EXECUTION_SIZE,
        TARGET_COLUMN,
        MIN_VOL_SPREAD_RATIO,
        config,
    )
