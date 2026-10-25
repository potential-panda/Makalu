"""The Spread and Volatility windows Makalu's activity curves are built from.

Both datasets are Hase's: `hase derive Volatility --execution-size X --params
H:L:S` and `hase derive Spread --execution-size X --params SPAN` compute them,
and `hase.derive.spread.measure` / `hase.derive.volatility.measure` are the one
formula for each. What Makalu owns is which windows its curves read, named here
once in Hase's partition spelling.
"""

from hase.derive import spread, volatility

STEP = 1

# (horizon, lookback, smooth) in seconds. Lookback is at least the horizon, so
# volatility is measured over several returns; smooth is shorter than lookback,
# so smoothing does not make it sluggish.
VOLATILITY_WINDOWS = [(60, 300, 120), (60, 900, 300), (180, 900, 300), (300, 900, 300)]

# Each volatility window is paired with the spread smoothed over its lookback,
# and each pair weighs the same in the curve: half volatility, half spread.
# That is why Spread g1s_ema900s appears three times.
ACTIVITY_SERIES = [
    (
        ("Volatility", volatility.param_id(STEP, h, lookback, smooth), "vol_bps_ema"),
        ("Spread", spread.param_id(STEP, lookback), "spread_bps_ema"),
    )
    for h, lookback, smooth in VOLATILITY_WINDOWS
]

SPREAD_SPANS = sorted({lookback for _, lookback, _ in VOLATILITY_WINDOWS})


def hase_params(dataset: str) -> list[str]:
    """The `--params` values that derive every window here, for `hase derive <dataset>`."""
    if dataset == "Spread":
        return [str(span) for span in SPREAD_SPANS]
    return [f"{h}:{lookback}:{smooth}" for h, lookback, smooth in VOLATILITY_WINDOWS]
