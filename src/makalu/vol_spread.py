"""The VolSpread parameterisations Makalu reads, and nothing about computing them.

VolSpread is Hase's: `hase derive VolSpread --execution-size ... --params H:L:S`
computes it, and `hase.derive.vol_spread.measure` is the one formula for spread
and volatility on a grid. What Makalu owns is which windows its activity curves
are built from, so they are named here once, in Hase's partition spelling.
"""

from hase.derive.vol_spread import param_id

# (horizon, lookback, smooth) in seconds, on a 1 s grid. Lookback is at least
# the horizon, so volatility is measured over several returns; smooth is
# shorter than lookback, so smoothing does not make it sluggish.
VOLSPREAD_PARAMS = [(60, 300, 120), (60, 900, 300), (180, 900, 300), (300, 900, 300)]
VOLSPREAD_PARAM_IDS = [
    param_id(1, h, lookback, smooth) for h, lookback, smooth in VOLSPREAD_PARAMS
]
