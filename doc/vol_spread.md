# Spread and Volatility

## Purpose

Hase's `Spread` and `Volatility` datasets together measure whether a market has enough short-term movement to overcome execution cost. (They were one dataset, `VolSpread`, until it was split: see below.)

Spread is the immediate loss paid when an order is filled. Volatility over a horizon is the amount of price movement available over that same horizon. Together they describe whether a trade has enough room to become profitable:

```text
tradability = volatility / spread
```

If volatility is lower than spread, the market is structurally hard to trade profitably unless the directional signal is very strong. If volatility is much larger than spread, the market has enough movement to pay transaction cost, but direction prediction still matters.

Important caveat: volatility is not directional profit. A one-sigma volatility value equal to spread does not mean profit is easy. Practical filters should usually require `volatility / spread` to be greater than 1, for example 1.5 or 2.0, depending on the strategy horizon and hit rate.

## Assumptions

The working assumptions behind this dataset are:

1. Volatility and spread are market characteristics.
2. These characteristics often show daily and weekly waves caused by trader activity patterns, exchange popularity, market hours, and trading rules.
3. The pattern should be relatively stable, but should drift slowly over time.
4. Trading strategies can reference the learned pattern to set parameters dynamically.

The goal of these datasets is therefore not only to observe current conditions, but also to support learning a daily or weekly tradability pattern.

## Where it is computed

Hase computes both, from one function each -- `hase.derive.spread.measure` and
`hase.derive.volatility.measure` -- and the labellers in `src/labeling/` call those too.
Makalu keeps the design (this page) and the windows its curves read
(`src/makalu/activity_windows.py`):

```bash
hase derive Spread --market GMO:BTC_JPY --execution-size 0.002 --start 2026-06-01 --end 2026-06-30 \
  --params 300 --params 900
hase derive Volatility --market GMO:BTC_JPY --execution-size 0.002 --start 2026-06-01 --end 2026-06-30 \
  --params 60:300:120 --params 60:900:300 --params 180:900:300 --params 300:900:300
```

## Source Data

Both use silver `MarketPrice` at an `execution_size` as their source (or, with no size,
the top of the book, partition `execution_size=touch`). Its bid and ask are effective
execution prices for that size, not just top-of-book prices, so `execution_size` is part
of both partition keys.

## Calculation Model

### 1. The quote grid (code, not a dataset)

`MarketPrice` is put on a 1-second clock first: the last bid and ask in each second,
carried forward through a second with no update (`hase.derive.grid`). This gives a regular
time grid and avoids tying volatility to irregular order book update frequency. Each day
is read with the 8 hours before it, so windows at 00:00 JST are full.

The grid is not stored: it is a tenth of a second a day from `MarketPrice`, with nothing
to choose but the step. `hase.derive.grid.quote_grid` returns it (`ts`, `bid`, `ask`,
`mid`, `spread_bps`) to anyone who wants the mid on a clock.

```text
mid[t] = (bid[t] + ask[t]) / 2
spread_bps[t] = (ask[t] - bid[t]) / mid[t] * 1e4
```

### 2. Spread

Instantaneous spread is noisy, and the curves learn stable patterns, so `Spread` stores it
smoothed. One parameter, the span:

```text
spread_bps_ema[t] = EMA(spread_bps, span=span)
```

### 3. Volatility

Three parameters: the return `horizon` relevant to the strategy (30s, 60s), the `lookback`
over which it is estimated, and the `smooth` span, so the ratio does not combine a noisy
numerator with a smoothed denominator.

```text
return_horizon[t] = mid[t] / mid[t - horizon] - 1
vol_bps[t]        = rolling_std(return_horizon, window=lookback) * 1e4
vol_bps_ema[t]    = EMA(vol_bps, span=smooth)
rv_bps[t]         = sqrt(sum over lookback of log(mid[t] / mid[t-1])^2) * 1e4
```

### 4. The ratio

Both are in bps, so:

```text
tradability[t] = vol_bps_ema[t] / spread_bps_ema[t]
```

It is not stored: which spread span to divide by which volatility window is a choice,
and storing every pairing would be storing the choice.

## Why two datasets

`VolSpread` held everything in one file per window set: the grid's bid, ask, mid and
spread, the spread EMA, and the volatility. Only the volatility depended on all of its
parameters. The grid columns were identical in every window set's file, about 40% of
each file written four times, and the spread EMA was tied to the volatility's `lookback`,
so a spread could not be smoothed differently without also changing the volatility.

Now each dataset holds only what its own parameters make, and each parameter belongs to
one of them.

## Parameters

| Dataset | Parameter | Description |
|---------|-----------|-------------|
| both | `execution_size` | Order size used by `MarketPrice`, or `touch` |
| both | `step` | Grid interval, seconds (1) |
| `Spread` | `span` | EMA span of the spread |
| `Volatility` | `horizon` | Horizon of the return used for volatility |
| `Volatility` | `lookback` | Rolling window volatility is estimated over |
| `Volatility` | `smooth` | EMA span smoothing it |

`param_id` spells them: `g1s_ema300s` for Spread, `g1s_h60s_lb300s_ema120s` for Volatility.

## Columns

| Dataset | Column | Definition |
|---------|--------|------------|
| both | `ts` | Unix timestamp of the grid point |
| `Spread` | `spread_bps_ema` | EMA of the spread, bps |
| `Volatility` | `rv_bps` | Realized volatility over `lookback`, bps |
| `Volatility` | `vol_bps` | Rolling std of `horizon` returns over `lookback`, bps |
| `Volatility` | `vol_bps_ema` | `vol_bps` smoothed over `smooth`, bps |

## Pattern Learning

`Spread` and `Volatility` are the observation datasets. A separate derived dataset (`ActivityCurve`) learns daily or weekly patterns from them.

For each window and `execution_size`, aggregate recent rows into time bins:

```text
day_of_week
second_of_day or time_bin
```

Recommended aggregate fields:

| Column | Meaning |
|--------|---------|
| `median_vol_bps_ema` | Typical EMA-smoothed volatility for the bin |
| `median_spread_bps_ema` | Typical spread for the bin |
| `median_vol_spread_ratio` | Typical tradability ratio for the bin |
| `p25_vol_spread_ratio` | Lower quartile ratio |
| `p75_vol_spread_ratio` | Upper quartile ratio |
| `sample_count` | Number of observations in the bin |
| `trained_from` | Start of trailing training window |
| `trained_to` | End of trailing training window |

Median and quantiles are preferred over mean because order book data can contain spikes.

To handle slow drift, rebuild the pattern every `N` days using a trailing window:

```text
pattern_asof_date = D
training_window = [D - N days, D)
```

## Active/Inactive Window Simplification

Individual `bucket_index` cells in the learned pattern are noisy: splitting a year of
training data into two independent halves (e.g. first half of the year vs second half)
shows bin-level correlation around 0.83-0.95 between the halves, with mean differences
comparable in size to the signal itself. The overall day-of-week / time-of-day shape is
real, but any single 5-minute cell is not precise enough to schedule trading against
directly.

A more robust simplification is to collapse each day-type (weekday vs weekend) down to
two contiguous time-of-day groups -- active and inactive -- rather than trusting the raw
bin values:

1. Average the bin-level curve across the relevant day-type rows (e.g. all weekday rows)
   to get one representative day curve.
2. Threshold at that representative day's own mean score.
3. Take the longest contiguous run of hours above the threshold as "active"; everything
   else on that day-type is "inactive".

This throws away resolution but keeps only the structure that survives being checked
against half the training data independently: the *ratio* between the active and
inactive group means stays stable (within ~1%) across a chronological half/half split,
even though absolute activity levels drift over the training window.

Exchange-specific closures should be handled explicitly rather than left to the
threshold. A market reopening after a scheduled maintenance window commonly produces a
short volatility/spread spike that is an artifact of the reopen, not organic
tradability, and can distort a naive threshold if left in. The simplest fix is to mark
the whole day containing the closure (e.g. all of Saturday, if the exchange runs weekly
maintenance on Saturday) as inactive outright, rather than trying to carve out just the
affected hours.

Report each resulting time-period group in both UTC (the curve's native timezone) and
the exchange's local time -- local time is usually what actually explains the pattern
(e.g. a market's own night/day cycle, or overlap with another market's trading hours).

Example (GMO ETH_JPY, execution_size=0.1, trained on 2025-07-01 to 2026-06-20):

| Group | UTC | JST |
|-------|-----|-----|
| Weekday active | 13:00 - 21:00 | 22:00 - 06:00 |
| Weekday inactive | 21:00 - 13:00 | 06:00 - 22:00 |
| Weekend (Sat + Sun) | inactive | inactive |

The weekday active window lines up with the earlier finding that this market moves most
while the US session is running, which lands overnight in Japan. The whole weekend is
marked inactive rather than only excluding GMO's Saturday maintenance window (09:00-11:00
JST / 00:00-02:00 UTC), because weekend activity overall correlates poorly (~0.44-0.52)
with the average day, not just around the reopen.

## Strategy Usage

Strategies can use the observed or learned ratio to set parameters dynamically:

```text
if vol_spread_ratio is high:
    market has enough movement relative to cost

if vol_spread_ratio is low:
    cost is high relative to movement
```

This should be treated as a market regime filter, not a directional signal. It says whether the market is structurally tradable for a given execution size and horizon. It does not say whether the next trade should be long or short.
