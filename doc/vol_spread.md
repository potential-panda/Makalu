# VolSpread

## Purpose

`VolSpread` measures whether a market has enough short-term movement to overcome execution cost.

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

The goal of `VolSpread` is therefore not only to observe current conditions, but also to support learning a daily or weekly tradability pattern.

## Source Data

`VolSpread` uses silver `MarketPrice` as its source.

`MarketPrice` is derived from order book snapshots and an `execution_size`. Its bid and ask represent effective execution prices for that size, not just top-of-book prices.

From `MarketPrice`:

```text
mid = (bid + ask) / 2
spread = ask - bid
spread_pct = spread / mid
spread_bps = spread_pct * 10000
```

Because `execution_size` changes the effective bid, ask, and spread, it must be part of the `VolSpread` partition key and parameter context.

## Calculation Model

The calculation should separate three concepts:

1. Base sampling granularity
2. Measurement horizon and lookback
3. Emission interval

### 1. Base Sampling Granularity

Normalize `MarketPrice` to 1-second bars before computing volatility and spread statistics:

```text
mid_1s = resample(MarketPrice.mid, 1s).last().ffill()
spread_1s = resample(MarketPrice.spread, 1s).last().ffill()
spread_pct_1s = spread_1s / mid_1s
```

This gives a regular time grid and avoids tying volatility to irregular order book update frequency.

### 2. Measurement Horizon and Lookback

There are two related but different time parameters:

| Parameter | Meaning |
|-----------|---------|
| `return_horizon_sec` | The price-change horizon relevant to the strategy, for example 30s or 60s |
| `lookback_sec` | The trailing window used to estimate average volatility and spread |

The return series should be calculated over the strategy horizon:

```text
return_horizon[t] = mid_1s[t] / mid_1s[t - return_horizon_sec] - 1
```

Raw volatility is then the rolling standard deviation of these horizon returns:

```text
volatility_raw[t] = rolling_std(return_horizon, window=lookback_sec)
```

Spread should also be averaged or smoothed over the lookback window, because instantaneous spread is noisy and the dataset is intended to learn stable patterns:

```text
spread_pct_ema[t] = EMA(spread_pct_1s, span=lookback_sec)
```

Volatility should be smoothed as well, so the ratio does not combine a noisy numerator with a smoothed denominator:

```text
volatility[t] = EMA(volatility_raw, span=smooth_sec)
```

Then:

```text
vol_spread_ratio[t] = volatility[t] / spread_pct_ema[t]
```

### 3. Emission Interval

The emission interval controls how often a row is written:

```text
emit_interval_sec = Y
```

The recommended approach is to compute from 1-second data, then emit every `Y` seconds. The emission interval should reduce storage volume, but should not change the underlying calculation.

## Recommended Parameters

Use explicit parameter names:

| Parameter | Description |
|-----------|-------------|
| `execution_size` | Order size used by `MarketPrice` to calculate effective bid and ask |
| `return_horizon_sec` | Horizon of the return used for volatility |
| `lookback_sec` | Rolling window used to estimate volatility and spread |
| `smooth_sec` | EMA smoothing span for volatility and spread-derived measures |
| `emit_interval_sec` | Output cadence |

Avoid overloading one parameter to mean both trading horizon and averaging window.

## Recommended Columns

`VolSpread` should store enough intermediate values to audit the calculation:

| Column | Definition |
|--------|------------|
| `ts` | Unix timestamp |
| `bid` | Effective bid execution price from `MarketPrice` |
| `ask` | Effective ask execution price from `MarketPrice` |
| `mid` | `(bid + ask) / 2` |
| `spread` | `ask - bid` |
| `spread_pct` | `spread / mid` |
| `spread_bps` | `spread_pct * 10000` |
| `volatility_raw` | Rolling std of horizon returns |
| `volatility` | Smoothed volatility |
| `spread_pct_ema` | Smoothed spread percentage |
| `vol_spread_ratio` | `volatility / spread_pct_ema` |

## Pattern Learning

`VolSpread` is the observation dataset. A separate derived dataset should learn daily or weekly patterns from it.

For each `param_id` and `execution_size`, aggregate recent `VolSpread` rows into time bins:

```text
day_of_week
second_of_day or time_bin
```

Recommended aggregate fields:

| Column | Meaning |
|--------|---------|
| `median_volatility` | Typical volatility for the bin |
| `median_spread_pct` | Typical spread for the bin |
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

## Strategy Usage

Strategies can use the observed or learned ratio to set parameters dynamically:

```text
if vol_spread_ratio is high:
    market has enough movement relative to cost

if vol_spread_ratio is low:
    cost is high relative to movement
```

This should be treated as a market regime filter, not a directional signal. It says whether the market is structurally tradable for a given execution size and horizon. It does not say whether the next trade should be long or short.
