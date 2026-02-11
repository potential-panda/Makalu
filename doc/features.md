# Feature Documentation

This document describes the features available for machine learning models in the Makalu pipeline.

---

## 1. Trade Aggression Features

Trade aggression features are derived from `PopulateAggressionStep` in `pipeline/trade_derived_steps.py`. They capture buy/sell pressure and trade flow dynamics over rolling time windows (e.g., 1s, 3s, 5s).

### 1.1 Raw Fields (per window)

| Field | Description |
|-------|-------------|
| `buy_qty` | Total buy volume in window |
| `sell_qty` | Total sell volume in window |
| `buy_count` | Number of buy trades |
| `sell_count` | Number of sell trades |

### 1.2 Pre-computed Derived Fields

These are already computed in `TradeAggressionBuilder`:

| Field | Formula | Description | Status |
|-------|---------|-------------|--------|
| `imbalance_qty` | buy_qty - sell_qty | Net volume imbalance | Intermediate (non-stationary) |
| `imbalance_count` | buy_count - sell_count | Net trade count imbalance | Intermediate (non-stationary) |
| `total_qty` | buy_qty + sell_qty | Total volume | Intermediate (non-stationary) |
| `total_count` | buy_count + sell_count | Total trade count | Intermediate (non-stationary) |
| `buy_sell_ratio` | buy_qty / sell_qty | Volume ratio | Unstable near zero |
| ~~`net_flow`~~ | (buy_qty - sell_qty) / total_qty | Normalized net flow | **Remove** (duplicate of vol_imbalance_ratio) |
| ~~`qty_per_sec`~~ | total_qty / window_sec | Volume velocity | **Remove** (use normalized version) |
| ~~`trades_per_sec`~~ | total_count / window_sec | Trade frequency | **Remove** (use normalized version) |
| ~~`buy_qty_per_sec`~~ | buy_qty / window_sec | Buy volume velocity | **Remove** (redundant with vol_imbalance_ratio) |
| ~~`sell_qty_per_sec`~~ | sell_qty / window_sec | Sell volume velocity | **Remove** (redundant with vol_imbalance_ratio) |
| `vwap` | Volume-weighted average price | Overall VWAP | Intermediate (need mid for offset) |
| `vwap_buy` | Buy-side VWAP | Average buy price | Intermediate (need mid for offset) |
| `vwap_sell` | Sell-side VWAP | Average sell price | Intermediate (need mid for offset) |
| ~~`avg_trade_size`~~ | total_qty / total_count | Average trade size | **Remove** (use normalized version) |
| `max_trade_size` | Maximum single trade size | Detects large trades | Normalize with z-score |
| `burst_flag` | Boolean burst detection | High activity spike | OK (already boolean) |
| `dominant_size_bucket` | Most common trade size bucket | Trade size distribution | Categorical |
| `dominant_bucket_count` | Count in dominant bucket | Bucket frequency | Non-stationary |
| `dominant_bucket_share` | Share of dominant bucket | Bucket concentration | OK (already normalized) |

> **Note**: Fields marked ~~strikethrough~~ should be removed or replaced with normalized versions. Fields marked "Intermediate" are used to compute normalized features but should not be used directly as ML features.

### 1.3 Additional Derived Features

These features can be computed from the raw fields:

#### Normalized Imbalances (scale-independent)

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `vol_imbalance_ratio` | (buy_qty - sell_qty) / (buy_qty + sell_qty) | Normalized directional pressure [-1, 1] |
| `count_imbalance_ratio` | (buy_count - sell_count) / (buy_count + sell_count) | Trade count bias [-1, 1] |

```python
vol_imbalance_ratio = (buy_qty - sell_qty) / (buy_qty + sell_qty + 1e-9)
count_imbalance_ratio = (buy_count - sell_count) / (buy_count + sell_count + 1e-9)
```

#### VWAP Features (Trade-Only)

VWAP features that can be computed from trade data alone (no order book mid required).

> **Note**: Raw `vwap`, `vwap_buy`, `vwap_sell` are absolute prices (non-stationary). For offset from mid, see Section 5 (Cross-Feature Interactions) which requires order book data.

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `vwap_buy_sell_spread_bps` | (vwap_buy - vwap_sell) / vwap * 1e4 | [0, ~2*spread] | Execution spread between buyers and sellers |

```python
# Spread between buy and sell execution prices (no mid required)
vwap_buy_sell_spread_bps = (vwap_buy - vwap_sell) / (vwap + 1e-9) * 10000
```

**Interpretation:**
- `vwap_buy_sell_spread_bps` small: Tight execution, liquid market
- `vwap_buy_sell_spread_bps` large: Wide execution spread, buyers paying significantly more than sellers receiving

#### Trade Size Features

Trade size features capture **who** is trading bigger and **whether** trades are large relative to recent history.

> **Note**: Raw `avg_buy_size` and `avg_sell_size` are non-stationary (scale varies by instrument and over time). Always use normalized versions.

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `size_asymmetry_norm` | (avg_buy - avg_sell) / (avg_buy + avg_sell) | [-1, 1] | Which side trades bigger |
| `avg_size_zscore` | (avg_size - rolling_mean) / rolling_std | ~[-3, 3] | Are trades unusually large/small |
| `avg_size_pct` | rolling percentile rank | [0, 1] | Trade size percentile |

```python
# Intermediate calculations (not features themselves)
avg_buy_size = buy_qty / (buy_count + 1e-9)
avg_sell_size = sell_qty / (sell_count + 1e-9)
avg_size = (buy_qty + sell_qty) / (buy_count + sell_count + 1e-9)

# Feature 1: Which side is trading bigger? (directional)
size_asymmetry_norm = (avg_buy_size - avg_sell_size) / (avg_buy_size + avg_sell_size + 1e-9)

# Feature 2: Are trades large compared to recent history? (magnitude, z-score)
rolling_mean = avg_size.rolling("5min").mean()
rolling_std = avg_size.rolling("5min").std()
avg_size_zscore = (avg_size - rolling_mean) / (rolling_std + 1e-9)

# Feature 3: Trade size percentile rank (magnitude, percentile)
avg_size_pct = avg_size.rolling("5min").rank(pct=True)
```

**Interpretation:**
- `size_asymmetry_norm > 0`: Buyers trading bigger clips than sellers
- `avg_size_zscore > 2`: Unusually large trades (institutional activity?)
- `avg_size_pct > 0.9`: Trade sizes in top 10% of recent history

#### Activity Level Features

Activity level features capture **how active** the market is compared to recent history.

> **Note**: Raw `qty_per_sec` and `trades_per_sec` are non-stationary (vary by instrument and time of day). Always use normalized versions. Directional activity (buy vs sell) is already captured by `vol_imbalance_ratio`.

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `qty_per_sec_zscore` | (qty_per_sec - μ) / σ | ~[-3, 3] | Volume velocity vs recent history |
| `trades_per_sec_zscore` | (trades_per_sec - μ) / σ | ~[-3, 3] | Trade frequency vs recent history |
| `qty_per_sec_pct` | rolling percentile | [0, 1] | Volume velocity percentile |
| `trades_per_sec_pct` | rolling percentile | [0, 1] | Trade frequency percentile |

```python
# Intermediate calculations (from TradeAggressionBuilder, not features)
qty_per_sec = total_qty / window_sec
trades_per_sec = total_count / window_sec

# Feature 1: Volume velocity z-score
qty_rolling_mean = qty_per_sec.rolling("5min").mean()
qty_rolling_std = qty_per_sec.rolling("5min").std()
qty_per_sec_zscore = (qty_per_sec - qty_rolling_mean) / (qty_rolling_std + 1e-9)

# Feature 2: Trade frequency z-score
trades_rolling_mean = trades_per_sec.rolling("5min").mean()
trades_rolling_std = trades_per_sec.rolling("5min").std()
trades_per_sec_zscore = (trades_per_sec - trades_rolling_mean) / (trades_rolling_std + 1e-9)

# Feature 3: Volume velocity percentile
qty_per_sec_pct = qty_per_sec.rolling("5min").rank(pct=True)

# Feature 4: Trade frequency percentile
trades_per_sec_pct = trades_per_sec.rolling("5min").rank(pct=True)
```

**Interpretation:**
- `qty_per_sec_zscore > 2`: Unusually high volume (news event? whale activity?)
- `trades_per_sec_pct > 0.9`: Trade frequency in top 10% of recent history
- `qty_per_sec_zscore > 2 & trades_per_sec_zscore < 1`: Large trades but not many (institutional)
- `trades_per_sec_zscore > 2 & qty_per_sec_zscore < 1`: Many small trades (retail or HFT)

#### Activity Concentration

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `buy_qty_share` | buy_qty / total_qty | Fraction of volume that is buys |
| `buy_count_share` | buy_count / total_count | Fraction of trades that are buys |
| `qty_count_divergence` | buy_qty_share - buy_count_share | Size vs count story difference |

```python
buy_qty_share = buy_qty / (total_qty + 1e-9)
buy_count_share = buy_count / (total_count + 1e-9)

# Divergence: do qty and count tell different stories?
# High divergence = few large trades vs many small trades
qty_count_divergence = buy_qty_share - buy_count_share
```

#### Cross-Window Features (momentum/acceleration)

These compare metrics across different time windows to detect momentum and acceleration in trade flow.

> **Note**: Use the normalized `vol_imbalance_ratio` ([-1, 1]) for cross-window comparisons, not raw `imbalance_qty` which is non-stationary.

**Momentum Features** - Is short-term pressure different from longer-term?

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `imbalance_momentum_1s_3s` | ratio_1s - ratio_3s | [-2, 2] | Short vs medium-term |
| `imbalance_momentum_1s_5s` | ratio_1s - ratio_5s | [-2, 2] | Short vs long-term |
| `imbalance_momentum_3s_5s` | ratio_3s - ratio_5s | [-2, 2] | Medium vs long-term |

**Direction Consistency** - Do windows agree on direction?

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `imbalance_sign_agree_1s_3s` | sign(ratio_1s) == sign(ratio_3s) | {0, 1} | Short/medium agreement |
| `imbalance_sign_agree_1s_5s` | sign(ratio_1s) == sign(ratio_5s) | {0, 1} | Short/long agreement |
| `imbalance_sign_agree_3s_5s` | sign(ratio_3s) == sign(ratio_5s) | {0, 1} | Medium/long agreement |
| `imbalance_sign_agree_all` | all three windows agree | {0, 1} | Full agreement |

**Magnitude Dynamics** - Is imbalance strengthening or weakening?

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `imbalance_strengthening_1s_3s` | \|ratio_1s\| > \|ratio_3s\| | {0, 1} | Short-term intensifying |
| `imbalance_strengthening_1s_5s` | \|ratio_1s\| > \|ratio_5s\| | {0, 1} | Building from baseline |
| `imbalance_weakening_1s_5s` | \|ratio_1s\| < \|ratio_5s\| | {0, 1} | Fading from baseline |

```python
# First, compute normalized imbalance ratio per window (these are already [-1, 1])
vol_imbalance_ratio_1s = (buy_qty_1s - sell_qty_1s) / (buy_qty_1s + sell_qty_1s + 1e-9)
vol_imbalance_ratio_3s = (buy_qty_3s - sell_qty_3s) / (buy_qty_3s + sell_qty_3s + 1e-9)
vol_imbalance_ratio_5s = (buy_qty_5s - sell_qty_5s) / (buy_qty_5s + sell_qty_5s + 1e-9)

# Momentum: difference between windows ([-2, 2])
imbalance_momentum_1s_3s = vol_imbalance_ratio_1s - vol_imbalance_ratio_3s
imbalance_momentum_1s_5s = vol_imbalance_ratio_1s - vol_imbalance_ratio_5s
imbalance_momentum_3s_5s = vol_imbalance_ratio_3s - vol_imbalance_ratio_5s

# Direction consistency: do windows agree? ({0, 1})
imbalance_sign_agree_1s_3s = (np.sign(vol_imbalance_ratio_1s) == np.sign(vol_imbalance_ratio_3s)).astype(int)
imbalance_sign_agree_1s_5s = (np.sign(vol_imbalance_ratio_1s) == np.sign(vol_imbalance_ratio_5s)).astype(int)
imbalance_sign_agree_3s_5s = (np.sign(vol_imbalance_ratio_3s) == np.sign(vol_imbalance_ratio_5s)).astype(int)
imbalance_sign_agree_all = (imbalance_sign_agree_1s_3s & imbalance_sign_agree_1s_5s).astype(int)

# Magnitude dynamics: is imbalance strengthening? ({0, 1})
imbalance_strengthening_1s_3s = (abs(vol_imbalance_ratio_1s) > abs(vol_imbalance_ratio_3s)).astype(int)
imbalance_strengthening_1s_5s = (abs(vol_imbalance_ratio_1s) > abs(vol_imbalance_ratio_5s)).astype(int)
imbalance_weakening_1s_5s = (abs(vol_imbalance_ratio_1s) < abs(vol_imbalance_ratio_5s)).astype(int)
```

**Interpretation:**
- `imbalance_momentum_1s_3s > 0`: Short-term more bullish than medium-term (impulse)
- `imbalance_sign_agree_all == 1`: All windows agree on direction (high conviction)
- `imbalance_strengthening_1s_5s == 1`: Imbalance magnitude increasing (trend building)

#### Signed Flow Indicators (categorical/boolean)

```python
# Strong buy signal: high buy imbalance + consistent direction + strengthening
strong_buy = (
    (vol_imbalance_ratio_1s > 0.3) &
    (imbalance_sign_agree_all == 1) &
    (imbalance_strengthening_1s_5s == 1)
)

# Strong sell signal: high sell imbalance + consistent direction + strengthening
strong_sell = (
    (vol_imbalance_ratio_1s < -0.3) &
    (imbalance_sign_agree_all == 1) &
    (imbalance_strengthening_1s_5s == 1)
)

# Exhaustion signal: imbalance fading while still directional
exhaustion = (
    (abs(vol_imbalance_ratio_5s) > 0.3) &  # Was strong
    (imbalance_weakening_1s_5s == 1) &      # Now fading
    (imbalance_sign_agree_1s_5s == 1)       # Same direction (not reversal)
)

# Reversal signal: short-term flipping against longer-term
reversal = (
    (abs(vol_imbalance_ratio_5s) > 0.3) &   # Strong longer-term
    (imbalance_sign_agree_1s_5s == 0)       # But short-term disagrees
)
```

### 1.4 Feature Summary Table

| Feature | Formula | Range | Use Case |
|---------|---------|-------|----------|
| `vol_imbalance_ratio` | (B-S)/(B+S) | [-1, 1] | Directional pressure |
| `count_imbalance_ratio` | (Bc-Sc)/(Bc+Sc) | [-1, 1] | Trade count bias |
| `vwap_buy_sell_spread_bps` | (vwap_buy-vwap_sell)/vwap * 1e4 | [0, ~2*spread] | Execution spread |
| `size_asymmetry_norm` | (avg_buy-avg_sell)/(avg_buy+avg_sell) | [-1, 1] | Which side trades bigger |
| `avg_size_zscore` | (avg_size - μ) / σ | ~[-3, 3] | Trade size magnitude (z-score) |
| `avg_size_pct` | rolling percentile | [0, 1] | Trade size magnitude (percentile) |
| `qty_per_sec_zscore` | (qty_per_sec - μ) / σ | ~[-3, 3] | Volume velocity (z-score) |
| `trades_per_sec_pct` | rolling percentile | [0, 1] | Trade frequency (percentile) |
| `qty_count_divergence` | B/(B+S) - Bc/(Bc+Sc) | [-1, 1] | Large vs small trade detection |
| `imbalance_momentum_1s_3s` | ratio_1s - ratio_3s | [-2, 2] | Short-term momentum |
| `imbalance_momentum_1s_5s` | ratio_1s - ratio_5s | [-2, 2] | Medium-term momentum |
| `imbalance_sign_agree_all` | all windows same sign | {0, 1} | Direction consistency |
| `imbalance_strengthening_1s_5s` | \|ratio_1s\| > \|ratio_5s\| | {0, 1} | Magnitude increasing |

---

## 2. Order Book State Features

Order book state features are derived from `PopulateBookStateStep` in `pipeline/market_data_steps.py`. They capture the instantaneous state of the order book including price levels, spread, and various imbalance metrics.

Source: `qate/trading/intent.py`

### 2.1 Basic Price Fields

| Field | Description |
|-------|-------------|
| `best_bid` | Best bid price |
| `best_ask` | Best ask price |
| `mid` | Mid price = (best_bid + best_ask) / 2 |
| `spread` | Spread = best_ask - best_bid |
| `spread_bps` | Spread in basis points = spread / mid * 10000 |

### 2.2 Volume Imbalance (`vol_imb_k`)

Measures the imbalance between bid and ask volume at the top k levels.

**Formula:**
```python
def compute_volume_imbalance(order_book, k):
    bid_volume = sum(level.amount for level in order_book.bids[:k])
    ask_volume = sum(level.amount for level in order_book.asks[:k])
    return (bid_volume - ask_volume) / (bid_volume + ask_volume)
```

| Field | k | Range | Interpretation |
|-------|---|-------|----------------|
| `vol_imb_1` | 1 | [-1, 1] | Top-of-book imbalance (most reactive) |
| `vol_imb_3` | 3 | [-1, 1] | Near-touch imbalance |
| `vol_imb_5` | 5 | [-1, 1] | Shallow book imbalance |
| `vol_imb_10` | 10 | [-1, 1] | Deeper book imbalance (more stable) |

**Interpretation:**
- `+1`: All volume on bid side (strong buy pressure)
- `-1`: All volume on ask side (strong sell pressure)
- `0`: Balanced book

### 2.3 Weighted Imbalance (`weighted_imb_k`)

Distance-weighted imbalance that gives more weight to levels closer to mid price.

**Formula:**
```python
def compute_weighted_imbalance(order_book, k):
    mid_price = (best_bid + best_ask) / 2

    weighted_bid = sum(
        level.amount * (1 / (1 + abs(mid_price - level.price)))
        for level in order_book.bids[:k]
    )
    weighted_ask = sum(
        level.amount * (1 / (1 + abs(level.price - mid_price)))
        for level in order_book.asks[:k]
    )
    return (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)
```

| Field | k | Range | Interpretation |
|-------|---|-------|----------------|
| `weighted_imb_1` | 1 | [-1, 1] | Same as vol_imb_1 (only one level) |
| `weighted_imb_3` | 3 | [-1, 1] | Weighted near-touch |
| `weighted_imb_5` | 5 | [-1, 1] | Weighted shallow book |
| `weighted_imb_10` | 10 | [-1, 1] | Weighted deeper book |

**Interpretation:**
- Emphasizes liquidity close to mid price
- Less sensitive to large orders far from touch

### 2.4 Depth Imbalance (`depth_imb_Xbps`)

Imbalance of volume within a price range from mid price (specified in basis points).

**Formula:**
```python
def compute_depth_imbalance(order_book, distance_rate):
    mid_price = (best_bid + best_ask) / 2
    price_distance = mid_price * distance_rate

    bid_volume = sum(
        level.amount for level in order_book.bids
        if mid_price - level.price <= price_distance
    )
    ask_volume = sum(
        level.amount for level in order_book.asks
        if level.price - mid_price <= price_distance
    )
    return (bid_volume - ask_volume) / (bid_volume + ask_volume)
```

| Field | Rate | Range | Interpretation |
|-------|------|-------|----------------|
| `depth_imb_10bps` | 0.001 | [-1, 1] | Within 10bps of mid |
| `depth_imb_50bps` | 0.005 | [-1, 1] | Within 50bps of mid |
| `depth_imb_100bps` | 0.01 | [-1, 1] | Within 100bps of mid |

**Interpretation:**
- Price-based rather than level-based
- Useful for comparing across instruments with different tick sizes

### 2.5 Microprice (`microprice_k`)

Volume-weighted mid price that adjusts for order book imbalance.

> **Note**: Raw `microprice_k` is an absolute price (non-stationary). Use `microprice_offset_bps_k` (see Section 2.6) as the actual feature.

**Formula:**
```python
def compute_microprice(order_book, k):
    bid_volume = sum(level.amount for level in order_book.bids[:k])
    ask_volume = sum(level.amount for level in order_book.asks[:k])

    return (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
```

| Field | k | Status |
|-------|---|--------|
| `microprice_1` | 1 | Intermediate (use offset_bps) |
| `microprice_3` | 3 | Intermediate (use offset_bps) |
| `microprice_5` | 5 | Intermediate (use offset_bps) |

**Interpretation:**
- When bid_volume > ask_volume: microprice > mid (tilted toward ask)
- When ask_volume > bid_volume: microprice < mid (tilted toward bid)
- Microprice is a better predictor of next trade price than mid

### 2.6 Additional Derived Features

#### Microprice Offset

```python
# Distance from mid to microprice in basis points
microprice_offset_bps_k = (microprice_k - mid) / mid * 10000
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `microprice_offset_bps_1` | (μ₁ - mid) / mid * 1e4 | (-spread_bps/2, spread_bps/2) | Directional tilt |
| `microprice_offset_bps_3` | (μ₃ - mid) / mid * 1e4 | Similar | 3-level tilt |

#### Imbalance Comparisons Across k

```python
# Difference between shallow and deep imbalance
imbalance_gradient = vol_imb_1 - vol_imb_5

# Divergence: top-of-book vs deeper book telling different stories
imbalance_divergence = abs(vol_imb_1 - vol_imb_10)
```

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `imbalance_gradient` | vol_imb_1 - vol_imb_5 | Shallow vs deep pressure |
| `imbalance_divergence` | abs(vol_imb_1 - vol_imb_10) | Book consistency |

#### Spread-Normalized Features

```python
# Normalize microprice offset by spread
microprice_offset_norm = microprice_offset_bps / spread_bps
```

#### Rolling Aggregates (computed in FeatureStep)

```python
# Rolling mean/std over time windows
vol_imb_1_mean_1s = vol_imb_1.rolling("1S").mean()
vol_imb_1_std_1s = vol_imb_1.rolling("1S").std()
spread_bps_mean_3s = spread_bps.rolling("3S").mean()
```

### 2.7 Feature Summary Table

| Feature | Formula | Range | Use Case |
|---------|---------|-------|----------|
| `vol_imb_k` | (Vb - Va) / (Vb + Va) | [-1, 1] | Book pressure |
| `weighted_imb_k` | Distance-weighted | [-1, 1] | Near-touch pressure |
| `depth_imb_Xbps` | Within X bps | [-1, 1] | Price-based depth |
| `microprice_offset_bps_k` | (μ - mid) / mid * 1e4 | ~[-spread/2, spread/2] | Directional tilt |
| `spread_bps` | spread / mid * 1e4 | [0, inf) | Liquidity cost |
| `imbalance_gradient` | vol_imb_1 - vol_imb_5 | [-2, 2] | Shallow vs deep |

> **Note**: Raw `microprice_k` is an absolute price (non-stationary). Use `microprice_offset_bps_k` instead.

---

## 3. Book Change Features

Book change features are derived from `PopulateBookChangeStep` in `pipeline/order_book_derived_steps.py`. They capture the dynamics of order book changes between snapshots, including depth changes, pulls (cancellations), and refills (new orders).

Source: `qate/trading/book_change.py`

### 3.1 Overview

BookChange computes **price-aligned diffs** between consecutive order book snapshots. Unlike simple depth comparison, it tracks changes at each price level to detect:
- Depth removal (pulls/cancellations)
- Depth addition (refills/new orders)
- Best level changes (price improvements)

### 3.2 Time Fields

| Field | Description |
|-------|-------------|
| `delta_sec` | Time elapsed since previous snapshot (seconds) |

### 3.3 Best Level Reset Flags

Boolean flags indicating if the best price changed.

| Field | Type | Interpretation |
|-------|------|----------------|
| `best_bid_reset_k` | bool | Best bid price changed (improvement or deterioration) |
| `best_ask_reset_k` | bool | Best ask price changed (improvement or deterioration) |

**Use Cases:**
- Detect price level transitions
- Filter noise during stable periods
- Identify aggressive order activity

### 3.4 Delta Depth Fields

Aggregate depth change across top-k levels.

> **Note**: Raw delta depth values are non-stationary (vary by instrument liquidity). Use normalized versions (Section 3.7) for ML features.

| Field | Description | Status |
|-------|-------------|--------|
| `delta_depth_bid_k` | Total bid depth change (current - previous) | Intermediate |
| `delta_depth_ask_k` | Total ask depth change (current - previous) | Intermediate |
| `delta_depth_bid_rate_k` | Bid depth change per second | Intermediate |
| `delta_depth_ask_rate_k` | Ask depth change per second | Intermediate |

**Formula:**
```python
delta_depth_bid = sum(curr_depth[price] - prev_depth[price] for all prices)
delta_depth_bid_rate = delta_depth_bid / delta_sec
```

**Interpretation:**
- Positive delta: Net depth added (more passive liquidity)
- Negative delta: Net depth removed (liquidity pulled or consumed)

### 3.5 Pull/Refill Rate Fields

Decomposition of depth changes into pulls (removals) and refills (additions).

> **Note**: Raw rates are non-stationary. Use normalized imbalances (`pull_imbalance_k`, `refill_imbalance_k`) or z-score versions for ML features.

| Field | Formula | Description | Status |
|-------|---------|-------------|--------|
| `pull_bid_rate_k` | max(-delta_depth_bid_rate, 0) | Rate of bid depth removal | Intermediate |
| `pull_ask_rate_k` | max(-delta_depth_ask_rate, 0) | Rate of ask depth removal | Intermediate |
| `refill_bid_rate_k` | max(delta_depth_bid_rate, 0) | Rate of bid depth addition | Intermediate |
| `refill_ask_rate_k` | max(delta_depth_ask_rate, 0) | Rate of ask depth addition | Intermediate |

**Interpretation:**
- High `pull_bid_rate`: Bid liquidity being removed (sellers aggressive or MMs pulling)
- High `refill_ask_rate`: Ask liquidity being added (sellers queuing passively)

### 3.6 k-Level Variants

Features are computed for multiple depth levels (e.g., k=1, 3, 5, 10):

| k | Focus | Sensitivity |
|---|-------|-------------|
| 1 | Top-of-book only | Most reactive, highest noise |
| 3 | Near-touch | Good balance |
| 5 | Shallow book | More stable |
| 10 | Deeper book | Most stable, slower signal |

### 3.7 Additional Derived Features

#### Net Depth Change (Imbalance)

```python
# Net depth change imbalance (non-stationary - don't use directly)
delta_depth_imbalance_k = delta_depth_bid_k - delta_depth_ask_k

# Normalized version (USE THIS)
delta_depth_imbalance_norm_k = (delta_depth_bid_k - delta_depth_ask_k) / (abs(delta_depth_bid_k) + abs(delta_depth_ask_k) + 1e-9)
```

| Feature | Formula | Range | Status |
|---------|---------|-------|--------|
| ~~`delta_depth_imbalance_k`~~ | Δbid - Δask | (-inf, inf) | **Remove** (non-stationary) |
| `delta_depth_imbalance_norm_k` | Normalized | [-1, 1] | ✓ Use this |

#### Pull/Refill Imbalance

```python
# Who's pulling more? Bids or asks?
pull_imbalance_k = (pull_bid_rate_k - pull_ask_rate_k) / (pull_bid_rate_k + pull_ask_rate_k + 1e-9)

# Who's refilling more?
refill_imbalance_k = (refill_bid_rate_k - refill_ask_rate_k) / (refill_bid_rate_k + refill_ask_rate_k + 1e-9)
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `pull_imbalance_k` | (Pb - Pa) / (Pb + Pa) | [-1, 1] | Which side losing liquidity faster |
| `refill_imbalance_k` | (Rb - Ra) / (Rb + Ra) | [-1, 1] | Which side gaining liquidity faster |

#### Activity Level Features (Normalized)

Raw activity rates are non-stationary. Use z-score or percentile versions to measure **how active** book changes are compared to recent history.

```python
# Intermediate calculations (not features)
total_pull_rate_k = pull_bid_rate_k + pull_ask_rate_k
total_refill_rate_k = refill_bid_rate_k + refill_ask_rate_k
total_activity_k = total_pull_rate_k + total_refill_rate_k

# Feature 1: Activity z-score (is book unusually active?)
activity_rolling_mean = total_activity_k.rolling("5min").mean()
activity_rolling_std = total_activity_k.rolling("5min").std()
total_activity_zscore_k = (total_activity_k - activity_rolling_mean) / (activity_rolling_std + 1e-9)

# Feature 2: Activity percentile
total_activity_pct_k = total_activity_k.rolling("5min").rank(pct=True)

# Feature 3: Pull rate z-score (is pulling unusually high?)
pull_rolling_mean = total_pull_rate_k.rolling("5min").mean()
pull_rolling_std = total_pull_rate_k.rolling("5min").std()
total_pull_zscore_k = (total_pull_rate_k - pull_rolling_mean) / (pull_rolling_std + 1e-9)

# Feature 4: Pull vs refill ratio (already somewhat normalized)
pull_refill_ratio_k = total_pull_rate_k / (total_refill_rate_k + 1e-9)
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `total_activity_zscore_k` | (activity - μ) / σ | ~[-3, 3] | Book activity vs recent history |
| `total_activity_pct_k` | rolling percentile | [0, 1] | Activity percentile |
| `total_pull_zscore_k` | (pull - μ) / σ | ~[-3, 3] | Pull intensity vs recent history |
| `pull_refill_ratio_k` | pull / refill | [0, inf) | Net liquidity drain (>1) or accumulation (<1) |

**Interpretation:**
- `total_activity_zscore_k > 2`: Unusually high book activity (volatility event?)
- `total_pull_zscore_k > 2`: Unusually high liquidity removal (panic or aggressive taking)
- `pull_refill_ratio_k > 1`: More pulling than refilling (liquidity draining)

#### Cross-k Features (Normalized)

Compare shallow vs deep book activity using imbalances (already normalized).

```python
# Shallow vs deep pull imbalance comparison
pull_imbalance_gradient = pull_imbalance_1 - pull_imbalance_5  # [-2, 2]

# Is pulling more one-sided at top-of-book vs deeper?
# Positive = top-of-book more bid-heavy pulling, deeper more balanced
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `pull_imbalance_gradient` | pull_imb_1 - pull_imb_5 | [-2, 2] | Shallow vs deep pull asymmetry |
| `refill_imbalance_gradient` | refill_imb_1 - refill_imb_5 | [-2, 2] | Shallow vs deep refill asymmetry |

### 3.8 Feature Summary Table

| Feature | Formula | Range | Use Case |
|---------|---------|-------|----------|
| `best_bid_reset_k` | Boolean | {0, 1} | Price level change |
| `best_ask_reset_k` | Boolean | {0, 1} | Price level change |
| `delta_depth_imbalance_norm_k` | (Δbid-Δask)/(|Δbid|+|Δask|) | [-1, 1] | Net flow direction |
| `pull_imbalance_k` | (Pb-Pa)/(Pb+Pa) | [-1, 1] | Pull asymmetry |
| `refill_imbalance_k` | (Rb-Ra)/(Rb+Ra) | [-1, 1] | Refill asymmetry |
| `total_activity_zscore_k` | (activity - μ) / σ | ~[-3, 3] | Book activity magnitude |
| `total_activity_pct_k` | rolling percentile | [0, 1] | Activity percentile |
| `total_pull_zscore_k` | (pull - μ) / σ | ~[-3, 3] | Pull intensity magnitude |
| `pull_refill_ratio_k` | pull / refill | [0, inf) | Liquidity drain ratio |
| `pull_imbalance_gradient` | pull_imb_1 - pull_imb_5 | [-2, 2] | Shallow vs deep asymmetry |

> **Note**: Raw rates (`delta_depth_bid_k`, `pull_bid_rate_k`, etc.) are non-stationary and should not be used directly as ML features. Use the normalized versions above.

---

## 4. Execution Price Features

Execution price features are derived from `PopulateMarketPriceStep` in `pipeline/market_data_steps.py`. They represent the **expected VWAP (Volume-Weighted Average Price)** for executing a specific order size, accounting for market depth and slippage.

Source: `qate/core/model.py` - `OrderBook.market_price()` and `_expected_execution_price()`

### 4.1 Concept: Execution Price vs Best Price

Unlike simple best bid/ask prices, execution prices answer the question:
> "If I want to buy/sell X units right now, what average price would I get?"

This accounts for:
- **Slippage**: Consuming multiple price levels for larger orders
- **Market Impact**: True cost of execution vs. displayed quotes
- **Depth Quality**: How much size is available at each level

### 4.2 Calculation Method

```python
def _expected_execution_price(order_levels, target_size):
    """
    Calculate VWAP for executing target_size across order book levels.

    Walks through order book levels, accumulating volume until target_size
    is filled, then returns the volume-weighted average price.
    """
    remaining = target_size
    notional = 0.0

    for level in order_levels:
        take = min(level.amount, remaining)
        notional += take * level.price
        remaining -= take

        if remaining <= 0:
            return notional / target_size

    return None  # Not enough liquidity
```

**Example:**
```
Order book asks:
  Level 1: 100.00 x 5 units
  Level 2: 100.10 x 10 units
  Level 3: 100.20 x 20 units

For target_size = 12:
  Take 5 @ 100.00 = 500.00
  Take 7 @ 100.10 = 700.70
  Total: 1200.70 / 12 = 100.058 (execution price)

Best ask was 100.00, but actual execution is 100.058 (5.8 bps slippage)
```

### 4.3 Raw Fields

Fields are computed for specific execution sizes (e.g., "1.0", "10.0", "100.0"):

> **Note**: Raw execution prices (`bid`, `ask`, `mid`, `spread`) are absolute values (non-stationary). Use normalized versions (bps, ratios, z-scores) for ML features.

| Field | Description | Status |
|-------|-------------|--------|
| `bid` | Expected execution price for selling target_size | Intermediate (non-stationary) |
| `ask` | Expected execution price for buying target_size | Intermediate (non-stationary) |
| `mid` | (bid + ask) / 2 | Intermediate (non-stationary) |
| `spread` | ask - bid | Intermediate (non-stationary) |
| `spread_bps` | spread / mid * 10000 | ✓ OK (normalized) |

### 4.4 Multi-Size Execution Prices

Different execution sizes reveal different aspects of market depth:

| Execution Size | Use Case | Sensitivity |
|----------------|----------|-------------|
| Small (e.g., 0.1 BTC) | Retail-sized orders | Close to best bid/ask |
| Medium (e.g., 1.0 BTC) | Typical trading size | Some slippage |
| Large (e.g., 10.0 BTC) | Institutional size | Significant slippage |

### 4.5 Additional Derived Features

#### Slippage Metrics

```python
# Slippage from best price (in bps)
slippage_bid_bps = (best_bid - exec_bid) / best_bid * 10000
slippage_ask_bps = (exec_ask - best_ask) / best_ask * 10000

# Total round-trip slippage
slippage_total_bps = slippage_bid_bps + slippage_ask_bps
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `slippage_bid_bps` | (best_bid - exec_bid) / best_bid * 1e4 | [0, inf) | Sell-side slippage |
| `slippage_ask_bps` | (exec_ask - best_ask) / best_ask * 1e4 | [0, inf) | Buy-side slippage |
| `slippage_total_bps` | slippage_bid + slippage_ask | [0, inf) | Round-trip cost |

#### Execution Spread vs Best Spread

```python
# How much wider is execution spread vs best spread? (both in bps for comparability)
spread_expansion_bps = exec_spread_bps - best_spread_bps

# Ratio version (already normalized)
spread_expansion_ratio = exec_spread_bps / (best_spread_bps + 1e-9)
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `spread_expansion_bps` | exec_spread_bps - best_spread_bps | [0, inf) | Additional cost for size (in bps) |
| `spread_expansion_ratio` | exec_spread / best_spread | [1, inf) | Multiplier on spread |

#### Slippage Magnitude Features (Normalized)

Raw slippage values vary by instrument and market conditions. Use z-score or percentile for magnitude features.

```python
# Intermediate calculations (not features themselves)
slippage_total_bps = slippage_bid_bps + slippage_ask_bps

# Feature 1: Slippage z-score (is slippage unusually high?)
slippage_rolling_mean = slippage_total_bps.rolling("5min").mean()
slippage_rolling_std = slippage_total_bps.rolling("5min").std()
slippage_zscore = (slippage_total_bps - slippage_rolling_mean) / (slippage_rolling_std + 1e-9)

# Feature 2: Slippage percentile
slippage_pct = slippage_total_bps.rolling("5min").rank(pct=True)

# Feature 3: Slippage asymmetry (already normalized)
slippage_asymmetry = (slippage_ask_bps - slippage_bid_bps) / (slippage_total_bps + 1e-9)
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `slippage_zscore` | (slippage - μ) / σ | ~[-3, 3] | Slippage vs recent history |
| `slippage_pct` | rolling percentile | [0, 1] | Slippage percentile |
| `slippage_asymmetry` | (ask_slip - bid_slip) / total | [-1, 1] | Which side is harder to execute |

#### Depth Quality Score

```python
# Inverse of slippage - higher = better depth
depth_quality = 1 / (slippage_total_bps + 1)
```

### 4.6 Comparison with Book State Features

| Aspect | Book State (best bid/ask) | Execution Price |
|--------|---------------------------|-----------------|
| **Size** | Top-of-book only | Accounts for target size |
| **Slippage** | None (ideal price) | Includes market impact |
| **Use Case** | Signal generation | Actual trading cost |
| **Update Freq** | Every tick | Every tick |

### 4.7 Trading Strategy Applications

#### Cost-Aware Signal Generation

```python
# Only trade when signal exceeds execution cost
signal_strength_bps = microprice_offset_bps  # From book state
trade_cost_bps = exec_spread_bps / 2  # Half spread to enter

net_expected_profit = signal_strength_bps - trade_cost_bps
should_trade = net_expected_profit > min_threshold
```

#### Optimal Size Selection

```python
# Find size where marginal slippage exceeds marginal profit
for size in [1, 5, 10, 20, 50]:
    marginal_slippage = slippage_bps[size] - slippage_bps[size-1]
    if marginal_slippage > expected_alpha_bps:
        optimal_size = size - 1
        break
```

#### Execution Quality Monitoring

```python
# Compare actual fill price vs expected execution price
execution_quality = (expected_exec_price - actual_fill_price) / expected_exec_price * 10000
# Positive = better than expected, Negative = worse (slippage)
```

### 4.8 Feature Summary Table

| Feature | Formula | Range | Use Case |
|---------|---------|-------|----------|
| `exec_spread_bps` | (exec_ask - exec_bid) / mid * 1e4 | [0, inf) | True trading cost |
| `slippage_bid_bps` | (best_bid - exec_bid) / best_bid * 1e4 | [0, inf) | Sell slippage |
| `slippage_ask_bps` | (exec_ask - best_ask) / best_ask * 1e4 | [0, inf) | Buy slippage |
| `slippage_zscore` | (slippage - μ) / σ | ~[-3, 3] | Slippage magnitude (normalized) |
| `slippage_pct` | rolling percentile | [0, 1] | Slippage percentile |
| `slippage_asymmetry` | (ask_slip - bid_slip) / total | [-1, 1] | Execution asymmetry |
| `spread_expansion_bps` | exec_spread_bps - best_spread_bps | [0, inf) | Size penalty |
| `spread_expansion_ratio` | exec_spread / best_spread | [1, inf) | Spread multiplier |
| `depth_quality` | 1 / (slippage + 1) | (0, 1] | Liquidity score |

> **Note**: Raw execution prices (`exec_bid`, `exec_ask`, `exec_mid`) are absolute prices (non-stationary) and should only be used as intermediate values for computing normalized features. Use `slippage_*_bps`, `slippage_zscore`, or `slippage_asymmetry` as ML features.

---

## 5. Cross-Feature Interactions

Cross-feature interactions combine signals from different data sources to create more robust and predictive features. Single-source features can be noisy or misleading; combining them often reveals higher-quality signals.

### 5.1 Trade Aggression × Order Book State

These interactions compare **what traders are doing** (trades) with **what they're showing** (order book).

#### Sign Agreement

Do trades and book tell the same story?

```python
# Trade imbalance direction matches book imbalance direction
sign_agree_trade_book = (
    (imbalance_qty_1s > 0) == (vol_imb_1 > 0)
).astype(int)

# Strong agreement: both signals strong and aligned
strong_agreement = (
    (abs(vol_imbalance_ratio_1s) > 0.3) &
    (abs(vol_imb_1) > 0.3) &
    (sign_agree_trade_book == 1)
).astype(int)
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `sign_agree_trade_book` | sign(trade_imb) == sign(book_imb) | {0, 1} | Confirmation signal |
| `strong_agreement` | Both strong + aligned | {0, 1} | High conviction signal |

**Interpretation:**
- `sign_agree = 1`: Trades confirm book pressure (momentum likely to continue)
- `sign_agree = 0`: Trades diverge from book (potential reversal or noise)

#### Divergence Signals

When trades and book disagree, something interesting is happening.

```python
# Use normalized imbalance ratios for stationarity
# Trade imbalance opposite to book imbalance
trade_book_divergence = (
    (vol_imbalance_ratio_1s > 0.1) & (vol_imb_1 < -0.2)  # Buying into ask-heavy book
) | (
    (vol_imbalance_ratio_1s < -0.1) & (vol_imb_1 > 0.2)   # Selling into bid-heavy book
)

# Magnitude of divergence (using normalized ratios, both in [-1, 1])
divergence_magnitude = vol_imbalance_ratio_1s * (-vol_imb_1)  # Positive when diverging
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `trade_book_divergence` | sign mismatch | {0, 1} | Aggressive traders fighting the book |
| `divergence_magnitude` | ratio_trade * (-vol_imb) | [-1, 1] | Strength of divergence (positive = diverging) |

**Use Case:** Divergence often precedes book imbalance flipping to match trade direction.

#### VWAP Offset Features (requires Order Book mid)

VWAP offset features measure where trades are executing relative to the order book mid price.

> **Note**: These features require joining trade aggression data with order book state data, as `mid` comes from the order book.

```python
# mid comes from order book
mid = (best_bid + best_ask) / 2  # From BookState

# VWAP offsets from mid (in basis points)
vwap_offset_bps = (vwap - mid) / mid * 10000
vwap_buy_offset_bps = (vwap_buy - mid) / mid * 10000
vwap_sell_offset_bps = (vwap_sell - mid) / mid * 10000
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `vwap_offset_bps` | (vwap - mid) / mid * 1e4 | ~[-spread, +spread] | Net execution tilt |
| `vwap_buy_offset_bps` | (vwap_buy - mid) / mid * 1e4 | ~[0, +spread] | How much buyers pay up |
| `vwap_sell_offset_bps` | (vwap_sell - mid) / mid * 1e4 | ~[-spread, 0] | How much sellers give up |

**Interpretation:**
- `vwap_offset_bps > 0`: Net buying pressure (trades tilted toward ask)
- `vwap_offset_bps < 0`: Net selling pressure (trades tilted toward bid)
- `vwap_buy_offset_bps` large: Buyers aggressive, paying up to get filled

#### VWAP vs Microprice Comparison

Compare where trades are executing vs where the book suggests fair value.

```python
# Microprice offset from mid (book state)
microprice_offset_bps = (microprice_1 - mid) / mid * 10000

# Agreement: both pointing same direction
vwap_microprice_agree = (vwap_offset_bps > 0) == (microprice_offset_bps > 0)

# Spread between the two signals
vwap_microprice_spread = vwap_offset_bps - microprice_offset_bps
```

| Feature | Interpretation |
|---------|----------------|
| `vwap_microprice_agree` | Trade execution confirms book tilt |
| `vwap_microprice_spread` | Trades ahead of book (positive) or behind (negative) |

---

### 5.2 Trade Aggression × Book Change

These interactions reveal **how the book reacts to trades**.

#### Trade Consumption Detection

Are trades consuming the book? Use normalized ratios for stationary features.

```python
# Heavy buying + ask depth dropping = trades eating through asks
# Use normalized vol_imbalance_ratio ([-1, 1]) and pull_imbalance ([-1, 1])
trade_consuming_asks = (
    (vol_imbalance_ratio_1s > 0.3) &    # Strong buy imbalance
    (pull_imbalance_1 < -0.3)            # More pulling on ask side
)

# Heavy selling + bid depth dropping = trades eating through bids
trade_consuming_bids = (
    (vol_imbalance_ratio_1s < -0.3) &   # Strong sell imbalance
    (pull_imbalance_1 > 0.3)             # More pulling on bid side
)

# Consumption alignment: trade direction matches pull direction
# Both values in [-1, 1], product in [-1, 1]
consumption_alignment = vol_imbalance_ratio_1s * (-pull_imbalance_1)
# Positive = trades aligned with consumption (momentum)
# Negative = trades opposite to consumption (absorption)
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `trade_consuming_asks` | buy + ask_pull | {0, 1} | Buy trades eating through asks |
| `trade_consuming_bids` | sell + bid_pull | {0, 1} | Sell trades eating through bids |
| `consumption_alignment` | trade_ratio * (-pull_imb) | [-1, 1] | Alignment of trade and consumption |

#### Passive Absorption

Book absorbing trades without price impact.

```python
# High trade imbalance but depth change opposite or neutral
# Use normalized delta_depth_imbalance_norm ([-1, 1])
absorption_bid = (
    (vol_imbalance_ratio_1s < -0.3) &       # Strong sell imbalance
    (delta_depth_imbalance_norm_1 >= 0)      # But bid depth holding or growing
)

absorption_ask = (
    (vol_imbalance_ratio_1s > 0.3) &        # Strong buy imbalance
    (delta_depth_imbalance_norm_1 <= 0)      # But ask depth holding or growing
)

# Absorption strength: how much is depth holding against trade pressure?
# Both in [-1, 1], product measures opposition strength
absorption_strength = vol_imbalance_ratio_1s * (-delta_depth_imbalance_norm_1)
# Positive = depth opposing trades (absorption)
# Negative = depth following trades (momentum)
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `absorption_bid` | sell + bid_hold | {0, 1} | Bids absorbing sell pressure |
| `absorption_ask` | buy + ask_hold | {0, 1} | Asks absorbing buy pressure |
| `absorption_strength` | trade_ratio * (-depth_imb) | [-1, 1] | Strength of absorption vs momentum |

**Use Case:** High absorption suggests hidden liquidity or strong passive interest.

---

### 5.3 Order Book State × Book Change

These interactions capture **book dynamics** - how the static picture is evolving.

#### Imbalance Momentum

Is book imbalance increasing or decreasing? Use normalized versions for stationarity.

```python
# Current imbalance vs normalized change imbalance
# Both in [-1, 1], product in [-1, 1]
imbalance_momentum = vol_imb_1 * delta_depth_imbalance_norm_1
# Positive = imbalance strengthening, Negative = weakening

# Imbalance acceleration using normalized refill/pull imbalances
# Both use imbalance ratios ([-1, 1])
imbalance_building = (vol_imb_1 > 0) & (refill_imbalance_1 > 0.2)  # Bid refills > ask refills
imbalance_fading = (vol_imb_1 > 0) & (pull_imbalance_1 > 0.2)      # Bid pulls > ask pulls
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `imbalance_momentum` | vol_imb * delta_imb_norm | [-1, 1] | Imbalance strengthening (+) or weakening (-) |
| `imbalance_building` | imb > 0 & refill_imb > 0 | {0, 1} | Imbalance being reinforced by refills |
| `imbalance_fading` | imb > 0 & pull_imb > 0 | {0, 1} | Imbalance being eroded by pulls |

#### Best Level Stability vs Imbalance

```python
# Strong imbalance but best levels stable (wall holding)
stable_wall_bid = (vol_imb_1 > 0.5) & (best_bid_reset_1 == 0)
stable_wall_ask = (vol_imb_1 < -0.5) & (best_ask_reset_1 == 0)

# Weak imbalance but best levels resetting (thin book)
unstable_book = (abs(vol_imb_1) < 0.2) & (best_bid_reset_1 | best_ask_reset_1)
```

---

### 5.4 Order Book State × Execution Price

These interactions compare **displayed liquidity** with **executable liquidity**.

#### Slippage vs Imbalance

Does book imbalance predict slippage?

```python
# Slippage asymmetry aligned with book imbalance
slippage_imbalance_align = (
    (vol_imb_1 > 0) & (slippage_ask_bps > slippage_bid_bps)  # Bid-heavy, harder to buy
) | (
    (vol_imb_1 < 0) & (slippage_bid_bps > slippage_ask_bps)  # Ask-heavy, harder to sell
)

# Slippage ratio vs imbalance
slippage_ratio = slippage_ask_bps / (slippage_bid_bps + 1e-9)
slippage_imbalance_corr = slippage_ratio * vol_imb_1
```

| Feature | Interpretation |
|---------|----------------|
| `slippage_imbalance_align` | Execution cost confirms book pressure |
| `slippage_imbalance_corr` | Positive = consistent, Negative = hidden liquidity |

#### Microprice vs Execution Mid

```python
# Difference between microprice (theoretical) and execution mid (practical)
microprice_exec_gap = (microprice_1 - exec_mid) / mid * 10000

# Positive = microprice suggests better price than executable
# Negative = can execute better than microprice suggests
```

| Feature | Interpretation |
|---------|----------------|
| `microprice_exec_gap` | Positive = microprice optimistic, Negative = hidden depth |

#### Depth Quality vs Spread

```python
# Spread-adjusted depth quality
depth_quality_per_spread = depth_quality / (spread_bps + 1e-9)

# Is spread justified by depth?
spread_depth_ratio = spread_bps / (slippage_total_bps + 1e-9)
# High ratio = wide spread but low slippage (good depth behind spread)
```

---

### 5.5 Trade Aggression × Execution Price

These interactions compare **trade activity** with **execution costs**.

#### Trade Intensity vs Slippage

Use normalized versions (z-scores, percentiles) for stationarity.

```python
# High trading activity during low slippage (good liquidity being used)
# Use normalized versions: trades_per_sec_pct (percentile) and slippage_pct (percentile)
efficient_trading = (trades_per_sec_pct > 0.7) & (slippage_pct < 0.3)

# High trading activity during high slippage (liquidity crisis)
stressed_trading = (trades_per_sec_pct > 0.7) & (slippage_pct > 0.7)

# Activity-slippage divergence: high activity but low slippage (or vice versa)
# Both in [0, 1], difference in [-1, 1]
activity_slippage_divergence = trades_per_sec_pct - slippage_pct
# Positive = activity high relative to slippage (efficient market)
# Negative = slippage high relative to activity (thin market)
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `efficient_trading` | high_activity & low_slip | {0, 1} | High activity with low cost |
| `stressed_trading` | high_activity & high_slip | {0, 1} | High activity straining liquidity |
| `activity_slippage_divergence` | activity_pct - slippage_pct | [-1, 1] | Efficiency measure |

#### VWAP vs Execution Price

```python
# Are trades executing better or worse than expected?
trade_execution_edge = (vwap_1s - exec_mid) / mid * 10000

# Positive = trades executing at better prices than current book suggests
# Negative = trades paying more than current book suggests
```

---

### 5.6 Multi-Way Interactions

Combining three or more feature sources. All inputs should be normalized/stationary.

#### Confirmation Score

How many signals agree on direction? Use only normalized features.

```python
# Count agreeing signals (all already in [-1, 1] range)
signals = [
    np.sign(vol_imbalance_ratio_1s),       # Trade flow (normalized)
    np.sign(vol_imb_1),                     # Book imbalance (normalized)
    np.sign(microprice_offset_bps_1),       # Microprice offset
    np.sign(delta_depth_imbalance_norm_1),  # Book change (normalized)
]

confirmation_score = sum(signals) / len(signals)  # [-1, 1]
signal_agreement = abs(confirmation_score)         # [0, 1], 1 = all agree
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `confirmation_score` | mean(sign(signals)) | [-1, 1] | Net directional signal |
| `signal_agreement` | abs(confirmation_score) | [0, 1] | Confidence level |

#### Regime Detection

Use normalized features and boolean conditions.

```python
# Trending regime: trades and book aligned, consuming depth
regime_trending = (
    (sign_agree_trade_book == 1) &
    (trade_consuming_asks | trade_consuming_bids) &
    (signal_agreement > 0.5)
)

# Mean-reverting regime: divergence with absorption
regime_mean_reverting = (
    (sign_agree_trade_book == 0) &
    (absorption_bid | absorption_ask) &
    (signal_agreement < 0.3)
)

# Volatile regime: high activity, low agreement
# Use normalized activity features
regime_volatile = (
    (trades_per_sec_pct > 0.8) &            # High trade activity (percentile)
    (total_activity_pct_1 > 0.8) &           # High book activity (percentile)
    (signal_agreement < 0.3)
)
```

| Feature | Range | Interpretation |
|---------|-------|----------------|
| `regime_trending` | {0, 1} | Momentum regime |
| `regime_mean_reverting` | {0, 1} | Mean reversion regime |
| `regime_volatile` | {0, 1} | High volatility regime |

#### Quality-Weighted Signal

```python
# Weight directional signal by confidence and liquidity
# All components are normalized/bounded
quality_weighted_signal = (
    confirmation_score *          # [-1, 1]
    signal_agreement *            # [0, 1]
    depth_quality *               # (0, 1]
    (1 / (spread_bps + 1))        # (0, 1] for reasonable spreads
)
# Result in approximately [-1, 1] for typical values
```

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `quality_weighted_signal` | conf * agree * quality * spread_factor | ~[-1, 1] | Composite signal with confidence |

---

### 5.7 Interaction Feature Summary Table

| Category | Feature | Range | Interpretation |
|----------|---------|-------|----------------|
| **Trade × Book** | `vwap_offset_bps` | ~[-spread, spread] | Trade execution vs mid |
| | `vwap_buy_offset_bps` | ~[0, spread] | How much buyers pay up |
| | `vwap_sell_offset_bps` | ~[-spread, 0] | How much sellers give up |
| | `sign_agree_trade_book` | {0, 1} | Direction confirmation |
| | `divergence_magnitude` | [-1, 1] | Divergence strength (normalized) |
| | `vwap_microprice_spread` | ~[-spread, spread] | Trade vs book fair value |
| **Trade × Change** | `consumption_alignment` | [-1, 1] | Trade-consumption alignment |
| | `absorption_strength` | [-1, 1] | Absorption vs momentum |
| | `trade_consuming_asks` | {0, 1} | Buying through asks |
| | `trade_consuming_bids` | {0, 1} | Selling through bids |
| **Book × Change** | `imbalance_momentum` | [-1, 1] | Imbalance trend |
| | `imbalance_building` | {0, 1} | Reinforcement signal |
| | `imbalance_fading` | {0, 1} | Erosion signal |
| **Book × Exec** | `microprice_exec_gap` | ~bps | Theory vs practice gap |
| | `slippage_imbalance_corr` | varies | Consistency check |
| **Trade × Exec** | `activity_slippage_divergence` | [-1, 1] | Efficiency measure |
| | `efficient_trading` | {0, 1} | High activity, low cost |
| | `stressed_trading` | {0, 1} | High activity, high cost |
| **Multi-Way** | `confirmation_score` | [-1, 1] | Net direction |
| | `signal_agreement` | [0, 1] | Confidence level |
| | `quality_weighted_signal` | ~[-1, 1] | Composite signal with confidence |
| | `regime_trending` | {0, 1} | Momentum regime |
| | `regime_mean_reverting` | {0, 1} | Mean reversion regime |
| | `regime_volatile` | {0, 1} | Volatility regime |

> **Note**: All interaction features use normalized/stationary inputs (imbalance ratios, bps offsets, percentiles). Avoid using raw values like `imbalance_qty`, `pull_rate`, or `trades_per_sec` in interactions.

---

## 6. Feature Engineering Guidelines

### 6.1 Stationarity Requirements

All ML features must be **stationary** - they should not vary systematically by instrument, time of day, or market conditions. Non-stationary features cause model drift and poor generalization.

#### Feature Categories by Stationarity

| Category | Status | Examples | Action |
|----------|--------|----------|--------|
| **Imbalance Ratios** | ✓ Stationary | vol_imb, pull_imbalance, vol_imbalance_ratio | Use directly |
| **Bps Offsets** | ✓ Stationary | microprice_offset_bps, vwap_offset_bps | Use directly |
| **Boolean/Categorical** | ✓ Stationary | best_bid_reset, burst_flag, sign_agree | Use directly |
| **Z-Scores** | ✓ Stationary | avg_size_zscore, slippage_zscore | Use directly |
| **Percentiles** | ✓ Stationary | trades_per_sec_pct, slippage_pct | Use directly |
| **Absolute Prices** | ✗ Non-stationary | mid, microprice, exec_bid | Convert to offset |
| **Raw Rates** | ✗ Non-stationary | qty_per_sec, pull_bid_rate | Use z-score/percentile |
| **Raw Counts/Volumes** | ✗ Non-stationary | buy_qty, imbalance_qty | Use ratios |

#### Normalization Patterns

```python
# Pattern 1: Imbalance ratio (for directional features)
feature = (A - B) / (A + B + 1e-9)  # Range: [-1, 1]

# Pattern 2: Bps offset (for price features)
feature = (price - reference) / reference * 10000  # Range: ~[-spread, spread]

# Pattern 3: Z-score (for magnitude features)
rolling_mean = raw.rolling("5min").mean()
rolling_std = raw.rolling("5min").std()
feature = (raw - rolling_mean) / (rolling_std + 1e-9)  # Range: ~[-3, 3]

# Pattern 4: Percentile (for relative rank features)
feature = raw.rolling("5min").rank(pct=True)  # Range: [0, 1]
```

### 6.2 Normalization Summary

- **Imbalances**: Already [-1, 1], no further normalization needed
- **Rates**: Use z-score or percentile (not raw values)
- **Prices**: Convert to bps offset from mid
- **Counts/Volumes**: Use ratios or percentiles
- **Magnitudes**: Use z-score or percentile for "how much" questions

### 6.3 Handling NaN/Inf

```python
# Replace inf with large values
df = df.replace([np.inf, -np.inf], np.nan)

# Forward-fill short gaps, drop remaining
df = df.ffill(limit=3).dropna()

# Or use default values
df = df.fillna(0)  # For imbalances
df = df.fillna(df.median())  # For rates
```

### 6.4 Feature Selection Priorities

1. **Start with normalized imbalances**: vol_imb, vol_imbalance_ratio, microprice_offset_bps
2. **Add dynamics**: pull_imbalance, refill_imbalance, imbalance_momentum
3. **Add activity levels**: total_activity_zscore, trades_per_sec_pct (NOT raw rates)
4. **Add interactions**: sign_agree_trade_book, confirmation_score
5. **Add execution features**: Only if modeling execution costs (slippage_asymmetry, spread_expansion_ratio)

### 6.5 Avoiding Lookahead Bias

- All features use only past/current data
- Rolling windows use `.rolling()` with `closed='left'`
- Cross-window comparisons (1s vs 3s) are both available at same timestamp

### 6.6 Feature Naming Conventions

| Suffix | Meaning | Range |
|--------|---------|-------|
| `_k` or `_k1`, `_k5` | Depth level (top k levels) | varies |
| `_1s`, `_3s`, `_5s` | Time window | varies |
| `_bps` | Basis points (normalized by price) | typically small |
| `_ratio` | Ratio of two values | [0, inf) or [-inf, inf] |
| `_imbalance`, `_imb` | Normalized imbalance | [-1, 1] |
| `_zscore` | Z-score normalized | ~[-3, 3] |
| `_pct` | Percentile rank | [0, 1] |
| `_norm` | Generic normalized version | varies |
| `_agree`, `_flag`, `_reset` | Boolean indicator | {0, 1} |

---

## 7. Backtest Module

The backtest module (`src/backtest/`) provides an event-driven backtesting framework for cross-exchange trading strategies. It processes market data from multiple exchanges in chronological order, calculates features in real-time, makes trading decisions using trained ML models, and tracks PnL.

### 7.1 Architecture Overview

```
Events (merged) → Simulator → FeatureCalculator (BINANCE)
                           → VolatilityTracker (GMO)
                           → PositionManager → TradeRecords
                           → SimulationResult
```

### 7.2 Components

| Component | File | Purpose |
|-----------|------|---------|
| `BacktestDataLoader` | `data_loader.py` | Load order books and trades from parquet stores |
| `EventMerger` | `event_merger.py` | Merge events from multiple sources chronologically |
| `FeatureCalculator` | `feature_calculator.py` | Calculate real-time features from BINANCE data |
| `VolatilityTracker` | `volatility_tracker.py` | Track volatility and spread from GMO data |
| `PositionManager` | `position_manager.py` | Handle trade entry/exit with triple barrier |
| `SimulatorV2` | `simulator.py` | Main simulation loop |
| `SimulationResult` | `result.py` | Results container with analysis methods |

### 7.3 Features Calculated in Real-Time

The `FeatureCalculator` computes features from streaming BINANCE data:

**Order Book Features:**
- `spread_bps`: Spread in basis points
- `microprice_vs_mid`: Microprice offset from mid
- `volume_imbalance_k`: Volume imbalance at top k levels
- `weighted_imbalance`: Distance-weighted imbalance
- `depth_imbalance_Xbps`: Depth within X bps of mid
- `depth_ratio_k`: Bid/total depth ratio

**Trade Aggression Features:**
- `aggression_imbalance_Xs`: Buy/sell volume imbalance over X seconds
- `trades_per_sec_Xs`: Trade frequency
- `qty_per_sec_Xs`: Volume velocity

**Book Change Features:**
- `pull_imbalance_k`: Which side losing liquidity faster
- `refill_imbalance_k`: Which side gaining liquidity faster
- `delta_depth_bid/ask_k`: Depth changes

### 7.4 Volatility Tracking

The `VolatilityTracker` calculates from GMO data:
- Rolling volatility using return standard deviation
- EMA-smoothed volatility and spread percentage
- Trade entry filter: `vol > spread * min_vol_spread_ratio`

### 7.5 Position Management

Triple barrier exit logic:
- **Profit-Take**: Price reaches `entry * (1 + barrier_pct)` for long
- **Stop-Loss**: Price reaches `entry * (1 - barrier_pct)` for long
- **Timeout**: Position held longer than `timeout_sec`

Barrier percentage: `barrier_pct = volatility_ema * barrier_multiplier`

### 7.6 Usage Example

```python
from backtest import (
    BacktestDataConfig, BacktestDataLoader, EventMerger,
    SimulatorConfig, SimulatorV2, FeatureConfig, VolatilityConfig, PositionConfig
)

# Configure
data_config = BacktestDataConfig(
    dt_range=DtRange("2025-01-15", "2025-01-16"),
    market_binance=Market(ExchangeName.BINANCE, Symbol.XRP_USDT),
    market_gmo=Market(ExchangeName.GMO, Symbol.XRP_JPY),
    parquet_root="/data/parquet"
)

sim_config = SimulatorConfig(
    model_path="model.pkl",
    feature_columns=[...],
    long_prediction=2,
    short_prediction=0,
)

# Load and merge
loader = BacktestDataLoader(data_config)
events = EventMerger.merge(
    loader.load_binance_orderbooks(),
    loader.load_binance_trades(),
    loader.load_gmo_orderbooks()
)

# Run simulation
result = SimulatorV2(sim_config).run(events)
result.print_summary()
result.plot_pnl()
```

See `notebooks/backtest_simulation.ipynb` for a complete example.
