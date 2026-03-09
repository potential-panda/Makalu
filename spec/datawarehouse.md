# Crypto Market Data Warehouse & LLM Research Interface

## 1. Overview

This document describes the architecture for a **research-oriented data warehouse** built on historical crypto market data.

The system integrates:

* Raw exchange data captured via WebSockets
* A reproducible Parquet-based data lake
* Analytical feature pipelines
* Strategy research datasets
* A controlled **LLM query interface**

The system supports:

* market microstructure research
* strategy development
* ML feature generation
* backtesting analysis
* interactive research via natural language

---

## 2. Design Goals

### Primary Goals

1. **Reproducibility**
2. **Separation of data maturity layers**
3. **Efficient analytical queries**
4. **Support for ML feature generation**
5. **LLM-assisted research exploration**

### Non-Goals

* Real-time trading execution
* Real-time warehouse serving
* Autonomous trading agents

---

## 3. High-Level Architecture

```
Exchange WebSockets
        │
        ▼
Raw msgpack archive (S3)
        │
        ▼
Makalu Fetch Pipeline
(msgpack → parquet)
        │
        ▼
Bronze Layer  /panda/makalu-data/bronze/
        │
        ▼
Silver Layer  /panda/makalu-data/silver/
        │
        ▼
Gold Layer    /panda/makalu-data/gold/
        │
        ▼
Feature Layer /panda/makalu-data/feature/
        │
        ▼
Research Layer /panda/makalu-data/research/
        │
        ▼
LLM Research Interface
```

---

## 4. [DONE] Source Data Archive

### Purpose

Preserve original exchange data with minimal loss.

### Storage

```
s3://market-data/
  exchange=<exchange>/
    symbol=<symbol>/
      date=<YYYYMMDD>/
        stream_A.msgpack
        stream_B.msgpack
```

### Capture Characteristics

* WebSocket ingestion
* Redundant connections (2 streams per market)
* Daily file rotation
* Immutable archive

### Properties

```
append-only
immutable
replayable
not queried directly
```

---

## 5. Bronze Layer

Bronze represents **raw structured datasets** derived directly from msgpack files.
No analytics, no derived metrics — only normalization required to write valid Parquet.

### Root

```
/panda/makalu-data/bronze/
```

### Directory Layout

```
dataset=Trade/
  exchange=BINANCE/
    symbol=XRP_USDT/
      date=20250702/
        data.parquet

dataset=OrderBook/
  exchange=BINANCE/
    symbol=XRP_USDT/
      date=20250702/
        data.parquet

dataset=BookChange/
  exchange=BINANCE/
    symbol=XRP_USDT/
      date=20250702/
        data.parquet

dataset=BookState/
  exchange=BINANCE/
    symbol=XRP_USDT/
      date=20250702/
        data.parquet
```

### Bronze Rules

```
no derived analytics
no rolling statistics
minimal normalization only
```

### Schemas

**Trade**
```
ts
side
price
size
```

**OrderBook**
```
ts
bid0_price  bid0_qty
...
bid19_price bid19_qty
ask0_price  ask0_qty
...
ask19_price ask19_qty
```

**BookChange**
```
ts
side
price
qty
```

**BookState**
```
ts
[same as OrderBook]
```

---

## 6. Silver Layer

Silver datasets provide **canonical market structure views** derived from Bronze.
Point-in-time metrics only — no rolling windows, no statistical signals.

### Root

```
/panda/makalu-data/silver/
```

### Directory Layout

```
dataset=MarketPrice/
  exchange=BINANCE/
    symbol=XRP_USDT/
      execution_size=150/
        date=20250702/
          data.parquet

dataset=TradeAggression/
  exchange=BINANCE/
    symbol=XRP_USDT/
      date=20250702/
        data.parquet
```

### Silver Rules

```
point-in-time only
no rolling statistics
no window-based signals
```

### Schemas

**MarketPrice**
```
ts
best_bid
best_ask
midprice
microprice
```

**TradeAggression**
```
ts
price
size
side
aggressor    (buyer / seller)
```

---

## 7. Gold Layer

Gold datasets contain **analytical primitives** — window-based and statistical signals
built from Silver. These are the building blocks for feature generation.

Spread metrics live here, not in Silver, because they involve market impact and
statistical context that depends on recent history.

### Root

```
/panda/makalu-data/gold/
```

### Directory Layout

```
dataset=Chart_CANDLE/
  exchange=BINANCE/
    symbol=XRP_USDT/
      param_id=<param_id>/
        date=20250702/
          data.parquet

dataset=LeadLag/
  market_a=BINANCE:XRP_USDT/
    market_b=GMO:XRP_JPY/
      param_id=<param_id>/
        date=20250702/
          data.parquet

dataset=VolSpread/
  exchange=BINANCE/
    symbol=XRP_USDT/
      execution_size=0.002/
        param_id=<param_id>/
          date=20250702/
            data.parquet
```

### Schemas

**Chart_CANDLE** (and other bar types)
```
ts
open
high
low
close
volume
tick_count
open_ts
close_ts
```

**LeadLag**
```
ts
lead_lag_correlation
lag_seconds
window
```

**VolSpread**
```
ts
realized_vol
spread
spread_bps
spread_pct
execution_cost
```

---

## 8. Feature Layer

Feature tables combine Silver and Gold metrics into **ML-ready datasets**.

### Root

```
/panda/makalu-data/feature/
```

### Directory Layout

```
dataset=Feature_Mid/
  exchange=BINANCE/
    symbol=XRP_USDT/
      date=20250702/
        data.parquet

dataset=Feature_NearHFT/
  exchange=BINANCE/
    symbol=XRP_USDT/
      date=20250702/
        data.parquet
```

### Purpose

```
model training
strategy signal evaluation
regime classification
```

---

## 9. Research Layer

Research datasets are **experiment outputs** — ephemeral and rebuildable.

### Root

```
/panda/makalu-data/research/
```

### Directory Layout

```
dataset=BacktestPnL/
  param_set_id=<param_set_id>/
    date=20260101/
      data.parquet

dataset=TripleBarrierLabel/
  exchange=BINANCE/
    symbol=XRP_USDT/
      param_id=<param_id>/
        date=20250702/
          data.parquet

dataset=BuyLabel_tmp/
  exchange=BINANCE/
    symbol=XRP_USDT/
      date=20250702/
        data.parquet
```

### Properties

```
not canonical
experiment-specific
rebuildable
```

---

## 10. Dataset Definitions

See companion document: **`dataset_definitions.md`**

This document formally defines every metric, feature, and label:
* exact formula
* grain (tick / second / bar)
* source dataset
* valid value range

This is the semantic layer the LLM relies on. Without it, terms like `spread`,
`spread_bps`, and `spread_pct` will be interpreted inconsistently across experiments.
All metric names used in queries must have an entry in `dataset_definitions.md`.

---

## 11. Metadata Catalog

A metadata catalog is required to support LLM queries.

### Schema

```
dataset_name
layer
description
grain
primary_timestamp
partition_columns
source_dataset
```

### Example Entry

```
dataset_name:      VolSpread
layer:             gold
grain:             1 second
description:       realized volatility and spread metrics per market
partition_columns: exchange, symbol, execution_size, param_id
source_dataset:    MarketPrice, Trade
```

---

## 12. Query Engine

Primary query engine: **DuckDB**

Advantages:

* direct Parquet support with Hive partition pruning
* fast analytical queries
* simple deployment
* suitable for local research

Example query:

```sql
SELECT
    date_trunc('minute', to_timestamp(ts)) AS minute,
    avg(spread_bps),
    avg(realized_vol)
FROM read_parquet('/panda/makalu-data/gold/dataset=VolSpread/**/*.parquet',
                  hive_partitioning=true)
WHERE exchange = 'BINANCE'
  AND symbol   = 'XRP_USDT'
  AND date BETWEEN '20250701' AND '20250731'
GROUP BY minute
ORDER BY minute
```

---

## 13. LLM Research Interface

The LLM system acts as a **research assistant**, not a compute engine.

### Workflow

```
User Question
     │
     ▼
Retrieve Metadata + dataset_definitions.md
     │
     ▼
Generate SQL Plan
     │
     ▼
Execute Query (DuckDB)
     │
     ▼
Return Result + Explanation
```

### Example Question

```
Show days where spread exceeded realized volatility during Tokyo hours.
```

LLM tasks:

```
identify relevant tables from catalog
construct SQL using dataset_definitions for correct column names
run query
summarize results
suggest follow-up analysis
```

---

## 14. LLM Guardrails

### Query Restrictions

```
read-only queries
no DDL operations
no data mutation
```

### Query Limits

```
maximum date range
maximum row output
execution timeout
```

### Validation

Before execution:

```
verify dataset exists in catalog
verify column names against dataset_definitions.md
prevent accidental cartesian joins
```

---

## 15. Integration with Existing Systems

### Makalu

Responsibilities:

```
bronze ingestion  (msgpack → parquet, written to /panda/makalu-data/bronze/)
silver derivation
gold metric generation
feature pipelines
```

### Potential-Panda Core

Provides reusable microstructure functions used by Makalu pipelines:

```
microprice
order imbalance
trade aggression
volatility metrics
lead-lag detection
```

### Himalaya

Produces strategy backtest outputs written to the research layer:

```
/panda/makalu-data/research/dataset=BacktestPnL/...
```

---

## 16. Key Principles

```
raw data is immutable
transformations are reproducible
metrics have consistent definitions (enforced by dataset_definitions.md)
research artifacts are isolated and rebuildable
LLM is an interface, not the compute engine
```
