# Makalu

Data pipeline for market microstructure feature engineering and ML model training.

## Overview

Makalu is a comprehensive data pipeline that processes high-frequency market data (order books and trades) to generate machine learning features for scalping and microstructure-based trading strategies.

## Features

### Data Processing Stages

1. **MarketDataStage**: Processes raw market data from msgpack files
   - Trade data ingestion
   - OrderBook snapshots (10Hz for Binance)
   - Market price calculations
   - Order imbalance metrics (volume, weighted, depth)
   - Microprice computations
   - BookChange tracking

2. **TradeDerivedStage**: Generates trade-derived metrics
   - Candle bars (time-based OHLCV)
   - Heikin-Ashi bars
   - Range bars
   - Trade aggression metrics (multi-window: 1s, 3s, 5s)

3. **FeatureStage**: Builds ML-ready feature set
   - Aligns data from multiple sources
   - Computes derived features (microprice_offset_bps, vwap_offset_bps)
   - Rolling/EMA aggregates (1s, 3s, 5s windows)
   - Cross-window features
   - Interaction features
   - Outputs to Parquet format

### Feature Categories

- **Trade Aggression**: volume imbalance, count imbalance, VWAP, dominant size buckets
- **OrderBook State**: spread, volume imbalance, weighted imbalance, depth imbalance, microprice
- **BookChange**: net depth changes, pull/refill rates, best price movements
- **Derived Features**: offset calculations, rolling statistics, cross-window comparisons
- **Interaction Features**: sign agreement, spread-conditioned signals

## Installation

```bash
# Clone the repository
git clone git@github.com:potential-panda/Makalu.git
cd Makalu

# Install dependencies
pip install -e .
```

## Usage

```bash
# Run the full pipeline
python src/pipeline.py --market BINANCE:BTC_USDT --from-dt 20250101 --to-dt 20250107
```

### Pipeline Arguments

- `--market`: Market in format EXCHANGE:SYMBOL (e.g., BINANCE:BTC_USDT)
- `--from-dt`: Start date in YYYYMMDD format
- `--to-dt`: End date in YYYYMMDD format


## Documentation

See [doc/spec.md](doc/spec.md) for comprehensive specification including:
- TradeAggression implementation details
- OrderBook metrics
- BookChange tracking
- ML feature engineering guide
- Label computation strategies

Dataset reference files owned by Makalu:
- `doc/catalog.yaml` — dataset inventory, layer, grain, and partition columns
- `doc/dataset_definitions.md` — column-level dataset definitions

## Grafana Dashboards

Pre-configured dashboards in `grafana/`:
- `trade_aggression.json`: Trade flow visualization
- `order_book.json`: OrderBook state metrics
- `book_change.json`: Liquidity behavior
- `chart.json`: OHLCV charts

## License

MIT
