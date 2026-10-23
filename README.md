# Makalu

Market data pipeline and backtest orchestration for the **orion** algorithmic trading strategy. Trades BTC/ETH/XRP spot on GMO exchange using BINANCE as the signal market.

## What it does

- Downloads raw tick data (trades + order book) from S3
- Converts and stores it as a layered parquet data warehouse
- Derives microstructure metrics (MarketPrice, VolSpread, BookState, etc.)
- Runs backtests via the Himalaya engine
- Exposes a job queue API (port 9776) with daily data-sync and weekly scan+evict maintenance schedulers

## Data warehouse layers

```
bronze   Raw Trade + OrderBook tick data
silver   Derived per-tick metrics: MarketPrice, BookState, BookChange, TradeAggression
gold     Window-based signals: VolSpread, LeadLag, Chart_*
```

The bronze layer contains only `Trade` and `OrderBook`. All other datasets are
derived and live in silver, gold, feature, or research layers.

Data lives at `/panda/makalu-data/`. Raw msgpack source files at `/panda/market-data/`.

`/panda/makalu-data/manifest.db` tracks parquet file metadata for eviction:
path, row count, file size, and write timestamp. The API schedules weekly
maintenance that runs `python src/cli.py scan` and then
`python src/cli.py evict --apply`. Data that has not been scanned yet is
treated as fresh/protected because `evict` only deletes files registered in the
manifest. Makalu does not depend on OS file-read access time. The parquet disk
cap is stored in the manifest settings table and can be set with
`python src/cli.py evict --set-max-size 500GiB`.

## Data availability

Treat **2025-07-01** as the official start date for reliable market data. Older
raw or parquet files may exist, but may have collection or correctness problems.

Known global outage: **2026-04-02 through 2026-04-09 inclusive**. Market data
collectors crashed during this window. Bronze `Trade` and `OrderBook` reads
model the gap as available-but-empty so downstream callers see empty data
instead of needing their own outage special case.

## Installation

```bash
git clone git@github.com:potential-panda/Makalu.git
cd Makalu
pip install -e .
```

## CLI

```bash
python src/cli.py <subcommand> [args]

# Examples
python src/cli.py market-data --market GMO:BTC_JPY --days 3
python src/cli.py populate bronze --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07
python src/cli.py data-check --dataset Trade --exchange GMO --symbol BTC_JPY
python src/cli.py query --sql "SELECT ts, price FROM trade WHERE exchange='GMO' AND symbol='BTC_JPY' AND date='2026-01-01' LIMIT 10"
```

## Documentation

| Audience | File |
|----------|------|
| Human overview | this file |
| Claude Code entry point | `CLAUDE.md` |
| Codex / other agents entry point | `AGENT.md` |
| Data pipeline operations | `doc/data_pipeline.md` |
| SQL querying | `doc/querying.md` |
| Backtest workflow | `doc/backtest.md` |
| VolSpread plots + Grafana export | `doc/plot.md` |
| Local infra (Docker / InfluxDB) | `doc/local_infra.md` |
| Implementing new metrics | `doc/implement_metric.md` |
| Dataset column definitions | `doc/dataset_definitions.md` |
| Dataset catalog | `doc/catalog.yaml` |

## Framework dependencies

Uses `kamakuraquantlab-qate>=0.4.0` and `kamakuraquantlab-qate-env>=0.2.0`.
Install this checkout with `python -m pip install -e .`. No legacy core checkout
or extension of the `qate` namespace is required. Storage and microstructure
helpers are owned by this project; qate supplies event models and trading APIs.

Raw msgpack/S3 operations use the private `k2-data` package. Credentials and
trading environments use qate-env's `QATE_SECRET_DIR`, `QATE_ENV_ROOT`, and
`.qate.json` settings. Hosts using the old `~/secret` directory should set
`QATE_SECRET_DIR` to that directory. Data roots default to `/panda/market-data`
and `/panda/makalu-data`; override them with `MAKALU_MARKET_DATA_DIR` and
`MAKALU_PARQUET_ROOT_DIR`.
