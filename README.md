# Makalu

Market datalake operator for the **orion** algorithmic trading strategy. Trades BTC/ETH/XRP spot on GMO exchange using BINANCE as the signal market.

## What it does

- Reads bronze tick data (trades + order book) through Komachi; Kitakamakura produces it
- Derives microstructure metrics (BookState, LeadLag, curves, etc.; MarketPrice, Spread and Volatility come from Hase) into silver/gold
- Builds static report pages (`report`) and serves them read-only (`serve`, port 9776, public at `/makalu/`)

## Data warehouse layers

```
bronze   Raw Trade + OrderBook tick data (Kitakamakura's, read-only here)
silver   Derived per-tick metrics: MarketPrice, BookState, BookChange, TradeAggression
gold     Window-based signals: Spread, Volatility (Hase), LeadLag, Chart_*
```

The bronze layer contains only `Trade` and `OrderBook`. All other datasets are
derived and live in silver, gold, feature, or research layers.

The datalake root is Komachi's `ROOT_PATH` (`~/.kamakuraquantlab.env`), which is
`/panda/makalu-data/` on the warehouse host. Makalu never downloads, merges or
writes bronze: Kitakamakura does, and Makalu finds a market-day's file with
`komachi.bronze.file`.

`<root>/manifest.db` tracks Makalu's parquet file metadata (every layer except
bronze) for eviction: path, row count, file size, and write timestamp. Run
`python src/cli.py scan` and then `python src/cli.py evict` by hand; nothing
schedules them. Data that has not been scanned yet is
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
hase derive MarketPrice --market GMO:BTC_JPY --execution-size 0.002 --start 2026-01-01 --end 2026-01-07   # silver MarketPrice comes from Hase
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
| Report site | `doc/report.md` |
| SQL querying | `doc/querying.md` |
| Volatility / Spread plots + Grafana export | `doc/plot.md` |
| Local infra (Docker / InfluxDB) | `doc/local_infra.md` |
| Implementing new metrics | `doc/implement_metric.md` |
| Dataset column definitions | `doc/dataset_definitions.md` |
| Dataset catalog | `doc/catalog.yaml` |

## Framework dependencies

Uses `kamakuraquantlab-qate>=0.4.0`, `kamakuraquantlab-qate-env>=0.2.0` and
`kamakuraquantlab-komachi>=0.8.0`.
Install this checkout with `python -m pip install -e .`. No legacy core checkout
or extension of the `qate` namespace is required. Storage and microstructure
helpers are owned by this project; qate supplies event models and trading APIs.

`metrics-data` uses the private `k2-data` package for S3 transfer. Credentials
and trading environments use qate-env's `QATE_SECRET_DIR`, `QATE_ENV_ROOT`, and
`.qate.json` settings. Hosts using the old `~/secret` directory should set
`QATE_SECRET_DIR` to that directory. The datalake root comes from Komachi
(`komachi.data_root()`); change it with `ROOT_PATH` in `~/.kamakuraquantlab.env`.
