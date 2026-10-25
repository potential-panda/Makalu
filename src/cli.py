#!/usr/bin/env python3
"""
Makalu data warehouse CLI.

PURPOSE
-------
Derive and manage the silver/gold layers of the parquet datalake, which stores
crypto market microstructure data. The datalake root is Komachi's ROOT_PATH
(komachi.data_root(), /panda/makalu-data/ on the warehouse host).

DATA FLOW (must be populated in this order)
-------------------------------------------
  bronze/  Trade, OrderBook            ← written by Kitakamakura; read via Komachi
      │
      ▼  populate BookChange / BookState / TradeAggression / MarketPrice
  silver/  derived per-tick metrics    ← requires bronze to exist first
      │
      ▼  hase derive Spread / Volatility; populate LeadLag / Chart_RANGE / ...
  gold/    Spread, Volatility, LeadLag, Charts  ← requires silver/MarketPrice to exist first

Makalu does not download, merge, or write bronze.

SUBCOMMANDS
-----------
  data-check    READ-ONLY.  Inspect what parquet data is on disk. Safe to run anytime.
  populate      WRITE.      Derive silver/gold parquet files. Skips writes whose
                target data.parquet already exists; use --recreate to overwrite.
  metrics-data  WRITE.      Download 5-min trading metrics from S3 and push to InfluxDB.
                Idempotent. Looks back up to 28 days for missing intervals.
  scan          WRITE.      Register existing non-bronze parquet files into the manifest
                database. Safe to run manually after populate.
  evict         DESTRUCTIVE. If Makalu's parquet usage (all layers but bronze) exceeds
                the configured cap, delete manifest-tracked files older than 6 months.
                Never touches bronze. Always dry-run by default; requires --apply.
  grafana-export WRITE.     Push a date range of parquet data to InfluxDB and
                return a Grafana dashboard URL. JSON output: {grafana_url, ...}.
  report        WRITE.      Build static HTML reports (e.g. 'report leadlag') into the site
                folder, ~/env/LUIGI_MARIO/site. Writes nothing to the warehouse.
  serve         READ-ONLY.  Serve the site folder as static files on 127.0.0.1:9776,
                public as http://<ip>:7700/makalu/ through the pisces-01 tunnel.

TYPICAL WORKFLOW
----------------
  1. Check what bronze data exists (gaps are Kitakamakura's to fill):
       python cli.py data-check --dataset Trade --exchange GMO --symbol BTC_JPY

  2. Derive silver MarketPrice with Hase, then populate what depends on it:
       hase derive MarketPrice --market GMO:BTC_JPY --execution-size 0.002 --start 2026-01-01 --end 2026-01-07
       hase derive Spread --market GMO:BTC_JPY --execution-size 0.002 --start 2026-01-01 --end 2026-01-07 \
          --params 300 --params 900
       hase derive Volatility --market GMO:BTC_JPY --execution-size 0.002 --start 2026-01-01 --end 2026-01-07 \
          --params 60:300:120 --params 60:900:300 --params 180:900:300 --params 300:900:300

  3. Update manifest after populating:
       python cli.py scan

  4. Configure and periodically review parquet cleanup:
       python cli.py evict --set-max-size 500GiB
       python cli.py evict           # dry-run: shows what would be deleted
       python cli.py evict --apply   # actually deletes
"""

import argparse
import logging
import sys

from qate_env.env import Env
from qate_env.env_name import EnvName

from makalu import settings as sys_env
from makalu.cli import (
    arb_spot,
    backtest,
    data_check,
    evict,
    grafana_export,
    infra,
    metrics_data,
    plot,
    populate,
    publish,
    query,
    report,
    scan,
    trading_chart,
)
from makalu.data_outage import install_bronze_outage_patch

ENV_NAME = EnvName("LUIGI_MARIO")


def _setup_env() -> None:
    install_bronze_outage_patch()
    env = Env(sys_env.get_env_root_dir(), ENV_NAME)
    env.visit()
    # Also log to stdout so the agent subprocess can read LOG output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("{asctime} {levelname:8s} {name} {message}", style="{")
    )
    logging.getLogger().addHandler(stream_handler)


def main():
    parser = argparse.ArgumentParser(
        description="Makalu crypto market data warehouse CLI",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    data_check.register_cli(sub)
    arb_spot.register_cli(sub)
    populate.register_cli(sub)
    publish.register_cli(sub)
    metrics_data.register_cli(sub)
    plot.register_cli(sub)
    grafana_export.register_cli(sub)
    scan.register_cli(sub)
    evict.register_cli(sub)
    query.register_cli(sub)
    report.register_cli(sub)
    backtest.register_cli(sub)
    trading_chart.register_cli(sub)
    infra.register_cli(sub)

    args = parser.parse_args()
    _setup_env()
    args.func(args)


if __name__ == "__main__":
    main()
