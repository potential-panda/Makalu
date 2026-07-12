#!/usr/bin/env python3
"""
Makalu data warehouse CLI.

PURPOSE
-------
Manage the Makalu parquet data warehouse, which stores crypto market microstructure
data across four processing layers. All data lives under /panda/makalu-data/.

DATA FLOW (must be populated in this order)
-------------------------------------------
  Raw msgpack files (/panda/market-data/)
      │
      ▼  populate bronze
  bronze/  Trade, OrderBook            ← raw tick data from exchanges
      │
      ▼  populate BookChange / BookState / TradeAggression / MarketPrice
  silver/  derived per-tick metrics    ← requires bronze to exist first
      │
      ▼  (gold layer — not yet in this CLI)
  gold/    VolSpread, LeadLag, Charts  ← requires silver/MarketPrice to exist first

SUBCOMMANDS
-----------
  data-check    READ-ONLY.  Inspect what parquet data is on disk. Safe to run anytime.
  populate      WRITE.      Convert or derive parquet files. Skips writes whose
                target data.parquet already exists; use --recreate to overwrite.
  market-data   WRITE.      Download raw msgpack market data from S3 to /panda/market-data/.
                Idempotent. Run this before 'populate bronze'.
  market-data-s3
                READ-ONLY.  Inspect raw market-data availability in S3.
                Cached locally; use --refresh to rescan S3.
  market-data-cleanup
                DESTRUCTIVE. Delete raw msgpack files already populated to bronze.
                Always dry-run by default; requires --apply to delete.
  metrics-data  WRITE.      Download 5-min trading metrics from S3 and push to InfluxDB.
                Idempotent. Looks back up to 28 days for missing intervals.
  scan          WRITE.      Register existing parquet files into the manifest database.
                Scheduled weekly by the API; safe to run manually after populate.
  evict         DESTRUCTIVE. If parquet usage exceeds the configured cap, delete
                manifest-tracked files past retention: bronze 12 months,
                all other layers 6 months.
                Always dry-run by default; requires --apply to delete.
  grafana-export WRITE.     Push a date range of parquet data to InfluxDB and
                return a Grafana dashboard URL. JSON output: {grafana_url, ...}.

TYPICAL WORKFLOW
----------------
  1. Check what bronze data exists:
       python cli.py data-check --dataset Trade --exchange GMO --symbol BTC_JPY

  2. Populate missing bronze data:
       python cli.py populate bronze --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07

  3. Populate silver datasets that depend on bronze:
       python cli.py populate MarketPrice --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07 \
          --execution-size 0.002

  4. Update manifest after populating, or rely on the weekly scheduled scan:
       python cli.py scan

  5. Configure and periodically review parquet cleanup:
       python cli.py evict --set-max-size 500GiB
       python cli.py evict           # dry-run: shows what would be deleted
       python cli.py evict --apply   # actually deletes
"""

import argparse
import logging
import sys

from qate.env import sys_env
from qate.env.env import Env
from qate.env.env_name import EnvName

from makalu.cli import (
    arb_spot,
    backtest,
    data_check,
    evict,
    grafana_export,
    infra,
    market_data,
    market_data_cleanup,
    market_data_s3,
    metrics_data,
    plot,
    populate,
    publish,
    query,
    scan,
    trading_chart,
)
from makalu.data_outage import install_qate_bronze_outage_patch

ENV_NAME = EnvName("LUIGI_MARIO")


def _setup_env() -> None:
    install_qate_bronze_outage_patch()
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
    market_data.register_cli(sub)
    market_data_s3.register_cli(sub)
    market_data_cleanup.register_cli(sub)
    metrics_data.register_cli(sub)
    plot.register_cli(sub)
    grafana_export.register_cli(sub)
    scan.register_cli(sub)
    evict.register_cli(sub)
    query.register_cli(sub)
    backtest.register_cli(sub)
    trading_chart.register_cli(sub)
    infra.register_cli(sub)

    args = parser.parse_args()
    _setup_env()
    args.func(args)


if __name__ == "__main__":
    main()
