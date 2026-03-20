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
  data-check  READ-ONLY.  Inspect what data is on disk. Safe to run anytime.
  populate    WRITE.      Convert or derive parquet files. Idempotent (safe to re-run).
  scan        WRITE.      Register existing parquet files into the manifest database.
              Run once after populate to keep the manifest in sync.
  evict       DESTRUCTIVE. Delete files that exceed the retention policy.
              Always dry-run by default; requires --apply to delete.

TYPICAL WORKFLOW
----------------
  1. Check what bronze data exists:
       python cli.py data-check --dataset Trade --exchange GMO --symbol BTC_JPY

  2. Populate missing bronze data:
       python cli.py populate bronze --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07

  3. Populate silver datasets that depend on bronze:
       python cli.py populate MarketPrice --exchange GMO --symbol BTC_JPY --start 2026-01-01 --end 2026-01-07 \
          --execution-size 0.002

  4. Update manifest after populating:
       python cli.py scan

  5. Periodically clean up old data:
       python cli.py evict           # dry-run: shows what would be deleted
       python cli.py evict --apply   # actually deletes
"""

import argparse
import logging
import sys

from qate.env import sys_env
from qate.env.env import Env
from qate.env.env_name import EnvName

from makalu.cli import data_check, evict, plot, populate, scan

ENV_NAME = EnvName.LUIGI_MARIO


def _setup_env() -> None:
    env = Env(sys_env.ENV_DIR, ENV_NAME)
    env.visit()
    # Also log to stdout so the agent subprocess can read LOG output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("{asctime} {levelname:8s} {name} {message}", style="{"))
    logging.getLogger().addHandler(stream_handler)


def main():
    parser = argparse.ArgumentParser(
        description="Makalu crypto market data warehouse CLI",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    data_check.register_cli(sub)
    populate.register_cli(sub)
    plot.register_cli(sub)
    scan.register_cli(sub)
    evict.register_cli(sub)

    args = parser.parse_args()
    _setup_env()
    args.func(args)


if __name__ == "__main__":
    main()
