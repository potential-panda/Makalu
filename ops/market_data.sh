#!/usr/bin/env bash
# Download raw market data from S3 to /panda/market-data/
# Called by cron daily at 04:00. Safe to run manually at any time.
set -euo pipefail
export PATH="/home/lulurun/.pyenv/shims:/home/lulurun/.pyenv/bin:$PATH"
cd /home/lulurun/workspace/Makalu
exec python cli.py market-data "$@"
