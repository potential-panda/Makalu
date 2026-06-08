#!/usr/bin/env bash
# Download trading metrics from S3 and push to InfluxDB.
# Called by cron every 11 minutes. Safe to run manually at any time.
set -euo pipefail
export PATH="/home/lulurun/.pyenv/shims:/home/lulurun/.pyenv/bin:$PATH"
cd /home/lulurun/workspace/Makalu
exec python src/cli.py metrics-data "$@"
