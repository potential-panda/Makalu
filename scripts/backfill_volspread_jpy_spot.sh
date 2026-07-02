#!/usr/bin/env bash
set -u

cd /home/lulurun/workspace/Makalu

START="${START:-2025-07-01}"
END="${END:-2026-06-20}"
LOG="${LOG:-/tmp/makalu_volspread_backfill_2025-07-01_2026-06-20.log}"

markets=(
  "GMO BTC_JPY 0.002"
  "GMO ETH_JPY 0.1"
  "GMO XRP_JPY 150"
  "GMO BTC_SPOT 0.002"
  "GMO ETH_SPOT 0.1"
  "GMO XRP_SPOT 150"
  "BITBANK BTC_SPOT 0.002"
  "BITBANK ETH_SPOT 0.1"
  "BITBANK XRP_SPOT 150"
)

{
  echo "script-start $(date -Is)"
  for entry in "${markets[@]}"; do
    set -- $entry
    exchange="$1"
    symbol="$2"
    execution_size="$3"

    echo "=== $(date -Is) START ${exchange} ${symbol} execution_size=${execution_size} ${START}..${END} ==="
    python src/cli.py populate VolSpread \
      --exchange "$exchange" \
      --symbol "$symbol" \
      --start "$START" \
      --end "$END" \
      --execution-size "$execution_size" \
      --recreate
    status=$?
    echo "=== $(date -Is) END ${exchange} ${symbol} status=${status} ==="

    if [ "$status" -ne 0 ]; then
      echo "Backfill failed at ${exchange} ${symbol}; stopping."
      exit "$status"
    fi
  done
  echo "=== $(date -Is) ALL DONE ==="
} 2>&1 | tee -a "$LOG"
