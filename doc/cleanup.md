# Data Cleanup Notes

## BINANCE order book malformed snapshots

Context: historical BINANCE bronze `OrderBook` parquet contains intermittent
partial/malformed snapshots. The upstream collector code is expected to be fixed,
but bronze partitions are append-once in normal operation, so historical parquet
will retain the bad rows unless we explicitly rewrite or filter them.

This note records the first sampling pass before any destructive cleanup.

### Sampled data

Dataset:

```text
/panda/makalu-data/bronze/dataset=OrderBook/exchange=BINANCE
```

Symbols:

```text
BTC_USDT
ETH_USDT
XRP_USDT
```

Sample windows:

```text
2026-07-20 -> 2026-07-24
2026-08-20 -> 2026-08-24
2026-09-12 -> 2026-09-16
```

Analysis script:

```bash
python scripts/analyze_orderbook_irregularity.py \
  --start 2026-09-12 --end 2026-09-16 \
  --csv /tmp/binance_orderbook_irregularity_20260912_20260916.csv
```

The script is read-only.

### Irregular snapshot definition

For the sampling pass, a row is counted as irregular if any of the following is
true:

```text
top bid/ask missing or invalid
OR ask <= bid
OR top spread > 10 bps
OR valid bid levels < 3
OR valid ask levels < 3
```

`Irregular` is a union of these conditions. The per-condition counts overlap and
must not be added together. For example, a snapshot can have both partial depth
and spread > 10 bps, and a row with missing top-of-book may be irregular even if
it is not counted in either of those two columns.

### Combined sampled rates

Across the 15 sampled days:

| Symbol | Rows sampled | Irregular rows | Irregular % | Spread >10 bps | Partial depth |
|---|---:|---:|---:|---:|---:|
| `BTC_USDT` | 12,857,136 | 55,583 | 0.432% | 0.150% | 0.311% |
| `ETH_USDT` | 12,782,622 | 65,973 | 0.516% | 0.166% | 0.364% |
| `XRP_USDT` | 9,323,091 | 90,833 | 0.974% | 0.222% | 0.662% |

Worst sampled day:

```text
XRP_USDT 2026-09-13
419,442 rows
8,040 irregular rows
1.917% irregular
```

Maximum observed top spread in the sample:

| Symbol | Max top spread |
|---|---:|
| `BTC_USDT` | 6,598.94 bps |
| `ETH_USDT` | 10,379.07 bps |
| `XRP_USDT` | 7,369.92 bps |

These extreme values are not plausible top-of-book market states. Spot checks
show brief partial books with many missing levels and far-away prices, followed
by quick reversion.

### Cleanup recommendation

Do not immediately mutate bronze without an explicit rewrite plan and backup.
Bronze is useful as an audit trail for upstream data quality.

For derived datasets and backtests, filter snapshots using the irregular
definition above. If we decide to rewrite historical bronze, use the same rule
and write the operation as an explicit cleanup command with:

```text
dry-run summary
backup or output-to-new-location mode
per-file row counts before/after
manifest scan after successful rewrite
```

The first cleanup candidate is BINANCE `OrderBook` for `BTC_USDT`, `ETH_USDT`,
and `XRP_USDT`.
