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

## What was actually done (2026-10-05)

The recommendation above — keep bronze as a raw audit trail, filter only in
derived datasets — was reversed. It is kept above as the record of why, because
the reversal has a cost worth being explicit about.

### Why it was reversed

Bronze is not only an internal input. Kitakamakura sells it, byte for byte, and
draws the website's per-day preview from it. Neither of those can apply a
derived-dataset filter: `verify` reads only `ts`, the schema and the metadata,
so it cannot reject an implausible quote, and the preview's vertical scale is
`price.min()`/`price.max()` over every row in the day. 0.3-0.9% of snapshots is
invisible in a mean and decisive in an extreme — two rows out of 860,000 are
enough to set a day's whole scale.

That is what happened. BINANCE `OrderBook` for 2026-09-17..21 was published
with its top-of-book artefacts intact and drew five days of ordinary trading as
flat lines: on 2026-09-19 the real range was 80,501..81,741 but the plot was
scaled 72,716..93,569, leaving the price line 1.6 pixels tall in a 27-pixel box.
It was not noticed for three weeks.

The boundary was 9/17 only because the one-shot run below cleaned everything up
to 9/16 and nothing afterwards. The raw defect rate is unchanged across it
(0.31% on 9/16, 0.35% on 9/17) — the cliff was ours, not the collector's.

### The audit trail is the msgpack, not bronze

Mutating bronze loses nothing that was not already held somewhere better. The
raw msgpack in `/panda/market-data/` and in S3 is the real record of what the
collector produced, it is retained, and bronze is fully reproducible from it
with `populate bronze --recreate`. Counting upstream defect rates should read
the msgpack — which is what established the rate was flat across 9/17, a fact
bronze could no longer answer once it had been partially cleaned.

### Where the rule lives now

`src/pipeline/order_book_quality.py` holds the thresholds and `is_malformed`.
Both bronze writers call it:

```text
makalu.cli.populate.execute_bronze      what the 04:00 scheduled job runs
pipeline.PopulateOrderBookStep          what src/bronze.py runs
```

Having two writers is the hazard here. The filter was first added only to
`PopulateOrderBookStep`, which looks like the bronze write path and is not the
one the daily job takes; a populate run proved it by writing 854,087 rows
unfiltered. The rule is shared so the two cannot disagree, but the write loop
is still duplicated — a third writer would silently miss it.

`PopulateBookStateStep` also gained the depth half of the rule; it had only the
spread half, so a partial book with a tight touch and nothing behind it was
passing into the imbalance and microprice fields computed from that depth.

Thresholds are `MAX_SPREAD_BPS = 10.0` and `MIN_VALID_LEVELS = 3`, identical to
`scripts/cleanup_orderbook_irregularity.py`. **Changing one without re-running
that script over history re-creates the cliff.** The two are cross-checked: on
2026-10-04 BTC_USDT the pipeline filter and the script both dropped exactly
7,850 of 854,087 rows.

### Cleanup actually applied

| When | Range | Scope |
|---|---|---|
| 2026-09-18 | start of history .. 2026-09-16 | 432/438/439 files, BTC/ETH/XRP |
| 2026-10-05 | 2026-09-17 .. 2026-10-04 | 52 files, 245,853 rows removed |

The 2026-10-05 pass also backfilled 2026-10-01, which was missing from bronze
for all 15 markets while its msgpack was present — a populate failure, not a
collection gap.

Two consequences of rewriting in place, both still open:

- **Files grow ~11%** (3.00 GB to 3.33 GB over 48 files) despite losing rows,
  because the pandas rewrite does not reproduce the original writer's row-group
  and encoding settings. The published `bytes` figures change accordingly.
- **The manifest is stale for every rewritten file.** `scan` only registers
  files it does not know, so it never corrects an existing row: 2026-07-22
  still reads 861,410 rows / 67.1 MB against an actual 857,567 / 83.0 MB.
  `evict` enforces the disk cap from `file_size`, so it is currently computing
  against numbers up to 20% low across roughly 1,300 files. Needs a targeted
  manifest update or a `--rescan-known` flag on `scan`.

### Still not covered

The collector fix (`potential-panda-core` cdc172a, carried into
`qate_exchanges`) was deployed to 1 of 4 hosts on 2026-10-05. Until the rest
follow, the filter above is what keeps bronze clean. It stays afterwards: a
resync can still race a reconnect, and a guard that only runs when it is
expected to fire is not a guard.

`kitakamakura verify` still cannot reject an implausible quote, and the
archive plot still scales on every row rather than the 120 it draws. Both are
unfixed.
