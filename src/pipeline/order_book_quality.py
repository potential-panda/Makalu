"""The rule for a book snapshot that is not fit to store.

One definition, because there are two bronze writers -- `execute_bronze` in
makalu.cli.populate (what the daily job runs) and PopulateOrderBookStep (what
src/bronze.py runs) -- and a rule that lives in both of them separately is a
rule that will differ in one of them.

Same thresholds as scripts/cleanup_orderbook_irregularity.py, so a day written
by the pipeline matches a day that script has already cleaned. Changing a
threshold here without re-running that script over history splits the archive
into two standards, which is how 2026-09-17 became a visible cliff.

Why these snapshots exist: a collector that loses the near levels on a resync
publishes a book whose best bid is whatever far-away dust order survived. On
2026-09-17 BINANCE:BTC_USDT that meant a best bid of 68,127.93 against a best
ask of 75,710.00 -- not crossed, so a crossed-book check alone passes it. The
cause was potential-panda-core cdc172a, where OrderBookTracker.on_snapshot was
a silent no-op; fixed there, but the fix is per-host and a resync can still
race a reconnect, so this is the floor rather than an assumption about the
collector.
"""

MAX_SPREAD_BPS = 10.0
MIN_VALID_LEVELS = 3


def is_malformed(order_book) -> bool:
    """True if this snapshot should not be stored.

    Ordered so the cheap structural checks short-circuit before `spread_bps`,
    which divides by the mid and raises on a one-sided book.
    """
    best_bid, best_ask = order_book.best_bid, order_book.best_ask
    if best_bid is None or best_ask is None:
        return True
    if best_bid <= 0 or best_ask <= 0:
        return True
    # Covers locked (bid == ask) as well as crossed. A locked book has zero
    # spread, so the bps test below would pass it.
    if best_ask <= best_bid:
        return True
    if order_book.spread_bps > MAX_SPREAD_BPS:
        return True
    # Depth, not just the touch: a partial book often keeps a plausible
    # top-of-book with almost nothing behind it.
    if len(order_book.bids) < MIN_VALID_LEVELS:
        return True
    if len(order_book.asks) < MIN_VALID_LEVELS:
        return True
    return False


def describe_rule() -> str:
    """For the log line that reports how many rows a run dropped."""
    return (
        f"spread > {MAX_SPREAD_BPS} bps, fewer than {MIN_VALID_LEVELS} levels "
        f"a side, or no usable top of book"
    )
