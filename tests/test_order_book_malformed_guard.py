"""PopulateOrderBookStep drops malformed book snapshots before they reach bronze.

The collector could lose the near levels on a resync and publish a book whose
best bid was a far-away dust order (potential-panda-core cdc172a). Those rows
are not quotes, and one of them sets a whole day's min/max -- which is how
~3-4k rows/day in 0.4% of snapshots drew a flat line on the published archive
preview for 2026-09-17..21. The rule here is the same one
scripts/cleanup_orderbook_irregularity.py applies to historical bronze, so a
day written by the pipeline matches a day that script already cleaned.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qate.core.model import OrderBook, OrderLevel  # noqa: E402

from pipeline.market_data_steps import Context, PopulateOrderBookStep  # noqa: E402


BASE = 1789570800.0


class _FakeStore:
    """Captures what the step would have written."""

    def __init__(self):
        self.written = []
        self.closed = False

    def add_order_book(self, order_book):
        self.written.append(order_book)

    def close(self):
        self.closed = True


class _FakeParquetStore:
    def __init__(self, store):
        self._store = store

    def create_order_book(self, market):
        return self._store


class _FakePipelineContext:
    def __init__(self, store):
        self.parquet_store = _FakeParquetStore(store)


def _book(ts: float, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> OrderBook:
    return OrderBook(
        bids=[OrderLevel(p, a) for p, a in bids],
        asks=[OrderLevel(p, a) for p, a in asks],
        symbol=None,
        exchange_name=None,
        exchange_ts=ts,
        received_ts=ts,
    )


def _levels(top: float, step: float, n: int) -> list[tuple[float, float]]:
    return [(top + step * i, 1.0) for i in range(n)]


def _healthy(ts: float) -> OrderBook:
    # 20 levels a side, 0.01 wide at the touch -- ~0.013 bps on a 75k book.
    return _book(ts, _levels(75_697.82, -0.01, 20), _levels(75_697.83, 0.01, 20))


def _run(books: list[OrderBook]) -> list[OrderBook]:
    store = _FakeStore()
    ctx = Context(populate_dt_range=None, market=None, trades=[], order_books=books)
    PopulateOrderBookStep().populate(_FakePipelineContext(store), ctx)
    assert store.closed, "the store must be closed even when rows are dropped"
    return store.written


def test_healthy_books_are_written_unchanged():
    books = [_healthy(BASE + i) for i in range(5)]
    written = _run(books)
    assert [b.exchange_ts for b in written] == [BASE + i for i in range(5)]


def test_partial_book_with_far_away_best_bid_is_dropped():
    # The real 2026-09-17 BINANCE:BTC_USDT shape: two dust bids ~10% below the
    # ask, one ask. Not crossed, so a crossed-book check alone would pass it.
    bad = _book(BASE, [(68_127.93, 0.00221), (60_558.16, 0.00231)], [(75_710.00, 0.70000)])
    written = _run([_healthy(BASE - 1), bad, _healthy(BASE + 1)])
    assert len(written) == 2
    assert bad not in written


def test_too_few_levels_is_dropped_even_with_a_tight_touch():
    # A partial book can keep a plausible top-of-book with nothing behind it,
    # which the spread check passes. Depth is what catches it.
    bad = _book(BASE, _levels(75_697.82, -0.01, 2), _levels(75_697.83, 0.01, 2))
    assert bad.spread_bps < PopulateOrderBookStep.MAX_SPREAD_BPS
    assert _run([bad]) == []


def test_exactly_min_valid_levels_is_kept():
    n = PopulateOrderBookStep.MIN_VALID_LEVELS
    ok = _book(BASE, _levels(75_697.82, -0.01, n), _levels(75_697.83, 0.01, n))
    assert _run([ok]) == [ok]


def test_wide_spread_is_dropped_at_the_threshold():
    mid = 75_000.0
    # 20 bps wide, well over the 10 bps limit, with full depth either side.
    half = mid * 20 / 10_000 / 2
    bad = _book(BASE, _levels(mid - half, -0.01, 20), _levels(mid + half, 0.01, 20))
    assert bad.spread_bps > PopulateOrderBookStep.MAX_SPREAD_BPS
    assert _run([bad]) == []


def test_empty_side_is_dropped_and_does_not_raise():
    # best_bid is None here: the guard must short-circuit before spread_bps,
    # which would raise on a one-sided book.
    assert _run([_book(BASE, [], _levels(75_697.83, 0.01, 20))]) == []
    assert _run([_book(BASE, _levels(75_697.82, -0.01, 20), [])]) == []


def test_crossed_book_is_dropped():
    bad = _book(BASE, _levels(75_700.00, -0.01, 20), _levels(75_690.00, 0.01, 20))
    assert _run([bad]) == []


def test_locked_book_is_dropped():
    # bid == ask. Zero spread passes a `> MAX_SPREAD_BPS` test, so it needs
    # the explicit ask <= bid check.
    bad = _book(BASE, _levels(75_697.83, -0.01, 20), _levels(75_697.83, 0.01, 20))
    assert _run([bad]) == []


def test_non_positive_prices_are_dropped():
    assert _run([_book(BASE, [(0.0, 1.0)] * 3, _levels(75_697.83, 0.01, 20))]) == []


def test_skip_count_is_reported(caplog):
    import logging

    books = [_healthy(BASE), _book(BASE + 1, [(68_127.93, 0.002)], [(75_710.0, 0.7)])]
    with caplog.at_level(logging.INFO, logger="pipeline.market_data_steps"):
        _run(books)
    text = caplog.text
    # The count must be visible: these rows went unnoticed for months because
    # nothing downstream said how many were being dropped.
    assert "Skipped 1 malformed snapshots of 2" in text
    assert "Wrote 1 order books" in text


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
