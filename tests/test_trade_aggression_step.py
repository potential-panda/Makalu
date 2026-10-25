"""PopulateAggressionStep emits one record per whole second (the last trade's
aggression in that second), not one per trade. Guards the downsampling that
keeps the stored TradeAggression dataset ~15x smaller while staying byte-
identical to the Kilimanjaro feature_frame consumer (last(col) per floor(ts))."""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qate.core.model import Side, Trade  # noqa: E402
from qate.util.dt_range import DtRange  # noqa: E402

from pipeline.trade_derived_steps import Context, PopulateAggressionStep  # noqa: E402

# A real 2026 second boundary (unix seconds), so DtRange.is_within passes.
BASE = 1780000000.0  # int == the second bucket


def _trade(ts: float, side: Side = Side.BUY) -> Trade:
    return Trade(
        trade_id="t",
        side=side,
        price=100.0,
        size=1.0,
        symbol=None,
        exchange_name=None,
        exchange_ts=ts,
        received_ts=ts,
    )


def _wide_range() -> DtRange:
    return DtRange(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2030, 1, 1, tzinfo=timezone.utc),
    )


def test_one_record_per_second_at_last_trade():
    trades = [
        _trade(BASE + 0.1, Side.BUY),
        _trade(BASE + 0.5, Side.BUY),
        _trade(BASE + 0.9, Side.SELL),
        _trade(BASE + 1.2, Side.BUY),
        _trade(BASE + 3.0, Side.SELL),  # gap at second BASE+2 -> no row
    ]
    ctx = Context(populate_dt_range=_wide_range(), market=None, trades=trades)
    step = PopulateAggressionStep(windows_sec=[1])
    out = list(step._populate(None, ctx))

    # one row per second that had trades (BASE+2 is skipped)
    assert [ts for ts, _ in out] == [BASE + 0.9, BASE + 1.2, BASE + 3.0]

    fields = {ts: f for ts, f in out}
    # 1s window is [now-1, now]; at BASE+0.9 -> +0.1,+0.5,+0.9 = 3
    assert fields[BASE + 0.9]["total_count_1s"] == 3
    # at BASE+1.2 -> +0.5,+0.9,+1.2 = 3
    assert fields[BASE + 1.2]["total_count_1s"] == 3
    # at BASE+3.0 -> only +3.0 (+2.x evicted) = 1
    assert fields[BASE + 3.0]["total_count_1s"] == 1


def test_out_of_range_seconds_not_emitted():
    # populate range covers only [BASE+1, BASE+2): the BASE+0 and BASE+3 rows
    # must not be emitted, but their trades still feed the rolling window.
    trades = [_trade(BASE + 0.5), _trade(BASE + 1.5), _trade(BASE + 3.5)]
    rng = DtRange(
        datetime.fromtimestamp(BASE + 1, tz=timezone.utc),
        datetime.fromtimestamp(BASE + 2, tz=timezone.utc),
    )
    ctx = Context(populate_dt_range=rng, market=None, trades=trades)
    step = PopulateAggressionStep(windows_sec=[1])
    out = list(step._populate(None, ctx))
    emitted = [ts for ts, _ in out]
    assert BASE + 1.5 in emitted
    assert BASE + 0.5 not in emitted and BASE + 3.5 not in emitted


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
