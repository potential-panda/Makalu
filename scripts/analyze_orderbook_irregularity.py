#!/usr/bin/env python3
"""Analyze order book snapshot irregularity in bronze parquet files.

This is read-only. It reports per-day rates for obvious malformed snapshots:
missing/invalid top of book, crossed/locked books, unusually wide top spread,
partial depth, and timestamp gaps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


ROOT = Path("/panda/makalu-data/bronze/dataset=OrderBook")


@dataclass(frozen=True)
class DayStats:
    symbol: str
    date: str
    rows: int
    irregular: int
    irregular_pct: float
    null_top: int
    crossed_or_locked: int
    spread_gt_1bp: int
    spread_gt_10bp: int
    spread_gt_100bp: int
    partial_depth: int
    gap_gt_1s: int
    max_gap_s: float
    max_spread_bps: float
    p999_spread_bps: float


def _daterange(start: str, end: str) -> list[str]:
    cur = date.fromisoformat(start)
    last = date.fromisoformat(end)
    days = []
    while cur <= last:
        days.append(cur.isoformat())
        cur += timedelta(days=1)
    return days


def _valid_depth(df: pd.DataFrame, side: str, levels: int) -> pd.Series:
    valid = pd.Series(0, index=df.index)
    for i in range(levels):
        price = pd.to_numeric(df.get(f"{side}{i}_price"), errors="coerce")
        qty = pd.to_numeric(df.get(f"{side}{i}_qty"), errors="coerce")
        valid += ((price > 0) & (qty > 0)).astype(int)
    return valid


def analyze_file(path: Path, symbol: str, day: str, min_valid_levels: int) -> DayStats:
    df = pd.read_parquet(path)
    rows = len(df)
    if rows == 0:
        return DayStats(symbol, day, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0)

    df = df.sort_values("ts").reset_index(drop=True)
    bid = pd.to_numeric(df["bid0_price"], errors="coerce")
    ask = pd.to_numeric(df["ask0_price"], errors="coerce")
    bid_qty = pd.to_numeric(df["bid0_qty"], errors="coerce")
    ask_qty = pd.to_numeric(df["ask0_qty"], errors="coerce")

    top_valid = (bid > 0) & (ask > 0) & (bid_qty > 0) & (ask_qty > 0)
    null_top = ~top_valid
    crossed_or_locked = top_valid & (ask <= bid)
    mid = (bid + ask) / 2
    spread_bps = ((ask - bid) / mid * 10000).where(top_valid & (mid > 0))

    bid_depth = _valid_depth(df, "bid", 20)
    ask_depth = _valid_depth(df, "ask", 20)
    partial_depth = top_valid & ((bid_depth < min_valid_levels) | (ask_depth < min_valid_levels))

    spread_gt_1bp = spread_bps > 1.0
    spread_gt_10bp = spread_bps > 10.0
    spread_gt_100bp = spread_bps > 100.0

    irregular_mask = (
        null_top
        | crossed_or_locked
        | spread_gt_10bp.fillna(False)
        | partial_depth
    )

    dt = pd.to_numeric(df["ts"], errors="coerce").diff()
    gap_gt_1s = dt > 1.0

    irregular = int(irregular_mask.sum())
    return DayStats(
        symbol=symbol,
        date=day,
        rows=rows,
        irregular=irregular,
        irregular_pct=irregular / rows * 100,
        null_top=int(null_top.sum()),
        crossed_or_locked=int(crossed_or_locked.sum()),
        spread_gt_1bp=int(spread_gt_1bp.sum()),
        spread_gt_10bp=int(spread_gt_10bp.sum()),
        spread_gt_100bp=int(spread_gt_100bp.sum()),
        partial_depth=int(partial_depth.sum()),
        gap_gt_1s=int(gap_gt_1s.sum()),
        max_gap_s=float(dt.max(skipna=True) or 0.0),
        max_spread_bps=float(spread_bps.max(skipna=True) or 0.0),
        p999_spread_bps=float(spread_bps.quantile(0.999) or 0.0),
    )


def summarize(stats: list[DayStats]) -> pd.DataFrame:
    df = pd.DataFrame([s.__dict__ for s in stats])
    if df.empty:
        return df
    grouped = df.groupby("symbol", as_index=False).agg(
        days=("date", "count"),
        rows=("rows", "sum"),
        irregular=("irregular", "sum"),
        null_top=("null_top", "sum"),
        crossed_or_locked=("crossed_or_locked", "sum"),
        spread_gt_1bp=("spread_gt_1bp", "sum"),
        spread_gt_10bp=("spread_gt_10bp", "sum"),
        spread_gt_100bp=("spread_gt_100bp", "sum"),
        partial_depth=("partial_depth", "sum"),
        gap_gt_1s=("gap_gt_1s", "sum"),
        max_gap_s=("max_gap_s", "max"),
        max_spread_bps=("max_spread_bps", "max"),
        p999_spread_bps_max=("p999_spread_bps", "max"),
    )
    grouped["irregular_pct"] = grouped["irregular"] / grouped["rows"] * 100
    grouped["spread_gt_10bp_pct"] = grouped["spread_gt_10bp"] / grouped["rows"] * 100
    grouped["partial_depth_pct"] = grouped["partial_depth"] / grouped["rows"] * 100
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", default="BINANCE")
    parser.add_argument("--symbols", nargs="+", default=["BTC_USDT", "ETH_USDT", "XRP_USDT"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--min-valid-levels", type=int, default=3)
    parser.add_argument("--csv", default=None, help="Optional path to write per-day CSV")
    args = parser.parse_args()

    stats: list[DayStats] = []
    missing: list[str] = []
    for symbol in args.symbols:
        for day in _daterange(args.start, args.end):
            path = ROOT / f"exchange={args.exchange}" / f"symbol={symbol}" / f"date={day}" / "data.parquet"
            if not path.exists():
                missing.append(f"{symbol}:{day}")
                continue
            stats.append(analyze_file(path, symbol, day, args.min_valid_levels))

    per_day = pd.DataFrame([s.__dict__ for s in stats])
    summary = summarize(stats)

    if args.csv:
        per_day.to_csv(args.csv, index=False)
        print(f"Wrote per-day CSV: {args.csv}")

    if missing:
        print("Missing files:")
        for item in missing:
            print(f"  {item}")
        print()

    if per_day.empty:
        print("No files analyzed.")
        return

    display_cols = [
        "symbol",
        "date",
        "rows",
        "irregular",
        "irregular_pct",
        "null_top",
        "spread_gt_10bp",
        "spread_gt_100bp",
        "partial_depth",
        "gap_gt_1s",
        "max_spread_bps",
    ]
    print("Per-day stats:")
    print(per_day[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("Summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
