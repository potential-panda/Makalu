#!/usr/bin/env python3
"""Remove malformed order book snapshots from bronze parquet files.

Default mode is a dry-run. Use --apply to rewrite parquet files in place.
The drop rule matches doc/cleanup.md:

  top bid/ask missing or invalid
  OR ask <= bid
  OR top spread > max-spread-bps
  OR valid bid levels < min-valid-levels
  OR valid ask levels < min-valid-levels
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


ROOT = Path("/panda/makalu-data/bronze/dataset=OrderBook")


@dataclass(frozen=True)
class FileResult:
    symbol: str
    date: str
    path: str
    rows_before: int
    rows_after: int
    removed: int
    removed_pct: float
    null_top: int
    crossed_or_locked: int
    spread_gt_threshold: int
    partial_depth: int
    changed: bool


def _daterange(start: str | None, end: str | None) -> set[str] | None:
    if not start and not end:
        return None
    if not start or not end:
        raise ValueError("--start and --end must be provided together")
    cur = date.fromisoformat(start)
    last = date.fromisoformat(end)
    days = set()
    while cur <= last:
        days.add(cur.isoformat())
        cur += timedelta(days=1)
    return days


def _discover_symbols(exchange: str) -> list[str]:
    base = ROOT / f"exchange={exchange}"
    if not base.exists():
        return []
    return sorted(p.name.removeprefix("symbol=") for p in base.glob("symbol=*") if p.is_dir())


def _discover_files(exchange: str, symbols: list[str], dates: set[str] | None) -> list[tuple[str, str, Path]]:
    files: list[tuple[str, str, Path]] = []
    for symbol in symbols:
        base = ROOT / f"exchange={exchange}" / f"symbol={symbol}"
        for parquet in sorted(base.glob("date=*/data.parquet")):
            day = parquet.parent.name.removeprefix("date=")
            if dates is not None and day not in dates:
                continue
            files.append((symbol, day, parquet))
    return files


def _valid_depth(df: pd.DataFrame, side: str, levels: int) -> pd.Series:
    valid = pd.Series(0, index=df.index)
    for i in range(levels):
        price = pd.to_numeric(df.get(f"{side}{i}_price"), errors="coerce")
        qty = pd.to_numeric(df.get(f"{side}{i}_qty"), errors="coerce")
        valid += ((price > 0) & (qty > 0)).astype(int)
    return valid


def _irregular_mask(
    df: pd.DataFrame,
    *,
    max_spread_bps: float,
    min_valid_levels: int,
) -> tuple[pd.Series, dict[str, int]]:
    bid = pd.to_numeric(df["bid0_price"], errors="coerce")
    ask = pd.to_numeric(df["ask0_price"], errors="coerce")
    bid_qty = pd.to_numeric(df["bid0_qty"], errors="coerce")
    ask_qty = pd.to_numeric(df["ask0_qty"], errors="coerce")

    top_valid = (bid > 0) & (ask > 0) & (bid_qty > 0) & (ask_qty > 0)
    null_top = ~top_valid
    crossed_or_locked = top_valid & (ask <= bid)
    mid = (bid + ask) / 2
    spread_bps = ((ask - bid) / mid * 10000).where(top_valid & (mid > 0))
    spread_gt_threshold = spread_bps > max_spread_bps

    bid_depth = _valid_depth(df, "bid", 20)
    ask_depth = _valid_depth(df, "ask", 20)
    partial_depth = top_valid & ((bid_depth < min_valid_levels) | (ask_depth < min_valid_levels))

    mask = null_top | crossed_or_locked | spread_gt_threshold.fillna(False) | partial_depth
    components = {
        "null_top": int(null_top.sum()),
        "crossed_or_locked": int(crossed_or_locked.sum()),
        "spread_gt_threshold": int(spread_gt_threshold.sum()),
        "partial_depth": int(partial_depth.sum()),
    }
    return mask, components


def process_file(
    symbol: str,
    day: str,
    path: Path,
    *,
    apply: bool,
    max_spread_bps: float,
    min_valid_levels: int,
    compression: str,
) -> FileResult:
    df = pd.read_parquet(path)
    rows_before = len(df)
    if rows_before == 0:
        return FileResult(symbol, day, str(path), 0, 0, 0, 0.0, 0, 0, 0, 0, False)

    mask, components = _irregular_mask(
        df,
        max_spread_bps=max_spread_bps,
        min_valid_levels=min_valid_levels,
    )
    removed = int(mask.sum())
    rows_after = rows_before - removed
    changed = removed > 0

    if apply and changed:
        cleaned = df.loc[~mask].copy()
        tmp_path = path.with_name(f".{path.name}.tmp")
        cleaned.to_parquet(tmp_path, index=False, compression=compression)
        os.replace(tmp_path, path)

    return FileResult(
        symbol=symbol,
        date=day,
        path=str(path),
        rows_before=rows_before,
        rows_after=rows_after,
        removed=removed,
        removed_pct=removed / rows_before * 100,
        null_top=components["null_top"],
        crossed_or_locked=components["crossed_or_locked"],
        spread_gt_threshold=components["spread_gt_threshold"],
        partial_depth=components["partial_depth"],
        changed=changed,
    )


def _print_summary(results: list[FileResult], apply: bool) -> None:
    df = pd.DataFrame([r.__dict__ for r in results])
    if df.empty:
        print("No files matched.")
        return

    summary = df.groupby("symbol", as_index=False).agg(
        files=("path", "count"),
        changed_files=("changed", "sum"),
        rows_before=("rows_before", "sum"),
        rows_after=("rows_after", "sum"),
        removed=("removed", "sum"),
        null_top=("null_top", "sum"),
        crossed_or_locked=("crossed_or_locked", "sum"),
        spread_gt_threshold=("spread_gt_threshold", "sum"),
        partial_depth=("partial_depth", "sum"),
    )
    summary["removed_pct"] = summary["removed"] / summary["rows_before"] * 100
    print("Mode:", "APPLY" if apply else "DRY-RUN")
    print("Summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    changed = df[df["removed"] > 0].sort_values("removed_pct", ascending=False)
    print()
    print("Worst files by removed_pct:")
    cols = ["symbol", "date", "rows_before", "removed", "removed_pct", "null_top", "spread_gt_threshold", "partial_depth"]
    print(changed[cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", default="BINANCE")
    parser.add_argument("--symbols", nargs="+", default=None, help="Default: all symbols under exchange")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--max-spread-bps", type=float, default=10.0)
    parser.add_argument("--min-valid-levels", type=int, default=3)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--csv", default=None, help="Optional per-file result CSV")
    parser.add_argument("--apply", action="store_true", help="Rewrite parquet files in place")
    args = parser.parse_args()

    symbols = args.symbols or _discover_symbols(args.exchange)
    dates = _daterange(args.start, args.end)
    files = _discover_files(args.exchange, symbols, dates)
    print(f"Exchange: {args.exchange}")
    print(f"Symbols: {', '.join(symbols) if symbols else '(none)'}")
    print(f"Files: {len(files)}")
    print(f"Rule: spread>{args.max_spread_bps}bps OR valid depth < {args.min_valid_levels} levels OR invalid/crossed top")

    results: list[FileResult] = []
    for idx, (symbol, day, path) in enumerate(files, start=1):
        result = process_file(
            symbol,
            day,
            path,
            apply=args.apply,
            max_spread_bps=args.max_spread_bps,
            min_valid_levels=args.min_valid_levels,
            compression=args.compression,
        )
        results.append(result)
        if idx % 25 == 0 or idx == len(files):
            removed = sum(r.removed for r in results)
            rows = sum(r.rows_before for r in results)
            pct = removed / rows * 100 if rows else 0.0
            print(f"Processed {idx}/{len(files)} files; removed={removed} ({pct:.4f}%)")

    if args.csv:
        pd.DataFrame([r.__dict__ for r in results]).to_csv(args.csv, index=False)
        print(f"Wrote CSV: {args.csv}")

    _print_summary(results, args.apply)


if __name__ == "__main__":
    main()
