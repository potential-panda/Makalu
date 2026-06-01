"""arb-spot: cross-exchange spot arbitrage analysis (step 4 of 4).

Reads already-populated MarketPrice data for GMO, BITBANK, and COINCHECK,
then computes cross-exchange spread and price-flip statistics.

Prerequisites (run these first):
  python cli.py market-data --market GMO:XRP_SPOT --days 7
  python cli.py market-data --market BITBANK:XRP_SPOT --days 7
  python cli.py market-data --market COINCHECK:XRP_SPOT --days 7
  python cli.py populate bronze  --exchange GMO      --symbol XRP_SPOT --start ... --end ...
  python cli.py populate bronze  --exchange BITBANK   --symbol XRP_SPOT --start ... --end ...
  python cli.py populate bronze  --exchange COINCHECK --symbol XRP_SPOT --start ... --end ...
  python cli.py populate MarketPrice --exchange GMO      --symbol XRP_SPOT --start ... --end ... --execution-size 150
  python cli.py populate MarketPrice --exchange BITBANK   --symbol XRP_SPOT --start ... --end ... --execution-size 150
  python cli.py populate MarketPrice --exchange COINCHECK --symbol XRP_SPOT --start ... --end ... --execution-size 150
  python cli.py scan
"""

import argparse
import sys
from datetime import date, timedelta
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd

LOG = getLogger(__name__)

EXCHANGES = ["GMO", "BITBANK", "COINCHECK"]

SYMBOL_ALIAS = {
    "btc_spot": "BTC_SPOT",
    "eth_spot": "ETH_SPOT",
    "xrp_spot": "XRP_SPOT",
}


def _execution_size(symbol_upper: str) -> str:
    from pipeline.market_pair import EXECUTION_SIZE
    from qate.core.model import Symbol
    sym = Symbol[symbol_upper]
    if sym not in EXECUTION_SIZE:
        raise ValueError(f"No canonical execution_size defined for {symbol_upper}. Add it to market_pair.EXECUTION_SIZE.")
    return EXECUTION_SIZE[sym]


def _date_range(days: int) -> tuple[str, str]:
    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=days - 1)
    return str(start), str(end)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_market_price(exchange: str, symbol: str, execution_size: str,
                       start: str, end: str) -> pd.DataFrame:
    import qate.env.sys_env as sys_env
    root = (
        Path(sys_env.get_parquet_root_dir())
        / "silver" / "dataset=MarketPrice"
        / f"exchange={exchange}" / f"symbol={symbol}"
        / f"execution_size={execution_size}"
    )
    dfs = []
    cur, end_d = date.fromisoformat(start), date.fromisoformat(end)
    while cur <= end_d:
        p = root / f"date={cur}" / "data.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p, columns=["ts", "bid", "ask"]))
        cur += timedelta(days=1)
    if not dfs:
        return pd.DataFrame(columns=["ts", "bid", "ask"])
    return pd.concat(dfs, ignore_index=True).sort_values("ts")


_JST = None  # lazy-loaded


def _jst():
    global _JST
    if _JST is None:
        from datetime import timezone
        _JST = timezone(timedelta(hours=9))
    return _JST


def _is_maintenance(ts_series: pd.Series) -> pd.Series:
    """Return boolean mask for GMO maintenance window: Saturday 08:00–12:00 JST."""
    dt_jst = pd.to_datetime(ts_series, unit="s", utc=True).dt.tz_convert(_jst())
    return (dt_jst.dt.dayofweek == 5) & (dt_jst.dt.hour >= 8) & (dt_jst.dt.hour < 12)


def _resample_1s(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 1-second bins (last tick per bin), add mid column."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("dt")[["bid", "ask"]].resample("1s").last().dropna()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["ts"] = df.index.view(np.int64) / 1e9
    return df.reset_index()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _count_flips(mid_a: np.ndarray, mid_b: np.ndarray) -> int:
    """
    Count sign changes in (mid_a - mid_b).
    A flip = the cheaper exchange switches identity.
    """
    signs = pd.Series(np.sign(mid_a - mid_b)).replace(0, np.nan).ffill().dropna()
    return int((signs.diff().abs() > 0).sum())


def _merge_pair(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame | None:
    merged = pd.merge_asof(
        df_a.sort_values("ts"),
        df_b.sort_values("ts").rename(columns={"bid": "bid_b", "ask": "ask_b", "mid": "mid_b", "dt": "dt_b"}),
        on="ts",
        tolerance=2.0,
        direction="nearest",
    ).dropna(subset=["bid", "ask", "bid_b", "ask_b"])
    if merged.empty:
        return None
    # Drop GMO maintenance window (Saturday 08:00–12:00 JST) from all pairs
    maint = _is_maintenance(merged["ts"])
    merged = merged[~maint].copy()
    if merged.empty:
        return None
    # positive: mid_a > mid_b  (A more expensive; buy A, sell B would be buy-side loss)
    merged["cross_bps"] = (merged["mid"] - merged["mid_b"]) / merged["mid_b"] * 10000
    # true arb: bid_a > ask_b  (buy on B, sell on A — profitable before fees)
    merged["arb_ab"] = (merged["bid"] - merged["ask_b"]) / merged["ask_b"] * 10000
    # true arb: bid_b > ask_a  (buy on A, sell on B — profitable before fees)
    merged["arb_ba"] = (merged["bid_b"] - merged["ask"]) / merged["ask"] * 10000
    merged["date"] = pd.to_datetime(merged["ts"], unit="s", utc=True).dt.date
    return merged


def _analyze_pair(df_a: pd.DataFrame, df_b: pd.DataFrame,
                  name_a: str, name_b: str) -> pd.DataFrame:
    merged = _merge_pair(df_a, df_b)
    if merged is None:
        return pd.DataFrame()
    records = []
    for d, g in merged.groupby("date"):
        n = len(g)
        hours = n / 3600
        arb_n = ((g["arb_ab"] > 0) | (g["arb_ba"] > 0)).sum()
        flips = _count_flips(g["mid"].values, g["mid_b"].values)
        records.append({
            "date": str(d),
            "pair": f"{name_a}/{name_b}",
            "samples": n,
            "arb_opp_pct": round(arb_n / n * 100, 2) if n else 0.0,
            "avg_bps": round(g["cross_bps"].mean(), 2),
            "max_bps": round(g["cross_bps"].max(), 2),
            "min_bps": round(g["cross_bps"].min(), 2),
            "std_bps": round(g["cross_bps"].std(), 2),
            "flips": flips,
            "flips_per_hr": round(flips / hours, 1) if hours > 0 else 0.0,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def _print_table(summary: pd.DataFrame) -> None:
    for pair, group in summary.groupby("pair", sort=False):
        name_a, name_b = pair.split("/")
        print(f"\n{'─'*74}")
        print(f"  {name_a}  vs  {name_b}"
              f"   (cross_bps = mid_{name_a} − mid_{name_b}; positive = {name_a} costlier)")
        print(f"{'─'*74}")
        print(f"  {'Date':<12} {'Arb%':>6} {'Avg':>8} {'Max':>8} {'Min':>8} {'Std':>6}  {'Flips':>6}  {'Flips/hr':>8}")
        print(f"  {'-'*70}")
        for _, r in group.sort_values("date").iterrows():
            print(
                f"  {r['date']:<12}"
                f" {r['arb_opp_pct']:>5.1f}%"
                f" {r['avg_bps']:>+8.1f}"
                f" {r['max_bps']:>+8.1f}"
                f" {r['min_bps']:>+8.1f}"
                f" {r['std_bps']:>6.1f} "
                f" {int(r['flips']):>6d} "
                f" {r['flips_per_hr']:>8.1f}"
            )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(resampled: dict[str, pd.DataFrame],
          pairs: list[tuple[str, str]],
          output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (name_a, name_b) in zip(axes, pairs):
        merged = _merge_pair(resampled[name_a], resampled[name_b])
        if merged is None:
            ax.text(0.5, 0.5, "No overlapping data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{name_a} vs {name_b}")
            continue

        dt = pd.to_datetime(merged["ts"], unit="s", utc=True)
        cross = merged["cross_bps"]

        ax.plot(dt, cross, lw=0.4, color="#4a90d9", alpha=0.8)
        ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)

        # Shade true-arb windows
        ax.fill_between(dt, cross, 0, where=(merged["arb_ab"] > 0).values,
                        color="#27ae60", alpha=0.4, label=f"Buy {name_b} / Sell {name_a}")
        ax.fill_between(dt, cross, 0, where=(merged["arb_ba"] > 0).values,
                        color="#e74c3c", alpha=0.4, label=f"Buy {name_a} / Sell {name_b}")

        # Orange tick at every flip
        signs = pd.Series(np.sign(cross.values)).replace(0, np.nan).ffill()
        flip_mask = (signs.diff().abs() > 0).values
        for flip_dt in dt[flip_mask]:
            ax.axvline(flip_dt, color="orange", lw=0.5, alpha=0.35)

        ax.set_title(f"{name_a} vs {name_b}  —  mid spread (bps)   orange = price flip", fontsize=10)
        ax.set_ylabel("bps", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d", tz="UTC"))
        ax.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Cross-exchange spot arbitrage — {list(resampled)[0].split('_')[0]}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def execute(args) -> None:
    symbol_upper = SYMBOL_ALIAS.get(args.symbol.lower(), args.symbol.upper())
    try:
        execution_size = _execution_size(symbol_upper)
    except (KeyError, ValueError) as e:
        LOG.error(str(e))
        sys.exit(1)

    start, end = _date_range(args.days)
    print(f"arb-spot: {symbol_upper}  |  {', '.join(EXCHANGES)}"
          f"  |  {start} → {end}  |  execution_size={execution_size}")
    print()

    # Load MarketPrice for each exchange
    resampled: dict[str, pd.DataFrame] = {}
    missing_hint: list[str] = []
    for exchange in EXCHANGES:
        df = _load_market_price(exchange, symbol_upper, execution_size, start, end)
        if df.empty:
            missing_hint.append(
                f"  python cli.py populate MarketPrice --exchange {exchange}"
                f" --symbol {symbol_upper} --start {start} --end {end}"
                f" --execution-size {execution_size}"
            )
            LOG.warning(f"No MarketPrice data for {exchange}:{symbol_upper} — skipping")
            continue
        rs = _resample_1s(df)
        resampled[exchange] = rs
        print(f"  {exchange}: {len(rs):,} 1s ticks  ({start} → {end})")

    if missing_hint:
        print("\nMissing data. Run these first:")
        for hint in missing_hint:
            print(hint)
        print()

    if len(resampled) < 2:
        LOG.error("Need data from at least 2 exchanges. Populate MarketPrice data first.")
        sys.exit(1)

    # All pairs among available exchanges
    available = [e for e in EXCHANGES if e in resampled]
    pairs = [(a, b) for i, a in enumerate(available) for b in available[i + 1:]]

    summaries = [
        _analyze_pair(resampled[a], resampled[b], a, b)
        for a, b in pairs
    ]
    summaries = [s for s in summaries if not s.empty]

    if not summaries:
        LOG.error("No overlapping data found for any pair.")
        sys.exit(1)

    _print_table(pd.concat(summaries, ignore_index=True))

    output_path = f"/tmp/arb_spot_{symbol_upper.lower()}.png"
    _plot(resampled, pairs, output_path)


BUCKET = "makalu_ArbSpot"
MEASUREMENT = "ArbSpot"


def _write_influx(resampled: dict[str, pd.DataFrame],
                  pairs: list[tuple[str, str]],
                  symbol: str) -> int:
    """Write 1s arb time-series to InfluxDB. Returns total points written."""
    import qate.env.sys_env as sys_env
    from influxdb_client import Point, WritePrecision
    from qate.store.influx_store import InfluxdbWriter

    writer = InfluxdbWriter(
        url=sys_env.get_influxdb()[1],
        token=sys_env.get_influxdb()[0],
        bucket_name=BUCKET,
        buffer_size=5000,
        no_retention=True,
    )

    total = 0
    BATCH = 5000

    for name_a, name_b in pairs:
        merged = _merge_pair(resampled[name_a], resampled[name_b])
        if merged is None:
            continue
        points = []
        for row in merged.itertuples(index=False):
            p = (
                Point(MEASUREMENT)
                .tag("exchange_a", name_a)
                .tag("exchange_b", name_b)
                .tag("symbol", symbol)
                .field("cross_bps", float(row.cross_bps))
                .field("arb_ab_bps", float(max(row.arb_ab, 0.0)))
                .field("arb_ba_bps", float(max(row.arb_ba, 0.0)))
                .time(int(row.ts * 1000), WritePrecision.MS)
            )
            points.append(p)
            if len(points) >= BATCH:
                writer.write(points)
                total += len(points)
                points = []
        if points:
            writer.write(points)
            total += len(points)
        print(f"  {name_a}/{name_b}: {total} points written")

    writer.close()
    return total


def execute_export(args) -> None:
    symbol_upper = SYMBOL_ALIAS.get(args.symbol.lower(), args.symbol.upper())
    try:
        execution_size = _execution_size(symbol_upper)
    except (KeyError, ValueError) as e:
        LOG.error(str(e))
        sys.exit(1)

    start, end = _date_range(args.days)
    print(f"arb-spot-export: {symbol_upper}  |  {', '.join(EXCHANGES)}"
          f"  |  {start} → {end}  →  InfluxDB bucket={BUCKET}")
    print()

    resampled: dict[str, pd.DataFrame] = {}
    for exchange in EXCHANGES:
        df = _load_market_price(exchange, symbol_upper, execution_size, start, end)
        if df.empty:
            LOG.warning(f"No MarketPrice data for {exchange}:{symbol_upper} — skipping")
            continue
        resampled[exchange] = _resample_1s(df)
        print(f"  {exchange}: {len(resampled[exchange]):,} ticks loaded")

    if len(resampled) < 2:
        LOG.error("Need data from at least 2 exchanges.")
        sys.exit(1)

    available = [e for e in EXCHANGES if e in resampled]
    pairs = [(a, b) for i, a in enumerate(available) for b in available[i + 1:]]

    print("\nWriting to InfluxDB...")
    total = _write_influx(resampled, pairs, symbol_upper)
    print(f"\nDone — {total:,} total points in bucket '{BUCKET}', measurement '{MEASUREMENT}'")


def register_cli(sub) -> None:
    p = sub.add_parser(
        "arb-spot",
        help="Cross-exchange spot arbitrage analysis — reads existing MarketPrice data",
        description=(
            "Reads already-populated MarketPrice data for GMO, BITBANK, and COINCHECK,\n"
            "then computes cross-exchange spread and price-flip statistics.\n\n"
            "Run market-data + populate bronze + populate MarketPrice + scan first.\n\n"
            "Columns:\n"
            "  Arb%      — % of 1s windows with a profitable cross-exchange opportunity\n"
            "  Avg/Max/Min/Std bps — mid-price spread statistics (positive = first exchange costlier)\n"
            "  Flips     — sign changes in mid spread (cheaper exchange switches)\n"
            "  Flips/hr  — flip rate per hour\n\n"
            "Output: table to stdout + PNG to /tmp/arb_spot_<symbol>.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--symbol",
        required=True,
        choices=["btc_spot", "eth_spot", "xrp_spot"],
        help="Spot symbol to analyse",
    )
    p.add_argument(
        "--days",
        type=int,
        default=7,
        metavar="N",
        help="Trailing days to analyse (default: 7)",
    )
    p.set_defaults(func=execute)

    q = sub.add_parser(
        "arb-spot-export",
        help="Push arb-spot time-series to InfluxDB (bucket: makalu_ArbSpot)",
        description=(
            f"Computes cross-exchange spread for GMO/BITBANK/COINCHECK and writes\n"
            f"1-second time-series to InfluxDB bucket '{BUCKET}'.\n\n"
            f"Measurement: {MEASUREMENT}\n"
            f"Tags:   exchange_a, exchange_b, symbol\n"
            f"Fields: cross_bps, arb_ab_bps, arb_ba_bps\n\n"
            f"Requires MarketPrice data already populated (run arb-spot first to check).\n"
            f"GMO maintenance window (Sat 08–12 JST) is excluded."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    q.add_argument("--symbol", required=True, choices=["btc_spot", "eth_spot", "xrp_spot"])
    q.add_argument("--days", type=int, default=7, metavar="N",
                   help="Trailing days to export (default: 7)")
    q.set_defaults(func=execute_export)
