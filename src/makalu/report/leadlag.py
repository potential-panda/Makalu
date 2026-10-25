"""Daily lead-lag statistics between two markets, for the lead-lag report.

One row per JST day, computed from silver MarketPrice mids at one execution
size. Unlike the gold LeadLag dataset (EWLL features over sliding windows, for
models), these are plain whole-day numbers meant to be read: where the
cross-correlation peaks, how high, and how much of the lead's move the lag
market has absorbed after a few seconds.

Three resolutions are measured, because the fine one is sensitive to the lead
market's tick size -- a coarse tick makes 100ms mid returns noisy, which
weakens every 100ms correlation without anything about the relationship
changing. The 1s and 5s numbers are the ones to trust for comparing periods.

  100ms  rho(k) for k = -1s..+3s, peak lag, impulse response beta(h)
  1s     rho at lag 0 and 1 bucket, beta over 0-3s
  5s     rho at lag 0 and 1 bucket, beta over 0-10s

plus the context that moves these numbers: volatility, tick size, update
counts and spreads.

MarketPrice is derived by Hase, which leaves out snapshots that cannot fill the
size or whose walked spread marks them as collector artefacts, and records how
many it left out in each file. The report carries that count per day, so a month
with an unusual number of artefacts is visible next to the numbers it fed.

The same 100ms and 1s numbers are also measured inside each JST hour
(`compute_day_hours`), to show how the relationship moves through the trading
day and to compare periods at the same hourly volatility.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from logging import getLogger

import numpy as np
import pandas as pd
from komachi import bronze
from qate.core.model import Market
from threadpoolctl import threadpool_limits

from makalu import settings as sys_env
from makalu.storage.parquet_store import ParquetStore

LOG = getLogger(__name__)

FINE_STEP_SEC = 0.1
FINE_LAGS = np.arange(-10, 31)  # -1.0s .. +3.0s in 100ms steps
CURVE_LAGS_MS = range(-500, 2001, 100)  # what the report stores and draws
BETA_HORIZONS_SEC = (0.2, 0.5, 1.0, 2.0, 3.0)
JST_OFFSET_SEC = 9 * 3600
HOUR_MIN_COVERAGE = (
    0.9  # share of an hour's minutes that must hold a quote on both sides
)
# GMO keeps publishing snapshots through its weekly maintenance (Sat ~9:00-11:00 JST) with the price frozen, so
# quote coverage alone lets those hours in. The lag market's mid must also change in this share of minutes; normal
# GMO hours move in 59-60 minutes of 60. Not applied to the lead: BINANCE BTC has genuinely quiet minutes.
HOUR_MIN_LAG_ACTIVE = 55 / 60


@dataclass(frozen=True)
class Pair:
    lead: Market
    lag: Market
    execution_size: str

    @property
    def key(self) -> str:
        return f"{self.lead.exchange_name.name}_{self.lead.symbol.name}__{self.lag.exchange_name.name}_{self.lag.symbol.name}"


def curve_column(lag_ms: int) -> str:
    return f"rho_lag_m{-lag_ms}" if lag_ms < 0 else f"rho_lag_{lag_ms}"


def _read_day(
    market: Market, execution_size: str, day: str
) -> tuple[pd.DataFrame, int]:
    """The day's MarketPrice, and how many snapshots Hase left out deriving it."""
    store = ParquetStore(sys_env.get_parquet_root_dir()).create_execution_price(
        market, execution_size
    )
    df = store.read_a_day(day)
    if df.empty:
        return df, 0
    dropped = df.attrs.get("dropped", {})
    df = df[["ts", "mid", "spread_bps"]].sort_values("ts").reset_index(drop=True)
    return df, int(dropped.get("unfilled", 0) + dropped.get("degenerate", 0))


def _grid(df: pd.DataFrame, t0: float, n: int, step: float) -> np.ndarray:
    """Last mid in each bucket of `step` seconds from t0, carried forward."""
    idx = np.floor((df["ts"].to_numpy() - t0) / step).astype(np.int64)
    keep = (idx >= 0) & (idx < n)
    out = np.full(n, np.nan)
    out[idx[keep]] = df["mid"].to_numpy()[keep]
    return pd.Series(out).ffill().to_numpy()


def _returns(
    lead_df: pd.DataFrame, lag_df: pd.DataFrame, step: float
) -> tuple[np.ndarray, np.ndarray]:
    t0 = np.floor(lead_df["ts"].iloc[0])
    n = int(np.ceil((lead_df["ts"].iloc[-1] - t0) / step)) + 1
    a, b = _grid(lead_df, t0, n, step), _grid(lag_df, t0, n, step)
    ok = ~(np.isnan(a) | np.isnan(b))
    first = int(np.argmax(ok))
    ra, rb = np.diff(np.log(a[first:])), np.diff(np.log(b[first:]))
    return ra - ra.mean(), rb - rb.mean()


def _cross_cov(ra: np.ndarray, rb: np.ndarray, lags) -> np.ndarray:
    """cov(rb[t], ra[t-k]) for each k: positive k means the lead moved first."""
    n = len(ra)
    out = np.empty(len(lags))
    for i, k in enumerate(lags):
        out[i] = rb[k:] @ ra[: n - k] if k >= 0 else rb[: n + k] @ ra[-k:]
    return out


def _tick(market: Market, day: str) -> float:
    """Price increment of a market, read off its best bid in bronze.

    MarketPrice's bid is an execution price for a size, not a quote, so it is
    not on the tick grid; the order book's top level is.
    """
    try:
        path = bronze.file(
            market.id, "OrderBook", day, root=sys_env.get_parquet_root_dir()
        )
    except bronze.MissingData:
        return np.nan
    bids = (
        pd.read_parquet(path, columns=["bid0_price"])["bid0_price"].dropna().to_numpy()
    )
    steps = np.abs(np.diff(bids))
    steps = steps[steps > 1e-12]
    return float(np.round(steps.min(), 10)) if len(steps) else np.nan


def compute_day(pair: Pair, day: str) -> dict | None:
    lead_df, lead_dropped = _read_day(pair.lead, pair.execution_size, day)
    lag_df, lag_dropped = _read_day(pair.lag, pair.execution_size, day)
    if lead_df.empty or lag_df.empty:
        return None

    row: dict = {"date": day}

    ra, rb = _returns(lead_df, lag_df, FINE_STEP_SEC)
    va, vb = ra @ ra, rb @ rb
    cc = _cross_cov(ra, rb, FINE_LAGS)
    rho = cc / np.sqrt(va * vb)
    peak = int(np.argmax(rho))
    beta = {
        h: cc[(FINE_LAGS >= 0) & (FINE_LAGS <= round(h / FINE_STEP_SEC))].sum() / va
        for h in BETA_HORIZONS_SEC
    }
    row.update(
        {
            "peak_lag_ms": int(FINE_LAGS[peak] * 100),
            "peak_rho": rho[peak],
            "rho0": rho[FINE_LAGS == 0][0],
            "rho_sum_0_3s": rho[FINE_LAGS >= 0].sum(),
            "rho_sum_neg": rho[FINE_LAGS < 0].sum(),
            **{f"beta_{str(h).replace('.', '_')}s": v for h, v in beta.items()},
            "vol_lead_bps_d": np.sqrt(va) * 1e4,
            "vol_lag_bps_d": np.sqrt(vb) * 1e4,
        }
    )
    for lag_ms in CURVE_LAGS_MS:
        row[curve_column(lag_ms)] = rho[FINE_LAGS == lag_ms // 100][0]

    for step, beta_buckets in ((1.0, 3), (5.0, 2)):
        ra, rb = _returns(lead_df, lag_df, step)
        va, vb = ra @ ra, rb @ rb
        cc = _cross_cov(ra, rb, range(0, 3))
        s = int(step)
        row[f"rho0_{s}s"] = cc[0] / np.sqrt(va * vb)
        row[f"rho1_{s}s"] = cc[1] / np.sqrt(va * vb)
        row[f"beta_{s}s_cum"] = cc[:beta_buckets].sum() / va

    lead_tick = _tick(pair.lead, day)
    row.update(
        {
            "lead_px": float(lead_df["mid"].median()),
            "lead_tick": lead_tick,
            "lead_tick_bps": lead_tick / float(lead_df["mid"].median()) * 1e4,
            "lead_updates": int((np.diff(lead_df["mid"].to_numpy()) != 0).sum()),
            "lag_updates": int((np.diff(lag_df["mid"].to_numpy()) != 0).sum()),
            "lead_spread_bps": float(lead_df["spread_bps"].median()),
            "lag_spread_bps": float(lag_df["spread_bps"].median()),
            "lead_dropped_rows": lead_dropped,
            "lag_dropped_rows": lag_dropped,
        }
    )
    return row


def _hour_stats(
    ra: np.ndarray, rb: np.ndarray, lags
) -> tuple[np.ndarray, float, float] | None:
    """Cross-covariance and variances of one hour's de-meaned returns, or None when either side never moved."""
    ra, rb = ra - ra.mean(), rb - rb.mean()
    va, vb = ra @ ra, rb @ rb
    if va == 0 or vb == 0:
        return None
    return _cross_cov(ra, rb, lags), va, vb


def compute_day_hours(pair: Pair, day: str) -> list[dict]:
    """One row per JST hour of the day: the 100ms and 1s numbers of `compute_day`, measured inside that hour.

    Both grids start at the day's JST midnight so buckets line up with clock hours.
    An hour is left out when either market has quotes in fewer than 90% of its minutes, or when the lag
    market's price changes in fewer than 55 of them (a venue in maintenance with its quotes frozen).
    """
    lead_df, _ = _read_day(pair.lead, pair.execution_size, day)
    lag_df, _ = _read_day(pair.lag, pair.execution_size, day)
    if lead_df.empty or lag_df.empty:
        return []
    t0 = (
        np.floor((lead_df["ts"].iloc[0] + JST_OFFSET_SEC) / 86400) * 86400
        - JST_OFFSET_SEC
    )
    mids = {}
    for step in (FINE_STEP_SEC, 1.0):
        n = int(round(86400 / step)) + 1
        # the day's first bucket opens a few ms before its first quote: fill it from the next one
        mids[step] = tuple(
            pd.Series(_grid(df, t0, n, step)).bfill(limit=50).to_numpy()
            for df in (lead_df, lag_df)
        )

    ta, tb = lead_df["ts"].to_numpy(), lag_df["ts"].to_numpy()
    rows = []
    for hour in range(24):
        lo, hi = t0 + hour * 3600, t0 + (hour + 1) * 3600
        qa, qb = (ta >= lo) & (ta < hi), (tb >= lo) & (tb < hi)
        coverage = (
            min(
                len(np.unique((ta[qa] - lo) // 60)), len(np.unique((tb[qb] - lo) // 60))
            )
            / 60
        )
        lag_ts, lag_mid = tb[qb], lag_df["mid"].to_numpy()[qb]
        lag_active = len(np.unique((lag_ts[1:][np.diff(lag_mid) != 0] - lo) // 60)) / 60
        if coverage < HOUR_MIN_COVERAGE or lag_active < HOUR_MIN_LAG_ACTIVE:
            continue
        stats = {}
        for step, lags in ((FINE_STEP_SEC, FINE_LAGS), (1.0, range(0, 3))):
            per = int(round(3600 / step))
            a, b = (m[hour * per : (hour + 1) * per + 1] for m in mids[step])
            if np.isnan(a).any() or np.isnan(b).any():
                break
            got = _hour_stats(np.diff(np.log(a)), np.diff(np.log(b)), lags)
            if got is None:
                break
            stats[step] = got
        else:
            cc, va, vb = stats[FINE_STEP_SEC]
            rho = cc / np.sqrt(va * vb)
            peak = int(np.argmax(rho))
            cc1, va1, vb1 = stats[1.0]
            lead_mid, lag_mid = (
                lead_df["mid"].to_numpy()[qa],
                lag_df["mid"].to_numpy()[qb],
            )
            row = {
                "date": day,
                "hour": hour,
                "peak_lag_ms": int(FINE_LAGS[peak] * 100),
                "peak_rho": rho[peak],
                "rho_sum_0_3s": rho[FINE_LAGS >= 0].sum(),
                "rho0_1s": cc1[0] / np.sqrt(va1 * vb1),
                "rho1_1s": cc1[1] / np.sqrt(va1 * vb1),
                "beta_1s_cum": cc1.sum() / va1,
                "vol_lead_bps_h": np.sqrt(va) * 1e4,
                "vol_lag_bps_h": np.sqrt(vb) * 1e4,
                "lead_updates": int((np.diff(lead_mid) != 0).sum()),
                "lag_updates": int((np.diff(lag_mid) != 0).sum()),
                "lead_spread_bps": float(
                    np.median(lead_df["spread_bps"].to_numpy()[qa])
                ),
                "lag_spread_bps": float(np.median(lag_df["spread_bps"].to_numpy()[qb])),
                "lead_px": float(np.median(lead_mid)),
            }
            for lag_ms in CURVE_LAGS_MS:
                row[curve_column(lag_ms)] = rho[FINE_LAGS == lag_ms // 100][0]
            rows.append(row)
    return rows


def available_days(pair: Pair) -> list[str]:
    """Days on which both markets have MarketPrice at the pair's execution size, sorted."""
    root = ParquetStore(sys_env.get_parquet_root_dir())
    held = []
    for market in (pair.lead, pair.lag):
        base = (
            root.create_execution_price(market, pair.execution_size)
            ._get_data_dir("x")
            .parent
        )
        held.append(
            {d.name[5:] for d in base.glob("date=*") if (d / "data.parquet").exists()}
            if base.exists()
            else set()
        )
    return sorted(held[0] & held[1])


_BLAS_LIMIT = None


def _one_blas_thread():
    """Pool initializer: one BLAS thread per worker, or 16 workers each start one per core and the box thrashes."""
    global _BLAS_LIMIT
    _BLAS_LIMIT = threadpool_limits(1)


def _compute_day_args(args):
    return compute_day(*args)


def compute_days(pair: Pair, days: list[str], workers: int = 16) -> pd.DataFrame:
    """Rows for every day both markets have MarketPrice; missing days are left out."""
    if not days:
        return pd.DataFrame()
    with ProcessPoolExecutor(
        max(1, min(workers, len(days))), initializer=_one_blas_thread
    ) as ex:
        rows = [r for r in ex.map(_compute_day_args, [(pair, d) for d in days]) if r]
    LOG.info(f"Lead-lag {pair.key}: computed {len(rows)} of {len(days)} day(s)")
    return pd.DataFrame(rows)


def _compute_day_hours_args(args):
    return compute_day_hours(*args)


def compute_hours(pair: Pair, days: list[str], workers: int = 16) -> pd.DataFrame:
    """Hourly rows for the given days; hours without enough quotes on both sides are left out."""
    if not days:
        return pd.DataFrame()
    with ProcessPoolExecutor(
        max(1, min(workers, len(days))), initializer=_one_blas_thread
    ) as ex:
        rows = [
            r
            for rs in ex.map(_compute_day_hours_args, [(pair, d) for d in days])
            for r in rs
        ]
    LOG.info(
        f"Lead-lag {pair.key}: computed {len(rows)} hour(s) over {len(days)} day(s)"
    )
    return pd.DataFrame(rows)
