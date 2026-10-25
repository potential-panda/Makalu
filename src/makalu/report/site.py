"""The static report site: what `report` writes and `serve` serves.

Everything is plain files under one folder, so any web server can host it:

    <site>/index.html                       list of reports
    <site>/static/...                       page code, styles, vendored d3
    <site>/leadlag/<pair>/index.html        one lead-lag report
    <site>/leadlag/<pair>/report.json       what the report is about
    <site>/leadlag/<pair>/daily.csv         its numbers, one row per day
    <site>/leadlag/<pair>/hourly.csv        the same per JST hour

Every link is relative, because the site is reached through a proxy that
mounts it under a path prefix (/makalu/ on the public host).

`daily.csv` and `hourly.csv` double as the cache: a rebuild computes only the
days they do not already hold, unless asked to recompute.
"""

from __future__ import annotations

import html
import json
import shutil
from datetime import date, datetime, timedelta, timezone
from logging import getLogger
from pathlib import Path

import pandas as pd

from makalu import settings as sys_env
from makalu.report import leadlag

LOG = getLogger(__name__)

STATIC_SRC = Path(__file__).parent / "static"
DEFAULT_COMPARISONS = [
    {"id": "q", "a": ["Q4 2025", "2025-10-01", "2025-12-31"], "b": ["Q2 2026", "2026-04-01", "2026-06-30"]},
    {"id": "h", "a": ["Jul–Dec 2025", "2025-07-01", "2025-12-31"], "b": ["Jan–Jun 2026", "2026-01-01", "2026-06-30"]},
]


def default_site_dir() -> Path:
    return Path(sys_env.get_env_root_dir()) / "LUIGI_MARIO" / "site"


def _days(start: str, end: str) -> list[str]:
    d, e = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while d <= e:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def install_static(site: Path) -> None:
    """Copy the page code into the site, replacing what was there."""
    dst = site / "static"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(STATIC_SRC, dst, ignore=shutil.ignore_patterns("*.template.html"))


def _cached(path: Path, wanted: list[str], recompute: bool, needs: list[str] = ()) -> tuple[pd.DataFrame, list[str]]:
    """Rows of a cache CSV inside the wanted days, and the wanted days it does not hold.

    A cache written before a column in `needs` existed is dropped whole, so every day is recomputed.
    """
    existing = pd.DataFrame()
    if path.exists() and not recompute:
        existing = pd.read_csv(path, dtype={"date": str})
        if not set(needs) <= set(existing.columns):
            LOG.info(f"{path.name} predates column(s) {sorted(set(needs) - set(existing.columns))}; recomputing it")
            existing = pd.DataFrame()
        else:
            existing = existing[existing["date"].isin(wanted)]
    return existing, sorted(set(wanted) - set(existing.get("date", [])))


def _merged(existing: pd.DataFrame, fresh: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = pd.concat([existing, fresh], ignore_index=True) if not fresh.empty else existing
    return out.sort_values(keys).round(6) if not out.empty else out


def build_leadlag(pair: leadlag.Pair, start: str, end: str, site: Path,
                  comparisons: list[dict] | None = None, recompute: bool = False) -> dict:
    out = site / "leadlag" / pair.key
    out.mkdir(parents=True, exist_ok=True)
    wanted = _days(start, end)

    existing, todo = _cached(out / "daily.csv", wanted, recompute)
    LOG.info(f"Lead-lag {pair.key}: {len(existing)} day(s) cached, {len(todo)} to compute")
    daily = _merged(existing, leadlag.compute_days(pair, todo), ["date"])
    if daily.empty:
        raise SystemExit(f"No MarketPrice for {pair.lead.id} and {pair.lag.id} at execution size "
                         f"{pair.execution_size} between {start} and {end}.")
    daily.to_csv(out / "daily.csv", index=False)

    # Only days the daily table holds can have hours; a day with no usable hour is recomputed next time.
    existing, todo = _cached(out / "hourly.csv", list(daily["date"]), recompute,
                             needs=["lead_px", leadlag.curve_column(0)])
    LOG.info(f"Lead-lag {pair.key}: hours cached for {existing['date'].nunique() if len(existing) else 0} day(s), {len(todo)} to compute")
    hourly = _merged(existing, leadlag.compute_hours(pair, todo), ["date", "hour"])
    if not hourly.empty:
        # the tick is read once per day from bronze; an hour's tick in bps is that over the hour's own price
        hourly["lead_tick_bps"] = (hourly["date"].map(daily.set_index("date")["lead_tick"]) / hourly["lead_px"] * 1e4).round(6)
    hourly.to_csv(out / "hourly.csv", index=False)

    meta = {
        "kind": "leadlag",
        "key": pair.key,
        "title": f"{pair.lead.symbol.name.split('_')[0]} lead-lag, {pair.lead.exchange_name.name} → {pair.lag.exchange_name.name}",
        "lead": pair.lead.id,
        "lag": pair.lag.id,
        "execution_size": pair.execution_size,
        "first_date": daily["date"].iloc[0],
        "last_date": daily["date"].iloc[-1],
        "days": int(len(daily)),
        "hours": int(len(hourly)),
        "comparisons": comparisons or DEFAULT_COMPARISONS,
        "curve_lags_ms": list(leadlag.CURVE_LAGS_MS),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "report.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    page = (STATIC_SRC / "leadlag.template.html").read_text()
    (out / "index.html").write_text(page.replace("{{TITLE}}", html.escape(meta["title"])))
    install_static(site)
    write_index(site)
    LOG.info(f"Wrote {out} ({meta['days']} days, {meta['hours']} hours, {meta['first_date']} → {meta['last_date']})")
    return meta


def write_index(site: Path) -> None:
    """The site's front page: one link per report found on disk."""
    reports = []
    for meta_path in sorted(site.glob("*/*/report.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            LOG.warning(f"Skipping unreadable {meta_path}: {e}")
            continue
        href = meta_path.parent.relative_to(site).as_posix() + "/"
        reports.append((meta, href))
    items = "\n".join(
        f'      <li><a href="{html.escape(href)}">{html.escape(m["title"])}</a>'
        f'<span>{html.escape(m["lead"])} → {html.escape(m["lag"])} · {m["first_date"]} to {m["last_date"]} · '
        f'updated {html.escape(m["generated_at"][:16].replace("T", " "))} UTC</span></li>'
        for m, href in reports
    ) or "      <li>No reports yet. Build one with <code>python src/cli.py report leadlag ...</code></li>"
    page = (STATIC_SRC / "index.template.html").read_text()
    (site / "index.html").write_text(page.replace("{{REPORTS}}", items))
