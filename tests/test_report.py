"""The lead-lag report: the numbers on a pair with a known lag, the site build, and the file server."""

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from qate.core.model import Market

from makalu.report import leadlag, site
from makalu.web import SiteHandler

LEAD = Market.from_str("BINANCE:XRP_USDT")
LAG = Market.from_str("GMO:XRP_JPY")
PAIR = leadlag.Pair(LEAD, LAG, "150")


def _synthetic(lag_steps: int, n: int = 200_000, seed: int = 0) -> dict:
    """Two mid series on a 100ms grid where the lag market copies the lead `lag_steps` buckets later."""
    rng = np.random.default_rng(seed)
    ts = 1_760_000_000 + np.arange(n) * 0.1
    lead_mid = 2.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    lag_mid = np.empty(n)
    lag_mid[lag_steps:] = lead_mid[:-lag_steps] * 300
    lag_mid[:lag_steps] = lead_mid[0] * 300
    lag_mid *= np.exp(rng.normal(0, 2e-5, n))

    def frame(mid):
        return pd.DataFrame({"ts": ts, "mid": mid, "spread_bps": np.full(n, 1.0)})

    return {LEAD.id: frame(lead_mid), LAG.id: frame(lag_mid)}


@pytest.fixture
def fake_market_data(monkeypatch):
    data = _synthetic(lag_steps=5)
    monkeypatch.setattr(
        leadlag, "_read_day", lambda market, size, day: (data[market.id], 0)
    )
    monkeypatch.setattr(leadlag, "_tick", lambda market, day: 0.0001)
    return data


def test_known_lag_is_found(fake_market_data):
    row = leadlag.compute_day(PAIR, "2026-01-01")
    assert row["peak_lag_ms"] == 500
    assert row["peak_rho"] > 0.9
    assert row["rho_lag_500"] == pytest.approx(row["peak_rho"])
    assert abs(row["rho0"]) < 0.1
    assert row["beta_1s_cum"] == pytest.approx(1.0, abs=0.05)
    assert row["lead_tick_bps"] == pytest.approx(0.0001 / row["lead_px"] * 1e4)


def test_known_lag_is_found_in_every_full_hour(fake_market_data):
    rows = leadlag.compute_day_hours(PAIR, "2026-01-01")
    # The synthetic day starts at 17:53 JST and runs 5.5 hours: 18:00..22:00 are full, the partial ends are left out.
    assert [r["hour"] for r in rows] == [18, 19, 20, 21, 22]
    for r in rows:
        assert r["peak_lag_ms"] == 500
        assert r["peak_rho"] > 0.9
        assert r["beta_1s_cum"] == pytest.approx(
            1.0, abs=0.1
        )  # 3,600 one-second returns: noisier than a day
        assert r["vol_lead_bps_h"] == pytest.approx(
            1e-4 * np.sqrt(36_000) * 1e4, rel=0.05
        )
        assert r["rho_lag_500"] == pytest.approx(r["peak_rho"])


def test_frozen_lag_hour_is_left_out(monkeypatch):
    """GMO maintenance: snapshots keep arriving, the price does not move. That hour is not a real hour of trading."""
    data = _synthetic(lag_steps=5)
    lag = data[LAG.id].copy()

    def jst_hour(t):
        return int(((t + 9 * 3600) % 86400) // 3600)

    frozen = lag["ts"].map(jst_hour) == 20
    lag.loc[frozen, "mid"] = lag.loc[frozen, "mid"].iloc[0]
    data = {LEAD.id: data[LEAD.id], LAG.id: lag}
    monkeypatch.setattr(
        leadlag, "_read_day", lambda market, size, day: (data[market.id], 0)
    )
    assert [r["hour"] for r in leadlag.compute_day_hours(PAIR, "2026-01-01")] == [
        18,
        19,
        21,
        22,
    ]


def test_missing_side_is_skipped(monkeypatch):
    monkeypatch.setattr(
        leadlag, "_read_day", lambda market, size, day: (pd.DataFrame(), 0)
    )
    assert leadlag.compute_day(PAIR, "2026-01-01") is None
    assert leadlag.compute_day_hours(PAIR, "2026-01-01") == []


def test_build_reuses_cached_days(tmp_path, monkeypatch):
    calls = []

    def fake_compute(pair, days, workers=16):
        calls.append(list(days))
        return pd.DataFrame(
            [
                {"date": d, "peak_lag_ms": 600, "rho1_1s": 0.5, "lead_tick": 0.0001}
                for d in days
            ]
        )

    hour_calls = []

    def fake_hours(pair, days, workers=16):
        hour_calls.append(list(days))
        return pd.DataFrame(
            [
                {"date": d, "hour": h, "rho1_1s": 0.4, "lead_px": 2.0, "rho_lag_0": 0.1}
                for d in days
                for h in (0, 1)
            ]
        )

    monkeypatch.setattr(leadlag, "compute_days", fake_compute)
    monkeypatch.setattr(leadlag, "compute_hours", fake_hours)
    meta = site.build_leadlag(PAIR, "2026-01-01", "2026-01-03", tmp_path)
    assert calls == [["2026-01-01", "2026-01-02", "2026-01-03"]]
    site.build_leadlag(PAIR, "2026-01-01", "2026-01-05", tmp_path)
    assert calls[-1] == ["2026-01-04", "2026-01-05"]
    assert hour_calls == [
        ["2026-01-01", "2026-01-02", "2026-01-03"],
        ["2026-01-04", "2026-01-05"],
    ]

    out = tmp_path / "leadlag" / meta["key"]
    assert json.loads((out / "report.json").read_text())["days"] == 5
    assert len(pd.read_csv(out / "daily.csv")) == 5
    assert len(pd.read_csv(out / "hourly.csv")) == 10
    assert json.loads((out / "report.json").read_text())["hours"] == 10
    assert pd.read_csv(out / "hourly.csv")["lead_tick_bps"].iloc[0] == pytest.approx(
        0.5
    )
    assert (tmp_path / "static" / "vendor" / "d3-7.9.0.min.js").exists()
    assert not (tmp_path / "static" / "leadlag.template.html").exists()
    assert (
        'href="leadlag/BINANCE_XRP_USDT__GMO_XRP_JPY/"'
        in (tmp_path / "index.html").read_text()
    )
    # Every link in a report page is relative, because nginx mounts the site under /makalu/.
    assert (
        'src="/' not in (out / "index.html").read_text()
        and 'href="/' not in (out / "index.html").read_text()
    )


@pytest.fixture
def server(tmp_path):
    (tmp_path / "index.html").write_text("front")
    (tmp_path / "leadlag").mkdir()
    from functools import partial
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SiteHandler, directory=str(tmp_path))
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _status(url: str) -> int:
    try:
        return urllib.request.urlopen(url).status
    except urllib.error.HTTPError as e:
        return e.code


def test_server_serves_files_only(server):
    assert _status(server + "/") == 200
    assert _status(server + "/leadlag/") == 404  # no directory listings
    assert _status(server + "/../../etc/passwd") == 404  # nothing outside the site
