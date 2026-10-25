# Reports — static pages served from Makalu

Makalu builds read-only HTML reports from the warehouse and serves them as
static files. There is no API: `report` writes files, `serve` hands them out.

```
python src/cli.py report leadlag ...   →  ~/env/LUIGI_MARIO/site/   ←  python src/cli.py serve (127.0.0.1:9776)
                                                                      ←  nginx on pisces-01 :7700/makalu/ via the eiger tunnel
```

Public URL: `http://157.120.33.185:7700/makalu/`. nginx strips the `/makalu/`
prefix, so every link inside the site is relative. The site has **no
authentication**: anyone with the URL can read the reports and their CSVs.

## Site layout

| Path | What |
|------|------|
| `index.html` | list of reports, rewritten on every build |
| `static/` | page code, `style.css`, vendored `d3-7.9.0.min.js` (copied from `src/makalu/report/static/`) |
| `leadlag/<LEAD>__<LAG>/index.html` | one lead-lag report |
| `leadlag/<LEAD>__<LAG>/report.json` | markets, execution size, date range, comparison periods |
| `leadlag/<LEAD>__<LAG>/daily.csv` | the report's numbers, one row per JST day; also the build cache |
| `leadlag/<LEAD>__<LAG>/hourly.csv` | the same per JST hour (sections 6–11, and 1–4 under "Per hour"); also a build cache |

## report leadlag

Daily lead-lag between two markets from silver `MarketPrice` (both markets,
same execution size). Code: `src/makalu/report/leadlag.py`.

```bash
python src/cli.py report leadlag --pair xrp                       # first day with MarketPrice on both sides .. today-2
python src/cli.py report leadlag --lead-market BINANCE:XRP_USDT --lag-market GMO:XRP_JPY \
  --start 2025-07-01 --end 2026-06-29 \
  --compare 'Q4 2025:2025-10-01:2025-12-31,Q2 2026:2026-04-01:2026-06-30'
python src/cli.py report leadlag --pair xrp --recompute           # ignore the cache
```

- Only days missing from `daily.csv` / `hourly.csv` are computed (a few seconds per year of days on 16 workers;
  each worker is held to one BLAS thread, without which 16 workers oversubscribe the box and a year takes 10+ minutes).
  After a `hase derive MarketPrice --recreate`, rebuild with `--recompute`, or the old numbers stay.
- Days without `MarketPrice` on either side are left out.
- Tick size is read from the bronze order book's best bid (through Komachi), since
  MarketPrice's bid is an execution price, not a quote.

Columns in `daily.csv`:

| Column | Meaning |
|--------|---------|
| `peak_lag_ms`, `peak_rho` | lag (ms, lead first) with the highest 100ms cross-correlation, and that correlation |
| `rho0`, `rho_lag_<k>` / `rho_lag_m<k>` | 100ms cross-correlation at lag 0 and at ±k ms |
| `rho_sum_0_3s`, `rho_sum_neg` | sum of 100ms ρ over 0..3s, and over negative lags (lag market leading) |
| `beta_<h>s` | 100ms impulse response: share of a lead move reflected after h s |
| `rho0_1s`, `rho1_1s`, `beta_1s_cum` | 1s grid: ρ same second, ρ next second, pass-through over 0–3s |
| `rho0_5s`, `rho1_5s`, `beta_5s_cum` | 5s grid: the same, pass-through over 0–10s |
| `vol_lead_bps_d`, `vol_lag_bps_d` | realised volatility of 100ms returns, bps per day |
| `lead_px`, `lead_tick`, `lead_tick_bps` | lead market median mid, tick, tick in bps |
| `lead_updates`, `lag_updates` | mid-price changes per day |
| `lead_spread_bps`, `lag_spread_bps` | median MarketPrice spread at the execution size |

The 100ms numbers are sensitive to the lead's tick size; compare periods on the 1s and 5s ones.

Columns in `hourly.csv` (one row per `date` and JST `hour`, 0–23), measured inside that hour with
returns de-meaned per hour; an hour is left out when either market has quotes in fewer than 54 of its 60 minutes,
or when the lag market's mid changes in fewer than 55 of them — GMO's weekly maintenance (Sat ~9:00–11:00 JST)
keeps publishing frozen quotes, which would otherwise pass as real hours with ρ near 0:

| Column | Meaning |
|--------|---------|
| `peak_lag_ms`, `peak_rho`, `rho_sum_0_3s`, `rho_lag_<k>` / `rho_lag_m<k>` | as in `daily.csv`, on the hour's 100ms returns |
| `rho0_1s`, `rho1_1s`, `beta_1s_cum` | as in `daily.csv`, on the hour's 1s returns (3,600 of them: noisier than a day) |
| `vol_lead_bps_h`, `vol_lag_bps_h` | realised volatility of 100ms returns, bps per hour |
| `lead_updates`, `lag_updates` | mid-price changes in the hour |
| `lead_spread_bps`, `lag_spread_bps` | median MarketPrice spread in the hour |
| `lead_px`, `lead_tick_bps` | lead median mid in the hour; the day's tick (from `daily.csv`) over it, in bps |

The page's "Per day / Per hour" switch (default: per hour) chooses which file sections 1–4 are drawn from.
Per hour, peak lag is the mean over hours rather than the median, since it moves in 100ms steps.
A `hourly.csv` written before a column above existed is recomputed whole on the next build.

## serve

```bash
python src/cli.py serve                      # 127.0.0.1:9776, site ~/env/LUIGI_MARIO/site
systemctl --user status makalu-web           # unit: deploy/makalu-web.service
```

Install or update the unit:

```bash
cp deploy/makalu-web.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now makalu-web
```

Rebuilding a report needs no restart; the server reads files on each request.
If the public URL times out while `curl localhost:9776` works, check the tunnel
(`systemctl --user status eiger-tunnel`, see `Eiger/doc/prod_infra.md`).
