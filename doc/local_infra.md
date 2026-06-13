# Local Infrastructure

## Docker Stack: Grafana + InfluxDB

Defined in `/panda-infra/Infra/docker/grafana/docker-compose.yml`.

| Service  | Port | Purpose |
|----------|------|---------|
| InfluxDB | 8086 | Time-series DB — receives backtest metrics and trading metrics |
| Grafana  | 3000 | Dashboard — reads from InfluxDB |

### Check health

```bash
python /home/lulurun/workspace/Makalu/src/cli.py docker-observability --format text
# JSON output (for programmatic use):
python /home/lulurun/workspace/Makalu/src/cli.py docker-observability
```

Output fields: `influxdb_up`, `grafana_up`, `healthy`, `warnings`, `docker_ps`.

### Start the stack (if down)

```bash
cd /panda-infra/Infra/docker/grafana && docker compose up -d
```

### When InfluxDB must be up

- `grafana-export` — pushes Parquet data to InfluxDB
- `metrics-data` — pushes prod trading metrics to InfluxDB
- Himalaya backtest — writes backtest metrics to InfluxDB

If InfluxDB is down before a backtest: warn the user, ask whether to start it or proceed anyway (backtest will run but metrics won't be saved).

### When Grafana must be up

- Viewing any Grafana dashboard
- `grafana-export` returns a Grafana URL — only useful if Grafana is running

---

## Key Paths

| Resource | Path |
|----------|------|
| Makalu CLI | `/home/lulurun/workspace/Makalu/src/cli.py` |
| Himalaya runner | `/home/lulurun/workspace/Himalaya/start.sh` |
| Raw market data | `/panda/market-data/` |
| Parquet warehouse | `/panda/makalu-data/` |
| Backtest env dir | `~/env/` |
| Dataset catalog | `/home/lulurun/workspace/Makalu/doc/catalog.yaml` |
| Dataset definitions | `/home/lulurun/workspace/Makalu/doc/dataset_definitions.md` |
| Strategy configs | `/home/lulurun/workspace/Makalu/openclaw/knowledge/strategies.json` |
| Docker compose | `/panda-infra/Infra/docker/grafana/` |
| Jobs dir | `/panda-infra/makalu/jobs/` |
