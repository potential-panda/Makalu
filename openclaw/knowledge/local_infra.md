# Local Infrastructure

## Docker Stack: Grafana + InfluxDB

Both services are defined in `/panda-infra/Infra/docker/grafana/docker-compose.yml`.

| Service  | Port | Purpose                                      |
|----------|------|----------------------------------------------|
| InfluxDB | 8086 | Time-series DB — receives backtest metrics and trading metrics |
| Grafana  | 3000 | Visualization dashboard — reads from InfluxDB |

### Check health

```bash
python /home/lulurun/workspace/Makalu/cli.py docker-observability --format text
```

Output includes: `influxdb_up`, `grafana_up`, `healthy`, `warnings`, `docker ps` snapshot.

### Start the stack (if down)

```bash
cd /panda-infra/Infra/docker/grafana
docker compose up -d
```

### When InfluxDB must be up

- `grafana-export` — pushes Parquet data to InfluxDB
- `metrics-data` — pushes prod trading metrics to InfluxDB
- Himalaya backtest — writes backtest metrics to InfluxDB alongside Parquet

### When Grafana must be up

- Viewing any dashboard
- `grafana-export` returns a Grafana URL — the URL is only useful if Grafana is running

## Data Roots

From `qate.env.sys_env`:

- Raw market data: `/panda/market-data/`
- Parquet warehouse: `/panda/makalu-data/`
- Env folders: `~/env/`
