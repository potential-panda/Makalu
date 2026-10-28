# Local Infrastructure

## 1 Docker Stack: Grafana + InfluxDB

Defined in `/panda-infra/Infra/docker/grafana/docker-compose.yml`.

| Service  | Port | Purpose |
|----------|------|---------|
| InfluxDB | 8086 | Time-series DB — receives exported datasets and trading metrics |
| Grafana  | 3000 | Dashboard — reads from InfluxDB |

### 1.1 Check health

```bash
python /home/lulurun/workspace/Makalu/src/cli.py docker-observability --format text
# JSON output (for programmatic use):
python /home/lulurun/workspace/Makalu/src/cli.py docker-observability
```

Output fields: `influxdb_up`, `grafana_up`, `healthy`, `warnings`, `docker_ps`.

### 1.2 Start the stack (if down)

```bash
cd /panda-infra/Infra/docker/grafana && docker compose up -d
```

### 1.3 When InfluxDB must be up

- `grafana-export` — pushes Parquet data to InfluxDB
- `metrics-data` — pushes prod trading metrics to InfluxDB

### 1.4 When Grafana must be up

- Viewing any Grafana dashboard
- `grafana-export` returns a Grafana URL — only useful if Grafana is running

---

## 2 Key Paths

| Resource | Path |
|----------|------|
| Makalu CLI | `/home/lulurun/workspace/Makalu/src/cli.py` |
| Parquet warehouse | `/panda/makalu-data/` |
| Dataset catalog | `/home/lulurun/workspace/Makalu/doc/catalog.yaml` |
| Dataset definitions | `/home/lulurun/workspace/Makalu/doc/dataset_definitions.md` |
| Docker compose | `/panda-infra/Infra/docker/grafana/` |
