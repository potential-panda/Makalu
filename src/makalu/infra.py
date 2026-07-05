"""
Local infrastructure health checks — Grafana and InfluxDB Docker stack.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import Any

_COMPOSE_DIR = Path("/panda-infra/Infra/docker/grafana")
_INFLUXDB_PORT = 8086
_GRAFANA_PORT = 3000


def _run(args: list[str], *, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout)


def _file_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
    }


def _port_open(port: int) -> bool:
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def docker_observability() -> dict[str, Any]:
    result: dict[str, Any] = {
        "compose_dir": str(_COMPOSE_DIR),
        "compose_file": _file_summary(_COMPOSE_DIR / "docker-compose.yml"),
        "data_dirs": {
            "grafana": _file_summary(_COMPOSE_DIR / "data" / "grafana"),
            "influxdb2": _file_summary(_COMPOSE_DIR / "data" / "influxdb2"),
            "influxdb2_config": _file_summary(
                _COMPOSE_DIR / "data" / "influxdb2-config"
            ),
        },
    }

    try:
        cp = _run(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"])
        result["docker_ps"] = [line for line in cp.stdout.splitlines() if line.strip()]
        result["docker_ps_stderr"] = cp.stderr.strip()
    except Exception as e:
        result["docker_error"] = str(e)

    result["influxdb_up"] = _port_open(_INFLUXDB_PORT)
    result["grafana_up"] = _port_open(_GRAFANA_PORT)
    result["healthy"] = result["influxdb_up"] and result["grafana_up"]

    warnings = []
    if not result["influxdb_up"]:
        warnings.append(f"InfluxDB (port {_INFLUXDB_PORT}) is not reachable")
    if not result["grafana_up"]:
        warnings.append(f"Grafana (port {_GRAFANA_PORT}) is not reachable")
    result["warnings"] = warnings

    return result
