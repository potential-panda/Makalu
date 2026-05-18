from __future__ import annotations

import json

from makalu.infra import docker_observability


def register_cli(subparsers) -> None:
    p = subparsers.add_parser("docker-observability", help="Check Grafana and InfluxDB Docker stack health")
    p.add_argument("--format", choices=["json", "text"], default="json")
    p.set_defaults(func=run)


def run(args) -> None:
    result = docker_observability()
    if args.format == "text":
        status = "healthy" if result["healthy"] else "UNHEALTHY"
        print(f"Docker stack: {status}")
        print(f"  InfluxDB (8086): {'up' if result['influxdb_up'] else 'DOWN'}")
        print(f"  Grafana  (3000): {'up' if result['grafana_up'] else 'DOWN'}")
        for w in result.get("warnings", []):
            print(f"  WARNING: {w}")
    else:
        print(json.dumps(result, indent=2, default=str))
