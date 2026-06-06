"""CLI command: generate trading metrics chart and optionally post to Discord."""

from __future__ import annotations

import argparse
import configparser
import json
import tempfile
from pathlib import Path

import requests

from makalu.trading.data import load_orders, load_pnl
from makalu.trading.chart import plot_trading_metrics

_DISCORD_KEYS = Path.home() / "secret" / "discord.keys"

_ENV_BUCKET = {
    "ASHLEY_XEON": "panda-cc-pisces-01",
    "ATSUSHI_EGAWA": "panda-cc-pisces-01",
    "QIN_TENGFEI": "panda-cc-orion-01",
    "QASIM_TARIQ": "panda-cc-orion-02",
}


def _read_webhook(key: str) -> str | None:
    if not _DISCORD_KEYS.exists():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(_DISCORD_KEYS)
    return cfg.get(key, "TOKEN", fallback=None)


def _post_to_discord(webhook_url: str, path: str, content: str):
    with open(path, "rb") as f:
        resp = requests.post(
            webhook_url,
            data={"content": content},
            files={"file": ("chart.png", f, "image/png")},
            timeout=15,
        )
    resp.raise_for_status()


def execute(args) -> None:
    env = args.env
    bucket = _ENV_BUCKET.get(env)
    if not bucket:
        print(json.dumps({"error": f"Unknown env '{env}'. Known: {list(_ENV_BUCKET)}"}))
        return

    order_df = load_orders(env, bucket)
    pnl_df = load_pnl(env, bucket)

    n_orders = len(order_df)
    n_pnl = len(pnl_df)
    taker_count = len(order_df[order_df["order_type"] == "DEFAULT"]) if n_orders else 0

    title = f"{env} — trading metrics"
    if not order_df.empty:
        start = order_df["ts"].min().strftime("%Y-%m-%d")
        end = order_df["ts"].max().strftime("%Y-%m-%d")
        title += f"  ({start} → {end})"

    fig = plot_trading_metrics(order_df, pnl_df, title=title)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=130, bbox_inches="tight")
        chart_path = tmp.name

    result = {
        "env": env,
        "orders": n_orders,
        "taker_orders": taker_count,
        "pnl_events": n_pnl,
        "chart": chart_path,
    }

    if args.discord_key:
        webhook = _read_webhook(args.discord_key)
        if not webhook:
            result["discord_error"] = f"No webhook found for key '{args.discord_key}'"
        else:
            net = pnl_df["cumulative_net"].iloc[-1] if n_pnl else 0.0
            content = (
                f"**{env}** trading metrics | "
                f"{taker_count} taker fills | "
                f"net PnL: ¥{net:.2f}"
            )
            _post_to_discord(webhook, chart_path, content)
            result["discord"] = "posted"

    print(json.dumps(result))


def register_cli(sub) -> None:
    p = sub.add_parser(
        "trading-chart",
        help="Generate trading metrics chart (slippage, time-to-fill, PnL) and post to Discord",
        description=(
            "Loads ORDER and PNL metrics from local msgpack files, generates a\n"
            "3-panel chart (taker slippage, taker time-to-fill, cumulative PnL),\n"
            "saves it as a PNG, and optionally posts it to a Discord webhook.\n\n"
            "Data source: /panda/metrics-data/\n"
            "Envs: ASHLEY_XEON, ATSUSHI_EGAWA, QIN_TENGFEI, QASIM_TARIQ"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--env",
        required=True,
        choices=list(_ENV_BUCKET),
        help="Environment name to chart.",
    )
    p.add_argument(
        "--discord-key",
        metavar="KEY",
        help="Section name in ~/secret/discord.keys to use for posting (e.g. ELEPHANT).",
    )
    p.set_defaults(func=execute)
