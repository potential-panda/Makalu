"""
BacktestRunner: writes env config files for Himalaya and publishes run-log results.
"""

from __future__ import annotations

import json
from logging import getLogger
from pathlib import Path

LOG = getLogger(__name__)

_KNOWLEDGE_PATH = Path(__file__).parents[2] / "openclaw" / "knowledge" / "strategies.json"


class BacktestRunner:
    def __init__(self, env_dir: str):
        self._env_dir = Path(env_dir)
        with open(_KNOWLEDGE_PATH, encoding="utf-8") as f:
            self._knowledge: dict = json.load(f)

    def _write_env(
        self,
        env_name: str,
        strategy: str,
        variant: str,
        trading_market: str,
        signal_market: str,
        order_size: float,
        magnifier: float,
        param_grid: list,
    ) -> None:
        env_path = self._env_dir / env_name
        env_path.mkdir(parents=True, exist_ok=True)

        trading = {
            "chat_config_key": None,
            "gateway_name": "SIMULATOR",
            "strategy_module_name": strategy,
            "trading_config_key": "DEFAULT",
            "variant": variant,
        }
        config = {
            "magnifier": magnifier,
            "market": trading_market,
            "order_size": order_size,
            "ref_market": signal_market,
        }

        for fname, data in [
            ("trading.json", trading),
            ("config.json", config),
            ("param_grid.json", param_grid),
        ]:
            target = env_path / fname
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            LOG.info("Wrote %s", target)

    def _publish_run_logs(self, env_name: str) -> int:
        run_log_dir = self._env_dir / env_name / "run_log"
        results_dir = self._env_dir / "KILIMANJARO_API" / "results"

        if not run_log_dir.exists():
            LOG.info("No run_log dir at %s, nothing to publish", run_log_dir)
            return 0

        results_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        for src in run_log_dir.glob("*.json"):
            try:
                with open(src, encoding="utf-8") as f:
                    run_log = json.load(f)
                bucket_name = run_log.get("bucket_name")
                if not bucket_name:
                    LOG.warning("No bucket_name in %s, skipping", src.name)
                    continue
                dest = results_dir / f"{bucket_name}.json"
                with open(dest, "w", encoding="utf-8") as f:
                    json.dump(run_log, f, indent=2)
                count += 1
                LOG.info("Published: %s", dest.name)
            except Exception as e:
                LOG.warning("Could not publish %s: %s", src.name, e)

        LOG.info("Published %s run-log(s) to %s", count, results_dir)
        return count


def _extract_asset(market: str) -> str:
    symbol = market.split(":")[1]
    return symbol.split("_")[0]


def _resolve_order_size(k: dict, asset: str) -> float:
    order_size = k.get("order_size_by_asset") or k.get("order_size")
    if isinstance(order_size, dict):
        if asset not in order_size:
            raise ValueError(f"No order_size defined for asset '{asset}'. Known: {list(order_size)}")
        return order_size[asset]
    return float(order_size)
