"""Local Python startup customizations for Makalu entrypoints."""

from __future__ import annotations

import configparser
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QATE_SRC = ROOT / "KQL" / "qate" / "src"
QATE_ENV_SRC = ROOT / "KQL" / "qate-env" / "src"
QATE_MARKET_DATA_SRC = ROOT / "qate-market-data" / "src"
POTENTIAL_PANDA_QATE = ROOT / "potential-panda-core" / "src" / "qate"


def _install_qate_env_alias() -> None:
    for path in (QATE_SRC, QATE_ENV_SRC, QATE_MARKET_DATA_SRC):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        import qate_env.env as env
        import qate_env.env_name as env_name
        import qate_env.sys_env as sys_env
    except ImportError:
        return

    sys_env.MARKET_DATA_DIR = "/panda/market-data"
    sys_env.PARQUET_ROOT_DIR = "/panda/makalu-data"
    sys_env.SECRET_DIR = Path.home() / "secret"
    sys_env.DEFAULT_SECRET_DIR = sys_env.SECRET_DIR
    sys_env.AWS_REGION = "ap-northeast-1"

    def get_market_data_dir():
        return sys_env.MARKET_DATA_DIR

    def get_parquet_root_dir():
        return sys_env.PARQUET_ROOT_DIR

    def get_s3_client():
        import boto3

        api_key, secret = sys_env.get_api_keys("aws-s3", "DEFAULT")
        return boto3.client(
            "s3",
            aws_access_key_id=api_key,
            aws_secret_access_key=secret,
            region_name=sys_env.AWS_REGION,
        )

    def get_discord(key_name: str):
        config = configparser.ConfigParser()
        config.read(os.path.join(sys_env.SECRET_DIR, "discord.keys"))
        return (
            config[key_name]["TOKEN"],
            int(config[key_name]["CHANNEL_ID"]),
            int(config[key_name]["ERRORS_CHANNEL_ID"]),
        )

    def get_ai(name: str | None = None):
        _, secret = sys_env.get_api_keys("ai", "DEFAULT")
        return secret

    sys_env.get_market_data_dir = get_market_data_dir
    sys_env.get_parquet_root_dir = get_parquet_root_dir
    sys_env.get_s3_client = get_s3_client
    sys_env.get_discord = get_discord
    sys_env.get_ai = get_ai

    package = types.ModuleType("qate.env")
    package.__path__ = []
    package.env = env
    package.env_name = env_name
    package.sys_env = sys_env

    sys.modules.setdefault("qate.env", package)
    sys.modules.setdefault("qate.env.env", env)
    sys.modules.setdefault("qate.env.env_name", env_name)
    sys.modules.setdefault("qate.env.sys_env", sys_env)


def _extend_qate_store_namespace() -> None:
    if not POTENTIAL_PANDA_QATE.exists():
        return
    try:
        import qate
    except ImportError:
        return

    qate_path = getattr(qate, "__path__", None)
    if qate_path is not None and str(POTENTIAL_PANDA_QATE) not in qate_path:
        qate_path.append(str(POTENTIAL_PANDA_QATE))


def _install_qate_core_metric_alias() -> None:
    try:
        import qate.util.buffer as buffer
    except ImportError:
        return

    metric = types.ModuleType("qate.core.metric")
    metric.BufferedWriter = buffer.BufferedWriter
    sys.modules.setdefault("qate.core.metric", metric)


_install_qate_env_alias()
_extend_qate_store_namespace()
_install_qate_core_metric_alias()
