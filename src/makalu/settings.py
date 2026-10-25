"""Where Makalu's data is; environment and credentials come from qate-env.

The datalake root is Komachi's answer (`ROOT_PATH` in ~/.kamakuraquantlab.env):
Kitakamakura writes bronze there, Komachi says where a market-day is, and
Makalu derives silver and gold beside it.
"""
import os

import komachi
from qate_env.sys_env import get_api_keys, get_env_root_dir, get_influxdb, get_secret_dir, read_keys  # noqa: F401

PARQUET_ROOT_DIR = str(komachi.data_root())
MANIFEST_PATH = os.path.join(PARQUET_ROOT_DIR, "manifest.db")

def get_parquet_root_dir():
    return PARQUET_ROOT_DIR

def get_s3_client():
    import boto3
    api_key, secret = get_api_keys("aws-s3")
    return boto3.client("s3", aws_access_key_id=api_key, aws_secret_access_key=secret,
                        region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-1"))

def get_discord(key_name):
    token, channel, errors = read_keys("discord", key_name, "TOKEN", "CHANNEL_ID", "ERRORS_CHANNEL_ID")
    return token, int(channel), int(errors)

def get_ai(name=None):
    return get_api_keys("ai", name or "DEFAULT")[1]
