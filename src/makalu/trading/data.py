"""Load ORDER and PNL measurements from local msgpack files into DataFrames."""

from __future__ import annotations

import glob
from pathlib import Path

import msgpack
import pandas as pd

_METRICS_DIR = Path("/panda/metrics-data")


def _iter_msgpack(pattern: str):
    for fname in sorted(glob.glob(pattern)):
        with open(fname, "rb") as f:
            yield from msgpack.Unpacker(f, raw=False)


def load_orders(env_name: str, bucket: str) -> pd.DataFrame:
    """Return DataFrame of ORDER measurements for the given env.

    Columns: ts, exchange, symbol, side, settle_type, order_type,
             price, exec_price, exec_size, slippage, time_to_create, time_to_fill
    """
    from qate.core.model import Measurement, Tag, Field, ExchangeName, Side, SettleType
    from qate.core.order import OrderType
    from qate.core.symbol import Symbol

    ORDER = Measurement.ORDER.value

    rows = []
    pattern = str(_METRICS_DIR / bucket / f"{env_name}_Metrics_*.msgpack*")
    for obj in _iter_msgpack(pattern):
        if obj[0] != ORDER:
            continue
        ts = obj[1]
        tags = {obj[2][i]: obj[2][i + 1] for i in range(0, len(obj[2]), 2)}
        fields = {obj[3][i]: obj[3][i + 1] for i in range(0, len(obj[3]), 2)}
        rows.append({
            "ts": pd.Timestamp(ts, unit="s", tz="Asia/Tokyo"),
            "exchange": ExchangeName(tags[Tag.EXCHANGE_NAME.value]).name,
            "symbol": Symbol(tags[Tag.SYMBOL.value]).name,
            "side": Side(tags[Tag.SIDE.value]).name,
            "settle_type": SettleType(tags[Tag.SETTLE_TYPE.value]).name,
            "order_type": OrderType(tags[Tag.ORDER_TYPE.value]).name,
            "price": fields.get(Field.PRICE.value),
            "exec_price": fields.get(Field.EXEC_PRICE.value),
            "exec_size": fields.get(Field.EXEC_SIZE.value),
            "slippage": fields.get(Field.SLIPPAGE.value),
            "time_to_create": fields.get(Field.TIME_TO_CREATE.value),
            "time_to_fill": fields.get(Field.TIME_TO_FILL.value),
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


def load_pnl(env_name: str, bucket: str) -> pd.DataFrame:
    """Return DataFrame of PNL measurements (CLOSE legs only) for the given env.

    Columns: ts, exchange, symbol, side, settle_type, pnl, fee, pnl_return
    """
    from qate.core.model import Measurement, Tag, Field, ExchangeName, Side, SettleType
    from qate.core.symbol import Symbol

    PNL = Measurement.PNL.value

    rows = []
    pattern = str(_METRICS_DIR / bucket / f"{env_name}_Metrics_*.msgpack*")
    for obj in _iter_msgpack(pattern):
        if obj[0] != PNL:
            continue
        ts = obj[1]
        tags = {obj[2][i]: obj[2][i + 1] for i in range(0, len(obj[2]), 2)}
        fields = {obj[3][i]: obj[3][i + 1] for i in range(0, len(obj[3]), 2)}
        settle_raw = tags.get(Tag.SETTLE_TYPE.value)
        if settle_raw is None:
            continue
        settle_type = SettleType(settle_raw).name
        if settle_type != "CLOSE":
            continue
        side_raw = tags.get(Tag.SIDE.value)
        rows.append({
            "ts": pd.Timestamp(ts, unit="s", tz="Asia/Tokyo"),
            "exchange": ExchangeName(tags[Tag.EXCHANGE_NAME.value]).name,
            "symbol": Symbol(tags[Tag.SYMBOL.value]).name,
            "side": Side(side_raw).name if side_raw is not None else "UNKNOWN",
            "settle_type": settle_type,
            "pnl": fields.get(Field.PNL.value, 0.0),
            "fee": fields.get(Field.FEE.value, 0.0),
            # Files written before qate packed Field.RETURN carry the key "return".
            "pnl_return": fields.get(Field.RETURN.value, fields.get("return", 0.0)),
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["cumulative_net"] = (df["pnl"] + df["fee"]).cumsum()
    return df
