"""Exercise the application with the current qate models and local storage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest
from qate.core.model import ExchangeName, Market, Side, Symbol
from qate.util.dt_range import DtRange

from makalu.storage.parquet_store import OrderBookStore, ParquetStore, TradeStore


def test_bronze_decode_and_empty_window(tmp_path):
    market = Market(ExchangeName.BINANCE, Symbol.BTC_USDT)
    window = DtRange.from_strings("2026-03-02 12:00:00", "2026-03-02 12:01:00")
    trades = TradeStore(str(tmp_path / "bronze"), market)
    assert trades.read_trades(window) == []
    path = trades._get_parquet_path(window.days[0])
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"ts": window.start_ts - 1, "side": 0, "price": 99.0, "size": 1.0},
            {"ts": window.start_ts, "side": 1, "price": 100.0, "size": 0.2},
        ]
    ).to_parquet(path, index=False)
    decoded = trades.read_trades(window)
    assert len(decoded) == 1
    assert decoded[0].side == Side.SELL
    assert decoded[0].market == market
    assert decoded[0].price == 100.0
    books = OrderBookStore(str(tmp_path / "bronze"), market, depth=1)
    path = books._get_parquet_path(window.days[0])
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts": window.start_ts,
                "bid0_price": 99.0,
                "bid0_qty": 0.2,
                "ask0_price": 101.0,
                "ask0_qty": 0.3,
            }
        ]
    ).to_parquet(path, index=False)
    book = books.read_order_books(window)[0]
    assert book.bids[0].price == 99.0
    assert book.asks[0].amount == 0.3


def test_credentials_use_qate_env_settings(tmp_path, monkeypatch):
    from makalu import settings

    monkeypatch.setenv("QATE_SECRET_DIR", str(tmp_path))
    (tmp_path / "aws-s3.keys").write_text(
        "[DEFAULT]\nAPI_KEY=test-key\nSECRET=test-secret\n"
    )
    assert settings.get_api_keys("aws-s3") == ("test-key", "test-secret")
    import qate

    assert all("potential-panda-core" not in str(path) for path in qate.__path__)


def test_bronze_is_read_through_komachi_and_never_written(tmp_path):
    from komachi.layout import data_path

    market = Market(ExchangeName.GMO, Symbol.BTC_JPY)
    trades = ParquetStore(str(tmp_path)).create_trade(market)
    assert trades._get_parquet_path("2026-03-02") == data_path(
        tmp_path, "GMO:BTC_JPY", "Trade", "2026-03-02"
    )
    with pytest.raises(TypeError):
        trades.add(0.0, {"side": 0, "price": 1.0, "size": 1.0})


def test_datalake_root_is_komachis():
    import komachi

    from makalu import settings

    assert settings.get_parquet_root_dir() == str(komachi.data_root())
