from logging import getLogger

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import WriteOptions, WriteType
from influxdb_client.domain import BucketRetentionRules
from qate.core.model import (
    DataType,
    ExchangeName,
    Field,
    Measurement,
    SettleType,
    Side,
    Tag,
)
from qate.core.order import OrderType
from qate.core.symbol import Symbol
from qate.util.buffer import BufferedWriter

BUCKET = "panda"
ORG = "panda-org"

LOG = getLogger(__name__)

def create_client(url, token):
    return InfluxDBClient(
        url, token, org=ORG, timeout=120_000
    )  # 120s for large batch writes


def normalize_points_timestamp_us(points):
    """
    Mutates a list of InfluxDB Point objects:
    - converts millisecond timestamps to microseconds (if not already US precision)
    - adds +1us offset when timestamps collide to avoid InfluxDB overwrite

    Safe to call multiple times on the same points: if a point is already
    WritePrecision.US it is treated as microseconds and not multiplied again.
    This prevents timestamp overflow when a write fails and the buffer is
    re-flushed by the next successful write cycle.
    """
    last_base_us = None
    seq = 0

    for p in points:
        if p._write_precision == WritePrecision.US:
            base_us = p._time  # already in microseconds — do not multiply
        else:
            base_us = p._time * 1_000  # milliseconds → microseconds

        if base_us == last_base_us:
            seq += 1
        else:
            seq = 0
            last_base_us = base_us

        p.time(base_us + seq, WritePrecision.US)


class InfluxdbWriter(BufferedWriter):
    def __init__(
        self,
        url: str,
        token: str,
        bucket_name: str,
        buffer_size: int = 2048,
        retention_seconds: int | None = None,
        param_set_id: str | None = None,
    ):
        super(InfluxdbWriter, self).__init__(buffer_size)
        self.client = create_client(url, token)
        self.bucket_name = bucket_name
        self.retention_seconds = retention_seconds
        self.param_set_id = param_set_id
        self._init_bucket()
        self.w = self.client.write_api(
            write_options=WriteOptions(
                write_type=WriteType.synchronous,
                batch_size=buffer_size,
            )
        )

    def _init_bucket(self):
        b = self.client.buckets_api()
        bucket = b.find_bucket_by_name(self.bucket_name)
        if not bucket:
            LOG.info(f"CREATE_BUCKET {self.bucket_name}")
            if self.retention_seconds is None:
                b.create_bucket(bucket_name=self.bucket_name)
            else:
                rules = BucketRetentionRules(
                    type="expire", every_seconds=self.retention_seconds
                )
                b.create_bucket(bucket_name=self.bucket_name, retention_rules=rules)

    def write(self, points: list[Point]):
        normalize_points_timestamp_us(points)
        if self.param_set_id:
            for p in points:
                p.tag("param_set_id", self.param_set_id)
        self.w.write(self.bucket_name, record=points, write_precision=WritePrecision.US)

    def close(self):
        super().close()
        self.w.close()


def _regroup_pairs(flat):
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def _get_tag_value(tag: Tag, value: int) -> str:
    if tag == Tag.DATA_TYPE:
        return DataType(value).name
    elif tag == Tag.EXCHANGE_NAME:
        return ExchangeName(value).name
    elif tag == Tag.ORDER_TYPE:
        return OrderType(value).name
    elif tag == Tag.SETTLE_TYPE:
        return SettleType(value).name
    elif tag == Tag.SIDE:
        return Side(value).name
    elif tag == Tag.SYMBOL:
        return Symbol(value).name
    else:
        return str(value)


def convert_to_point(obj: list) -> Point:
    measurement = Measurement(obj[0])
    ts = obj[1]
    tags = _regroup_pairs(obj[2])
    fields = _regroup_pairs(obj[3])
    point = Point(measurement.name).time(int(ts * 1000), WritePrecision.MS)
    for k, v in tags:
        if type(k) is int:
            tag = Tag(k)
            k = tag.name
            v = _get_tag_value(tag, v)
        point.tag(k, v)
    for k, v in fields:
        if type(k) is int:
            k = Field(k).name
        point.field(k, v)
    return point


class InfluxdbStore:
    def __init__(self, config):
        (token, write_url, read_url) = config
        self._token = token
        self._write_url = write_url
        self._read_url = read_url

    def create_writer(
        self,
        bucket_name: str,
        buffer_size=2048,
        retention_seconds: int | None = None,
        param_set_id: str | None = None,
    ):
        return InfluxdbWriter(
            self._write_url,
            self._token,
            bucket_name,
            buffer_size,
            retention_seconds,
            param_set_id,
        )

    def delete_bucket(self, bucket_name: str) -> None:
        """Drop an entire InfluxDB bucket (all data, all time)."""
        client = create_client(self._write_url, self._token)
        try:
            b = client.buckets_api()
            bucket = b.find_bucket_by_name(bucket_name)
            if bucket:
                b.delete_bucket(bucket)
                LOG.info(f"DELETE_BUCKET {bucket_name}")
            else:
                LOG.info(f"DELETE_BUCKET {bucket_name} — not found, skipping")
        finally:
            client.close()

    def delete_range(self, bucket_name: str, start: str, stop: str) -> None:
        """Delete all data in bucket for the given time range (RFC3339 strings)."""
        client = create_client(self._write_url, self._token)
        try:
            client.delete_api().delete(
                start=start,
                stop=stop,
                predicate="",
                bucket=bucket_name,
                org=ORG,
            )
            LOG.info(f"DELETE_RANGE bucket={bucket_name} {start} → {stop}")
        finally:
            client.close()
