import pyarrow as pa
import pyarrow.parquet as pq

from makalu.cli.scan import scan
from makalu.storage.manifest import Manifest


def _write(root, rel):
    path = root / rel / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"ts": [1.0, 2.0]}), path)
    return path


def test_scan_registers_new_files_and_prunes_deleted_ones(tmp_path):
    root = tmp_path / "lake"
    keep = _write(root, "gold/dataset=Spread/date=2026-01-01")
    drop = _write(root, "gold/dataset=VolSpread/date=2026-01-01")
    _write(root, "bronze/dataset=Trade/date=2026-01-01")
    manifest = Manifest(str(tmp_path / "manifest.db"))

    assert scan(root, manifest) == (2, 0)  # bronze is never tracked
    drop.unlink()
    assert scan(root, manifest) == (0, 1)
    assert [r["path"] for r in manifest.get_all()] == [str(keep)]


def test_scan_will_not_prune_everything_when_the_lake_looks_empty(tmp_path):
    """An unmounted root walks as empty; that is not every file being deleted."""
    root = tmp_path / "lake"
    path = _write(root, "gold/dataset=Spread/date=2026-01-01")
    manifest = Manifest(str(tmp_path / "manifest.db"))
    scan(root, manifest)
    path.unlink()
    assert scan(root, manifest) == (0, 0)
    assert len(manifest.get_all()) == 1
