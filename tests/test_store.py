"""The portable store: written by rankfield alone, read back into the same parts."""
from pathlib import Path

import numpy as np
import pytest

import rankfield as rf
from conftest import logits

zarr = pytest.importorskip("zarr")
pytest.importorskip("duckn")


def _parts():
    a = rf.encode(logits(K=5, shape=(7, 9, 11), seed=1), depth=4)
    a.labels = [0, 10, 11, 12, 13]
    a.geometry = rf.Geometry(shape=(7, 9, 11), origin=(1.0, -2.0, 5.0),        # an oblique array
                             directions=((0.0, 0.0, 3.0), (2.0, 0.0, 0.0), (0.0, -2.0, 0.0)))
    b = rf.encode(logits(K=3, shape=(7, 9, 11), seed=2), depth=3)
    b.labels = [0, 20, 21]
    b.geometry = a.geometry
    return [rf.Part(field=a, name="organs"), rf.Part(field=b, envelope_start=(0, 0, 0), name="ribs")]


@pytest.mark.parametrize("suffix", [".duckn", ".duckn.zip"])
def test_write_then_read_round_trips_the_parts_and_the_restore(tmp_path, suffix):
    parts = _parts()
    out = rf.store.write_parts(tmp_path / f"s{suffix}", parts)
    back = rf.store.read_parts(out)
    assert [p.name for p in back] == ["organs", "ribs"]
    for p, q in zip(parts, back):
        np.testing.assert_array_equal(np.asarray(q.field.ranks[:]), p.field.ranks)
        np.testing.assert_array_equal(np.asarray(q.field.support[:]), p.field.support)
        assert q.field.labels == p.field.labels
        g, h = p.field.geometry, q.field.geometry
        np.testing.assert_allclose(h.origin, g.origin)
        np.testing.assert_allclose(h.directions, g.directions, atol=1e-9)
        assert q.field.meta["version"] == p.field.meta["version"]
    grid = rf.Grid.isotropic(1.5, like=rf.array_grid(parts[0]))
    np.testing.assert_array_equal(rf.restore(back, grid=grid).labels, rf.restore(parts, grid=grid).labels)


def test_the_root_and_arrays_validate_under_duckn(tmp_path):
    from duckn import DucknMetadata, validate_against_shape
    out = rf.store.write_parts(tmp_path / "s.duckn", _parts())
    root = rf.store.open_group(out)
    for i in (0, 1):
        for name in ("ranks", "support", "tail"):
            if name not in root[f"parts/{i}"]:
                continue                                   # exhaustive: no tail to write
            arr = root[f"parts/{i}/{name}"]
            validate_against_shape(DucknMetadata.model_validate(arr.attrs.asdict()["duckn"]), tuple(arr.shape))


def test_a_haversack_store_reads_as_parts():
    """The reader is the same one haversack uses on its own stores: a demo store beside this
    checkout, when there is one."""
    demo = Path(__file__).resolve().parents[2] / "haversack" / "data" / "duckn_demo" / "idc-torso1" / "body.duckn"
    if not demo.exists():
        pytest.skip("no haversack demo store beside this checkout")
    parts = rf.store.read_parts(demo)
    assert parts and parts[0].field.frame is None and parts[0].field.classes == 3
    labels = rf.restore(parts, grid=6.0, device="cpu").labels
    assert labels.any()
