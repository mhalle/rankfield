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


@pytest.mark.parametrize("recorded_shape", [True, False])
def test_the_host_decoders_read_parts_straight_from_a_store(tmp_path, recorded_shape):
    """read_parts hands back lazy zarr arrays; margin/deficit/probabilities read them a plane
    at a time and give what they give on the in-memory field. ``recorded_shape=False`` is a
    store whose ranked block has no ``shape`` (haversack writes none): the planes say it."""
    parts = _parts()
    back = rf.store.read_parts(rf.store.write_parts(tmp_path / "s.duckn.zip", parts))
    for p, q in zip(parts, back):
        assert not isinstance(q.field.support, np.ndarray)          # lazy, as read
        if not recorded_shape:
            del q.field.meta["shape"]
        for c in range(p.field.classes):
            np.testing.assert_array_equal(rf.margin(q.field, c), rf.margin(p.field, c))
            np.testing.assert_array_equal(rf.deficit(q.field, c), rf.deficit(p.field, c))
        for a, b in zip(rf.probabilities(q.field), rf.probabilities(p.field)):
            np.testing.assert_array_equal(a, b)
        groups = [[c] for c in range(p.field.classes)]
        np.testing.assert_array_equal(rf.decode_groups(q.field, groups).numpy(),
                                      rf.decode_groups(p.field, groups).numpy())


def test_a_device_resident_field_is_refused_by_name():
    torch = pytest.importorskip("torch")
    code = rf.to_device(_parts()[0].field, "cpu")
    assert isinstance(code.support, torch.Tensor)
    with pytest.raises(TypeError, match="decode_groups"):
        rf.margin(code, 0)


class CountingPlanes:
    """An array that counts its plane reads - what a zarr array costs per read."""

    def __init__(self, a):
        self.a, self.shape, self.reads = a, a.shape, 0

    def __getitem__(self, i):
        self.reads += 1
        return self.a[i]


def test_margin_and_deficit_read_each_plane_once():
    f = _parts()[0].field
    for fn in (rf.margin, rf.deficit):
        ranks, support = CountingPlanes(f.ranks), CountingPlanes(f.support)
        lazy = rf.RankField(ranks=ranks, support=support, tail=None, meta=f.meta)
        np.testing.assert_array_equal(fn(lazy, 2), fn(f, 2))
        assert (ranks.reads, support.reads) == (f.ranks.shape[0], f.support.shape[0])


def test_a_meta_shape_that_is_not_the_planes_is_refused():
    f = _parts()[0].field
    f.meta["shape"] = [7, 9, 12]
    with pytest.raises(ValueError, match="shape"):
        rf.margin(f, 0)


def test_a_torch_plane_beside_numpy_ones_is_refused_by_name():
    torch = pytest.importorskip("torch")
    f = _parts()[0].field
    f.ranks = torch.from_numpy(f.ranks)
    with pytest.raises(TypeError, match="ranks is a torch tensor"):
        rf.deficit(f, 0)
