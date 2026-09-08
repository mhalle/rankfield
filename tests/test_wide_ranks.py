"""More than 254 classes: the rank plane widens to uint16, and every path has to follow.

`rank_dtype` picks uint16 as soon as `class + 1` will not fit a byte, so a store with 255
classes or more exercises code that a 118-class store never reaches. The GPU kernels were
the part that did not follow: their label-table bounds check reduced the rank plane on the
device, and torch's MPS backend has no max reduction for uint16, so the check raised
`max_reduction_ushort_ushort` before the kernel ran - on exactly the stores that need it.
"""
import numpy as np
import pytest
import torch

import rankfield as rf
from rankfield.backends import max_class
from conftest import devices, logits

SHAPE = (5, 8, 10)


def part_of(K, *, depth=5, labels=None, shape=SHAPE):
    code = rf.encode(logits(K=K, shape=shape, noise=0.5), depth=depth)
    code.labels = list(range(K)) if labels is None else list(labels)
    code.geometry = rf.Geometry.aligned(shape, (2.0, 2.0, 2.0))
    return rf.Part(field=code)


class TestTheRankPlaneWidens:
    @pytest.mark.parametrize("K, want", [(200, np.uint8), (254, np.uint8), (255, np.uint16),
                                         (300, np.uint16), (600, np.uint16)])
    def test_the_dtype_follows_the_class_count(self, K, want):
        code = rf.encode(logits(K=K, shape=SHAPE), depth=4)
        assert code.ranks.dtype == want
        assert code.meta["rank_dtype"] == np.dtype(want).name
        assert int(code.ranks.max()) <= K                      # class + 1, and it fits
        assert (code.ranks[0].astype(np.int64) - 1 == logits(K=K, shape=SHAPE).argmax(0).numpy()).all()

    def test_the_numpy_decoders_read_a_wide_field(self):
        code = part_of(300).field
        assert rf.margin(code, 7).shape == SHAPE
        assert rf.deficit(code, 299).shape == SHAPE
        ids, p = rf.probabilities(code)
        np.testing.assert_allclose(p.sum(0) + code.tail / 65535.0, 1.0, atol=2.0 / 65535)


class TestEveryDeviceAgrees:
    @pytest.mark.parametrize("K", [300, 600])
    def test_the_restore_matches_the_cpu_on_every_device(self, K):
        p = part_of(K)
        grid = rf.Grid.isotropic(1.3, like=rf.array_grid(p))
        want = rf.restore([p], grid=grid, device="cpu").labels
        assert want.dtype == np.uint16
        for dev in devices():
            got = np.asarray(rf.restore([p], grid=grid, device=dev).labels)
            np.testing.assert_array_equal(got, want, err_msg=f"{dev} differs from cpu at {K} classes")

    def test_decode_groups_reads_a_wide_field_on_every_device(self):
        code = part_of(300).field
        want = rf.decode_groups(code, [[0, 1], [299]]).numpy()
        for dev in devices():
            got = rf.decode_groups(rf.to_device(code, dev), [[0, 1], [299]], device=dev)
            np.testing.assert_allclose(got.cpu().numpy(), want, atol=1e-6)

    def test_a_wide_LABEL_table_over_narrow_ranks_widens_only_the_output(self):
        """The output dtype follows the LABELS, the rank dtype follows the CLASS COUNT, and
        the two are independent: six classes mapped onto labels up to 65535."""
        p = part_of(6, depth=4, labels=[0, 300, 700, 1200, 40000, 65535])
        assert p.field.ranks.dtype == np.uint8
        grid = rf.Grid.isotropic(1.4, like=rf.array_grid(p))
        want = rf.restore([p], grid=grid, device="cpu").labels
        assert want.dtype == np.uint16 and int(want.max()) > 255
        for dev in devices():
            np.testing.assert_array_equal(np.asarray(rf.restore([p], grid=grid, device=dev).labels), want)


class TestTheBoundsCheckSurvivesTheDtype:
    def test_max_class_agrees_with_a_plain_reduction_on_every_device(self):
        a = np.array([[0, 1, 40000], [65535, 2, 3]], np.uint16)
        for dev in devices():
            t = torch.from_numpy(a).to(torch.device(dev))
            assert max_class(t) == 65535
        assert max_class(torch.from_numpy(a.astype(np.uint8) * 0)) == 0
        assert max_class(torch.zeros((0,), dtype=torch.uint16)) == 0

    def test_a_short_label_table_is_still_refused_on_every_device(self):
        p = part_of(300)
        p.field.labels = list(range(299))                       # one short
        grid = rf.Grid.isotropic(1.4, like=rf.array_grid(p))
        for dev in devices():
            with pytest.raises(ValueError, match="label table"):
                rf.restore([p], grid=grid, device=dev)


class TestTheStoreCarriesTheWidth:
    """The store writes whatever dtype the planes are; nothing in it is byte-shaped."""

    def test_a_wide_store_round_trips_and_restores_the_same(self, tmp_path):
        pytest.importorskip("zarr")
        pytest.importorskip("duckn")
        p = part_of(300)
        grid = rf.Grid.isotropic(1.4, like=rf.array_grid(p))
        want = rf.restore([p], grid=grid, device="cpu").labels
        for suffix in (".duckn", ".duckn.zip"):
            out = rf.store.write_parts(tmp_path / f"wide{suffix}", [p])
            back = rf.store.read_parts(out)
            np.testing.assert_array_equal(np.asarray(back[0].field.ranks[:]), p.field.ranks)
            assert np.asarray(back[0].field.ranks[:]).dtype == np.uint16
            np.testing.assert_array_equal(rf.restore(back, grid=grid, device="cpu").labels, want)
