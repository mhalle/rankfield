"""The restore on in-memory parts: against the float64 reference, across devices, and the
rules the reviews forced (paint on the decision, ties to the stored winner, far-edge grids)."""
import numpy as np
import pytest

import rankfield as rf
from rankfield import reference_restore
from conftest import devices, logits


def part(code, labels=None, spacing=(2.0, 2.0, 2.0), name="p"):
    K = code.classes
    code.labels = list(range(K)) if labels is None else list(labels)
    shape = tuple(code.ranks.shape[1:])
    code.geometry = rf.Geometry(spacing_zyx=spacing, shape_zyx=shape)
    return rf.Part(field=code, name=name)


def test_the_argmax_of_the_interpolated_deficit_matches_the_float64_reference():
    """On the store's own decoded field, the candidate restore equals reference.py's
    K-channel interpolation + argmax everywhere but at exact ties (which go to the stored
    winner here, argmax's lowest index there)."""
    lg = logits(K=10, shape=(9, 11, 13), noise=0.2)
    code = rf.encode(lg, depth=6)
    p = part(code)
    grid = rf.Grid.isotropic(1.3, like=rf.array_grid(p))
    res = rf.restore([p], grid=grid)
    ref, margin = reference_restore(p, grid)
    diff = res.labels != ref
    assert diff.mean() < 1e-3
    assert (margin[diff] < 1e-5).all()                  # only exact ties differ


def test_nearest_is_the_stored_winner():
    code = rf.encode(logits(K=6), depth=4)
    p = part(code)
    res = rf.restore([p], grid=rf.array_grid(p), interp="nearest")
    np.testing.assert_array_equal(res.labels, code.ranks[0].astype(np.int64) - 1)


def test_the_identity_restore_is_the_labelmap_even_at_quantized_ties():
    ranks = np.zeros((2, 3, 3, 4), np.uint8)
    support = np.zeros((1, 3, 3, 4), np.uint8)
    ranks[0] = 4
    ranks[1] = 2
    support[0] = 255                                    # tied within the quantum
    ranks[0, :, :, 2:] = 2
    ranks[1, :, :, 2:] = 4
    code = rf.encode(logits(K=4, shape=(3, 3, 4)), depth=2)
    code.ranks, code.support = ranks, support
    p = part(code)
    for dev in devices():
        res = rf.restore([p], grid=rf.array_grid(p), device=dev)
        np.testing.assert_array_equal(res.labels, np.asarray([3, 3, 1, 1])[None, None, :].repeat(3, 0).repeat(3, 1))


def test_a_grid_past_the_far_edge_restores_in_small_slabs_on_every_device():
    code = rf.encode(logits(K=6, shape=(12, 10, 10), seed=3), depth=6)
    p = part(code)
    grid = rf.Grid(shape=(40, 22, 22), spacing=(1.0, 1.0, 1.0), origin=(-3.0, -2.0, -2.0))
    big = rf.restore([p], grid=grid, slab_voxels=1 << 20).labels
    for slab in (22 * 22 * 3, 700):
        np.testing.assert_array_equal(rf.restore([p], grid=grid, slab_voxels=slab).labels, big)
    assert (big[-3:] == 0).all() and (big[:2] == 0).all()
    for dev in devices()[1:]:
        np.testing.assert_array_equal(rf.restore([p], grid=grid, device=dev).labels, big)


def test_paint_is_transparent_for_class_zero_not_for_label_zero():
    code = rf.encode(logits(K=4, shape=(6, 6, 6), seed=5, noise=0.3), depth=4)
    grid = rf.Grid(shape=(9, 9, 9), spacing=(1.3, 1.3, 1.3))
    base = rf.encode(logits(K=4, shape=(6, 6, 6), seed=6, noise=0.3), depth=4)
    for dev in devices():
        a = part(base, labels=[0, 9, 9, 9], name="a")
        b = part(code, labels=[5, 1, 0, 3], name="b")          # class 0 -> 5, class 2 -> 0
        out = rf.restore([a, b], grid=grid, device=dev).labels
        plain = rf.restore([part(code, labels=[0, 1, 2, 3])], grid=grid, device=dev).labels
        assert not (out == 5).any()                           # class 0 never paints
        assert (out[plain == 2] == 0).all()                   # class 2 paints label 0
        assert (out[plain == 0] == rf.restore([a], grid=grid, device=dev).labels[plain == 0]).all()


def test_a_later_part_never_narrows_an_earlier_parts_labels():
    a = rf.encode(logits(K=3, shape=(6, 6, 6), seed=7), depth=3)
    b = rf.encode(logits(K=3, shape=(6, 6, 6), seed=8), depth=3)
    grid = rf.Grid(shape=(8, 8, 8), spacing=(1.5, 1.5, 1.5))
    for dev in devices():
        out = rf.restore([part(a, [0, 300, 301], name="a"), part(b, [0, 7, 8], name="b")], grid=grid, device=dev).labels
        assert out.dtype == np.uint16 and (out > 255).any() and set(np.unique(out)) <= {0, 7, 8, 300, 301}


def test_an_roi_beyond_a_parts_z_extent_is_background_on_every_device():
    code = rf.encode(logits(K=3, shape=(6, 6, 6), seed=9), depth=3)
    p = part(code)
    grid = rf.Grid(shape=(10, 8, 8), spacing=(2.0, 2.0, 2.0), origin=(30.0, 0.0, 0.0))
    for dev in devices():
        assert (rf.restore([p], grid=grid, device=dev).labels == 0).all()


def test_gpu_paths_match_the_torch_path_bit_for_bit():
    if len(devices()) < 2:
        pytest.skip("no GPU")
    for seed in range(4):
        K, depth = (10, 6) if seed % 2 == 0 else (5, 5)
        code = rf.encode(logits(K=K, shape=(11, 13, 17), seed=seed, noise=0.3), depth=depth)
        for spacing in (0.7, 1.3, 2.0):
            p = part(code)
            grid = rf.Grid.isotropic(spacing, like=rf.array_grid(p))
            cpu = rf.restore([p], grid=grid).labels
            for dev in devices()[1:]:
                np.testing.assert_array_equal(rf.restore([p], grid=grid, device=dev).labels, cpu)
                box = ((1, grid.shape[0] - 1), (0, grid.shape[1]), (2, grid.shape[2]))
                np.testing.assert_array_equal(rf.restore([p], grid=grid, device=dev, roi=box).labels,
                                              cpu[tuple(slice(a, b) for a, b in box)])


def test_two_parts_must_share_a_placement():
    a = rf.encode(logits(K=3, shape=(6, 6, 6), seed=1), depth=3)
    b = rf.encode(logits(K=3, shape=(6, 6, 6), seed=2), depth=3)
    pa, pb = part(a, name="a"), part(b, name="b")
    pb.field.geometry = rf.Geometry(spacing_zyx=(2.0, 2.0, 2.0), shape_zyx=(6, 6, 6), origin_xyz=(5.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="placement"):
        rf.restore([pa, pb], grid=2.0)


def test_roi_of_bounds_a_structure():
    code = rf.encode(logits(K=5, shape=(8, 9, 10), seed=4), depth=4)
    p = part(code)
    grid = rf.Grid.isotropic(1.0, like=rf.array_grid(p))
    full = rf.restore([p], grid=grid).labels
    win = code.ranks[0].astype(int) - 1
    for c in range(1, 5):
        if not (win == c).any():
            continue
        idx = np.argwhere(win == c)
        ext = [int(v) for lo, hi in zip(idx.min(0), idx.max(0)) for v in (lo, hi)]
        box = rf.roi_of([p], [(0, ext)], grid)
        sl = tuple(slice(a, b) for a, b in box)
        assert (full[sl] == c).sum() == (full == c).sum()
