"""Restoring onto a world geometry through a general affine map (2026-09-25).

A field its model computed on a grid it resampled in WORLD space - FastSurfer's conformed
1 mm grid - is related to an oblique input by a rotation, which no per-axis Mapping holds.
These hold the new path to the old one wherever both apply, exactly where the arithmetic is
exact (flips and permutations with binary-exact coordinates), and to the world where only it
applies (an off-center ball on a rotated grid - a centered one would pass a wrong rotation).
"""
import numpy as np
import pytest
import torch

import rankfield as rf
from rankfield.mapping import Affine
from rankfield.restore import _affine_part, _restore_part
from conftest import devices, logits


def part(code, geometry, name="p"):
    code.labels = list(range(code.classes))
    code.geometry = geometry
    return rf.Part(field=code, name=name)


def test_a_world_grid_that_lines_up_is_the_ordinary_restore():
    code = rf.encode(logits(K=8, shape=(9, 11, 13), noise=0.2), depth=6)
    origin = (-3.0, 7.0, 11.0)
    p = part(code, rf.Geometry.aligned((9, 11, 13), 2.0, origin=origin))
    world = rf.Geometry.aligned((13, 16, 19), 1.3, origin=origin)
    for dev in devices():
        a = rf.restore([p], grid=world, device=dev)
        b = rf.restore([p], grid=rf.Grid((13, 16, 19), spacing=(1.3,) * 3), device=dev)
        np.testing.assert_array_equal(a.labels, b.labels)
        np.testing.assert_allclose(a.geometry.origin, origin)
        np.testing.assert_allclose(a.geometry.directions, world.directions)


@pytest.mark.parametrize("interp", ["linear", "nearest"])
def test_the_affine_path_is_the_per_axis_path_where_both_apply(interp):
    code = rf.encode(logits(K=10, shape=(9, 11, 13), noise=0.3), depth=6)
    p = part(code, rf.Geometry.aligned((9, 11, 13), 2.0))
    a, b = (0.61, 0.53, 0.47), (-0.3, 0.2, 0.9)
    box = ((0, 15), (2, 20), (1, 27))
    shape = tuple(hi - lo for lo, hi in box)
    per_axis = np.zeros(shape, np.uint8)
    general = np.zeros(shape, np.uint8)
    _restore_part(p, rf.Grid((15, 20, 27)), box, interp, per_axis, paint=False,
                  dev=torch.device("cpu"), slab_voxels=1 << 20, gpu=False, mapping=rf.Mapping(a, b))
    _affine_part(p, Affine(np.diag(a), b), box, interp, general, paint=False,
                 dev=torch.device("cpu"), slab_voxels=97)       # small slabs: the seams too
    np.testing.assert_array_equal(general, per_axis)


def _flipped_and_swapped(K=9):
    """One field stored two ways: plainly, and with its x axis reversed and its y and x axes
    swapped - the same world content. Spacings and origins are binary-exact, so both maps
    produce identical coordinates and the restores must agree bit for bit."""
    lg = logits(K=K, shape=(8, 10, 12), noise=0.3)
    plain = part(rf.encode(lg, depth=6), rf.Geometry.aligned((8, 10, 12), 2.0, origin=(4.0, -6.0, 8.0)))
    # stored[z, x', y] = lg[z, y, 11 - x']: array axis 1 walks world x backwards, axis 2 walks y
    stored = torch.flip(lg, dims=(3,)).transpose(2, 3).contiguous()
    geo = rf.Geometry(shape=(8, 12, 10),
                      directions=((0.0, 0.0, 2.0), (-2.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
                      origin=(4.0 + 2.0 * 11, -6.0, 8.0))
    other = part(rf.encode(stored, depth=6), geo)
    return plain, other


@pytest.mark.parametrize("interp", ["linear", "nearest"])
def test_a_flip_and_an_axis_swap_restore_exactly(interp):
    plain, other = _flipped_and_swapped()
    assert Affine.between(rf.Geometry.aligned((1, 1, 1), 1.0), other.field.geometry).separable is None
    world = rf.Geometry.aligned((17, 21, 25), 1.0, origin=(4.5, -6.0, 7.5))
    for dev in devices():
        a = rf.restore([other], grid=world, interp=interp, device=dev)
        b = rf.restore([plain], grid=world, interp=interp, device=dev)
        np.testing.assert_array_equal(a.labels, b.labels)


def test_an_roi_is_the_same_voxels_as_the_whole():
    plain, other = _flipped_and_swapped()
    world = rf.Geometry.aligned((17, 21, 25), 1.0, origin=(4.5, -6.0, 7.5))
    whole = rf.restore([other], grid=world)
    roi = ((3, 11), (5, 19), (2, 9))
    part_ = rf.restore([other], grid=world, roi=roi)
    np.testing.assert_array_equal(part_.labels, whole.labels[3:11, 5:19, 2:9])
    np.testing.assert_allclose(part_.geometry.world((0, 0, 0)), whole.geometry.world((3, 5, 2)))


def test_a_rotated_model_grid_puts_an_off_center_ball_where_it_is():
    """A ball of radius 9 mm at an off-center world point, computed on a 1 mm model grid
    rotated 30 degrees about z; restored onto an axis-aligned 0.8 mm world grid it must sit
    at the same world point, the same size."""
    th = np.deg2rad(30.0)
    rows = ((0.0, 0.0, 1.0), (-np.sin(th), np.cos(th), 0.0), (np.cos(th), np.sin(th), 0.0))
    geo = rf.Geometry(shape=(40, 48, 48), directions=rows, origin=(-5.0, -30.0, -20.0))
    center = np.array([12.0, 6.0, 1.5])                       # world x, y, z
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in geo.shape], indexing="ij"), -1)
    d = np.linalg.norm(geo.world(idx) - center, axis=-1)
    lg = torch.from_numpy(np.stack([np.zeros_like(d), 9.0 - d]).astype(np.float32))
    p = part(rf.encode(lg, depth=2), geo)
    world = rf.Geometry.aligned((50, 70, 70), 0.8, origin=(-24.0, -26.0, -22.0))
    for dev in devices():
        r = rf.restore([p], grid=world, device=dev)
        ball = np.argwhere(r.labels == 1)
        got = r.geometry.world(ball.mean(0))
        assert np.abs(got - center).max() < 0.2, (dev, got)
        vol = len(ball) * 0.8 ** 3
        assert abs(vol - 4 / 3 * np.pi * 9 ** 3) / (4 / 3 * np.pi * 9 ** 3) < 0.03, vol


def test_a_framed_part_refuses_a_world_grid():
    code = rf.encode(logits(K=4), depth=4)
    p = part(code, rf.Geometry.aligned((6, 10, 12), 2.0))
    code.frame = {"placeholder": True}
    with pytest.raises(ValueError, match="through its frame"):
        rf.restore([p], grid=rf.Geometry.aligned((6, 10, 12), 2.0))


def test_roi_of_bounds_a_structure_on_a_world_grid():
    """The ball of the rotated test, found by its stored extent through the inverse map."""
    th = np.deg2rad(30.0)
    rows = ((0.0, 0.0, 1.0), (-np.sin(th), np.cos(th), 0.0), (np.cos(th), np.sin(th), 0.0))
    geo = rf.Geometry(shape=(40, 48, 48), directions=rows, origin=(-5.0, -30.0, -20.0))
    center = np.array([12.0, 6.0, 1.5])
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in geo.shape], indexing="ij"), -1)
    d = np.linalg.norm(geo.world(idx) - center, axis=-1)
    lg = torch.from_numpy(np.stack([np.zeros_like(d), 9.0 - d]).astype(np.float32))
    p = part(rf.encode(lg, depth=2), geo)
    inside = np.argwhere(d < 9.0)
    ext = [int(v) for pair in zip(inside.min(0), inside.max(0)) for v in pair]
    world = rf.Geometry.aligned((50, 70, 70), 0.8, origin=(-24.0, -26.0, -22.0))
    grid = rf.Grid(world.shape, spacing=world.spacing)
    box = rf.roi_of([p], [(0, ext)], grid, world=world)
    whole = rf.restore([p], grid=world).labels
    ball = np.argwhere(whole == 1)
    for ax, (a, b) in enumerate(box):
        assert a <= ball[:, ax].min() and ball[:, ax].max() < b, (box, ball.min(0), ball.max(0))


@pytest.mark.parametrize("interp", ["linear", "nearest"])
def test_a_pure_flip_takes_the_general_path_and_restores_exactly(interp):
    """A flip alone - a negative diagonal, nothing off it - is no per-axis Mapping (its
    factors are non-negative by construction), so it must take the general path."""
    lg = logits(K=7, shape=(8, 10, 12), noise=0.3)
    plain = part(rf.encode(lg, depth=6), rf.Geometry.aligned((8, 10, 12), 2.0, origin=(4.0, -6.0, 8.0)))
    flipped = part(rf.encode(torch.flip(lg, dims=(3,)).contiguous(), depth=6),
                   rf.Geometry(shape=(8, 10, 12), directions=((0.0, 0.0, 2.0), (0.0, 2.0, 0.0), (-2.0, 0.0, 0.0)),
                               origin=(4.0 + 2.0 * 11, -6.0, 8.0)))
    world = rf.Geometry.aligned((17, 21, 25), 1.0, origin=(4.5, -6.0, 7.5))
    assert Affine.between(world, flipped.field.geometry).separable is None
    np.testing.assert_array_equal(rf.restore([flipped], grid=world, interp=interp).labels,
                                  rf.restore([plain], grid=world, interp=interp).labels)
