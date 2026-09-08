"""Geometry: one order - per-axis rows in array order, vectors in world (x, y, z)."""
import numpy as np
import pytest

import rankfield as rf


def test_aligned_puts_array_axes_along_world_axes():
    g = rf.Geometry.aligned((4, 5, 6), (3.0, 2.0, 1.0), origin=(1.0, 2.0, 3.0))
    assert g.shape == (4, 5, 6)
    assert g.spacing == (3.0, 2.0, 1.0)
    assert g.directions == ((0.0, 0.0, 3.0), (0.0, 2.0, 0.0), (1.0, 0.0, 0.0))
    assert g.cosines == ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    np.testing.assert_allclose(g.world((1, 1, 1)), (2.0, 4.0, 6.0))       # one step along each axis
    assert rf.Geometry.aligned((4, 5, 6), 2.0).spacing == (2.0, 2.0, 2.0)


def test_direction_rows_carry_the_spacing_and_the_matrix_is_the_rows():
    rows = ((0.0, 0.0, 3.0), (2.0, 0.0, 0.0), (0.0, -2.0, 0.0))                  # oblique: Y along x, X along -y
    g = rf.Geometry(shape=(7, 9, 11), directions=rows, origin=(1.0, -2.0, 5.0))
    assert g.spacing == (3.0, 2.0, 2.0)
    assert g.cosines == ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    idx = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2.5, 1.5, 0.5]])
    expect = np.asarray(g.origin) + idx @ np.asarray(rows)
    np.testing.assert_allclose(g.world(idx), expect)
    np.testing.assert_allclose((g.matrix @ np.c_[idx, np.ones(len(idx))].T)[:3].T, expect)


def test_regrid_keeps_the_orientation_and_moves_along_the_array_axes():
    rows = ((0.0, 0.0, 3.0), (2.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    g = rf.Geometry(shape=(7, 9, 11), directions=rows, origin=(1.0, -2.0, 5.0))
    h = g.regrid((3, 4, 5), 1.5, offset=(3.0, 2.0, 2.0))                          # one voxel in along each axis
    assert h.shape == (3, 4, 5)
    assert h.cosines == g.cosines
    assert h.spacing == (1.5, 1.5, 1.5)
    np.testing.assert_allclose(h.origin, g.world((1, 1, 1)))


def test_the_record_round_trips_and_an_old_one_is_refused():
    g = rf.Geometry(shape=(7, 9, 11), directions=((0.0, 0.0, 3.0), (2.0, 0.0, 0.0), (0.0, -2.0, 0.0)),
                    origin=(1.0, -2.0, 5.0))
    assert rf.Geometry.from_meta(g.to_meta()) == g
    with pytest.raises(ValueError, match="spacing_zyx"):
        rf.Geometry.from_meta({"spacing_zyx": [1, 1, 1], "shape_zyx": [2, 2, 2], "origin_xyz": [0, 0, 0],
                               "direction_xyz": [1, 0, 0, 0, 1, 0, 0, 0, 1]})


@pytest.mark.parametrize("rows", [
    ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),                          # a zero row
    ((0.0, 0.0, 1.0), (0.0, 0.0, 2.0), (1.0, 0.0, 0.0)),                          # two rows along one line
])
def test_degenerate_directions_are_refused(rows):
    with pytest.raises(ValueError):
        rf.Geometry(shape=(2, 2, 2), directions=rows)


def test_an_oblique_part_restores_with_its_orientation_and_its_box_offset():
    from conftest import logits
    rows = ((0.0, 0.0, 3.0), (2.0, 0.0, 0.0), (0.0, -2.0, 0.0))
    code = rf.encode(logits(K=4, shape=(6, 7, 8)), depth=3)
    code.labels = [0, 1, 2, 3]
    code.geometry = rf.Geometry(shape=(6, 7, 8), directions=rows, origin=(1.0, -2.0, 5.0))
    grid = rf.Grid.isotropic(1.0, like=rf.array_grid(rf.Part(field=code)))
    box = ((2, 5), (1, 4), (3, 9))
    r = rf.restore([rf.Part(field=code)], grid=grid, roi=box)
    assert r.labels.shape == (3, 3, 6)
    assert r.geometry.shape == (3, 3, 6)
    assert r.geometry.cosines == code.geometry.cosines
    assert r.geometry.spacing == (1.0, 1.0, 1.0)
    mm = np.asarray(grid.origin) + np.asarray([a for a, _ in box]) * np.asarray(grid.spacing)   # along the array axes
    np.testing.assert_allclose(r.geometry.origin, code.geometry.world(mm / np.asarray(code.geometry.spacing)))
