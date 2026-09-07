"""The restore against the logits it was encoded from.

Every other restore test compares the candidate restore with reference.py's float64 restore
OF THE SAME STORED FIELD. Both sides read the same bytes, so they agree about everything the
encoding threw away, and no test in this suite could see the keep rule's cost. Here the
ORIGINAL logits are interpolated on the same grid and the labels compared, which is the only
comparison that can (docs/format.md, "What the keep rule does not promise").
"""
import numpy as np
import torch

import rankfield as rf
from rankfield.restore import mapping_of
from rankfield.tables import build_tables
from conftest import logits


def _part(code, spacing=(2.0, 2.0, 2.0)):
    code.labels = list(range(code.classes))
    code.geometry = rf.Geometry(spacing_zyx=spacing, shape_zyx=tuple(code.ranks.shape[1:]))
    return rf.Part(field=code)


def interpolated_argmax(dense, part, grid):
    """reference_restore's own interpolation, over any dense (K, Z, Y, X) float64 field."""
    tz, ty, tx = build_tables(grid.shape, tuple(dense.shape[1:]), mapping_of(part, grid),
                              interp="linear", outside="background")

    def g(iz, iy, ix):
        return dense[:, iz[:, None, None], iy[None, :, None], ix[None, None, :]]

    def idx(t):
        return np.where(t.i0 < 0, 0, t.i0).astype(np.int64), t.i1.astype(np.int64), t.f64

    z0, z1, zf = idx(tz)
    y0, y1, yf = idx(ty)
    x0, x1, xf = idx(tx)
    wx, wy, wz = xf[None, None, None, :], yf[None, None, :, None], zf[None, :, None, None]
    c00 = g(z0, y0, x0) * (1 - wx) + g(z0, y0, x1) * wx
    c01 = g(z0, y1, x0) * (1 - wx) + g(z0, y1, x1) * wx
    c10 = g(z1, y0, x0) * (1 - wx) + g(z1, y0, x1) * wx
    c11 = g(z1, y1, x0) * (1 - wx) + g(z1, y1, x1) * wx
    return ((c00 * (1 - wy) + c01 * wy) * (1 - wz) + (c10 * (1 - wy) + c11 * wy) * wz).argmax(0)


def label_error(lg, depth, spacing=(2.0, 2.0, 2.0), step=0.9):
    """The fraction of restored voxels whose label the original logits do not agree with."""
    code = rf.encode(lg, depth=depth)
    part = _part(code, spacing)
    grid = rf.Grid.isotropic(step, like=rf.array_grid(part))
    deficit = (lg.double() - lg.double().max(0, keepdim=True).values).numpy()
    truth = interpolated_argmax(deficit, part, grid)
    got = rf.restore([part], grid=grid).labels
    return float((got != truth).mean())


class TestAgainstTheLogits:
    def test_a_smooth_field_restores_its_own_argmax_almost_everywhere(self):
        assert label_error(logits(K=12), depth=6) < 0.002

    def test_the_error_falls_with_depth_and_settles_on_the_byte_quantum(self):
        """Depth is the knob the format document points at. The floor it settles on is
        quantization, which depth cannot reach."""
        lg = logits(K=12, noise=1.5)
        e = {d: label_error(lg, depth=d) for d in (4, 6, 8, 12)}
        assert e[4] > e[6] > e[8] > e[12], f"error must fall with depth, got {e}"
        assert e[12] < e[4] / 4, f"depth should buy at least 4x, got {e}"
        assert e[12] > 0.0, "some error survives full depth: the byte quantum"

    def test_more_classes_competing_costs_more_at_a_fixed_depth(self):
        assert label_error(logits(K=30, noise=1.5), depth=6) > label_error(logits(K=12, noise=1.5), depth=6)


class TestTheDocumentedLimits:
    """The keep rule's costs, pinned where the specification claims them. These assert what
    the encoding DOES, not what it should do; a fix here is a documentation change too.
    """

    def test_a_dropped_class_at_the_floor_can_still_win_an_interpolation(self):
        """format.md, "A dropped class still reads at the floor". Needs enough other classes
        to fill the depth - with a plane per class the field keeps everything and is right."""
        K = 8
        cols = np.full((2, K), -500.0)
        cols[0, 0], cols[1, 0] = 0.0, -40.0        # A wins voxel 0
        cols[0, 1], cols[1, 1] = -40.0, 0.0        # B wins voxel 1
        cols[0, 2], cols[1, 2] = -1.0, -100.0      # C: close at voxel 0, far at voxel 1
        lg = torch.tensor(cols.T, dtype=torch.float32)[:, None, None, :]
        part = _part(rf.encode(lg, depth=6), spacing=(1.0, 1.0, 1.0))
        grid = rf.Grid(shape=(1, 1, 1), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.4))
        true_argmax = int((0.6 * cols[0] + 0.4 * cols[1]).argmax())
        assert true_argmax == 0
        assert int(rf.restore([part], grid=grid).labels.ravel()[0]) == 2, "C, at the -clip floor"
        assert float(rf.deficit(part.field, 2)[0, 0, 1]) == -part.field.clip

    def test_the_nearest_kept_class_is_not_always_the_true_runner_up(self):
        """format.md, "`ranks[1]` is not always the true runner-up"."""
        lg = logits(K=30, noise=1.5)
        code = rf.encode(lg, depth=6)
        true_second = np.argsort(-lg.double().numpy(), axis=0)[1]
        stored_second = code.ranks[1].astype(np.int64) - 1
        live = stored_second >= 0
        assert (stored_second[live] != true_second[live]).mean() > 0.01

    def test_the_float64_reference_is_not_candidate_restricted(self):
        """format.md, "The float64 reference is not candidate-restricted". reference.py lets a
        class stored at no corner win from the -clip floor; restore() never considers it, so
        the two answer differently and only one of them can be called the restore."""
        cols = np.array([[0.0, -40.0], [-40.0, 0.0], [-9.0, -9.0]]).T     # (voxel, class)
        lg = torch.tensor(cols.T, dtype=torch.float32)[:, None, None, :]
        part = _part(rf.encode(lg, depth=2), spacing=(1.0, 1.0, 1.0))
        assert not (part.field.ranks == 3).any(), "C must be stored at neither voxel"
        grid = rf.Grid(shape=(1, 1, 1), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.4))
        assert int((0.6 * cols[0] + 0.4 * cols[1]).argmax()) == 2         # the logits say C
        assert int(rf.restore([part], grid=grid).labels.ravel()[0]) == 0
        assert int(rf.reference_restore(part, grid)[0].ravel()[0]) == 2

    def test_depth_closes_the_runner_up_gap(self):
        lg = logits(K=30, noise=1.5)

        def wrong(depth):
            code = rf.encode(lg, depth=depth)
            true_second = np.argsort(-lg.double().numpy(), axis=0)[1]
            stored = code.ranks[1].astype(np.int64) - 1
            live = stored >= 0
            return float((stored[live] != true_second[live]).mean())

        assert wrong(12) < wrong(6) / 4
