"""Float64 reference restore - tiny shapes only: the K-channel deficit field interpolated
with float64 weights, then argmax (lowest index on a tie). Returns ``(labels, winner
margin)`` so a test can tell a tie from a defect.

NOT the definition of :func:`~rankfield.restore`. This builds the whole dense field, so a
class stored at no corner of the stencil still reads ``-clip`` here and can win; the restore
considers only the classes stored at the eight corners. The two disagree exactly there
(docs/format.md, "The restore"). On real fields that difference does not arise, which is
what makes this a useful check on the kernels - it is a check, not a specification."""
from __future__ import annotations

import numpy as np

from .code import Part, levels
from .restore import mapping_of
from .tables import build_tables


def reference_restore(part: Part, grid):
    f = part.field
    K = f.classes
    lv = levels(f.meta)
    shape = tuple(f.ranks.shape[1:])
    dense = np.full((K,) + shape, -f.clip, np.float64)
    for j in range(f.depth):
        level = np.zeros(shape) if j == 0 else -lv[f.support[j - 1]].astype(np.float64)
        for c in range(1, K + 1):
            sel = f.ranks[j] == c
            dense[c - 1][sel] = level[sel]
    tz, ty, tx = build_tables(grid.shape, shape, mapping_of(part, grid), interp="linear", outside="background")

    def g(iz, iy, ix):
        return dense[:, iz[:, None, None], iy[None, :, None], ix[None, None, :]]

    def idx(t):
        return np.where(t.i0 < 0, 0, t.i0).astype(np.int64), t.i1.astype(np.int64), t.f64
    z0, z1, zf = idx(tz)
    y0, y1, yf = idx(ty)
    x0, x1, xf = idx(tx)
    wx = xf[None, None, None, :]
    wy = yf[None, None, :, None]
    wz = zf[None, :, None, None]
    c00 = g(z0, y0, x0) * (1 - wx) + g(z0, y0, x1) * wx
    c01 = g(z0, y1, x0) * (1 - wx) + g(z0, y1, x1) * wx
    c10 = g(z1, y0, x0) * (1 - wx) + g(z1, y0, x1) * wx
    c11 = g(z1, y1, x0) * (1 - wx) + g(z1, y1, x1) * wx
    values = (c00 * (1 - wy) + c01 * wy) * (1 - wz) + (c10 * (1 - wy) + c11 * wy) * wz
    valid = (tz.i0 >= 0)[:, None, None] & (ty.i0 >= 0)[None, :, None] & (tx.i0 >= 0)[None, None, :]
    lut = np.asarray(f.labels, dtype=np.int64)
    best = values.argmax(0)
    labels = np.where(valid, lut[best], 0)
    srt = np.sort(values, axis=0)
    return labels, srt[-1] - srt[-2]
