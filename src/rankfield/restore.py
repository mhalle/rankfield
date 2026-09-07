"""Labels on any grid from encoded parts: the restore.

At an output voxel the only classes that can win are the ones stored at the eight model
voxels around it (a class stored nowhere in the stencil interpolates to exactly ``-clip``,
below any stored one), so the restore per voxel is: gather the eight corners' planes,
interpolate the DEFICIT of each candidate (zero where it won, minus its gap where it
trailed, minus the clip where it is not stored), take the largest. Cells whose eight
corners share a winner take it outright. With the ``shell`` keep rule every class that
wins at any corner is stored at every corner, so the floor never decides a winner.

*Ties.* Two candidates that tie after interpolation (the byte quantum can make one) are
ordered by the weight of the corners where each is the stored winner, then by class index;
so on a model voxel the answer is that voxel's stored winner. This is a deliberate
deviation from argmax's first-maximal-index rule at a genuine tie of interpolated logits;
it makes the identity restore exact and is a no-op on real interpolated output.

*Paint.* With several parts, a part's DECISION background (class 0) is transparent,
whatever its label table maps class 0 to - the pipeline's rule.

*Placement.* A part with a ``frame`` restores onto grids in the frame's source space,
exactly as the pipeline composed the mapping (``Frame.mapping(grid)`` plus the envelope
crop); a part without one restores onto grids in its own array frame - voxel 0 at 0 mm,
the array's true spacing - and cannot resolve ``"input"``. Every part of a multi-part
restore must share one placement.

Work is bounded by the region asked for: an ROI touches only the model box under it, read
once per part. The GPU paths (Metal, Triton) make the same decisions bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._torch import torch
from .code import Part, levels
from .frame import Frame
from .geometry import Geometry
from .grid import Grid
from .mapping import Mapping
from .tables import AxisTable, build_tables

LABEL_MAX = 65535


@dataclass
class Restored:
    """Labels on ``grid`` (Z, Y, X) in the canonical frame, with what places them."""

    labels: np.ndarray
    grid: Grid
    geometry: Geometry
    frame: Frame | None
    parts: list[str]
    interp: str
    roi: tuple | None = None


def array_grid(part: Part) -> Grid:
    """The part's stored array as a Grid in its OWN frame: voxel 0 at 0 mm, true spacing."""
    g = part.field.geometry
    return Grid(shape=tuple(int(v) for v in part.field.support.shape[1:]) if part.field.ranks is None
                else tuple(int(v) for v in part.field.ranks.shape[1:]),
                spacing=tuple(float(v) for v in g.spacing_zyx))


def mapping_of(part: Part, grid: Grid) -> Mapping:
    """Output index -> stored-array coordinate."""
    if part.field.frame:
        mp = Frame.from_meta(part.field.frame).mapping(grid)
        start = tuple(int(v) for v in part.envelope_start)
        if any(start):
            mp = mp >> Mapping((1.0, 1.0, 1.0), tuple(-float(v) for v in start))
        return mp
    return Mapping.between(grid, array_grid(part))


def resolve_grid(part: Part, grid) -> tuple[Grid, Frame | None]:
    """``"input"`` (needs the frame), a spacing in mm, or a Grid -> the output Grid."""
    if part.field.frame:
        fr = Frame.from_meta(part.field.frame)
        return fr.resolve_grid(grid), fr
    if grid == "input" or grid is None:
        raise ValueError("this part carries no frame: name the grid (a spacing in mm, or a Grid "
                         "in the part's array frame)")
    if isinstance(grid, Grid):
        return grid, None
    return Grid.isotropic(float(grid), like=array_grid(part)), None


def _same_placement(a: Part, b: Part) -> bool:
    ga, gb = a.field.geometry, b.field.geometry
    if bool(a.field.frame) != bool(b.field.frame):
        return False
    if a.field.frame:
        # the same source frame: everything but the per-part model grid must agree
        skip = ("model_shape", "model_spacing")
        fa = {k: v for k, v in a.field.frame.items() if k not in skip}
        fb = {k: v for k, v in b.field.frame.items() if k not in skip}
        return fa == fb
    return (np.allclose(ga.origin_xyz, gb.origin_xyz) and np.allclose(ga.direction_xyz, gb.direction_xyz)
            and np.allclose(ga.spacing_zyx, gb.spacing_zyx)
            and tuple(a.field.ranks.shape[1:]) == tuple(b.field.ranks.shape[1:]))


def output_geometry(part: Part, grid: Grid, fr: Frame | None, box) -> Geometry:
    """World placement of the labels on ``grid`` (or of its ``box``), canonical orientation."""
    if fr is not None:
        geo = fr.output_geometry(grid)
        D = np.asarray(geo.direction_xyz).reshape(3, 3)
        origin = np.asarray(geo.origin_xyz)
        direction = geo.direction_xyz
    else:
        g = part.field.geometry
        D = np.asarray(g.direction_xyz).reshape(3, 3)
        origin = np.asarray(g.origin_xyz) + D @ np.asarray(grid.origin, float)[::-1]
        direction = g.direction_xyz
    off_xyz = np.asarray([box[2][0], box[1][0], box[0][0]], float) * np.asarray(grid.spacing)[::-1]
    return Geometry(spacing_zyx=tuple(float(v) for v in grid.spacing),
                    shape_zyx=tuple(b - a for a, b in box),
                    origin_xyz=tuple(float(v) for v in origin + D @ off_xyz),
                    direction_xyz=tuple(float(v) for v in direction))


def roi_of(parts, extents, grid_out: Grid, *, halo: int = 1) -> tuple:
    """The output-index box holding ``extents`` - ``[(part index, inclusive 6-box on that
    part's array), ...]`` - ``halo`` model voxels wider, through each part's inverse
    mapping. A heuristic: a class can be elected a little past where it won."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for i, e in extents:
        inv = mapping_of(parts[i], grid_out).inverse()
        for corner in (np.asarray(e[0::2], float) - halo, np.asarray(e[1::2], float) + halo):
            o = inv.apply(corner)
            lo = np.minimum(lo, np.floor(o))
            hi = np.maximum(hi, np.ceil(o))
    if not np.isfinite(lo).all():
        raise ValueError("no extents")
    lo = np.clip(lo, 0, np.asarray(grid_out.shape) - 1).astype(int)
    hi = np.clip(hi + 1, 1, np.asarray(grid_out.shape)).astype(int)
    return tuple((int(a), int(b)) for a, b in zip(lo, hi))


# ----------------------------------------------------------------------------------------

def restore(parts, *, grid="input", interp: str = "linear", roi=None, device=None,
            slab_voxels: int = 1 << 20, progress=None) -> Restored:
    """Labels on ``grid`` from ``parts`` (in paint order). ``roi`` is an output-index box
    ``((z0, z1), (y0, y1), (x0, x1))``, half-open; only that box is computed and returned."""
    if interp not in ("linear", "nearest"):
        raise ValueError(f"interp must be 'linear' or 'nearest'; got {interp!r}")
    parts = list(parts)
    if not parts:
        raise ValueError("no parts")
    for p in parts:
        if p.field.meta.get("mode", "ranked") != "ranked":
            raise NotImplementedError(f"mode {p.field.meta.get('mode')!r} restore is not built")
        if p.field.labels is None or p.field.geometry is None:
            raise ValueError("every part needs labels and a geometry")
        if not _same_placement(parts[0], p):
            raise ValueError(f"part {p.name!r} is not placed like part {parts[0].name!r}; "
                             "a multi-part restore needs one placement")
    grid_out, fr = resolve_grid(parts[0], grid)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    box = roi or tuple((0, n) for n in grid_out.shape)
    for (a, b), n in zip(box, grid_out.shape):
        if not (0 <= a < b <= n):
            raise ValueError(f"roi {roi} escapes the output grid {grid_out.shape}")
    out_shape = tuple(b - a for a, b in box)
    max_label = max(int(np.max(p.field.labels)) for p in parts)
    if max_label > LABEL_MAX:
        raise ValueError(f"a label of {max_label} does not fit uint16")
    dtype = np.uint8 if max_label < 256 else np.uint16
    gpu = interp == "linear" and _gpu_kernel(dev, parts)
    if gpu:
        out_t = torch.zeros(out_shape, dtype=torch.uint8 if dtype is np.uint8 else torch.uint16, device=dev)
        out = None
    else:
        out = np.zeros(out_shape, dtype)             # zero = background; paint leaves it
        out_t = None
    names = []
    for n_part, p in enumerate(parts):
        names.append(p.name or str(n_part))
        if progress:
            progress(f"restore {names[-1]} ({n_part + 1}/{len(parts)})")
        _restore_part(p, grid_out, box, interp, out if out is not None else out_t,
                      paint=len(parts) > 1, dev=dev, slab_voxels=slab_voxels, gpu=gpu)
    if gpu:
        out = out_t.cpu().numpy()
    geo = output_geometry(parts[0], grid_out, fr, box)
    return Restored(labels=out, grid=grid_out, geometry=geo, frame=fr, parts=names, interp=interp, roi=roi)


def _gpu_kernel(dev, parts) -> bool:
    if dev.type == "mps":
        from .backends import metal
        return metal.available() and all(p.field.depth <= metal.RANKED_NMAX for p in parts)
    if dev.type == "cuda":
        from .backends import triton_gpu
        return triton_gpu.available()
    return False


def _spans(tabs):
    out = []
    for i0, i1, _f in tabs:
        ok = i0 >= 0
        if not ok.any():
            return None
        out.append((int(min(i0[ok].min(), i1[ok].min())), int(max(i0[ok].max(), i1[ok].max())) + 1))
    return out


def _shifted(i0, i1, f, lo):
    """Tables relative to a model box starting at ``lo``; an outside row stays -1."""
    return (np.where(i0 >= 0, i0 - lo, -1).astype(np.int32),
            np.where(i0 >= 0, i1 - lo, 0).astype(np.int32),
            f.astype(np.float32))


def _restore_part(part: Part, grid_out: Grid, box, interp, out, *, paint, dev, slab_voxels, gpu):
    f = part.field
    lut_np = np.asarray([int(v) for v in f.labels], dtype=np.int64)
    if lut_np.max() > (255 if (out.dtype == np.uint8 if isinstance(out, np.ndarray) else out.dtype == torch.uint8) else LABEL_MAX):
        raise ValueError(f"a label of {lut_np.max()} does not fit the output buffer")
    if lut_np.min() < 0:
        raise ValueError(f"a label of {lut_np.min()} is negative; labels are unsigned")
    if len(lut_np) < f.classes:
        raise ValueError(f"the label table has {len(lut_np)} entries for {f.classes} classes")
    clip = f.clip
    lut_levels = levels(f.meta)
    ranks_arr, sup_arr = f.ranks, f.support
    N = int(ranks_arr.shape[0])
    src_shape = tuple(int(v) for v in ranks_arr.shape[1:])
    mp = mapping_of(part, grid_out)
    tz, ty, tx = build_tables(grid_out.shape, src_shape, mp, interp=interp, outside="background")
    tabs = [(t.i0[a:b], t.i1[a:b], t.f[a:b]) for t, (a, b) in zip((tz, ty, tx), box)]
    spans = _spans(tabs)
    if spans is None:
        return                                       # nothing of this part is under the ROI
    sz, sy, sx = spans
    R = np.ascontiguousarray(ranks_arr[:, sz[0]:sz[1], sy[0]:sy[1], sx[0]:sx[1]])
    S = (np.ascontiguousarray(sup_arr[:, sz[0]:sz[1], sy[0]:sy[1], sx[0]:sx[1]]) if N > 1
         else np.zeros((0,) + R.shape[1:], np.uint8))
    shifted = [_shifted(i0, i1, fv, lo) for (i0, i1, fv), (lo, _) in zip(tabs, (sz, sy, sx))]
    if gpu:
        _restore_part_gpu(R, S, shifted, lut_np, lut_levels, clip, out, paint, dev)
        return
    _restore_part_torch(R, S, shifted, lut_np, lut_levels, clip, out, paint, dev, slab_voxels, interp)


def _restore_part_gpu(R, S, shifted, lut_np, lut_levels, clip, out_t, paint, dev):
    from .backends import metal, triton_gpu
    Rt = torch.from_numpy(R).to(dev)
    if Rt.dtype not in (torch.uint8, torch.uint16):
        Rt = Rt.to(torch.uint16)
    St = torch.from_numpy(S).to(dev)
    tables = tuple(AxisTable(i0, i1, fv, fv.astype(np.float64)) for i0, i1, fv in shifted)
    runner = metal.run_ranked if dev.type == "mps" else triton_gpu.run_ranked
    runner(Rt, St, out_t, tables, lut_np, levels=lut_levels, clip=clip, paint=paint)


def _restore_part_torch(R, S, shifted, lut_np, lut_levels, clip, out, paint, dev, slab_voxels, interp):
    lut = torch.as_tensor(lut_np, device=dev)
    lv = torch.from_numpy(lut_levels).to(dev)
    N = R.shape[0]
    Rz, Ry, Rx = R.shape[1:]
    Rt = torch.from_numpy(R).reshape(N, -1).to(dev)           # bytes; widened per gather
    St = torch.from_numpy(S).reshape(max(N - 1, 0), -1).to(dev) if N > 1 else None
    (z0s, z1s, zf), (y0s, y1s, yf), (x0s, x1s, xf) = shifted
    nz_out, ny_out, nx_out = out.shape
    rows_per_slab = max(1, slab_voxels // max(1, ny_out * nx_out))

    def axis(i0, i1, fv):
        v = torch.from_numpy(i0 >= 0).to(dev)
        a = torch.from_numpy(np.where(i0 >= 0, i0, 0)).to(dev).long()
        b = torch.from_numpy(np.where(i0 >= 0, i1, 0)).to(dev).long()
        return v, a, b, torch.from_numpy(fv).to(dev)
    vy, ay, by, wy = axis(y0s, y1s, yf)
    vx, ax_, bx, wx = axis(x0s, x1s, xf)
    for za in range(0, nz_out, rows_per_slab):
        zb = min(za + rows_per_slab, nz_out)
        vz, az, bz, wz = axis(z0s[za:zb], z1s[za:zb], zf[za:zb])
        valid = (vz[:, None, None] & vy[None, :, None] & vx[None, None, :]).reshape(-1)
        V = valid.numel()
        corners, weights = [], []
        for iz, fz in ((az, 1 - wz), (bz, wz)):
            for iy, fy in ((ay, 1 - wy), (by, wy)):
                for ix, fx in ((ax_, 1 - wx), (bx, wx)):
                    idx = (iz[:, None, None] * Ry + iy[None, :, None]) * Rx + ix[None, None, :]
                    corners.append(idx.reshape(V))
                    weights.append((fz[:, None, None] * fy[None, :, None] * fx[None, None, :]).reshape(V))
        assert all(int(c.min()) >= 0 and int(c.max()) < Rz * Ry * Rx for c in corners)
        win = torch.stack([Rt[0][c].to(torch.int32) for c in corners])
        uniform = (win == win[0:1]).all(0)
        best = win[0].clone()
        hard = (~uniform & valid).nonzero(as_tuple=True)[0]
        if hard.numel() and interp == "linear":
            best[hard] = _decide(Rt, St, [c[hard] for c in corners], [w[hard] for w in weights], N, clip, lv)
        lab = lut[(best - 1).clamp(min=0)]
        keep = valid & (best != 1) if paint else valid
        lab = lab.reshape(zb - za, ny_out, nx_out).cpu().numpy()
        keep = keep.reshape(zb - za, ny_out, nx_out).cpu().numpy()
        out[za:zb] = np.where(keep, lab.astype(out.dtype), out[za:zb] if paint else 0)


def _decide(R, S, corners, weights, N, clip, lv):
    """The argmax of the interpolated deficit over every class stored at any corner; ties to
    the candidate with more winner mass, then the lower index (the kernels do the same, in
    the same order of operations)."""
    Vh = corners[0].numel()
    dev = R.device
    cls = [[R[j][c].to(torch.int32) for j in range(N)] for c in corners]
    lvl = [[torch.zeros(Vh, device=dev) if j == 0 else -lv[S[j - 1][c].long()] for j in range(N)]
           for c in corners]
    best_val = torch.full((Vh,), -float("inf"), device=dev)
    best_mass = torch.zeros(Vh, device=dev)
    best_cls = torch.zeros(Vh, dtype=torch.int32, device=dev)
    floor = torch.full((Vh,), -clip, device=dev)
    for q in range(8):
        for j in range(N):
            cand = cls[q][j]
            live = cand != 0
            val = torch.zeros(Vh, device=dev)
            mass = torch.zeros(Vh, device=dev)
            for q2 in range(8):
                d = floor
                for j2 in range(N):
                    d = torch.where(cls[q2][j2] == cand, lvl[q2][j2], d)
                val = val + weights[q2] * d
                mass = mass + weights[q2] * (cls[q2][0] == cand).to(torch.float32)
            val = torch.where(live, val, torch.full_like(val, -float("inf")))
            better = val > best_val
            tie = (val == best_val) & ((mass > best_mass) | ((mass == best_mass) & (cand < best_cls)))
            take = better | tie
            best_val = torch.where(take, val, best_val)
            best_mass = torch.where(take, mass, best_mass)
            best_cls = torch.where(take, cand, best_cls)
    return best_cls
