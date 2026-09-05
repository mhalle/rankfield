"""CUDA: the store-backed restore as two Triton kernels.

Pass one settles every output voxel whose eight corners share a winner and flags the rest;
pass two decides the flagged ones from the candidates at the eight corners, blending each
candidate's deficit with the same weights in the same order as the torch path. Fused
multiply-add is off (``enable_fp_fusion=False``) and levels come from a 256-entry table, so
the decisions match the torch and Metal paths bit for bit.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR = None
except Exception as _e:                                     # pragma: no cover - depends on the box
    triton = None
    _TRITON_IMPORT_ERROR = _e


def _has_fp_fusion_option() -> bool:
    """``enable_fp_fusion=False`` is load-bearing (an FMA decides near-ties differently from
    the CPU); Triton drops an unknown launch option silently, so its absence counts as
    unavailable rather than as a wrong answer."""
    try:
        from triton.backends.nvidia.compiler import CUDAOptions
        return "enable_fp_fusion" in getattr(CUDAOptions, "__dataclass_fields__", {})
    except Exception:                                       # pragma: no cover
        return False


def available() -> bool:
    return triton is not None and torch.cuda.is_available() and _has_fp_fusion_option()


def why_unavailable() -> str:
    if triton is None:
        return f"triton is not installed ({_TRITON_IMPORT_ERROR})"
    if not torch.cuda.is_available():
        return "no CUDA device"
    if not _has_fp_fusion_option():
        return "this triton has no enable_fp_fusion launch option (need >= 3.0 on the NVIDIA backend)"
    return ""


if triton is not None:

    @triton.jit
    def ranked_uniform(ranks_ptr, z0_ptr, z1_ptr, y0_ptr, y1_ptr, x0_ptr, x1_ptr, lut_ptr,
                       out_ptr, flag_ptr, Zs, Ys, Xs, Ya, Xa, n_out, background,
                       PAINT: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out
        x = offs % Xa
        y = (offs // Xa) % Ya
        z = offs // (Xa * Ya)
        z0 = tl.load(z0_ptr + z, mask=mask, other=0)
        y0 = tl.load(y0_ptr + y, mask=mask, other=0)
        x0 = tl.load(x0_ptr + x, mask=mask, other=0)
        inside = (z0 >= 0) & (y0 >= 0) & (x0 >= 0)
        z0 = tl.where(inside, z0, 0)
        y0 = tl.where(inside, y0, 0)
        x0 = tl.where(inside, x0, 0)
        z1 = tl.load(z1_ptr + z, mask=mask, other=0)
        y1 = tl.load(y1_ptr + y, mask=mask, other=0)
        x1 = tl.load(x1_ptr + x, mask=mask, other=0)
        plane = Ys * Xs
        lm = mask & inside
        r0 = tl.load(ranks_ptr + z0 * plane + y0 * Xs + x0, mask=lm, other=0).to(tl.int32)
        same = r0 == tl.load(ranks_ptr + z0 * plane + y0 * Xs + x1, mask=lm, other=0).to(tl.int32)
        same &= r0 == tl.load(ranks_ptr + z0 * plane + y1 * Xs + x0, mask=lm, other=0).to(tl.int32)
        same &= r0 == tl.load(ranks_ptr + z0 * plane + y1 * Xs + x1, mask=lm, other=0).to(tl.int32)
        same &= r0 == tl.load(ranks_ptr + z1 * plane + y0 * Xs + x0, mask=lm, other=0).to(tl.int32)
        same &= r0 == tl.load(ranks_ptr + z1 * plane + y0 * Xs + x1, mask=lm, other=0).to(tl.int32)
        same &= r0 == tl.load(ranks_ptr + z1 * plane + y1 * Xs + x0, mask=lm, other=0).to(tl.int32)
        same &= r0 == tl.load(ranks_ptr + z1 * plane + y1 * Xs + x1, mask=lm, other=0).to(tl.int32)
        label = tl.load(lut_ptr + tl.maximum(r0 - 1, 0), mask=lm, other=background)
        out_dtype = out_ptr.dtype.element_ty
        if PAINT:
            tl.store(out_ptr + offs, label.to(out_dtype), mask=lm & same & (r0 != 1))
        else:
            tl.store(out_ptr + offs, tl.where(inside, label, background).to(out_dtype), mask=mask & (same | ~inside))
        tl.store(flag_ptr + offs, (inside & ~same).to(tl.int8), mask=mask)

    @triton.jit
    def ranked_decide(ranks_ptr, support_ptr, levels_ptr, idx_ptr, z0_ptr, z1_ptr, zf_ptr,
                      y0_ptr, y1_ptr, yf_ptr, x0_ptr, x1_ptr, xf_ptr, lut_ptr, out_ptr,
                      N, Zs, Ys, Xs, Ya, Xa, n_idx, clip, PAINT: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        k = pid * BLOCK + tl.arange(0, BLOCK)
        mask = k < n_idx
        offs = tl.load(idx_ptr + k, mask=mask, other=0)
        x = offs % Xa
        y = (offs // Xa) % Ya
        z = offs // (Xa * Ya)
        z0 = tl.load(z0_ptr + z, mask=mask, other=0)
        y0 = tl.load(y0_ptr + y, mask=mask, other=0)
        x0 = tl.load(x0_ptr + x, mask=mask, other=0)
        z1 = tl.load(z1_ptr + z, mask=mask, other=0)
        y1 = tl.load(y1_ptr + y, mask=mask, other=0)
        x1 = tl.load(x1_ptr + x, mask=mask, other=0)
        zf = tl.load(zf_ptr + z, mask=mask, other=0.0)
        yf = tl.load(yf_ptr + y, mask=mask, other=0.0)
        xf = tl.load(xf_ptr + x, mask=mask, other=0.0)
        plane = Ys * Xs
        cs = Zs * plane
        wz0 = 1.0 - zf
        wy0 = 1.0 - yf
        wx0 = 1.0 - xf
        b0 = z0 * plane + y0 * Xs + x0
        b1 = z0 * plane + y0 * Xs + x1
        b2 = z0 * plane + y1 * Xs + x0
        b3 = z0 * plane + y1 * Xs + x1
        b4 = z1 * plane + y0 * Xs + x0
        b5 = z1 * plane + y0 * Xs + x1
        b6 = z1 * plane + y1 * Xs + x0
        b7 = z1 * plane + y1 * Xs + x1
        w0 = (wz0 * wy0) * wx0
        w1 = (wz0 * wy0) * xf
        w2 = (wz0 * yf) * wx0
        w3 = (wz0 * yf) * xf
        w4 = (zf * wy0) * wx0
        w5 = (zf * wy0) * xf
        w6 = (zf * yf) * wx0
        w7 = (zf * yf) * xf
        best_v = tl.full((BLOCK,), float("-inf"), tl.float32)
        best_m = tl.zeros((BLOCK,), tl.float32)
        best_c = tl.zeros((BLOCK,), tl.int32)
        for q in tl.static_range(8):
            if q == 0:
                bq = b0
            elif q == 1:
                bq = b1
            elif q == 2:
                bq = b2
            elif q == 3:
                bq = b3
            elif q == 4:
                bq = b4
            elif q == 5:
                bq = b5
            elif q == 6:
                bq = b6
            else:
                bq = b7
            for j in range(0, N):
                c = tl.load(ranks_ptr + j * cs + bq, mask=mask, other=0).to(tl.int32)
                live = c != 0
                val = tl.zeros((BLOCK,), tl.float32)
                mass = tl.zeros((BLOCK,), tl.float32)
                for q2 in tl.static_range(8):
                    if q2 == 0:
                        bq2 = b0
                        wq2 = w0
                    elif q2 == 1:
                        bq2 = b1
                        wq2 = w1
                    elif q2 == 2:
                        bq2 = b2
                        wq2 = w2
                    elif q2 == 3:
                        bq2 = b3
                        wq2 = w3
                    elif q2 == 4:
                        bq2 = b4
                        wq2 = w4
                    elif q2 == 5:
                        bq2 = b5
                        wq2 = w5
                    elif q2 == 6:
                        bq2 = b6
                        wq2 = w6
                    else:
                        bq2 = b7
                        wq2 = w7
                    d = tl.full((BLOCK,), 0.0, tl.float32) - clip
                    r0 = tl.load(ranks_ptr + bq2, mask=mask, other=0).to(tl.int32)
                    found = r0 == c
                    d = tl.where(found, 0.0, d)
                    mass = mass + wq2 * found.to(tl.float32)
                    for j2 in range(1, N):
                        r = tl.load(ranks_ptr + j2 * cs + bq2, mask=mask, other=0).to(tl.int32)
                        hit = (r == c) & ~found
                        s = tl.load(support_ptr + (j2 - 1) * cs + bq2, mask=mask, other=0).to(tl.int32)
                        lvl = -tl.load(levels_ptr + s, mask=mask, other=0.0)
                        d = tl.where(hit, lvl, d)
                        found = found | hit
                    val = val + wq2 * d
                val = tl.where(live, val, float("-inf"))
                tie = (val == best_v) & ((mass > best_m) | ((mass == best_m) & (c < best_c)))
                take = (val > best_v) | tie
                best_v = tl.where(take, val, best_v)
                best_m = tl.where(take, mass, best_m)
                best_c = tl.where(take, c, best_c)
        label = tl.load(lut_ptr + tl.maximum(best_c - 1, 0), mask=mask, other=0)
        out_dtype = out_ptr.dtype.element_ty
        if PAINT:
            tl.store(out_ptr + offs, label.to(out_dtype), mask=mask & (best_c != 1))
        else:
            tl.store(out_ptr + offs, label.to(out_dtype), mask=mask)


@torch.no_grad()
def run_ranked(ranks: torch.Tensor, support: torch.Tensor, out: torch.Tensor, tables, lut, *,
               levels, clip: float, paint: bool, background: int = 0, block: int = 256) -> None:
    if not available():
        raise RuntimeError(f"rankfield.backends.triton_gpu: {why_unavailable()}")
    if ranks.device.type != "cuda" or out.device.type != "cuda":
        raise ValueError("rankfield.backends.triton_gpu: ranks and out must be on a CUDA device")
    if ranks.dtype not in (torch.uint8, torch.uint16, torch.int32):
        raise TypeError(f"rankfield.backends.triton_gpu: ranks must be uint8/uint16/int32; got {ranks.dtype}")
    if out.dtype not in (torch.uint8, torch.uint16):
        raise TypeError(f"rankfield.backends.triton_gpu: out must be uint8 or uint16; got {out.dtype}")
    ranks = ranks.contiguous()
    N, Zs, Ys, Xs = (int(v) for v in ranks.shape)
    if N * Zs * Ys * Xs >= 2 ** 31:
        raise ValueError("rankfield.backends.triton_gpu: N * source volume must be < 2^31 (32-bit offsets)")
    support = support.contiguous() if N > 1 else torch.zeros((1,), dtype=torch.uint8, device=ranks.device)
    if N > 1 and tuple(support.shape) != (N - 1, Zs, Ys, Xs):
        raise ValueError("rankfield.backends.triton_gpu: support does not match ranks")
    Za, Ya, Xa = (int(v) for v in out.shape)
    n_out = Za * Ya * Xa
    if n_out >= 2 ** 31:
        raise ValueError("rankfield.backends.triton_gpu: output volume must be < 2^31 voxels")
    limit = 255 if out.dtype == torch.uint8 else 65535
    if int(np.max(lut)) > limit:
        raise ValueError(f"rankfield.backends.triton_gpu: a label of {int(np.max(lut))} does not fit {out.dtype}")
    lv = np.ascontiguousarray(levels, dtype=np.float32)
    if lv.shape != (256,):
        raise ValueError("rankfield.backends.triton_gpu: levels must be a 256-entry table")
    dev = ranks.device

    def to_i(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)).to(dev)

    def to_f(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)

    tz, ty, tx = tables
    if (len(tz.i0), len(ty.i0), len(tx.i0)) != (Za, Ya, Xa):
        raise ValueError("rankfield.backends.triton_gpu: the tables do not match the output shape")
    z0, z1, zf = to_i(tz.i0), to_i(tz.i1), to_f(tz.f)
    y0, y1, yf = to_i(ty.i0), to_i(ty.i1), to_f(ty.f)
    x0, x1, xf = to_i(tx.i0), to_i(tx.i1), to_f(tx.f)
    lut_t = to_i(lut)
    lv_t = to_f(lv)
    flag = torch.zeros(n_out, dtype=torch.int8, device=dev)
    ranked_uniform[(triton.cdiv(n_out, block),)](
        ranks, z0, z1, y0, y1, x0, x1, lut_t, out, flag, Zs, Ys, Xs, Ya, Xa, n_out,
        int(background), PAINT=bool(paint), BLOCK=block)
    idx = flag.nonzero(as_tuple=True)[0].to(torch.int32)
    n_idx = int(idx.numel())
    if n_idx:
        ranked_decide[(triton.cdiv(n_idx, block),)](
            ranks, support, lv_t, idx, z0, z1, zf, y0, y1, yf, x0, x1, xf, lut_t, out,
            N, Zs, Ys, Xs, Ya, Xa, n_idx, float(clip), PAINT=bool(paint), BLOCK=block,
            enable_fp_fusion=False)
