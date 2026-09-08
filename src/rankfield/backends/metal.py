"""Apple GPU: the store-backed restore as a Metal kernel (PyTorch >= 2.7, compile_shader).

One thread per output voxel over the RANK and SUPPORT planes. The candidates are the
classes stored at the eight corners; each candidate's deficit (0 where it won, minus its
level where it trailed, minus the clip where absent) is blended with the same eight
weights, in the same order, as the torch path in ``rankfield.restore``, with fp contraction
off, so the two agree bit for bit. Levels come from a 256-entry table, never a formula.
"""
from __future__ import annotations

import numpy as np

from . import max_class
from .._torch import have_torch, no_grad, torch

RANKED_NMAX = 8             # rank planes the candidate list is sized for (8 corners x N)
_VARIANTS = [("uchar", "uchar"), ("uchar", "ushort"), ("ushort", "uchar"), ("ushort", "ushort")]

_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
{PRAGMA}
"""

_KERNEL = r"""
kernel void {NAME}(
    device const {RANK_T}*  ranks   [[buffer(0)]],
    device const uchar*     support [[buffer(1)]],
    device const float*     levels  [[buffer(2)]],
    device const int*       iparams [[buffer(3)]],
    device const float*     fparams [[buffer(4)]],
    device const int*       tz0     [[buffer(5)]],
    device const int*       tz1     [[buffer(6)]],
    device const float*     tzf     [[buffer(7)]],
    device const int*       ty0     [[buffer(8)]],
    device const int*       ty1     [[buffer(9)]],
    device const float*     tyf     [[buffer(10)]],
    device const int*       tx0     [[buffer(11)]],
    device const int*       tx1     [[buffer(12)]],
    device const float*     txf     [[buffer(13)]],
    device const int*       lut     [[buffer(14)]],
    device {OUT_T}*         out     [[buffer(15)]],
    uint elem [[thread_position_in_grid]])
{
    const int n_slab = iparams[8];
    if ((int)elem >= n_slab) return;
    const int N = iparams[0], Zt = iparams[1], Yt = iparams[2], Xt = iparams[3];
    const int Ya = iparams[5], Xa = iparams[6];
    const int z_offset = iparams[7];
    const int paint = iparams[9], background = iparams[10];
    const float clip = fparams[0];

    const uint plane_out = (uint)Xa * (uint)Ya;
    const int x = (int)(elem % (uint)Xa);
    const int y = (int)((elem / (uint)Xa) % (uint)Ya);
    const int z = z_offset + (int)(elem / plane_out);
    const long oidx = (long)z * (long)plane_out + (long)y * (long)Xa + (long)x;

    const int z0 = tz0[z], y0 = ty0[y], x0 = tx0[x];
    if (z0 < 0 || y0 < 0 || x0 < 0) {
        if (!paint) out[oidx] = ({OUT_T})background;
        return;
    }
    const int z1 = tz1[z], y1 = ty1[y], x1 = tx1[x];
    const float zf = tzf[z], yf = tyf[y], xf = txf[x];
    const float wz[2] = {1.0f - zf, zf};
    const float wy[2] = {1.0f - yf, yf};
    const float wx[2] = {1.0f - xf, xf};
    const int plane = Yt * Xt;
    int b[8];
    float w[8];
    {
        const int zz[2] = {z0, z1}, yy[2] = {y0, y1}, xx[2] = {x0, x1};
        int q = 0;
        for (int cz = 0; cz < 2; cz++)
            for (int cy = 0; cy < 2; cy++)
                for (int cx = 0; cx < 2; cx++) {
                    b[q] = zz[cz] * plane + yy[cy] * Xt + xx[cx];
                    w[q] = (wz[cz] * wy[cy]) * wx[cx];     // the torch path's product order
                    q++;
                }
    }
    const long cs = (long)Zt * (long)plane;

    int best_c = (int)ranks[b[0]];
    bool uniform = true;
    for (int q = 1; q < 8; q++) if ((int)ranks[b[q]] != best_c) { uniform = false; break; }
    if (!uniform) {
        float best_v = -INFINITY;
        float best_m = 0.0f;
        best_c = 0;
        int seen[{NMAX8}];
        int n_seen = 0;
        for (int q = 0; q < 8; q++) {
            for (int j = 0; j < N; j++) {
                const int c = (int)ranks[(long)j * cs + b[q]];
                if (c == 0) break;                            // sentinels form a suffix
                bool dup = false;
                for (int t = 0; t < n_seen; t++) if (seen[t] == c) { dup = true; break; }
                if (dup) continue;
                seen[n_seen++] = c;
                float val = 0.0f;
                float mass = 0.0f;
                for (int q2 = 0; q2 < 8; q2++) {
                    float d = -clip;
                    for (int j2 = 0; j2 < N; j2++) {
                        const int r = (int)ranks[(long)j2 * cs + b[q2]];
                        if (r == 0) break;
                        if (r == c) {
                            d = (j2 == 0) ? 0.0f : -levels[(int)support[(long)(j2 - 1) * cs + b[q2]]];
                            break;
                        }
                    }
                    val = val + w[q2] * d;
                    mass = mass + w[q2] * (((int)ranks[b[q2]] == c) ? 1.0f : 0.0f);
                }
                if (val > best_v || (val == best_v && (mass > best_m || (mass == best_m && c < best_c)))) {
                    best_v = val; best_m = mass; best_c = c;
                }
            }
        }
    }
    if (paint && best_c == 1) return;                         // class 0 = background: transparent
    out[oidx] = ({OUT_T})lut[best_c - 1];
}
"""

_LIB = None
_FP_CONTRACT: str | None = None


def available() -> bool:
    return bool(have_torch() and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                and hasattr(torch.mps, "compile_shader"))


def source(fp_contract_off: bool = True) -> str:
    pragma = "#pragma clang fp contract(off)" if fp_contract_off else ""
    parts = [_HEADER.replace("{PRAGMA}", pragma)]
    for rt, ot in _VARIANTS:
        parts.append(_KERNEL.replace("{NAME}", f"rk_{rt}_{ot}").replace("{RANK_T}", rt)
                     .replace("{OUT_T}", ot).replace("{NMAX8}", str(8 * RANKED_NMAX)))
    return "\n".join(parts)


def library():
    """Compile once, with fused-multiply-add contraction off so the blend rounds exactly
    like the torch path; falls back to the compiler default if the pragma is rejected."""
    global _LIB, _FP_CONTRACT
    if _LIB is None:
        if not available():
            raise RuntimeError("rankfield.backends.metal: needs an MPS device and torch.mps.compile_shader")
        try:
            _LIB = torch.mps.compile_shader(source(True))
            _FP_CONTRACT = "off"
        except Exception:
            _LIB = torch.mps.compile_shader(source(False))
            _FP_CONTRACT = "default"
    return _LIB


def fp_contract() -> str | None:
    library()
    return _FP_CONTRACT


@no_grad
def run_ranked(ranks: torch.Tensor, support: torch.Tensor, out: torch.Tensor, tables, lut, *,
               levels, clip: float, paint: bool, background: int = 0, slab_voxels: int = 1 << 26,
               group_size: int = 256) -> None:
    """``ranks`` (N, Zt, Yt, Xt) uint8/uint16 and ``support`` (N-1, ...) uint8 on 'mps' ->
    labels into ``out`` (uint8/uint16, same device) through the host tables, the label LUT
    and the 256-entry ``levels`` table."""
    if ranks.device.type != "mps" or out.device.type != "mps" or support.device.type != "mps":
        raise ValueError("rankfield.backends.metal: ranks, support and out must be on the 'mps' device")
    rt = {torch.uint8: "uchar", torch.uint16: "ushort"}.get(ranks.dtype)
    ot = {torch.uint8: "uchar", torch.uint16: "ushort"}.get(out.dtype)
    if rt is None or ot is None:
        raise TypeError(f"rankfield.backends.metal: ranks {ranks.dtype} / out {out.dtype} must be uint8 or uint16")
    if support.dtype != torch.uint8:
        raise TypeError(f"rankfield.backends.metal: support must be uint8; got {support.dtype}")
    N, Zt, Yt, Xt = (int(v) for v in ranks.shape)
    if N > RANKED_NMAX:
        raise ValueError(f"rankfield.backends.metal: the kernel holds up to {RANKED_NMAX} planes; got {N}")
    if N > 1 and tuple(support.shape) != (N - 1, Zt, Yt, Xt):
        raise ValueError("rankfield.backends.metal: support does not match ranks")
    if Zt * Yt * Xt >= 2 ** 31:
        raise ValueError("rankfield.backends.metal: per-plane volume must be < 2^31 voxels")
    ranks = ranks.contiguous()
    support = support.contiguous() if N > 1 else torch.zeros((1,), dtype=torch.uint8, device=ranks.device)
    if not out.is_contiguous():
        raise ValueError("rankfield.backends.metal: out must be contiguous")
    dev = ranks.device
    Za, Ya, Xa = (int(s) for s in out.shape)
    if Za * Ya * Xa >= 2 ** 31:
        raise ValueError("rankfield.backends.metal: output volume must be < 2^31 voxels")
    tz, ty, tx = tables
    if (len(tz.i0), len(ty.i0), len(tx.i0)) != (Za, Ya, Xa):
        raise ValueError("rankfield.backends.metal: the tables do not match the output shape")
    limit = 255 if out.dtype == torch.uint8 else 65535
    if int(np.max(lut)) > limit:
        raise ValueError(f"rankfield.backends.metal: a label of {int(np.max(lut))} does not fit {out.dtype}")
    top = max_class(ranks)                            # the kernel indexes lut[class - 1] unchecked
    if len(lut) < top:
        raise ValueError(f"rankfield.backends.metal: the label table has {len(lut)} entries for a class "
                         f"{top}")
    lv = np.ascontiguousarray(levels, dtype=np.float32)
    if lv.shape != (256,):
        raise ValueError("rankfield.backends.metal: levels must be a 256-entry table")

    def to_i(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)).to(dev)

    def to_f(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)

    bufs = [to_i(tz.i0), to_i(tz.i1), to_f(tz.f), to_i(ty.i0), to_i(ty.i1), to_f(ty.f),
            to_i(tx.i0), to_i(tx.i1), to_f(tx.f)]
    lut_t = to_i(lut)
    lv_t = to_f(lv)
    fparams = torch.tensor([float(clip)], dtype=torch.float32, device=dev)
    kernel = getattr(library(), f"rk_{rt}_{ot}")
    plane = Ya * Xa
    planes_per_launch = max(1, int(slab_voxels) // plane)
    for z_off in range(0, Za, planes_per_launch):
        nz = min(planes_per_launch, Za - z_off)
        n_slab = nz * plane
        iparams = torch.tensor([N, Zt, Yt, Xt, Za, Ya, Xa, z_off, n_slab, int(bool(paint)), int(background)],
                               dtype=torch.int32, device=dev)
        kernel(ranks, support, lv_t, iparams, fparams, *bufs, lut_t, out, threads=n_slab, group_size=group_size)
