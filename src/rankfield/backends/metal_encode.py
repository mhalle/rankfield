"""Apple GPU: the encoder's selection and order as one Metal kernel, a thread per voxel.

What it replaces, on MPS: the shell's 27 scatters, the key, ``_select`` (two ``topk`` and a
walk over the classes), both ``settle_ties`` and the stable sort - steps whose ``topk`` and
``sort`` over millions of rows of a few classes took 9 of the 13.5 s a K=5 field's encode
took on an M2 (2026-09-23). The same work is a handful of comparisons per class in a thread.

What it returns is what ``encode._select_and_order`` returns, bit for bit: the class indices
in stored order, their true gaps (copied, never recomputed) and whether each is kept. It makes
the same decisions from the same numbers:

* the key is ``min(gap, gap_range)``, minus ``big`` for a class that wins at the voxel or at
  one of its 26 neighbors (one IEEE add, the torch path's ``key.add_(sh * -big)``), and the
  winner comes first whatever its key - a flag here, where the torch path writes ``-inf``;
* the N smallest keys are kept, the lowest class index winning a tie at the cut (``_select``);
* ``kept`` is the winner, a shell class, or a gap under ``clip`` - everything when exhaustive;
* the kept are ordered by (true gap, class index), the dropped after them by class index -
  the torch path's stable sort with the dropped at ``inf``, then ``settle_ties``.

Only comparisons, ``min`` and that one add run here; ``exp``, the gap byte and the tail stay
on torch, so no transcendental has to agree between two implementations. Compiled with fp
contraction off, as the restore kernel is.
"""
from __future__ import annotations

from .._torch import have_torch, no_grad, torch

NMAX = 16                   # kept planes a thread holds in registers; deeper encodes use torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
{PRAGMA}

// (key, class) of a SELECTED entry, the winner first: a < b in the torch path's cut order
inline bool key_before(float ka, int ca, bool wa, float kb, int cb, bool wb) {
    if (wa != wb) return wa;
    return ka < kb || (ka == kb && ca < cb);
}

kernel void rk_encode_select(
    device const float* gaps    [[buffer(0)]],   // (K, V) true gaps, >= 0
    device const long*  around  [[buffer(1)]],   // (zs + 2, Y + 2, X + 2) winners with a halo
    device const long*  win     [[buffer(2)]],   // (V) the winner, as torch's max chose it
    device const int*   iparams [[buffer(3)]],   // K, N, zs, Y, X, exhaustive
    device const float* fparams [[buffer(4)]],   // gap_range, big, clip
    device long*        idx_out [[buffer(5)]],   // (N, V) class index, stored order
    device float*       g_out   [[buffer(6)]],   // (N, V) its true gap
    device uchar*       kept_out [[buffer(7)]],  // (N, V), read by torch as its 1-byte bool
    uint v [[thread_position_in_grid]])
{
    const int K = iparams[0], N = iparams[1], zs = iparams[2], Y = iparams[3], X = iparams[4];
    const bool exhaustive = iparams[5] != 0;
    const uint V = (uint)zs * (uint)Y * (uint)X;
    if (v >= V) return;
    const float gap_range = fparams[0], big = fparams[1], clip = fparams[2];
    const int x = (int)(v % (uint)X);
    const int y = (int)((v / (uint)X) % (uint)Y);
    const int z = (int)(v / ((uint)X * (uint)Y));

    // the classes that win at the voxel or a neighbor, each once
    int shell[27];
    int n_shell = 0;
    for (int dz = 0; dz < 3; dz++)
        for (int dy = 0; dy < 3; dy++)
            for (int dx = 0; dx < 3; dx++) {
                const int c = (int)around[((long)(z + dz) * (Y + 2) + (y + dy)) * (X + 2) + (x + dx)];
                bool seen = false;
                for (int t = 0; t < n_shell; t++) if (shell[t] == c) { seen = true; break; }
                if (!seen) shell[n_shell++] = c;
            }

    // the N smallest (key, class), the winner first: an insertion walk in class order, so an
    // equal key never displaces a lower class - the lowest indices win the tie at the cut
    const int w = (int)win[v];
    float sk[{NMAX}];
    int sc[{NMAX}];
    bool sw[{NMAX}];
    int n = 0;
    for (int c = 0; c < K; c++) {
        const bool is_w = (c == w);
        float k = 0.0f;
        if (!is_w) {
            k = min(gaps[(long)c * V + v], gap_range);
            for (int t = 0; t < n_shell; t++) if (shell[t] == c) { k = k + (-big); break; }
        }
        int pos;
        if (n < N) {
            pos = n++;
        } else if (key_before(k, c, is_w, sk[N - 1], sc[N - 1], sw[N - 1])) {
            pos = N - 1;
        } else {
            continue;
        }
        while (pos > 0 && key_before(k, c, is_w, sk[pos - 1], sc[pos - 1], sw[pos - 1])) {
            sk[pos] = sk[pos - 1]; sc[pos] = sc[pos - 1]; sw[pos] = sw[pos - 1];
            pos--;
        }
        sk[pos] = k; sc[pos] = c; sw[pos] = is_w;
    }

    // kept: position 0 of the cut order (the winner), a shell class, a gap under the clip
    float g[{NMAX}];
    bool kp[{NMAX}];
    for (int j = 0; j < N; j++) {
        const int c = sc[j];
        g[j] = gaps[(long)c * V + v];
        bool in_shell = false;
        for (int t = 0; t < n_shell; t++) if (shell[t] == c) { in_shell = true; break; }
        kp[j] = exhaustive || j == 0 || in_shell || g[j] < clip;
    }

    // stored order: the kept by (true gap, class), then the dropped by class
    int ord[{NMAX}];
    for (int j = 0; j < N; j++) {
        int pos = j;
        while (pos > 0) {
            const int a = ord[pos - 1];
            bool before;                             // does entry j go ahead of entry a?
            if (kp[j] != kp[a]) before = kp[j];
            else if (kp[j]) before = g[j] < g[a] || (g[j] == g[a] && sc[j] < sc[a]);
            else before = sc[j] < sc[a];
            if (!before) break;
            ord[pos] = a;
            pos--;
        }
        ord[pos] = j;
    }
    for (int j = 0; j < N; j++) {
        const int s = ord[j];
        const long o = (long)j * V + v;
        idx_out[o] = (long)sc[s];
        g_out[o] = g[s];
        kept_out[o] = kp[s] ? (uchar)1 : (uchar)0;
    }
}
"""

_LIB = None
_FP_CONTRACT: str | None = None


def available() -> bool:
    return bool(have_torch() and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                and hasattr(torch.mps, "compile_shader"))


def source(fp_contract_off: bool = True) -> str:
    pragma = "#pragma clang fp contract(off)" if fp_contract_off else ""
    return _SOURCE.replace("{PRAGMA}", pragma).replace("{NMAX}", str(NMAX))


def library():
    """Compile once, fp contraction off (the key's one add must round as torch's does)."""
    global _LIB, _FP_CONTRACT
    if _LIB is None:
        if not available():
            raise RuntimeError("rankfield.backends.metal_encode: needs an MPS device and "
                               "torch.mps.compile_shader")
        try:
            _LIB = torch.mps.compile_shader(source(True))
            _FP_CONTRACT = "off"
        except Exception:
            _LIB = torch.mps.compile_shader(source(False))
            _FP_CONTRACT = "default"
    return _LIB


@no_grad
def select(gaps: torch.Tensor, around: torch.Tensor, win: torch.Tensor, *, N: int,
           exhaustive: bool, gap_range: float, big: float, clip: float, group_size: int = 256):
    """``gaps`` (K, zs, Y, X) float32, ``around`` (zs + 2, Y + 2, X + 2) and ``win`` (zs, Y, X)
    int64, all on 'mps' -> ``(idx, g_sel, kept)``, each (N, zs, Y, X): int64, float32, bool."""
    if not (1 <= N <= NMAX):
        raise ValueError(f"rankfield.backends.metal_encode: the kernel holds 1 to {NMAX} planes; got {N}")
    K, zs, Y, X = (int(v) for v in gaps.shape)
    if tuple(around.shape) != (zs + 2, Y + 2, X + 2) or tuple(win.shape) != (zs, Y, X):
        raise ValueError("rankfield.backends.metal_encode: the winner maps do not match the gaps")
    if gaps.dtype != torch.float32 or around.dtype != torch.int64 or win.dtype != torch.int64:
        raise TypeError("rankfield.backends.metal_encode: gaps float32, winner maps int64")
    V = zs * Y * X
    if K * V >= 2 ** 62 or V >= 2 ** 32:
        raise ValueError("rankfield.backends.metal_encode: a slab past 2^32 voxels")
    dev = gaps.device
    gaps, around, win = gaps.contiguous(), around.contiguous(), win.contiguous()
    idx = torch.empty((N, zs, Y, X), dtype=torch.int64, device=dev)
    g_sel = torch.empty((N, zs, Y, X), dtype=torch.float32, device=dev)
    kept = torch.empty((N, zs, Y, X), dtype=torch.bool, device=dev)
    iparams = torch.tensor([K, N, zs, Y, X, int(bool(exhaustive))], dtype=torch.int32, device=dev)
    fparams = torch.tensor([float(gap_range), float(big), float(clip)], dtype=torch.float32, device=dev)
    library().rk_encode_select(gaps, around, win, iparams, fparams, idx, g_sel, kept,
                               threads=V, group_size=group_size)
    return idx, g_sel, kept
