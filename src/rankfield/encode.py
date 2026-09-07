"""The encoder: logits -> :class:`RankField`, in slabs on the logits' own device.

What is kept per voxel (format 0.3, the ``shell`` rule): every class that wins at the
voxel or at any of its 26 neighbours, with its true gap - so any class that wins at one
corner of an interpolation stencil is present, with its gap, at every other corner, and a
restore never scores it at the floor. Then the closest non-winners within ``clip``. Kept
entries are ordered by gap, so ``ranks[1]`` is the true runner-up, and dropped entries are
the sentinel and come last (a reader may stop at the first one).

Why: a class dropped by the clip used to be floored at ``-clip`` by the restore, an
UPPER bound on its deficit, which over-credited it exactly where it mattered - thin
structures grew (a 12th rib by 19 %). Keeping the shell removes that bias at the cost of
6 % more bytes on a whole-body case; measured in the format document.

The gap byte follows the meta's curve (``log`` over ``gap_range``): a shell class can be
30 logits behind at the far corner and still be recorded, while the quantum near zero,
where a boundary is decided, stays at a few thousandths of a logit.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .code import (CLIP, DEFAULT_DEPTH, FORMAT_VERSION, GAP_ORIGIN, GAP_RANGE, GAP_UNIT,
                   SUPPORT_MAX, TAIL_MAX, ZERO_LEVEL, RankField, byte_of_gap, rank_dtype)


def _check(logits) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor; got {type(logits).__name__}")
    if logits.ndim != 4:
        raise ValueError(f"logits must be (K, Z, Y, X); got shape {tuple(logits.shape)}")
    if not logits.dtype.is_floating_point:
        raise TypeError(f"logits must be floating point; got {logits.dtype}")
    return logits


def settle_ties(top: torch.Tensor, idx: torch.Tensor, N: int) -> None:
    """Order equal-valued entries of a sort by ascending class index, in place.

    ``topk`` leaves the order among EQUAL values undefined, and CUDA's is not the CPU's, so
    without this the stored bytes depend on where they were encoded. Lower index wins, which
    is argmax's own convention, so ``ranks[0]`` remains exactly the argmax every labelmap was
    made with. Bubble, because a tie can run longer than a pair; ``N`` is small.

    This orders what was selected; WHICH is selected is ``_select``'s, which takes the lowest
    indices among equal keys at the depth cut - so a tie straddling the boundary is resolved
    the same way on every device too (``topk`` alone left that to the backend: 216 of 693
    voxels of a tie-heavy field differed between cpu and mps).
    """
    for _ in range(N - 1):
        for j in range(N - 1):
            swap = (top[j] == top[j + 1]) & (idx[j] > idx[j + 1])
            lo = torch.where(swap, idx[j + 1], idx[j])
            hi = torch.where(swap, idx[j], idx[j + 1])
            idx[j], idx[j + 1] = lo, hi


def _shell(win_ext: torch.Tensor, K: int, halo_lo: int, zs: int) -> torch.Tensor:
    """``(K, zs, Y, X)`` bool: class c wins at the voxel or at one of its 26 neighbours.

    ``win_ext`` is the winner map of the slab with one plane of halo on each side where the
    volume has one (``halo_lo`` says whether the first plane is halo). Edges replicate."""
    Ze, Y, X = win_ext.shape
    pad = torch.nn.functional.pad(win_ext[None, None].to(torch.float32), (1, 1, 1, 1, 0, 0),
                                  mode="replicate")[0, 0].to(torch.int64)      # y/x replicate
    zpad = torch.cat([pad[:1], pad, pad[-1:]], 0)                             # z replicate
    z0 = 1 + halo_lo                                                          # slab start in zpad
    shell = torch.zeros((K, zs, Y, X), dtype=torch.bool, device=win_ext.device)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = zpad[z0 + dz:z0 + dz + zs, 1 + dy:1 + dy + Y, 1 + dx:1 + dx + X]
                shell.scatter_(0, w[None], True)
    return shell


def _select(key: torch.Tensor, N: int) -> torch.Tensor:
    """Indices (N, ...) of the N smallest keys along dim 0, choosing the LOWEST class index
    among equal keys at the cut. ``torch.topk`` picks arbitrarily among ties and differently
    per device, which made the same logits encode to different ranks on cpu and mps
    (``settle_ties`` orders what was selected; it cannot repair what was selected)."""
    # Memory: ``key`` is (K, slab) float32, a gigabyte on a whole-body part, and this runs
    # on a 16 GB laptop's GPU beside the network - so no K-sized temporaries beyond two bool
    # masks, and the selection is written back into ``key`` (the caller's throwaway copy).
    cut = torch.topk(-key, N, dim=0).values[N - 1:N].neg()     # the N-th smallest key
    chosen = key < cut                                          # always in
    at = key == cut                                             # the tie at the cut
    room = (N - chosen.sum(0, dtype=torch.int32)).to(torch.int16)
    seen = torch.zeros_like(room)
    for k in range(key.shape[0]):                               # walk the tie in class order
        seen += at[k]
        chosen[k] |= at[k] & (seen <= room)                     # its lowest indices
    del at
    key.masked_fill_(~chosen, float("inf"))
    return torch.topk(-key, N, dim=0).indices                   # exactly the chosen N; any order


def _ordered_sum(w: torch.Tensor) -> torch.Tensor:
    """Sum over dim 0 in index order, one elementwise add per plane: the same IEEE operations
    in the same order on every device."""
    acc = w[0].clone()
    for k in range(1, w.shape[0]):
        acc = acc + w[k]
    return acc


def encode(logits: torch.Tensor, *, depth: int = DEFAULT_DEPTH, clip: float = CLIP,
           keep: str = "shell", curve: str = "log", gap_range: float = GAP_RANGE,
           gap_origin: float = GAP_ORIGIN, slab: int | None = None,
           with_tail: bool = True,
           tail_temperatures: Sequence[float] = (1.0,)) -> RankField:
    """``(K, Z, Y, X)`` logits -> :class:`RankField`.

    ``keep`` is ``"shell"`` (format 0.3) or ``"clip"`` (0.2's rule: within the clip only);
    ``curve`` ``"log"`` or ``"uniform"``; with ``keep="clip"``, ``curve="uniform"`` and
    ``gap_range=clip`` the bytes are format 0.2's. ``slab`` bounds peak memory (planes of
    the slab are promoted to fp32, and the shell mask is K bytes per voxel of the slab).
    ``tail_temperatures`` are the softmax temperatures the dropped mass is measured at, each
    positive; the default ``(1.0,)`` writes the single ``tail`` plane of format 0.3, and any
    other temperature is written to ``tails`` beside it.
    """
    lg_all = _check(logits)
    K = int(lg_all.shape[0])
    N = max(1, min(int(depth), K))
    Z, Y, X = (int(v) for v in lg_all.shape[1:])
    if keep not in ("shell", "clip"):
        raise ValueError(f"keep must be 'shell' or 'clip'; got {keep!r}")
    if keep == "clip" and curve == "uniform" and float(gap_range) != float(clip):
        raise ValueError("with keep='clip' the uniform range is the clip")
    exhaustive = N >= K
    rdt = rank_dtype(K)
    meta = {"version": FORMAT_VERSION, "mode": "ranked", "classes": K, "depth": N,
            "clip": float(clip), "gap_unit": GAP_UNIT, "gap_curve": curve,
            "gap_range": float(gap_range), "gap_origin": float(gap_origin), "keep": keep,
            "rank_dtype": np.dtype(rdt).name, "support_max": SUPPORT_MAX, "rank_sentinel": 0,
            "exhaustive": bool(exhaustive), "shape": [Z, Y, X]}
    if slab is None:                                  # ~256 MB for the shell mask
        slab = max(1, min(Z, (1 << 28) // max(1, K * Y * X)))
    ranks = np.empty((N, Z, Y, X), rdt)
    support = np.empty((N - 1, Z, Y, X), np.uint8) if N > 1 else np.empty((0, Z, Y, X), np.uint8)
    temps = [float(t) for t in tail_temperatures]
    if any(not t > 0 for t in temps):
        raise ValueError(f"tail_temperatures must be positive, got {temps}")
    tail = None if (exhaustive or not with_tail or 1.0 not in temps) else np.empty((Z, Y, X), np.uint16)
    # format 0.4: the dropped mass at OTHER temperatures too - what a distillation at T
    # needs and cannot recover from the kept classes (at T=4 a torso store drops 3.4 % of
    # the mass on average, a third of its voxels over 1 %)
    tails = ({} if (exhaustive or not with_tail) else
             {t: np.empty((Z, Y, X), np.uint16) for t in temps if t != 1.0})
    max_tail = 0.0
    max_tails = {t: 0.0 for t in tails}
    big = 1e6
    for z0 in range(0, Z, slab):
        z1 = min(z0 + slab, Z)
        lg = lg_all[:, z0:z1].float()
        top0 = lg.max(0, keepdim=True).values
        gaps = top0 - lg                                        # (K, zs, Y, X) >= 0
        if keep == "shell":
            lo, hi = max(0, z0 - 1), min(Z, z1 + 1)
            win_ext = lg_all[:, lo:hi].float().argmax(0)
            sh = _shell(win_ext, K, z0 - lo, z1 - z0)
            key = torch.where(sh, gaps - big, gaps)
        else:
            sh = None
            key = gaps
        idx = _select(key, N)                                     # the N smallest keys, by index on ties
        ktop = torch.gather(key, 0, idx)                          # (key's unchosen entries are now inf)
        settle_ties(ktop, idx, N)
        g_sel = torch.gather(gaps, 0, idx)                        # true gaps of the kept
        sh_sel = torch.gather(sh, 0, idx) if sh is not None else torch.zeros_like(g_sel, dtype=torch.bool)
        kept = sh_sel | (g_sel < clip)
        kept[0] = True                                            # the winner, always
        # order the kept by true gap so ranks[1] is the runner-up; the dropped go last
        order_key = torch.where(kept, g_sel, torch.full_like(g_sel, big))
        order_key, perm = torch.sort(order_key, dim=0, stable=True)
        idx = torch.gather(idx, 0, perm)
        g_sel = torch.gather(g_sel, 0, perm)
        sh_sel = torch.gather(sh_sel, 0, perm)
        kept = torch.gather(kept, 0, perm)
        settle_ties(order_key, idx, N)
        r = (idx + 1).to(torch.int32)
        if N > 1:
            sup = byte_of_gap(g_sel[1:], meta).round()
            # a kept class stays a class even past the range (byte 1); a dropped one is
            # the sentinel in both arrays
            sup = torch.where(kept[1:], sup.clamp(min=1.0), torch.zeros_like(sup))
            r[1:][~kept[1:]] = 0
            support[:, z0:z1] = sup.to(torch.uint8).cpu().numpy()
        ranks[:, z0:z1] = r.cpu().numpy().astype(rdt, copy=False)
        if tail is not None:
            # summed in class order, one add at a time: a device's own reduction order would
            # move the rounded byte by one between machines, and a store's bytes must not
            # depend on where they were written
            z_full = _ordered_sum(torch.exp(-gaps))
            kept_mass = torch.exp(-g_sel)
            kept_mass[~kept] = 0.0
            t = ((z_full - _ordered_sum(kept_mass)) / z_full).clamp(0, 1)
            max_tail = max(max_tail, float(t.max()))
            tail[z0:z1] = (t * TAIL_MAX).round().cpu().numpy().astype(np.uint16)
        for temp, arr in tails.items():
            z_t = _ordered_sum(torch.exp(-gaps / temp))
            kept_t = torch.exp(-g_sel / temp)
            kept_t[~kept] = 0.0
            t = ((z_t - _ordered_sum(kept_t)) / z_t).clamp(0, 1)
            max_tails[temp] = max(max_tails[temp], float(t.max()))
            arr[z0:z1] = (t * TAIL_MAX).round().cpu().numpy().astype(np.uint16)
        del lg, gaps, key, ktop, idx, g_sel, sh, sh_sel, kept
    # what was dropped: 0 when nothing could be, the measured maximum when the tail was
    # written, and unknown - not zero - when it was not
    meta["max_tail"] = 0.0 if exhaustive else (max_tail if tail is not None else None)
    meta["tail_max"] = None if (tail is None and not tails) else TAIL_MAX
    stored = ([1.0] if tail is not None else []) + sorted(tails)
    meta["tail_temperatures"] = stored
    meta["max_tail_at_temperature"] = [(0.0 if exhaustive else (max_tail if t == 1.0 else max_tails[t]))
                                       for t in stored]
    return RankField(ranks=ranks, support=support, tail=tail, meta=meta, tails=tails or None)


def encode_regions(logits: torch.Tensor, *, clip: float = CLIP, threshold: float = 0.0,
                   slab: int = 32) -> RankField:
    """Sigmoid-head logits -> one signed margin plane per region (no ranks, no tail).

    The logit is already the margin, referenced to the decision threshold and quantized
    uniformly: 1 = clip below the boundary, ``ZERO_LEVEL`` exactly on it, 255 = clip above;
    0 stays the fill sentinel.
    """
    lg_all = _check(logits)
    K = int(lg_all.shape[0])
    Z, Y, X = (int(v) for v in lg_all.shape[1:])
    support = np.empty((K, Z, Y, X), np.uint8)
    for z0 in range(0, Z, max(1, int(slab))):
        z1 = min(z0 + max(1, int(slab)), Z)
        m = lg_all[:, z0:z1].float() - float(threshold)
        q = ((m / clip).clamp(-1, 1) * (SUPPORT_MAX - ZERO_LEVEL) + ZERO_LEVEL).round()
        support[:, z0:z1] = q.clamp(1, SUPPORT_MAX).to(torch.uint8).cpu().numpy()
        del m, q
    return RankField(ranks=None, support=support, tail=None, meta={
        "version": FORMAT_VERSION, "mode": "regions", "classes": K, "depth": K,
        "clip": float(clip), "gap_unit": GAP_UNIT, "threshold": float(threshold),
        "support_max": SUPPORT_MAX, "signed_support": True, "support_zero": ZERO_LEVEL,
        "exhaustive": True, "max_tail": 0.0, "tail_max": None, "shape": [Z, Y, X]})
