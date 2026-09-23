"""The encoder: logits -> :class:`RankField`, in slabs on the logits' own device.

What is kept per voxel (format 0.3, the ``shell`` rule): every class that wins at the
voxel or at any of its 26 neighbors, with its true gap; then the closest non-winners
within ``clip``; until the planes are full. Kept entries are ordered by gap, and dropped
entries are the sentinel and come last (a reader may stop at the first one).

Why: a class dropped by the clip is floored at ``-clip`` by the restore, an UPPER bound on
its deficit, which over-credits it exactly where it matters - thin structures grew (a 12th
rib by 19 %). Carrying the neighbors' winners narrows that, because a class that wins at
one corner of an interpolation stencil is then present, with its gap, at every corner the
depth has room for.

Narrows, not removes. Three things the rule above does not promise, each pinned by a test
in ``tests/test_review_fixes.py`` and measured in the format document under "What the keep
rule does not promise":

* ``ranks[1]`` is the nearest KEPT class, not always the true runner-up: shell classes take
  the planes first, so a closer non-winner can be evicted by one that wins at a neighbor.
* A 27-voxel neighborhood can hold more winners than ``depth`` planes. The class dropped
  then is the farthest shell class - never the winner, which the depth cut keeps.
* A class kept at one corner and dropped at another still reads at the floor there, so it
  can still win an interpolation it should lose.

Depth is the knob that closes all three, and the format document carries the curve.

The gap byte follows the meta's curve (``log`` over ``gap_range``): a shell class can be
30 logits behind at the far corner and still be recorded, while the quantum near zero,
where a boundary is decided, stays at a few thousandths of a logit.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._torch import torch
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
    """``(K, zs, Y, X)`` bool: class c wins at the voxel or at one of its 26 neighbors.

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
    # Memory: ``key`` is (K, slab) float32, the largest plane the encoder holds, and this runs
    # on a 16 GB laptop's GPU beside the network - so no K-sized temporaries beyond topk's
    # negated key and two bool masks, and the selection is written back into ``key`` (the
    # caller's throwaway copy). slab_bytes counts all of them.
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


# What a slab costs at the encoder's peak, in bytes: a*K + b*N + c*T + d per voxel of the
# slab, plus e per voxel of the padded winner maps the shell is built from.
# Counted, not guessed: the tensors the encoder's ops return, alive at once, over 240
# configurations (K 2..118, depth 1..K, 1 or 3 temperatures, both keep rules, with and
# without a tail, fp16 and fp32 input) and planes down to 1x1 - tests/test_encode.py
# re-counts a sample of them. What the count cannot see: scratch a kernel allocates inside
# itself, and on a GPU the per-slab copies to the host, so the budget is working memory as
# the ops define it, not a promise about a device pool. What it is:
#   K: gaps, key, the shell mask and exp's input and output are alive together when the
#      partition function is taken - 17 bytes a class (4 + 4 + 1 + 4 + 4)
#   N: the selection's int64 indices, its sort and gathers - 24 to 46 bytes a kept plane
#   T: a partition function per tail temperature, and its temporaries
#   e: _shell's winner maps with their halo planes and replicate padding (int64 and fp32,
#      (planes + 4) x (Y + 2) x (X + 2)) - negligible on a real plane, dominant on a tiny one
# The old rule budgeted ONE byte a class (the shell mask), so at small K the slab became
# the whole volume: a 40-Mvoxel K=5 field wanted ~10 GB of MPS pool in one slab.
SLAB_BYTES_PER_CLASS = 18
SLAB_BYTES_PER_PLANE = 48
SLAB_BYTES_PER_TEMPERATURE = 16
SLAB_BYTES_FIXED = 64
SLAB_BYTES_PER_PADDED_VOXEL = 32
# A small-footprint default, not a property of any machine: the output is identical at any
# slab, so a caller that has measured its device's budget passes it as ``memory_budget``.
DEFAULT_MEMORY_BUDGET = 1 << 30


def slab_bytes(planes: int, rows: int, columns: int, classes: int, depth: int,
               temperatures: int = 1) -> int:
    """Upper bound on the encoder's peak working memory, in bytes, for a slab of ``planes`` z
    planes of ``rows`` x ``columns``, with ``classes`` logits kept to ``depth`` planes and
    ``temperatures`` tail planes."""
    per_voxel = (SLAB_BYTES_PER_CLASS * int(classes) + SLAB_BYTES_PER_PLANE * int(depth)
                 + SLAB_BYTES_PER_TEMPERATURE * int(temperatures) + SLAB_BYTES_FIXED)
    padded = (int(planes) + 4) * (int(rows) + 2) * (int(columns) + 2)
    return per_voxel * int(planes) * int(rows) * int(columns) + SLAB_BYTES_PER_PADDED_VOXEL * padded


def choose_slab(shape, classes: int, depth: int, temperatures: int = 1,
                memory_budget: int = DEFAULT_MEMORY_BUDGET) -> int:
    """The most z planes per slab whose :func:`slab_bytes` fits ``memory_budget``; at least 1
    (a plane is the smallest unit a slab has - a budget under one plane's cost is exceeded,
    not refused) and at most the volume."""
    Z, Y, X = (int(v) for v in shape)
    return _most_planes(Z, lambda planes: slab_bytes(planes, Y, X, classes, depth, temperatures),
                        memory_budget)


# encode_regions' peak, per voxel of the slab and class: the fp32 margin plane, which every
# step rewrites in place, and the uint8 plane it is converted to - 4 + 1, with no fixed term
# (no halo, no padding). Counted the same way as SLAB_BYTES_*, over K 1..118, planes 1x1 to
# 64x64, slabs of 1 and 3, fp32/fp16/bf16 input: exactly 5.0 bytes every time. Written out
# of place (m, then a temporary for each of the division, clamp, scale, offset, round and
# clamp) the same bytes cost 13. The count's blind spots are encode's: on a GPU, the per-slab
# host copy of the uint8 plane is outside it.
REGION_BYTES_PER_CLASS = 5


def region_slab_bytes(planes: int, rows: int, columns: int, classes: int) -> int:
    """Upper bound on :func:`encode_regions`' peak working memory, in bytes, for a slab of
    ``planes`` z planes of ``rows`` x ``columns`` with ``classes`` region logits."""
    return REGION_BYTES_PER_CLASS * int(classes) * int(planes) * int(rows) * int(columns)


def choose_region_slab(shape, classes: int, memory_budget: int = DEFAULT_MEMORY_BUDGET) -> int:
    """The most z planes per :func:`encode_regions` slab whose :func:`region_slab_bytes` fits
    ``memory_budget``; at least 1 and at most the volume, as :func:`choose_slab`."""
    Z, Y, X = (int(v) for v in shape)
    return _most_planes(Z, lambda planes: region_slab_bytes(planes, Y, X, classes), memory_budget)


def _most_planes(Z: int, cost, memory_budget: int) -> int:
    """The most planes, 1..Z, whose ``cost(planes)`` - affine in the planes - fits the budget."""
    fixed = cost(0)
    per_plane = cost(1) - fixed
    return max(1, min(Z, (int(memory_budget) - fixed) // max(1, per_plane)))


def _slab_and_budget(slab, memory_budget, choose) -> int:
    """A given ``slab`` as a positive int (the budget not consulted), else ``choose(budget)``."""
    if memory_budget is not None and not int(memory_budget) > 0:
        raise ValueError(f"memory_budget must be a positive number of bytes, got {memory_budget}")
    if slab is None:
        return int(choose(DEFAULT_MEMORY_BUDGET if memory_budget is None else memory_budget))
    if int(slab) != slab or int(slab) < 1:
        # 0 used to die in range(); a negative slab ran no slab at all and returned the
        # uninitialized np.empty planes as a field
        raise ValueError(f"slab must be a positive integer number of z planes, got {slab!r}")
    return int(slab)


def encode(logits: torch.Tensor, *, depth: int = DEFAULT_DEPTH, clip: float = CLIP,
           keep: str = "shell", curve: str = "log", gap_range: float = GAP_RANGE,
           gap_origin: float = GAP_ORIGIN, slab: int | None = None,
           memory_budget: int | None = None, with_tail: bool = True,
           tail_temperatures: Sequence[float] = (1.0,)) -> RankField:
    """``(K, Z, Y, X)`` logits -> :class:`RankField`.

    ``keep`` is ``"shell"`` (format 0.3) or ``"clip"`` (0.2's rule: within the clip only);
    ``curve`` ``"log"`` or ``"uniform"``; with ``keep="clip"``, ``curve="uniform"`` and
    ``gap_range=clip`` the bytes are format 0.2's. ``slab`` is the z planes encoded at once,
    a positive integer; left as None it is sized so the working set (:func:`slab_bytes`)
    stays within ``memory_budget`` bytes - 1 GiB by default, a small footprint rather than a
    measurement, so a caller that knows its device's budget should pass it. A given ``slab``
    is taken as is and the budget is not consulted. Neither changes a byte of the output.
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
    n_tails = (tail is not None) + len(tails)
    slab = _slab_and_budget(slab, memory_budget,
                            lambda budget: choose_slab((Z, Y, X), K, N, n_tails, budget))
    # The offset that puts every shell class under every other class. A CONSTANT of the
    # format, not a property of the data: the key's gaps are clamped at the range first, so
    # nothing beyond it can inflate the offset. Two earlier offsets could not do this. 1e6
    # sits where float32 steps by 0.0625, and the logits' own range let one outlying voxel
    # push the offset back up there - either way gaps under the step collapsed into a tie
    # that the depth cut broke by class index, and the winner itself could be the entry it
    # dropped. Here the step is ~8e-6 logits whatever the data, and the winner is pinned
    # below every key by -inf, so no step can reach it.
    big = float(gap_range) + 1.0
    for z0 in range(0, Z, slab):
        z1 = min(z0 + slab, Z)
        lg = lg_all[:, z0:z1].float()
        top0, win = lg.max(0, keepdim=True)     # the winner's index comes free with its value
        gaps = top0 - lg                                        # (K, zs, Y, X) >= 0
        del lg, top0            # (K, slab) planes are the peak: each goes as soon as it is read
        if keep == "shell":
            lo, hi = max(0, z0 - 1), min(Z, z1 + 1)
            win_ext = lg_all[:, lo:hi].float().argmax(0)
            sh = _shell(win_ext, K, z0 - lo, z1 - z0)
            # Two (K, slab) planes live at the peak, which is what `torch.where(sh, gaps -
            # big, gaps)` cost before. `key[sh] -= big` would read better and cost far more:
            # a boolean mask materializes int64 indices, three bytes of temporary per byte
            # of key.
            key = gaps.clamp(max=float(gap_range))
            key.add_(sh * -big)                             # shell first, by clamped gap
            key.scatter_(0, win, float("-inf"))             # the winner, exactly, at any scale
            del win_ext
        else:
            sh = None
            key = gaps          # no offset, so the winner's gap of 0 is always among the N
        # Before the cut: _select fills the unchosen with inf IN PLACE, and with keep="clip"
        # the key is `gaps` itself (the shell branch copies it, that branch has nothing to
        # copy), so a partition function read afterwards would score every dropped class as
        # zero mass and report a tail of nothing.
        z_full = _ordered_sum(torch.exp(-gaps)) if tail is not None else None
        z_temps = {t: _ordered_sum(torch.exp(-gaps / t)) for t in tails}
        idx = _select(key, N)                                     # the N smallest keys, by index on ties
        ktop = torch.gather(key, 0, idx)                          # (key's unchosen entries are now inf)
        del key
        settle_ties(ktop, idx, N)
        del ktop
        g_sel = torch.gather(gaps, 0, idx)                        # true gaps of the kept
        sh_sel = torch.gather(sh, 0, idx) if sh is not None else torch.zeros_like(g_sel, dtype=torch.bool)
        del gaps, sh, win       # from here on the slab is (N, slab): nothing K-sized is left
        # With a plane per class there is nothing to gain by dropping one: the clip would
        # leave a class the field has room for at the -clip floor and cost a tail to
        # describe. `exhaustive` says nothing was dropped, so nothing is.
        kept = (torch.ones_like(g_sel, dtype=torch.bool) if exhaustive
                else sh_sel | (g_sel < clip))
        kept[0] = True                                            # the winner, always
        # order the kept by true gap so ranks[1] is the runner-up; the dropped go last
        # inf, not the offset: a KEPT shell class can trail by more than the range, and a
        # dropped entry has to sort behind it for the sentinels to stay a suffix
        order_key = torch.where(kept, g_sel, torch.full_like(g_sel, float("inf")))
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
            # Summed in class order, one add at a time: a device's own reduction order would
            # move the rounded value between machines. That buys the RANK and SUPPORT planes
            # exactly - they are decided by comparisons, which are exact even when the values
            # are not - but it cannot buy the tail, which quantizes a float that came out of
            # exp(). IEEE 754 requires correct rounding for + - * / and sqrt and nothing else;
            # CUDA documents expf at 2 ULP, on the multi-function unit's hardware exp2. So the
            # uint16 tail can land one unit either side of a rounding boundary - 1 voxel in
            # 594 on a tie-heavy field, unchanged since 0.2.1, and 1/65535 is far under the
            # gap byte's own quantum. docs/format.md, "The restore", carries the citations.
            kept_mass = torch.exp(-g_sel)
            kept_mass[~kept] = 0.0
            t = ((z_full - _ordered_sum(kept_mass)) / z_full).clamp(0, 1)
            max_tail = max(max_tail, float(t.max()))
            tail[z0:z1] = (t * TAIL_MAX).round().cpu().numpy().astype(np.uint16)
        for temp, arr in tails.items():
            z_t = z_temps[temp]
            kept_t = torch.exp(-g_sel / temp)
            kept_t[~kept] = 0.0
            t = ((z_t - _ordered_sum(kept_t)) / z_t).clamp(0, 1)
            max_tails[temp] = max(max_tails[temp], float(t.max()))
            arr[z0:z1] = (t * TAIL_MAX).round().cpu().numpy().astype(np.uint16)
        del idx, g_sel, sh_sel, kept, z_full, z_temps
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
                   slab: int | None = None, memory_budget: int | None = None) -> RankField:
    """Sigmoid-head logits -> one signed margin plane per region (no ranks, no tail).

    The logit is already the margin, referenced to the decision threshold and quantized
    uniformly: 1 = clip below the boundary, ``ZERO_LEVEL`` exactly on it, 255 = clip above;
    0 stays the fill sentinel. ``slab`` and ``memory_budget`` are :func:`encode`'s: left as
    None the slab is the most z planes whose :func:`region_slab_bytes` fits the budget (1 GiB
    by default), a given ``slab`` is a positive integer taken as is, and neither changes a
    byte of the output.
    """
    lg_all = _check(logits)
    K = int(lg_all.shape[0])
    Z, Y, X = (int(v) for v in lg_all.shape[1:])
    slab = _slab_and_budget(slab, memory_budget,
                            lambda budget: choose_region_slab((Z, Y, X), K, budget))
    support = np.empty((K, Z, Y, X), np.uint8)
    for z0 in range(0, Z, slab):
        z1 = min(z0 + slab, Z)
        # One fp32 plane rewritten in place, then the uint8 one (region_slab_bytes). A copy
        # even from fp32: .float() would hand back the caller's logits to be written over.
        m = lg_all[:, z0:z1].to(torch.float32, copy=True)
        m.sub_(float(threshold)).div_(clip).clamp_(-1, 1)
        m.mul_(SUPPORT_MAX - ZERO_LEVEL).add_(ZERO_LEVEL).round_().clamp_(1, SUPPORT_MAX)
        support[:, z0:z1] = m.to(torch.uint8).cpu().numpy()
        del m
    return RankField(ranks=None, support=support, tail=None, meta={
        "version": FORMAT_VERSION, "mode": "regions", "classes": K, "depth": K,
        "clip": float(clip), "gap_unit": GAP_UNIT, "threshold": float(threshold),
        "support_max": SUPPORT_MAX, "signed_support": True, "support_zero": ZERO_LEVEL,
        "exhaustive": True, "max_tail": 0.0, "tail_max": None, "shape": [Z, Y, X]})
