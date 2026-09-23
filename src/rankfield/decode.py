"""The decoders: planes -> fields. Every byte -> gap goes through :func:`levels`."""
from __future__ import annotations

import sys

import numpy as np

from ._torch import torch
from .code import SUPPORT_MAX, TAIL_MAX, ZERO_LEVEL, RankField, levels


def _host(code: RankField, who: str) -> None:
    """Refuse a code whose planes are torch tensors. Anything else that indexes like numpy
    is read: an ndarray, or the lazy zarr arrays :func:`rankfield.store.read_parts` hands
    back, which the decoders read a plane at a time (:func:`_plane`) rather than whole.

    Never imports torch - these decoders must run without it - so a tensor can only be
    one if torch was already imported by whoever made it."""
    torch_mod = sys.modules.get("torch")
    for name in ("ranks", "support", "tail"):
        a = getattr(code, name)
        if a is None:
            continue
        if torch_mod is not None and isinstance(a, torch_mod.Tensor):
            raise TypeError(f"{who}() decodes numpy (or zarr) planes, but this code's {name} is a "
                            f"torch tensor on {a.device}. Use decode_groups(), which decodes "
                            f"where the planes are, or pass the planes as numpy (.cpu().numpy()).")
        if not hasattr(a, "shape") or not hasattr(a, "__getitem__"):
            raise TypeError(f"{who}() needs the planes as arrays (numpy, or zarr arrays from "
                            f"read_parts); this code's {name} is {type(a).__name__}")


def _plane(a, i: int) -> np.ndarray:
    """One plane of a stored array as numpy: a view of an ndarray, a read of a zarr array."""
    return np.asarray(a[i])


def _shape(code: RankField) -> tuple[int, ...]:
    """``(Z, Y, X)``, from the planes themselves: a store written by another writer
    (haversack's) records no ``shape``. Where the meta does record one it must agree - a
    meta describing other planes than it came with is refused here rather than failing as a
    mismatched boolean index somewhere below."""
    shape = tuple(int(v) for v in code.support.shape[1:])
    stated = code.meta.get("shape")
    if stated is not None and tuple(int(v) for v in stated) != shape:
        raise ValueError(f"the meta's shape {list(stated)} is not the planes' {list(shape)}")
    return shape


def margin(code: RankField, channel: int) -> np.ndarray:
    """``l_c - max_{j != c} l_j``: positive inside by the lead, zero on the boundary, negative
    outside, ``-clip`` where the channel is absent. The field to render and mesh; NOT the
    field to restore from (see :func:`deficit`).

    The lead is measured against the nearest class the encoding KEPT, so it is an upper
    bound on the true lead: where a closer competitor lost the depth cut to a shell class,
    this reads high (docs/format.md, "What the keep rule does not promise").

    Each call reads every rank and support plane once. On the lazy arrays of
    :func:`rankfield.store.read_parts` that is a read and a decompression per call, so to
    decode many channels read the planes once first (``np.asarray`` each into a
    :class:`RankField`)."""
    _host(code, "margin")
    clip = code.clip
    if code.meta.get("mode") == "regions":
        _shape(code)
        q = _plane(code.support, int(channel)).astype(np.float32)
        out = (q - ZERO_LEVEL) / (SUPPORT_MAX - ZERO_LEVEL) * clip
        out[q == 0] = -clip
        return out
    return _field(code, channel, floor=True)


def _field(code: RankField, channel: int, *, floor: bool, winner: float | None = None) -> np.ndarray:
    """The channel's level at every voxel: the nearest kept competitor's gap where it wins,
    minus its own gap where it trails, ``-clip`` where it is unnamed. ``floor`` clamps every stored level to
    the clip as well - the rendering field, where the clip is the range of what is shown -
    while the restore field keeps a shell class at its true gap (format.md, "The restore").
    A winner whose runner-up is unnamed (support byte 0, the sentinel) leads by AT LEAST the
    clip: that byte is not a level and must never be read as one - ``levels[0]`` is the
    curve's far end, 64 logits under the default log byte, not 8. ``winner``, when given, is
    the level where the channel wins instead of its lead (the deficit's 0).

    Every plane is read once: ``support[0]`` serves both the winner's lead and rank 1."""
    clip = code.clip
    lut = levels(code.meta)
    out = np.full(_shape(code), -clip, np.float32)
    want = int(channel) + 1
    sel = _plane(code.ranks, 0) == want
    first = _plane(code.support, 0) if code.support.shape[0] else None
    if winner is not None:
        out[sel] = winner
    elif first is not None:
        s0 = first[sel]
        lead = lut[s0]
        if floor:
            lead = np.minimum(lead, clip)
        out[sel] = np.where(s0 == 0, np.float32(clip), lead)
    else:
        out[sel] = clip
    for j in range(1, code.ranks.shape[0]):
        sel = _plane(code.ranks, j) == want
        sup = first if j == 1 else _plane(code.support, j - 1)
        level = -lut[sup[sel]]
        out[sel] = np.maximum(level, -clip) if floor else level
    return out


def deficit(code: RankField, channel: int) -> np.ndarray:
    """``l_c - max_j l_j``: zero where c wins, negative behind (a shell class at its TRUE
    gap, an unnamed one at ``-clip``). The logits up to a per-voxel constant shared by every
    channel - the field a restore interpolates, bit for bit what ``restore`` reads.

    Each call reads every rank and support plane once. On the lazy arrays of
    :func:`rankfield.store.read_parts` that is a read and a decompression per call, so to
    decode many channels read the planes once first (``np.asarray`` each into a
    :class:`RankField`)."""
    if code.meta.get("mode") == "regions":
        return margin(code, channel)
    _host(code, "deficit")
    return _field(code, channel, floor=False, winner=0.0)


def to_device(code: RankField, device) -> RankField:
    """The planes on ``device`` once, so repeated decodes do not re-upload."""
    dev = torch.device(device)
    def mv(a):
        return None if a is None else torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    return RankField(ranks=mv(code.ranks), support=mv(code.support), tail=mv(code.tail),
                     meta=dict(code.meta), labels=code.labels, geometry=code.geometry,
                     frame=code.frame,
                     # the extra-temperature planes travel too: the meta names them, and a
                     # field whose meta promises a temperature it no longer carries makes
                     # tail_at() raise on a field it was just handed
                     tails=None if code.tails is None else {t: mv(a) for t, a in code.tails.items()})


def decode_groups(code: RankField, groups, *, device=None, quantize: bool = False) -> torch.Tensor:
    """One signed margin field per GROUP of channels, in one pass over the planes.

    A class at rank j sits at level ``-gap_j``; keeping max over members and max over
    non-members yields the union's own margin ``m_S = d_S - d_notS`` exactly - not the max of
    member margins, which would put a surface at every internal boundary the group dissolves.
    Disjoint groups take a byte-valued fast path, bit-identical to the general one.
    ``quantize`` returns uint8 with 128 on the boundary and 0 reserved as absent. Levels are
    floored at ``-clip``: a shell class recorded 30 logits behind reads as ``-clip`` here,
    because this is the rendering field (the restore reads the true level).
    """
    clip = code.clip
    K = code.classes
    shape = _shape(code)
    resident = isinstance(code.support, torch.Tensor)
    dev = torch.device(device) if device is not None else (code.support.device if resident else torch.device("cpu"))
    groups = [[int(c) for c in g] for g in groups]
    G = len(groups)

    def plane(a):
        if isinstance(a, torch.Tensor):
            return a if a.device == dev else a.to(dev)
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

    if code.meta.get("mode") == "regions":
        sup = plane(code.support)
        out = torch.full((G,) + shape, -clip, dtype=torch.float32, device=dev)
        for g, members in enumerate(groups):
            for c in members:
                q = sup[c].to(torch.float32)
                m = (q - ZERO_LEVEL) / (SUPPORT_MAX - ZERO_LEVEL) * clip
                m = torch.where(q == 0, torch.full_like(m, -clip), m)
                torch.maximum(out[g], m, out=out[g])
        return _quantize_margin(out, clip) if quantize else out

    lut = torch.from_numpy(levels(code.meta)).to(dev)
    ranks = plane(code.ranks)
    sup = plane(code.support)
    memb = torch.zeros((G, K + 1), dtype=torch.bool, device=dev)
    owner = torch.full((K + 1,), -1, dtype=torch.int64, device=dev)
    disjoint = True
    for g, members in enumerate(groups):
        for c in members:
            memb[g, c + 1] = True
            if owner[c + 1] >= 0:
                disjoint = False
            owner[c + 1] = g
    if disjoint and G:
        out = _decode_disjoint(ranks, sup, owner, G, shape, clip, lut, dev)
        return _quantize_margin(out, clip) if quantize else out
    d_in = torch.full((G,) + shape, -clip, dtype=torch.float32, device=dev)
    d_out = torch.full((G,) + shape, -clip, dtype=torch.float32, device=dev)
    floor = torch.tensor(-clip, dtype=torch.float32, device=dev)
    for j in range(ranks.shape[0]):
        rj = ranks[j].long()
        present = rj != 0
        level = torch.zeros(shape, dtype=torch.float32, device=dev) if j == 0 else -lut[sup[j - 1].long()]
        mine = memb[:, rj]
        torch.maximum(d_in, torch.where(mine & present, level, floor), out=d_in)
        torch.maximum(d_out, torch.where(~mine & present, level, floor), out=d_out)
        del rj, present, level, mine
    out = d_in.sub_(d_out)
    return _quantize_margin(out, clip) if quantize else out


def _decode_disjoint(ranks, sup, owner, G, shape, clip, lut, dev):
    """Disjoint groups: the winner's group needs the best non-member level, every other
    group its best member level, maxima kept as BYTES and converted once. Levels are
    monotone in the byte, so a byte maximum is a level maximum."""
    V = 1
    for n in shape:
        V *= n
    r0 = ranks[0].long().reshape(V)
    g0 = owner[r0]
    best = torch.zeros((G, V), dtype=torch.uint8, device=dev)
    out_b = torch.zeros(V, dtype=torch.uint8, device=dev)
    idx = torch.arange(V, device=dev)
    has = g0 >= 0
    best[g0[has], idx[has]] = SUPPORT_MAX
    for j in range(1, ranks.shape[0]):
        rj = ranks[j].long().reshape(V)
        present = rj != 0
        s = sup[j - 1].reshape(V)
        gj = owner[rj]
        mine = present & (gj >= 0)
        gi, ii = gj[mine], idx[mine]
        best[gi, ii] = torch.maximum(best[gi, ii], s[mine])
        other = present & (gj != g0) & has
        out_b = torch.where(other, torch.maximum(out_b, s), out_b)

    def level(b):                       # floored at -clip, as the general path's max does
        return torch.maximum(-lut[b.long()], torch.tensor(-clip, dtype=torch.float32, device=dev))
    out = torch.empty((G, V), dtype=torch.float32, device=dev)
    lv_out = level(out_b)
    for g in range(G):
        out[g] = torch.where(g0 == g, 0.0 - lv_out, level(best[g]) - 0.0)
    return out.reshape((G,) + shape)


def _quantize_margin(m: torch.Tensor, clip: float) -> torch.Tensor:
    """Signed margin -> uint8 with 128 exactly on the boundary, 0 reserved as absent."""
    q = (m / clip).clamp(-1, 1) * (SUPPORT_MAX - ZERO_LEVEL) + ZERO_LEVEL
    return q.round().clamp(1, SUPPORT_MAX).to(torch.uint8)


def tail_at(code: RankField, temperature: float = 1.0):
    """The stored dropped-mass plane at ``temperature`` (uint16 over ``tail_max``), or None for
    an exhaustive field. Raises when the field is not exhaustive and no tail was stored at
    that temperature: renormalizing with the wrong one would misstate every probability."""
    t = float(temperature)
    if code.meta.get("exhaustive"):
        return None
    if t == 1.0 and code.tail is not None:
        return code.tail
    if code.tails and t in code.tails:
        return code.tails[t]
    have = ([1.0] if code.tail is not None else []) + sorted(code.tails or {})
    raise ValueError(f"no tail stored at temperature {t:g} (stored: {have or 'none'}); the mass "
                     f"of the dropped classes at that temperature is not recoverable")


def probabilities(code: RankField, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(class_ids, p)`` for the stored channels at ``temperature``; absent classes get id
    -1 and p 0. Ranked: ``p_j = exp(-g_j / T) / Z`` with ``Z = Z_kept / (1 - tail_T)`` - the
    stored classes carry ``1 - tail_T`` of the mass and the tail is the rest, so a consumer
    sums to one only by counting the tail as one more target (docs/format.md,
    "Distillation"). Needs a tail stored at ``T`` (:func:`tail_at`); regions: sigmoids."""
    _host(code, "probabilities")
    T = float(temperature)
    if not T > 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if code.meta.get("mode") == "regions":
        m = np.stack([margin(code, c) for c in range(code.classes)])
        ids = np.broadcast_to(np.arange(code.classes, dtype=np.int64)[:, None, None, None], m.shape)
        return ids.copy(), 1.0 / (1.0 + np.exp(-m))
    lut = levels(code.meta)
    support = np.asarray(code.support)                  # every plane at once: read it whole
    gaps = np.concatenate([np.zeros((1, *support.shape[1:]), np.float32),
                           lut[support]], axis=0)
    w = np.exp(-gaps / T)
    ids = np.asarray(code.ranks).astype(np.int64) - 1
    w[ids < 0] = 0.0
    z = w.sum(axis=0)
    tail = tail_at(code, T)
    if tail is not None:
        tail_max = float(code.meta.get("tail_max") or TAIL_MAX)   # the uint16 quantum
        z = z / np.clip(1.0 - np.asarray(tail).astype(np.float32) / tail_max, 1e-6, None)
    p = w / z
    p[ids < 0] = 0.0
    return ids, p
