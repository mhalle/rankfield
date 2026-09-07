"""The data structure: what is stored per voxel, and the facts that give the bytes meaning.

Per voxel::

    ranks    (N, Z, Y, X)    class + 1 for the N kept classes, 0 = "not this class"
    support  (N-1, Z, Y, X)  how far each trails the winner, as a byte: 255 = tied, 1 = at
                             the range, 0 = absent (the sentinel, like the rank)
    tail     (Z, Y, X)       probability mass of everything NOT kept, uint16 (optional)

Zero means "nothing here" in every array, so an unwritten block, a zeroed buffer or a
forgotten fill decodes as absent - the safe answer - and small integers leave the high
byte constant for a byte-shuffling compressor.

The byte -> gap map is a CURVE with a RANGE (format 0.3: ``log`` over 64 logits with a
0.5-logit origin, fine near zero where boundaries live, coarse far out where nothing is
decided) and the kept set is decided by a KEEP RULE (``shell``: every class that wins at
the voxel or at one of its 26 neighbours, always, with its true gap; then the closest
non-winners within ``clip``). Format 0.2 stores are the special case ``uniform`` over
``clip`` with ``clip`` keeping; every decoder reads both through :func:`levels`.

Two fields fall out of the same bytes and are not interchangeable: the DEFICIT
``l_c - max_j l_j`` (zero where c wins) is the logits up to a per-voxel constant, so it is
the field to interpolate before an argmax; the MARGIN ``l_c - max_{j != c} l_j`` (positive
inside by c's lead) is the signed field whose zero level set is c's surface - the field
to render and mesh. Region (sigmoid) heads store their margin directly, referenced to the
decision threshold, with ``ZERO_LEVEL`` on the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FORMAT_VERSION = "0.4"
GAP_UNIT = "logit"       # what gaps, levels, the clip and the range are measured in for a softmax head
CLIP = 8.0               # non-winners farther behind than this are not kept
DEFAULT_DEPTH = 6        # planes; the shell never needed more than 6 on real anatomy
GAP_RANGE = 64.0         # the gap byte 1 stands for
GAP_ORIGIN = 0.5         # the log curve's knee: below it the quantum is a few thousandths of a logit
SUPPORT_MAX = 255        # 255 = tied with the winner
ZERO_LEVEL = 128         # regions only: the level the decision boundary lands on exactly
TAIL_MAX = 65535         # uint16 tail: quantum 1/65535 of the mass
FLOOR = "clip"           # an absent class is scored at -clip by the restore


@dataclass
class RankField:
    """One part's encoded field: the planes plus the facts that give them meaning.

    ``meta`` is what a container serializes verbatim (the ``ranked`` block of a store);
    :func:`levels` turns it into the byte -> gap map, so a reader never hard-codes a curve.
    ``labels`` maps channel -> label value; ``geometry`` places the array in the world;
    ``frame`` is the pipeline's record of how the array was derived from a source grid
    (``Frame.to_meta``), which is what makes a restore onto the source exact.
    """

    ranks: np.ndarray | None
    support: np.ndarray
    tail: np.ndarray | None
    meta: dict = field(default_factory=dict)
    labels: list | None = None
    geometry: object | None = None       # rankfield.geometry.Geometry of the stored array
    frame: dict | None = None            # Frame.to_meta()
    tails: dict | None = None            # {temperature: uint16 plane} beyond the T=1 `tail` (format 0.4)

    @property
    def depth(self) -> int:
        return int(self.meta["depth"])

    @property
    def classes(self) -> int:
        return int(self.meta["classes"])

    @property
    def clip(self) -> float:
        return float(self.meta["clip"])

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.ranks, self.support, self.tail) if a is not None)

    def __repr__(self) -> str:
        shape = "x".join(str(s) for s in self.support.shape[1:])
        return (f"RankField({self.meta.get('mode', '?')} {self.meta.get('version', '?')}, "
                f"K={self.meta.get('classes')}, depth={self.meta.get('depth')}, {shape}, "
                f"{self.nbytes / 1e6:.1f} MB raw)")


@dataclass
class Part:
    """A field placed for the restore: its channel table and, when known, the envelope crop
    of the model grid it was cut from. ``envelope_start`` is the crop's first index per axis
    on the model grid (all zero when the array is the whole grid)."""

    field: RankField
    envelope_start: tuple = (0, 0, 0)
    name: str = ""


# ----------------------------------------------------------------------------------------
# the byte curve

def _curve(meta: dict) -> tuple[str, float, float]:
    """``(curve, range, origin)`` for a meta, with 0.2's uniform-over-clip as the default."""
    curve = str(meta.get("gap_curve", "uniform"))
    rng = float(meta.get("gap_range", meta["clip"]))
    origin = float(meta.get("gap_origin", GAP_ORIGIN))
    if curve not in ("uniform", "log"):
        raise ValueError(f"unknown gap_curve {curve!r}")
    if not rng > 0 or (curve == "log" and not origin > 0):
        # a table built from these would be zeros, negative or NaN - and every decoder
        # would index it without a word (the JavaScript reader refuses the same values)
        raise ValueError(f"gap_range {rng} / gap_origin {origin} do not describe a curve")
    return curve, rng, origin


def levels(meta: dict) -> np.ndarray:
    """The byte -> gap map as a 256-entry float32 table: ``levels[s]`` is the gap a support
    byte ``s`` stands for (``levels[0]`` is the range, what an absent class would decode to
    if it were read; readers floor absent classes at the clip instead).

    A TABLE, deliberately: every decoder - numpy, torch, the Metal and Triton kernels, the
    JavaScript port - indexes the same 256 float32 values, so they agree bit for bit
    whatever their division rounds like.
    """
    curve, rng, origin = _curve(meta)
    s = np.arange(256, dtype=np.float64) / SUPPORT_MAX
    if curve == "uniform":
        g = (1.0 - s) * rng
    else:
        g = origin * np.expm1((1.0 - s) * np.log1p(rng / origin))
    return g.astype(np.float32)


def byte_of_gap(gaps, meta: dict):
    """Gaps -> support bytes (float, unrounded, 0..255) under ``meta``'s curve; torch or numpy."""
    curve, rng, origin = _curve(meta)
    if hasattr(gaps, "clamp"):                         # torch
        import torch
        if curve == "uniform":
            u = 1.0 - gaps / rng
        else:
            u = 1.0 - torch.log1p(gaps / origin) / float(np.log1p(rng / origin))
        return u.clamp(0, 1) * SUPPORT_MAX
    gaps = np.asarray(gaps, dtype=np.float64)
    if curve == "uniform":
        u = 1.0 - gaps / rng
    else:
        u = 1.0 - np.log1p(gaps / origin) / np.log1p(rng / origin)
    return np.clip(u, 0, 1) * SUPPORT_MAX


def rank_dtype(K: int):
    """``class + 1`` must fit, and the sentinel is 0. Declared in meta, never assumed."""
    return np.uint8 if K + 1 <= np.iinfo(np.uint8).max else np.uint16
