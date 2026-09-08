"""Where a (Z, Y, X) array sits in the world: duckn's form, which is NRRD's.

One order. ``shape`` and ``directions`` are per array axis, in array order; ``origin`` and
each direction row are world-space vectors, LPS millimeters, components (x, y, z). Nothing
here holds array-order and world-order quantities under one name, and the index-to-world
map is the rows as they stand: ``world = origin + k * directions[0] + j * directions[1] +
i * directions[2]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Vec3 = tuple[float, float, float]


def _vec3(v, name: str) -> Vec3:
    t = tuple(float(x) for x in v)
    if len(t) != 3:
        raise ValueError(f"{name} must have 3 entries; got {v!r}")
    return t


@dataclass(frozen=True)
class Geometry:
    """``shape`` samples per array axis; ``directions[i]`` is the world step from one sample
    to the next along array axis ``i`` - a direction VECTOR, not a cosine: its length is the
    spacing (NRRD's ``space directions``, duckn's ``space_direction``); ``origin`` is the world
    position of sample (0, 0, 0). :meth:`aligned` builds the common case, array axes along the
    world axes; :attr:`spacing` and :attr:`cosines` are derived, never stored."""

    shape: tuple[int, int, int]
    directions: tuple[Vec3, Vec3, Vec3]
    origin: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self):
        shape = tuple(int(s) for s in self.shape)
        if len(shape) != 3 or any(s < 1 for s in shape):
            raise ValueError(f"shape must be 3 positive ints (Z, Y, X); got {self.shape!r}")
        rows = tuple(self.directions)
        if len(rows) != 3:
            raise ValueError(f"directions must be 3 rows, one per array axis; got {self.directions!r}")
        rows = tuple(_vec3(r, "a direction row") for r in rows)
        if any(np.linalg.norm(r) == 0 for r in rows):
            raise ValueError(f"a direction row has zero length: {rows}")
        if np.linalg.det(np.asarray(rows)) == 0:
            raise ValueError(f"directions do not span space: {rows}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "directions", rows)
        object.__setattr__(self, "origin", _vec3(self.origin, "origin"))

    # -- constructors -----------------------------------------------------------------
    @classmethod
    def aligned(cls, shape, spacing, origin=(0.0, 0.0, 0.0)) -> "Geometry":
        """Array axes along the world axes - Z along z, Y along y, X along x - at ``spacing``
        (array order, or one number for all three)."""
        s = (spacing,) * 3 if np.isscalar(spacing) else tuple(spacing)
        s = _vec3(s, "spacing")
        return cls(shape=tuple(shape),
                   directions=((0.0, 0.0, s[0]), (0.0, s[1], 0.0), (s[2], 0.0, 0.0)),
                   origin=origin)

    # -- derived ------------------------------------------------------------------------
    @property
    def spacing(self) -> Vec3:
        """Sample step per array axis, mm: the length of each direction row."""
        return tuple(float(np.linalg.norm(r)) for r in self.directions)

    @property
    def cosines(self) -> tuple[Vec3, Vec3, Vec3]:
        """The direction rows at unit length: ITK's direction cosines, one row per array axis."""
        return tuple(tuple(float(v) for v in np.asarray(r) / np.linalg.norm(r)) for r in self.directions)

    @property
    def matrix(self) -> np.ndarray:
        """4x4 index -> world: ``world = matrix @ (k, j, i, 1)``, index in array order."""
        m = np.eye(4)
        m[:3, :3] = np.asarray(self.directions, float).T
        m[:3, 3] = self.origin
        return m

    def world(self, index) -> np.ndarray:
        """Continuous array-order index (..., 3) -> world position (..., 3), mm."""
        return np.asarray(self.origin) + np.asarray(index, float) @ np.asarray(self.directions, float)

    def regrid(self, shape, spacing, offset=(0.0, 0.0, 0.0)) -> "Geometry":
        """The same orientation on another grid: ``shape`` samples at ``spacing``, sample
        (0, 0, 0) moved ``offset`` mm along each of this geometry's array axes."""
        cos = np.asarray(self.cosines)
        s = _vec3((spacing,) * 3 if np.isscalar(spacing) else spacing, "spacing")
        origin = np.asarray(self.origin) + np.asarray(_vec3(offset, "offset")) @ cos
        return Geometry(shape=tuple(shape),
                        directions=tuple(tuple(float(v) for v in c * k) for c, k in zip(cos, s)),
                        origin=tuple(float(v) for v in origin))

    # -- records ------------------------------------------------------------------------
    def to_meta(self) -> dict:
        return {"shape": list(self.shape), "directions": [list(r) for r in self.directions],
                "origin": list(self.origin)}

    @classmethod
    def from_meta(cls, meta: dict) -> "Geometry":
        if "directions" not in meta:
            raise ValueError("a geometry record needs shape, directions and origin; "
                             f"got keys {sorted(meta)} (a record with spacing_zyx / direction_xyz "
                             "predates rankfield 0.3 and has to be rewritten)")
        return cls(shape=tuple(meta["shape"]), directions=tuple(tuple(r) for r in meta["directions"]),
                   origin=tuple(meta.get("origin", (0.0, 0.0, 0.0))))
