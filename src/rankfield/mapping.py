"""Maps between index spaces: per-axis (:class:`Mapping`), and general (:class:`Affine`)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid, Vec3, _vec3


@dataclass(frozen=True)
class Mapping:
    """``x_to = a * x_from + b`` per axis (Z, Y, X).

    Maps integer voxel indices of a *from* grid to continuous voxel
    coordinates of a *to* grid. Separable: no rotation, no shear - flips and
    permutations belong to the caller's frame, not here. ``a >= 0``.

    Compose with ``>>``: ``m1 >> m2`` applies ``m1`` first.
    """

    a: Vec3
    b: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "a", _vec3(self.a, "a"))
        object.__setattr__(self, "b", _vec3(self.b, "b"))
        if any(x < 0 for x in self.a):
            raise ValueError(f"a must be >= 0 on every axis (flips belong to the frame); got {self.a}")

    def apply(self, x_from) -> np.ndarray:
        """Apply to coordinates (..., 3)."""
        return np.asarray(x_from, dtype=np.float64) * np.asarray(self.a) + np.asarray(self.b)

    def then(self, other: "Mapping") -> "Mapping":
        """``self`` first, then ``other``."""
        a1, b1 = np.asarray(self.a), np.asarray(self.b)
        a2, b2 = np.asarray(other.a), np.asarray(other.b)
        return Mapping(tuple(a2 * a1), tuple(a2 * b1 + b2))

    __rshift__ = then

    def inverse(self) -> "Mapping":
        if any(x == 0 for x in self.a):
            raise ValueError("mapping with a zero factor has no inverse")
        a = 1.0 / np.asarray(self.a)
        return Mapping(tuple(a), tuple(-np.asarray(self.b) * a))

    # -- constructors -----------------------------------------------------
    @classmethod
    def identity(cls) -> "Mapping":
        return cls((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    @classmethod
    def center(cls, shape_from, shape_to) -> "Mapping":
        """Voxel-center (half-pixel) rule: ``x_to = (x_from + 0.5) * n_to / n_from - 0.5``.

        The rule of ``skimage.transform.resize``, ITK, ``F.interpolate(align_corners=False)``
        and therefore nnU-Net's own resampler. ``Mapping.center(n_src, n_model)`` maps
        source-image indices to model-grid coordinates for a model grid produced
        by that resampler - i.e. it inverts the forward resample exactly.
        """
        n_from = np.asarray(shape_from, dtype=np.float64)
        n_to = np.asarray(shape_to, dtype=np.float64)
        a = n_to / n_from
        return cls(tuple(a), tuple(0.5 * a - 0.5))

    @classmethod
    def corner(cls, shape_from, shape_to) -> "Mapping":
        """Voxel-corner rule: ``x_to = x_from * (n_to - 1) / (n_from - 1)``.

        The rule of ``scipy.ndimage.zoom(grid_mode=False)`` and
        ``F.interpolate(align_corners=True)`` - what TotalSegmentator's
        ``change_spacing`` uses on both legs. An axis with a single sample maps
        to coordinate 0.
        """
        n_from = np.asarray(shape_from, dtype=np.float64)
        n_to = np.asarray(shape_to, dtype=np.float64)
        a = np.where(n_from > 1, (n_to - 1) / np.maximum(n_from - 1, 1), 0.0)
        return cls(tuple(a), (0.0, 0.0, 0.0))

    @classmethod
    def spacing(cls, spacing_from, spacing_to, shift=(0.0, 0.0, 0.0)) -> "Mapping":
        """Origin-aligned, spacing-exact rule: ``x_to = x_from * s_from / s_to + shift``.

        Voxel (0, 0, 0) of both grids coincide (``shift = 0``); this is the
        the MLX toolkit's kernel's convention (its ``s2t = acq / target``).
        """
        s_from = np.asarray(_vec3(spacing_from, "spacing_from"))
        s_to = np.asarray(_vec3(spacing_to, "spacing_to"))
        return cls(tuple(s_from / s_to), _vec3(shift, "shift"))

    @classmethod
    def between(cls, grid_from: Grid, grid_to: Grid) -> "Mapping":
        """Physical mapping: index on ``grid_from`` -> coordinate on ``grid_to``
        through millimeters. Identity when the grids coincide."""
        gf, gt = Grid.like(grid_from), Grid.like(grid_to)
        s_from, s_to = np.asarray(gf.spacing), np.asarray(gt.spacing)
        b = (np.asarray(gf.origin) - np.asarray(gt.origin)) / s_to
        return cls(tuple(s_from / s_to), tuple(b))

    def __repr__(self) -> str:
        a = ", ".join(f"{x:g}" for x in self.a)
        b = ", ".join(f"{x:g}" for x in self.b)
        return f"Mapping(a=({a}), b=({b}))"


@dataclass(frozen=True)
class Affine:
    """``x_to = x_from @ m + b``: a general affine map between index spaces (array order,
    row vectors), rotation and shear included.

    For two grids whose axes do not line up in the world - a model grid a model resampled
    in world space, like FastSurfer's conformed 1 mm grid, against an oblique acquisition -
    no per-axis :class:`Mapping` relates them. :meth:`between` builds this one from their
    two :class:`~rankfield.geometry.Geometry` records: output index -> world -> source index.
    Every restore decision (the inside test, the edge clamp, the corners and weights) stays
    per axis, on the coordinates this produces (``restore._affine_part``).
    """

    m: tuple
    b: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self):
        m = np.asarray(self.m, dtype=np.float64)
        if m.shape != (3, 3) or not np.isfinite(m).all():
            raise ValueError(f"m must be a finite 3x3 matrix; got {self.m!r}")
        object.__setattr__(self, "m", tuple(tuple(float(v) for v in r) for r in m))
        object.__setattr__(self, "b", _vec3(self.b, "b"))

    def apply(self, x_from) -> np.ndarray:
        """Apply to coordinates (..., 3)."""
        return np.asarray(x_from, dtype=np.float64) @ np.asarray(self.m) + np.asarray(self.b)

    def inverse(self) -> "Affine":
        mi = np.linalg.inv(np.asarray(self.m))
        return Affine(mi, tuple(-np.asarray(self.b) @ mi))

    @property
    def separable(self) -> Mapping | None:
        """The same map as a per-axis :class:`Mapping` - no rotation, no shear, no flip -
        or None. A restore takes the per-axis path (and the fused kernels) whenever this
        exists, so two grids that happen to line up restore exactly as they always did."""
        m = np.asarray(self.m)
        d = np.diag(m)
        if np.any(m - np.diag(d)) or np.any(d < 0):
            return None
        return Mapping(tuple(d), self.b)

    @classmethod
    def between(cls, geo_from, geo_to) -> "Affine":
        """Index on ``geo_from`` -> continuous index on ``geo_to``, through the world: both
        are :class:`~rankfield.geometry.Geometry` (array-order direction rows, LPS mm)."""
        d_from = np.asarray(geo_from.directions, dtype=np.float64)
        d_to_inv = np.linalg.inv(np.asarray(geo_to.directions, dtype=np.float64))
        b = (np.asarray(geo_from.origin, dtype=np.float64) - np.asarray(geo_to.origin, dtype=np.float64)) @ d_to_inv
        return cls(d_from @ d_to_inv, tuple(b))

    def __repr__(self) -> str:
        return f"Affine(m={np.round(np.asarray(self.m), 6).tolist()}, b={tuple(round(v, 6) for v in self.b)})"
