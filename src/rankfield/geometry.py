"""Voxel-grid placement in physical space, SimpleITK's conventions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    """Voxel-grid placement in physical space, SimpleITK's conventions.

    ``spacing_zyx`` / ``shape_zyx`` are in array order; ``origin_xyz`` and the 9-tuple
    row-major ``direction_xyz`` are in SimpleITK's X, Y, Z order.
    """

    spacing_zyx: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction_xyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "spacing_zyx", tuple(float(s) for s in self.spacing_zyx))
        object.__setattr__(self, "shape_zyx", tuple(int(s) for s in self.shape_zyx))
        object.__setattr__(self, "origin_xyz", tuple(float(o) for o in self.origin_xyz))
        object.__setattr__(self, "direction_xyz", tuple(float(d) for d in self.direction_xyz))
        if len(self.direction_xyz) != 9:
            raise ValueError(f"direction_xyz must have 9 entries; got {len(self.direction_xyz)}")
