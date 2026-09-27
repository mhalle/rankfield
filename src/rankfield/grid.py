"""Axis-aligned sampling grids in a canonical frame: labelfield's, re-exported.

rankfield's geometry - Grid, the per-axis Mapping, the general Affine and the tables - is
labelfield's (github.com/mhalle/labelfield) since 0.3.8, so a restore from a store and a restore
from live logits (haversack) decide with the same code. These modules keep rankfield's paths.
"""
from __future__ import annotations

from labelfield.grid import Grid, Shape3, Vec3, _shape3, _vec3

__all__ = ["Grid", "Shape3", "Vec3"]
