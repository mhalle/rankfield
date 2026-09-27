"""Host-side per-axis index / weight tables and the per-axis decision rule: labelfield's,
re-exported (see rankfield.grid)."""
from __future__ import annotations

from labelfield.tables import INTERP, OUTSIDE, AxisTable, axis_coords, axis_table, build_tables, normalize_interp

__all__ = ["INTERP", "OUTSIDE", "AxisTable", "axis_coords", "axis_table", "build_tables", "normalize_interp"]
