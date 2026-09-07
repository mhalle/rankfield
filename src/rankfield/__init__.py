"""rankfield: a per-class field kept as ranks.

A network's output is a field: one value per class per voxel, of which only the
differences matter. rankfield keeps, per voxel, the few classes that can still win and how
far each trails the winner, in one byte each, and gives back the fields (margins, deficits,
probabilities) and the labels on any grid - the restore - without ever materializing the
K-channel volume. The rules are in ``docs/format.md``; the encoder is ``encode``, the
decoders ``margin``/``deficit``/``decode_groups``/``probabilities``, the restore ``restore``.
"""
from .code import (CLIP, DEFAULT_DEPTH, FORMAT_VERSION, GAP_UNIT, SUPPORT_MAX, TAIL_MAX,
                   ZERO_LEVEL, Part, RankField, byte_of_gap, levels)
from .decode import decode_groups, deficit, margin, probabilities, tail_at, to_device
from .encode import encode, encode_regions, settle_ties
from .frame import Frame
from .geometry import Geometry
from .grid import Grid
from .mapping import Mapping
from .reference import reference_restore
from . import store  # noqa: F401 - zarr at call time only
from .restore import Restored, array_grid, mapping_of, resolve_grid, restore, roi_of

__all__ = ["CLIP", "DEFAULT_DEPTH", "FORMAT_VERSION", "GAP_UNIT", "SUPPORT_MAX", "TAIL_MAX",
           "ZERO_LEVEL", "Frame", "Geometry", "Grid", "Mapping", "Part", "RankField", "Restored",
           "array_grid", "byte_of_gap", "decode_groups", "deficit", "encode", "encode_regions", "levels",
           "mapping_of", "margin", "probabilities", "reference_restore", "resolve_grid", "restore", "roi_of",
           "settle_ties", "tail_at", "to_device"]
__version__ = "0.2.0"
