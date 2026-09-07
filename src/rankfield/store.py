"""A ranked store on disk: the portable form, read and written without haversack.

A store is a standard zarr v3 group - a directory, or a zip of one - whose ``parts/<i>``
groups each hold one part's planes (``ranks``, ``support``, optional ``tail`` and the
per-temperature tails of format 0.4) with the ``ranked`` block in their duckn attributes
and the array's placement in duckn's geometry vocabulary (``space``, ``space_origin``,
``axes`` with ``space_direction``). That is everything a reader needs to rebuild
:class:`Part`\\ s and restore, decode or distill from them; haversack's stores carry more
(the seg extension, provenance, derived layers) on top of exactly this.

Reading needs ``zarr`` (the ``store`` extra). Writing builds the geometry attributes with
duckn's own models rather than hand-written dicts (duckn is not on PyPI:
``pip install "duckn @ git+https://github.com/mhalle/duckn.git"``).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .code import FORMAT_VERSION, Part, RankField
from .geometry import Geometry

KNOWN_VERSIONS = ("0.2", "0.3", "0.4")
DUCKN_VERSION = "1.0"
CHUNK = 64

_SPACE_TO_LPS = {"left-posterior-superior": (1.0, 1.0, 1.0),
                 "right-anterior-superior": (-1.0, -1.0, 1.0),
                 "left-anterior-superior": (1.0, -1.0, 1.0)}


def is_zip(path) -> bool:
    return str(path).lower().endswith(".zip")


def open_group(path, mode: str = "r"):
    """The store's root zarr group. ``mode`` is ``"r"`` only: writing goes through
    :func:`write_parts`, which stages and packs (a zip entry cannot be rewritten)."""
    import zarr
    from zarr.storage import LocalStore, ZipStore
    if mode != "r":
        raise ValueError("open_group is read-only; write with write_parts")
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))
    st = ZipStore(str(p), mode="r") if is_zip(p) else LocalStore(str(p), read_only=True)
    return zarr.open_group(store=st, mode="r")


# ----------------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------------

def array_geometry(arr) -> Geometry:
    """The stored array's own placement, from its duckn attributes, in SimpleITK's terms
    (LPS, direction cosines by column x, y, z)."""
    d = arr.attrs.asdict()["duckn"]
    axes = [a for a in d["axes"] if a.get("space_direction")]
    if len(axes) != 3:
        raise ValueError("ranks array: need three spatial axes with space_direction")
    space = str(d.get("space", "left-posterior-superior")).replace("-time", "")
    if space not in _SPACE_TO_LPS:
        raise ValueError(f"ranks array: space {space!r} is not one this reader places")
    flip = np.asarray(_SPACE_TO_LPS[space])
    dirs = [np.asarray(a["space_direction"], float) * flip for a in axes]      # z, y, x in LPS
    spacing = [float(np.linalg.norm(v)) for v in dirs]
    cos = [v / n for v, n in zip(dirs, spacing)]
    D = np.stack([cos[2], cos[1], cos[0]], axis=1)                              # columns x, y, z
    origin = np.asarray(d.get("space_origin", (0.0, 0.0, 0.0)), float) * flip
    return Geometry(spacing_zyx=tuple(spacing), shape_zyx=tuple(int(v) for v in arr.shape[1:]),
                    origin_xyz=tuple(float(v) for v in origin),
                    direction_xyz=tuple(float(v) for v in D.reshape(-1)))


def part_indices(root) -> list[int]:
    """Paint order: ``part_order`` from a ``haversack`` or ``rankfield`` root block, else
    ``parts/0, 1, ...`` as found."""
    ext = root.attrs.asdict().get("duckn", {}).get("extensions", {})
    for key in ("haversack", "rankfield"):
        order = (ext.get(key) or {}).get("part_order")
        if order:
            return [int(p["index"]) for p in order]
    idx = []
    while f"parts/{len(idx)}" in root:
        idx.append(len(idx))
    return idx


def read_parts(store) -> list[Part]:
    """The store's parts in paint order, planes as the zarr arrays themselves (read where a
    restore needs them, never whole). ``store`` is a path or an open root group."""
    root = open_group(store) if isinstance(store, (str, Path)) else store
    out = []
    for i in part_indices(root):
        g = root[f"parts/{i}"]
        m = dict(g.attrs.asdict()["duckn"]["extensions"]["ranked"])
        if str(m.get("version")) not in KNOWN_VERSIONS:
            raise ValueError(f"parts/{i}: ranked format {m.get('version')!r}; this reader knows "
                             f"{', '.join(KNOWN_VERSIONS)}")
        env = m.get("envelope")
        start = tuple(int(a) for a in env[0::2]) if isinstance(env, (list, tuple)) else (
            tuple(int(a) for a in env["start"]) if isinstance(env, dict) and "start" in env else (0, 0, 0))
        tails = {float(t): g[tail_array_name(t)] for t in m.get("tail_temperatures", [])
                 if tail_array_name(t) in g}
        field = RankField(ranks=g["ranks"], support=g["support"],
                          tail=g["tail"] if "tail" in g else None, meta=m,
                          labels=[int(v) for v in m["labels"]], geometry=array_geometry(g["ranks"]),
                          frame=m.get("frame"), tails=tails or None)
        out.append(Part(field=field, envelope_start=start, name=str(m.get("part", i))))
    return out


def tail_array_name(temperature: float) -> str:
    """``tail`` at temperature 1, ``tail_temperature_<T>`` otherwise (``4`` not ``4.0``)."""
    t = float(temperature)
    if t == 1.0:
        return "tail"
    return f"tail_temperature_{int(t) if t.is_integer() else t}"


# ----------------------------------------------------------------------------------------
# writing: duckn's models for the geometry, staged then packed
# ----------------------------------------------------------------------------------------

def _grid_attrs(geo: Geometry, *, list_axis: bool, centering: str) -> dict:
    from duckn import AxisMetadata, DucknMetadata
    from duckn.models import duckn_attrs
    D = np.asarray(geo.direction_xyz, float).reshape(3, 3)
    cols = [D[:, 2], D[:, 1], D[:, 0]]                              # array axes Z, Y, X
    axes = [AxisMetadata(kind="space", centering=centering, unit="mm",
                         space_direction=[round(float(v), 9) for v in (c * s)])
            for c, s in zip(cols, geo.spacing_zyx)]
    if list_axis:
        axes = [AxisMetadata(kind="list")] + axes
    return duckn_attrs(DucknMetadata(version=DUCKN_VERSION, space="left-posterior-superior",
                                     space_origin=[float(v) for v in geo.origin_xyz], axes=axes))


def _block_attrs(block: dict) -> dict:
    from duckn import DucknMetadata
    from duckn.models import duckn_attrs
    return duckn_attrs(DucknMetadata(version=DUCKN_VERSION, extensions={"ranked": block}))


def _root_attrs(order: list[dict]) -> dict:
    from duckn import DucknMetadata
    from duckn.models import duckn_attrs
    return duckn_attrs(DucknMetadata(version=DUCKN_VERSION, extensions={
        "rankfield": {"version": FORMAT_VERSION, "part_order": order}}))


def _write_array(g, name, data, attrs):
    import zarr
    data = np.asarray(data)
    chunks = (1,) * (data.ndim - 3) + (CHUNK,) * 3
    arr = g.create_array(name, shape=data.shape, dtype=data.dtype, chunks=chunks,
                         compressors=zarr.codecs.ZstdCodec(level=9), attributes=attrs)
    arr[:] = data


def write_parts(path, parts, *, centering: str = "node") -> Path:
    """Write ``parts`` as a bare ranked store at ``path`` (a directory, or a ``.zip``).

    Each part's planes and its ``ranked`` block (the meta plus ``labels``, ``part``,
    ``envelope`` and ``frame`` when known) go under ``parts/<i>``; the arrays are placed by
    the part's :class:`Geometry`. ``centering`` is duckn's word for what a sample is - ``node``
    for a corner-rule model grid (the usual), ``cell`` for an image grid. A zip is staged
    beside the target and packed uncompressed, so a reader can range-read its members.
    """
    import zarr
    from zarr.storage import LocalStore
    parts = list(parts)
    if not parts:
        raise ValueError("write_parts: no parts")
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + ".staging-", dir=out.parent))
    try:
        root = zarr.create_group(store=LocalStore(str(staging)))
        order = []
        for i, p in enumerate(parts):
            f = p.field
            if f.geometry is None or f.labels is None:
                raise ValueError(f"part {i}: a stored part needs labels and a geometry")
            geo = f.geometry
            shape = tuple(int(v) for v in np.asarray(f.ranks).shape[1:])
            block = dict(f.meta)
            block.update({"labels": [int(v) for v in f.labels], "part": p.name or str(i),
                          "shape": list(shape),
                          "envelope": [int(v) for pair in zip(p.envelope_start, [s + e - 1 for s, e in zip(p.envelope_start, shape)]) for v in pair]})
            if f.frame:
                block["frame"] = f.frame
            if f.tails:
                block["tail_temperatures"] = sorted(float(t) for t in f.tails)
            g = root.create_group(f"parts/{i}", attributes=_block_attrs(block))
            _write_array(g, "ranks", f.ranks, _grid_attrs(geo, list_axis=True, centering=centering))
            _write_array(g, "support", f.support, _grid_attrs(geo, list_axis=True, centering=centering))
            if f.tail is not None:
                _write_array(g, "tail", f.tail, _grid_attrs(geo, list_axis=False, centering=centering))
            for t, arr in (f.tails or {}).items():
                if float(t) != 1.0:
                    _write_array(g, tail_array_name(t), arr, _grid_attrs(geo, list_axis=False, centering=centering))
            order.append({"index": i, "name": p.name or str(i)})
        root.attrs.update(_root_attrs(order))
        if is_zip(out):
            partial = out.with_name(out.name + ".partial")
            with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                for fp in sorted(q for q in staging.rglob("*") if q.is_file()):
                    zf.write(fp, fp.relative_to(staging).as_posix())
            os.replace(partial, out)
        else:
            if out.exists():
                shutil.rmtree(out)
            os.replace(staging, out)
            staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return out
