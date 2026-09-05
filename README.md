# rankfield

> **Alpha, under heavy development.** The API, the format and the numbers in the
> specification change between commits without notice; nothing here is stable yet, and
> releases are snapshots for the consumers that pin them, not promises. Use at your own
> risk, and pin a tag.

A per-class field kept as ranks. A network's output is a field, one value per class per
voxel, of which only the differences matter. rankfield keeps, per voxel, the few classes
that can still win and how far each trails the winner, one byte each, and gives back the
fields (margins, deficits, probabilities) and the labels on any grid - the restore -
without ever materializing the K-channel volume.

- `docs/format.md` is the specification: what is stored, the level table, the keep rule,
  the two fields, the restore and its rules, with the measurements behind each.
- `rankfield.encode` / `encode_regions`: logits -> `RankField`.
- `margin`, `deficit`, `decode_groups`, `probabilities`: the fields.
- `restore`: labels on any grid from `Part`s, with a Metal and a Triton kernel that agree
  with the torch path bit for bit.

In-memory only for now: serialization stays with the first consumer (haversack's ranked
store) until it is locked down. numpy is required; torch for encode and the restore's
blend; triton (>= 3.0) for CUDA.

    uv pip install -e .            # from a checkout; or: pip install -e .
    pip install "rankfield @ git+https://github.com/mhalle/rankfield.git@v0.1.2"
    pytest tests

Apache-2.0.
