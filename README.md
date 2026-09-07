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
  what that rule does NOT promise and what it costs, the two fields, the restore and its
  rules, with the measurements behind each.
- `rankfield.encode` / `encode_regions`: logits -> `RankField`.
- `margin`, `deficit`, `decode_groups`, `probabilities`, `tail_at`: the fields.
- `restore`: labels on any grid from `Part`s, with a Metal and a Triton kernel that agree
  with the torch path bit for bit.
- `rankfield.store`: read and write the portable zarr v3 store.

The restore is an approximation, and the format document says how good. On a synthetic field
built to stress it - 12 classes, heavy noise - labels differ from the interpolated original
logits at 0.9 % of voxels at depth 6 and 0.15 % at depth 12, the remainder being byte
quantization. Depth is the knob. On one real torso case the figure is 0.0037 %.

numpy is all the decoders need. torch is the `torch` extra, for `encode`, `decode_groups`
and the restore's blend; `zarr` is the `store` extra; triton (>= 3.0) is the CUDA kernel.
Python 3.12 or newer.

    uv sync --extra test           # from a checkout
    pip install "rankfield[torch] @ git+https://github.com/mhalle/rankfield.git@v0.2.1"
    pytest tests

Apache-2.0.
