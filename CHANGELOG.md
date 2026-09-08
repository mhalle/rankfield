# Changelog

## 0.3.1 - 2026-09-07

- A multi-part restore compared two FRAMED parts' placement with `==` on the nested frame
  record, while two UNFRAMED parts were compared with a tolerance on origin and directions -
  the same geometric question answered two ways, and the exact way was the one handling the
  record with the most chances to differ innocently. Parts being composited can derive their
  frame independently from the same input, and a number off by one ULP, or a sequence a
  producer built as a tuple where `Frame.to_meta` emits a list, was refused as a different
  placement. `[6, 10, 12] != (6, 10, 12)` needs no arithmetic to happen. The comparison
  walks the record now: numbers to `np.isclose`'s tolerance, which is what the unframed
  branch already used, sequences element-wise whatever their container, an absent key as
  null, and everything else exactly.
- The placement check had no test in either branch. `tests/test_placement.py` covers what
  counts as one placement and what does not, both framed and unframed.

## 0.3.0 - 2026-09-07

`Geometry` is one order now. It used to pack array-order quantities (`spacing_zyx`,
`shape_zyx`) beside world-order ones (`origin_xyz`, a row-major 3x3 `direction_xyz` of
cosines by column x, y, z) - SimpleITK's fields next to numpy's array - and every reader
and writer of it reversed and transposed at the boundary. It is duckn's form, which is
NRRD's: `shape` and `directions` per array axis in array order, each direction row a
world-space vector whose length is the spacing, and `origin` a world point; `spacing`,
`cosines`, the 4x4 `matrix` and `world()` are derived. `Geometry.aligned(shape, spacing,
origin)` is the axis-aligned case; `regrid` is the same orientation on another grid.
No bytes move in a store: its arrays already carried exactly these rows, and the reader
now takes them as they stand rather than normalizing into cosines and back. The frame's
`canonical` record changes to `{shape, directions, origin}`; a record with the old keys is
refused with a message that says so. Breaking for every constructor call and field read.

- More than 254 classes works on the GPU. `rank_dtype` widens the rank plane to uint16 at
  255 classes, and both kernels' label-table bounds check reduced that plane with
  `ranks.max()` - which torch implements on none of the three backends (`max_reduction_
  ushort_ushort` on MPS, `max_all not implemented for UInt16` on the CPU, `max_all_cuda ...`
  on CUDA). So the check raised before the kernel ran, on exactly the stores that need it. `backends.max_class()`
  widens a chunk at a time instead, which also keeps the temporary off a whole-body part's
  gigabyte. `tests/test_wide_ranks.py` covers the width from the encoder through the numpy
  decoders, every device, the store and back; the output dtype follows the LABELS and the
  rank dtype the CLASS COUNT, and the two are independent. Checked on Metal and, on a Modal
  A10, on CUDA: the Triton restore of a 300- and a 600-class store is the CPU's, voxel for
  voxel.

## 0.2.4 - 2026-09-07

Documentation and tests; no encoder or decoder change, and no bytes move.

- The rejected-remedy experiments in `docs/format.md` ran on the dense decoded field, where a
  class stored at no corner still reads `-clip` and can win. The restore never considers such
  a class, so those numbers described a different algorithm. Re-run through `restore()`: the
  rejection stands, but the reason given was wrong. The floor is load-bearing for CANDIDATES -
  a class stored at some corners of the stencil and absent at others - not for classes dropped
  everywhere, which cannot be the answer at all. The union remedy's cost was also understated,
  at 9.1 and 11.9 planes per voxel measured over a 6-neighbourhood where the rule says 26; it
  is 10.5 and 15.6.
- `reference_restore` is not candidate-restricted and is not the definition of `restore`. It
  interpolates the whole K-channel dense field, so a class stored nowhere can win it. The two
  disagree exactly there, which `tests/test_against_logits.py` now pins with the case that
  shows it. On real fields they agree at 0.0000 % of voxels, so the reference remains a check
  on the kernels - the test that compares them says so now, rather than implying it defines
  what the restore computes.

## 0.2.3 - 2026-09-07

- The depth cut could still drop the winner. 0.2.2 replaced the `1e6` shell offset with the
  logits' own range, which narrowed the failure without removing its cause: a large float32
  offset merges gaps that differ by less than its step, and the tie is broken by class index.
  One outlying voxel anywhere in the volume pushed the offset back to where the step is
  0.0625 - at junctions whose own class competition that voxel does not touch - and gaps
  under the step collapsed there regardless. The offset is a constant of the format now,
  `gap_range + 1`, with the key's gaps clamped at the range so no magnitude in the data can
  inflate it, and the winner is pinned below every key at `-inf`, which no step can reach.
  `ranks[0]` is the argmax unconditionally. Dropped entries now sort behind `inf` rather than
  behind the offset, which a kept shell class can exceed.

  Stored bytes are unchanged on shell, noisy, clip-rule and exhaustive fields. Encode
  allocates the same 3330.7 MB on the 60-class benchmark it did before, to the megabyte, and
  runs within noise of 0.2.2: the winner's index comes back from the `max` the encoder
  already takes, and the offset is applied in place over the one temporary the key was
  always going to need.

The review's second finding of this round - that the rejected-remedy experiments were run on
the dense decoded field, where a class stored at no corner can win, rather than through the
candidate restore, where it cannot - is NOT addressed here. Re-measured through `restore()`
the rejection stands (lowering the floor is ~3x worse on both fields), but the reason given
in `docs/format.md` is wrong, and the union remedy's plane counts are understated because
they were measured over a 6-neighbourhood where the rule says 26.

## 0.2.2 - 2026-09-07

Four defects a second adversarial review reproduced, and the documentation the review said
was writing cheques the code could not cash.

- The depth cut offset shell classes by `1e6`, where float32 steps by 0.0625. Gaps a
  thousandth of a logit apart collapsed into a tie with the winner, broken by class index -
  so at a junction of more shell winners than the depth holds, the winner itself could be
  dropped and a trailing class stored as `ranks[0]`, which breaks the identity restore. The
  offset is the logits' own range now, taken over the whole volume so `slab` still cannot
  move a byte. What the depth cannot hold is the farthest shell class.
- `keep="clip"` passed `gaps` itself as the selection key and the cut fills the unchosen
  with infinity in place, so the tail was measured over the kept classes alone and came out
  zero. The partition functions are taken before the cut.
- `exhaustive` was `depth >= classes`: capacity, not retention. The clip still dropped
  classes and an exhaustive field writes no tail to describe them, so probabilities read 1
  and 0 where the truth at T=4 was 0.905 and 0.095. A field with a plane per class now keeps
  every class; there is no byte to save by dropping one. This changes the bytes of an
  exhaustive field that had a class past the clip.
- `to_device()` left the extra-temperature tails behind while copying the meta that names
  them, so `tail_at()` raised on a field it had just been handed.

The review's other two findings are keep-rule decisions, not local defects, and stand:
`ranks[1]` is the nearest kept class rather than always the true runner-up, and a class kept
at one corner and dropped at the next still reads at the `-clip` floor and can win an
interpolation it should lose. Both are now documented, measured and pinned rather than
denied. `docs/format.md` gains "What the keep rule does not promise" with the label error by
depth, the runner-up disagreement rate, and two remedies measured and rejected - lowering the
floor is nearly 3x worse, and keeping the neighbours' kept set costs more planes than the
depth that matches it.

- `tests/test_against_logits.py` compares a restore against the ORIGINAL logits. Every other
  restore test compares it with a float64 reference restore of the same stored field, which
  shares the representation with what it checks and cannot see any of this.
- The specification was still titled 0.3; it describes 0.4's per-temperature tails and the
  new meta keys now, and says `exhaustive` means nothing was dropped.
- README: the store is no longer "in-memory only", numpy alone runs the decoders, and the
  restore is introduced as an approximation with a number attached.

## 0.2.1 - 2026-09-07

- `import rankfield` needed torch, which is the `torch` extra: `decode.py`, `encode.py` and
  `restore.py` imported it at module scope and `__init__` imports all three, so a default
  install (numpy only) raised `ModuleNotFoundError` and no numpy decoder was reachable.
  Those modules bind a proxy now that imports torch on the first attribute read and names
  the extra when there is nothing to import; annotations are postponed package-wide, so a
  `torch.Tensor` in a signature never resolves it. `margin()`, `deficit()`,
  `probabilities()`, `levels()` and the store reader run with no torch installed
  (`tests/test_no_torch.py` decodes a field in a child interpreter with torch blocked).
- The GPU backends resolved torch at import through `@torch.no_grad()`; `run_ranked` enters
  the context when it runs instead, and `available()` answers False when torch cannot be
  imported rather than raising.

## 0.2.0 - 2026-09-07

Format 0.4, and the store readable without haversack.

- `encode()` takes `tail_temperatures`. The default `(1.0,)` writes format 0.3's single tail
  plane and the same bytes as before; any other temperature is measured over the same slabs
  and kept in `RankField.tails`. A distillation at T needs the mass the encoding dropped at
  T, which the T=1 tail does not give it - at T=4 a torso store drops 3.4 % of the mass on
  average, a third of its voxels over 1 %.
- `tail_at()` serves a stored plane and refuses a temperature that was never written, rather
  than renormalizing with the wrong tail and misstating every probability; `probabilities()`
  takes the temperature to decode at.
- `rankfield.store` reads and writes the bare zarr v3 store - parts in paint order, geometry
  from duckn's own models on write, per-temperature tails beside the T=1 plane. Reading needs
  the `store` extra; writing also needs duckn, which is not on PyPI.
- `requires-python` is now `>=3.12`: the zarr floor the store reads against needs it, and the
  3.11 split had no solution.

The prerelease of this work read `tail_temperatures` in `encode()` without taking it as an
argument, so every call raised `NameError`. It is a parameter now, and `tests/test_encode.py`
covers both the default and a second temperature.

## 0.1.3 - 2026-09-05

- `levels()` refuses a non-positive `gap_range`, or `gap_origin` on the log curve, instead of
  building a table of zeros or NaN - what the JavaScript reader already refused.

## 0.1.2 - 2026-09-05

- `_select` (0.1.1's deterministic depth cut) allocated several (K, slab) temporaries and put
  a whole-body encode past a 16 GB laptop's GPU budget; it now works in place on the key
  with two bool masks and a class-order walk, same result (`tests/test_review_fixes.py`).

## 0.1.1 - 2026-09-05

Defects an adversarial review of 0.1.0 reproduced, all pinned by `tests/test_review_fixes.py`:

- `margin()` read a winner's sentinel byte (no stored runner-up) as level 0 - the curve's far
  end, 64 logits under the log byte - instead of the clip, and did not floor a shell class at
  `-clip`; 15 % of the voxels of a real field read 64 where the rendering field says 8. It now
  floors as the spec says, and `deficit()` keeps the true level the restore reads. The test
  that compared it with `decode_groups` used a field where neither path was reachable.
- Encoding was not reproducible across devices: `topk` selects arbitrarily among equal keys at
  the depth cut (`settle_ties` only orders what was selected), and the tail's mass summed in
  the device's reduction order. `_select` takes the lowest class indices at the cut and the
  tail sums in class order; cpu and mps now encode a tie-heavy field byte for byte.
- The Metal and Triton kernels indexed the label table without a length check (Metal returned
  garbage where the CPU raised); `restore` refuses a short or negative table up front.
- `max_tail` said 0 when the tail was not written (it is now `None`); `probabilities()` divided
  a missing `tail_max` by 255 instead of the uint16 quantum; two framed parts were assumed to
  share a placement without comparing their frames.

## 0.1.0 - 2026-09-05

Cut out of haversack's `ranked` module and its restore. Format 0.3: the shell keep rule and
the log byte over a range of 64, which remove the clip's bias (a 12th rib grew 19 %; now
0.4 %) at the bytes of the old encoding; levels from a table so every decoder agrees bit
for bit; ties to the stored winner; paint on the decision. See `docs/format.md`.
