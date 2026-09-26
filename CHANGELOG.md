# Changelog

## Unreleased

- **A world restore between grids that line up takes the fast path again.** Two oblique grids of
  one orientation compose to off-diagonal terms of ~1e-17, which `Affine.separable` read as a
  rotation: the per-axis path and its kernels were skipped (~40x slower on MPS). Terms below
  `SEPARABLE_TOLERANCE` (1e-9 of the diagonal) are float noise now.
- **Documented: nearest under a flip differs at exact half-sample ties** (the per-axis rule rounds
  up in array order); linear is exact. Tests now cover an roi on a world grid at a spacing other
  than 1 mm and several parts painting through the general path (review, 2026-09-26).

## 0.3.7 - 2026-09-25

- **Restore onto a world geometry.** `restore(parts, grid=<Geometry>)` puts an unframed part's
  labels on any world grid through its own array geometry: output index -> world -> stored
  array, a general affine (`rankfield.Affine`, built by `Affine.between(geo_from, geo_to)`).
  This is how a field on a grid its model resampled in world space - FastSurfer's conformed
  1 mm grid - comes back onto an oblique input, which no per-axis `Mapping` can describe.
  When the two grids line up the map IS per axis and the restore is the ordinary one, fused
  kernels included; otherwise coordinates are computed per output voxel on the torch path,
  with every decision (the inside test, the edge clamp, corners, weights, the deficit) the
  per-axis path's. Held to it bit for bit where both apply, and exactly under flips and axis
  swaps with binary-exact coordinates. A framed part refuses a world grid: its frame is the
  exact rule its model grid was made by. `roi_of(..., world=)` bounds a structure on one.
  No stored byte changes.

## 0.3.6 - 2026-09-23

The encoder's selection moves off `topk` and `sort` where they were slow. No bytes move: both
new paths are held to the torch path byte for byte, and it stays as the reference.

- **On MPS, a Metal kernel selects and orders the kept classes, a thread per voxel**
  (`backends/metal_encode.py`, the shell rule, depth up to 16). Profiling a K=5 field
  (29 Mvoxels, `lung_vessels` on a 0.625 mm CTPA) on an M2 put 9 of its 13.5 s in three
  calls: `topk` twice in `_select` and the stable `sort`, each over millions of rows of five
  classes - 0.49, 0.46 and 0.38 s on the same machine's CPU, and no faster with the class
  axis last. The kernel makes the torch path's decisions from the same numbers: the key
  `min(gap, gap_range)` less `big` in the shell (one IEEE add), the winner first, the N
  smallest keys with the lowest index taking a tie at the cut, kept = winner, shell or under
  the clip, and the kept ordered by (true gap, class index). `exp`, the gap byte and the
  tail stay on torch. K=5: 13.2 s -> 0.74 s (x17.7); K=118 on 0.8 Mvoxels: 0.65 -> 0.15 s.
  Through haversack the fine stage's stored arrays were identical to 0.3.5's.
- **When depth reaches K every class is kept, and one stable sort of the gaps is the whole
  selection**: the torch path's order then comes down to (true gap, class index). It skips
  the shell, the key, `_select`, both `settle_ties` and the second sort, on every device:
  K=5 above on the CPU 8.2 -> 1.09 s, on an A10 1.12 -> 0.49 s.
- `EXHAUSTIVE_SHORTCUT` and `METAL_ENCODE` turn the two paths off; `tests/test_fast_paths.py`
  compares each with the torch path (after showing it ran) over tie-heavy fields - small
  integer logits, gaps exactly at the clip, shell gaps past the range - every depth to K,
  both keep rules, extra tail temperatures and one-plane slabs. 12 mutants: 9 killed; the
  three survivors are equivalent (the center voxel's winner, the order of dropped entries)
  or unobservable here (an unstable sort that happens to be stable on these rows).
- A slab now costs the kernel path a fixed overhead that the torch path's selection hid:
  one plane a slab took 2.09 s against 0.75 s at the 1 GiB default. A caller passing a
  tiny `memory_budget` pays that.
- Found, not changed: support can differ by one step between CUDA and the CPU (3 of 116 M
  bytes on the field above, identically in 0.3.5), because `byte_of_gap` evaluates the log
  curve with each device's `log` - 4 of a million random gaps round differently. Ranks
  match; MPS matched the CPU.

## 0.3.5 - 2026-09-23

Two defects found running a 0.625 mm CTPA through `lung_vessels` (K=5) on a 16 GB M2, and the
same slab fix carried to the regions encoder. No bytes move: every change is to how much
memory the encoders take and what the host decoders accept.

- `encode()` sized its slab for the one-byte shell mask alone (~256 MB of it), so at small K
  the slab was the whole volume - and a voxel of a slab really costs ~17 bytes a class plus
  ~24-46 per kept plane (fp32 gaps, key and exp temporaries; int64 selection, sort and gather
  indices). A 40-Mvoxel K=5 field asked for ~10 GB of MPS pool in one slab and ran out.
  The slab is now the most z planes whose `slab_bytes(planes, rows, columns, K, depth,
  temperatures)` fits `memory_budget` bytes. That bound was fitted by counting the tensors the
  encoder's ops return, alive at once, and checked over 1,104 configurations (planes down to
  1x1, slabs of 1-3, both keep rules, fp16 and fp32): the peak is 0.05-0.86 of it, 0.20-0.86 on
  planes of 16x16 and up. The count cannot see scratch a kernel allocates inside itself, nor
  on a GPU the per-slab copies to the host, so the budget is working memory as the ops define
  it, not a promise about a device pool.
- `memory_budget` defaults to 1 GiB: a small footprint, not a measurement of any machine. A
  caller that knows its device's budget should pass it. At the default, large-K fields take
  thinner slabs than before (whole-body K=118 at 236x167x167: 81 planes -> 15), which is more
  per-slab launches and ~13 % more halo work; the throughput cost has not been measured.
- The K-sized planes are freed as soon as the selection has read them, so the (N, slab) phase
  no longer sits on top of them: 315 -> 242 B/voxel at K=5, 460 -> 404 at K=12; at K=118
  (2,174 -> 2,158) the peak is the K phase itself.
- `slab=` must be a positive integer. 0 used to fail inside `range()`, and a negative slab ran
  no slab at all and returned the uninitialized planes as a field; both now raise
  `ValueError`. A given `slab` is taken as is, without consulting `memory_budget`.
- New: `choose_slab`, `slab_bytes`, `DEFAULT_MEMORY_BUDGET`.
- `encode_regions()` had a fixed `slab=32`, whatever K and the plane, and held 13 bytes per
  voxel and region at its peak: the fp32 margin plus a new temporary for each step of the
  quantization (division, clamp, scale, offset, round, clamp) and the uint8 plane. At 118
  regions on 512 x 512 that is ~13 GB in one slab. The margin is now rewritten in place, so
  the peak is exactly 5 bytes (the fp32 plane and the uint8 plane). The bytes do not change:
  it is the same operations in the same order, checked against the out-of-place form on the
  CPU and MPS. It takes `memory_budget` (the same 1 GiB default) and sizes its slab with
  `choose_region_slab(shape, classes, memory_budget)`, the most planes whose
  `region_slab_bytes(planes, rows, columns, classes)` fits. That bound is 5 bytes per voxel
  and region and has no fixed term. It was counted as `encode`'s was, over K 1-118, planes
  1x1 to 64x64, slabs of 1 and 3, and fp32, fp16 and bf16 input, and the count equals it every
  time. It has the same blind spot: on a GPU, the per-slab host copy is not counted. A given
  `slab` must be a positive integer, as in `encode`; 0 and negative values used to be taken
  as 1 without a word.
- `margin()`, `deficit()` and `probabilities()` refused a field from `store.read_parts()`: its
  planes are lazy zarr arrays, and the host check read `.device` off them (`AttributeError`).
  They now refuse only torch-tensor planes (any of ranks, support, tail, by name) and read
  anything else that indexes like numpy. `margin`/`deficit` read each plane once per call -
  `support[0]` used to be read twice and `ranks[0]` again by `deficit` - which on a zarr array
  is a decompression per read; to decode many channels, read the planes once first.
- The field's shape comes from the planes: haversack's stores record no `shape`, which was
  the `KeyError: 'shape'` behind the first error (`decode_groups()` had it too). A meta
  `shape` that disagrees with the planes is refused with both shapes named, where it used to
  fail as a mismatched boolean index.

## 0.3.4 - 2026-09-22

- The house spelling is American, and a test holds it there: `tests/test_american_spelling.py`
  (ported from haversack) scans every tracked text file for British forms, in prose and in
  identifiers. The 28 that had drifted into the docs, docstrings and tests are respelled. No
  public name, stored key or format field changes; no bytes move.

## 0.3.3 - 2026-09-22

- duckn is pinned at `v0.5.1` (seg extension 0.9). Nothing here changes: the store writer uses
  duckn's geometry models and the `ranked` extension block, which 0.4.0 leaves as they were,
  and the store tests pass against it.
- The README names no release in its install line; `TAG` is a release tag, listed by
  `git ls-remote --tags`, so the page is not wrong by the next release.

## 0.3.2 - 2026-09-08

Tooling and one corrected claim. No encoder or decoder change; no bytes move.

- `tools/cuda_check.py` runs the suite on a CUDA box through Modal, mounting the working
  tree rather than a tag. The Triton kernel is the one path no machine here can execute, and
  it is where a defect survives longest - the uint16 bounds check fixed in 0.3.0 was dead on
  CUDA too, which only a GPU run showed.
- Its first run found that the tail plane differs between the CPU and CUDA. Ordering the sum
  removes a backend's reduction order, which is what 0.1.1 fixed, but `exp()` is the device's
  own: CUDA's differs from the CPU's by a float32 ulp on about a third of a sampled range, so
  a value on a rounding boundary lands one unit either side. Measured at 1 voxel in 594 on a
  tie-heavy field and identical at 0.2.1, 0.2.2, 0.2.4 and 0.3.1, so it is not a regression -
  it was never visible because no CUDA machine ran the tests. It is also within spec, not a
  torch defect: IEEE 754 requires correct rounding only for `+ - * /` and `sqrt`, and CUDA
  documents `expf` at a maximum error of 2 ULP on the multi-function unit's hardware `exp2`.
  The measured difference is half a float32 ULP, inside that budget. `docs/format.md` carries
  the citations. The RANK and SUPPORT planes,
  decided by comparisons, are exact everywhere. The encoder's comment, `docs/format.md` and
  the test now say that rather than claiming a store's bytes cannot depend on where they were
  written. One unit is 1/65535 of the dropped mass, far under the gap byte's own quantum, but
  a store written on a GPU is not byte-identical to one written on a CPU.
- `duckn` is a dependency group with the tag haversack pins, so a clean `uv sync --extra test`
  runs the four store-writer tests instead of skipping them. It is a group, not part of the
  `test` extra, because a git source in published metadata would make `pip install
  rankfield[test]` unresolvable.
- README pinned v0.2.1, six releases stale.

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
  at 9.1 and 11.9 planes per voxel measured over a 6-neighborhood where the rule says 26; it
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
they were measured over a 6-neighborhood where the rule says 26.

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
floor is nearly 3x worse, and keeping the neighbors' kept set costs more planes than the
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
