# Changelog

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
