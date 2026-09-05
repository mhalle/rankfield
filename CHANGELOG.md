# Changelog

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
