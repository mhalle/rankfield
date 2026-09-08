# rankfield format 0.4

The rules, in one place. Everything below is measured on real anatomy (TotalSegmentator
`total_fast` on an IDC torso CT, 118 classes, 3 mm model grid, restored to 1.5 mm) unless it
says otherwise; the experiment scripts that produced the numbers are described at the end.

## What is stored

Per model voxel, for the `N` kept classes (`depth`):

| array | shape | dtype | meaning |
|---|---|---|---|
| `ranks` | (N, Z, Y, X) | uint8 (uint16 above 254 classes) | class + 1; **0 = not this class** |
| `support` | (N-1, Z, Y, X) | uint8 | how far rank j trails the winner, as a byte through the level table; **0 = absent** |
| `tail` | (Z, Y, X) | uint16, optional | probability mass of everything not kept, in 1/65535 |
| `tail_temperature_<T>` | (Z, Y, X) | uint16, optional | the same, at softmax temperature T (0.4) |

Zero is the sentinel in every array. An unwritten chunk, a zeroed buffer, a forgotten fill
value all decode as "absent", the safe answer, and small integers keep the high byte
constant for a byte-shuffling compressor. Rank 0 is never the sentinel: every voxel has a
winner, so `ranks[0] - 1` is a labelmap with no special cases. Sentinels form a suffix:
once plane j is absent at a voxel, every deeper plane is too, and a reader may stop at the
first one.

Per part, the block that gives the bytes meaning:

| key | 0.4 value | meaning |
|---|---|---|
| `version` | `"0.4"` | readers refuse what they do not know |
| `mode` | `ranked` or `regions` | softmax head, or independent sigmoid heads (one signed byte per region, 128 on the boundary) |
| `classes`, `depth` | K, N | |
| `gap_unit` | `logit` | what gaps, levels, the clip and the range are measured in |
| `gap_curve`, `gap_range`, `gap_origin` | `log`, 64, 0.5 | the byte -> gap map, see below |
| `clip` | 8 | non-winners farther behind than this are not kept; the restore's floor |
| `keep` | `shell` | the keep rule, see below |
| `support_max`, `rank_sentinel`, `tail_max` | 255, 0, 65535 | constants of the format |
| `tail_temperatures` | `[1.0]` | the temperatures a tail plane was written at (0.4) |
| `max_tail_at_temperature` | | the largest dropped fraction at each of those, same order (0.4) |
| `exhaustive`, `max_tail`, `rank_dtype`, `shape` | | facts of this part |

`exhaustive` says NOTHING WAS DROPPED, so no tail is needed to describe what was. A field with
a plane per class keeps every class to make that true: there is no byte to save by dropping
one, and dropping left a class the field had room for at the `-clip` floor. Before 0.4 the
flag was `depth >= classes` - capacity, not retention - and an exhaustive field could drop a
class past the clip and carry no tail to account for it.

A tail at temperature T is what a distillation at T needs and cannot recover from the kept
classes; the T=1 plane does not give it. Reading one back is `tail_at(code, T)`, which refuses
a temperature that was never written rather than substituting another - renormalizing with the
wrong tail misstates every probability. The temperatures are a write-time decision: adding one
later means re-encoding from logits.

Format 0.2 is the special case `keep = clip`, `gap_curve = uniform`, `gap_range = clip`.
Readers derive the level table from the block and read both.

## The level table

`levels[s]` is the gap a support byte `s` stands for. It is a 256-entry float32 TABLE,
computed once from the block, and every decoder - numpy, torch, the Metal and Triton
kernels, the JavaScript port - indexes the same values, so they agree bit for bit
whatever their division rounds like.

    uniform:  levels[s] = (1 - s/255) * range
    log:      levels[s] = origin * expm1((1 - s/255) * log1p(range / origin))

The log curve spends the byte unevenly: a few thousandths of a logit per step near zero,
where a boundary is decided, and a quarter of a logit out at 30, where nothing is. A
companded byte at a FIXED range buys almost nothing (sqrt and log shaved 15 % off the
differing voxels and 6-11 % off the bytes at range 16, and left the structure-level error
exactly where uniform had it); it is what makes a range of 64 affordable, and the range
is what the keep rule below needs.

## The keep rule: shell

Kept at a voxel, in this order, until the planes are full:

1. every class that wins at the voxel or at any of its 26 neighbours, with its true gap,
   however far behind it is here (byte 1 at least);
2. then the non-winners within `clip`, closest first.

Kept entries are ordered by gap; dropped entries are the sentinel and come last. `ranks[1]`
is the nearest KEPT class, which is the true runner-up only where the runner-up survived the
depth cut - see "What the keep rule does not promise" below.

Why: an interpolation stencil's eight corners are mutual neighbours, so a class that wins at
one corner is present, with its gap, at every corner the depth has room for, and the restore
does not score THAT class at the floor. Under the 0.2 rule a class dropped by the clip was
floored at `-clip`, an UPPER bound on its deficit, which over-credited it exactly where it
mattered:

| encoding | voxels differing from the run's own labels | worst structure volume change | bytes |
|---|---|---|---|
| 0.2: keep within clip 8, uniform byte | 0.060 % | 12th rib +18.8 % | 1.96 MB |
| keep within clip 16, uniform byte | 0.0056 % | +2.0 % | 4.01 MB |
| **0.3: shell, log byte range 64, clip 8** | **0.0037 %** | **+0.4 %** | **2.08 MB** |

Depth is measured, not chosen - on that case. The shell held at most 6 classes on the torso
(1.18 on average) and never overflowed depth 6; depth 8 is the cap for the stencil (an output
voxel consults at most 8 corners), depth 4 dropped a neighbouring winner at 141 voxels and
still lost nothing at four decimals. Deeper planes are sentinel almost everywhere and cost
bytes accordingly (nothing) and restore time on the CPU (a half more at 8 than at 6).

That is one anatomy at one spacing. A 27-voxel neighbourhood can hold 27 distinct winners,
and how close a real model comes to that is a property of the model, not of the format.

## What the keep rule does not promise

The shell rule narrows the floor's bias. It does not bound it. Three things are easy to read
into the rule above, and none of them hold.

**`ranks[1]` is not always the true runner-up.** Shell classes take the planes first, so a
closer non-winner can be evicted by a class that wins at a neighbour and trails badly here.
`margin()` then reports the lead over the nearest kept class, an upper bound on the true lead.

**The shell can overflow.** When more classes win in the neighbourhood than the depth holds,
one is dropped. It is the farthest shell class, never the winner. The cut ranks shell classes
under everything else by subtracting a constant of the format - `gap_range + 1`, with the
key's gaps clamped at the range first, so no logit magnitude in the data can inflate it - and
pins the winner below every key at `-inf`. Both matter: an offset large enough to be coarse in
float32 merges gaps that differ by less than its step, and a tie there is broken by class
index, which cost the winner its place when the offset was `1e6`, and again when it was the
logits' own range and one outlying voxel raised it.

**A dropped class still reads at the floor.** A class kept at one corner with its true gap and
dropped at the next reads `-clip` there, so the interpolation sees it far above the truth and
it can win between the corners. This is the 0.2 bias, narrowed to the corners where a class
crosses the clip rather than removed. With logits `A = [0, -40]`, `B = [-40, 0]`, `C = [-1,
-100]` on two adjacent voxels and enough other classes to fill the depth, the true
interpolated argmax 40 % along is A, and the restore returns C.

### Measured

Synthetic fields - blobby logits plus gaussian noise, 10x14x16, restored at 0.9 mm and
compared against interpolating the ORIGINAL logits. That last part is the point: comparing a
restore against a float64 reference restore of the same stored field, which is what the tests
do, shares the representation between both sides and cannot see any of this. A benchmark that
measures a restore against its own model's logits is the thing this repository still lacks.

Labels differing from the interpolated original logits:

| classes | noise | depth 6 | depth 8 | depth 10 | depth 12 | depth 16 |
|---|---|---|---|---|---|---|
| 12 | 1.5 | 0.937 % | 0.334 % | 0.159 % | 0.151 % | 0.151 % |
| 30 | 1.5 | 5.234 % | 1.796 % | 0.595 % | 0.285 % | 0.253 % |

The plateau is byte quantization, not the keep rule; depth cannot reach it. On smooth logits
(no noise, 12 classes) depth 6 already differs at 0.045 %.

`tests/test_against_logits.py` is this comparison, and its `label_error` helper reproduces
every figure in the table above from `conftest.logits(K, shape=(10, 14, 16), noise=1.5)`.

`ranks[1]` disagrees with the true runner-up at depth 6 nowhere on smooth logits, at 1.5 % of
voxels at 12 classes with noise and 9.3 % at 30. Where it disagrees the reported margin is off
by 0.49 logits on average at 12 classes and 0.61 at 30, worst cases 1.8 and 2.4, against a
field clipped at 8. Where it agrees the error is 0.01, which is the byte quantum. At depth 10
the disagreement is 0.04 %.

### Two remedies, measured and rejected

All of these run through `restore()`, on the two fields above.

| rule | 12 classes | 30 classes |
|---|---|---|
| current, depth 6 | 0.937 % at 5.7 planes | 5.234 % at 6.0 planes |
| lower the floor to -1000 | 2.546 % at 5.7 | 15.017 % at 6.0 |
| keep every class the neighbours kept | 0.155 % at 10.5 | 0.248 % at 15.6 |
| plain depth 10 | 0.159 % at 7.9 | 0.595 % at 9.9 |
| plain depth 12 | 0.151 % at 12.0 | 0.285 % at 11.8 |
| plain depth 16 | 0.151 % at 12.0 | 0.253 % at 15.1 |

*Lower the floor* so an absent class cannot win: **worse**, by 2.7x and 2.9x. The floor is
load-bearing - but not because a class dropped everywhere can win. Such a class is stored at
no corner, so the restore never considers it and it cannot be the answer at all. The floor
works for CANDIDATES: a class stored at some corners of the stencil and absent at others reads
`-clip` at the absent ones, and that upper bound is what lets a genuinely close class win
where it should. The same floor that hands a far class a win it should not have is what
carries these, which is why the bias cannot simply be lowered away.

*Keep every class the neighbours kept*, not just their winners: reaches 0.155 % and 0.248 %,
but needs 10.5 and 15.6 planes per voxel over the 26-neighbourhood the rule specifies. Plain
depth reaches the same error for the same or fewer planes - depth 10 and depth 16 in the
table. The rule change buys nothing that depth does not buy more cheaply.

So depth is the knob, and these are synthetic fields chosen to stress the rule rather than
anatomy. The torso figures earlier in this document are what the rule does on one real case.
What the repository still lacks is a benchmark over several real models, cases and spacings,
measured against their own logits.

## The two fields

`deficit_c = l_c - max_j l_j` (zero where c wins) is the logits up to a per-voxel constant
shared by every class - a gauge transformation - so interpolating it and taking the argmax
is exactly interpolating the logits and taking the argmax. **That is the field a restore
interpolates.**

`margin_c = l_c - max_{j != c} l_j` (positive inside by c's lead) is the signed field whose
zero level set is c's surface, with an interior gradient to shade and mesh. **That is the
field a renderer wants**, and it is not gauge-equivalent. The union of a group's members has
its own margin, `d_S - d_notS`, which is not the max of member margins (that would put a
surface at every internal boundary the group dissolves). Rendering floors levels at
`-clip`; the restore reads the true level.

## The restore

Per output voxel: the candidates are the classes stored at the eight corners; each
candidate's deficit is blended with the eight trilinear weights (zero where it won, minus
its level where it trailed, minus the clip where absent); the largest wins. Cells whose
eight corners share a winner take it outright.

*Ties.* Equal float32 sums - the byte quantum makes them - go to the candidate with more
winner mass (the weight of the corners where it is rank 0), then to the lower class index.
This is a deliberate deviation from argmax's first-maximal-index rule at a genuine tie of
interpolated logits: it makes the identity restore exact (a model voxel restores to its
stored winner) and is a no-op on real interpolated output.

*Paint.* Several parts composite in order; a part's DECISION background (class 0) is
transparent, whatever its label table maps class 0 to.

*Nearest* interpolation is the stored winner at the nearest model voxel, exact by
construction.

*Placement.* A part carries either a frame record (how the array was derived from a source
grid: source grid, crop, model shape, resample convention) and restores onto grids in the
source space exactly as the pipeline composed the mapping; or only its array geometry, and
restores onto grids in its own array frame (voxel 0 at 0 mm, true spacing). All parts of a
multi-part restore share one placement.

A geometry is duckn's, which is NRRD's: per array axis, in array order, a world-space
direction vector from one sample to the next (its length is the spacing), plus the world
position of sample (0, 0, 0). World is LPS millimeters. It is the same record the store's
arrays carry in their duckn attributes and the frame carries as ``canonical``; spacing and
direction cosines are derived from it, never stored beside it in another order.

*The float64 reference is not candidate-restricted.* `reference.py` builds the whole
K-channel dense field, in which a class stored at NO corner still reads `-clip` and can win the
argmax. `restore()` considers only the classes stored at the eight corners. The two implement
different rules and disagree exactly where a class absent from the entire stencil would have
won: with `A = [0, -40]`, `B = [-40, 0]`, `C = [-9, -9]` on two adjacent voxels at depth 2, the
true interpolated argmax 40 % along is C, the reference returns C, and the restore returns A.
On the fields behind the table above the two disagree at 0.0000 % of voxels, so the reference
is still a good check on the kernels - but it is not the definition of what the restore
computes, and a test that compares the two cannot tell the rules apart.

*Bit-exactness.* The torch path, the Metal kernel and the Triton kernel make the same
decisions bit for bit: the same corner order, the same weight products, fused multiply-add
off, levels from the table. Measured on the torso (52.5 M output voxels): 0 differing
voxels between all three.

## What remains

Against the labels the pipeline wrote from its live logits, 0.3 differs at 0.0037 % of
voxels, within the byte quantum of a tie; against a float64 reference restore of the
stored field, at exact ties only - that second comparison shares the representation with
what it checks, so it measures the kernels, not the encoding. On that case the truncation
floor decided no winner. It can: see "What the keep rule does not promise".
Reproduced through haversack's product path on 2026-09-05 (`haversack segment ... -o torso.duckn.zip`,
then `haversack restore --spacing 1.5` against `haversack segment --spacing 1.5`): 0.0037 % of
voxels, left 12th rib +0.40 %.

## Experiments

One `total_fast` run on the torso with its logits captured, re-encoded under every rule
and curve, each store restored to 1.5 mm and compared with the run's own labels
(per-structure Dice and volume change, differing-voxel count, bytes on disk). The scripts
lived in a scratch directory; the harness is `haversack`'s `segment_to_store` with the
encoder replaced, and is a day to rebuild from this description.
