# rankfield format 0.3

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

Zero is the sentinel in every array. An unwritten chunk, a zeroed buffer, a forgotten fill
value all decode as "absent", the safe answer, and small integers keep the high byte
constant for a byte-shuffling compressor. Rank 0 is never the sentinel: every voxel has a
winner, so `ranks[0] - 1` is a labelmap with no special cases. Sentinels form a suffix:
once plane j is absent at a voxel, every deeper plane is too, and a reader may stop at the
first one.

Per part, the block that gives the bytes meaning:

| key | 0.3 value | meaning |
|---|---|---|
| `version` | `"0.3"` | readers refuse what they do not know |
| `mode` | `ranked` or `regions` | softmax head, or independent sigmoid heads (one signed byte per region, 128 on the boundary) |
| `classes`, `depth` | K, N | |
| `gap_unit` | `logit` | what gaps, levels, the clip and the range are measured in |
| `gap_curve`, `gap_range`, `gap_origin` | `log`, 64, 0.5 | the byte -> gap map, see below |
| `clip` | 8 | non-winners farther behind than this are not kept; the restore's floor |
| `keep` | `shell` | the keep rule, see below |
| `support_max`, `rank_sentinel`, `tail_max` | 255, 0, 65535 | constants of the format |
| `exhaustive`, `max_tail`, `rank_dtype`, `shape` | | facts of this part |

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

Kept entries are ordered by gap, so `ranks[1]` is the true runner-up wherever a runner-up
is kept; dropped entries are the sentinel and come last.

Why: an interpolation stencil's eight corners are mutual neighbours, so any class that
wins at one corner is present, with its gap, at every corner, and the restore never scores
it at the floor. Under the 0.2 rule a class dropped by the clip was floored at `-clip`, an
UPPER bound on its deficit, which over-credited it exactly where it mattered:

| encoding | voxels differing from the run's own labels | worst structure volume change | bytes |
|---|---|---|---|
| 0.2: keep within clip 8, uniform byte | 0.060 % | 12th rib +18.8 % | 1.96 MB |
| keep within clip 16, uniform byte | 0.0056 % | +2.0 % | 4.01 MB |
| **0.3: shell, log byte range 64, clip 8** | **0.0037 %** | **+0.4 %** | **2.08 MB** |

Depth is measured, not chosen: the shell held at most 6 classes on the torso (1.18 on
average) and never overflowed depth 6; depth 8 is the cap (an output voxel consults at most
8 corners), depth 4 dropped a neighbouring winner at 141 voxels and still lost nothing at
four decimals. Deeper planes are sentinel almost everywhere and cost bytes accordingly
(nothing) and restore time on the CPU (a half more at 8 than at 6).

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

*Bit-exactness.* The torch path, the Metal kernel and the Triton kernel make the same
decisions bit for bit: the same corner order, the same weight products, fused multiply-add
off, levels from the table. Measured on the torso (52.5 M output voxels): 0 differing
voxels between all three.

## What remains

Against the labels the pipeline wrote from its live logits, 0.3 differs at 0.0037 % of
voxels, within the byte quantum of a tie; against a float64 reference restore of the
stored field, at exact ties only. The truncation floor no longer decides a winner. What a
finer quantum near zero would buy is below what the tests can measure.

## Experiments

One `total_fast` run on the torso with its logits captured, re-encoded under every rule
and curve, each store restored to 1.5 mm and compared with the run's own labels
(per-structure Dice and volume change, differing-voxel count, bytes on disk). The scripts
lived in a scratch directory; the harness is `haversack`'s `segment_to_store` with the
encoder replaced, and is a day to rebuild from this description.
