# rankfield in one page

*Written for someone who ships a segmentation tool. Companion to `format.md`, which is the
specification; this is the vocabulary. Describes format 0.4, which every release since v0.2.0 writes.*

## The problem

A network's last layer is a field: one float per class per voxel. TotalSegmentator's
`total_fast` on a torso CT is 118 classes on a 3 mm grid - about 1.5 GB as dense fp16 - and
essentially all of it is discarded the moment `argmax` runs. What survives is a labelmap, on
the model's grid, at the interpolation the pipeline happened to pick. Everything else you
might want later - labels at another spacing, nearest instead of linear, a confidence gate, a
signed field to mesh, a probability for a downstream classifier - now means re-running the
network.

rankfield keeps, per voxel, only the classes that can still win and how far each trails the
winner, one byte each. The same torso is **2.08 MB**.

## What is stored

Per voxel, `depth` planes deep:

| term | meaning |
|---|---|
| **rank** | which class, ordered by how close it is. `ranks[0]` is the winner, so `ranks[0] - 1` is the labelmap. `0` means absent - the sentinel everywhere, so an unwritten chunk decodes as "nothing here". |
| **gap** (`support`) | how far rank *j* trails the winner, in logits. One byte, through the **level table**. The winner's gap is 0 by definition, so there is one fewer gap plane than rank plane. |
| **level table** | the 256-entry byte → logit map, computed from the header, identical in every decoder (numpy, torch, Metal, Triton, JS). Log-spaced by default: a few thousandths of a logit per step near 0, where boundaries are decided, a quarter of a logit out at 30, where nothing is. |
| **tail** | the softmax mass of everything *not* kept, as uint16. Without it the kept probabilities do not normalize. |

## The knobs

| term | default | what it does |
|---|---|---|
| **depth** | 6 | planes per voxel = how many classes can be kept. The one knob that trades size for accuracy. On real anatomy the rule below never needed more than 6 of 118. |
| **clip** | 8 logits | two jobs. At encode: a non-winner further behind than this is not worth keeping. At restore: the level a class reads at a corner where it is absent - an *upper* bound on how badly it trails, which is what lets a genuinely close class win. |
| **keep** | `shell` | which classes get the planes: (1) every class winning at this voxel **or any of its 26 neighbours**, at its true gap however far behind; then (2) the rest within `clip`, closest first. The neighbour part matters because an interpolation stencil's eight corners are mutual neighbours - so a class that wins anywhere in the cell is present, with a real number, at every corner. |
| **exhaustive** | derived | nothing was dropped, so no tail is needed to describe what was. |
| **mode** | `ranked` | softmax head. `regions` is the sigmoid-head form: one signed byte per region, 128 exactly on the decision boundary, no ranks and no tail. |

## What you get back

Two different fields, and the distinction is the one thing worth reading twice.

**deficit** `= l_c − max_j l_j`. Zero where *c* wins, negative behind. This is the logits up
to a per-voxel constant shared by every class - a gauge transformation - so interpolating it
and taking the argmax is *exactly* interpolating the logits and taking the argmax. **This is
the field a restore interpolates.**

**margin** `= l_c − max_{j≠c} l_j`. Positive inside by *c*'s lead, zero on *c*'s surface,
negative outside. **This is the field a renderer or a mesher wants**, and it is not
gauge-equivalent to the deficit. A group of classes has its own margin - `d_S − d_notS`, not
the max of its members' margins, which would put a surface at every internal boundary the
group was supposed to dissolve.

**probabilities(T)** gives `(class_ids, p)` at temperature *T*. The kept classes carry
`1 − tail` of the mass; you sum to one only by counting the tail as one more target.

**tail at temperature.** A distillation at *T* = 0.4 needs the dropped mass measured at 0.4,
and that cannot be recovered from the *T* = 1 plane. Temperatures are a **write-time**
decision - adding one later means re-encoding from logits - and `tail_at()` refuses a
temperature that was never written rather than substituting a neighbour, because
renormalizing with the wrong tail misstates every probability.

**restore** gives labels on any grid. Per output voxel: the candidates are the classes stored
at the eight surrounding corners; each candidate's deficit is blended with the trilinear
weights; the largest wins. The K-channel volume is never materialized. Torch, Metal and
Triton paths agree bit for bit (0 differing voxels in 52.5 M).

## Honest limits

It is an approximation, and the size of the error is measured rather than asserted.

- Real torso, restored to 1.5 mm against the labels that run's own live logits produced:
  **0.0037 % of voxels differ**, worst structure 12th rib **+0.40 %** by volume. On synthetic
  fields built to stress the rule (12 classes, heavy noise): 0.94 % at depth 6, 0.15 % at
  depth 12, the plateau past that being byte quantization.
- `ranks[1]` is **not** always the true runner-up - a shell class can evict a closer
  non-winner - so `margin()` reports a lead over the nearest *kept* class, an upper bound.
- A class kept at one corner and dropped at the next reads `−clip` there, above the truth, so
  it can win between corners. Lowering the floor makes this *worse*: the same floor is what
  carries the genuinely close candidates.
- A store written on a GPU is not byte-identical to one from a CPU. Ranks and gaps are decided
  by comparisons and are exact; the tail quantizes an `exp()`, which is the device's own.

The benchmark this still lacks is several real models, cases and spacings, each against its
own logits.
