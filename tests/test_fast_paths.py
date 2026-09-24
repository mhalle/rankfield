"""The encoder's fast paths store exactly what its general torch path stores (2026-09-23).

Two paths skip the general one (``encode._select_and_order``): the exhaustive shortcut - when
depth reaches K every class is kept, and one stable sort of the gaps gives the stored order -
and, on MPS, the Metal select kernel (``backends/metal_encode.py``). Both are held here to the
general path on the same device, byte for byte, over the fields where a difference would
hide: ties at the winner, at the depth cut and inside the shell (small-integer logits), fp16
input, every depth from 1 to K, both keep rules, extra tail temperatures, and slabs of one
plane. Each test first shows the path it names actually ran, so none of this passes vacuously.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

import rankfield as rf
from conftest import devices, logits

enc = sys.modules["rankfield.encode"]       # the module: `rankfield.encode` is the function

GPU = [d for d in devices() if d != "cpu"]


def ties(K, shape=(5, 7, 9), seed=0, top=3):
    """Small-integer logits: exact ties everywhere - with the winner, at the cut, in the shell."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, top, (K, *shape), generator=g).float()


FIELDS = {
    "blobby": lambda K: logits(K=K, shape=(5, 7, 9), seed=K),
    "blobby fp16": lambda K: logits(K=K, shape=(5, 7, 9), seed=K + 1, noise=0.3).half(),
    "ties": lambda K: ties(K, seed=K),
    "ties of two values": lambda K: ties(K, seed=K + 7, top=2),
    # gaps of exactly 0, 8 and 16 with clip 8: a gap AT the clip (kept only if it is under it)
    # and neighbors' winners beyond it (kept for the shell alone). Mutation found both
    # decisions unobserved by the fields above (2026-09-23).
    "ties at the clip": lambda K: ties(K, seed=K + 3) * 8.0,
    # gaps of 30, 70 and 100 with gap_range 64: shell keys past the range, where the clamp
    # makes 70 and 100 tie and the lower class index decides the cut
    "gaps past the range": lambda K: torch.tensor([0.0, 70.0, 100.0])[ties(K, seed=K + 5).long()],
}
NMAX = 16          # backends.metal_encode.NMAX: past it the torch path runs (a test says so)
CASES = [(K, d) for K in (1, 2, 3, 5, 8, 16, 20) for d in sorted({1, 2, 3, 6, K})
         if d <= K and d <= NMAX] + [(118, 6), (118, 16)]


def _arrays(code):
    out = {"ranks": np.asarray(code.ranks), "support": np.asarray(code.support)}
    if code.tail is not None:
        out["tail"] = np.asarray(code.tail)
    for t, a in (code.tails or {}).items():
        out[f"tail@{t}"] = np.asarray(a)
    return out


def _same(a, b, what):
    ea, eb = _arrays(a), _arrays(b)
    assert ea.keys() == eb.keys(), what
    for k in ea:
        np.testing.assert_array_equal(ea[k], eb[k], err_msg=f"{what}: {k}")
    assert a.meta == b.meta, what


def _general(monkeypatch, lg, **kw):
    """The same encode with every fast path off."""
    with monkeypatch.context() as m:
        m.setattr(enc, "EXHAUSTIVE_SHORTCUT", False)
        m.setattr(enc, "METAL_ENCODE", False)
        return rf.encode(lg, **kw)


class _Count:
    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, *a, **k):
        self.n += 1
        return self.fn(*a, **k)


# -- the exhaustive shortcut, on every device -------------------------------------------

@pytest.mark.parametrize("dev", devices())
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("K", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("keep", ["shell", "clip"])
def test_the_exhaustive_shortcut_stores_what_the_general_path_stores(monkeypatch, dev, field,
                                                                    K, keep):
    lg = FIELDS[field](K).to(dev)
    kw = dict(depth=K + 1, keep=keep)                        # depth past K: exhaustive
    general = _general(monkeypatch, lg, **kw)
    spy = _Count(enc._select_and_order)
    with monkeypatch.context() as m:
        m.setattr(enc, "_select_and_order", spy)
        m.setattr(enc, "METAL_ENCODE", False)               # the shortcut itself, not the kernel
        fast = rf.encode(lg, **kw)
    assert spy.n == 0, "the general path ran: the shortcut was not taken"
    assert fast.meta["exhaustive"]
    _same(fast, general, f"{dev} {field} K={K} {keep}")


# -- the Metal kernel ---------------------------------------------------------------------

@pytest.mark.parametrize("dev", GPU if "mps" in GPU else [pytest.param("mps", marks=pytest.mark.skip("no MPS"))])
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("K,depth", CASES)
def test_the_metal_kernel_stores_what_the_general_path_stores(monkeypatch, dev, field, K, depth):
    if dev != "mps":
        pytest.skip("the kernel is Metal")
    from rankfield.backends import metal_encode
    lg = FIELDS[field](K).to(dev)
    kw = dict(depth=depth, tail_temperatures=(1.0, 4.0))
    general = _general(monkeypatch, lg, **kw)
    spy = _Count(metal_encode.select)
    with monkeypatch.context() as m:
        m.setattr(metal_encode, "select", spy)
        fast = rf.encode(lg, **kw)
    assert spy.n > 0, "the kernel did not run"
    _same(fast, general, f"{field} K={K} depth={depth}")


@pytest.mark.skipif("mps" not in GPU, reason="the kernel is Metal")
@pytest.mark.parametrize("slab", [1, 2, 1000])
def test_the_kernel_does_not_care_how_the_volume_is_slabbed(monkeypatch, slab):
    lg = ties(8, shape=(6, 11, 13), seed=3).to("mps")
    _same(rf.encode(lg, depth=4, slab=slab), _general(monkeypatch, lg, depth=4, slab=slab),
          f"slab {slab}")


@pytest.mark.skipif("mps" not in GPU, reason="the kernel is Metal")
def test_the_kernel_agrees_with_the_cpu_on_tie_heavy_fields():
    """The reason the torch path settles ties by index at all: the bytes must not depend on
    the device. The kernel is a third implementation of the same rule."""
    for seed in range(4):
        lg = ties(12, shape=(6, 8, 10), seed=seed)
        cpu, mps = rf.encode(lg, depth=5), rf.encode(lg.to("mps"), depth=5)
        np.testing.assert_array_equal(cpu.ranks, mps.ranks)
        np.testing.assert_array_equal(cpu.support, mps.support)


@pytest.mark.skipif("mps" not in GPU, reason="the kernel is Metal")
def test_a_depth_past_the_kernels_registers_takes_the_torch_path(monkeypatch):
    from rankfield.backends import metal_encode
    assert metal_encode.NMAX == NMAX
    lg = logits(K=40, shape=(3, 5, 6)).to("mps")
    spy = _Count(metal_encode.select)
    monkeypatch.setattr(metal_encode, "select", spy)
    code = rf.encode(lg, depth=metal_encode.NMAX + 1)
    assert spy.n == 0 and code.ranks.shape[0] == metal_encode.NMAX + 1


@pytest.mark.skipif("mps" not in GPU, reason="the kernel is Metal")
def test_the_clip_rule_stays_on_torch(monkeypatch):
    """Format 0.2's rule has no shell and no pinned winner; the kernel is the shell rule's."""
    from rankfield.backends import metal_encode
    spy = _Count(metal_encode.select)
    monkeypatch.setattr(metal_encode, "select", spy)
    rf.encode(logits(K=8).to("mps"), depth=4, keep="clip")
    assert spy.n == 0
