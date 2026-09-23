"""The encoder: what it stores, what it recovers, what zero means, and the shell rule."""
import numpy as np
import pytest
import torch
import torch.utils._python_dispatch

import rankfield as rf
from conftest import logits


def uniform02(lg, depth=6, clip=8.0):
    """Format 0.2's bytes: keep within the clip, uniform byte over the clip."""
    return rf.encode(lg, depth=depth, clip=clip, keep="clip", curve="uniform", gap_range=clip)


class TestPlanes:
    def test_rank_zero_is_the_argmax_and_ranks_store_class_plus_one(self, lg):
        code = rf.encode(lg, depth=3)
        np.testing.assert_array_equal(code.ranks[0].astype(np.int64) - 1, lg.argmax(0).numpy())
        assert code.meta["rank_sentinel"] == 0 and (code.ranks[0] != 0).all()

    def test_zero_means_absent_in_both_arrays_and_sentinels_form_a_suffix(self):
        code = rf.encode(logits(K=12), depth=4, clip=2.0)
        masked = code.ranks[1:] == 0
        assert masked.any()
        assert (code.support[masked] == 0).all()
        for j in range(2, code.depth):
            assert not ((code.ranks[j - 1] == 0) & (code.ranks[j] != 0)).any()

    def test_a_present_rank_never_sits_over_a_zero_support(self):
        code = rf.encode(logits(K=12), depth=6, clip=2.0)
        for j in range(1, code.depth):
            assert not ((code.ranks[j] != 0) & (code.support[j - 1] == 0)).any()

    def test_kept_entries_are_ordered_by_gap_so_rank_one_is_the_runner_up(self, lg):
        code = rf.encode(lg, depth=6)
        top = torch.topk(lg, 2, dim=0).indices.numpy()
        second = code.ranks[1].astype(np.int64) - 1
        live = code.ranks[1] != 0
        # wherever a runner-up is kept it is the true runner-up (or tied with it)
        d = lg.numpy()
        gap_true = np.take_along_axis(d, top[0:1], 0)[0] - np.take_along_axis(d, top[1:2], 0)[0]
        gap_kept = np.take_along_axis(d, top[0:1], 0)[0] - np.take_along_axis(d, second[None], 0)[0]
        assert np.allclose(gap_kept[live], gap_true[live], atol=1e-5)

    def test_the_block_states_its_format(self, lg):
        code = rf.encode(lg, depth=3)
        for k in ("version", "gap_unit", "gap_curve", "gap_range", "gap_origin", "keep", "tail_max"):
            assert k in code.meta
        assert code.meta["version"] == rf.FORMAT_VERSION and code.meta["keep"] == "shell"
        assert code.tail.dtype == np.uint16
        full = rf.encode(logits(K=4), depth=4)
        assert full.tail is None and full.meta["tail_max"] is None

    def test_slabbing_does_not_change_the_result(self):
        lg = logits(shape=(9, 8, 8))
        a, b = rf.encode(lg, depth=4, slab=2), rf.encode(lg, depth=4, slab=1000)
        np.testing.assert_array_equal(a.ranks, b.ranks)
        np.testing.assert_array_equal(a.support, b.support)
        np.testing.assert_array_equal(a.tail, b.tail)

    def test_it_refuses_input_that_is_not_logits(self):
        with pytest.raises(ValueError):
            rf.encode(torch.zeros(4, 5, 6))
        with pytest.raises(TypeError):
            rf.encode(torch.zeros(2, 3, 4, 5, dtype=torch.int32))


class LiveBytes(torch.utils._python_dispatch.TorchDispatchMode):
    """The peak bytes of the storages that ops inside the mode return, alive at once - counted
    per storage as ops return and as their last tensor dies, so it does not depend on the
    allocator, the machine or what else is running (process RSS does). It cannot see scratch
    a kernel allocates inside itself; exact for what the ops hand back, no more."""

    def __init__(self, *outside):
        super().__init__()
        self.outside = {t.untyped_storage().data_ptr() for t in outside}
        self.refs, self.size, self.live, self.peak = {}, {}, 0, 0

    def _drop(self, ptr):
        self.refs[ptr] -= 1
        if not self.refs[ptr]:
            del self.refs[ptr]
            self.live -= self.size.pop(ptr)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        import weakref
        from torch.utils._pytree import tree_leaves
        out = func(*args, **(kwargs or {}))
        for t in tree_leaves(out):
            if not isinstance(t, torch.Tensor):
                continue
            st = t.untyped_storage()
            ptr = st.data_ptr()
            if ptr in self.outside or st.nbytes() == 0:
                continue
            if ptr not in self.refs:
                self.refs[ptr], self.size[ptr] = 0, st.nbytes()
                self.live += st.nbytes()
                self.peak = max(self.peak, self.live)
            self.refs[ptr] += 1
            weakref.finalize(t, self._drop, ptr)
        return out


class TestSlab:
    def test_a_few_classes_on_a_wide_plane_get_a_thin_slab(self):
        """The 2026-09-23 failure: a 0.625 mm CTPA, K=5 (lung_vessels). The old rule budgeted
        the one-byte shell mask alone and took the whole volume as one slab (~10 GB of MPS
        pool at 40 Mvoxel); the slab must be sized by what a voxel really costs."""
        Z, Y, X = 152, 512, 512                                    # ~40 Mvoxel
        slab = rf.choose_slab((Z, Y, X), classes=5, depth=5, temperatures=1)
        assert slab <= 16                                          # well under Z
        assert rf.slab_bytes(slab, Y, X, 5, 5, 1) <= rf.DEFAULT_MEMORY_BUDGET
        assert rf.slab_bytes(slab + 1, Y, X, 5, 5, 1) > rf.DEFAULT_MEMORY_BUDGET   # the most that fits
        assert (1 << 28) // (5 * Y * X) >= Z                       # the old rule: all of it
        # a plane over the budget still encodes, one plane at a time
        assert rf.choose_slab((Z, Y, X), 118, 6, memory_budget=1) == 1
        assert rf.choose_slab((3, 8, 8), 5, 5) == 3                # never past the volume

    @pytest.mark.parametrize("K,depth,temps,keep", [
        (5, 6, (1.0,), "shell"), (2, 6, (1.0,), "shell"), (12, 4, (1.0, 2.0, 4.0), "shell"),
        (40, 6, (1.0,), "shell"), (12, 12, (1.0,), "shell"), (12, 6, (1.0,), "clip")])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_the_counted_peak_stays_within_the_bound(self, K, depth, temps, keep, dtype):
        """What the encoder's ops allocate, counted while alive: the peak is under slab_bytes
        for the slab, whatever the depth, tails, keep rule or dtype - and on a tiny plane,
        where the shell's padded winner maps outweigh the voxels (a 2x2 plane once ran 4 %
        over a bound that was per voxel only)."""
        N = min(depth, K)
        for Y in (2, 24):
            lg = logits(K=K, shape=(6, Y, Y), seed=K, noise=0.3).to(dtype)
            for slab in (1, 3):
                with LiveBytes(lg) as m:
                    rf.encode(lg, depth=depth, slab=slab, keep=keep, tail_temperatures=temps)
                bound = rf.slab_bytes(slab, Y, Y, K, N, 0 if N >= K else len(temps))
                assert 0 < m.peak <= bound, (Y, slab, m.peak, bound)

    def test_the_budget_is_kept_and_the_bytes_do_not_move(self):
        Z, Y, X = 12, 32, 32
        lg = logits(K=5, shape=(Z, Y, X), seed=3, noise=0.3)
        budget = rf.slab_bytes(3, Y, X, 5, 5, 0)                    # three planes' worth
        with LiveBytes(lg) as m:
            a = rf.encode(lg, memory_budget=budget)
        assert m.peak <= budget
        with LiveBytes(lg) as whole:
            b = rf.encode(lg, slab=Z)
        assert whole.peak > budget                                  # the test can fail
        np.testing.assert_array_equal(a.ranks, b.ranks)
        np.testing.assert_array_equal(a.support, b.support)
        assert a.meta == b.meta
        with pytest.raises(ValueError):
            rf.encode(lg, memory_budget=0)

    @pytest.mark.parametrize("slab", [0, -2, 1.5])
    def test_a_slab_that_is_not_a_positive_count_is_refused(self, slab):
        """0 used to die inside range(); -2 ran no slab and returned np.empty's garbage."""
        with pytest.raises(ValueError, match="slab"):
            rf.encode(logits(K=3, shape=(4, 5, 6)), slab=slab)


class TestShell:
    def test_a_neighboring_winner_is_kept_with_its_true_gap(self):
        """The rule the bias fix rests on: whatever wins at any of the 26 neighbors is
        present here, however far behind, so no stencil corner ever floors it."""
        lg = logits(K=10, shape=(8, 9, 9), noise=0.3)
        code = rf.encode(lg, depth=6, clip=8.0)
        win = lg.argmax(0).numpy()
        lv = rf.levels(code.meta)
        Z, Y, X = win.shape
        checked = 0
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    neigh = win[max(0, z - 1):z + 2, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
                    kept = {int(v) - 1: j for j, v in enumerate(code.ranks[:, z, y, x]) if v}
                    for c in np.unique(neigh):
                        assert int(c) in kept, (z, y, x, c)
                        j = kept[int(c)]
                        if j:
                            true_gap = float(lg[win[z, y, x], z, y, x] - lg[c, z, y, x])
                            got = float(lv[code.support[j - 1, z, y, x]])
                            assert abs(got - true_gap) <= max(0.02, 0.06 * true_gap) or true_gap > 64, (true_gap, got)
                            checked += 1
        assert checked > 50

    def test_the_shell_is_bounded_by_the_junction_and_fits_the_planes(self):
        """The largest shell is the number of winners meeting in a 3x3x3 block: 6 on real
        anatomy, more only on a random field like this one. Whatever it is, a depth of that
        size keeps every neighboring winner."""
        lg = logits(K=12, shape=(10, 12, 12))
        win = lg.argmax(0).numpy()
        Z, Y, X = win.shape
        worst = 0
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    worst = max(worst, len(np.unique(win[max(0, z - 1):z + 2, max(0, y - 1):y + 2, max(0, x - 1):x + 2])))
        assert worst <= 8
        code = rf.encode(lg, depth=worst)
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    neigh = set(np.unique(win[max(0, z - 1):z + 2, max(0, y - 1):y + 2, max(0, x - 1):x + 2]).tolist())
                    kept = {int(v) - 1 for v in code.ranks[:, z, y, x] if v}
                    assert neigh <= kept

    def test_format_02_bytes_are_the_special_case(self, lg):
        code = uniform02(lg)
        assert code.meta["keep"] == "clip" and code.meta["gap_curve"] == "uniform"
        lv = rf.levels(code.meta)
        np.testing.assert_allclose(lv, (1 - np.arange(256) / 255) * 8.0, atol=1e-6)
        masked = code.ranks[1:] == 0
        assert (code.support[masked] == 0).all()


class TestLevels:
    def test_the_table_is_monotone_and_spans_the_range(self):
        meta = rf.encode(logits(K=4), depth=3).meta
        lv = rf.levels(meta)
        assert lv[255] == 0.0 and abs(lv[1] - 64.0) < 2.0 and abs(lv[0] - 64.0) < 1e-3
        assert (np.diff(lv) < 0).all()
        assert lv[254] < 0.01                              # the quantum at the boundary
        assert lv[1] - lv[2] > 0.5                         # coarse out at the range

    def test_byte_of_gap_inverts_the_table(self):
        meta = rf.encode(logits(K=4), depth=3).meta
        lv = rf.levels(meta)
        s = np.arange(256)
        back = np.round(rf.byte_of_gap(lv, meta)).astype(int)
        assert (np.abs(back[1:] - s[1:]) <= 1).all()


class TestFields:
    def test_margin_is_positive_inside_negative_outside(self, lg):
        code = rf.encode(lg, depth=4)
        winner = lg.argmax(0).numpy()
        for c in range(int(lg.shape[0])):
            m = rf.margin(code, c)
            assert (m[winner == c] >= 0).all() and (m[winner != c] <= 0).all()

    def test_margin_recovers_the_true_gap_within_the_local_quantum(self, lg):
        code = rf.encode(lg.float(), depth=6, clip=8.0)
        top2 = torch.topk(lg.double(), 2, dim=0).values
        truth = (top2[0] - top2[1]).numpy()
        got = np.stack([rf.margin(code, c) for c in range(int(lg.shape[0]))]).max(0)
        lv = rf.levels(code.meta)
        quantum = np.abs(np.diff(lv)).max()            # coarse only far out
        near = truth < 8.0
        assert np.abs(got[near] - truth[near]).max() < quantum
        fine = truth < 1.0
        if fine.any():
            assert np.abs(got[fine] - truth[fine]).max() < 0.02

    def test_the_winners_margin_is_the_gap_to_the_runner_up(self):
        lg = torch.zeros(3, 1, 1, 1)
        lg[0] = 6.0
        assert abs(float(rf.margin(rf.encode(lg, depth=3), 0).ravel()[0]) - 6.0) < 0.1

    def test_probabilities_round_trip_against_a_direct_softmax(self):
        lg = logits(K=10)
        code = rf.encode(lg, depth=10)
        assert code.meta["exhaustive"] and code.tail is None
        ids, p = rf.probabilities(code)
        truth = torch.softmax(lg.double(), 0).numpy()
        live = ids >= 0
        got = np.take_along_axis(truth, np.where(live, ids, 0), axis=0)
        assert np.abs(p[live] - got[live]).max() < 0.02

    def test_the_tail_is_the_mass_of_everything_not_stored(self):
        lg = logits(K=12)
        code = rf.encode(lg, depth=3)
        ids, p = rf.probabilities(code)
        np.testing.assert_allclose(p.sum(0) + code.tail / 65535.0, 1.0, atol=2.0 / 65535)

    def test_the_default_encode_asks_for_the_one_temperature_of_format_0_3(self):
        code = rf.encode(logits(K=12), depth=3)
        assert code.meta["tail_temperatures"] == [1.0] and code.tails is None
        assert code.tail is not None and rf.tail_at(code, 1.0) is code.tail

    def test_a_second_temperature_measures_the_mass_dropped_at_that_temperature(self):
        lg, T = logits(K=12), 4.0
        code = rf.encode(lg, depth=3, tail_temperatures=(1.0, T))
        assert code.meta["tail_temperatures"] == [1.0, T]
        np.testing.assert_array_equal(code.tail, rf.encode(lg, depth=3).tail)  # T=1 bytes unchanged
        ids, _ = rf.probabilities(code)
        soft = torch.softmax(lg.double() / T, 0).numpy()
        live = ids >= 0
        kept = np.where(live, np.take_along_axis(soft, np.where(live, ids, 0), axis=0), 0.0).sum(0)
        np.testing.assert_allclose(rf.tail_at(code, T) / 65535.0 + kept, 1.0, atol=1e-3)

    def test_a_temperature_that_is_not_positive_is_refused(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="must be positive"):
                rf.encode(logits(K=6), depth=3, tail_temperatures=(bad,))

    def test_deficit_is_the_logits_up_to_a_per_voxel_constant(self):
        lg = logits(K=6)
        code = rf.encode(lg, depth=6, clip=40.0, gap_range=64.0)
        d = np.stack([rf.deficit(code, c) for c in range(6)])
        shift = lg.numpy() - d
        np.testing.assert_allclose(shift.max(0), shift.min(0), atol=0.3)

    def test_a_group_of_one_reproduces_the_numpy_margin_and_the_fast_path_matches(self):
        code = rf.encode(logits(K=10, shape=(5, 9, 11)), depth=4)
        groups = [[0], [1, 2], [3, 4, 5], [7]]
        fast = rf.decode_groups(code, groups)
        general = rf.decode_groups(code, groups + [[7, 8]])[:4]
        assert torch.equal(fast, general)
        np.testing.assert_allclose(fast[0].numpy(), rf.margin(code, 0), atol=1e-6)
        assert torch.equal(rf.decode_groups(code, groups, quantize=True),
                           rf.decode_groups(code, groups + [[7, 8]], quantize=True)[:4])

    def test_a_union_is_not_the_max_of_member_margins(self):
        """max() gets the sign right and underestimates the magnitude, so it would put a
        surface at every internal boundary."""
        code = rf.encode(logits(K=6), depth=6)
        union = rf.decode_groups(code, [[0, 1, 2]])[0].numpy()
        naive = np.maximum.reduce([rf.margin(code, c) for c in (0, 1, 2)])
        np.testing.assert_array_equal(union > 0, naive > 0)
        assert float((union - naive).max()) > 0.5
        assert float((union - naive).min()) >= -1e-5

    def test_internal_boundaries_vanish(self):
        code = rf.encode(logits(K=5), depth=5, clip=40.0, gap_range=64.0)
        a, b = rf.margin(code, 1), rf.margin(code, 2)
        union = rf.decode_groups(code, [[1, 2]])[0].numpy()
        internal = ((a > 0) & (np.roll(b, -1, axis=0) > 0))[:-1]
        assert internal.any()
        crossed = internal & ((union > 0) != (np.roll(union, -1, axis=0) > 0))[:-1]
        assert int(crossed.sum()) == 0

    def test_quantize_reserves_zero_and_puts_the_boundary_on_128(self):
        code = rf.encode(logits(K=5), depth=5)
        q = rf.decode_groups(code, [[1]], quantize=True)[0].numpy()
        assert q.min() >= 1
        m = rf.decode_groups(code, [[1]])[0].numpy()
        assert (q[m > 0.2] > 128).all() and (q[m < -0.2] < 128).all()


class TestTies:
    def test_ranks0_is_exactly_argmax_when_the_top_two_tie(self):
        lg = torch.zeros(5, 1, 1, 1)
        lg[3] = lg[1] = 7.0
        code = rf.encode(lg, depth=4)
        assert int(code.ranks[0].ravel()[0]) - 1 == 1

    def test_encoding_is_deterministic_over_tie_heavy_fp16(self):
        lg = logits(K=8).half().float()
        a, b = rf.encode(lg, depth=5), rf.encode(lg, depth=5)
        np.testing.assert_array_equal(a.ranks, b.ranks)
        np.testing.assert_array_equal(a.support, b.support)


class TestRegions:
    def test_overlapping_regions_are_all_kept_and_zero_is_the_boundary(self):
        lg = torch.full((3, 4, 5, 6), 2.0)
        code = rf.encode_regions(lg)
        for c in range(3):
            assert (rf.margin(code, c) > 0).all()
        at = rf.encode_regions(torch.full((2, 3, 3, 3), 1.5), threshold=1.5)
        assert float(rf.margin(at, 0).max()) == 0.0
        assert code.ranks is None and code.tail is None and code.meta["mode"] == "regions"

    def test_a_wide_plane_with_many_regions_gets_a_thin_slab(self):
        """The fixed slab=32 held K x 32 x Y x X voxels at once, at 13 bytes each as it was
        written: 118 regions on 512 x 512 was ~13 GB in one slab. The slab is sized now."""
        Z, Y, X = 152, 512, 512
        slab = rf.choose_region_slab((Z, Y, X), 118)
        assert slab < 32
        assert rf.region_slab_bytes(slab, Y, X, 118) <= rf.DEFAULT_MEMORY_BUDGET
        assert rf.region_slab_bytes(slab + 1, Y, X, 118) > rf.DEFAULT_MEMORY_BUDGET
        assert rf.choose_region_slab((Z, Y, X), 118, memory_budget=1) == 1
        assert rf.choose_region_slab((3, 8, 8), 2) == 3

    @pytest.mark.parametrize("K", [1, 3, 40])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_the_counted_peak_stays_within_the_bound(self, K, dtype):
        """The margin plane is rewritten in place and converted once: the count is exactly the
        bound, 5 bytes a voxel and class, on any plane and at any slab."""
        for Y, X in ((1, 1), (7, 5), (24, 24)):
            lg = (torch.randn(K, 6, Y, X, generator=torch.Generator().manual_seed(K)) * 20).to(dtype)
            for slab in (1, 3):
                with LiveBytes(lg) as m:
                    rf.encode_regions(lg, slab=slab)
                assert 0 < m.peak <= rf.region_slab_bytes(slab, Y, X, K), (Y, X, slab, m.peak)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_the_budget_is_kept_and_the_bytes_do_not_move(self, dtype):
        Z, Y, X, K = 12, 32, 32, 6
        lg = (logits(K=K, shape=(Z, Y, X), seed=5, noise=0.3) * 3).to(dtype)
        before = lg.clone()
        budget = rf.region_slab_bytes(3, Y, X, K)                   # three planes' worth
        with LiveBytes(lg) as m:
            a = rf.encode_regions(lg, threshold=0.4, memory_budget=budget)
        assert m.peak <= budget
        with LiveBytes(lg) as whole:
            b = rf.encode_regions(lg, threshold=0.4, slab=Z)
        assert whole.peak > budget                                  # the test can fail
        c = rf.encode_regions(lg, threshold=0.4, slab=5)            # a slab that leaves a remainder
        for other in (b, c):
            np.testing.assert_array_equal(a.support, other.support)
            assert a.meta == other.meta
        # and the bytes are the quantization as written out of place, before the in-place rewrite
        q = (((lg.float() - 0.4) / rf.CLIP).clamp(-1, 1) * (rf.SUPPORT_MAX - rf.ZERO_LEVEL)
             + rf.ZERO_LEVEL).round().clamp(1, rf.SUPPORT_MAX).to(torch.uint8).numpy()
        np.testing.assert_array_equal(a.support, q)
        assert torch.equal(lg, before)                              # the caller's logits untouched
        with pytest.raises(ValueError):
            rf.encode_regions(lg, memory_budget=0)

    @pytest.mark.parametrize("slab", [0, -2, 1.5])
    def test_a_slab_that_is_not_a_positive_count_is_refused(self, slab):
        """0 and -2 used to be quietly taken as 1."""
        with pytest.raises(ValueError, match="slab"):
            rf.encode_regions(logits(K=3, shape=(4, 5, 6)), slab=slab)
