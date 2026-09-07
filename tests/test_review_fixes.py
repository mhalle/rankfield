"""Defects an adversarial review of 0.1.0 reproduced (2026-09-05), pinned.

Each test is the review's minimal input and the corrected outcome.
"""
import numpy as np
import pytest
import torch

import rankfield as rf
from conftest import devices, logits


def _lone_and_shell():
    """A field with lone winners (nothing else within the clip, no shell member) and shell
    classes far behind: a 1-D ramp between two classes, two more nowhere near.

    Four classes against a depth of three on purpose. A field with a plane per class keeps
    every class (`exhaustive`), so it has no sentinel to read and no lone winner to find.
    """
    n = 40
    x = torch.arange(n, dtype=torch.float32)
    la = 12.0 * (20.5 - x)                             # steep, no tie: the neighbor's gap is past the clip
    lb = -la
    lc = torch.full((n,), -200.0)
    ld = torch.full((n,), -300.0)
    return torch.stack([la, lb, lc, ld]).reshape(4, 1, 1, n)


class TestMarginFloors:
    def test_a_winner_without_a_runner_up_leads_by_the_clip_not_the_range(self):
        code = rf.encode(_lone_and_shell(), depth=3)
        lone = (code.ranks[0] == 1) & (code.support[0] == 0)
        assert lone.any(), "the fixture must contain lone winners"
        m = rf.margin(code, 0)
        assert np.all(m[lone] == code.clip), "the sentinel byte is not a level"
        assert np.all(np.abs(m) <= code.clip), "the rendering field is bounded by the clip"

    def test_margin_and_decode_groups_agree_where_they_once_could_not(self):
        code = rf.encode(_lone_and_shell(), depth=3)
        for c in range(4):
            np.testing.assert_allclose(rf.decode_groups(code, [[c]])[0].numpy(), rf.margin(code, c), atol=1e-6)

    def test_deficit_keeps_a_shell_class_at_its_true_gap(self):
        code = rf.encode(_lone_and_shell(), depth=3)
        d = rf.deficit(code, 1)
        shell = (code.ranks[1] == 2) & (rf.levels(code.meta)[code.support[0]] > code.clip)
        assert shell.any(), "the fixture must store a shell class beyond the clip"
        assert np.all(d[shell] < -code.clip), "the restore field reads the true level"
        assert np.all(rf.margin(code, 1)[shell] == -code.clip), "the rendering field floors it"


class TestEncodingIsTheSameEverywhere:
    @pytest.mark.parametrize("dev", [d for d in devices() if d != "cpu"])
    def test_tie_heavy_logits_encode_identically_on_every_device(self, dev):
        lg = (logits(K=16, shape=(6, 9, 11), seed=3) * 4).round() / 4      # many exact ties
        a = rf.encode(lg, depth=6)
        b = rf.encode(lg.to(dev), depth=6)
        np.testing.assert_array_equal(a.ranks, b.ranks)
        np.testing.assert_array_equal(a.support, b.support)
        np.testing.assert_array_equal(a.tail, b.tail)

    def test_the_cut_takes_the_lowest_indices_among_equal_keys(self):
        lg = torch.zeros(6, 1, 1, 1)
        lg[0] = 1.0                                     # one winner, five tied runners-up
        code = rf.encode(lg, depth=3, keep="clip")
        assert list(code.ranks[:, 0, 0, 0]) == [1, 2, 3]

    def test_a_dropped_tail_is_recorded_as_unknown_not_zero(self):
        lg = logits(K=20, shape=(4, 6, 8))
        with_ = rf.encode(lg, depth=4)
        without = rf.encode(lg, depth=4, with_tail=False)
        assert with_.meta["max_tail"] > 0.0
        assert without.meta["max_tail"] is None and without.meta["tail_max"] is None


class TestProbabilities:
    def test_the_tail_quantum_is_uint16_when_the_meta_does_not_say(self):
        code = rf.encode(logits(K=12, shape=(4, 6, 8)), depth=4)
        _, stated = rf.probabilities(code)
        del code.meta["tail_max"]
        _, assumed = rf.probabilities(code)
        np.testing.assert_array_equal(stated, assumed)
        assert 0.5 < float(stated.sum(0).mean()) < 1.0      # the stored classes' share, tail aside


class TestRestoreRefuses:
    def _part(self, labels):
        code = rf.encode(logits(K=5, shape=(6, 7, 8)), depth=4)
        code.labels = labels
        code.geometry = rf.Geometry(spacing_zyx=(2.0, 2.0, 2.0), shape_zyx=(6, 7, 8))
        return rf.Part(field=code)

    @pytest.mark.parametrize("dev", devices())
    def test_a_short_label_table_is_an_error_not_an_out_of_bounds_read(self, dev):
        with pytest.raises(ValueError, match="label table"):
            rf.restore([self._part([10, 11, 12])], grid=1.5, device=dev)

    def test_a_negative_label_is_refused(self):
        with pytest.raises(ValueError, match="negative"):
            rf.restore([self._part([0, -3, 2, 3, 4])], grid=1.5)

    def test_two_framed_parts_must_share_the_source_frame(self):
        def framed(src_n, seed):
            code = rf.encode(logits(K=3, shape=(6, 6, 6), seed=seed), depth=3)
            code.labels = [0, 1, 2]
            code.geometry = rf.Geometry(spacing_zyx=(2.0, 2.0, 2.0), shape_zyx=(6, 6, 6))
            fr = rf.Frame(source=rf.Grid(shape=(src_n,) * 3, spacing=(1.0, 1.0, 1.0)),
                          model_shape=(6, 6, 6), model_spacing=(2.0, 2.0, 2.0), convention="corner",
                          canonical=rf.Geometry(spacing_zyx=(1.0, 1.0, 1.0), shape_zyx=(src_n,) * 3))
            code.frame = fr.to_meta()
            return rf.Part(field=code)
        with pytest.raises(ValueError, match="placement"):
            rf.restore([framed(12, 1), framed(40, 2)], grid=2.0)
        rf.restore([framed(12, 1), framed(12, 2)], grid=2.0)         # the same frame: fine


class TestLevelsRefuse:
    @pytest.mark.parametrize("meta", [
        {"clip": 8.0, "gap_curve": "sqrt"},
        {"clip": 8.0, "gap_curve": "log", "gap_range": 0.0},
        {"clip": 8.0, "gap_curve": "log", "gap_range": 64.0, "gap_origin": -1.0},
        {"clip": 8.0, "gap_range": -8.0},
    ])
    def test_a_block_that_describes_no_curve_is_refused(self, meta):
        with pytest.raises(ValueError):
            rf.levels(meta)


class TestSecondReview:
    """Defects a second adversarial review reproduced (2026-09-07), pinned.

    Its two remaining findings are not here: a dropped non-winner still reads at the
    ``-clip`` floor and can win an interpolation, and a shell class still outranks a closer
    true runner-up. Both are keep-rule decisions, not defects with a local fix.
    """

    def _junction(self, step, K=7):
        """K winners meeting at one voxel: the centre's own, and one per neighbour. The
        centre's trailing classes sit ``step`` logits apart, so the depth cut has to tell
        gaps of that size apart to keep the winner."""
        lg = np.full((K, 3, 3, 3), -5.0)
        for c in range(K):
            lg[c][1, 1, 1] = 0.0 if c == K - 1 else -step * (c + 1)
        for c, v in enumerate([(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]):
            lg[c][v] = 0.0
        return torch.tensor(lg, dtype=torch.float32)

    @pytest.mark.parametrize("step", [1.0, 0.1, 0.01, 0.001, 1e-4, 1e-8, 0.0])
    def test_the_depth_cut_keeps_the_winner_however_close_the_shell_crowds_it(self, step):
        """Offsetting the shell by a large float32 merged distinct gaps into a tie with the
        winner, which the cut then broke by class index - dropping the winner. 1e6 sits where
        float32 steps by 0.0625; the logits' own range only moved the step. The winner is
        pinned by -inf now, so no step can reach it."""
        lg = self._junction(step)
        code = rf.encode(lg, depth=6)
        stored = code.ranks[:, 1, 1, 1].astype(np.int64) - 1
        assert int(stored[0]) == int(lg[:, 1, 1, 1].argmax()), "ranks[0] is not the argmax"
        assert int(lg[:, 1, 1, 1].argmax()) in stored

    @pytest.mark.parametrize("step", [1e-4, 1e-8])
    @pytest.mark.parametrize("outlier", [1e6, -1e9])
    def test_one_outlying_voxel_does_not_cost_a_winner_anywhere_else(self, step, outlier):
        """The offset was the logits' own range for one release, which a single extreme voxel
        pushed back up to where float32 cannot separate the keys - at a voxel whose own class
        competition it does not touch. The offset is a constant of the format now."""
        lg = self._junction(step)
        lg[:, 0, 0, 0] += outlier
        code = rf.encode(lg, depth=6)
        assert int(code.ranks[0, 1, 1, 1]) - 1 == int(lg[:, 1, 1, 1].argmax())

    def test_a_kept_shell_class_past_the_range_still_outranks_a_sentinel(self):
        """The dropped entries were ordered behind the kept with the shell offset, which is
        smaller than a gap can be; they sort behind inf now, so sentinels stay a suffix."""
        n = 30
        x = torch.arange(n, dtype=torch.float32)
        far = torch.stack([100.0 * (15.5 - x), -100.0 * (15.5 - x)] +
                          [torch.full((n,), -v) for v in (9.0, 40.0, 300.0)])
        code = rf.encode(far.reshape(5, 1, 1, n), depth=3)
        for j in range(1, code.depth):
            assert not ((code.ranks[j - 1] == 0) & (code.ranks[j] != 0)).any()

    def test_the_shell_class_the_depth_cannot_hold_is_the_farthest_one(self):
        lg = self._junction(0.001)
        code = rf.encode(lg, depth=6)
        stored = set((code.ranks[:, 1, 1, 1].astype(np.int64) - 1).tolist())
        gaps = (lg[:, 1, 1, 1].max() - lg[:, 1, 1, 1]).numpy()
        dropped = set(range(7)) - stored
        assert len(dropped) == 1
        assert gaps[dropped.pop()] == gaps.max()

    def test_the_clip_rule_measures_the_mass_it_dropped(self):
        """``keep="clip"`` passed `gaps` itself as the selection key, and the cut fills the
        unchosen with inf in place - so the tail was taken over the kept classes alone and
        came out zero."""
        lg = torch.tensor([[0.0], [-1.0], [-2.0]], dtype=torch.float32)[:, None, None, :]
        code = rf.encode(lg, depth=1, keep="clip", curve="uniform", gap_range=8.0)
        truth = 1.0 - float(torch.softmax(lg[:, 0, 0, 0].double(), 0)[0])
        assert code.tail is not None
        np.testing.assert_allclose(code.tail.ravel()[0] / 65535.0, truth, atol=2.0 / 65535)

    def test_the_clip_and_shell_rules_agree_on_the_mass_they_dropped(self):
        lg = logits(K=10, shape=(4, 6, 8))
        a = rf.encode(lg, depth=4, keep="clip", curve="uniform", gap_range=8.0)
        b = rf.encode(lg, depth=4, keep="shell")
        assert float(a.tail.mean()) > 0.0
        assert abs(float(a.tail.mean()) - float(b.tail.mean())) / 65535.0 < 0.05

    def test_an_exhaustive_field_really_did_keep_every_class(self):
        """`exhaustive` was `N >= K` - capacity, not retention. The clip still dropped
        classes, and an exhaustive field writes no tail to describe them with."""
        lg = torch.tensor([[0.0, 0.0], [-9.0, -9.0]], dtype=torch.float32)[:, None, None, :]
        code = rf.encode(lg, depth=2)
        assert code.meta["exhaustive"] and code.tail is None
        assert (code.ranks != 0).all(), "an exhaustive field has no sentinel"
        for T in (1.0, 4.0):
            _, p = rf.probabilities(code, T)
            truth = torch.softmax(lg[:, 0, 0, 0].double() / T, 0).numpy()
            np.testing.assert_allclose(p[:, 0, 0, 0], truth, atol=0.01)

    def test_an_exhaustive_field_sums_to_one_without_a_tail(self):
        code = rf.encode(logits(K=5, shape=(4, 6, 8)), depth=5)
        assert code.meta["exhaustive"] and code.meta["max_tail"] == 0.0
        _, p = rf.probabilities(code)
        np.testing.assert_allclose(p.sum(0), 1.0, atol=1e-5)

    def test_to_device_carries_the_extra_temperature_tails(self):
        code = rf.encode(logits(K=8, shape=(3, 5, 6)), depth=4, tail_temperatures=(1.0, 4.0))
        moved = rf.to_device(code, "cpu")
        assert sorted(moved.tails) == sorted(code.tails) == [4.0]
        for T in moved.meta["tail_temperatures"]:
            assert rf.tail_at(moved, T) is not None
        np.testing.assert_array_equal(np.asarray(moved.tails[4.0]), code.tails[4.0])
