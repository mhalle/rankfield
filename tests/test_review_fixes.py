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
    classes far behind: a 1-D ramp between two classes, a third class nowhere near."""
    n = 40
    x = torch.arange(n, dtype=torch.float32)
    la = 12.0 * (20.5 - x)                             # steep, no tie: the neighbor's gap is past the clip
    lb = -la
    lc = torch.full((n,), -200.0)
    return torch.stack([la, lb, lc]).reshape(3, 1, 1, n)


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
        for c in range(3):
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
