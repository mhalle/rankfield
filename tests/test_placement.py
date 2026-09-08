"""One placement per multi-part restore: what counts as the same, and what does not.

The check had no test in either branch. Two UNFRAMED parts were compared with a tolerance
on origin and directions; two FRAMED parts with `==` on the nested record - the same
geometric question answered two ways, and the strict way was the one handling the record
with the most chances to differ innocently. Parts being composited can derive their frame
independently from the same input, so a last-bit difference, or a tuple where `to_meta`
emits a list, is the same placement.
"""
import copy
import math

import pytest

import rankfield as rf
from rankfield.restore import _same_record
from conftest import logits

SHAPE = (6, 10, 12)
GEO = rf.Geometry.aligned(SHAPE, (2.0, 2.0, 2.0), origin=(1.0, -2.0, 5.0))


def frame_meta():
    return rf.Frame(source=rf.Grid(shape=SHAPE, spacing=(2.0, 2.0, 2.0)),
                    model_shape=SHAPE, model_spacing=(2.0, 2.0, 2.0),
                    convention="corner", canonical=GEO,
                    original_orientation="LPS").to_meta()


def part(name, *, frame=None, geometry=GEO, seed=0):
    code = rf.encode(logits(K=5, shape=SHAPE, seed=seed), depth=4)
    code.labels = [0, 1, 2, 3, 4]
    code.geometry = geometry
    code.frame = frame
    return rf.Part(field=code, name=name)


def restores(a, b, grid="input"):
    try:
        rf.restore([a, b], grid=grid)
        return True
    except ValueError as e:
        if "one placement" not in str(e):
            raise
        return False


class TestFramedPartsTolerateWhatDoesNotMatter:
    def test_the_same_record_composites(self):
        m = frame_meta()
        assert restores(part("a", frame=m), part("b", frame=m, seed=1))

    def test_a_number_off_by_one_ulp_is_the_same_placement(self):
        m, m2 = frame_meta(), copy.deepcopy(frame_meta())
        m2["source"]["spacing"][0] = math.nextafter(m2["source"]["spacing"][0], math.inf)
        assert m2 != m                                            # exact equality would refuse
        assert restores(part("a", frame=m), part("b", frame=m2, seed=1))

    def test_a_tuple_where_the_writer_emitted_a_list_is_the_same_placement(self):
        m, m2 = frame_meta(), copy.deepcopy(frame_meta())
        m2["source"]["shape"] = tuple(m2["source"]["shape"])
        m2["canonical"]["directions"] = tuple(tuple(r) for r in m2["canonical"]["directions"])
        assert m2 != m
        assert restores(part("a", frame=m), part("b", frame=m2, seed=1))

    def test_an_omitted_optional_field_matches_one_written_as_null(self):
        m, m2 = frame_meta(), copy.deepcopy(frame_meta())
        assert m2["model_source"] is None
        del m2["model_source"]
        assert restores(part("a", frame=m), part("b", frame=m2, seed=1))

    def test_the_per_part_model_grid_is_allowed_to_differ(self):
        m, m2 = frame_meta(), copy.deepcopy(frame_meta())
        m2["model_shape"], m2["model_spacing"] = [3, 5, 6], [4.0, 4.0, 4.0]
        assert restores(part("a", frame=m), part("b", frame=m2, seed=1))


class TestFramedPartsStillRefuseWhatDoes:
    @pytest.mark.parametrize("mutate", [
        pytest.param(lambda m: m["source"]["spacing"].__setitem__(0, 3.0), id="a different spacing"),
        pytest.param(lambda m: m["source"]["origin"].__setitem__(2, 40.0), id="a different crop origin"),
        pytest.param(lambda m: m["source"]["shape"].__setitem__(1, 99), id="a different source shape"),
        pytest.param(lambda m: m.__setitem__("convention", "center"), id="a different convention"),
        pytest.param(lambda m: m.__setitem__("original_orientation", "RAS"), id="a different orientation"),
        pytest.param(lambda m: m["canonical"]["origin"].__setitem__(0, 9.0), id="a different world origin"),
    ])
    def test_a_real_difference_is_refused(self, mutate):
        m, m2 = frame_meta(), copy.deepcopy(frame_meta())
        mutate(m2)
        assert not restores(part("a", frame=m), part("b", frame=m2, seed=1))

    def test_a_framed_part_does_not_composite_with_an_unframed_one(self):
        assert not restores(part("a", frame=frame_meta()), part("b", seed=1))


class TestUnframedPartsAreUnchanged:
    def test_the_same_array_placement_composites(self):
        assert restores(part("a"), part("b", seed=1), grid=2.0)

    def test_a_different_array_placement_is_refused(self):
        other = rf.Geometry.aligned(SHAPE, (2.0, 2.0, 3.0), origin=(1.0, -2.0, 5.0))
        assert not restores(part("a"), part("b", geometry=other, seed=1), grid=2.0)


class TestTheComparisonItself:
    @pytest.mark.parametrize("a, b, same", [
        ({"x": [1.0, 2.0]}, {"x": (1.0, 2.0)}, True),
        ({"x": 1.0}, {"x": 1}, True),
        ({"x": None}, {}, True),
        ({"x": [1.0, 2.0]}, {"x": [1.0, 2.0, 3.0]}, False),
        ({"x": "corner"}, {"x": "center"}, False),
        ({"x": True}, {"x": 1}, False),                  # bool is an int; do not conflate them
        ({"x": False}, {"x": 0}, False),
        ({"x": {"y": 1.0}}, {"x": 1.0}, False),
        ({"x": float("nan")}, {"x": float("nan")}, False),
    ])
    def test_the_shapes_and_types_it_distinguishes(self, a, b, same):
        assert _same_record(a, b) is same

    def test_it_tolerates_float_noise_and_not_a_real_step(self):
        assert _same_record([1.5], [math.nextafter(1.5, math.inf)])
        assert not _same_record([1.5], [1.5001])
        assert _same_record([0.0], [0.0])
