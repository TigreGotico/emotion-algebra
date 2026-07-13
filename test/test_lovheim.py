"""Tests for Lövheim's neurochemical cube."""
import numpy as np
import pytest

from emotion_algebra.emotions import get_emotion
from emotion_algebra.float_emotion import FloatEmotion
from emotion_algebra.lovheim import (
    BASELINE,
    CORNER_ANCHORS,
    CORNER_POINTS,
    CORNERS,
    LovheimPoint,
    closest_affect,
)


class TestConstruction:
    def test_defaults_to_baseline(self):
        assert LovheimPoint().as_array.tolist() == list(BASELINE)

    def test_is_baseline(self):
        assert LovheimPoint(*BASELINE).is_baseline
        assert not LovheimPoint(1.0, 0.5, 0.5).is_baseline

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
    def test_out_of_cube_rejected(self, bad):
        with pytest.raises(ValueError):
            LovheimPoint(bad, 0.5, 0.5)

    @pytest.mark.parametrize("bad", ["1.0", None, True])
    def test_non_numeric_rejected(self, bad):
        with pytest.raises(TypeError):
            LovheimPoint(bad, 0.5, 0.5)

    def test_frozen(self):
        p = LovheimPoint(0.5, 0.5, 0.5)
        with pytest.raises(Exception):
            p.serotonin = 0.9

    def test_hashable(self):
        assert len({LovheimPoint(0.5, 0.5, 0.5), LovheimPoint(0.5, 0.5, 0.5)}) == 1

    def test_adrenaline_aliases_noradrenaline(self):
        p = LovheimPoint(0.1, 0.2, 0.3)
        assert p.adrenaline == p.noradrenaline == 0.3


class TestCorners:
    def test_eight_distinct_corners(self):
        assert len(CORNERS) == 8
        assert len(set(CORNERS.values())) == 8

    def test_every_corner_has_an_anchor(self):
        assert set(CORNER_ANCHORS) == set(CORNERS.values())

    @pytest.mark.parametrize("name,point", sorted(CORNER_POINTS.items()))
    def test_corner_names_itself(self, name, point):
        assert point.closest_affect() == name
        assert point.name == name
        assert str(point) == name

    @pytest.mark.parametrize("name,point", sorted(CORNER_POINTS.items()))
    def test_corner_blend_is_a_delta(self, name, point):
        blend = point.affect_blend()
        assert blend[name] == pytest.approx(1.0)
        assert sum(blend.values()) == pytest.approx(1.0)

    def test_convenience_wrapper(self):
        assert closest_affect(1.0, 1.0, 0.0) == CORNERS[(1, 1, 0)]

    def test_paper_corner_assignment(self):
        # Lövheim (2012), Med Hypotheses 78(2):341-8 — the cube's own labels.
        assert CORNERS[(0, 0, 0)].startswith("shame")
        assert CORNERS[(0, 1, 0)].startswith("fear")
        assert CORNERS[(0, 1, 1)].startswith("anger")
        assert CORNERS[(1, 1, 1)].startswith("interest")


class TestBlend:
    def test_blend_is_a_distribution(self):
        blend = LovheimPoint(0.3, 0.7, 0.2).affect_blend()
        assert sum(blend.values()) == pytest.approx(1.0)
        assert all(0.0 <= w <= 1.0 for w in blend.values())
        assert set(blend) == set(CORNERS.values())

    def test_centre_is_uniform(self):
        blend = LovheimPoint(0.5, 0.5, 0.5).affect_blend()
        assert all(w == pytest.approx(0.125) for w in blend.values())

    def test_blend_is_exact_on_an_edge(self):
        # A point on the serotonin edge splits between exactly two corners.
        blend = LovheimPoint(0.25, 1.0, 1.0).affect_blend()
        heavy = {k: v for k, v in blend.items() if v > 1e-9}
        assert len(heavy) == 2
        assert sum(heavy.values()) == pytest.approx(1.0)


class TestBridge:
    @pytest.mark.parametrize("name,point", sorted(CORNER_POINTS.items()))
    def test_corner_round_trips(self, name, point):
        back = LovheimPoint.from_float_emotion(point.to_float_emotion())
        assert back.closest_affect() == name

    def test_cube_centre_is_hedonically_negative_not_neutral(self):
        # Five of Tomkins' eight affects are negative (shame, distress, fear,
        # anger, contempt), so the centroid of the corner anchors is negative.
        # The cube's baseline is therefore NOT the Hourglass origin — a real
        # property of the affect set, not a calibration error.
        fe = LovheimPoint(*BASELINE).to_float_emotion()
        assert (fe.sensitivity, fe.attention) == (0.0, 0.0)  # these axes do cancel
        assert fe.pleasantness < 0
        assert fe.aptitude < 0

    def test_neutral_maps_back_to_baseline(self):
        # The inverse short-circuits neutral to baseline rather than to the exact
        # pre-image, so the round trip is deliberately asymmetric here.
        assert LovheimPoint.from_float_emotion(FloatEmotion(0, 0, 0, 0)).is_baseline

    def test_interior_point_round_trips_through_its_image(self):
        p = LovheimPoint(0.3, 0.8, 0.15)
        back = LovheimPoint.from_float_emotion(p.to_float_emotion())
        # The map is not injective, so the pre-image need not be p — but it
        # must land on the same image, exactly.
        assert np.allclose(
            back.to_float_emotion().as_array, p.to_float_emotion().as_array, atol=1e-6
        )

    def test_cube_cannot_express_positive_aptitude(self):
        # Tomkins has no trust/admiration affect, so no corner anchors a
        # positive Aptitude and no blend of them can produce one.
        assert all(a[3] <= 0.0 for a in CORNER_ANCHORS.values())
        rng = np.random.default_rng(0)
        for s, d, n in rng.random((200, 3)):
            assert LovheimPoint(s, d, n).to_float_emotion().aptitude <= 1e-9

    def test_deltas_from_baseline_are_signed(self):
        # Downstream ordering: (dopamine, serotonin, adrenaline).
        dopamine, serotonin, adrenaline = LovheimPoint(0.9, 0.1, 0.5).deltas_from_baseline()
        assert serotonin > 0
        assert dopamine < 0
        assert adrenaline == pytest.approx(0.0)

    def test_joy_reads_as_high_serotonin_dopamine(self):
        p = LovheimPoint.from_float_emotion(FloatEmotion.from_emotion(get_emotion("ecstasy")))
        assert p.serotonin > 0.5
        assert p.dopamine > 0.5


class TestSerialization:
    def test_dict_round_trip(self):
        p = LovheimPoint(0.2, 0.4, 0.6)
        assert LovheimPoint.from_dict(p.to_dict()) == p

    def test_json_round_trip(self):
        from emotion_algebra.serialization import from_json

        p = LovheimPoint(0.2, 0.4, 0.6)
        assert from_json(p.to_json()) == p

    def test_repr_is_evaluable(self):
        p = LovheimPoint(0.25, 0.5, 0.75)
        assert eval(repr(p)) == p  # noqa: S307 — repr contract
