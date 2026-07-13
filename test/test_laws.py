"""Property-based tests for the algebraic laws in ``docs/laws.md``.

Every law documented there is machine-checked here with `hypothesis`, over the
whole space of named emotions, flows and float vectors, rather than over a
handful of hand-picked examples. If a law is stated in the docs it must appear
here; if it is checked here it must be stated in the docs.

The CI profile is derandomised, so a failure is always reproducible.
"""
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from emotion_algebra.base import hourglass_polarity
from emotion_algebra.composite_emotions import CompositeEmotion
from emotion_algebra.distance import (
    MAX_EMOTION_DISTANCE,
    emotion_distance,
    emotion_similarity,
)
from emotion_algebra.emotions import EMOTIONS, get_emotion
from emotion_algebra.feelings import FEELINGS
from emotion_algebra.float_emotion import FloatEmotion
from emotion_algebra.lovheim import CORNER_POINTS, LovheimPoint
from emotion_algebra.pad import from_pad, to_pad
from emotion_algebra.plutchik import DIMENSIONS, Emotion, Neutrality
from emotion_algebra.serialization import from_dict, to_dict

settings.register_profile("ci", derandomize=True, max_examples=200,
                          suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("ci")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

EMOTION_NAMES = sorted(EMOTIONS.keys())
FEELING_NAMES = sorted(FEELINGS.keys())
AXES = sorted(DIMENSIONS.keys())

emotion_names = st.sampled_from(EMOTION_NAMES)
feeling_names = st.sampled_from(FEELING_NAMES)
axes = st.sampled_from(AXES)
flows = st.integers(min_value=-6, max_value=6)
in_range_flows = st.integers(min_value=-3, max_value=3)

#: Finite floats on the Hourglass scale.
axis_values = st.floats(
    min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False, width=32
)
#: Cube coordinates.
unit_values = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32
)


@st.composite
def emotions(draw):
    """Any of the 24 named emotions."""
    return get_emotion(draw(emotion_names))


@st.composite
def float_emotions(draw):
    """Any point in the Hourglass float space."""
    return FloatEmotion(*[draw(axis_values) for _ in range(4)])


@st.composite
def lovheim_points(draw):
    """Any point in the cube."""
    return LovheimPoint(*[draw(unit_values) for _ in range(3)])


# ---------------------------------------------------------------------------
# Law 1 — Neutrality is the additive identity
# ---------------------------------------------------------------------------

class TestIdentityLaws:
    @given(emotions())
    def test_neutrality_is_right_identity(self, e):
        assert (e + Neutrality()) == e

    @given(float_emotions())
    def test_neutrality_is_right_identity_for_floats(self, fe):
        assert (fe + FloatEmotion()) == fe

    @given(axes, in_range_flows)
    def test_neutrality_on_axis_plus_flow_constructs_that_emotion(self, axis, flow):
        """``Neutrality(axis) + flow`` is the canonical constructor for a flow."""
        result = Neutrality(dimension=axis) + flow
        if flow == 0:
            assert result.emotional_flow == 0
        else:
            assert isinstance(result, Emotion)
            assert result.emotional_flow == flow
            assert result.dimension.axis == axis


# ---------------------------------------------------------------------------
# Law 2 — Negation is an involution
# ---------------------------------------------------------------------------

class TestNegationLaws:
    @given(emotions())
    def test_negation_is_involutive(self, e):
        assert -(-e) == e

    @given(float_emotions())
    def test_float_negation_is_involutive(self, fe):
        assert -(-fe) == fe

    @given(emotions())
    def test_negation_flips_flow_sign(self, e):
        assert (-e).emotional_flow == -e.emotional_flow

    @given(emotions())
    def test_emotion_plus_its_negation_is_neutral(self, e):
        """``e + (-e)`` cancels to zero flow — the additive-inverse law."""
        assert (e + (-e)).emotional_flow == 0

    @given(emotions())
    def test_negation_preserves_arousal(self, e):
        """Flipping the pole does not change how activated you are."""
        assert (-e).arousal == e.arousal


# ---------------------------------------------------------------------------
# Law 3 — Same-axis addition is commutative and associative
# ---------------------------------------------------------------------------

class TestAdditionLaws:
    @given(emotions(), emotions())
    def test_addition_is_commutative(self, a, b):
        assert (a + b).as_array.tolist() == (b + a).as_array.tolist()

    @given(emotions(), emotions(), emotions())
    def test_addition_is_associative_within_range(self, a, b, c):
        """Associative *within range* — saturation is not, and cannot be, associative.

        Once any partial sum clamps at the ±3 lattice bound, information is
        gone, and the grouping decides how much: ``anger + (terror + terror)``
        clamps the inner sum to terror and yields apprehension, while
        ``(anger + terror) + terror`` clamps only at the end and yields a hyper
        terror. That is true of every saturating arithmetic, not a defect here,
        so the law is scoped to where no clamping occurs.
        """
        # Every *intermediate* must stay in range too, not just the total.
        # `anger + anger + apprehension` sums to 3, but the grouping
        # `(anger + anger)` hits 4 and clamps on the way, which is precisely
        # how the two groupings come apart.
        assume(np.all(np.abs(a.as_array + b.as_array) <= 3))
        assume(np.all(np.abs(b.as_array + c.as_array) <= 3))
        assume(np.all(np.abs(a.as_array + b.as_array + c.as_array) <= 3))
        left = (a + b) + c
        right = a + (b + c)
        assert np.allclose(left.as_array, right.as_array)

    @given(axes, flows, flows)
    def test_same_axis_addition_saturates_at_the_hyper_bound(self, axis, f1, f2):
        """Flow saturates at ±9; nothing ever escapes the representable range."""
        base = DIMENSIONS[axis].basic_emotion
        result = base.emotion_from_flow(f1) + f2
        assert -9 <= int(result) <= 9

    @given(axes)
    def test_cancelling_an_emotion_keeps_its_axis(self, axis):
        """``e - e`` is neutral *on e's axis*, not a dimensionless neutral.

        Dropping the axis would strand the value: the algebra could not be
        continued on an axis it had forgotten.
        """
        e = DIMENSIONS[axis].mild_emotion
        result = e - e
        assert result.emotional_flow == 0
        assert result.dimension is not None
        assert result.dimension.axis == axis
        # ...and it is still usable as an operand on that axis.
        assert (result >> 2).emotional_flow == 2


# ---------------------------------------------------------------------------
# Law 4 — Shifts are inverse
# ---------------------------------------------------------------------------

class TestShiftLaws:
    @given(axes, in_range_flows, st.integers(min_value=-3, max_value=3))
    def test_rshift_then_lshift_round_trips(self, axis, flow, shift):
        e = DIMENSIONS[axis].basic_emotion.emotion_from_flow(flow)
        assume(abs(flow + shift) <= 3)  # stay inside the lattice, no saturation
        assert ((e >> shift) << shift).emotional_flow == e.emotional_flow

    @given(axes, in_range_flows, in_range_flows)
    def test_shift_by_emotion_is_shift_by_its_flow(self, axis, flow, other_flow):
        dim = DIMENSIONS[axis]
        e = dim.basic_emotion.emotion_from_flow(flow)
        other = dim.basic_emotion.emotion_from_flow(other_flow)
        assume(other_flow != 0)
        assert (e << other).emotional_flow == (e << other_flow).emotional_flow


# ---------------------------------------------------------------------------
# Law 5 — Distance is a true metric
# ---------------------------------------------------------------------------

class TestMetricLaws:
    @given(emotions(), emotions())
    def test_non_negative(self, a, b):
        assert emotion_distance(a, b) >= 0.0

    @given(emotions())
    def test_identity_of_indiscernibles(self, a):
        assert emotion_distance(a, a) == 0.0

    @given(emotions(), emotions())
    def test_symmetry(self, a, b):
        assert emotion_distance(a, b) == pytest.approx(emotion_distance(b, a))

    @given(emotions(), emotions(), emotions())
    def test_triangle_inequality(self, a, b, c):
        ab = emotion_distance(a, b)
        bc = emotion_distance(b, c)
        ac = emotion_distance(a, c)
        assert ac <= ab + bc + 1e-9

    @given(emotions(), emotions())
    def test_distance_never_exceeds_the_documented_maximum(self, a, b):
        assert emotion_distance(a, b) <= MAX_EMOTION_DISTANCE + 1e-9


class TestSimilarityLaws:
    @given(emotions(), emotions(), st.sampled_from(["distance", "cosine"]))
    def test_similarity_is_bounded(self, a, b, metric):
        assert 0.0 <= emotion_similarity(a, b, metric=metric) <= 1.0

    @given(emotions(), emotions(), st.sampled_from(["distance", "cosine"]))
    def test_similarity_is_symmetric(self, a, b, metric):
        assert emotion_similarity(a, b, metric=metric) == pytest.approx(
            emotion_similarity(b, a, metric=metric)
        )

    @given(emotions())
    def test_self_similarity_is_one(self, a):
        assert emotion_similarity(a, a) == pytest.approx(1.0)

    @given(emotions())
    def test_extreme_of_the_opposite_pole_is_least_similar_on_the_axis(self, a):
        """The least similar emotion on an axis is the *tertiary* at the far pole.

        Note this is not ``-a``. Negation mirrors an emotion at its own
        intensity tier — ``-acceptance`` (aptitude +1) is ``boredom`` (−1) —
        but ``loathing`` (−3) is farther still. "Opposite" and "most distant"
        are different questions, and conflating them is easy to do.

        Nor is the extreme the least similar *overall*: the space is
        4-dimensional and Euclidean, so a mid-intensity emotion on another axis
        can out-distance a near opposite on this one. Claiming otherwise would
        assert the axes are commensurable, which they are not.
        """
        dim = DIMENSIONS[a.dimension.axis]
        extreme = dim.intense_opposite if a.emotional_flow > 0 else dim.intense_emotion
        same_axis = [e for e in EMOTIONS.values() if e.dimension.axis == dim.axis]
        for other in same_axis:
            assert emotion_similarity(a, extreme) <= emotion_similarity(a, other) + 1e-9

    @given(emotions())
    def test_negation_mirrors_at_the_same_intensity_tier(self, a):
        assert (-a).arousal == a.arousal
        assert (-a).emotional_flow == -a.emotional_flow

    @given(emotions())
    def test_tertiary_opposites_are_maximally_dissimilar(self, a):
        """The poles of one axis at full intensity are as far apart as the model goes."""
        assume(a.dimension.intense_emotion.name == a.name)
        assert emotion_similarity(a, -a) == pytest.approx(0.0)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            emotion_similarity(get_emotion("joy"), get_emotion("fear"), metric="euclidean")


# ---------------------------------------------------------------------------
# Law 6 — polarity is bounded, and agrees with valence where both apply
# ---------------------------------------------------------------------------

class TestPolarityLaws:
    @given(float_emotions())
    def test_polarity_is_bounded(self, fe):
        assert -1.0 <= fe.polarity <= 1.0

    @given(emotions())
    def test_negation_flips_polarity_sign_on_hedonic_axes(self, e):
        """On the signed (hedonic) axes, negating the emotion negates polarity.

        Not true on Sensitivity/Attention, where polarity uses |value| — that
        asymmetry is the whole point of Cambria's formula, and is asserted
        separately.
        """
        assume(e.dimension.axis in ("pleasantness", "aptitude"))
        assert (-e).polarity == pytest.approx(-e.polarity)

    @given(emotions())
    def test_sensitivity_is_aversive_at_both_poles(self, e):
        assume(e.dimension.axis == "sensitivity")
        assert e.polarity < 0

    @given(emotions())
    def test_attention_is_engaging_at_both_poles(self, e):
        assume(e.dimension.axis == "attention")
        assert e.polarity > 0

    @given(float_emotions())
    def test_polarity_matches_the_published_formula(self, fe):
        assert fe.polarity == pytest.approx(hourglass_polarity(fe.as_array))


# ---------------------------------------------------------------------------
# Law 7 — Conversion coherence
# ---------------------------------------------------------------------------

class TestConversionLaws:
    @given(emotions())
    def test_emotion_to_float_preserves_the_vector(self, e):
        assert np.allclose(FloatEmotion.from_emotion(e).as_array, e.as_array)

    @given(emotions())
    def test_emotion_float_roundtrip_preserves_arousal_and_valence(self, e):
        fe = FloatEmotion.from_emotion(e)
        assert fe.arousal == pytest.approx(e.arousal)
        assert fe.valence == pytest.approx(e.valence)
        assert fe.polarity == pytest.approx(e.polarity)

    @given(emotion_names)
    def test_named_emotions_are_their_own_nearest_name(self, name):
        """``closest_emotion`` on an emotion's own vector returns that emotion."""
        from emotion_algebra.distance import closest_emotion
        e = get_emotion(name)
        assert emotion_distance(closest_emotion(e.as_array), e) == 0.0

    @given(feeling_names)
    def test_named_feelings_are_their_own_nearest_name(self, name):
        from emotion_algebra.distance import closest_emotion
        f = FEELINGS[name]
        assert emotion_distance(closest_emotion(f.as_array), f) == 0.0


# ---------------------------------------------------------------------------
# Law 8 — Lövheim cube
# ---------------------------------------------------------------------------

class TestLovheimLaws:
    @given(lovheim_points())
    def test_affect_blend_is_a_distribution(self, p):
        weights = p.affect_blend()
        assert len(weights) == 8
        assert all(w >= -1e-12 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0)

    @given(lovheim_points())
    def test_closest_affect_is_always_a_named_corner(self, p):
        from emotion_algebra.lovheim import CORNERS
        assert p.closest_affect() in set(CORNERS.values())

    @given(lovheim_points())
    def test_closest_affect_is_the_heaviest_blend_weight(self, p):
        """The nearest corner is also the one carrying the most trilinear weight.

        Both rank the corners the same way, so the discrete and continuous
        readings of a cube point can never contradict each other.
        """
        weights = p.affect_blend()
        heaviest = max(weights.values())
        assert weights[p.closest_affect()] == pytest.approx(heaviest)

    def test_corners_round_trip_exactly(self):
        for name, point in CORNER_POINTS.items():
            back = LovheimPoint.from_float_emotion(point.to_float_emotion())
            assert np.allclose(back.as_array, point.as_array), name
            assert back.closest_affect() == name

    @given(lovheim_points())
    def test_projection_is_idempotent_on_the_image(self, p):
        """Projecting a point's own image back gives something with the same image.

        The map is not injective, so the *coordinates* may legitimately differ —
        but the projection must be a genuine best fit, so the image must match.
        """
        image = p.to_float_emotion()
        back = LovheimPoint.from_float_emotion(image)
        assert np.allclose(back.to_float_emotion().as_array, image.as_array, atol=1e-6)

    @given(unit_values, unit_values)
    def test_cube_never_reaches_positive_aptitude(self, d, n):
        """Tomkins has no trust/admiration affect, so the cube cannot express one.

        A documented limit of the bridge, asserted so it stays documented.
        """
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert LovheimPoint(s, d, n).to_float_emotion().as_array[3] <= 1e-9

    def test_forward_map_is_not_injective(self):
        """Two distinct cube points share the Hourglass origin.

        Fear and anger are exact negatives in Plutchik but *adjacent corners* in
        Lövheim, so a blend across that edge cancels. Same for surprise and
        interest. This is the two theories disagreeing, and it is why
        ``from_float_emotion`` is a projection rather than an inverse.
        """
        a = LovheimPoint(0.0, 1.0, 0.5)   # midway fear ↔ anger
        b = LovheimPoint(1.0, 0.5, 1.0)   # midway surprise ↔ interest
        assert not np.allclose(a.as_array, b.as_array)
        assert np.allclose(a.to_float_emotion().as_array, 0.0)
        assert np.allclose(b.to_float_emotion().as_array, 0.0)

    def test_neutral_emotion_maps_to_baseline(self):
        """An inert state means no monoamine displacement."""
        assert LovheimPoint.from_float_emotion(FloatEmotion()).is_baseline

    @given(unit_values, unit_values, unit_values, unit_values)
    def test_raising_serotonin_never_moves_toward_a_low_serotonin_corner(
        self, s_lo, s_hi, d, n
    ):
        """Monotonicity: more serotonin cannot make a low-serotonin affect *more* likely.

        The only symmetry claim made here is one the cube's own geometry
        guarantees — no symmetry is invented that the paper does not state.
        """
        assume(s_lo < s_hi)
        from emotion_algebra.lovheim import CORNERS
        low = LovheimPoint(s_lo, d, n).affect_blend()
        high = LovheimPoint(s_hi, d, n).affect_blend()
        for corner, affect in CORNERS.items():
            if corner[0] == 0:  # low-serotonin corner
                assert high[affect] <= low[affect] + 1e-9


# ---------------------------------------------------------------------------
# Law 9 — PAD projection
# ---------------------------------------------------------------------------

class TestPADLaws:
    @given(emotions())
    def test_pad_components_are_in_range(self, e):
        p = to_pad(e)
        assert -1.0 <= p.pleasure <= 1.0
        assert 0.0 <= p.arousal <= 1.0
        assert -1.0 <= p.dominance <= 1.0

    @given(emotions())
    def test_pleasure_is_polarity(self, e):
        assert to_pad(e).pleasure == pytest.approx(e.polarity)

    def test_anger_dominant_fear_submissive(self):
        """The one fact PAD's third axis exists to capture."""
        assert to_pad(get_emotion("rage")).dominance > 0
        assert to_pad(get_emotion("terror")).dominance < 0

    @given(
        st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False, width=32),
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False, width=32),
        st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False, width=32),
    )
    def test_from_pad_stays_on_the_hourglass_scale(self, p, a, d):
        fe = from_pad(p, a, d)
        assert np.all(np.abs(fe.as_array) <= 3.0 + 1e-9)

    @given(
        st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False, width=32),
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False, width=32),
        st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False, width=32),
    )
    def test_pad_roundtrip_preserves_dominance_sign(self, p, a, d):
        """PAD → Hourglass → PAD is lossy (3 numbers cannot pin down 4 axes),
        but it must never flip the sign of dominance — that is the axis's
        entire purpose."""
        assume(abs(d) > 0.05)
        back = to_pad(from_pad(p, a, d))
        assert math.copysign(1, back.dominance) == math.copysign(1, d)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"pleasure": 2.0, "arousal": 0.5, "dominance": 0.0},
            {"pleasure": 0.0, "arousal": -0.5, "dominance": 0.0},
            {"pleasure": 0.0, "arousal": 0.5, "dominance": 9.0},
            {"pleasure": float("nan"), "arousal": 0.5, "dominance": 0.0},
        ],
    )
    def test_out_of_range_pad_raises(self, kwargs):
        with pytest.raises(ValueError):
            from_pad(**kwargs)


# ---------------------------------------------------------------------------
# Law 10 — Serialization round-trips
# ---------------------------------------------------------------------------

class TestSerializationLaws:
    @given(emotions())
    def test_emotion_roundtrip(self, e):
        assert from_dict(to_dict(e)).name == e.name

    @given(float_emotions())
    def test_float_emotion_roundtrip(self, fe):
        assert np.allclose(from_dict(to_dict(fe)).as_array, fe.as_array)

    @given(lovheim_points())
    def test_lovheim_roundtrip(self, p):
        assert np.allclose(from_dict(to_dict(p)).as_array, p.as_array)

    @given(feeling_names)
    def test_feeling_roundtrip(self, name):
        f = FEELINGS[name]
        assert np.allclose(from_dict(to_dict(f)).as_array, f.as_array)
