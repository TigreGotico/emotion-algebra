"""The formal laws of the affect core, machine-checked with hypothesis.

The core is **not** a vector space — but it is very much an algebra. Three
structures, each with laws that hold exactly:

1. ``(S, blend)`` is a **barycentric algebra** (a convex space): idempotent,
   unital, skew-commutative, barycentric-associative. By Stone's theorem the
   models of this theory are exactly the convex subsets of real vector spaces —
   so we lose no rigour, we merely say *which* subset.
2. ``(S, d)`` is a **compact metric space**.
3. ``{relax_t}`` is a **contraction semigroup** whose unique fixed point is the
   set point. Banach then guarantees every state converges to it, exponentially.

The laws that do NOT hold are stated too, at the bottom, with the reason. An
algebra is defined as much by what it refuses to do.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from emotion_algebra.affect import CORE_AXES, ORIGIN, AffectState, mixture
from emotion_algebra.homeostasis import SET_POINT, drive_magnitude, relax

settings.register_profile(
    "ci", derandomize=True, max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("ci")

TOL = 1e-9
HALF_LIFE = 300.0


@st.composite
def states(draw):
    """Any point in the core."""
    return AffectState(
        positivity=draw(st.floats(0.0, 1.0, allow_nan=False)),
        negativity=draw(st.floats(0.0, 1.0, allow_nan=False)),
        potency=draw(st.floats(-1.0, 1.0, allow_nan=False)),
        arousal=draw(st.floats(0.0, 1.0, allow_nan=False)),
        unpredictability=draw(st.floats(0.0, 1.0, allow_nan=False)),
    )


weights = st.floats(0.0, 1.0, allow_nan=False)
times = st.floats(0.0, 2000.0, allow_nan=False)


def close(a: AffectState, b: AffectState, tol: float = TOL) -> bool:
    return bool(np.allclose(a.as_array, b.as_array, atol=tol))


class TestBarycentricAlgebra:
    """``(S, blend)`` is a convex space.

    These four laws *are* the axioms of a barycentric algebra (Stone 1949).
    Together they say: blending is what it looks like, and nothing more.
    """

    @given(states(), weights)
    def test_idempotence(self, a, w):
        # Blending a thing with itself changes nothing, at any weight.
        assert close(a.blend(a, w), a)

    @given(states(), states())
    def test_unit_laws(self, a, b):
        assert close(a.blend(b, 0.0), a)
        assert close(a.blend(b, 1.0), b)

    @given(states(), states(), weights)
    def test_skew_commutativity(self, a, b, w):
        # blend(a, b, w) == blend(b, a, 1-w). Mixing is symmetric once you
        # account for which side the weight is measured from.
        assert close(a.blend(b, w), b.blend(a, 1.0 - w))

    @given(states(), states(), states(), weights, weights)
    def test_barycentric_associativity(self, a, b, c, p, q):
        # The real associativity law of a convex space. Naive associativity is
        # FALSE — the weights must be reparametrized, because a weight means
        # "share of what is left", not "share of the whole".
        s = 1.0 - (1.0 - p) * (1.0 - q)
        assume(s > 1e-6)
        r = q / s
        assume(0.0 <= r <= 1.0)
        left = a.blend(b, p).blend(c, q)
        right = a.blend(b.blend(c, r), s)
        assert close(left, right, tol=1e-7)

    @given(states(), states(), weights)
    def test_blend_never_leaves_the_space(self, a, b, w):
        # Convexity means closure is FREE: no clamping is needed, ever. This is
        # exactly what a vector-space model could not promise.
        m = a.blend(b, w)
        assert 0.0 <= m.positivity <= 1.0
        assert 0.0 <= m.negativity <= 1.0
        assert -1.0 <= m.potency <= 1.0
        assert 0.0 <= m.arousal <= 1.0
        assert 0.0 <= m.unpredictability <= 1.0

    @given(states(), states(), weights)
    def test_mixture_agrees_with_blend(self, a, b, w):
        assert close(mixture([a, b], [1.0 - w, w]), a.blend(b, w))

    @given(states(), states(), states())
    def test_mixture_is_permutation_invariant(self, a, b, c):
        assert close(mixture([a, b, c]), mixture([c, a, b]))

    @given(states(), states())
    def test_mixture_concentrated_on_one_is_that_one(self, a, b):
        assert close(mixture([a, b], [1.0, 0.0]), a)


class TestMetric:
    """``(S, d)`` is a metric space."""

    @given(states(), states())
    def test_non_negative_and_symmetric(self, a, b):
        assert a.distance(b) >= 0.0
        assert a.distance(b) == pytest.approx(b.distance(a))

    @given(states())
    def test_identity_of_indiscernibles(self, a):
        assert a.distance(a) == pytest.approx(0.0)

    @given(states(), states(), states())
    def test_triangle_inequality(self, a, b, c):
        assert a.distance(c) <= a.distance(b) + b.distance(c) + 1e-9

    @given(states(), states(), weights)
    def test_blend_lies_between_its_endpoints(self, a, b, w):
        # A convex combination is on the segment, so it can be no further from
        # either endpoint than they are from each other.
        m = a.blend(b, w)
        assert a.distance(m) <= a.distance(b) + 1e-9
        assert b.distance(m) <= a.distance(b) + 1e-9


class TestContractionSemigroup:
    """``{relax_t}`` is a contraction semigroup with a unique fixed point.

    This is the theorem that makes the set point an *attractor* rather than a
    hopeful default: Banach guarantees convergence, from anywhere.
    """

    @given(states())
    def test_zero_time_is_the_identity(self, a):
        assert close(relax(a, 0.0, HALF_LIFE), a)

    @given(states(), times, times)
    def test_semigroup_law(self, a, t1, t2):
        # relax(t1) then relax(t2) == relax(t1 + t2). Exactly.
        assert close(
            relax(relax(a, t1, HALF_LIFE), t2, HALF_LIFE),
            relax(a, t1 + t2, HALF_LIFE),
            tol=1e-7,
        )

    @given(times)
    def test_the_set_point_is_a_fixed_point(self, t):
        assert close(relax(SET_POINT, t, HALF_LIFE), SET_POINT)

    @given(states(), states(), times)
    def test_it_is_a_contraction_with_a_known_factor(self, a, b, t):
        # d(relax(a), relax(b)) == k * d(a, b), with k = 2^(-t/h) <= 1.
        k = 0.5 ** (t / HALF_LIFE)
        expected = k * a.distance(b)
        assert relax(a, t, HALF_LIFE).distance(relax(b, t, HALF_LIFE)) == pytest.approx(
            expected, abs=1e-7
        )

    @given(states())
    def test_every_state_converges_to_the_set_point(self, a):
        # Banach, made concrete. Wherever you start, you end up at rest.
        assert close(relax(a, 100_000.0, HALF_LIFE), SET_POINT, tol=1e-6)

    @given(states(), times)
    def test_relaxing_never_increases_the_drive(self, a, t):
        assert drive_magnitude(relax(a, t, HALF_LIFE)) <= drive_magnitude(a) + 1e-9

    @given(states())
    def test_the_attractor_is_not_the_origin(self, a):
        # The point of the whole exercise. Relaxation goes home, and home is not
        # zero — an organism at rest is mildly positive, not blank.
        settled = relax(a, 100_000.0, HALF_LIFE)
        assert not settled.is_origin
        assert settled.valence > 0.0


class TestIntensification:
    """``intensify`` is a partial monoid action of (R>=0, x) on the core."""

    @given(states())
    def test_unit(self, a):
        assert close(a.intensify(1.0), a)

    @given(states())
    def test_annihilator_is_the_origin_not_the_set_point(self, a):
        # Scaling to zero lands on the ORIGIN — which is precisely why the
        # origin is a coordinate fact and not a psychological state.
        assert close(a.intensify(0.0), ORIGIN)

    @given(states(), st.floats(0.0, 1.0), st.floats(0.0, 1.0))
    def test_composition(self, a, j, k):
        # Monoid action: intensify(intensify(a, j), k) == intensify(a, j*k).
        # Only where nothing clamps — hence factors <= 1.
        assert close(a.intensify(j).intensify(k), a.intensify(j * k), tol=1e-7)

    @given(states(), states(), st.floats(0.0, 1.0), weights)
    def test_distributes_over_blend(self, a, b, k, w):
        # k*(mix of a and b) == mix of (k*a) and (k*b). Linearity, where it is
        # safe to claim it.
        assert close(
            a.blend(b, w).intensify(k),
            a.intensify(k).blend(b.intensify(k), w),
            tol=1e-7,
        )

    @given(states(), st.floats(0.0, 1.0))
    def test_shrinking_never_grows_the_state(self, a, k):
        assert a.intensify(k).intensity <= a.intensity + 1e-9

    @given(states(), st.floats(-10.0, -1e-6))
    def test_negative_factors_are_refused(self, a, k):
        # There is no negation, and there will be no negation by the back door.
        with pytest.raises(ValueError):
            a.intensify(k)


class TestTheLawsThatDoNotHold:
    """What the core refuses to do, and why. An algebra is defined by this too."""

    def test_there_is_no_additive_inverse(self):
        # In a vector space, every a has a -a summing to zero. Here there is no
        # sum at all, and no negation: sadness is not "minus joy", because it
        # has its own action readiness (withdraw), not "negative approach".
        assert not hasattr(AffectState, "__neg__")
        assert not hasattr(AffectState, "__sub__")
        assert not hasattr(AffectState, "__add__")
        assert not hasattr(AffectState, "__mul__")

    def test_naive_associativity_of_blend_is_false(self):
        # blend(blend(a,b,p), c, q) is NOT blend(a, blend(b,c,p), q).
        #
        # An existence claim, so it gets a concrete witness rather than a
        # property test: for *some* inputs the two coincide by luck, which is
        # exactly why the honest law is the reparametrized one in
        # TestBarycentricAlgebra.test_barycentric_associativity. A weight is a
        # share of *what remains*, not a share of the whole.
        a = AffectState(positivity=1.0)
        b = AffectState(negativity=1.0)
        c = AffectState(potency=1.0)
        left = a.blend(b, 0.5).blend(c, 0.5)
        right = a.blend(b.blend(c, 0.5), 0.5)
        assert not close(left, right, tol=1e-3)
