"""Acceptance tests for the affect core.

These are not unit tests. Each one is a claim the literature makes about human
emotion, turned into an assertion the model has to earn. If any of them fail,
the model is wrong — not the test.
"""
import itertools

import numpy as np
import pytest

import emotion_algebra.views  # noqa: F401 — registers the conversion graph
from emotion_algebra.affect import ORIGIN, AffectState, mixture
from emotion_algebra.evidence import CITABLE, Grade, all_evidence, grade_of
from emotion_algebra.homeostasis import (
    SET_POINT,
    Temperament,
    at_rest,
    drive,
    drive_magnitude,
    perturb,
    relax,
)
from emotion_algebra.lovheim import LovheimPoint
from emotion_algebra.neuro import BASELINE, NeuroState
from emotion_algebra.projection import (
    CORE,
    Fidelity,
    convert,
    explain_loss,
    fidelity,
    get_view,
    views,
)
from emotion_algebra.prototypes import PROTOTYPES, prototype
from emotion_algebra.readout import dominant, entropy, label
from emotion_algebra.tendency import action_readiness, dominant_tendency


class TestAngerFearCrux:
    """The case that breaks every valence/arousal model, and the wheel with it.

    Anger and fear are both negative and both high-arousal, so valence and
    arousal cannot separate them. Four independent programmes (Smith & Ellsworth
    1985; Roseman 1996; Scherer's SECs; Lerner & Keltner 2001, with causal
    mediation) find that control/coping does.
    """

    def test_both_are_negative_and_activated(self):
        for name in ("anger", "fear"):
            assert prototype(name).valence < 0
            assert prototype(name).arousal > 0.5

    def test_potency_is_what_separates_them(self):
        assert prototype("anger").potency > 0 > prototype("fear").potency
        assert prototype("rage").potency > 0 > prototype("terror").potency

    def test_they_are_indistinguishable_on_valence_and_arousal(self):
        # The reason valence/arousal models cannot tell them apart, and the
        # reason PAD needs a third axis at all.
        anger, fear = prototype("anger"), prototype("fear")
        assert anger.valence == pytest.approx(fear.valence, abs=0.05)
        assert anger.arousal == pytest.approx(fear.arousal, abs=0.05)

    def test_they_are_not_antipodal(self):
        # Plutchik's wheel says terror == -rage: every coordinate flipped. If
        # that were so, anger and fear would have OPPOSITE valence and arousal.
        # They have the same. They differ on potency — one axis, not all of them.
        anger, fear = prototype("anger"), prototype("fear")
        assert np.sign(anger.valence) == np.sign(fear.valence)
        assert np.sign(anger.arousal) == np.sign(fear.arousal)
        assert np.sign(anger.potency) != np.sign(fear.potency)

        # And concretely: fear is not the reflection of anger.
        reflected = AffectState(
            positivity=anger.negativity,
            negativity=anger.positivity,
            potency=-anger.potency,
            arousal=anger.arousal,
            unpredictability=anger.unpredictability,
        )
        assert reflected.distance(fear) > 0.5, "fear is not '-anger'"

    def test_the_midpoint_of_rage_and_terror_is_not_neutrality(self):
        # THE bug this whole core exists to fix. On the Hourglass axes,
        # (rage + terror) / 2 == [0,0,0,0] — blend the two most intense negative
        # states and get calm.
        mid = prototype("rage").blend(prototype("terror"), 0.5)
        assert not mid.is_origin
        assert not at_rest(mid)
        assert mid.valence < -0.5, "the blend must stay strongly unpleasant"
        assert mid.arousal > 0.5, "and stay activated"

    def test_the_midpoint_is_distress(self):
        # Control cancels out; unpleasantness and activation do not. That state
        # has a name, and it is not 'neutrality'.
        mid = prototype("rage").blend(prototype("terror"), 0.5)
        assert abs(mid.potency) < 0.2
        assert dominant(mid) == "distress"


class TestNoVectorSpace:
    """The core is a convex cone with a metric, not a vector space."""

    def test_there_is_no_negation(self):
        assert not hasattr(AffectState, "__neg__")

    def test_there_is_no_subtraction(self):
        assert not hasattr(AffectState, "__sub__")

    def test_intensify_rejects_negative_factors(self):
        # Negation by the back door.
        with pytest.raises(ValueError):
            prototype("joy").intensify(-1.0)

    def test_blend_stays_inside_the_space(self):
        for a, b in itertools.combinations(list(PROTOTYPES)[:8], 2):
            mixed = prototype(a).blend(prototype(b), 0.5)
            assert 0.0 <= mixed.positivity <= 1.0
            assert -1.0 <= mixed.potency <= 1.0

    @pytest.mark.parametrize("weight", [-0.1, 1.1])
    def test_blend_rejects_non_convex_weights(self, weight):
        with pytest.raises(ValueError):
            prototype("joy").blend(prototype("fear"), weight)

    def test_distance_is_a_metric(self):
        a, b, c = (prototype(n) for n in ("joy", "fear", "grief"))
        assert a.distance(a) == pytest.approx(0.0)
        assert a.distance(b) == pytest.approx(b.distance(a))
        assert a.distance(c) <= a.distance(b) + b.distance(c) + 1e-9

    def test_mixture_of_many_is_convex(self):
        states = [prototype(n) for n in ("joy", "fear", "grief", "rage")]
        m = mixture(states)
        assert 0.0 <= m.positivity <= 1.0
        assert -1.0 <= m.potency <= 1.0

    def test_mixture_rejects_empty_and_bad_weights(self):
        with pytest.raises(ValueError):
            mixture([])
        with pytest.raises(ValueError):
            mixture([prototype("joy")], weights=[-1.0])


class TestAmbivalence:
    """Happy and sad co-activate. A signed valence axis cannot represent that.

    Larsen, McGraw & Cacioppo (2001): in predictably ambivalent situations
    (graduation day) people report both, simultaneously.
    """

    def test_bittersweet_is_representable(self):
        graduation = AffectState(positivity=0.8, negativity=0.6, arousal=0.7)
        assert graduation.ambivalence > 0.5

    def test_signed_valence_destroys_it(self):
        # This is the information a signed-only model throws away: bittersweet
        # and flat are indistinguishable once you collapse to one axis.
        graduation = AffectState(positivity=0.8, negativity=0.6, arousal=0.7)
        flat = AffectState(positivity=0.2, negativity=0.0, arousal=0.7)
        assert graduation.valence == pytest.approx(flat.valence)
        assert graduation.ambivalence > flat.ambivalence

    def test_pure_states_are_not_ambivalent(self):
        assert prototype("joy").ambivalence == pytest.approx(0.0)
        assert prototype("grief").ambivalence == pytest.approx(0.0)


class TestHomeostasis:
    """The origin is not a state. Rest is a set point, and it is not at zero."""

    def test_the_origin_is_not_rest(self):
        # Core affect is always on (Barrett & Bliss-Moreau). Nothing lives at
        # the coordinate origin.
        assert ORIGIN.is_origin
        assert not at_rest(ORIGIN)

    def test_rest_is_mildly_positive(self):
        # The positivity offset: an organism at rest is disposed to explore.
        assert SET_POINT.valence > 0
        assert SET_POINT.positivity > SET_POINT.negativity
        assert at_rest(SET_POINT)

    def test_rest_is_low_arousal(self):
        assert SET_POINT.arousal < 0.35

    def test_decay_returns_to_the_set_point_not_the_origin(self):
        settled = relax(prototype("terror"), dt=10_000, half_life=300)
        assert at_rest(settled)
        assert not settled.is_origin

    def test_recovery_passes_through_milder_states(self):
        # terror -> fear -> ... -> rest. A recovery trajectory, not a jump.
        terror = prototype("terror")
        early = relax(terror, dt=150, half_life=300)
        late = relax(terror, dt=900, half_life=300)
        assert drive_magnitude(terror) > drive_magnitude(early) > drive_magnitude(late)

    def test_drive_points_home(self):
        d = drive(prototype("terror"))
        assert d["negativity"] < 0     # negativity must come down
        assert d["potency"] > 0        # control must be restored
        assert d["arousal"] < 0        # and activation must settle

    def test_drive_is_zero_at_rest(self):
        assert drive_magnitude(SET_POINT) == pytest.approx(0.0)

    def test_negativity_bias(self):
        # "Bad is stronger than good" (Baumeister et al. 2001). Equal pushes,
        # unequal landings.
        up = perturb(SET_POINT, {"positivity": 0.3}).positivity - SET_POINT.positivity
        down = perturb(SET_POINT, {"negativity": 0.3}).negativity - SET_POINT.negativity
        assert down > up

    def test_temperament_validates(self):
        with pytest.raises(ValueError):
            Temperament(negativity_bias=0.0)
        with pytest.raises(ValueError):
            Temperament(resilience=-1.0)

    @pytest.mark.parametrize("bad", [(-1.0, 300.0), (1.0, 0.0)])
    def test_relax_validates(self, bad):
        dt, half_life = bad
        with pytest.raises(ValueError):
            relax(prototype("joy"), dt=dt, half_life=half_life)


class TestActionReadiness:
    """Motivational direction tracks potency, not valence."""

    def test_anger_approaches_while_being_unpleasant(self):
        # Carver & Harmon-Jones (2009). This single fact refutes every
        # "negative = avoid" sentiment model.
        anger = prototype("anger")
        readiness = action_readiness(anger)
        assert anger.valence < 0
        assert readiness["approach"] > readiness["avoidance"]

    def test_anger_moves_against_fear_moves_away(self):
        assert dominant_tendency(prototype("anger")) == "antagonism"
        assert dominant_tendency(prototype("fear")) == "avoidance"

    def test_arousal_separates_fleeing_from_giving_up(self):
        # Fear and sadness are both negative and both low-potency. What differs
        # is activation: fear flees, sadness withdraws.
        assert dominant_tendency(prototype("fear")) == "avoidance"
        assert dominant_tendency(prototype("sadness")) == "withdrawal"

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("rage", "antagonism"),
            ("terror", "avoidance"),
            ("grief", "withdrawal"),
            ("joy", "affiliation"),
            ("trust", "affiliation"),
            ("disgust", "rejection"),
            ("surprise", "attending"),
            ("shame", "withdrawal"),
        ],
    )
    def test_readiness_matches_the_literature(self, name, expected):
        assert dominant_tendency(prototype(name)) == expected

    def test_at_rest_nothing_is_demanded(self):
        assert dominant_tendency(SET_POINT) == "rest"

    def test_readiness_is_a_distribution(self):
        r = action_readiness(prototype("anger"))
        assert sum(r.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in r.values())


class TestReadout:
    """Names are a distribution over regions, never a basis."""

    @pytest.mark.parametrize("name", sorted(PROTOTYPES))
    def test_every_prototype_names_itself(self, name):
        assert dominant(prototype(name)) == name

    def test_label_is_a_distribution(self):
        d = label(prototype("rage"))
        assert sum(d.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in d.values())

    def test_neighbours_are_not_zero(self):
        # Cowen & Keltner: categories bridged by continuous gradients, not
        # separated by walls. Rage's neighbours must stay visible.
        d = label(prototype("rage"))
        assert d["rage"] > d["anger"] > 0.0

    def test_in_between_states_have_high_entropy(self):
        # A state between categories should SAY it is between categories, not
        # pick one confidently.
        mid = prototype("joy").blend(prototype("fear"), 0.5)
        assert entropy(mid) > entropy(prototype("joy"))

    def test_top_k_renormalizes(self):
        d = label(prototype("joy"), top_k=3)
        assert len(d) == 3
        assert sum(d.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_temperature_must_be_positive(self, bad):
        with pytest.raises(ValueError):
            label(prototype("joy"), temperature=bad)


class TestNeuro:
    """Neuromodulators map to computational roles, never to emotion names."""

    def test_baseline_chemistry_is_rest(self):
        assert at_rest(BASELINE.to_affect())

    def test_coping_flips_anger_and_fear_under_identical_threat(self):
        # The whole model in one test. Same threat; only the coping chemistry
        # differs. Dopamine/testosterone (agency) vs cortisol (helplessness).
        cannot_cope = NeuroState(noradrenaline=0.95, cortisol=0.95, dopamine=0.15)
        can_cope = NeuroState(noradrenaline=0.9, dopamine=0.85, testosterone=0.9)

        assert cannot_cope.to_affect().potency < 0
        assert can_cope.to_affect().potency > 0

    def test_noradrenaline_drives_arousal(self):
        calm = NeuroState(noradrenaline=0.2).to_affect()
        alarmed = NeuroState(noradrenaline=0.9).to_affect()
        assert alarmed.arousal > calm.arousal

    def test_noradrenaline_drives_unexpected_uncertainty(self):
        assert (
            NeuroState(noradrenaline=0.9).to_affect().unpredictability
            > NeuroState(noradrenaline=0.2).to_affect().unpredictability
        )

    def test_cortisol_lowers_coping(self):
        assert NeuroState(cortisol=0.95).to_affect().potency < SET_POINT.potency

    def test_opioids_drive_liking(self):
        assert NeuroState(opioids=0.95).to_affect().positivity > SET_POINT.positivity

    def test_round_trip_preserves_the_emotion(self):
        for name in ("rage", "terror", "joy", "grief"):
            back = NeuroState.from_affect(prototype(name))
            assert dominant(back.to_affect()) == name

    def test_adrenaline_aliases_noradrenaline(self):
        ns = NeuroState(noradrenaline=0.7)
        assert ns.adrenaline == ns.noradrenaline

    def test_deltas_are_signed(self):
        d = NeuroState(dopamine=0.9, serotonin=0.1).deltas_from_baseline()
        assert d["dopamine"] > 0
        assert d["serotonin"] < 0

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
    def test_levels_are_bounded(self, bad):
        with pytest.raises(ValueError):
            NeuroState(dopamine=bad)

    def test_serialization_round_trip(self):
        ns = NeuroState(dopamine=0.7, cortisol=0.3)
        assert NeuroState.from_dict(ns.to_dict()) == ns


class TestTotalConversion:
    """Every model reaches every other, and every map declares what it loses."""

    SAMPLES = {
        CORE: prototype("anger"),
        "pad": (-0.6, 0.8, 0.6),
        "circumplex": (-0.6, 0.8),
        "hourglass": np.array([2.0, 0.0, -2.0, 0.0]),
        "plutchik": "anger",
        "neuro": NeuroState(dopamine=0.8),
        "lovheim": LovheimPoint(0.3, 0.8, 0.8),
    }

    def test_every_model_is_registered(self):
        assert set(views()) == set(self.SAMPLES)

    def test_the_graph_is_total(self):
        for source, target in itertools.product(views(), views()):
            convert(self.SAMPLES[source], source, target)

    def test_every_lossy_view_declares_its_loss(self):
        for name in views():
            if name == CORE:
                continue
            view = get_view(name)
            if view.fidelity is not Fidelity.EXACT:
                assert view.loses, f"{name} is {view.fidelity} but declares no loss"

    def test_pad_carries_its_own_axes_exactly(self):
        # Pleasure IS valence, Arousal IS arousal, Dominance IS potency. No
        # fitted weights, no unreachable region.
        for name in sorted(PROTOTYPES):
            state = prototype(name)
            p, a, d = convert(state, CORE, "pad")
            assert p == pytest.approx(state.valence)
            assert a == pytest.approx(state.arousal)
            assert d == pytest.approx(state.potency)

    def test_pad_round_trips_on_its_own_axes(self):
        for pleasure, arousal, dominance in [
            (-0.6, 0.8, 0.6),
            (0.5, 0.3, -0.2),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ]:
            back = convert((pleasure, arousal, dominance), "pad", CORE)
            assert back.valence == pytest.approx(pleasure)
            assert back.arousal == pytest.approx(arousal)
            assert back.potency == pytest.approx(dominance)

    def test_circumplex_cannot_tell_anger_from_fear(self):
        # The honest failure. Drop potency and the two collapse together — which
        # is exactly why the circumplex needs a third axis.
        anger = convert(prototype("anger"), CORE, "circumplex")
        fear = convert(prototype("fear"), CORE, "circumplex")
        assert anger == pytest.approx(fear)

    def test_fidelity_is_the_weakest_leg(self):
        assert fidelity("pad", CORE) is Fidelity.LOSSY
        assert fidelity("hourglass", "pad") is Fidelity.HEURISTIC

    def test_explain_loss_is_never_empty_for_a_lossy_map(self):
        assert "unpredictability" in explain_loss(CORE, "pad")
        assert explain_loss(CORE, CORE).endswith("nothing is lost.")

    def test_lovheim_stays_reachable_for_downstream(self):
        point = convert(prototype("rage"), CORE, "lovheim")
        assert isinstance(point, LovheimPoint)
        assert 0.0 <= point.dopamine <= 1.0


class TestEvidence:
    """The library states how much to trust each of its own parts."""

    def test_the_load_bearing_claims_are_established(self):
        assert grade_of("circumplex") is Grade.ESTABLISHED
        assert grade_of("appraisal.control_separates_anger_from_fear") is Grade.ESTABLISHED
        assert grade_of("categories.no_discrete_signatures") is Grade.ESTABLISHED

    def test_the_unsupported_models_are_graded_as_such(self):
        assert grade_of("plutchik.antipodal") is Grade.METAPHOR
        assert grade_of("hourglass") is Grade.METAPHOR
        assert grade_of("lovheim.cube") is Grade.SPECULATIVE

    def test_speculative_and_metaphor_constructs_are_not_citable(self):
        assert not all_evidence()["lovheim.cube"].citable
        assert not all_evidence()["plutchik.antipodal"].citable

    def test_contested_constructs_are_declared_not_resolved(self):
        assert grade_of("valence.bipolarity") is Grade.CONTESTED

    def test_every_entry_carries_a_citation(self):
        for key, entry in all_evidence().items():
            assert entry.cite.strip(), f"{key} has no citation"

    def test_non_established_entries_explain_themselves(self):
        for key, entry in all_evidence().items():
            if entry.grade not in CITABLE:
                assert entry.note.strip(), f"{key} is {entry.grade} but says nothing"
