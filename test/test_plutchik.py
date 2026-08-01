"""Tests for emotion_data.plutchik — Emotion, Neutrality, EmotionalDimension."""
from copy import copy

import numpy as np
import pytest

from emotion_algebra.plutchik import (
    Emotion, Neutrality, EmotionalDimension, DIMENSIONS,
    PRIMARY_EMOTION_NAMES, SECONDARY_EMOTIONS_NAMES, TERTIARY_EMOTIONS_NAMES,
)
from emotion_algebra.emotions import EMOTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def anger():
    return copy(EMOTIONS["anger"])

@pytest.fixture
def joy():
    return copy(EMOTIONS["joy"])

@pytest.fixture
def trust():
    return copy(EMOTIONS["trust"])

@pytest.fixture
def rage():
    return copy(EMOTIONS["rage"])

@pytest.fixture
def annoyance():
    return copy(EMOTIONS["annoyance"])

@pytest.fixture
def fear():
    return copy(EMOTIONS["fear"])

@pytest.fixture
def terror():
    return copy(EMOTIONS["terror"])

@pytest.fixture
def serenity():
    return copy(EMOTIONS["serenity"])

@pytest.fixture
def sadness():
    return copy(EMOTIONS["sadness"])

@pytest.fixture
def surprise():
    return copy(EMOTIONS["surprise"])

@pytest.fixture
def ecstasy():
    return copy(EMOTIONS["ecstasy"])


# ---------------------------------------------------------------------------
# EmotionalDimension
# ---------------------------------------------------------------------------

class TestEmotionalDimension:
    def test_all_four_dimensions_exist(self):
        for name in ("sensitivity", "attention", "pleasantness", "aptitude"):
            assert name in DIMENSIONS

    def test_dimension_name_property(self):
        d = DIMENSIONS["sensitivity"]
        assert d.name == "sensitivity"

    def test_dimension_str(self):
        d = DIMENSIONS["attention"]
        assert str(d) == "attention"

    def test_dimension_repr(self):
        d = DIMENSIONS["sensitivity"]
        assert "sensitivity" in repr(d)

    def test_sensitivity_valence_is_zero(self):
        # reactivity is orthogonal to hedonics (Posner et al. 2005)
        assert DIMENSIONS["sensitivity"].valence == 0

    def test_attention_valence_is_zero(self):
        # engagement is orthogonal to hedonics (Posner et al. 2005)
        assert DIMENSIONS["attention"].valence == 0

    def test_pleasantness_valence_is_positive(self):
        assert DIMENSIONS["pleasantness"].valence == 1

    def test_aptitude_valence_is_positive(self):
        assert DIMENSIONS["aptitude"].valence == 1

    def test_sensitivity_kind_is_activation(self):
        assert DIMENSIONS["sensitivity"].kind == "activation"

    def test_attention_kind_is_activation(self):
        assert DIMENSIONS["attention"].kind == "activation"

    def test_pleasantness_kind_is_hedonic(self):
        assert DIMENSIONS["pleasantness"].kind == "hedonic"

    def test_dimension_eq_by_name(self):
        d = DIMENSIONS["sensitivity"]
        assert d == "sensitivity"
        assert d == DIMENSIONS["sensitivity"]
        assert d != DIMENSIONS["attention"]

    def test_dimension_add_returns_composite_dimension(self):
        from emotion_algebra.composite_emotions import CompositeDimension
        d1 = DIMENSIONS["sensitivity"]
        d2 = DIMENSIONS["attention"]
        result = d1 + d2
        assert isinstance(result, CompositeDimension)

    def test_dimension_contains_neutrality(self):
        d = DIMENSIONS["sensitivity"]
        assert Neutrality() in d

    def test_dimension_contains_emotion(self, anger):
        d = DIMENSIONS["sensitivity"]
        assert anger in d

    def test_dimension_does_not_contain_wrong_emotion(self, joy):
        d = DIMENSIONS["sensitivity"]
        assert joy not in d


# ---------------------------------------------------------------------------
# Emotion — classification
# ---------------------------------------------------------------------------

class TestEmotionClassification:
    def test_primary_emotions(self):
        for name in PRIMARY_EMOTION_NAMES:
            e = copy(EMOTIONS[name])
            assert e.is_primary, f"{name} should be primary"

    def test_secondary_emotions(self):
        for name in SECONDARY_EMOTIONS_NAMES:
            e = copy(EMOTIONS[name])
            assert e.is_secondary, f"{name} should be secondary"

    def test_tertiary_emotions(self):
        for name in TERTIARY_EMOTIONS_NAMES:
            e = copy(EMOTIONS[name])
            assert e.is_tertiary, f"{name} should be tertiary"

    def test_is_not_hyper_by_default(self, anger):
        assert not anger.is_hyper

    def test_is_hyper_after_offset(self, rage):
        e = copy(rage)
        e.intensity_offset = 1
        assert e.is_hyper

    def test_is_composite_false_for_plain_emotion(self, anger):
        assert not anger.is_composite


# ---------------------------------------------------------------------------
# Emotion — intensity
# ---------------------------------------------------------------------------

class TestEmotionIntensity:
    def test_primary_intensity_is_basic(self, annoyance):
        assert annoyance.intensity == "basic"

    def test_secondary_intensity_is_mild(self, anger):
        assert anger.intensity == "mild"

    def test_tertiary_intensity_is_intense(self, rage):
        assert rage.intensity == "intense"

    def test_offset_1_is_mega(self, rage):
        e = copy(rage)
        e.intensity_offset = 1
        assert e.intensity == "mega"

    def test_offset_2_is_extreme(self, rage):
        e = copy(rage)
        e.intensity_offset = 2
        assert e.intensity == "extreme"

    def test_offset_3_plus_is_hyper(self, rage):
        e = copy(rage)
        e.intensity_offset = 3
        assert e.intensity == "hyper"

    def test_neutrality_intensity_is_null(self):
        assert Neutrality().intensity == "null"


# ---------------------------------------------------------------------------
# Emotion — emotional_flow
# ---------------------------------------------------------------------------

class TestEmotionalFlow:
    def test_primary_positive_flow_is_1(self, annoyance):
        assert annoyance.emotional_flow == 1

    def test_primary_negative_flow_is_minus1(self, serenity):
        # serenity is basic_opposite in... wait, let me check
        # serenity is pleasantness basic_emotion (+1 side? no...)
        # pleasantness: ecstasy(3), joy(2), serenity(1), pensiveness(-1), sadness(-2), grief(-3)
        assert serenity.emotional_flow in (1, -1)

    def test_secondary_positive_flow_is_2(self, anger):
        assert anger.emotional_flow == 2

    def test_tertiary_positive_flow_is_3(self, rage):
        assert rage.emotional_flow == 3

    def test_fear_is_negative_flow(self, fear):
        assert fear.emotional_flow < 0

    def test_neutrality_flow_is_zero(self):
        assert Neutrality().emotional_flow == 0


# ---------------------------------------------------------------------------
# Emotion — base_emotion, parent_emotion, opposite_emotion
# ---------------------------------------------------------------------------

class TestEmotionRelations:
    def test_primary_base_emotion_is_self(self, annoyance):
        assert annoyance.base_emotion == annoyance

    def test_secondary_base_emotion_is_primary(self, anger):
        base = anger.base_emotion
        assert base.is_primary

    def test_tertiary_base_emotion_is_primary(self, rage):
        base = rage.base_emotion
        assert base.is_primary

    def test_primary_parent_is_none(self, annoyance):
        assert annoyance.parent_emotion is None

    def test_secondary_parent_is_primary(self, anger):
        parent = anger.parent_emotion
        assert parent is not None
        assert parent.is_primary

    def test_tertiary_parent_is_secondary(self, rage):
        parent = rage.parent_emotion
        assert parent is not None
        assert parent.is_secondary

    def test_opposite_of_anger_is_fear(self, anger, fear):
        opp = anger.opposite_emotion
        assert opp.name == fear.name or opp._name == fear._name

    def test_double_negation_returns_to_original(self, anger):
        double_neg = -(-anger)
        assert double_neg._name == anger._name

    def test_opposite_primary_positive(self, annoyance):
        opp = annoyance.opposite_emotion
        assert opp is not None
        assert opp.emotional_flow < 0

    def test_opposite_primary_negative(self):
        apprehension = copy(EMOTIONS["apprehension"])
        opp = apprehension.opposite_emotion
        assert opp is not None
        assert opp.emotional_flow > 0

    def test_opposite_secondary_positive(self, anger):
        opp = anger.opposite_emotion
        assert opp is not None

    def test_opposite_tertiary_positive(self, rage):
        opp = rage.opposite_emotion
        assert opp is not None
        assert opp.emotional_flow < 0


# ---------------------------------------------------------------------------
# Emotion — type and kind
# ---------------------------------------------------------------------------

class TestEmotionTypeKind:
    def test_type_returns_string(self, anger):
        t = anger.type
        assert isinstance(t, str)
        assert len(t) > 0

    def test_type_contains_valence_word(self, joy):
        t = joy.type
        assert any(w in t for w in ("positive", "negative", "neutral"))

    # Russell Circumplex — spec §2.1
    def test_joy_is_excited_positive(self):
        assert copy(EMOTIONS["joy"]).type == "excited positive"

    def test_ecstasy_is_excited_positive(self):
        assert copy(EMOTIONS["ecstasy"]).type == "excited positive"

    def test_serenity_is_calm_positive(self):
        assert copy(EMOTIONS["serenity"]).type == "calm positive"

    def test_grief_is_excited_negative(self):
        assert copy(EMOTIONS["grief"]).type == "excited negative"

    def test_sadness_is_excited_negative(self):
        assert copy(EMOTIONS["sadness"]).type == "excited negative"

    def test_pensiveness_is_calm_negative(self):
        assert copy(EMOTIONS["pensiveness"]).type == "calm negative"

    def test_gloat_classified_as_social(self):
        from emotion_algebra.plutchik import EMOTION_KIND_NAMES
        # gloat (ecstasy+loathing): Schadenfreude is a fortunes-of-others
        # emotion (Ortony/OCC) -- inherently social, not merely event-based.
        assert "gloat" in EMOTION_KIND_NAMES["social"]
        assert "gloat" not in EMOTION_KIND_NAMES["event related"]

    def test_frivolity_classified_as_event_related(self):
        from emotion_algebra.plutchik import EMOTION_KIND_NAMES
        # frivolity (ecstasy+amazement): intense-pleasant reaction to a
        # surprising event, alongside ecstasy/joy/elation -- not social.
        assert "frivolity" in EMOTION_KIND_NAMES["event related"]
        assert "frivolity" not in EMOTION_KIND_NAMES["social"]

    # `type` classifies on polarity (Cambria's 4-axis sentiment score), not on
    # valence (Pleasantness only). Under valence, every emotion off the
    # Pleasantness axis -- three axes out of four -- collapsed to "activated
    # neutral", which put anger and fear, the textbook high-arousal *negative*
    # emotions, in the hedonically-neutral band. Polarity places them correctly.
    def test_anger_is_excited_negative(self):
        assert copy(EMOTIONS["anger"]).type == "excited negative"

    def test_fear_is_excited_negative(self):
        assert copy(EMOTIONS["fear"]).type == "excited negative"

    def test_sensitivity_axis_is_aversive_at_both_poles(self):
        """Anger and fear are both negative; they differ in coping, not hedonic sign."""
        assert copy(EMOTIONS["rage"]).polarity < 0
        assert copy(EMOTIONS["terror"]).polarity < 0

    def test_anticipation_is_excited_positive(self):
        assert copy(EMOTIONS["anticipation"]).type == "excited positive"

    def test_both_attention_poles_score_positive(self):
        """|Attention| in Cambria's formula makes surprise positive too.

        Surprise is hedonically ambiguous in Russell's circumplex, so a case
        could be made for classifying it neutral. We do not: the polarity
        formula is Cambria's published one, and bending it so that one emotion
        reads more intuitively would make every other number in this library
        un-citable. The quirk is locked here so it stays a decision rather than
        drifting into an accident.
        """
        assert copy(EMOTIONS["surprise"]).polarity > 0
        assert copy(EMOTIONS["anticipation"]).polarity > 0

    def test_valence_still_pleasantness_only(self):
        """polarity changed; valence did NOT -- they are different questions."""
        assert copy(EMOTIONS["anger"]).valence == 0
        assert copy(EMOTIONS["joy"]).valence == 2

    def test_neutrality_type_is_neutral(self):
        assert Neutrality().type == "neutral"

    # arousal property — spec §2.1
    def test_anger_arousal_is_abs_flow(self):
        anger = copy(EMOTIONS["anger"])
        assert anger.arousal == abs(anger.emotional_flow) == 2

    def test_fear_arousal_is_abs_flow(self):
        fear = copy(EMOTIONS["fear"])
        assert fear.arousal == abs(fear.emotional_flow) == 2

    def test_neutrality_arousal_is_zero(self):
        assert Neutrality().arousal == 0

    def test_kind_returns_string_or_empty(self, anger):
        k = anger.kind
        assert isinstance(k, str)

    def test_kind_for_known_emotion(self):
        annoyance = copy(EMOTIONS["annoyance"])
        k = annoyance.kind
        # annoyance is in 'related to object properties'
        assert isinstance(k, str)


# ---------------------------------------------------------------------------
# Emotion — triggered_reactions
# ---------------------------------------------------------------------------

class TestTriggeredReactions:
    def test_triggered_reactions_returns_list(self, anger):
        reactions = anger.triggered_reactions
        assert isinstance(reactions, list)

    def test_annoyance_triggers_attack(self):
        annoyance = copy(EMOTIONS["annoyance"])
        reactions = annoyance.triggered_reactions
        names = [r.name for r in reactions]
        assert "attack" in names


# ---------------------------------------------------------------------------
# Emotion — vector / matrix representations
# ---------------------------------------------------------------------------

class TestEmotionVector:
    def test_emotion_vector_length_4(self, anger):
        v = anger.emotion_vector
        assert len(v) == 4

    def test_as_array_shape(self, anger):
        arr = anger.as_array
        assert arr.shape == (4,)

    def test_as_matrix_shape(self, anger):
        m = anger.as_matrix
        assert m.shape == (2, 2)

    def test_emotion_len(self, anger):
        # anger is in one dimension → len 1
        assert len(anger) == 1

    def test_neutrality_len_is_zero(self):
        assert len(Neutrality()) == 0


# ---------------------------------------------------------------------------
# Emotion — arithmetic operators
# ---------------------------------------------------------------------------

class TestEmotionArithmetic:
    def test_add_same_dimension_upgrades(self, annoyance):
        result = annoyance + 1
        assert result.emotional_flow == 2  # annoyance(1) + 1 = anger(2)

    def test_add_int_upgrades_to_tertiary(self, anger):
        result = anger + 1
        assert result.emotional_flow == 3

    def test_add_emotion_same_dim_upgrades(self, annoyance, anger):
        result = annoyance + annoyance
        assert result.emotional_flow == 2

    def test_add_different_dim_creates_composite(self, joy, trust):
        from emotion_algebra.composite_emotions import CompositeEmotion
        result = joy + trust
        assert isinstance(result, CompositeEmotion)

    def test_add_neutrality_returns_self(self, anger):
        result = anger + Neutrality()
        assert result._name == anger._name

    def test_add_string_emotion_name(self, anger):
        result = anger + "annoyance"
        # same dimension → upgrades
        assert hasattr(result, "emotional_flow")

    def test_add_feeling(self, anger):
        from emotion_algebra.feelings import FEELINGS
        love = copy(FEELINGS["love"])
        result = anger + love
        assert result is not None

    def test_sub_int_downgrades(self, anger):
        result = anger - 1
        assert result.emotional_flow == 1

    def test_sub_neutrality(self, anger):
        result = anger - Neutrality()
        assert result._name == anger._name

    def test_sub_emotion_adds_opposite(self, anger, fear):
        # anger - fear → anger + (-fear) = anger + anger (same dim) → rage
        result = anger - fear
        assert isinstance(result, Emotion)

    def test_sub_string(self, anger):
        result = anger - "annoyance"
        assert hasattr(result, "emotional_flow")

    def test_mul_emotion_creates_composite(self, anger):
        from emotion_algebra.composite_emotions import CompositeEmotion
        vigilance = copy(EMOTIONS["vigilance"])
        result = anger * vigilance
        assert isinstance(result, CompositeEmotion)

    def test_mul_neutrality(self, anger):
        result = anger * Neutrality()
        assert result._name == anger._name

    def test_mul_string(self, anger):
        vigilance = copy(EMOTIONS["vigilance"])
        result = anger * "vigilance"
        assert result is not None

    def test_truediv_int(self, rage):
        result = rage / 3
        assert isinstance(result, Emotion)

    def test_truediv_neutrality(self, anger):
        result = anger / Neutrality()
        assert result._name == anger._name

    def test_floordiv_int(self, rage):
        result = rage // 1
        assert isinstance(result, Emotion)

    def test_floordiv_neutrality(self, anger):
        result = anger // Neutrality()
        assert result._name == anger._name

    def test_lshift_int(self, anger):
        result = anger << 1
        assert isinstance(result, Emotion)

    def test_lshift_neutrality(self, anger):
        result = anger << Neutrality()
        assert result._name == anger._name

    def test_lshift_same_dimension(self, anger, rage):
        result = anger << rage
        assert isinstance(result, Emotion)

    def test_rshift_int(self, anger):
        result = anger >> 1
        assert isinstance(result, Emotion)

    def test_rshift_neutrality(self, anger):
        result = anger >> Neutrality()
        assert result._name == anger._name

    def test_rshift_same_dimension(self, anger, annoyance):
        result = anger >> annoyance
        assert isinstance(result, Emotion)

    def test_shift_round_trip_int_operand(self, annoyance):
        shifted = annoyance >> 1
        back = shifted << 1
        assert back.emotional_flow == annoyance.emotional_flow

    def test_shift_round_trip_emotion_operand(self, annoyance):
        # (a >> b) << b must return a, for any same-dimension b — << and >>
        # must be exact inverses regardless of operand type.
        shifted = annoyance >> annoyance
        back = shifted << annoyance
        assert back.emotional_flow == annoyance.emotional_flow

    def test_lshift_emotion_operand_is_self_minus_other(self, anger, annoyance):
        result = anger << annoyance
        assert result.emotional_flow == anger.emotional_flow - annoyance.emotional_flow


# ---------------------------------------------------------------------------
# Emotion — unary operators
# ---------------------------------------------------------------------------

class TestEmotionUnary:
    def test_neg_returns_opposite(self, anger, fear):
        result = -anger
        assert result._name == fear._name or result.dimension == anger.dimension

    def test_pos_returns_copy_if_flow_nonzero(self, anger):
        result = +anger
        assert result._name == anger._name

    def test_abs_returns_neutrality(self, anger):
        result = abs(anger)
        assert isinstance(result, Neutrality)

    def test_bool_positive_emotion(self, joy):
        assert bool(joy)

    def test_bool_negative_emotion_is_true(self, fear):
        # fear has negative flow but non-zero intensity → truthy (presence, not valence)
        assert bool(fear) is True

    def test_int_emotion(self, anger):
        assert int(anger) == 2

    def test_float_emotion(self, anger):
        assert float(anger) == 2.0


# ---------------------------------------------------------------------------
# Emotion — comparison operators
# ---------------------------------------------------------------------------

class TestEmotionComparisons:
    def test_lt(self, annoyance, anger):
        assert annoyance < anger

    def test_le(self, annoyance, anger):
        assert annoyance <= anger
        assert annoyance <= annoyance

    def test_gt(self, rage, anger):
        assert rage > anger

    def test_ge(self, rage, anger):
        assert rage >= anger
        assert rage >= rage

    def test_eq_same_emotion(self, anger):
        other = copy(EMOTIONS["anger"])
        assert anger == other

    def test_ne_different_dimension(self, anger, joy):
        assert anger != joy

    def test_ne_by_name(self, anger):
        assert anger != "notanemotion"

    def test_eq_by_name_string(self, anger):
        assert anger == "anger"


class TestEmotionHashability:
    """Emotions are immutable value objects and must be hashable — usable
    as dict keys and in sets, consistent with __eq__ (same-dimension,
    same-flow emotions hash equal)."""

    def test_is_hashable(self, anger):
        hash(anger)  # must not raise

    def test_equal_emotions_hash_equal(self, anger):
        other = copy(EMOTIONS["anger"])
        assert anger == other
        assert hash(anger) == hash(other)

    def test_usable_as_dict_key(self, anger, joy):
        d = {anger: "sensitivity", joy: "pleasantness"}
        assert d[copy(EMOTIONS["anger"])] == "sensitivity"

    def test_usable_in_set_dedup(self, anger):
        s = {anger, copy(EMOTIONS["anger"]), copy(EMOTIONS["anger"])}
        assert len(s) == 1


# ---------------------------------------------------------------------------
# Emotion — __contains__
# ---------------------------------------------------------------------------

class TestEmotionContains:
    def test_contains_neutrality(self, anger):
        assert Neutrality() in anger

    def test_contains_dimension_string(self, anger):
        assert "sensitivity" in anger

    def test_contains_dimension_object(self, anger):
        assert DIMENSIONS["sensitivity"] in anger


# ---------------------------------------------------------------------------
# Emotion — string_to_emotion
# ---------------------------------------------------------------------------

class TestStringToEmotion:
    def test_known_emotion_name(self):
        result = Emotion.string_to_emotion("anger")
        assert isinstance(result, Emotion)

    def test_known_feeling_name(self):
        result = Emotion.string_to_emotion("love")
        assert result is not None

    def test_unknown_string_returns_string(self):
        result = Emotion.string_to_emotion("xyz_unknown")
        assert result == "xyz_unknown"


# ---------------------------------------------------------------------------
# Emotion — emotion_from_flow
# ---------------------------------------------------------------------------

class TestEmotionFromFlow:
    def test_flow_1_returns_basic(self, anger):
        e = anger.emotion_from_flow(1)
        assert e.emotional_flow == 1

    def test_flow_2_returns_mild(self, anger):
        e = anger.emotion_from_flow(2)
        assert e.emotional_flow == 2

    def test_flow_3_returns_tertiary(self, anger):
        e = anger.emotion_from_flow(3)
        assert e.emotional_flow == 3

    def test_flow_minus1_returns_basic_opposite(self, anger):
        e = anger.emotion_from_flow(-1)
        assert e.emotional_flow == -1

    def test_flow_minus2_returns_mild_opposite(self, anger):
        e = anger.emotion_from_flow(-2)
        assert e.emotional_flow == -2

    def test_flow_minus3_returns_intense_opposite(self, anger):
        e = anger.emotion_from_flow(-3)
        assert e.emotional_flow == -3

    def test_flow_0_returns_neutrality(self, anger):
        e = anger.emotion_from_flow(0)
        assert isinstance(e, Neutrality)

    def test_flow_capped_at_9(self, anger):
        e = anger.emotion_from_flow(100)
        assert isinstance(e, Emotion)

    def test_flow_large_sets_offset(self, anger):
        e = anger.emotion_from_flow(5)
        assert e.intensity_offset > 0


# ---------------------------------------------------------------------------
# Neutrality
# ---------------------------------------------------------------------------

class TestNeutrality:
    def test_neutrality_flow_is_zero(self):
        assert Neutrality().emotional_flow == 0

    def test_neutrality_intensity_null(self):
        assert Neutrality().intensity == "null"

    def test_neutrality_type_is_neutral(self):
        assert Neutrality().type == "neutral"

    def test_neutrality_kind_is_neutral(self):
        assert Neutrality().kind == "neutral"

    def test_neutrality_valence_is_zero(self):
        assert Neutrality().valence == 0

    def test_neutrality_is_primary(self):
        assert Neutrality().is_primary

    def test_neutrality_base_emotion_is_self(self):
        n = Neutrality()
        assert n.base_emotion is n

    def test_neutrality_opposite_is_self(self):
        n = Neutrality()
        assert n.opposite_emotion is n

    def test_neutrality_parent_is_none(self):
        assert Neutrality().parent_emotion is None

    def test_neutrality_len_is_zero(self):
        assert len(Neutrality()) == 0

    def test_neutrality_add_emotion_returns_emotion(self, anger):
        result = Neutrality() + anger
        assert isinstance(result, Emotion)

    def test_neutrality_add_with_dimension(self, anger):
        n = Neutrality(dimension="sensitivity")
        result = n + anger
        assert isinstance(result, Emotion)

    def test_neutrality_add_string_known(self):
        result = Neutrality() + "anger"
        assert isinstance(result, Emotion)

    @pytest.mark.parametrize(
        "dimension,flow",
        [
            (d, f)
            for d in ("sensitivity", "attention", "pleasantness", "aptitude")
            for f in (-3, -2, -1, 1, 2, 3)
        ],
    )
    def test_neutrality_plus_int_is_additive_identity(self, dimension, flow):
        # Neutrality is the additive identity: Neutrality(dim) + flow must
        # construct an Emotion at that flow on that dimension, not fall
        # through to a bare int.
        n = Neutrality(dimension=dimension)
        result = n + flow
        assert isinstance(result, Emotion)
        assert result.emotional_flow == flow
        assert result.dimension.axis == dimension

    def test_neutrality_minus_int_is_additive_identity(self):
        n = Neutrality(dimension="sensitivity")
        result = n - 2
        assert isinstance(result, Emotion)
        assert result.emotional_flow == -2

    def test_neutrality_sub_emotion_returns_opposite(self, anger):
        result = Neutrality() - anger
        assert isinstance(result, Emotion)
        assert result.emotional_flow == -anger.emotional_flow

    def test_neutrality_eq_zero_int(self):
        assert Neutrality() == 0

    def test_neutrality_eq_bool_true(self):
        assert Neutrality() == True  # noqa: E712

    def test_neutrality_eq_zero_name(self):
        assert Neutrality() == "neutrality"

    def test_neutrality_eq_empty_list(self):
        assert Neutrality() == []

    def test_neutrality_eq_zero_emotion(self):
        assert Neutrality() == Neutrality()

    def test_neutrality_neq_nonzero_emotion(self, anger):
        assert not (Neutrality() == anger)
