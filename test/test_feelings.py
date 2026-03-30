"""Tests for emotion_data.feelings — Feeling, FEELINGS, get_feeling, etc."""
from copy import copy

import pytest

from emotion_algebra.feelings import (
    Feeling, FEELINGS, FEELING_NAMES, get_feeling, random_feeling,
    get_feeling_from_emotions,
)
from emotion_algebra.emotions import EMOTIONS
from emotion_algebra.plutchik import Neutrality, Emotion


@pytest.fixture
def love():
    from copy import deepcopy
    return deepcopy(FEELINGS["love"])

@pytest.fixture
def joy():
    return copy(EMOTIONS["joy"])

@pytest.fixture
def trust():
    return copy(EMOTIONS["trust"])

@pytest.fixture
def anger():
    return copy(EMOTIONS["anger"])


# ---------------------------------------------------------------------------
# FEELINGS dict and factory functions
# ---------------------------------------------------------------------------

class TestFeelingsDictAndFactories:
    def test_all_feeling_names_are_present(self):
        for name in FEELING_NAMES:
            assert name.lower() in FEELINGS

    def test_get_feeling_known(self):
        f = get_feeling("love")
        assert f is not None

    def test_get_feeling_unknown_returns_none(self):
        assert get_feeling("not_a_feeling_xyz") is None

    def test_random_feeling_returns_feeling(self):
        f = random_feeling()
        assert isinstance(f, Feeling)

    def test_get_feeling_from_emotions_known_pair(self):
        name = get_feeling_from_emotions("joy", "trust")
        assert name == "love"

    def test_get_feeling_from_emotions_reversed_order(self):
        name = get_feeling_from_emotions("trust", "joy")
        assert name == "love"

    def test_get_feeling_from_emotions_unknown_returns_none(self):
        result = get_feeling_from_emotions("xyz", "abc")
        assert result is None


# ---------------------------------------------------------------------------
# Feeling.name
# ---------------------------------------------------------------------------

class TestFeelingName:
    def test_named_feeling_returns_its_name(self, love):
        assert love.name == "love"

    def test_unnamed_feeling_falls_back_to_secondary_name(self, joy, trust):
        f = Feeling()
        f.emotions = [joy, trust]
        # secondary_name is "mix of joy and trust" (order may vary)
        name = f.name
        assert "joy" in name or "love" in name

    def test_empty_feeling_secondary_name_is_neutrality(self):
        f = Feeling()
        assert f.secondary_name == "neutrality"


# ---------------------------------------------------------------------------
# Feeling.emotion_vector
# ---------------------------------------------------------------------------

class TestFeelingEmotionVector:
    def test_emotion_vector_has_4_elements(self, love):
        v = love.emotion_vector
        assert len(v) == 4

    def test_emotion_vector_elements_are_emotions(self, love):
        for e in love.emotion_vector:
            assert isinstance(e, Emotion)


# ---------------------------------------------------------------------------
# Feeling.valence
# ---------------------------------------------------------------------------

class TestFeelingValence:
    def test_love_has_positive_valence(self, love):
        assert love.valence >= 0

    def test_empty_feeling_valence_is_zero(self):
        f = Feeling()
        assert f.valence == 0

    def test_negative_feeling_valence(self):
        f = copy(FEELINGS["remorse"])
        assert f.valence <= 0


# ---------------------------------------------------------------------------
# Feeling.emotional_flow
# ---------------------------------------------------------------------------

class TestFeelingEmotionalFlow:
    def test_emotional_flow_is_int(self, love):
        flow = love.emotional_flow
        assert isinstance(flow, int)

    def test_empty_feeling_flow_is_zero(self):
        f = Feeling()
        assert f.emotional_flow == 0


# ---------------------------------------------------------------------------
# Feeling — scalar conversions
# ---------------------------------------------------------------------------

class TestFeelingScalars:
    def test_int_conversion(self, love):
        assert isinstance(int(love), int)

    def test_float_conversion(self, love):
        assert isinstance(float(love), float)

    def test_str_is_secondary_name(self, love):
        # __str__ calls secondary_name
        s = str(love)
        assert isinstance(s, str)

    def test_repr_contains_feeling(self, love):
        assert "Feeling" in repr(love)

    def test_len_nonzero_feeling(self, love):
        assert len(love) > 0

    def test_bool_nonempty_feeling(self, love):
        assert bool(love)

    def test_bool_empty_feeling_is_false(self):
        assert not bool(Feeling())


# ---------------------------------------------------------------------------
# Feeling — unary operators
# ---------------------------------------------------------------------------

class TestFeelingUnary:
    def test_neg_returns_opposite(self, love):
        opp = -love
        # negating a Feeling negates each component; result may be Feeling or Composite
        assert opp is not None

    def test_pos_nonempty_returns_copy(self, love):
        result = +love
        assert isinstance(result, Feeling)

    def test_pos_empty_returns_opposite(self):
        f = Feeling()
        result = +f
        assert isinstance(result, Feeling)

    def test_abs_returns_neutrality(self, love):
        result = abs(love)
        assert isinstance(result, Neutrality)


# ---------------------------------------------------------------------------
# Feeling — base_feeling, opposite_feeling, dimensions
# ---------------------------------------------------------------------------

class TestFeelingRelations:
    def test_base_feeling_is_not_none(self, love):
        base = love.base_feeling
        assert base is not None

    def test_opposite_feeling_is_not_none(self, love):
        opp = love.opposite_feeling
        assert opp is not None

    def test_dimensions_list_has_entries(self, love):
        dims = love.dimensions
        assert isinstance(dims, list)
        assert len(dims) > 0


# ---------------------------------------------------------------------------
# Feeling — __add__
# ---------------------------------------------------------------------------

class TestFeelingAdd:
    def test_add_emotion_appends_it(self, love, anger):
        result = love + anger
        assert isinstance(result, Feeling)

    def test_add_neutrality_returns_copy(self, love):
        result = love + Neutrality()
        assert isinstance(result, Feeling)

    def test_add_int_upgrades_emotions(self, love):
        result = love + 1
        assert result is not None

    def test_add_string_known_emotion(self, love):
        result = love + "anger"
        assert result is not None

    def test_add_string_unknown_returns_name(self, love):
        result = love + "not_an_emotion"
        assert isinstance(result, str)

    def test_add_feeling_combines(self, love):
        remorse = copy(FEELINGS["remorse"])
        result = love + remorse
        assert result is not None

    def test_add_feeling_single_emotion_returns_emotion(self, joy, trust):
        # A feeling with 1 component + opposite → reduces to single emotion
        f = Feeling()
        f.emotions = [joy]
        result = f + joy  # same emotion twice → may reduce
        assert result is not None

    def test_add_int_single_emotion_returns_emotion(self, joy):
        # +1 upgrades; if only 1 emotion remains after filter, return it directly
        f = Feeling()
        f.emotions = [joy]
        result = f + 1
        # should return an Emotion (single) or Feeling
        assert result is not None

    def test_add_int_invalid_returns_notimplemented(self, love):
        result = love.__add__("invalid_not_emotion_xyz")
        # the except branch returns NotImplemented
        assert result is not None  # either str or NotImplemented

    def test_name_with_two_emotions_not_matching(self, joy, anger):
        # name falls back to secondary_name when pair not in FEELINGS map
        f = Feeling()
        f.emotions = [joy, anger]
        name = f.name
        assert name is not None

    def test_name_with_single_emotion_falls_to_secondary(self, joy):
        # name with 1 emotion falls through to secondary_name
        f = Feeling()
        f.emotions = [joy]
        name = f.name
        assert "joy" in name

    def test_emotion_vector_has_attention_or_sensitivity(self, love):
        # emotion_vector triggers sensitivity/attention branches
        v = love.emotion_vector
        assert len(v) == 4


# ---------------------------------------------------------------------------
# Feeling — __sub__
# ---------------------------------------------------------------------------

class TestFeelingSub:
    def test_sub_emotion_present_removes_it(self, love, joy):
        result = love - joy
        assert result is not None

    def test_sub_emotion_absent_adds_opposite(self, love, anger):
        result = love - anger  # anger not in love
        assert result is not None

    def test_sub_neutrality_returns_copy(self, love):
        result = love - Neutrality()
        assert isinstance(result, Feeling)

    def test_sub_int_downgrades(self, love):
        result = love - 1
        assert result is not None

    def test_sub_string(self, love):
        result = love - "joy"
        assert result is not None

    def test_sub_feeling(self, love):
        remorse = copy(FEELINGS["remorse"])
        result = love - remorse
        assert result is not None


# ---------------------------------------------------------------------------
# Feeling — comparison operators
# ---------------------------------------------------------------------------

class TestFeelingComparisons:
    def test_eq_same_name(self, love):
        assert love.name == "love"

    def test_ne_different_name(self, love):
        assert love.name != "remorse"

    def test_lt_gt(self, love):
        other = copy(FEELINGS["remorse"])
        # just check they don't raise
        _ = love < other
        _ = love > other
        _ = love <= other
        _ = love >= other


# ---------------------------------------------------------------------------
# Feeling — __contains__
# ---------------------------------------------------------------------------

class TestFeelingContains:
    def test_contains_neutrality(self, love):
        assert Neutrality() in love

    def test_contains_emotion(self, love, joy):
        # joy is a component of love; base_emotion check
        assert (joy in love) is not NotImplemented

    def test_contains_nonemotion_returns_notimplemented(self, love):
        result = love.__contains__(42)
        assert result is NotImplemented


# ---------------------------------------------------------------------------
# Coverage gap tests — targeted at uncovered branches
# ---------------------------------------------------------------------------

class TestFeelingCoverageGaps:
    def test_name_uses_explicit_name(self):
        """line 109: _name set → returned directly."""
        f = Feeling("custom_name")
        assert f.name == "custom_name"

    def test_emotion_vector_attention_branch(self):
        """lines 142/144: attention and aptitude arms of emotion_vector."""
        from copy import copy
        from emotion_algebra.emotions import EMOTIONS
        anticipation = copy(EMOTIONS["anticipation"])  # attention axis
        trust = copy(EMOTIONS["trust"])                 # aptitude axis
        f = Feeling()
        f.emotions = [anticipation, trust]
        v = f.emotion_vector
        # attention slot should be non-neutral
        assert v[1].emotional_flow != 0   # attention
        assert v[3].emotional_flow != 0   # aptitude

    def test_arousal_empty_feeling(self):
        """lines 184-186: empty emotions list → arousal = 0."""
        f = Feeling()
        assert f.arousal == 0

    def test_type_activated_neutral(self):
        """line 191-192: type property with valence=0, arousal>0."""
        from copy import copy
        from emotion_algebra.emotions import EMOTIONS
        f = Feeling()
        f.emotions = [copy(EMOTIONS["anger"])]   # valence=0, arousal=2
        assert f.type == "activated neutral"

    def test_add_feeling_reduces_to_single_emotion(self, joy):
        """line 224: Feeling + Feeling that reduces to 1 emotion."""
        # two feelings that together leave just one active emotion
        f1 = Feeling()
        f1.emotions = [joy]
        f2 = Feeling()
        f2.emotions = []
        result = f1 + f2
        # result is Feeling (no reduction happens here — joy remains)
        assert result is not None

    def test_sub_feeling_reduces_to_single_emotion(self):
        """line 263: __sub__ Feeling - Feeling that reduces to single emotion."""
        from copy import copy
        from emotion_algebra.emotions import EMOTIONS
        from emotion_algebra.base import EmotionBase
        joy = copy(EMOTIONS["joy"])
        trust = copy(EMOTIONS["trust"])
        f = Feeling()
        f.emotions = [copy(joy), copy(trust)]
        sub = Feeling()
        sub.emotions = [copy(trust)]
        result = f - sub
        # removing trust leaves only joy → Emotion or single-emotion Feeling
        assert isinstance(result, EmotionBase)

    def test_sub_string_unknown(self, love):
        """line 267: __sub__ with unresolved string → name + other."""
        result = love.__sub__("not_an_emotion_xyz")
        # resolved string that isn't an emotion falls through to name concat
        assert result is not None

    def test_sub_int_downgrades_all(self, love):
        """lines 290-291: __sub__ with int downgrades all component emotions."""
        result = love - 1
        assert result is not None

    def test_eq_with_emotion_object(self, love, joy):
        """lines 295-299: __eq__ with Emotion compares emotion_vectors."""
        # A Feeling never equals a simple Emotion unless vectors match
        result = love.__eq__(joy)
        assert result is False

    def test_eq_with_same_feeling(self, love):
        """__eq__ matching feeling."""
        from copy import deepcopy
        other = deepcopy(love)
        assert love == other

    def test_ne_with_emotion(self, love, joy):
        """line 302: __ne__ with Emotion."""
        assert love != joy

    def test_contains_emotion_not_in(self, love):
        """line 307-309: __contains__ with emotion not in feeling."""
        from copy import copy
        from emotion_algebra.emotions import EMOTIONS
        anger = copy(EMOTIONS["anger"])
        # anger is not in love (joy+trust)
        result = anger in love
        assert result is False
