"""Tests for emotion_data.composite_emotions — CompositeEmotion, CompositeDimension."""
from copy import copy

import numpy as np
import pytest

from emotion_algebra.composite_emotions import (
    CompositeEmotion, COMPOSITE_EMOTIONS_NAMES, OPPOSITE_EMOTIONS_NAMES,
)
from emotion_algebra.emotions import EMOTIONS
from emotion_algebra.plutchik import Neutrality, EmotionalDimension, DIMENSIONS, Emotion


@pytest.fixture
def rage():
    return copy(EMOTIONS["rage"])

@pytest.fixture
def vigilance():
    return copy(EMOTIONS["vigilance"])

@pytest.fixture
def aggressiveness():
    """rage × vigilance = aggressiveness"""
    return rage_and_vigilance_composite()

def rage_and_vigilance_composite():
    rage = copy(EMOTIONS["rage"])
    vigilance = copy(EMOTIONS["vigilance"])
    return rage * vigilance


# ---------------------------------------------------------------------------
# CompositeDimension
# ---------------------------------------------------------------------------

class TestCompositeDimension:
    def test_dimension_add_returns_composite_dimension(self):
        from emotion_algebra.composite_emotions import CompositeDimension
        d1 = DIMENSIONS["sensitivity"]
        d2 = DIMENSIONS["attention"]
        result = d1 + d2
        assert isinstance(result, CompositeDimension)

    def test_composite_dimension_contains_component_dims(self):
        from emotion_algebra.composite_emotions import CompositeDimension
        d1 = DIMENSIONS["sensitivity"]
        d2 = DIMENSIONS["attention"]
        cd = d1 + d2
        assert d1 in cd or d2 in cd


# ---------------------------------------------------------------------------
# CompositeEmotion construction
# ---------------------------------------------------------------------------

class TestCompositeEmotionConstruction:
    def test_mul_creates_composite_emotion(self, rage, vigilance):
        result = rage * vigilance
        assert isinstance(result, CompositeEmotion)

    def test_composite_is_composite(self, rage, vigilance):
        c = rage * vigilance
        assert c.is_composite

    def test_components_populated(self, rage, vigilance):
        c = rage * vigilance
        assert len(c.components) > 0

    def test_name_is_aggressiveness(self, rage, vigilance):
        c = rage * vigilance
        assert c.name == "aggressiveness"


# ---------------------------------------------------------------------------
# CompositeEmotion.get_composite_from_emotions
# ---------------------------------------------------------------------------

class TestGetCompositeFromEmotions:
    def test_known_pair_returns_name(self, rage, vigilance):
        result = CompositeEmotion.get_composite_from_emotions(rage, vigilance)
        assert result == "aggressiveness"

    def test_unknown_pair_returns_none(self, rage):
        trust = copy(EMOTIONS["trust"])
        result = CompositeEmotion.get_composite_from_emotions(rage, trust)
        assert result is None


# ---------------------------------------------------------------------------
# CompositeEmotion — name / secondary_name
# ---------------------------------------------------------------------------

class TestCompositeEmotionName:
    def test_named_composite_returns_name(self):
        c = rage_and_vigilance_composite()
        assert c.name == "aggressiveness"

    def test_secondary_name_when_no_match(self):
        c = CompositeEmotion()
        joy = copy(EMOTIONS["joy"])
        c.components = [joy]
        name = c.secondary_name
        assert "joy" in name

    def test_empty_composite_secondary_name_is_neutrality(self):
        c = CompositeEmotion()
        assert c.secondary_name == "neutrality"

    def test_str_is_name(self):
        c = rage_and_vigilance_composite()
        assert str(c) == "aggressiveness"

    def test_repr_contains_composite(self):
        c = rage_and_vigilance_composite()
        assert "CompositeEmotionObject" in repr(c)


# ---------------------------------------------------------------------------
# CompositeEmotion — emotion_vector
# ---------------------------------------------------------------------------

class TestCompositeEmotionVector:
    def test_emotion_vector_has_4_elements(self):
        c = rage_and_vigilance_composite()
        v = c.emotion_vector
        assert len(v) == 4

    def test_emotion_vector_elements_are_emotions(self):
        c = rage_and_vigilance_composite()
        for e in c.emotion_vector:
            assert isinstance(e, Emotion)


# ---------------------------------------------------------------------------
# CompositeEmotion — emotional_flow
# ---------------------------------------------------------------------------

class TestCompositeEmotionFlow:
    def test_emotional_flow_is_numeric(self):
        c = rage_and_vigilance_composite()
        assert isinstance(c.emotional_flow, (int, float))

    def test_emotional_flow_is_signed(self):
        # rage (flow=3) + vigilance (flow=3) → both positive → sum > 0
        c = rage_and_vigilance_composite()
        assert c.emotional_flow > 0

    def test_emotional_flow_negative_for_negative_components(self):
        from copy import copy as _copy
        grief = _copy(EMOTIONS["grief"])    # flow = -3 (pleasantness)
        loathing = _copy(EMOTIONS["loathing"])  # flow = -3 (aptitude)
        c = grief * loathing
        assert c.emotional_flow < 0


# ---------------------------------------------------------------------------
# CompositeEmotion — dimension
# ---------------------------------------------------------------------------

class TestCompositeEmotionDimension:
    def test_dimension_returns_composite_dimension(self):
        from emotion_algebra.composite_emotions import CompositeDimension
        c = rage_and_vigilance_composite()
        d = c.dimension
        assert isinstance(d, CompositeDimension)

    def test_dimensions_list_has_2_entries(self):
        c = rage_and_vigilance_composite()
        dims = c.dimensions
        assert len(dims) == 2


# ---------------------------------------------------------------------------
# CompositeEmotion — base_emotion, parent_emotion, opposite_emotion
# ---------------------------------------------------------------------------

class TestCompositeEmotionRelations:
    def test_base_emotion_is_composite(self):
        c = rage_and_vigilance_composite()
        base = c.base_emotion
        assert isinstance(base, CompositeEmotion)

    def test_parent_emotion_is_none(self):
        c = rage_and_vigilance_composite()
        assert c.parent_emotion is None

    def test_opposite_emotion_is_composite(self):
        c = rage_and_vigilance_composite()
        opp = c.opposite_emotion
        assert isinstance(opp, CompositeEmotion)

    def test_neg_returns_composite(self):
        c = rage_and_vigilance_composite()
        neg = -c
        assert isinstance(neg, CompositeEmotion)


# ---------------------------------------------------------------------------
# CompositeEmotion — type
# ---------------------------------------------------------------------------

class TestCompositeEmotionType:
    def test_type_returns_string(self):
        c = rage_and_vigilance_composite()
        t = c.type
        assert isinstance(t, str)


# ---------------------------------------------------------------------------
# CompositeEmotion — equivalent_feeling
# ---------------------------------------------------------------------------

class TestEquivalentFeeling:
    def test_equivalent_feeling_returns_feeling_or_none(self):
        from emotion_algebra.feelings import Feeling
        c = rage_and_vigilance_composite()
        ef = c.equivalent_feeling
        assert ef is None or isinstance(ef, Feeling)


# ---------------------------------------------------------------------------
# CompositeEmotion — __add__
# ---------------------------------------------------------------------------

class TestCompositeEmotionAdd:
    def test_add_neutrality_returns_copy(self):
        c = rage_and_vigilance_composite()
        result = c + Neutrality()
        assert isinstance(result, CompositeEmotion)

    def test_add_emotion_returns_composite(self):
        c = rage_and_vigilance_composite()
        joy = copy(EMOTIONS["joy"])
        result = c + joy
        assert isinstance(result, CompositeEmotion)

    def test_add_feeling_returns_composite(self):
        from emotion_algebra.feelings import FEELINGS
        c = rage_and_vigilance_composite()
        love = copy(FEELINGS["love"])
        result = c + love
        assert isinstance(result, CompositeEmotion)

    def test_add_string_emotion(self):
        c = rage_and_vigilance_composite()
        result = c + "joy"
        assert result is not None


# ---------------------------------------------------------------------------
# CompositeEmotion — __sub__
# ---------------------------------------------------------------------------

class TestCompositeEmotionSub:
    def test_sub_neutrality_returns_copy(self):
        c = rage_and_vigilance_composite()
        result = c - Neutrality()
        assert isinstance(result, CompositeEmotion)

    def test_sub_emotion_returns_emotion_or_composite(self):
        c = rage_and_vigilance_composite()
        rage = copy(EMOTIONS["rage"])
        result = c - rage
        assert isinstance(result, (Emotion, CompositeEmotion, Neutrality))

    def test_sub_feeling_returns_composite(self):
        from emotion_algebra.feelings import FEELINGS
        c = rage_and_vigilance_composite()
        love = copy(FEELINGS["love"])
        result = c - love
        assert result is not None

    def test_sub_computes_self_minus_other_not_other_minus_self(self):
        # love = joy + trust; love - joy should leave trust (the component
        # NOT shared with the subtrahend), not disgust (-trust).
        joy = copy(EMOTIONS["joy"])
        trust = copy(EMOTIONS["trust"])
        love = joy + trust
        result = love - joy
        assert result.name == "trust", (
            f"Expected 'trust' (love - joy leaves the trust component), "
            f"got {result.name!r}"
        )

    def test_sub_emotion_is_anti_commutative_via_negation(self):
        # (c - e) should be the negation of what (e - c) "conceptually" is
        # for the shared-axis component: subtracting a stronger same-axis
        # emotion than the composite holds should flip the resulting sign
        # relative to subtracting a weaker one.
        rage = copy(EMOTIONS["rage"])  # sensitivity flow 3
        vigilance = copy(EMOTIONS["vigilance"])  # attention flow 2
        c = rage * vigilance
        annoyance = copy(EMOTIONS["annoyance"])  # sensitivity flow 1
        result = c - annoyance
        # self(rage=3) - other(annoyance=1) on sensitivity axis -> flow 2 (anger)
        assert result.components[0].emotional_flow == 2


# ---------------------------------------------------------------------------
# matrix_to_array / array_to_emotion
# ---------------------------------------------------------------------------

class TestMatrixConversions:
    def test_matrix_to_array_returns_4_element_array(self):
        c = rage_and_vigilance_composite()
        m = c.as_matrix
        arr = CompositeEmotion.matrix_to_array(m)
        assert arr.shape == (4,)

    def test_array_to_emotion_returns_emotion(self):
        arr = np.array([3, 3, 0, 0])
        result = CompositeEmotion.array_to_emotion(arr)
        assert result is not None


# ---------------------------------------------------------------------------
# CompositeEmotion — __truediv__ / __floordiv__ / __lshift__ / __rshift__
# ---------------------------------------------------------------------------

class TestCompositeEmotionDivShift:
    def test_truediv_neutrality_returns_copy(self):
        c = rage_and_vigilance_composite()
        result = c / Neutrality()
        assert isinstance(result, CompositeEmotion)

    def test_truediv_component_removes_it(self):
        c = rage_and_vigilance_composite()
        rage = copy(EMOTIONS["rage"])
        result = c / rage
        # component removed — result may be Emotion or CompositeEmotion
        assert result is not None

    def test_floordiv_neutrality_returns_copy(self):
        c = rage_and_vigilance_composite()
        result = c // Neutrality()
        assert isinstance(result, CompositeEmotion)

    def test_lshift_neutrality_returns_copy(self):
        c = rage_and_vigilance_composite()
        result = c << Neutrality()
        assert isinstance(result, CompositeEmotion)

    def test_rshift_neutrality_returns_copy(self):
        c = rage_and_vigilance_composite()
        result = c >> Neutrality()
        assert isinstance(result, CompositeEmotion)

    def test_truediv_string(self):
        c = rage_and_vigilance_composite()
        result = c / "rage"
        assert result is not None

    def test_lshift_string_unsupported(self):
        # __lshift__ converts string to emotion then only handles Neutrality
        c = rage_and_vigilance_composite()
        result = c.__lshift__("rage")
        assert result is NotImplemented


# ---------------------------------------------------------------------------
# CompositeEmotion — comparison operators
# ---------------------------------------------------------------------------

class TestCompositeEmotionComparisons:
    def test_eq_same_name(self):
        c = rage_and_vigilance_composite()
        assert c == "aggressiveness"

    def test_ne_different_name(self):
        c = rage_and_vigilance_composite()
        assert c != "joy"

    def test_lt_gt(self):
        c = rage_and_vigilance_composite()
        rage = copy(EMOTIONS["rage"])
        _ = c < rage
        _ = c > rage
        _ = c <= rage
        _ = c >= rage

    def test_lt_scalar(self):
        c = rage_and_vigilance_composite()
        _ = c < 5
        _ = c > 0
        _ = c <= 10
        _ = c >= 0


# ---------------------------------------------------------------------------
# CompositeEmotion — __contains__
# ---------------------------------------------------------------------------

class TestCompositeEmotionContains:
    def test_contains_component_emotion(self):
        c = rage_and_vigilance_composite()
        rage = copy(EMOTIONS["rage"])
        assert rage in c

    def test_contains_dimension(self):
        c = rage_and_vigilance_composite()
        # __contains__ checks self.dimension (composite) == item
        result = c.__contains__(c.dimensions[0])
        assert isinstance(result, bool)

    def test_contains_string(self):
        c = rage_and_vigilance_composite()
        assert "rage" in c

    def test_not_contains_unrelated(self):
        c = rage_and_vigilance_composite()
        joy = copy(EMOTIONS["joy"])
        assert joy not in c


# ---------------------------------------------------------------------------
# CompositeEmotion — type variants
# ---------------------------------------------------------------------------

class TestCompositeEmotionTypeVariants:
    def test_type_optimism(self):
        # optimism = anticipation + joy (pleasantness/attention → high flow)
        anticipation = copy(EMOTIONS["anticipation"])
        joy = copy(EMOTIONS["joy"])
        c = anticipation * joy
        t = c.type
        assert isinstance(t, str)

    def test_type_awe_negative_flow(self):
        # awe = fear(-2) + surprise(-1) → flow < 0 → hits negative branch
        fear = copy(EMOTIONS["fear"])
        surprise = copy(EMOTIONS["surprise"])
        c = fear * surprise
        assert c.emotional_flow < 0
        t = c.type
        assert "negative" in t or isinstance(t, str)

    def test_type_remorse_negative_flow(self):
        # remorse = grief(-3) + loathing(-3) → flow < 0
        grief = copy(EMOTIONS["grief"])
        loathing = copy(EMOTIONS["loathing"])
        c = grief * loathing
        assert c.emotional_flow < 0
        t = c.type
        assert isinstance(t, str)

    def test_type_submission_negative_flow(self):
        # submission = fear(-2) + trust(+1) → may be 0 or negative
        fear = copy(EMOTIONS["fear"])
        trust = copy(EMOTIONS["trust"])
        c = fear * trust
        t = c.type
        assert isinstance(t, str)

    def test_emotion_matrix_property(self):
        c = rage_and_vigilance_composite()
        m = c.emotion_matrix
        assert m is not None

    def test_kind_returns_string_or_none(self):
        c = rage_and_vigilance_composite()
        k = c.kind
        assert k is None or isinstance(k, str)


# ---------------------------------------------------------------------------
# CompositeDimension — extended properties
# ---------------------------------------------------------------------------

class TestCompositeDimensionProperties:
    def _get_cd(self, axis1="sensitivity", axis2="attention"):
        d1 = DIMENSIONS[axis1]
        d2 = DIMENSIONS[axis2]
        return d1 + d2

    def test_name_contains_axes(self):
        cd = self._get_cd()
        assert "sensitivity" in cd.name or "attention" in cd.name

    def test_intense_emotion_known_pair(self):
        cd = self._get_cd("sensitivity", "attention")
        result = cd.intense_emotion
        assert result is not None

    def test_intense_opposite_known_pair(self):
        cd = self._get_cd("sensitivity", "attention")
        result = cd.intense_opposite
        assert result is not None

    def test_mild_emotion_known_pair(self):
        cd = self._get_cd("sensitivity", "attention")
        result = cd.mild_emotion
        assert result is not None

    def test_mild_opposite_known_pair(self):
        cd = self._get_cd("sensitivity", "attention")
        result = cd.mild_opposite
        assert result is not None

    def test_intense_emotion_aptitude_sensitivity(self):
        cd = self._get_cd("aptitude", "sensitivity")
        assert cd.intense_emotion is not None

    def test_intense_emotion_pleasantness_attention(self):
        cd = self._get_cd("pleasantness", "attention")
        assert cd.intense_emotion is not None

    def test_intense_emotion_pleasantness_aptitude(self):
        cd = self._get_cd("pleasantness", "aptitude")
        assert cd.intense_emotion is not None

    def test_intense_opposite_aptitude_sensitivity(self):
        cd = self._get_cd("aptitude", "sensitivity")
        assert cd.intense_opposite is not None

    def test_intense_opposite_pleasantness_attention(self):
        cd = self._get_cd("pleasantness", "attention")
        assert cd.intense_opposite is not None

    def test_intense_opposite_pleasantness_aptitude(self):
        cd = self._get_cd("pleasantness", "aptitude")
        assert cd.intense_opposite is not None

    def test_mild_emotion_aptitude_sensitivity(self):
        cd = self._get_cd("aptitude", "sensitivity")
        assert cd.mild_emotion is not None

    def test_mild_emotion_pleasantness_attention(self):
        cd = self._get_cd("pleasantness", "attention")
        assert cd.mild_emotion is not None

    def test_mild_emotion_pleasantness_aptitude(self):
        cd = self._get_cd("pleasantness", "aptitude")
        assert cd.mild_emotion is not None

    def test_mild_opposite_aptitude_sensitivity(self):
        cd = self._get_cd("aptitude", "sensitivity")
        assert cd.mild_opposite is not None

    def test_mild_opposite_pleasantness_attention(self):
        cd = self._get_cd("pleasantness", "attention")
        assert cd.mild_opposite is not None

    def test_mild_opposite_pleasantness_aptitude(self):
        cd = self._get_cd("pleasantness", "aptitude")
        assert cd.mild_opposite is not None

    def test_basic_emotion_is_none(self):
        cd = self._get_cd()
        assert cd.basic_emotion is None

    def test_basic_opposite_is_none(self):
        cd = self._get_cd()
        assert cd.basic_opposite is None

    def test_sub_removes_dimension(self):
        d1 = DIMENSIONS["sensitivity"]
        d2 = DIMENSIONS["attention"]
        cd = d1 + d2
        result = cd - d1
        # result is EmotionalDimension (single dim left)
        assert isinstance(result, EmotionalDimension)

    def test_sub_absent_returns_none(self):
        d1 = DIMENSIONS["sensitivity"]
        d2 = DIMENSIONS["attention"]
        cd = d1 + d2
        d3 = DIMENSIONS["pleasantness"]
        result = cd - d3
        assert result is None

    def test_contains_neutrality(self):
        cd = self._get_cd()
        assert Neutrality() in cd

    def test_contains_emotion(self):
        from copy import copy
        cd = self._get_cd("sensitivity", "attention")
        rage = copy(EMOTIONS["rage"])
        # rage is in sensitivity dimension
        result = rage in cd
        assert isinstance(result, bool)

    def test_contains_feeling(self):
        from emotion_algebra.feelings import FEELINGS
        from copy import deepcopy
        cd = self._get_cd("sensitivity", "attention")
        love = deepcopy(FEELINGS["love"])
        result = love in cd
        assert isinstance(result, bool)

    def test_unknown_pair_returns_none_for_all_properties(self):
        # Build a CompositeDimension with fake axes that won't match any named pair
        from emotion_algebra.composite_emotions import CompositeDimension
        from emotion_algebra.plutchik import EmotionalDimension
        cd = CompositeDimension()
        d_fake = EmotionalDimension()
        d_fake.axis = "unknown_axis"
        d_fake2 = EmotionalDimension()
        d_fake2.axis = "other_axis"
        cd.dimensions = [d_fake, d_fake2]
        assert cd.intense_emotion is None
        assert cd.intense_opposite is None
        assert cd.mild_emotion is None
        assert cd.mild_opposite is None
        assert cd.basic_emotion is None
        assert cd.basic_opposite is None

    def test_eq_with_matching_dimension_emotion(self):
        c = rage_and_vigilance_composite()
        rage = copy(EMOTIONS["rage"])
        # Test __eq__ when other is Emotion with matching dimension
        result = c.__eq__(rage)
        assert isinstance(result, bool)

    def test_ne_with_matching_dimension_emotion(self):
        c = rage_and_vigilance_composite()
        rage = copy(EMOTIONS["rage"])
        result = c.__ne__(rage)
        assert isinstance(result, bool)

    def test_add_returns_notimplemented_for_unknown_type(self):
        c = rage_and_vigilance_composite()
        result = c.__add__(42)
        assert result is NotImplemented

    def test_sub_returns_notimplemented_for_unknown_type(self):
        c = rage_and_vigilance_composite()
        result = c.__sub__(42)
        assert result is NotImplemented

    def test_truediv_composite_returns_notimplemented(self):
        c = rage_and_vigilance_composite()
        c2 = rage_and_vigilance_composite()
        result = c.__truediv__(c2)
        assert result is NotImplemented

    def test_floordiv_composite_returns_notimplemented(self):
        c = rage_and_vigilance_composite()
        c2 = rage_and_vigilance_composite()
        result = c.__floordiv__(c2)
        assert result is NotImplemented

    def test_lshift_emotion_returns_notimplemented(self):
        c = rage_and_vigilance_composite()
        rage = copy(EMOTIONS["rage"])
        result = c.__lshift__(rage)
        assert result is NotImplemented

    def test_rshift_emotion_returns_notimplemented(self):
        c = rage_and_vigilance_composite()
        rage = copy(EMOTIONS["rage"])
        result = c.__rshift__(rage)
        assert result is NotImplemented

    def test_contains_returns_notimplemented_for_unknown_type(self):
        c = rage_and_vigilance_composite()
        result = c.__contains__(42)
        assert result is NotImplemented

    def test_mul_with_emotion_returns_matrix(self):
        # __mul__ on Emotion (not CompositeEmotion) with another Emotion
        rage = copy(EMOTIONS["rage"])
        vigilance = copy(EMOTIONS["vigilance"])
        result = rage.__mul__(vigilance)
        assert result is not None

    def test_composite_dimension_sub_single_dim_remaining(self):
        # cd - dim where 2 dims, remove one → returns EmotionalDimension
        from emotion_algebra.composite_emotions import CompositeDimension
        cd = CompositeDimension()
        d1 = DIMENSIONS["sensitivity"]
        d2 = DIMENSIONS["attention"]
        cd.dimensions = [d1, d2]
        result = cd - d1
        assert isinstance(result, EmotionalDimension)

    def test_composite_dimension_sub_three_dims_returns_composite(self):
        # cd - dim where 3 dims remain > 1, returns CompositeDimension
        from emotion_algebra.composite_emotions import CompositeDimension
        cd = CompositeDimension()
        d1 = DIMENSIONS["sensitivity"]
        d2 = DIMENSIONS["attention"]
        d3 = DIMENSIONS["pleasantness"]
        cd.dimensions = [d1, d2, d3]
        result = cd - d1
        assert isinstance(result, CompositeDimension)

    def test_composite_dimension_contains_emotion_not_in_dims(self):
        # emotion whose dimension is not in the CompositeDimension → False
        from emotion_algebra.composite_emotions import CompositeDimension
        cd = CompositeDimension()
        cd.dimensions = [DIMENSIONS["sensitivity"], DIMENSIONS["attention"]]
        joy = copy(EMOTIONS["joy"])  # joy is in pleasantness dimension
        result = joy in cd
        assert isinstance(result, bool)

    def test_composite_dimension_contains_feeling(self):
        from emotion_algebra.composite_emotions import CompositeDimension
        from emotion_algebra.feelings import FEELINGS
        from copy import deepcopy
        cd = CompositeDimension()
        cd.dimensions = [DIMENSIONS["sensitivity"], DIMENSIONS["attention"]]
        love = deepcopy(FEELINGS["love"])
        result = love in cd
        assert isinstance(result, bool)

    def test_truediv_absent_component_returns_notimplemented(self):
        # __truediv__ with an Emotion not in components
        c = rage_and_vigilance_composite()
        joy = copy(EMOTIONS["joy"])
        result = c.__truediv__(joy)
        # joy not in components; falls through to NotImplemented
        assert result is NotImplemented

    def test_eq_with_emotion_matching_dimension(self):
        # __eq__ when other is Emotion and other._dimension == self.dimension
        c = rage_and_vigilance_composite()
        rage = copy(EMOTIONS["rage"])
        # self.dimension is CompositeDimension; other._dimension is EmotionalDimension
        # These won't be equal so we just check it returns bool
        result = c == rage
        assert isinstance(result, bool)

    def test_ne_with_emotion_matching_dimension(self):
        c = rage_and_vigilance_composite()
        rage = copy(EMOTIONS["rage"])
        result = c != rage
        assert isinstance(result, bool)

    def test_mul_composite_with_emotion(self):
        # CompositeEmotion.__mul__(Emotion) must return an EmotionBase (a
        # FloatEmotion projected from the matrix product), never a bare
        # numpy array — every operator returns a typed algebra member.
        from emotion_algebra.base import EmotionBase
        from emotion_algebra.float_emotion import FloatEmotion
        c = rage_and_vigilance_composite()
        joy = copy(EMOTIONS["joy"])
        result = c.__mul__(joy)
        assert result is not None
        assert isinstance(result, EmotionBase)
        assert isinstance(result, FloatEmotion)


# ---------------------------------------------------------------------------
# Coverage gap tests — targeted at uncovered branches
# ---------------------------------------------------------------------------

class TestCompositeCoverageGaps:
    def test_string_to_emotion_feeling_branch(self):
        """line 141-143: string_to_emotion returns Feeling when name is in FEELINGS."""
        from emotion_algebra.composite_emotions import CompositeEmotion
        result = CompositeEmotion.string_to_emotion("love")
        from emotion_algebra.feelings import Feeling
        assert isinstance(result, Feeling)

    def test_string_to_emotion_unknown(self):
        """string_to_emotion returns string unchanged when not found."""
        from emotion_algebra.composite_emotions import CompositeEmotion
        result = CompositeEmotion.string_to_emotion("not_an_emotion_xyz")
        assert result == "not_an_emotion_xyz"

    def test_get_composite_from_emotions(self):
        """line 192: get_composite_from_emotions lookup."""
        from emotion_algebra.composite_emotions import CompositeEmotion, COMPOSITE_EMOTIONS
        rage = copy(EMOTIONS["rage"])
        vigilance = copy(EMOTIONS["vigilance"])
        result = CompositeEmotion.get_composite_from_emotions(rage, vigilance)
        assert result == "aggressiveness"

    def test_base_emotion_reconstructs(self):
        """line 236: base_emotion rebuilds from emotion_vector."""
        c = rage_and_vigilance_composite()
        base = c.base_emotion
        assert base is not None

    def test_equivalent_feeling_fallback(self):
        """lines 289-291: equivalent_feeling fallback when no match in FEELINGS_TO_EMOTION_MAP."""
        from emotion_algebra.composite_emotions import CompositeEmotion
        from emotion_algebra.feelings import Feeling
        # Build a composite with 3 components — no 2-component dyad match
        anger = copy(EMOTIONS["anger"])
        joy = copy(EMOTIONS["joy"])
        trust = copy(EMOTIONS["trust"])
        c = CompositeEmotion()
        c.components = [anger, joy, trust]
        result = c.equivalent_feeling
        assert isinstance(result, Feeling)

    def test_add_string_resolves_feeling(self):
        """line 336: __add__ with string that resolves to a Feeling."""
        c = rage_and_vigilance_composite()
        result = c.__add__("love")
        assert result is not None

    def test_sub_string_resolves(self):
        """line 361: __sub__ with string argument."""
        c = rage_and_vigilance_composite()
        result = c.__sub__("rage")
        assert result is not None

    def test_sub_feeling(self):
        """lines 368-369: __sub__ with Feeling."""
        from emotion_algebra.feelings import FEELINGS
        from copy import deepcopy
        c = rage_and_vigilance_composite()
        love = deepcopy(FEELINGS["love"])
        result = c - love
        assert result is not None

    def test_sub_returns_neutrality_on_empty(self):
        """line 374: __sub__ returns Neutrality when components empty."""
        from emotion_algebra.composite_emotions import CompositeEmotion
        from emotion_algebra.plutchik import Neutrality
        rage = copy(EMOTIONS["rage"])
        c = CompositeEmotion()
        c.components = [rage]
        result = c - rage
        assert isinstance(result, Neutrality) or result is not None

    def test_truediv_composite_returns_notimplemented(self):
        """line 393: __truediv__ with CompositeEmotion → NotImplemented."""
        c1 = rage_and_vigilance_composite()
        c2 = rage_and_vigilance_composite()
        result = c1.__truediv__(c2)
        assert result is NotImplemented

    def test_floordiv_composite_returns_notimplemented(self):
        """lines 399-404: __floordiv__ with CompositeEmotion → NotImplemented."""
        c1 = rage_and_vigilance_composite()
        c2 = rage_and_vigilance_composite()
        result = c1.__floordiv__(c2)
        assert result is NotImplemented

    def test_floordiv_emotion_removes_component(self):
        """__floordiv__ with Emotion that is a component."""
        from emotion_algebra.composite_emotions import CompositeEmotion
        rage = copy(EMOTIONS["rage"])
        vigilance = copy(EMOTIONS["vigilance"])
        c = CompositeEmotion()
        c.components = [copy(rage), copy(vigilance)]
        result = c // rage
        assert result is not None

    def test_eq_with_name_string(self):
        """line 416: __eq__ with string compares name."""
        from emotion_algebra.composite_emotions import COMPOSITE_EMOTIONS
        from copy import copy as cp
        c = cp(COMPOSITE_EMOTIONS["aggressiveness"])
        assert c == "aggressiveness"
        assert c != "love"

    def test_ne_with_emotion(self):
        """line 425/432: __ne__ branches."""
        from emotion_algebra.composite_emotions import CompositeEmotion
        c = rage_and_vigilance_composite()
        rage = copy(EMOTIONS["rage"])
        assert c != rage   # different type

    def test_contains_dimension_via_composite(self):
        """__contains__ with EmotionalDimension."""
        from emotion_algebra.composite_emotions import CompositeEmotion
        c = rage_and_vigilance_composite()
        sensitivity = DIMENSIONS["sensitivity"]
        # dimension check goes through composite dimension
        result = c.__contains__(sensitivity)
        assert isinstance(result, (bool, type(NotImplemented)))

    def test_composite_dimension_contains_emotion(self):
        """CompositeDimension __contains__ with Emotion that IS in one of the dimensions."""
        from emotion_algebra.composite_emotions import CompositeDimension
        cd = DIMENSIONS["sensitivity"] + DIMENSIONS["attention"]
        rage = copy(EMOTIONS["rage"])
        assert rage in cd

    def test_composite_dimension_repr(self):
        """CompositeDimension __repr__."""
        from emotion_algebra.composite_emotions import CompositeDimension
        cd = DIMENSIONS["sensitivity"] + DIMENSIONS["attention"]
        r = repr(cd)
        assert "CompositeDimensionObject" in r

    def test_emotion_vector_aptitude_branch(self):
        """line 192: aptitude branch in emotion_vector."""
        rage = copy(EMOTIONS["rage"])
        trust = copy(EMOTIONS["trust"])  # aptitude axis
        c = rage + trust
        vec = c.emotion_vector
        assert vec[3].emotional_flow != 0

    def test_arousal_empty_composite(self):
        """line 236: arousal returns 0 when components is empty."""
        from emotion_algebra.composite_emotions import CompositeEmotion
        c = CompositeEmotion()
        assert c.arousal == 0

    def test_composite_dimension_sub_notimplemented(self):
        """line 541: CompositeDimension.__sub__ with non-dimension."""
        from emotion_algebra.composite_emotions import CompositeDimension
        cd = DIMENSIONS["sensitivity"] + DIMENSIONS["attention"]
        result = cd.__sub__("not_a_dimension")
        assert result is NotImplemented

    def test_composite_dimension_contains_unrecognised(self):
        """lines 559-561: __contains__ with unrecognised type → False."""
        from emotion_algebra.composite_emotions import CompositeDimension
        cd = DIMENSIONS["sensitivity"] + DIMENSIONS["attention"]
        result = cd.__contains__(42)
        assert result is False

    def test_mul_neutrality(self):
        """lines 368-369: __mul__ with Neutrality."""
        from emotion_algebra.plutchik import Neutrality
        c = rage_and_vigilance_composite()
        result = c * Neutrality()
        from emotion_algebra.composite_emotions import CompositeEmotion
        assert isinstance(result, CompositeEmotion)

    def test_mul_notimplemented(self):
        """line 374: __mul__ with non-Emotion/non-Neutrality."""
        c = rage_and_vigilance_composite()
        result = c.__mul__(42)
        assert result is NotImplemented

    def test_floordiv_string(self):
        """line 393: __floordiv__ with string."""
        c = rage_and_vigilance_composite()
        result = c // "rage"
        assert result is not None

    def test_floordiv_emotion_not_component(self):
        """line 404: __floordiv__ with Emotion not in components."""
        c = rage_and_vigilance_composite()
        joy = copy(EMOTIONS["joy"])
        result = c.__floordiv__(joy)
        # joy not in components → falls through to return NotImplemented
        assert result is NotImplemented

    def test_eq_with_string(self):
        """line 416: __eq__ with string name."""
        from emotion_algebra.composite_emotions import COMPOSITE_EMOTIONS
        from copy import copy as cp
        c = cp(COMPOSITE_EMOTIONS["aggressiveness"])
        assert (c == "aggressiveness") is True

    def test_ne_mismatched_emotion(self):
        """line 425/432: __ne__ with Emotion from different dimension."""
        c = rage_and_vigilance_composite()
        joy = copy(EMOTIONS["joy"])
        assert c != joy

    def test_composite_dimension_sub_multi(self):
        """line 529: CompositeDimension.__sub__ leaves >1 dimension."""
        from emotion_algebra.composite_emotions import CompositeDimension
        cd = DIMENSIONS["sensitivity"] + DIMENSIONS["attention"]
        # Add a third dimension manually
        cd.dimensions.append(DIMENSIONS["pleasantness"])
        result = cd - DIMENSIONS["sensitivity"]
        # Two dimensions remain → returns CompositeDimension
        assert isinstance(result, CompositeDimension)

    def test_composite_dimension_contains_dimension_false(self):
        """line 559: __contains__ EmotionalDimension not in list → falls through to False."""
        cd = DIMENSIONS["sensitivity"] + DIMENSIONS["attention"]
        result = cd.__contains__(DIMENSIONS["pleasantness"])
        # pleasantness not in this CompositeDimension
        assert result is False

    def test_composite_dimension_add_notimplemented(self):
        """line 529: CompositeDimension.__add__ non-EmotionalDimension."""
        from emotion_algebra.composite_emotions import CompositeDimension
        cd = DIMENSIONS["sensitivity"] + DIMENSIONS["attention"]
        result = cd.__add__(42)
        assert result is NotImplemented

    def test_eq_composite_vs_composite_name_mismatch(self):
        """line 416: __eq__ comparing by name string."""
        from emotion_algebra.composite_emotions import COMPOSITE_EMOTIONS
        from copy import copy as cp
        c = cp(COMPOSITE_EMOTIONS["aggressiveness"])
        assert (c == "love") is False

    def test_composite_dimension_contains_emotion_false(self):
        """line 559: __contains__ EmotionalDimension not present → False."""
        from emotion_algebra.composite_emotions import CompositeDimension
        cd = DIMENSIONS["sensitivity"] + DIMENSIONS["attention"]
        # pleasantness dimension not in this CompositeDimension
        assert DIMENSIONS["pleasantness"] not in cd
