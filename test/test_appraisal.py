"""Tests for emotion_algebra.appraisal — Appraisal, appraisal_to_emotion, continuous appraisal."""
import pytest

from emotion_algebra.appraisal import (
    Appraisal, appraisal_to_emotion, appraisal_to_float_emotion,
    float_emotion_to_neuro_deltas,
)
from emotion_algebra.float_emotion import FloatEmotion
from emotion_algebra.plutchik import Neutrality


class TestAppraisal:
    def test_default_all_none(self):
        a = Appraisal()
        assert a.novelty is None
        assert a.goal_relevance is None
        assert a.goal_congruence is None
        assert a.agency is None
        assert a.coping_potential is None

    def test_set_fields(self):
        a = Appraisal(novelty="unexpected", goal_relevance="relevant")
        assert a.novelty == "unexpected"
        assert a.goal_relevance == "relevant"

    def test_to_dict_has_all_keys(self):
        a = Appraisal(novelty="unexpected")
        d = a.to_dict()
        assert "novelty" in d
        assert "goal_relevance" in d
        assert "goal_congruence" in d
        assert "agency" in d
        assert "coping_potential" in d

    def test_from_dict_roundtrip(self):
        a = Appraisal(
            novelty="unexpected",
            goal_relevance="relevant",
            goal_congruence="incongruent",
            agency="other",
            coping_potential="high",
        )
        a2 = Appraisal.from_dict(a.to_dict())
        assert a2.novelty == a.novelty
        assert a2.goal_relevance == a.goal_relevance
        assert a2.goal_congruence == a.goal_congruence
        assert a2.agency == a.agency
        assert a2.coping_potential == a.coping_potential

    def test_from_dict_partial(self):
        d = {"novelty": "unexpected", "goal_relevance": None}
        a = Appraisal.from_dict(d)
        assert a.novelty == "unexpected"
        assert a.goal_relevance is None


class TestAppraisalToEmotion:
    def test_anger_pattern(self):
        a = Appraisal(
            goal_relevance="relevant",
            goal_congruence="incongruent",
            agency="other",
            coping_potential="high",
        )
        result = appraisal_to_emotion(a)
        assert result.name == "anger"

    def test_fear_pattern(self):
        a = Appraisal(
            goal_relevance="relevant",
            goal_congruence="incongruent",
            agency="other",
            coping_potential="low",
        )
        result = appraisal_to_emotion(a)
        assert result.name == "fear"

    def test_joy_congruent_self(self):
        a = Appraisal(
            goal_relevance="relevant",
            goal_congruence="congruent",
            agency="self",
        )
        result = appraisal_to_emotion(a)
        assert result.name == "joy"

    def test_trust_congruent_other(self):
        a = Appraisal(
            goal_relevance="relevant",
            goal_congruence="congruent",
            agency="other",
        )
        result = appraisal_to_emotion(a)
        assert result.name == "trust"

    def test_serenity_congruent_circumstance(self):
        a = Appraisal(
            goal_relevance="relevant",
            goal_congruence="congruent",
            agency="circumstance",
        )
        result = appraisal_to_emotion(a)
        assert result.name == "serenity"

    def test_boredom_irrelevant(self):
        a = Appraisal(goal_relevance="irrelevant")
        result = appraisal_to_emotion(a)
        assert result.name == "boredom"

    def test_surprise_unexpected_irrelevant(self):
        a = Appraisal(novelty="unexpected", goal_relevance="irrelevant")
        result = appraisal_to_emotion(a)
        assert result.name == "surprise"

    def test_surprise_unexpected_alone(self):
        a = Appraisal(novelty="unexpected")
        result = appraisal_to_emotion(a)
        assert result.name == "surprise"

    def test_disgust_incongruent_self_high(self):
        a = Appraisal(
            goal_relevance="relevant",
            goal_congruence="incongruent",
            agency="self",
            coping_potential="high",
        )
        result = appraisal_to_emotion(a)
        assert result.name == "disgust"

    def test_sadness_incongruent_self_low(self):
        a = Appraisal(
            goal_relevance="relevant",
            goal_congruence="incongruent",
            agency="self",
            coping_potential="low",
        )
        result = appraisal_to_emotion(a)
        assert result.name == "sadness"

    def test_empty_appraisal_returns_neutrality(self):
        a = Appraisal()
        result = appraisal_to_emotion(a)
        assert isinstance(result, Neutrality)

    def test_returns_emotion_base(self):
        from emotion_algebra.base import EmotionBase
        a = Appraisal(goal_relevance="relevant", goal_congruence="congruent", agency="self")
        result = appraisal_to_emotion(a)
        assert isinstance(result, EmotionBase)

    def test_anticipation_incongruent_circumstance_high(self):
        a = Appraisal(
            goal_relevance="relevant",
            goal_congruence="incongruent",
            agency="circumstance",
            coping_potential="high",
        )
        result = appraisal_to_emotion(a)
        assert result.name == "anticipation"


class TestAppraisalToFloat:
    """Test Appraisal.to_float() normalisation."""

    def test_categorical_novelty(self):
        a = Appraisal(novelty="unexpected").to_float()
        assert a.novelty == 1.0
        a2 = Appraisal(novelty="expected").to_float()
        assert a2.novelty == 0.0

    def test_categorical_agency(self):
        for val, expected in [("self", 1.0), ("other", 0.5), ("circumstance", 0.0)]:
            assert Appraisal(agency=val).to_float().agency == expected

    def test_none_maps_to_half(self):
        a = Appraisal().to_float()
        assert a.novelty == 0.5
        assert a.goal_relevance == 0.5
        assert a.intrinsic_pleasantness == 0.5

    def test_float_passthrough(self):
        a = Appraisal(novelty=0.7, goal_relevance=0.3).to_float()
        assert a.novelty == pytest.approx(0.7)
        assert a.goal_relevance == pytest.approx(0.3)

    def test_float_clamped(self):
        a = Appraisal(novelty=1.5, goal_relevance=-0.2).to_float()
        assert a.novelty == 1.0
        assert a.goal_relevance == 0.0

    def test_intrinsic_pleasantness_categorical(self):
        assert Appraisal(intrinsic_pleasantness="pleasant").to_float().intrinsic_pleasantness == 1.0
        assert Appraisal(intrinsic_pleasantness="unpleasant").to_float().intrinsic_pleasantness == 0.0

    def test_to_dict_includes_pleasantness(self):
        a = Appraisal(intrinsic_pleasantness=0.7)
        d = a.to_dict()
        assert "intrinsic_pleasantness" in d
        assert d["intrinsic_pleasantness"] == 0.7

    def test_from_dict_with_pleasantness(self):
        a = Appraisal.from_dict({"intrinsic_pleasantness": "pleasant", "novelty": "unexpected"})
        assert a.intrinsic_pleasantness == "pleasant"
        assert a.novelty == "unexpected"

    def test_invalid_cross_field_value_raises(self):
        """e.g. agency='low' should be rejected, not silently mapped."""
        with pytest.raises(ValueError, match="agency"):
            Appraisal(agency="low").to_float()

    def test_unknown_string_value_raises(self):
        with pytest.raises(ValueError, match="novelty"):
            Appraisal(novelty="bogus").to_float()


def _axes(fe):
    """Extract named axes from a FloatEmotion as_array [sens, attn, pleas, apt]."""
    v = fe.as_array
    return float(v[0]), float(v[1]), float(v[2]), float(v[3])


class TestAppraisalToFloatEmotion:
    """Test appraisal_to_float_emotion() continuous mapping."""

    def test_returns_float_emotion(self):
        a = Appraisal(novelty=0.8, goal_relevance=0.9)
        result = appraisal_to_float_emotion(a)
        assert isinstance(result, FloatEmotion)

    def test_novel_relevant_congruent_positive_pleasantness(self):
        """Unexpected + relevant + congruent → high pleasantness."""
        a = Appraisal(novelty=1.0, goal_relevance=1.0, goal_congruence=1.0,
                       intrinsic_pleasantness=1.0)
        sens, attn, pleas, apt = _axes(appraisal_to_float_emotion(a))
        assert pleas > 0

    def test_incongruent_low_coping_negative_sensitivity(self):
        """Relevant + incongruent + can't cope → negative sensitivity (fear)."""
        a = Appraisal(goal_relevance=1.0, goal_congruence=0.0, coping_potential=0.0)
        sens, attn, pleas, apt = _axes(appraisal_to_float_emotion(a))
        assert sens < 0

    def test_congruent_high_coping_positive_aptitude(self):
        """Congruent + can cope → high aptitude (trust)."""
        a = Appraisal(goal_congruence=1.0, coping_potential=1.0)
        sens, attn, pleas, apt = _axes(appraisal_to_float_emotion(a))
        assert apt > 0

    def test_novel_negative_attention(self):
        """Unexpected → negative attention (surprise pole)."""
        a = Appraisal(novelty=1.0, goal_relevance=0.5)
        sens, attn, pleas, apt = _axes(appraisal_to_float_emotion(a))
        assert attn < 0

    def test_all_neutral_near_zero(self):
        """All None → 0.5 → all four axes near zero."""
        a = Appraisal()  # all None → 0.5
        sens, attn, pleas, apt = _axes(appraisal_to_float_emotion(a))
        assert abs(sens) < 0.1
        assert abs(attn) < 0.1
        assert abs(pleas) < 0.1
        assert abs(apt) < 0.1

    def test_categorical_inputs_work(self):
        """Categorical values are normalised internally."""
        a = Appraisal(novelty="unexpected", goal_relevance="relevant",
                       goal_congruence="congruent")
        sens, attn, pleas, apt = _axes(appraisal_to_float_emotion(a))
        assert pleas > 0

    def test_unpleasant_lowers_pleasantness(self):
        """Unpleasant stimulus lowers pleasantness axis."""
        pleasant = Appraisal(intrinsic_pleasantness=1.0, goal_congruence=0.5,
                              goal_relevance=0.5)
        unpleasant = Appraisal(intrinsic_pleasantness=0.0, goal_congruence=0.5,
                                goal_relevance=0.5)
        _, _, pleas_p, _ = _axes(appraisal_to_float_emotion(pleasant))
        _, _, pleas_u, _ = _axes(appraisal_to_float_emotion(unpleasant))
        assert pleas_p > pleas_u


class TestFloatEmotionToNeuroDeltas:
    """Test float_emotion_to_neuro_deltas() inverse mapping."""

    def test_returns_three_floats(self):
        fe = FloatEmotion(sensitivity=0.5, attention=0.5, pleasantness=0.3, aptitude=0.3)
        result = float_emotion_to_neuro_deltas(fe)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_high_arousal_positive_dopamine(self):
        fe = FloatEmotion(sensitivity=1.0, attention=1.0, pleasantness=0.0, aptitude=0.0)
        d, s, a = float_emotion_to_neuro_deltas(fe)
        assert d > 0

    def test_positive_valence_positive_serotonin(self):
        fe = FloatEmotion(sensitivity=0.0, attention=0.0, pleasantness=1.0, aptitude=1.0)
        d, s, a = float_emotion_to_neuro_deltas(fe)
        assert s > 0
        assert a == 0.0

    def test_negative_valence_positive_adrenaline(self):
        fe = FloatEmotion(sensitivity=0.0, attention=0.0, pleasantness=-1.0, aptitude=-1.0)
        d, s, a = float_emotion_to_neuro_deltas(fe)
        assert a > 0
        assert s == 0.0

    def test_negative_sensitivity_still_raises_dopamine(self):
        # Fear (sensitivity < 0) is high-arousal/high-salience just like
        # anger (sensitivity > 0) — the dopamine term represents arousal
        # magnitude, not the hedonic sign of sensitivity, so it must not
        # go negative or lower than the zero-arousal baseline.
        fear_like = FloatEmotion(sensitivity=-2.0, attention=0.0, pleasantness=0.0, aptitude=0.0)
        anger_like = FloatEmotion(sensitivity=2.0, attention=0.0, pleasantness=0.0, aptitude=0.0)
        neutral = FloatEmotion(sensitivity=0.0, attention=0.0, pleasantness=0.0, aptitude=0.0)
        d_fear, _, _ = float_emotion_to_neuro_deltas(fear_like)
        d_anger, _, _ = float_emotion_to_neuro_deltas(anger_like)
        d_neutral, _, _ = float_emotion_to_neuro_deltas(neutral)
        assert d_fear > d_neutral
        assert d_fear == pytest.approx(d_anger)

    def test_scale_parameter(self):
        fe = FloatEmotion(sensitivity=1.0, attention=1.0, pleasantness=0.0, aptitude=0.0)
        d1, _, _ = float_emotion_to_neuro_deltas(fe, scale=0.15)
        d2, _, _ = float_emotion_to_neuro_deltas(fe, scale=0.30)
        assert d2 == pytest.approx(d1 * 2.0)

    def test_round_trip_relevant_congruent(self):
        """relevant + congruent + high coping → positive serotonin."""
        a = Appraisal(novelty=0.5, goal_relevance=1.0, goal_congruence=1.0,
                       coping_potential=1.0, intrinsic_pleasantness=0.8)
        fe = appraisal_to_float_emotion(a)
        d, s, adr = float_emotion_to_neuro_deltas(fe)
        assert s > 0, "Congruent+pleasant should produce serotonin"

    def test_round_trip_threatening(self):
        """relevant + incongruent + low coping → adrenaline."""
        a = Appraisal(goal_relevance=1.0, goal_congruence=0.0,
                       coping_potential=0.0, intrinsic_pleasantness=0.0)
        fe = appraisal_to_float_emotion(a)
        d, s, adr = float_emotion_to_neuro_deltas(fe)
        assert adr > 0, "Threat should produce adrenaline"
