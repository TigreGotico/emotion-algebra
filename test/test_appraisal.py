"""Tests for emotion_algebra.appraisal — Appraisal, appraisal_to_emotion."""
import pytest

from emotion_algebra.appraisal import Appraisal, appraisal_to_emotion
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
