"""Tests for emotion_algebra.needs — need-to-emotion mapping."""
import pytest

from emotion_algebra.needs import (
    need_deficit_to_emotion,
    need_deficit_to_float_emotion,
    NEED_DEFICIT_EMOTIONS,
    MAXNEEF_DEFICIT_EMOTIONS,
    MURRAY_DEFICIT_EMOTIONS,
)
from emotion_algebra.float_emotion import FloatEmotion
from emotion_algebra.base import EmotionBase


class TestNeedDeficitToEmotion:
    def test_all_maxneef_needs_have_emotions(self):
        for need in MAXNEEF_DEFICIT_EMOTIONS:
            emo = need_deficit_to_emotion(need)
            assert emo is not None, f"No emotion for Max-Neef need {need!r}"
            assert isinstance(emo, EmotionBase)

    def test_all_murray_needs_have_emotions(self):
        for need in MURRAY_DEFICIT_EMOTIONS:
            emo = need_deficit_to_emotion(need)
            assert emo is not None, f"No emotion for Murray need {need!r}"

    def test_protection_deficit_is_fear(self):
        assert need_deficit_to_emotion("protection").name == "fear"

    def test_subsistence_deficit_is_sadness(self):
        assert need_deficit_to_emotion("subsistence").name == "sadness"

    def test_freedom_deficit_is_anger(self):
        assert need_deficit_to_emotion("freedom").name == "anger"

    def test_creation_deficit_is_boredom(self):
        assert need_deficit_to_emotion("creation").name == "boredom"

    def test_harm_avoidance_deficit_is_fear(self):
        assert need_deficit_to_emotion("harm_avoidance").name == "fear"

    def test_unknown_need_returns_none(self):
        assert need_deficit_to_emotion("nonexistent") is None


class TestNeedDeficitToFloatEmotion:
    def test_returns_float_emotion(self):
        fe = need_deficit_to_float_emotion("protection")
        assert isinstance(fe, FloatEmotion)

    def test_protection_has_negative_sensitivity(self):
        """Fear → negative sensitivity axis."""
        fe = need_deficit_to_float_emotion("protection")
        assert float(fe.as_array[0]) < 0

    def test_subsistence_has_negative_pleasantness(self):
        """Sadness → negative pleasantness axis."""
        fe = need_deficit_to_float_emotion("subsistence")
        assert float(fe.as_array[2]) < 0

    def test_freedom_has_positive_sensitivity(self):
        """Anger → positive sensitivity axis."""
        fe = need_deficit_to_float_emotion("freedom")
        assert float(fe.as_array[0]) > 0

    def test_unknown_need_returns_none(self):
        assert need_deficit_to_float_emotion("nonexistent") is None

    def test_all_needs_produce_nonzero_vector(self):
        import numpy as np
        for need in NEED_DEFICIT_EMOTIONS:
            fe = need_deficit_to_float_emotion(need)
            assert fe is not None
            assert np.any(fe.as_array != 0), f"Need {need!r} produced zero vector"
