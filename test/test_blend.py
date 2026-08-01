"""Tests for FloatEmotion.blend() — multi-emotion blending."""
import pytest
import numpy as np

from emotion_algebra.float_emotion import FloatEmotion
from emotion_algebra.emotions import get_emotion


class TestFloatEmotionBlend:
    def test_blend_two_emotions(self):
        joy = get_emotion("joy")
        trust = get_emotion("trust")
        fe = FloatEmotion.blend(joy, trust)
        assert isinstance(fe, FloatEmotion)
        # Joy contributes pleasantness, trust contributes aptitude
        assert float(fe.as_array[2]) > 0  # pleasantness
        assert float(fe.as_array[3]) > 0  # aptitude

    def test_blend_single_emotion(self):
        anger = get_emotion("anger")
        fe = FloatEmotion.blend(anger)
        expected = FloatEmotion.from_emotion(anger)
        assert np.allclose(fe.as_array, expected.as_array)

    def test_blend_equal_weights(self):
        joy = get_emotion("joy")
        sadness = get_emotion("sadness")
        fe = FloatEmotion.blend(joy, sadness)
        # Joy (+2 pleasantness) + sadness (-2 pleasantness) → 0
        assert abs(float(fe.as_array[2])) < 0.01

    def test_blend_custom_weights(self):
        joy = get_emotion("joy")
        trust = get_emotion("trust")
        fe = FloatEmotion.blend(joy, trust, weights=[0.8, 0.2])
        # Joy dominates → pleasantness > aptitude
        assert float(fe.as_array[2]) > float(fe.as_array[3])

    def test_blend_with_scale(self):
        joy = get_emotion("joy")
        fe_full = FloatEmotion.blend(joy, scale=1.0)
        fe_half = FloatEmotion.blend(joy, scale=0.5)
        assert np.allclose(fe_half.as_array, fe_full.as_array * 0.5)

    def test_blend_three_emotions(self):
        joy = get_emotion("joy")
        trust = get_emotion("trust")
        interest = get_emotion("interest")
        fe = FloatEmotion.blend(joy, trust, interest)
        # All three contribute to different axes
        assert float(fe.as_array[1]) > 0  # attention from interest
        assert float(fe.as_array[2]) > 0  # pleasantness from joy
        assert float(fe.as_array[3]) > 0  # aptitude from trust

    def test_blend_wrong_weight_count_raises(self):
        joy = get_emotion("joy")
        trust = get_emotion("trust")
        with pytest.raises(ValueError):
            FloatEmotion.blend(joy, trust, weights=[0.5])

    def test_blend_empty_returns_zero(self):
        fe = FloatEmotion.blend()
        assert np.allclose(fe.as_array, 0.0)
