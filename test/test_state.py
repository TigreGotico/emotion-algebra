"""Tests for emotion_algebra.state — EmotionalState, EmotionTimeline."""
from copy import copy

import numpy as np
import pytest

from emotion_algebra.state import EmotionalState, EmotionTimeline
from emotion_algebra.emotions import get_emotion


@pytest.fixture
def anger():
    return copy(get_emotion("anger"))

@pytest.fixture
def fear():
    return copy(get_emotion("fear"))

@pytest.fixture
def joy():
    return copy(get_emotion("joy"))


class TestEmotionalState:
    def test_default_is_zero(self):
        s = EmotionalState()
        assert np.allclose(s.snapshot(), 0)

    def test_init_with_vector(self):
        s = EmotionalState([1.0, 2.0, 3.0, 4.0])
        assert s.snapshot().tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_apply_changes_vector(self, anger):
        s = EmotionalState()
        s.apply(anger)
        assert not np.allclose(s.snapshot(), 0)

    def test_apply_weight(self, anger):
        s1 = EmotionalState()
        s2 = EmotionalState()
        s1.apply(anger, weight=1.0)
        s2.apply(anger, weight=2.0)
        assert np.allclose(s2.snapshot(), s1.snapshot() * 2)

    def test_apply_returns_self(self, anger):
        s = EmotionalState()
        result = s.apply(anger)
        assert result is s

    def test_decay_reduces_magnitude(self, anger):
        s = EmotionalState()
        s.apply(anger)
        before = np.linalg.norm(s.snapshot())
        s.decay(0.5)
        after = np.linalg.norm(s.snapshot())
        assert after < before

    def test_decay_returns_self(self, anger):
        s = EmotionalState()
        s.apply(anger)
        result = s.decay(0.9)
        assert result is s

    def test_full_decay_approaches_zero(self, anger):
        s = EmotionalState()
        s.apply(anger)
        for _ in range(100):
            s.decay(0.5)
        assert np.allclose(s.snapshot(), 0, atol=1e-10)

    def test_reset_zeros_vector(self, anger):
        s = EmotionalState()
        s.apply(anger)
        s.reset()
        assert np.allclose(s.snapshot(), 0)

    def test_reset_returns_self(self):
        s = EmotionalState()
        assert s.reset() is s

    def test_dominant_returns_emotion(self, anger):
        s = EmotionalState()
        s.apply(anger, weight=5.0)
        dom = s.dominant()
        assert dom is not None

    def test_dominant_none_when_neutral(self):
        s = EmotionalState()
        assert s.dominant() is None

    def test_snapshot_is_copy(self, anger):
        s = EmotionalState()
        s.apply(anger)
        snap = s.snapshot()
        snap[0] = 999.0
        assert s.snapshot()[0] != 999.0

    def test_valence_method(self, joy):
        s = EmotionalState()
        s.apply(joy)
        assert s.valence() > 0

    def test_arousal_method(self, anger):
        s = EmotionalState()
        s.apply(anger)
        assert s.arousal() > 0

    def test_to_dict_from_dict_roundtrip(self, anger, fear):
        s = EmotionalState()
        s.apply(anger).apply(fear, 0.5)
        data = s.to_dict()
        s2 = EmotionalState.from_dict(data)
        assert s == s2

    def test_to_dict_has_vector_key(self, anger):
        s = EmotionalState()
        s.apply(anger)
        assert "vector" in s.to_dict()

    def test_add_two_states(self, anger, fear):
        s1 = EmotionalState()
        s1.apply(anger)
        s2 = EmotionalState()
        s2.apply(fear)
        s3 = s1 + s2
        assert isinstance(s3, EmotionalState)
        assert not np.allclose(s3.snapshot(), s1.snapshot())

    def test_mul_scales_state(self, anger):
        s = EmotionalState()
        s.apply(anger)
        s2 = s * 2.0
        assert np.allclose(s2.snapshot(), s.snapshot() * 2)

    def test_eq_equal_states(self, anger):
        s1 = EmotionalState()
        s1.apply(anger)
        s2 = EmotionalState()
        s2.apply(anger)
        assert s1 == s2

    def test_eq_different_states(self, anger, fear):
        s1 = EmotionalState()
        s1.apply(anger)
        s2 = EmotionalState()
        s2.apply(fear)
        assert s1 != s2

    def test_repr_contains_name(self, anger):
        s = EmotionalState()
        s.apply(anger, weight=5.0)
        r = repr(s)
        assert "EmotionalState" in r

    def test_repr_neutral(self):
        s = EmotionalState()
        assert "neutral" in repr(s)


class TestEmotionTimeline:
    def test_empty_timeline_len_zero(self):
        tl = EmotionTimeline()
        assert len(tl) == 0

    def test_append_increases_length(self):
        tl = EmotionTimeline()
        s = EmotionalState([1, 0, 0, 0])
        tl.append(s)
        assert len(tl) == 1

    def test_append_returns_self(self):
        tl = EmotionTimeline()
        s = EmotionalState()
        assert tl.append(s) is tl

    def test_drift_fewer_than_two_is_zero(self):
        tl = EmotionTimeline()
        assert np.allclose(tl.drift(), 0)

    def test_drift_two_snapshots(self):
        tl = EmotionTimeline()
        s1 = EmotionalState([1, 0, 0, 0])
        s2 = EmotionalState([3, 0, 0, 0])
        tl.append(s1).append(s2)
        assert tl.drift()[0] == pytest.approx(2.0)

    def test_dominant_sequence_length(self):
        tl = EmotionTimeline()
        tl.append(EmotionalState([2, 0, 0, 0]))
        tl.append(EmotionalState([0, 0, 2, 0]))
        seq = tl.dominant_sequence()
        assert len(seq) == 2

    def test_to_dict_from_dict_roundtrip(self):
        tl = EmotionTimeline(label="test")
        tl.append(EmotionalState([2, 0, 0, 0]))
        tl.append(EmotionalState([0, 1, 1, 0]))
        data = tl.to_dict()
        tl2 = EmotionTimeline.from_dict(data)
        assert tl2.label == "test"
        assert len(tl2) == 2
        assert np.allclose(tl2._snapshots[0], tl._snapshots[0])

    def test_repr(self):
        tl = EmotionTimeline("conv")
        assert "EmotionTimeline" in repr(tl)
        assert "conv" in repr(tl)
