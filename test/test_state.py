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


class TestMood:
    """Mood is a slow EMA over applied emotions; emotion is the instant reading.

    The distinction is Mehrabian's (temperament vs emotion): a single event must
    not swing an agent's temperament, but a sustained run of them must.
    """

    def test_starts_neutral(self):
        assert list(EmotionalState().mood) == [0.0, 0.0, 0.0, 0.0]

    def test_one_event_barely_moves_mood(self):
        state = EmotionalState()
        state.apply(get_emotion("ecstasy"))
        # The instant emotion is fully felt...
        assert np.max(np.abs(state.snapshot())) > 1.0
        # ...but mood has only crept toward it.
        assert np.max(np.abs(state.mood)) < np.max(np.abs(state.snapshot()))

    def test_sustained_events_move_mood(self):
        brief, sustained = EmotionalState(), EmotionalState()
        brief.apply(get_emotion("ecstasy"))
        for _ in range(50):
            sustained.apply(get_emotion("ecstasy"))
        assert np.max(np.abs(sustained.mood)) > np.max(np.abs(brief.mood))

    def test_mood_tracks_the_sign_of_sustained_input(self):
        state = EmotionalState()
        for _ in range(50):
            state.apply(get_emotion("grief"))
        assert state.mood[2] < 0  # pleasantness axis

    def test_dominant_mood_names_the_sustained_emotion(self):
        state = EmotionalState()
        for _ in range(50):
            state.apply(get_emotion("ecstasy"))
        assert state.dominant_mood() is not None
        assert state.dominant_mood().polarity > 0


class TestHalfLifeDecay:
    def test_one_half_life_halves_the_state(self):
        state = EmotionalState()
        state.apply(get_emotion("ecstasy"))
        before = state.snapshot().copy()
        state.decay_halflife(dt=10.0, half_life=10.0)
        assert list(state.snapshot()) == pytest.approx(list(before * 0.5))

    def test_two_half_lives_quarter_it(self):
        state = EmotionalState()
        state.apply(get_emotion("ecstasy"))
        before = state.snapshot().copy()
        state.decay_halflife(dt=20.0, half_life=10.0)
        assert list(state.snapshot()) == pytest.approx(list(before * 0.25))

    def test_zero_elapsed_time_is_a_no_op(self):
        state = EmotionalState()
        state.apply(get_emotion("ecstasy"))
        before = state.snapshot().copy()
        state.decay_halflife(dt=0.0, half_life=10.0)
        assert list(state.snapshot()) == pytest.approx(list(before))

    @pytest.mark.parametrize("half_life", [0.0, -1.0])
    def test_non_positive_half_life_rejected(self, half_life):
        with pytest.raises(ValueError):
            EmotionalState().decay_halflife(dt=1.0, half_life=half_life)

    def test_negative_dt_rejected(self):
        with pytest.raises(ValueError):
            EmotionalState().decay_halflife(dt=-1.0, half_life=10.0)

    @pytest.mark.parametrize("factor", [0.0, -0.1, 1.5])
    def test_factor_decay_validates_its_range(self, factor):
        with pytest.raises(ValueError):
            EmotionalState().decay(factor)


class TestLovheimReadout:
    def test_state_projects_into_the_cube(self):
        state = EmotionalState()
        state.apply(get_emotion("ecstasy"))
        point = state.to_lovheim()
        assert point.serotonin > 0.5
        assert point.dopamine > 0.5

    def test_neutral_state_sits_at_baseline(self):
        assert EmotionalState().to_lovheim().is_baseline

    def test_timeline_yields_a_neurochemical_sequence(self):
        timeline = EmotionTimeline()
        for name in ("ecstasy", "grief"):
            state = EmotionalState()
            state.apply(get_emotion(name))
            timeline.append(state)
        sequence = timeline.lovheim_sequence()
        assert len(sequence) == 2
        assert sequence[0].serotonin > sequence[1].serotonin
