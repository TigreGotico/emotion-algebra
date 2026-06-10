"""Tests for emotion_algebra.distance — emotion_distance, closest_emotion, emotion_clusters."""
import random
from copy import copy

import numpy as np
import pytest

from emotion_algebra.distance import (
    NEUTRAL_RADIUS,
    _nearest,
    closest_emotion,
    emotion_clusters,
    emotion_distance,
)
from emotion_algebra.base import EmotionBase
from emotion_algebra.emotions import EMOTIONS, get_emotion
from emotion_algebra.feelings import FEELINGS, Feeling
from emotion_algebra.plutchik import Emotion, Neutrality


@pytest.fixture
def anger():
    return copy(EMOTIONS["anger"])

@pytest.fixture
def rage():
    return copy(EMOTIONS["rage"])

@pytest.fixture
def joy():
    return copy(EMOTIONS["joy"])

@pytest.fixture
def sadness():
    return copy(EMOTIONS["sadness"])


class TestEmotionDistance:
    def test_same_emotion_is_zero(self, anger):
        assert emotion_distance(anger, anger) == 0.0

    def test_adjacent_intensity_is_one(self, anger, rage):
        # rage flow=3, anger flow=2 — same axis, distance 1
        assert emotion_distance(rage, anger) == pytest.approx(1.0)

    def test_opposite_pleasantness_emotions(self, joy, sadness):
        # joy flow=+2, sadness flow=-2 on pleasantness axis → distance 4
        assert emotion_distance(joy, sadness) == pytest.approx(4.0)

    def test_neutrality_to_joy(self, joy):
        n = Neutrality()
        assert emotion_distance(n, joy) == pytest.approx(2.0)

    def test_returns_float(self, anger, joy):
        d = emotion_distance(anger, joy)
        assert isinstance(d, float)

    def test_symmetric(self, anger, joy):
        assert emotion_distance(anger, joy) == emotion_distance(joy, anger)

    def test_cross_axis_distance(self, anger, joy):
        # anger is [2,0,0,0], joy is [0,0,2,0] — distance = sqrt(4+4) = 2√2
        assert emotion_distance(anger, joy) == pytest.approx(np.sqrt(8))


class TestClosestEmotion:
    def test_exact_joy_vector(self):
        result = closest_emotion([0, 0, 2, 0])
        assert result.name == "joy"

    def test_exact_anger_vector(self):
        result = closest_emotion([2, 0, 0, 0])
        assert result.name == "anger"

    def test_near_anger_vector(self):
        # Slightly off anger — should still return anger or an adjacent emotion
        result = closest_emotion([2.1, 0.0, 0.0, 0.0])
        assert result is not None

    def test_zero_vector_returns_emotion(self):
        # Contract: the origin is equidistant from every primary emotion, so no
        # basis name is honest — it resolves to Neutrality instead of an
        # arbitrary winner decided by iteration order.
        result = closest_emotion([0.0, 0.0, 0.0, 0.0])
        assert result is not None
        assert isinstance(result, Neutrality)
        assert result.name == "neutrality"

    def test_returns_named_emotion(self):
        result = closest_emotion([0, 0, 3, 0])
        assert result.name in EMOTIONS

    def test_numpy_array_input(self):
        result = closest_emotion(np.array([0.0, 0.0, -2.0, 0.0]))
        assert result.name == "sadness"


class TestClosestEmotionFeelings:
    """Feelings in the candidate set: mixed-axis states get honest names."""

    def test_all_positive_mixed_vector_names_positive_feeling(self):
        # [0.6, 0.6, 0.6, 0.6] is equidistant from all four positive basic
        # emotions; with feelings included it resolves to a strictly closer
        # positive dyad — never "annoyance".
        result = closest_emotion([0.6, 0.6, 0.6, 0.6])
        assert result.name != "annoyance"
        assert result.name == "acknowledgement"  # serenity + acceptance
        assert isinstance(result, Feeling)
        assert result.valence > 0

    def test_feeling_pole_names_the_feeling(self):
        # love = joy + trust → [0, 0, 2, 2]
        result = closest_emotion([0, 0, 2, 2])
        assert result.name == "love"

    def test_include_feelings_false_restricts_to_basis(self):
        result = closest_emotion([0.6, 0.6, 0.6, 0.6], include_feelings=False)
        assert isinstance(result, Emotion)
        assert result.name in EMOTIONS
        # deterministic tie-break: equal distance and alignment across the four
        # positive basics → lexicographically smallest name
        assert result.name == "acceptance"

    def test_basis_poles_still_name_basis_emotion(self):
        # exact basis vectors win over any feeling, for all 24 named emotions
        for name, emo in EMOTIONS.items():
            assert closest_emotion(emo.as_array).name == name

    def test_return_type_is_emotion_base_with_name(self):
        # backward-compat: callers rely on `.name` (str) on the returned object
        for vec in ([0, 0, 2, 0], [0.6, 0.6, 0.6, 0.6], [0, 0, 0, 0]):
            result = closest_emotion(vec)
            assert isinstance(result, EmotionBase)
            assert isinstance(result.name, str) and result.name


class TestClosestEmotionNeutral:
    def test_zero_vector_is_neutral(self):
        result = closest_emotion([0, 0, 0, 0])
        assert isinstance(result, Neutrality)
        assert result.name == "neutrality"
        assert np.allclose(result.as_array, 0)

    def test_inside_neutral_radius_is_neutral(self):
        # norm = 0.2 < NEUTRAL_RADIUS
        result = closest_emotion([0.1, 0.1, 0.1, 0.1])
        assert result.name == "neutrality"

    def test_outside_neutral_radius_is_named(self):
        # norm = 0.4 >= NEUTRAL_RADIUS → a real named state
        result = closest_emotion([0.2, 0.2, 0.2, 0.2])
        assert result.name != "neutrality"

    def test_neutral_radius_relative_to_intensity_one_shell(self):
        # basic emotions sit on the intensity-1 shell; the neutral zone must
        # stay well inside it so faint-but-real emotions are not swallowed
        assert 0 < NEUTRAL_RADIUS < 1
        assert closest_emotion([0.9, 0, 0, 0]).name == "annoyance"


class TestClosestEmotionDeterminism:
    def test_deterministic_across_permuted_candidate_order(self):
        candidates = list(EMOTIONS.values()) + list(FEELINGS.values())
        vec = np.array([0.6, 0.6, 0.6, 0.6])
        baseline = _nearest(vec, candidates).name
        rng = random.Random(1234)
        for _ in range(10):
            shuffled = candidates[:]
            rng.shuffle(shuffled)
            assert _nearest(vec, shuffled).name == baseline

    def test_tie_break_prefers_alignment_over_name(self):
        # [1.5, 0, 0, 0] is equidistant (0.5) from annoyance [1,0,0,0] and
        # anger [2,0,0,0]; anger has higher dot product with the query and must
        # win even though "annoyance" sorts first lexicographically.
        assert closest_emotion([1.5, 0, 0, 0]).name == "anger"

    def test_repeated_calls_agree(self):
        vec = [0.6, 0.6, 0.6, 0.6]
        names = {closest_emotion(vec).name for _ in range(5)}
        assert len(names) == 1


class TestEmotionClusters:
    def test_returns_list_of_lists(self):
        clusters = emotion_clusters()
        assert isinstance(clusters, list)
        assert all(isinstance(c, list) for c in clusters)

    def test_all_emotions_covered(self):
        clusters = emotion_clusters()
        covered = {e.name for cluster in clusters for e in cluster}
        assert covered == set(EMOTIONS.keys())

    def test_tight_threshold_more_clusters(self):
        tight = emotion_clusters(threshold=0.5)
        loose = emotion_clusters(threshold=3.0)
        assert len(tight) >= len(loose)

    def test_no_duplicates(self):
        clusters = emotion_clusters()
        seen = []
        for cluster in clusters:
            for e in cluster:
                seen.append(e.name)
        assert len(seen) == len(set(seen))
