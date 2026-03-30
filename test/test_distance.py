"""Tests for emotion_algebra.distance — emotion_distance, closest_emotion, emotion_clusters."""
from copy import copy

import numpy as np
import pytest

from emotion_algebra.distance import emotion_distance, closest_emotion, emotion_clusters
from emotion_algebra.emotions import EMOTIONS, get_emotion
from emotion_algebra.plutchik import Neutrality


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
        # [0,0,0,0] — equidistant from many primary emotions; returns one deterministically
        result = closest_emotion([0.0, 0.0, 0.0, 0.0])
        assert result is not None

    def test_returns_named_emotion(self):
        result = closest_emotion([0, 0, 3, 0])
        assert result.name in EMOTIONS

    def test_numpy_array_input(self):
        result = closest_emotion(np.array([0.0, 0.0, -2.0, 0.0]))
        assert result.name == "sadness"


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
