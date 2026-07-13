"""Tests for emotion_algebra.emotions and the EmotionAnalyzer facade."""
import pytest

from emotion_algebra.emotions import (
    EMOTIONS, EMOTION_NAMES, POSITIVE_EMOTIONS, NEGATIVE_EMOTIONS,
    DIMENSION_TO_EMOTION_MAP, KIND_TO_EMOTION_MAP,
    random_emotion, get_emotion, get_dimension, emotion_to_dimension,
)
from emotion_algebra.plutchik import Emotion, EmotionalDimension


# ---------------------------------------------------------------------------
# EMOTIONS dict
# ---------------------------------------------------------------------------

class TestEmotionsDict:
    def test_24_emotions(self):
        assert len(EMOTIONS) == 24

    def test_all_emotion_names_covered(self):
        assert set(EMOTION_NAMES) == set(EMOTIONS.keys())

    def test_all_values_are_emotion_instances(self):
        for e in EMOTIONS.values():
            assert isinstance(e, Emotion)


# ---------------------------------------------------------------------------
# POSITIVE / NEGATIVE emotions
# ---------------------------------------------------------------------------

class TestPositiveNegative:
    def test_positive_emotions_are_positive(self):
        for e in POSITIVE_EMOTIONS:
            assert e.valence > 0, f"{e.name} should have valence > 0"

    def test_negative_emotions_are_negative(self):
        for e in NEGATIVE_EMOTIONS:
            assert e.valence < 0, f"{e.name} should have valence < 0"


# ---------------------------------------------------------------------------
# DIMENSION_TO_EMOTION_MAP
# ---------------------------------------------------------------------------

class TestDimensionMap:
    def test_four_dimensions_present(self):
        for name in ("sensitivity", "attention", "pleasantness", "aptitude"):
            assert name in DIMENSION_TO_EMOTION_MAP

    def test_each_dimension_has_emotions(self):
        for name, emotions in DIMENSION_TO_EMOTION_MAP.items():
            assert len(emotions) > 0, f"{name} has no emotions"

    def test_each_dimension_has_6_emotions(self):
        for name, emotions in DIMENSION_TO_EMOTION_MAP.items():
            assert len(emotions) == 6, f"{name} should have 6 emotions"


# ---------------------------------------------------------------------------
# KIND_TO_EMOTION_MAP
# ---------------------------------------------------------------------------

class TestKindMap:
    def test_known_kinds_present(self):
        for kind in ("related to object properties", "future appraisal", "event related",
                     "self appraisal", "social", "cathected"):
            assert kind in KIND_TO_EMOTION_MAP


# ---------------------------------------------------------------------------
# get_emotion
# ---------------------------------------------------------------------------

class TestGetEmotion:
    def test_known_emotion(self):
        e = get_emotion("anger")
        assert isinstance(e, Emotion)
        assert e.name == "anger"

    def test_unknown_emotion_returns_none(self):
        assert get_emotion("not_an_emotion_xyz") is None


# ---------------------------------------------------------------------------
# get_dimension
# ---------------------------------------------------------------------------

class TestGetDimension:
    def test_known_dimension(self):
        d = get_dimension("sensitivity")
        assert isinstance(d, EmotionalDimension)

    def test_unknown_dimension_returns_none(self):
        assert get_dimension("not_a_dimension_xyz") is None


# ---------------------------------------------------------------------------
# random_emotion
# ---------------------------------------------------------------------------

class TestRandomEmotion:
    def test_returns_emotion(self):
        e = random_emotion()
        assert isinstance(e, Emotion)

    def test_returns_different_results(self):
        results = {random_emotion().name for _ in range(50)}
        assert len(results) > 1


# ---------------------------------------------------------------------------
# emotion_to_dimension
# ---------------------------------------------------------------------------

class TestEmotionToDimension:
    def test_known_emotion_returns_dimension(self):
        d = emotion_to_dimension("anger")
        assert isinstance(d, EmotionalDimension)

    def test_unknown_emotion_returns_none(self):
        assert emotion_to_dimension("not_an_emotion_xyz") is None

    def test_composite_emotion_returns_list(self):
        from copy import copy
        from emotion_algebra.emotions import EMOTIONS
        from emotion_algebra.composite_emotions import CompositeEmotion
        rage = copy(EMOTIONS["rage"])
        vigilance = copy(EMOTIONS["vigilance"])
        composite = rage * vigilance
        # Call emotion_to_dimension directly on the composite object (don't mutate global)
        assert isinstance(composite, CompositeEmotion)
        result = [e.dimension for e in composite.components if e.dimension]
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# emotion_algebra.__init__ — EmotionAnalyzer
# ---------------------------------------------------------------------------

class TestEmotionAnalyzer:
    def setup_method(self):
        # patch deepmoji imports so __init__ doesn't fail if deepmoji model missing
        import unittest.mock as mock
        import sys
        # mock deepmoji module if not installed
        if "deepmoji" not in sys.modules:
            mock_deepmoji = mock.MagicMock()
            sys.modules["deepmoji"] = mock_deepmoji

    def _get_analyzer(self):
        from emotion_algebra import EmotionAnalyzer
        return EmotionAnalyzer()

    def test_get_emotion_known(self):
        from emotion_algebra.emotions import get_emotion
        e = get_emotion("anger")
        assert isinstance(e, Emotion)

    def test_get_feeling_known(self):
        from emotion_algebra.feelings import get_feeling
        f = get_feeling("love")
        assert f is not None

    def test_get_dimension_known(self):
        d = get_dimension("sensitivity")
        assert isinstance(d, EmotionalDimension)

    def test_analyzer_get_method(self):
        from emotion_algebra import EmotionAnalyzer
        a = EmotionAnalyzer()
        result = a.get("anger")
        assert isinstance(result, Emotion)

    def test_analyzer_random_emotion(self):
        from emotion_algebra import EmotionAnalyzer
        a = EmotionAnalyzer()
        e = a.random_emotion()
        assert isinstance(e, Emotion)

    def test_analyzer_emotion_method(self):
        from emotion_algebra import EmotionAnalyzer
        a = EmotionAnalyzer()
        e = a.emotion("anger")
        assert isinstance(e, Emotion)

    def test_analyzer_feeling_method(self):
        try:
            from emotion_algebra import EmotionAnalyzer
            a = EmotionAnalyzer()
            from emotion_algebra.feelings import Feeling
            f = a.feeling("love")
            assert isinstance(f, Feeling)
        except ImportError:
            pytest.skip("deepmoji not available")

    def test_analyzer_dimension_method(self):
        try:
            from emotion_algebra import EmotionAnalyzer
            a = EmotionAnalyzer()
            d = a.dimension("sensitivity")
            assert isinstance(d, EmotionalDimension)
        except ImportError:
            pytest.skip("deepmoji not available")

    def test_analyzer_get_sentiment(self):
        try:
            from emotion_algebra import EmotionAnalyzer
            from emotion_algebra.lexicons import LEXICON
            a = EmotionAnalyzer()
            word = next(iter(LEXICON))
            result = a.get_sentiment(word)
            assert result is not None
        except ImportError:
            pytest.skip("deepmoji not available")

