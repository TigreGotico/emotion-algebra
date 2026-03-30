"""Unit tests for emotion_data algebra — deterministic, no external dependencies required."""
import pytest
from copy import copy

from emotion_data.emotions import EMOTIONS
from emotion_data.feelings import FEELINGS
from emotion_data.plutchik import Emotion, Neutrality, DIMENSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def emo(name: str) -> Emotion:
    """Return a copy of the named emotion from the EMOTIONS registry."""
    e = EMOTIONS[name]
    return copy(e)


# ---------------------------------------------------------------------------
# Intensity arithmetic
# ---------------------------------------------------------------------------

class TestIntensityArithmetic:
    def test_annoyance_plus_1_equals_anger(self) -> None:
        result = emo("annoyance") + 1
        assert result.name == "anger", f"Expected anger, got {result.name}"

    def test_anger_plus_1_equals_rage(self) -> None:
        result = emo("anger") + 1
        assert result.name == "rage", f"Expected rage, got {result.name}"

    def test_serenity_minus_1_goes_down(self) -> None:
        # serenity is primary (flow=1), subtracting 1 brings flow to 0 → neutrality
        result = emo("serenity") - 1
        assert result.emotional_flow == 0, (
            f"Expected flow==0 (neutrality), got flow={result.emotional_flow}"
        )

    def test_anger_minus_1_equals_annoyance(self) -> None:
        result = emo("anger") - 1
        assert result.name == "annoyance", f"Expected annoyance, got {result.name}"

    def test_int_of_emotion_equals_flow(self) -> None:
        assert int(emo("anger")) == 2  # secondary → flow 2
        assert int(emo("annoyance")) == 1  # primary → flow 1
        assert int(emo("rage")) == 3  # tertiary → flow 3


# ---------------------------------------------------------------------------
# Negation / opposite
# ---------------------------------------------------------------------------

class TestNegation:
    def test_neg_anger_equals_fear(self) -> None:
        result = -emo("anger")
        assert result.name == "fear", f"Expected fear, got {result.name}"

    def test_neg_joy_equals_sadness(self) -> None:
        result = -emo("joy")
        assert result.name == "sadness", f"Expected sadness, got {result.name}"

    def test_neg_rage_equals_terror(self) -> None:
        result = -emo("rage")
        assert result.name == "terror", f"Expected terror, got {result.name}"

    def test_neg_trust_equals_disgust(self) -> None:
        result = -emo("trust")
        assert result.name == "disgust", f"Expected disgust, got {result.name}"


# ---------------------------------------------------------------------------
# Hyperintensity (offsets beyond flow=3)
# ---------------------------------------------------------------------------

class TestHyperintensity:
    def test_rage_plus_large_number_gives_hyper_name(self) -> None:
        result = emo("rage") + 5  # flow=3, +5 → offset=5
        assert "hyper" in result.name.lower(), (
            f"Expected 'hyper' in name, got {result.name!r}"
        )

    def test_rage_plus_1_gives_mega_prefix(self) -> None:
        result = emo("rage") + 1  # flow=3, +1 → offset=1
        assert "mega" in result.name.lower(), (
            f"Expected 'mega' in name, got {result.name!r}"
        )

    def test_intensity_offset_stored(self) -> None:
        result = emo("rage") + 3
        assert result.intensity_offset == 3


# ---------------------------------------------------------------------------
# Composition → Feeling
# ---------------------------------------------------------------------------

class TestComposition:
    def test_joy_plus_trust_returns_composite(self) -> None:
        from emotion_data.composite_emotions import CompositeEmotion
        result = emo("joy") + emo("trust")
        assert isinstance(result, CompositeEmotion), f"Expected CompositeEmotion, got {type(result)}"

    def test_joy_plus_surprise_returns_composite(self) -> None:
        from emotion_data.composite_emotions import CompositeEmotion
        result = emo("joy") + emo("surprise")
        assert isinstance(result, CompositeEmotion), f"Expected CompositeEmotion, got {type(result)}"

    def test_fear_plus_sadness_returns_composite(self) -> None:
        from emotion_data.composite_emotions import CompositeEmotion
        result = emo("fear") + emo("sadness")
        assert isinstance(result, CompositeEmotion), f"Expected CompositeEmotion, got {type(result)}"


# ---------------------------------------------------------------------------
# Emotion vectors (4D)
# ---------------------------------------------------------------------------

class TestEmotionVectors:
    def test_emotion_vector_has_4_elements(self) -> None:
        vec = emo("joy").emotion_vector
        assert len(vec) == 4, f"Expected 4 elements, got {len(vec)}"

    def test_emotion_vector_elements_are_emotion_instances(self) -> None:
        vec = emo("joy").emotion_vector
        for elem in vec:
            assert isinstance(elem, Emotion), f"Expected Emotion, got {type(elem)}"

    def test_as_array_shape(self) -> None:
        import numpy as np
        arr = emo("joy").as_array
        assert isinstance(arr, np.ndarray), f"Expected ndarray, got {type(arr)}"
        assert arr.shape == (4,), f"Expected shape (4,), got {arr.shape}"

    def test_joy_in_pleasantness_dimension(self) -> None:
        # joy belongs to pleasantness → its vector should have non-neutral pleasantness
        vec = emo("joy").emotion_vector
        sensitivity, attention, pleasantness, aptitude = vec
        assert pleasantness.emotional_flow != 0, "joy should have non-zero pleasantness flow"
        assert sensitivity.emotional_flow == 0
        assert attention.emotional_flow == 0
        assert aptitude.emotional_flow == 0


# ---------------------------------------------------------------------------
# Valence property
# ---------------------------------------------------------------------------

class TestValence:
    def test_joy_valence_is_pleasantness_flow(self) -> None:
        # joy is Pleasantness +2
        assert emo("joy").valence == 2

    def test_ecstasy_valence_is_plus_3(self) -> None:
        assert emo("ecstasy").valence == 3

    def test_serenity_valence_is_plus_1(self) -> None:
        assert emo("serenity").valence == 1

    def test_sadness_valence_is_minus_2(self) -> None:
        assert emo("sadness").valence == -2

    def test_grief_valence_is_minus_3(self) -> None:
        assert emo("grief").valence == -3

    def test_pensiveness_valence_is_minus_1(self) -> None:
        assert emo("pensiveness").valence == -1

    def test_neutrality_valence_is_zero(self) -> None:
        assert Neutrality().valence == 0

    def test_anger_valence_is_zero(self) -> None:
        # Sensitivity axis — arousal is orthogonal to hedonics
        assert emo("anger").valence == 0

    def test_fear_valence_is_zero(self) -> None:
        # Sensitivity axis — arousal is orthogonal to hedonics
        assert emo("fear").valence == 0

    def test_anticipation_valence_is_zero(self) -> None:
        # Attention axis — engagement is orthogonal to hedonics
        assert emo("anticipation").valence == 0

    def test_trust_valence_is_zero(self) -> None:
        # Aptitude axis — not Pleasantness, so valence = 0
        assert emo("trust").valence == 0


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------

class TestComparisons:
    def test_rage_greater_than_anger(self) -> None:
        assert emo("rage") > emo("anger"), "rage (flow=3) should be > anger (flow=2)"

    def test_anger_greater_than_annoyance(self) -> None:
        assert emo("anger") > emo("annoyance"), "anger (flow=2) > annoyance (flow=1)"

    def test_annoyance_less_than_rage(self) -> None:
        assert emo("annoyance") < emo("rage")

    def test_equal_emotions(self) -> None:
        assert emo("joy") == emo("joy")

    def test_unequal_different_dimension(self) -> None:
        # joy vs anger are in different dimensions → not equal
        assert emo("joy") != emo("anger")


# ---------------------------------------------------------------------------
# Neutrality identity element
# ---------------------------------------------------------------------------

class TestNeutralityIdentity:
    def test_emotion_plus_neutrality_returns_self(self) -> None:
        joy = emo("joy")
        result = joy + Neutrality()
        assert result.name == joy.name

    def test_neutrality_plus_emotion_returns_emotion(self) -> None:
        joy = emo("joy")
        result = Neutrality() + joy
        assert result.name == joy.name

    def test_neutrality_flow_is_zero(self) -> None:
        assert Neutrality().emotional_flow == 0

    def test_neutrality_intensity_is_null(self) -> None:
        assert Neutrality().intensity == "null"
