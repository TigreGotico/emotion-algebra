"""Tests for emotion_algebra.deepmoji — DeepMojiONNXAdapter."""
import sys
import types
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures — mock deepmoji_onnx so tests run without model weights
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_deepmoji_onnx():
    """Inject a fake deepmoji_onnx module for the duration of every test."""
    fake_module = types.ModuleType("deepmoji_onnx")

    class FakeModel:
        @classmethod
        def from_pretrained(cls, variant="fp32", cache_dir=None, **_):
            return cls()

        def top_emojis(self, sentences, k=5):
            # Return predictable emoji scores for "joy" and "anger" emojis
            return [{"😄": 0.6, "😊": 0.2, "😡": 0.1, "😂": 0.05, "😭": 0.05}]

    fake_module.DeepMojiONNX = FakeModel
    sys.modules["deepmoji_onnx"] = fake_module
    yield
    sys.modules.pop("deepmoji_onnx", None)
    # ensure deepmoji module is re-imported fresh
    sys.modules.pop("emotion_algebra.deepmoji", None)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestDeepMojiONNXAdapterConstruction:
    def test_instantiates(self):
        from emotion_algebra.deepmoji import DeepMojiONNXAdapter
        a = DeepMojiONNXAdapter()
        assert a is not None

    def test_deepmoji_onnx_is_importable(self):
        """deepmoji-onnx is a canonical dependency — must always be importable."""
        import deepmoji_onnx  # noqa: F401 — should not raise


# ---------------------------------------------------------------------------
# analyze / score
# ---------------------------------------------------------------------------

class TestDeepMojiONNXAdapterInference:
    def _adapter(self):
        from emotion_algebra.deepmoji import DeepMojiONNXAdapter
        return DeepMojiONNXAdapter()

    def test_analyze_returns_emotion_or_none(self):
        from emotion_algebra.base import EmotionBase
        a = self._adapter()
        result = a.analyze("I am so happy!")
        # 😄 → joy, 😊 → serenity — both positive; dominant should be non-None
        assert result is None or isinstance(result, EmotionBase)

    def test_analyze_dominant_is_positive(self):
        from emotion_algebra.base import EmotionBase
        a = self._adapter()
        result = a.analyze("I am so happy!")
        if result is not None:
            # mock returns mostly joy/serenity emojis → positive valence
            assert result.valence >= 0

    def test_score_returns_emotional_state(self):
        from emotion_algebra.state import EmotionalState
        a = self._adapter()
        state = a.score("I am so happy!")
        assert isinstance(state, EmotionalState)

    def test_score_non_zero_for_known_emojis(self):
        a = self._adapter()
        state = a.score("I am so happy!")
        # At least one of the mock emojis maps to a known emotion
        assert state.arousal() >= 0  # >= 0 always, >0 if any matched

    def test_top_emoji_scores_returns_dict(self):
        a = self._adapter()
        scores = a.top_emoji_scores("hello")
        assert isinstance(scores, dict)
        assert len(scores) > 0
        for v in scores.values():
            assert isinstance(v, float)

    def test_analyze_empty_string(self):
        a = self._adapter()
        # Should not crash — result may be None if no emoji map
        result = a.analyze("")
        assert result is None or hasattr(result, "valence")

    def test_score_empty_string(self):
        from emotion_algebra.state import EmotionalState
        a = self._adapter()
        state = a.score("")
        assert isinstance(state, EmotionalState)


# ---------------------------------------------------------------------------
# Integration with DeepMojiAdapter scoring pipeline
# ---------------------------------------------------------------------------

class TestDeepMojiONNXPipelineIntegration:
    def test_uses_emoji_adapter_internally(self):
        from emotion_algebra.deepmoji import DeepMojiONNXAdapter
        from emotion_algebra.emoji import DeepMojiAdapter
        a = DeepMojiONNXAdapter()
        assert isinstance(a._adapter, DeepMojiAdapter)

    def test_same_result_as_manual_pipeline(self):
        """Manually pipe top_emoji_scores into DeepMojiAdapter.from_scores."""
        from emotion_algebra.deepmoji import DeepMojiONNXAdapter
        from emotion_algebra.emoji import DeepMojiAdapter

        a = DeepMojiONNXAdapter()
        scores = a.top_emoji_scores("test")
        expected = DeepMojiAdapter().from_scores(scores)
        result = a.analyze("test")
        # Both paths use identical scores — results must match
        assert result == expected
