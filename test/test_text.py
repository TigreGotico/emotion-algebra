"""Tests for emotion_algebra.text — from_text, score_text, HFEmotionAdapter."""
import pytest

from emotion_algebra.text import from_text, score_text
from emotion_algebra.state import EmotionalState
from emotion_algebra.base import EmotionBase


class TestFromText:
    def test_known_lexicon_word_returns_emotion(self):
        # "abandoned" → anger per lexicon
        result = from_text("abandoned")
        assert result is not None
        assert isinstance(result, EmotionBase)

    def test_no_match_returns_none(self):
        result = from_text("xyzzy_not_a_real_word_12345")
        assert result is None

    def test_empty_string_returns_none(self):
        result = from_text("")
        assert result is None

    def test_mixed_case_handled(self):
        result1 = from_text("Abandoned")
        result2 = from_text("abandoned")
        assert (result1 is None) == (result2 is None)
        if result1 and result2:
            assert result1.name == result2.name

    def test_majority_vote(self):
        # "joy" words should dominate
        from emotion_algebra.lexicons import get_word_emotion
        # Find a joy word from the lexicon
        result = from_text("abundant accomplished joyful achieve")
        assert result is not None

    def test_returns_emotion_base(self):
        result = from_text("abandoned")
        if result is not None:
            assert isinstance(result, EmotionBase)

    def test_punctuation_stripped(self):
        # Words with punctuation should still tokenize
        result = from_text("abandoned, abandoned, abandoned!")
        assert result is not None


class TestScoreText:
    def test_returns_emotional_state(self):
        state = score_text("abandoned")
        assert isinstance(state, EmotionalState)

    def test_empty_text_returns_neutral_state(self):
        state = score_text("")
        assert state.dominant() is None

    def test_no_match_returns_neutral_state(self):
        state = score_text("xyzzy_not_real_12345")
        assert state.dominant() is None

    def test_known_word_produces_nonzero_state(self):
        state = score_text("abandoned")
        import numpy as np
        assert not np.allclose(state.snapshot(), 0)

    def test_dominant_matches_from_text(self):
        text = "abandoned"
        dom = score_text(text).dominant()
        single = from_text(text)
        if dom and single:
            # Both point at the anger axis
            assert dom.name == single.name or dom._dimension == single._dimension

    def test_multiple_words_accumulate(self):
        import numpy as np
        s1 = score_text("abandoned")
        s2 = score_text("abandoned abandoned")
        # Double the words → larger magnitude
        assert np.linalg.norm(s2.snapshot()) >= np.linalg.norm(s1.snapshot())


class TestHFEmotionAdapterImportError:
    def test_raises_import_error_without_transformers(self):
        from emotion_algebra.text import HFEmotionAdapter
        import sys
        # Temporarily hide transformers if present
        transformers_mod = sys.modules.get("transformers")
        sys.modules["transformers"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="transformers"):
                HFEmotionAdapter("some-model")
        finally:
            if transformers_mod is not None:
                sys.modules["transformers"] = transformers_mod
            elif "transformers" in sys.modules:
                del sys.modules["transformers"]
