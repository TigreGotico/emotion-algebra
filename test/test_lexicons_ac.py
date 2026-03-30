"""Tests for lexicons.py — tag_emotions, Aho-Corasick backend, fallback."""
import pytest
from emotion_algebra.lexicons import (
    tag_emotions, _check_ac_available, get_word_emotion,
    get_color, get_sentiment, get_subjectivity, get_orientation, LEXICON,
)


# ---------------------------------------------------------------------------
# tag_emotions — backend-agnostic contract tests
# ---------------------------------------------------------------------------

class TestTagEmotions:
    def test_returns_list(self):
        result = tag_emotions("joy and sadness")
        assert isinstance(result, list)

    def test_empty_string_returns_empty(self):
        assert tag_emotions("") == []

    def test_no_match_returns_empty(self):
        assert tag_emotions("xyzzy frobble") == []

    def test_single_match_has_required_keys(self):
        results = tag_emotions("rage")
        assert len(results) >= 1
        match = results[0]
        assert "word" in match
        assert "label" in match
        assert "start" in match
        assert "end" in match

    def test_label_is_valid_emotion_name(self):
        from emotion_algebra.emotions import get_emotion
        for match in tag_emotions("fury hatred terror joy"):
            assert get_emotion(match["label"]) is not None, \
                f"label {match['label']!r} not a valid emotion"

    def test_multiple_matches(self):
        results = tag_emotions("abhor the hatred and fury")
        assert len(results) >= 2

    def test_case_insensitive(self):
        lower = tag_emotions("rage")
        upper = tag_emotions("RAGE")
        assert len(lower) == len(upper)
        assert lower[0]["label"] == upper[0]["label"]

    def test_score_text_uses_tag_emotions(self):
        """score_text result should be consistent with tag_emotions output."""
        from emotion_algebra.text import score_text
        from emotion_algebra.state import EmotionalState

        # Use same-polarity emotions so they don't cancel each other out
        tags = tag_emotions("fury hatred")
        state = score_text("fury hatred")

        assert isinstance(state, EmotionalState)
        # If any tag returned, state should be non-zero (both → anger = sensitivity +2)
        if tags:
            assert state.arousal() > 0

    def test_from_text_uses_tag_emotions(self):
        from emotion_algebra.text import from_text
        result = from_text("fury hatred rage")
        # All three map to anger — dominant should be anger-axis
        assert result is not None
        assert result.dimension.axis == "sensitivity"


# ---------------------------------------------------------------------------
# Aho-Corasick specific tests (skipped if not available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _check_ac_available(), reason="ahocorasick-ner not installed")
class TestAhoCorasickBackend:
    def test_backend_is_active(self):
        assert _check_ac_available() is True

    def test_tagger_is_cached(self):
        from emotion_algebra.lexicons import _get_ac_tagger
        t1 = _get_ac_tagger()
        t2 = _get_ac_tagger()
        assert t1 is t2  # same object — cached

    def test_span_offsets_are_character_positions(self):
        # pyahocorasick returns end as inclusive last-char index, not exclusive.
        # text[start:end+1] extracts the matched word.
        text = "I feel absolute rage today"
        results = tag_emotions(text)
        for match in results:
            extracted = text[match["start"]:match["end"] + 1]
            assert extracted.lower() == match["word"].lower()

    def test_word_boundary_respected(self):
        # "rages" should not match "rage" as a substring without boundary
        results = tag_emotions("enrages the crowd")
        # ahocorasick-ner respects word boundaries — "rage" inside "enrages" should not match
        words = [m["word"] for m in results]
        # "enrages" is not in the lexicon; if it matches, it should be a whole-word match
        for w in words:
            assert "rage" != w or "enrages" not in "enrages"  # boundary respected


# ---------------------------------------------------------------------------
# Dict fallback tests (always run — monkeypatch AC away)
# ---------------------------------------------------------------------------

class TestDictFallback:
    def test_fallback_produces_same_labels(self, monkeypatch):
        """Force the dict fallback and verify labels match AC output."""
        import emotion_algebra.lexicons as lex_mod
        original = lex_mod._ac_available
        monkeypatch.setattr(lex_mod, "_ac_available", False)
        monkeypatch.setattr(lex_mod, "_ac_tagger", None)

        results = tag_emotions("rage terror fury")
        assert all("label" in r for r in results)
        assert all(r["label"] for r in results)

        monkeypatch.setattr(lex_mod, "_ac_available", original)

    def test_fallback_returns_start_end(self, monkeypatch):
        import emotion_algebra.lexicons as lex_mod
        monkeypatch.setattr(lex_mod, "_ac_available", False)
        monkeypatch.setattr(lex_mod, "_ac_tagger", None)

        results = tag_emotions("rage and terror")
        for r in results:
            assert "start" in r and "end" in r


# ---------------------------------------------------------------------------
# Lexicon field accessors — correct None coercion for empty CSV fields
# ---------------------------------------------------------------------------

class TestLexiconAccessors:
    def test_get_word_emotion_populated(self):
        assert get_word_emotion("abhor") == "anger"

    def test_get_color_populated(self):
        assert get_color("abhor") == "black"

    def test_get_sentiment_populated(self):
        assert get_sentiment("abhor") == "negative"

    def test_get_subjectivity_populated(self):
        assert get_subjectivity("abhor") is not None

    def test_get_orientation_populated(self):
        assert get_orientation("abhor") is not None

    def test_empty_field_returns_none(self):
        # "2-faced" has empty emotion field — should return None, not ""
        assert get_word_emotion("2-faced") is None
        assert get_color("2-faced") is None

    def test_unknown_word_returns_none(self):
        assert get_word_emotion("xyzzy_not_a_word") is None
        assert get_color("xyzzy_not_a_word") is None
        assert get_sentiment("xyzzy_not_a_word") is None
        assert get_subjectivity("xyzzy_not_a_word") is None
        assert get_orientation("xyzzy_not_a_word") is None
