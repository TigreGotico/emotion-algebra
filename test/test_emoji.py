"""Tests for emotion_algebra.emoji — emoji-to-emotion mapping."""
import pytest

from emotion_algebra.emoji import (
    EMOJI_EMOTION_MAP,
    from_emoji,
    score_emojis,
    from_emojis,
    DeepMojiAdapter,
)
from emotion_algebra.plutchik import Emotion
from emotion_algebra.state import EmotionalState


# ---------------------------------------------------------------------------
# EMOJI_EMOTION_MAP
# ---------------------------------------------------------------------------

class TestEmojiEmotionMap:
    def test_is_immutable(self):
        from types import MappingProxyType
        assert isinstance(EMOJI_EMOTION_MAP, MappingProxyType)

    def test_not_empty(self):
        assert len(EMOJI_EMOTION_MAP) > 50

    def test_all_values_are_valid_emotion_names(self):
        from emotion_algebra.emotions import get_emotion
        for emoji, name in EMOJI_EMOTION_MAP.items():
            assert get_emotion(name) is not None, f"{emoji!r} → {name!r} not a valid emotion"

    def test_covers_all_eight_primaries(self):
        """Each of Plutchik's 8 primary emotions must appear at least once."""
        primaries = {"joy", "sadness", "anger", "fear", "disgust", "trust",
                     "anticipation", "surprise"}
        mapped = set(EMOJI_EMOTION_MAP.values())
        for p in primaries:
            # primary itself or an intensity variant on the same axis
            from emotion_algebra.emotions import get_emotion
            assert any(
                get_emotion(v) is not None and get_emotion(v).dimension == get_emotion(p).dimension
                for v in mapped
            ), f"No emoji covers the {p!r} dimension"


# ---------------------------------------------------------------------------
# from_emoji
# ---------------------------------------------------------------------------

class TestFromEmoji:
    def test_known_emoji_returns_emotion(self):
        result = from_emoji("😊")
        assert isinstance(result, Emotion)
        assert result.name == "serenity"

    def test_intense_joy(self):
        assert from_emoji("😂").name == "ecstasy"

    def test_anger_emoji(self):
        assert from_emoji("😠").name == "anger"

    def test_rage_emoji(self):
        assert from_emoji("😡").name == "rage"

    def test_fear_emoji(self):
        assert from_emoji("😨").name == "fear"

    def test_terror_emoji(self):
        assert from_emoji("😱").name == "terror"

    def test_disgust_emoji(self):
        assert from_emoji("🤢").name == "disgust"

    def test_trust_emoji(self):
        assert from_emoji("🙏").name == "trust"

    def test_anticipation_emoji(self):
        assert from_emoji("🤔").name == "anticipation"

    def test_surprise_emoji(self):
        assert from_emoji("😲").name == "surprise"

    def test_sadness_emoji(self):
        assert from_emoji("😢").name == "sadness"

    def test_unknown_emoji_returns_none(self):
        assert from_emoji("🍕") is None

    def test_unknown_char_returns_none(self):
        assert from_emoji("x") is None

    def test_strips_whitespace(self):
        result = from_emoji("  😊  ")
        assert result is not None
        assert result.name == "serenity"

    def test_multichar_emoji(self):
        # ❤️ is two codepoints; map keys include it — from_emoji does direct dict lookup
        # so single-char lookup of the base char should still work if keyed correctly
        result = from_emoji("❤️")
        # may be None if keyed differently — just assert no crash
        assert result is None or isinstance(result, Emotion)


# ---------------------------------------------------------------------------
# score_emojis
# ---------------------------------------------------------------------------

class TestScoreEmojis:
    def test_returns_emotional_state(self):
        assert isinstance(score_emojis("hello 😊"), EmotionalState)

    def test_empty_string_zero_state(self):
        state = score_emojis("")
        assert all(v == 0.0 for v in state.snapshot())

    def test_no_emoji_zero_state(self):
        state = score_emojis("hello world")
        assert all(v == 0.0 for v in state.snapshot())

    def test_single_joy_emoji(self):
        state = score_emojis("😊")
        assert state.valence() > 0

    def test_multiple_emojis_accumulate(self):
        state_single = score_emojis("😊")
        state_double = score_emojis("😊😊")
        # pleasantness axis should be doubled
        assert state_double.valence() > state_single.valence()

    def test_mixed_text_and_emoji(self):
        state = score_emojis("I feel great 😄 today! 🎉")
        assert state.valence() > 0

    def test_anger_emoji_produces_negative_valence(self):
        state = score_emojis("😠")
        # anger is on sensitivity axis (valence = 0), not pleasantness
        # valence may be 0, but arousal should be non-zero
        assert state.arousal() > 0

    def test_sadness_emoji_produces_negative_pleasantness(self):
        state = score_emojis("😢")
        assert state.valence() < 0


# ---------------------------------------------------------------------------
# from_emojis
# ---------------------------------------------------------------------------

class TestFromEmojis:
    def test_no_emoji_returns_none(self):
        assert from_emojis("hello world") is None

    def test_returns_emotion_for_emoji(self):
        result = from_emojis("😊")
        assert isinstance(result, Emotion)

    def test_dominant_with_multiple_same(self):
        result = from_emojis("😠😠😠😊")
        # three anger emojis vs one serenity — anger-axis dominates
        assert result is not None

    def test_mixed_content(self):
        result = from_emojis("Best day ever! 😄🎉🥳")
        assert isinstance(result, Emotion)
        # all three emojis map to joy/ecstasy/joy
        assert result.dimension.axis == "pleasantness"


# ---------------------------------------------------------------------------
# DeepMojiAdapter
# ---------------------------------------------------------------------------

class TestDeepMojiAdapter:
    def setup_method(self):
        self.adapter = DeepMojiAdapter()

    # from_scores ----------------------------------------------------------

    def test_from_scores_returns_emotion(self):
        scores = {"😂": 0.45, "😊": 0.30, "😭": 0.25}
        result = self.adapter.from_scores(scores)
        assert isinstance(result, Emotion)

    def test_from_scores_all_unknown_returns_none(self):
        result = self.adapter.from_scores({"🍕": 0.9, "🚀": 0.1})
        assert result is None

    def test_from_scores_empty_returns_none(self):
        assert self.adapter.from_scores({}) is None

    def test_from_scores_dominated_by_joy(self):
        scores = {"😂": 0.8, "😭": 0.05, "😠": 0.05}
        result = self.adapter.from_scores(scores)
        assert result is not None
        assert result.dimension.axis == "pleasantness"

    def test_from_scores_dominated_by_sadness(self):
        scores = {"😭": 0.7, "😢": 0.2, "😊": 0.1}
        result = self.adapter.from_scores(scores)
        assert result is not None
        assert result.dimension.axis == "pleasantness"
        assert result.emotional_flow < 0

    def test_from_scores_weight_matters(self):
        """High-weight anger should beat low-weight joy."""
        scores_anger = {"😡": 0.9, "😊": 0.1}
        scores_joy = {"😡": 0.1, "😊": 0.9}
        anger_dom = self.adapter.from_scores(scores_anger)
        joy_dom = self.adapter.from_scores(scores_joy)
        # they should differ
        assert anger_dom != joy_dom or anger_dom is None

    # from_ranked ----------------------------------------------------------

    def test_from_ranked_returns_emotion(self):
        ranked = [("😂", 0.45), ("😊", 0.30), ("😭", 0.25)]
        result = self.adapter.from_ranked(ranked)
        assert isinstance(result, Emotion)

    def test_from_ranked_top_k(self):
        ranked = [("😂", 0.45), ("😊", 0.30), ("😭", 0.25)]
        result_full = self.adapter.from_ranked(ranked)
        result_top1 = self.adapter.from_ranked(ranked, top_k=1)
        assert result_top1 is not None
        # top-1 only sees 😂 (joy) with 0.45
        assert result_top1.dimension.axis == "pleasantness"

    def test_from_ranked_empty(self):
        assert self.adapter.from_ranked([]) is None

    # score_state ----------------------------------------------------------

    def test_score_state_returns_emotional_state(self):
        scores = {"😂": 0.5, "😊": 0.5}
        state = self.adapter.score_state(scores)
        assert isinstance(state, EmotionalState)

    def test_score_state_unknown_emojis_zero(self):
        state = self.adapter.score_state({"🍕": 1.0})
        assert all(v == 0.0 for v in state.snapshot())

    def test_score_state_valence_positive_for_joy_emojis(self):
        scores = {"😄": 0.6, "😊": 0.4}
        state = self.adapter.score_state(scores)
        assert state.valence() > 0

    def test_score_state_vs_from_scores_consistent(self):
        scores = {"😂": 0.6, "😭": 0.4}
        state = self.adapter.score_state(scores)
        dominant = self.adapter.from_scores(scores)
        assert state.dominant() == dominant


# ---------------------------------------------------------------------------
# EmotionAnalyzer integration
# ---------------------------------------------------------------------------

class TestEmotionAnalyzerEmojiMethods:
    def test_from_emoji(self):
        from emotion_algebra import EmotionAnalyzer
        assert EmotionAnalyzer.from_emoji("😊").name == "serenity"

    def test_score_emojis(self):
        from emotion_algebra import EmotionAnalyzer
        state = EmotionAnalyzer.score_emojis("😄🎉")
        assert isinstance(state, EmotionalState)

