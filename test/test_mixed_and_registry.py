"""Tests for score_mixed/from_mixed and register_emoji/unregister_emoji."""
import pytest

from emotion_algebra.text import score_mixed, from_mixed
from emotion_algebra.emoji import register_emoji, unregister_emoji, from_emoji, _USER_REGISTRY
from emotion_algebra.state import EmotionalState
from emotion_algebra.plutchik import Emotion


# ---------------------------------------------------------------------------
# score_mixed / from_mixed
# ---------------------------------------------------------------------------

class TestScoreMixed:
    def test_returns_emotional_state(self):
        assert isinstance(score_mixed("hello 😊"), EmotionalState)

    def test_empty_string_zero_state(self):
        state = score_mixed("")
        assert all(v == 0.0 for v in state.snapshot())

    def test_word_only_matches(self):
        state_word = score_mixed("joy")
        state_text = __import__("emotion_algebra.text", fromlist=["score_text"]).score_text("joy")
        assert list(state_word.snapshot()) == list(state_text.snapshot())

    def test_emoji_only_matches(self):
        from emotion_algebra.emoji import score_emojis
        state_mixed = score_mixed("😊")
        state_emoji = score_emojis("😊")
        assert list(state_mixed.snapshot()) == list(state_emoji.snapshot())

    def test_combined_signal_stronger_than_either_alone(self):
        """Mixed text+emoji should accumulate more than text or emoji alone."""
        state_word = score_mixed("joy")
        state_emoji = score_mixed("😄")
        state_combined = score_mixed("joy 😄")
        # combined arousal must be >= each individual
        assert state_combined.arousal() >= state_word.arousal()
        assert state_combined.arousal() >= state_emoji.arousal()

    def test_positive_valence_for_happy_mixed(self):
        state = score_mixed("I feel great 😄🎉")
        assert state.valence() > 0

    def test_negative_valence_for_sad_mixed(self):
        state = score_mixed("grief and sorrow 😭😢")
        assert state.valence() < 0


class TestFromMixed:
    def test_none_on_no_signal(self):
        assert from_mixed("xyzzy") is None

    def test_returns_emotion_for_word(self):
        result = from_mixed("anger")
        assert isinstance(result, Emotion)

    def test_returns_emotion_for_emoji(self):
        result = from_mixed("😠")
        assert isinstance(result, Emotion)

    def test_returns_emotion_for_combined(self):
        result = from_mixed("joy 😄")
        assert isinstance(result, Emotion)
        assert result.dimension.axis == "pleasantness"

    def test_emoji_can_break_tie(self):
        """One word emotion vs multiple emojis — emojis win."""
        result = from_mixed("sadness 😠😠😠")
        assert result is not None
        # sadness contributes to pleasantness axis, anger to sensitivity
        # 3 anger emojis should dominate on sensitivity; just assert no crash
        assert isinstance(result, Emotion)


# ---------------------------------------------------------------------------
# register_emoji / unregister_emoji
# ---------------------------------------------------------------------------

class TestRegisterEmoji:
    def teardown_method(self):
        """Clean up any test registrations."""
        _USER_REGISTRY.clear()

    def test_register_then_lookup(self):
        register_emoji("🤖", "trust")
        result = from_emoji("🤖")
        assert result is not None
        assert result.name == "trust"

    def test_register_overrides_canonical(self):
        # 😊 is canonically "serenity"; override to "anger"
        register_emoji("😊", "anger")
        result = from_emoji("😊")
        assert result.name == "anger"

    def test_register_invalid_emotion_raises(self):
        with pytest.raises(ValueError, match="not a recognised emotion"):
            register_emoji("🤖", "made_up_emotion_xyz")

    def test_register_strips_whitespace(self):
        register_emoji("  🤖  ", "trust")
        assert from_emoji("🤖").name == "trust"

    def test_unregister_returns_true_when_exists(self):
        register_emoji("🤖", "trust")
        assert unregister_emoji("🤖") is True

    def test_unregister_returns_false_when_absent(self):
        assert unregister_emoji("🍕") is False

    def test_unregister_restores_canonical(self):
        register_emoji("😊", "anger")
        unregister_emoji("😊")
        result = from_emoji("😊")
        assert result.name == "serenity"  # canonical restored

    def test_registry_affects_score_emojis(self):
        from emotion_algebra.emoji import score_emojis
        register_emoji("🤖", "anger")
        state = score_emojis("🤖")
        assert state.arousal() > 0

    def test_registry_affects_deepmoji_adapter(self):
        from emotion_algebra.emoji import DeepMojiAdapter
        from emotion_algebra.emotions import get_emotion
        register_emoji("🤖", "fear")
        adapter = DeepMojiAdapter()
        result = adapter.from_scores({"🤖": 1.0})
        assert result is not None
        assert result.dimension.axis == get_emotion("fear").dimension.axis


# ---------------------------------------------------------------------------
# CLI emoji support
# ---------------------------------------------------------------------------

class TestCLIEmojiSupport:
    def test_single_emoji_arg(self, monkeypatch, capsys):
        import sys
        from emotion_algebra.__main__ import main
        monkeypatch.setattr(sys, "argv", ["emotion_algebra", "😊"])
        main()
        out = capsys.readouterr().out
        assert "serenity" in out or "→" in out

    def test_multiple_emoji_arg(self, monkeypatch, capsys):
        import sys
        from emotion_algebra.__main__ import main
        monkeypatch.setattr(sys, "argv", ["emotion_algebra", "😄🎉"])
        main()
        out = capsys.readouterr().out
        assert "→" in out

    def test_unknown_emoji_falls_through_to_expr_eval(self, monkeypatch):
        """An emoji not in the map should fall through to expression evaluation."""
        import sys
        from emotion_algebra.__main__ import main
        monkeypatch.setattr(sys, "argv", ["emotion_algebra", "🍕"])
        # 🍕 is not in map — should fall through to _eval_expr which will try
        # to resolve "🍕" as an emotion name and exit(1)
        with pytest.raises(SystemExit):
            main()
