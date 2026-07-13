"""Tests for the DeepMoji text->affect probe."""
import numpy as np
import pytest

from emotion_algebra.affect import AffectState
from emotion_algebra.homeostasis import SET_POINT
from emotion_algebra.neural import (
    PROBE_AXES,
    affect_from_text,
    affect_from_texts,
    _probe,
)


class TestProbe:
    def test_probe_ships_with_the_package(self):
        w = _probe()
        assert w.shape == (65, 5)          # 64 emoji features + intercept
        assert np.isfinite(w).all()

    def test_axis_order_matches_the_core(self):
        assert PROBE_AXES == (
            "positivity", "negativity", "potency", "arousal", "unpredictability"
        )


class TestAffectFromText:
    def test_returns_an_affect_state(self):
        assert isinstance(affect_from_text("hello"), AffectState)

    def test_blank_input_returns_the_set_point(self):
        assert affect_from_text("   ") == SET_POINT

    def test_batching_agrees_with_single(self):
        texts = ["I am furious", "I am terrified"]
        batched = affect_from_texts(texts)
        for t, b in zip(texts, batched):
            assert b.as_array == pytest.approx(affect_from_text(t).as_array)

    def test_rejects_empty_and_non_strings(self):
        with pytest.raises(ValueError):
            affect_from_texts([])
        with pytest.raises(ValueError):
            affect_from_texts([None])

    def test_output_stays_inside_the_space(self):
        for text in ("I am furious", "", "wonderful!!!", "meh"):
            s = affect_from_text(text or " ")
            assert 0.0 <= s.positivity <= 1.0
            assert -1.0 <= s.potency <= 1.0


class TestItRecoversPotencyFromRealText:
    """The reason this module exists.

    Anger and fear are both negative and both aroused. What separates them is
    potency — and it is the difference between a user who escalates and one who
    quietly leaves. The bag-of-words lexicon cannot recover it: on "I'm scared
    I've broken something" it fires on `broken -> anger` and reports an angry,
    approach-motivated user, which is the opposite of the truth.
    """

    ANGRY = "This is the third time your app has lost my work. Fix it."
    AFRAID = "I don't know if I'm doing this right and I'm scared I've broken something."

    @pytest.fixture(scope="class")
    def states(self):
        angry, afraid = affect_from_texts([self.ANGRY, self.AFRAID])
        return angry, afraid

    def test_both_read_as_negative(self, states):
        angry, afraid = states
        assert angry.valence < 0
        assert afraid.valence < 0

    def test_valence_barely_distinguishes_them(self, states):
        # Which is exactly why a valence-only sentiment model treats them the
        # same, and gets one of the two responses wrong every time.
        angry, afraid = states
        assert abs(angry.valence - afraid.valence) < 0.15

    def test_potency_separates_them(self, states):
        angry, afraid = states
        assert angry.potency > 0 > afraid.potency

    @pytest.mark.parametrize(
        "text,expect_potency_positive",
        [
            ("I am furious about this", True),
            ("I'm terrified something has gone wrong", False),
        ],
    )
    def test_explicit_emotion_words(self, text, expect_potency_positive):
        s = affect_from_text(text)
        assert s.valence < 0
        assert (s.potency > 0) is expect_potency_positive
