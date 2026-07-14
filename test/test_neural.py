"""Tests for the DeepMoji text->affect probe.

These run **offline**. ``test/deepmoji_fixtures.json`` holds real encoder outputs
for the sentences below, captured once, so the probe can be exercised without a
90 MB model download on every run.

That split is also the honest one: the probe, and the projection into the core,
are what this library owns. The encoder is a third-party model — testing it here
would be testing ``deepmoji-onnx``, not us.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from emotion_algebra.affect import AffectState
from emotion_algebra.neural import PROBE_AXES, _probe, affect_from_features

FIXTURES = json.loads((Path(__file__).parent / "deepmoji_fixtures.json").read_text())
SENTENCES = FIXTURES["sentences"]
FEATURES = np.array(FIXTURES["features"], dtype=float)


def state_for(sentence: str) -> AffectState:
    """The core state for one fixture sentence."""
    i = SENTENCES.index(sentence)
    return affect_from_features(FEATURES[i : i + 1])[0]


class TestProbe:
    def test_probe_ships_with_the_package(self):
        w = _probe()
        assert w.shape == (73, 5)          # 64 emoji + 8 typographic + intercept
        assert np.isfinite(w).all()

    def test_typography_feeds_arousal_and_nothing_else(self):
        # Emoji carry valence; typography carries activation. The 8 typographic
        # rows are wired to arousal ONLY -- letting them touch the other axes
        # measurably degraded valence.
        w = _probe()
        typo_rows = w[64:72]
        assert np.abs(typo_rows[:, [0, 1, 2, 4]]).max() == 0.0   # not valence, potency...
        assert np.abs(typo_rows[:, 3]).max() > 0.0               # ...only arousal

    def test_axis_order_matches_the_core(self):
        assert PROBE_AXES == (
            "positivity", "negativity", "potency", "arousal", "unpredictability"
        )

    def test_fixtures_are_real_encoder_output(self):
        assert FEATURES.shape == (len(SENTENCES), 72)   # 64 emoji + 8 typographic
        assert (FEATURES >= 0).all()


class TestAffectFromFeatures:
    def test_returns_one_state_per_row(self):
        states = affect_from_features(FEATURES)
        assert len(states) == len(SENTENCES)
        assert all(isinstance(s, AffectState) for s in states)

    def test_output_stays_inside_the_space(self):
        for s in affect_from_features(FEATURES):
            assert 0.0 <= s.positivity <= 1.0
            assert 0.0 <= s.negativity <= 1.0
            assert -1.0 <= s.potency <= 1.0
            assert 0.0 <= s.arousal <= 1.0
            assert 0.0 <= s.unpredictability <= 1.0

    @pytest.mark.parametrize(
        "bad", [np.zeros((3, 5)), np.zeros(64), np.zeros((2, 2, 64))]
    )
    def test_wrong_shape_is_rejected(self, bad):
        with pytest.raises(ValueError):
            affect_from_features(bad)

    def test_raw_emoji_features_need_the_texts_too(self):
        # (n, 64) alone cannot produce arousal -- the typographic cues carry it.
        with pytest.raises(ValueError):
            affect_from_features(np.zeros((2, 64)))


class TestArousal:
    """Emphasis changes how LOUD something is, not whether it is good or bad.

    DeepMoji reads "!" as excitement, and excitement looks positive to it -- so
    left alone, "this is wonderful!!!" and "this is unacceptable!!!" BOTH drift
    toward neutral valence. The encoder is therefore shown the text stripped of
    emphasis, while the typographic cues see it intact.
    """

    def test_emphasis_raises_arousal(self):
        calm = state_for("this is wonderful")
        loud = state_for("this is wonderful!!!")
        assert loud.arousal > calm.arousal + 0.2

    def test_emphasis_does_not_move_valence(self):
        calm = state_for("this is wonderful")
        loud = state_for("this is wonderful!!!")
        assert loud.valence == pytest.approx(calm.valence, abs=0.05)

    def test_shouting_a_complaint_stays_a_complaint(self):
        # Capitals and !!! must not launder an insult into neutrality.
        quiet = state_for("this is completely unacceptable")
        shouted = state_for("THIS IS COMPLETELY UNACCEPTABLE!!!")
        assert shouted.valence < -0.3
        assert shouted.valence == pytest.approx(quiet.valence, abs=0.05)
        assert shouted.arousal > quiet.arousal + 0.2


class TestItRecoversPotencyFromRealText:
    """The reason this module exists.

    Anger and fear are both negative and both aroused. What separates them is
    potency — and that is the difference between a user who escalates and one who
    quietly leaves.

    The bag-of-words lexicon cannot recover it: on "I'm scared I've broken
    something" it fires on ``broken -> anger`` and reports an angry,
    approach-motivated user, which is the opposite of the truth.
    """

    ANGRY = "This is the third time your app has lost my work. Fix it."
    AFRAID = "I don't know if I'm doing this right and I'm scared I've broken something."

    @pytest.fixture(scope="class")
    def states(self):
        return state_for(self.ANGRY), state_for(self.AFRAID)

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
        "sentence,expect_potency_positive",
        [
            ("I am furious about this", True),
            ("I'm terrified something has gone wrong", False),
        ],
    )
    def test_explicit_emotion_words(self, sentence, expect_potency_positive):
        s = state_for(sentence)
        assert s.valence < 0
        assert (s.potency > 0) is expect_potency_positive

    def test_positive_text_reads_positive(self):
        assert state_for("this is wonderful, thank you so much").valence > 0
