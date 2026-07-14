"""The multilingual probe.

Runs **offline**. ``test/multilingual_fixtures.json`` holds real encoder outputs
for the sentences below, captured once, so the probe can be exercised without
downloading a 470 MB model on every run.

That split is also the honest one: the probe and the projection into the core are
what this library owns. The encoder is a third-party model, and testing it here
would be testing ``paraphrase-multilingual-MiniLM-L12-v2``, not us.

The load-bearing test is :class:`TestTheCrux`. Everything else in the library
follows from anger and fear separating on **potency** — the angry customer
escalates, the frightened one leaves without a word — and a language where that
fails is a language this library cannot read, however confident its numbers look.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from emotion_algebra.affect import AffectState
from emotion_algebra.lang import UnsupportedLanguageError
from emotion_algebra.multilingual import (
    N_EMBED,
    PROBE_PATH,
    _probe,
    affect_from_features,
    design_matrix,
)
from emotion_algebra.neural import PROBE_AXES

FIXTURES = json.loads(
    (Path(__file__).parent / "multilingual_fixtures.json").read_text()
)
CASES = FIXTURES["cases"]


def case(lang: str, index: int = 0):
    matches = [c for c in CASES if c["lang"] == lang]
    return matches[index]


def state_of(c) -> AffectState:
    embedding = np.array([c["embedding"]], dtype=float)
    return affect_from_features(embedding, [c["text"]], lang=c["lang"])[0]


class TestTheCrux:
    """Anger and fear, in each language. The reason the library exists.

    Both are unpleasant and both are highly aroused, so valence and arousal
    cannot tell them apart. Potency can — and if it cannot, in some language,
    then this library does not read that language and must say so.
    """

    @pytest.mark.parametrize("lang", ["en", "pt", "ar"])
    def test_the_complaint_and_the_goodbye_do_not_collapse_together(self, lang):
        angry = state_of(case(lang, 0))     # "your app lost my work. Fix it."
        afraid = state_of(case(lang, 1))    # "I'm scared I've broken something."

        assert angry.potency > afraid.potency, (
            f"in {lang}, the person who will escalate and the person who will "
            f"quietly leave are indistinguishable. That is the one thing this "
            f"library must never do."
        )

    @pytest.mark.parametrize("lang", ["en", "pt", "ar"])
    def test_both_are_unpleasant_which_is_why_valence_cannot_do_this(self, lang):
        # The whole argument in one assertion: valence agrees, potency does not.
        angry = state_of(case(lang, 0))
        afraid = state_of(case(lang, 1))

        assert angry.valence < 0.25 and afraid.valence < 0.25
        assert abs(angry.potency - afraid.potency) > abs(
            angry.valence - afraid.valence
        ), f"in {lang}, potency must carry more of the distinction than valence"


class TestProbe:
    def test_the_probe_has_the_shape_the_module_claims(self):
        probe = _probe()
        assert probe.shape == (N_EMBED + 8 + 1, len(PROBE_AXES))

    def test_typography_touches_arousal_and_nothing_else(self):
        # Embeddings carry valence; typography carries activation. Letting the
        # typographic rows reach valence is how "this is unacceptable!!!" ends
        # up scored as mildly positive.
        probe = _probe()
        typo = probe[N_EMBED:N_EMBED + 8]
        for i, axis in enumerate(PROBE_AXES):
            if axis == "arousal":
                assert np.abs(typo[:, i]).sum() > 0, "arousal must use them"
            else:
                assert np.abs(typo[:, i]).sum() == 0, f"{axis} must not"

    def test_it_records_that_it_was_fitted_on_english_only(self):
        # The provenance is not a comment: it ships inside the artefact.
        meta = json.loads(PROBE_PATH.read_text())
        assert "ENGLISH ONLY" in meta["note"]
        assert "ZERO-SHOT" in meta["note"]

    def test_states_are_in_range(self):
        for c in CASES:
            s = state_of(c)
            assert 0.0 <= s.positivity <= 1.0
            assert 0.0 <= s.negativity <= 1.0
            assert -1.0 <= s.potency <= 1.0
            assert 0.0 <= s.arousal <= 1.0
            assert 0.0 <= s.unpredictability <= 1.0


class TestTypographyCrossesTheScript:
    """The design matrix must read each language's own orthography."""

    def test_arabic_emphasis_raises_arousal(self):
        # "This is unacceptable!!! Why???" — with the ARABIC question mark. Under
        # an English profile the questioning cue would read 0.0 here.
        loud = case("ar", 2)
        calm = case("ar", 1)
        assert state_of(loud).arousal > state_of(calm).arousal

    def test_the_typographic_block_differs_by_language(self):
        c = case("ar", 2)
        embedding = np.array([c["embedding"]], dtype=float)

        as_arabic = design_matrix(embedding, [c["text"]], lang="ar")
        as_english = design_matrix(embedding, [c["text"]], lang="en")

        typo_ar = as_arabic[0, N_EMBED:N_EMBED + 8]
        typo_en = as_english[0, N_EMBED:N_EMBED + 8]
        assert not np.allclose(typo_ar, typo_en), (
            "reading Arabic with English typography must produce DIFFERENT "
            "features — if it did not, the bug this module fixes would be "
            "invisible"
        )

    def test_an_unregistered_language_raises(self):
        c = case("en", 0)
        with pytest.raises(UnsupportedLanguageError):
            affect_from_features(
                np.array([c["embedding"]]), [c["text"]], lang="de"
            )


class TestShapes:
    def test_wrong_embedding_width_raises(self):
        with pytest.raises(ValueError, match="384"):
            design_matrix(np.zeros((1, 12)), ["hi"], lang="en")

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            design_matrix(np.zeros((2, N_EMBED)), ["only one"], lang="en")
