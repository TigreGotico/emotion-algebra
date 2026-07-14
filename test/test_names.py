"""Names move between languages. Coordinates do not.

That sentence is the entire contract of :mod:`emotion_algebra.names`, and most of
these tests exist to pin it down, because the tempting bug is a subtle one: it
would be very easy to let a translated label quietly imply a translated *finding*.

`raiva` returns the same point as `anger` because this library **assumes** the
point is the same, not because it has checked. It cannot check: there is no
human-rated Arabic affective lexicon in existence. The assumption is graded
`CONTESTED` in `evidence.py`, and these tests make sure the code says exactly what
the grade says and nothing more.
"""
import pytest

from emotion_algebra import dominant, label, prototype
from emotion_algebra.lang import UnsupportedLanguageError
from emotion_algebra.names import NAMES, canonical, localized, names_in
from emotion_algebra.prototypes import PROTOTYPES


class TestNamesMoveButCoordinatesDoNot:
    def test_three_names_one_point(self):
        assert prototype("anger") == prototype("raiva") == prototype("غضب")

    def test_the_geometry_is_identical_in_every_language(self):
        # The probabilities must not shift by a hair. If they did, the language
        # would be changing the model rather than the label — which is the thing
        # this module exists NOT to do.
        state = prototype("anger")
        assert (
            list(label(state).values())
            == list(label(state, lang="pt").values())
            == list(label(state, lang="ar").values())
        )

    def test_the_label_is_translated(self):
        state = prototype("anger")
        assert dominant(state, lang="en") == "anger"
        assert dominant(state, lang="pt") == "raiva"
        assert dominant(state, lang="ar") == "غضب"

    def test_the_flagship_blend_survives_translation(self):
        # rage + terror -> distress, not neutrality. The blend that Plutchik's
        # geometry gets wrong, in three scripts.
        mixed = prototype("fúria").blend(prototype("رعب"), 0.5)
        assert dominant(mixed, lang="en") == "distress"
        assert dominant(mixed, lang="pt") == "angústia"
        assert dominant(mixed, lang="ar") == "كرب"


class TestCoverage:
    @pytest.mark.parametrize("lang", ["pt", "ar"])
    def test_every_prototype_has_a_name(self, lang):
        missing = set(PROTOTYPES) - set(NAMES[lang])
        assert not missing, f"{lang} is missing: {sorted(missing)}"

    @pytest.mark.parametrize("lang", ["pt", "ar"])
    def test_no_two_emotions_share_a_name(self, lang):
        # A collision would silently merge two prototypes into one label, and the
        # readout would become unable to say which of them it meant.
        names = list(NAMES[lang].values())
        assert len(names) == len(set(names)), f"{lang} has a duplicate label"

    @pytest.mark.parametrize("lang", ["en", "pt", "ar"])
    def test_names_in_round_trips(self, lang):
        for localised, name in names_in(lang).items():
            assert canonical(localised) == name

    @pytest.mark.parametrize("lang", ["pt", "ar"])
    def test_the_intensity_gradient_survives(self, lang):
        # annoyance < anger < rage must still escalate after translation, or the
        # readout returns the right emotion at the wrong strength.
        mild, mid, strong = (
            prototype(NAMES[lang][n]) for n in ("annoyance", "anger", "rage")
        )
        assert mild.arousal < mid.arousal < strong.arousal


class TestNormalisation:
    def test_arabic_diacritics_are_folded(self):
        # Nobody types harakat. The same word with and without them is the same
        # word, and a lookup that disagrees is a lookup that fails on real input.
        assert canonical("غَضَب") == "anger"

    def test_kashida_is_folded(self):
        assert canonical("غضـــب") == "anger"

    def test_case_and_space_are_folded(self):
        assert canonical("  RAIVA ") == "anger"

    def test_an_alias_resolves(self):
        assert canonical("luto") == "grief"
        assert canonical("قلق") == "apprehension"

    def test_an_unknown_name_is_none_not_a_guess(self):
        assert canonical("blorp") is None

    def test_prototype_raises_on_an_unknown_name(self):
        with pytest.raises(KeyError):
            prototype("blorp")


class TestLocalized:
    def test_it_falls_back_to_canonical_rather_than_raising(self):
        # A missing translation should degrade to English, not explode.
        assert localized("anger", lang="en") == "anger"

    def test_an_unsupported_language_raises(self):
        with pytest.raises(UnsupportedLanguageError):
            localized("anger", lang="de")

    def test_readout_refuses_an_unsupported_language(self):
        with pytest.raises(UnsupportedLanguageError):
            dominant(prototype("anger"), lang="de")
