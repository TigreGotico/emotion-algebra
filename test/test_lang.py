"""The language boundary.

Two things are being tested here, and only one of them is ordinary.

The ordinary one: each language's typographic cues are read with that language's
orthography, so an Arabic question mark counts as a question and a Portuguese
intensifier counts as an intensifier.

The other one is the point of the module. **Reading a language with the wrong
profile does not raise, and does not return zero-across-the-board.** It returns a
number that looks entirely reasonable. Several tests below assert the size of
that error directly, because a silent wrong answer is the failure this code
exists to prevent, and a test that merely checks the right answer would pass just
as happily if the wrong answer were still reachable.
"""
import re

import pytest

from emotion_algebra import lang
from emotion_algebra.lang import (
    CHANNELS,
    KASHIDA,
    LanguageProfile,
    UnsupportedLanguageError,
    detect_language,
    profile,
    resolve,
    strip_emphasis,
    typographic_features,
)


def channel(features, name):
    return features[CHANNELS.index(name)]


class TestTheDeadChannels:
    """Each of these is a cue that reads 0.0 on Arabic under the English profile.

    Not "reads low" — reads exactly zero, on every input, forever.
    """

    def test_arabic_questions_are_invisible_to_the_english_profile(self):
        # "Why? What happened?" — two questions, both with U+061F.
        text = "لماذا؟ ماذا حدث؟"

        ar, _ = typographic_features(text, "ar")
        en, _ = typographic_features(text, "en")

        assert channel(ar, "questioning") > 0, "the Arabic profile must see them"
        assert channel(en, "questioning") == 0.0, (
            "this is the bug: the English profile counts '?', and Arabic does "
            "not use '?'. Every Arabic question ever written scores zero."
        )

    def test_arabic_cannot_shout_in_capitals(self):
        # There is no such thing as a capital Arabic letter, so a case ratio is
        # not low here — it is undefined. Arabic shouts by stretching instead.
        loud = "مرحبـــــــا"
        ar, available = typographic_features(loud, "ar")
        en, _ = typographic_features(loud, "en")

        assert channel(ar, "shouting") > 0, "kashida IS the Arabic shout"
        assert channel(en, "shouting") == 0.0
        assert available[CHANNELS.index("shouting")] == 1.0, (
            "Arabic realises the channel, just not with case"
        )

    def test_the_emphasis_guard_silently_fails_on_arabic(self):
        # This is the nastiest one. Stripping emphasis before the encoder is what
        # stops "!" being read as excitement and neutralising the valence of an
        # angry sentence. The guard is an ASCII regex.
        text = "هذا غير مقبول؟؟؟"

        assert "؟" not in strip_emphasis(text, "ar")
        assert "؟" in strip_emphasis(text, "en"), (
            "the ASCII [!?]+ leaves U+061F exactly where it was, so on Arabic "
            "the guard does nothing at all and the bug it was written to fix "
            "comes straight back — without raising"
        )

    def test_portuguese_intensifiers_are_invisible_to_the_english_profile(self):
        text = "estou muito irritado"
        pt, _ = typographic_features(text, "pt")
        en, _ = typographic_features(text, "en")

        assert channel(pt, "intensification") > 0
        assert channel(en, "intensification") == 0.0

    def test_english_elongation_regex_misses_portuguese(self):
        # "nãããão" — the ASCII [a-z] class does not contain "ã".
        ascii_only = re.compile(r"([a-z])\1{2,}")
        assert not ascii_only.findall("nãããão")

        pt, _ = typographic_features("nãããão", "pt")
        assert channel(pt, "elongation") > 0


class TestRefusal:
    def test_an_unregistered_language_raises(self):
        with pytest.raises(UnsupportedLanguageError):
            profile("de")

    def test_the_error_says_what_it_can_read(self):
        with pytest.raises(UnsupportedLanguageError) as e:
            profile("de")
        assert "en" in str(e.value)
        assert e.value.lang == "de"

    def test_an_undetectable_string_is_refused_rather_than_guessed(self):
        # A string with no evidence in it is not English by default. Guessing
        # would be the whole mistake, in miniature.
        with pytest.raises(UnsupportedLanguageError):
            resolve("...", lang=None)

    def test_an_explicit_lang_always_wins_over_detection(self):
        # Detection is a convenience. It is never evidence.
        assert resolve("I am fine", lang="pt").code == "pt"


class TestDetection:
    @pytest.mark.parametrize("text,expected", [
        ("لا أعرف إن كنت أفعل هذا بشكل صحيح", "ar"),
        ("Não sei se estou a fazer isto bem", "pt"),
        ("I don't know if I'm doing this right", "en"),
    ])
    def test_the_three_supported_languages(self, text, expected):
        assert detect_language(text) == expected

    def test_script_beats_everything(self):
        # Arabic characters live in their own Unicode blocks and nothing else
        # does, so script identification is the one part of this that is solid.
        assert detect_language("ok غاضب") == "ar"

    def test_it_abstains_rather_than_guessing(self):
        # Telling Portuguese from English on a short Latin string is genuinely
        # hard, and this function does not pretend to have solved it.
        assert detect_language("hmm") is None
        assert detect_language("") is None


class TestAvailabilityMask:
    def test_a_language_with_no_shouting_reports_it_as_missing(self):
        # The distinction that matters: a channel a language cannot express must
        # be reported as ABSENT, not as a measured zero. One is missing data;
        # the other is evidence of calm, and they are not the same thing.
        mute = LanguageProfile(
            code="zxx",
            script="latin",
            emphasis=re.compile(r"[!?]+"),
            question="?",
            elongation=re.compile(r"([a-z])\1{2,}"),
            intensifiers=("very",),
            has_case=False,
            shouting=None,
        )
        lang.PROFILES["zxx"] = mute
        try:
            features, available = typographic_features("HELLO", "zxx")
            i = CHANNELS.index("shouting")
            assert features[i] == 0.0
            assert available[i] == 0.0, (
                "without this, the probe cannot tell 'this writer is not "
                "shouting' from 'this language cannot shout'"
            )
        finally:
            del lang.PROFILES["zxx"]

    def test_the_supported_languages_realise_every_channel(self):
        for code in ("en", "pt", "ar"):
            _, available = typographic_features("ok", code)
            assert all(a == 1.0 for a in available), code

    def test_features_and_mask_are_the_same_width(self):
        for code in ("en", "pt", "ar"):
            features, available = typographic_features("ok", code)
            assert len(features) == len(available) == len(CHANNELS)


class TestProfileIntegrity:
    def test_a_caseless_language_cannot_shout_in_capitals(self):
        with pytest.raises(ValueError, match="no case"):
            lang.register_profile(LanguageProfile(
                code="xx",
                script="arabic",
                emphasis=re.compile("[!]+"),
                question="؟",
                elongation=re.compile(r"([ء-ي])\1{2,}"),
                intensifiers=("جدا",),
                has_case=False,
                shouting="case",
            ))

    def test_a_language_cannot_be_registered_twice(self):
        with pytest.raises(ValueError):
            lang.register_profile(lang.PROFILES["en"])

    def test_intensifiers_match_words_not_substrings(self):
        # "very" is inside "every", and counting it there would be nonsense.
        en, _ = typographic_features("every single one", "en")
        assert channel(en, "intensification") == 0.0

        en, _ = typographic_features("very good", "en")
        assert channel(en, "intensification") > 0

    def test_arabic_kashida_is_the_documented_codepoint(self):
        assert KASHIDA == "ـ"
