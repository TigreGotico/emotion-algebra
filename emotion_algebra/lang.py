"""What a language's typography does, and what it cannot do.

The core of this library is language-neutral. :class:`~emotion_algebra.affect.AffectState`
holds five floats; appraisal, tendency, homeostasis and the neuromodulator readout
transform floats into floats. None of them has ever seen a word.

**The language boundary is exactly one arrow: text -> AffectState.** This module,
and the encoder in :mod:`emotion_algebra.neural`, are the whole of it.

Typography is orthography, not universals
-----------------------------------------
Emoji tell you how someone feels; punctuation tells you how *loudly*. That is a
real finding — the typographic cues predict arousal about twice as well as the
64-dimensional emoji distribution does — but the cues themselves are written in a
particular script, and a cue is only a cue where the script affords it.

Read a sentence of Arabic with English cues and four of the eight channels do not
merely weaken, they go **structurally dead**:

* Arabic asks questions with ``؟`` (U+061F), not ``?``. A rule that counts ``?``
  scores **zero on every Arabic question that has ever been written.**
* Arabic is **caseless**. There is no such thing as an Arabic capital letter, so
  a SHOUTING ratio is not *low* here — it is *undefined*, and it reports ``0.0``
  forever.
* ``([a-z])\\1{2,}`` matches no Arabic at all, and misses Portuguese ``nãããão``.
* An English intensifier list finds no Arabic and no Portuguese intensifiers.

A dead channel does not announce itself. It returns ``0.0``, which is a perfectly
plausible number, and the probe downstream multiplies it by a weight and carries
on. That is the failure mode this module exists to prevent: **not an error, but a
confident wrong answer.**

Channels, not characters
------------------------
So the eight features are defined as *functions* — emphasis, questioning, visual
shouting, elongation, length, trailing-off, intensification — and each
:class:`LanguageProfile` says how its language *realises* that function.

The most interesting case is shouting. Arabic cannot capitalise, but Arabic
writers do stretch words for emphasis, using the **kashida** (tatweel, U+0640):
``مرحبـــــا`` is the written equivalent of raising your voice. Same channel,
same feature index, different orthography.

Where a language realises a channel with nothing at all, the profile says so, and
:func:`typographic_features` returns an explicit **availability mask** alongside
the values. A missing channel is then a *known absence* rather than a silent zero
— something the evaluation can measure the cost of, instead of something the
probe quietly mistakes for evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

#: The eight functional channels, in feature order. These are *what the cue does*,
#: not what character it is written with.
CHANNELS = (
    "emphasis_density",      # how often the writer emphasises
    "emphasis_saturation",   # "!!!" — bounded, so one shouty message cannot dominate
    "questioning",           # "?" / "؟"
    "shouting",              # CAPS in a cased script; kashida in Arabic
    "elongation",            # "sooo", "nãããão", "مرحبـــــا"
    "length",
    "trailing_off",          # "..." — the one channel that means LOW arousal
    "intensification",       # "very", "muito", "???"
)

#: Arabic script, including the Supplement and Extended-A blocks.
_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")

#: The kashida (tatweel). Arabic's emphasis stretch.
KASHIDA = "ـ"

#: Trailing off. The one cue that means the writer is going *quiet* rather than
#: raising their voice — which is why it is wired to arousal with the opposite
#: sign to everything else here.
ELLIPSIS = re.compile(r"[.…]{2,}|…")

#: "sooo", "ahhh" — and, because the vowel class includes the accented range,
#: Portuguese "nãããão", which an ASCII-only ``[a-z]`` silently misses.
_LATIN_ELONGATION = re.compile(r"([a-zà-ÿ])\1{2,}", re.IGNORECASE)

#: Arabic stretches a word by repeating a letter, or by inserting kashida.
_ARABIC_ELONGATION = re.compile(rf"([ء-ي])\1{{2,}}|{KASHIDA}{{2,}}")


class UnsupportedLanguageError(ValueError):
    """The text is in a language this library has not been evaluated on.

    Raised rather than guessed, and that is the entire point.

    Running Arabic through the English pipeline does not fail — it returns a
    number. The number is confident, plausible, and wrong, and it will be
    consumed by a caller who has no way to know that. A wrong affect reading
    propagates into every decision made downstream from it, silently, forever.

    An exception is recoverable. A quietly wrong float is not.
    """

    def __init__(self, lang: str, supported: Optional[Iterable[str]] = None) -> None:
        known = ", ".join(sorted(supported if supported is not None else PROFILES))
        super().__init__(
            f"language {lang!r} is not supported here; this reads: {known}. "
            f"Reading it with another language's encoder and typography does not "
            f"fail — it returns a confident wrong answer — so it is refused "
            f"instead. Note that only the text layer is language-bound: the rest "
            f"of the library (appraisal, tendency, homeostasis, the "
            f"neuromodulator readout) is language-neutral and will work on any "
            f"AffectState you can supply from elsewhere."
        )
        self.lang = lang
        self.supported = tuple(sorted(supported if supported is not None else PROFILES))


@dataclass(frozen=True)
class LanguageProfile:
    """How one language realises the eight typographic channels.

    Attributes
    ----------
    code:
        BCP-47-ish language code. ``"en"``, ``"pt"``, ``"ar"``.
    script:
        ``"latin"`` or ``"arabic"``. Decides how text is detected, and whether
        the emphasis-stripping regex will match anything at all.
    emphasis:
        Marks that make a sentence *louder* without changing whether the thing
        described is good or bad. Stripped before the sentence reaches the
        encoder — see :mod:`emotion_algebra.neural` — and counted here, where
        they belong.
    question:
        The question mark. ``"?"`` in Latin script, ``"؟"`` (U+061F) in Arabic.
    elongation:
        Stretched letters. English ``sooo``; Portuguese, which must include its
        accented vowels, ``nãããão``; Arabic, which stretches with the kashida.
    intensifiers:
        Words that turn the dial up. Matched on word boundaries, because ``so``
        inside ``some`` is not an intensifier.
    has_case:
        ``False`` for Arabic. A caseless script cannot shout in capitals.
    shouting:
        How this language realises deliberate visual emphasis. ``"case"`` uses
        the ratio of capital letters. ``"kashida"`` uses the ratio of tatweel
        characters, which is what Arabic writers actually stretch words with.
        ``None`` means the language has **no** realisation of this channel — the
        feature is then reported as unavailable rather than as zero.
    """

    code: str
    script: str
    emphasis: re.Pattern
    question: str
    elongation: re.Pattern
    intensifiers: Tuple[str, ...]
    ellipsis: re.Pattern = ELLIPSIS
    has_case: bool = True
    shouting: Optional[str] = "case"
    _intensifier_re: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        pattern = "|".join(re.escape(w) for w in self.intensifiers)
        object.__setattr__(
            self, "_intensifier_re",
            re.compile(rf"(?<!\w)(?:{pattern})(?!\w)", re.IGNORECASE),
        )

    def count_intensifiers(self, text: str) -> int:
        return len(self._intensifier_re.findall(text))


#: Bounds on the counting channels, so that one twenty-exclamation-mark message
#: cannot saturate a feature and drown out everything else in the batch. These
#: are squashing constants, not measurements — any probe fitted downstream of
#: them absorbs their exact values into its weights.
SATURATION_CAPS = {"punctuation": 5, "length": 50, "kashida": 10}

PROFILES: Dict[str, LanguageProfile] = {}


def register_profile(profile: LanguageProfile) -> LanguageProfile:
    """Add a language. Raises if it is already registered."""
    if profile.code in PROFILES:
        raise ValueError(f"language already registered: {profile.code!r}")
    if profile.shouting not in (None, "case", "kashida"):
        raise ValueError(f"unknown shouting realisation: {profile.shouting!r}")
    if profile.shouting == "case" and not profile.has_case:
        raise ValueError(
            f"{profile.code!r} cannot shout in capitals: it has no case"
        )
    PROFILES[profile.code] = profile
    return profile


register_profile(LanguageProfile(
    code="en",
    script="latin",
    emphasis=re.compile(r"[!?]+"),
    question="?",
    elongation=_LATIN_ELONGATION,
    intensifiers=("very", "extremely", "so", "really", "totally"),
))

register_profile(LanguageProfile(
    code="pt",
    script="latin",
    emphasis=re.compile(r"[!?]+"),
    question="?",
    # Portuguese stretches accented vowels — "nãããão" — which an ASCII-only
    # [a-z] class silently misses.
    elongation=_LATIN_ELONGATION,
    intensifiers=(
        "muito", "muita", "muitos", "muitas", "super", "bastante", "imenso",
        "imensa", "extremamente", "mesmo", "tão", "demasiado", "completamente",
        "totalmente", "mega",
    ),
))

register_profile(LanguageProfile(
    code="ar",
    script="arabic",
    # U+061F ARABIC QUESTION MARK. An ASCII [!?]+ leaves it in place, which means
    # the emphasis-stripping guard in neural.py silently does nothing on Arabic —
    # and the valence-neutralisation bug it was written to fix comes straight back.
    emphasis=re.compile("[!؟‼]+"),
    question="؟",
    elongation=_ARABIC_ELONGATION,
    intensifiers=(
        "جدا",          # jiddan — very (MSA)
        "جدًا",
        "للغاية",  # lil-ghaaya — extremely
        "كثيرا",        # katheeran — greatly
        "كتير",              # kteer — a lot (Levantine)
        "أوي",                    # awi — very (Egyptian)
        "تماما",        # tamaaman — completely
        "مرة",                    # marra — very (Gulf)
        "وايد",              # waayid — a lot (Gulf)
    ),
    # There is no Arabic capital letter. Not "rarely used" — the category does
    # not exist. So the channel is realised the way Arabic writers actually
    # realise it: by stretching the word with kashida.
    has_case=False,
    shouting="kashida",
))


def detect_language(text: str) -> Optional[str]:
    """Guess the language of *text*, or ``None`` if it cannot be told.

    Deliberately small. Script identification is reliable — Arabic characters
    live in their own Unicode blocks and nothing else does — while telling
    Portuguese from English on Latin script is a genuinely hard problem that this
    function does **not** claim to have solved. It looks for Portuguese function
    words and diacritics, and abstains when it sees neither.

    ``None`` is the honest answer for a short or ambiguous string, and callers
    must treat it as one. **Pass ``lang=`` explicitly whenever you know it.**
    Detection is a convenience, not evidence.
    """
    text = str(text)
    if _ARABIC.search(text):
        return "ar"

    lowered = text.lower()
    if any(c in lowered for c in "ãõçêôáíú"):
        return "pt"
    words = set(re.findall(r"[a-zà-ÿ]+", lowered))
    pt_markers = {
        "que", "não", "de", "para", "com", "uma", "por", "está", "isto",
        "meu", "minha", "você", "muito", "mais", "como", "foi", "sou", "e",
    }
    en_markers = {
        "the", "is", "and", "to", "of", "it", "you", "my", "this", "that", "was",
        "for", "with", "not", "are", "have", "am",
    }
    pt_hits = len(words & pt_markers)
    en_hits = len(words & en_markers)
    if pt_hits > en_hits:
        return "pt"
    if en_hits > pt_hits:
        return "en"
    return None


def profile(lang: str) -> LanguageProfile:
    """Look up a :class:`LanguageProfile`.

    Raises
    ------
    UnsupportedLanguageError
        If the language is not registered. It is not silently approximated with
        the nearest one — see the exception's docstring for why.
    """
    try:
        return PROFILES[lang]
    except KeyError:
        raise UnsupportedLanguageError(lang) from None


def resolve(text: str, lang: Optional[str] = None) -> LanguageProfile:
    """The profile for *text*, using *lang* if given and detection if not.

    Raises
    ------
    UnsupportedLanguageError
        If *lang* is unregistered, or if it is ``None`` and the language cannot
        be detected. Abstaining is the point: a string we cannot identify is a
        string we cannot honestly read.
    """
    if lang is not None:
        return profile(lang)
    detected = detect_language(text)
    if detected is None:
        raise UnsupportedLanguageError("undetermined")
    return profile(detected)


def typographic_features(
    text: str, lang: str = "en"
) -> Tuple[List[float], List[float]]:
    """The eight arousal cues, read with the right language's eyes.

    Emoji tell you how someone *feels*; punctuation tells you how *loudly*.
    These are the "loudly" half, and on English they predict arousal about twice
    as well as the entire 64-dimensional emoji distribution does — which is why
    getting them right in a second language matters as much as the encoder does.

    Returns
    -------
    (features, available)
        *features* is the eight channel values in :data:`CHANNELS` order.
        *available* is ``1.0`` where the language realises that channel and
        ``0.0`` where it has no way to. An unavailable channel reads ``0.0`` in
        *features* too — but now the caller can **tell that zero apart from a
        measured absence of shouting**, which is the difference between missing
        data and evidence of calm.
    """
    p = profile(lang)
    text = str(text)
    words = text.split()
    n = max(len(words), 1)

    exclamations = text.count("!")
    questions = text.count(p.question)

    available = [1.0] * len(CHANNELS)
    shout = 0.0
    if p.shouting == "case":
        letters = [c for c in text if c.isalpha()]
        shout = sum(c.isupper() for c in letters) / max(len(letters), 1)
    elif p.shouting == "kashida":
        # Arabic's stretch. There is no capital letter to count, so count the
        # tatweel: a word held out is a word shouted.
        shout = min(text.count(KASHIDA), SATURATION_CAPS["kashida"]) / SATURATION_CAPS["kashida"]
    else:
        available[CHANNELS.index("shouting")] = 0.0

    punct = SATURATION_CAPS["punctuation"]
    features = [
        exclamations / n,
        min(exclamations, punct) / punct,
        min(questions, punct) / punct,
        shout,
        len(p.elongation.findall(text)) / n,
        min(len(words), SATURATION_CAPS["length"]) / SATURATION_CAPS["length"],
        len(p.ellipsis.findall(text)) / n,
        p.count_intensifiers(text) / n,
    ]
    return features, available


def strip_emphasis(text: str, lang: str = "en") -> str:
    """Remove emphasis marks, so the encoder judges *what* is said, not how loudly.

    DeepMoji reads ``!`` as excitement, and excitement looks positive to it. Left
    in place, both "this is wonderful!!!" and "this is unacceptable!!!" drift
    toward **neutral valence** — nonsense in one direction and dangerously wrong
    in the other.

    So the encoder is shown the sentence with emphasis stripped, and the
    typographic cues see it intact. Emphasis changes how *loud* something is; it
    does not change whether it is good or bad.

    This only works if the regex matches the language's actual punctuation. An
    ASCII ``[!?]+`` leaves ``?`` (U+061F) exactly where it was.
    """
    return profile(lang).emphasis.sub(" ", text).strip()
