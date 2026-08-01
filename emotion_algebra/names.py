"""What the emotions are called, in each language.

**This module translates names. It does not translate coordinates, and the
distinction is the whole point.**

A prototype is a point in the five-axis core — `anger` is at some particular
valence, potency, arousal and unpredictability. Two separate questions follow:

1. What is that point *called* in Portuguese? (`raiva`. This module.)
2. Is Portuguese `raiva` *at the same point* as English `anger`?

The second question is the interesting one, and **this library does not know the
answer**. It cannot: there is no human-rated Arabic valence/arousal/dominance
lexicon in existence, and the only openly available Portuguese affective norms
are Brazilian rather than European. The machine-translated "Arabic" lexicons that
do exist record English raters' judgements relabelled in Arabic, and using them
would launder an English opinion into an Arabic-looking number.

So the coordinates stay **canonical and language-neutral**, and only the labels
move. ``prototypes.cross_lingual_transfer`` is graded ``CONTESTED`` to say so out
loud: the GRID work supports the four *dimensions* replicating across cultures,
but it does not establish that the *per-term positions* are stable, and nobody
should read a translated label as a claim that they are.

    >>> from emotion_algebra import dominant, prototype
    >>> dominant(prototype("raiva"), lang="pt")
    'raiva'

What that gives you is a **label in the user's language**. What it does not give
you is a *finding* about Portuguese emotion terms, and the difference between
those two things is exactly what this package exists to keep straight.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional

from emotion_algebra.lang import KASHIDA, PROFILES

#: canonical (English) name -> the name in each language.
#:
#: These are the intensity-graded Plutchik terms, so the translations must
#: preserve the *gradient* — Portuguese `irritação` / `raiva` / `fúria` has to
#: escalate the way `annoyance` / `anger` / `rage` does, or the readout will
#: return a label that is the right emotion at the wrong strength.
NAMES: Dict[str, Dict[str, str]] = {
    "pt": {
        "serenity": "serenidade",
        "joy": "alegria",
        "ecstasy": "êxtase",
        "pensiveness": "melancolia",
        "sadness": "tristeza",
        "grief": "desgosto",
        "annoyance": "irritação",
        "anger": "raiva",
        "rage": "fúria",
        "apprehension": "apreensão",
        "fear": "medo",
        "terror": "terror",
        "acceptance": "aceitação",
        "trust": "confiança",
        "admiration": "admiração",
        "boredom": "tédio",
        "disgust": "nojo",
        "loathing": "repugnância",
        "interest": "interesse",
        "anticipation": "expectativa",
        "vigilance": "vigilância",
        "distraction": "distração",
        "surprise": "surpresa",
        "amazement": "espanto",
        "distress": "angústia",
        "shame": "vergonha",
    },
    "ar": {
        "serenity": "طمأنينة",
        "joy": "فرح",
        "ecstasy": "نشوة",
        "pensiveness": "كآبة",
        "sadness": "حزن",
        "grief": "أسى",
        "annoyance": "انزعاج",
        "anger": "غضب",
        "rage": "غيظ",
        "apprehension": "تخوف",
        "fear": "خوف",
        "terror": "رعب",
        "acceptance": "قبول",
        "trust": "ثقة",
        "admiration": "إعجاب",
        "boredom": "ملل",
        "disgust": "اشمئزاز",
        "loathing": "مقت",
        "interest": "اهتمام",
        "anticipation": "ترقب",
        "vigilance": "تيقظ",
        "distraction": "تشتت",
        "surprise": "دهشة",
        "amazement": "ذهول",
        "distress": "كرب",
        "shame": "خجل",
    },
}

#: Extra spellings that resolve to the same prototype. Not new emotions — just
#: the other ways people write the ones we have.
ALIASES: Dict[str, Dict[str, str]] = {
    "pt": {
        "luto": "grief",
        "raiva intensa": "rage",
        "furia": "rage",
        "ansiedade": "apprehension",
        "receio": "apprehension",
        "repulsa": "disgust",
        "aborrecimento": "boredom",
        "curiosidade": "interest",
        "antecipacao": "anticipation",
    },
    "ar": {
        "سخط": "rage",
        "قلق": "apprehension",
        "فزع": "terror",
        "اشتياق": "anticipation",
        "عار": "shame",
        "كره": "loathing",
    },
}

#: Arabic diacritics (harakat). Nobody types them, and the same word with and
#: without them is the same word.
_HARAKAT = re.compile(r"[ً-ْٰـ]")


def normalise(name: str) -> str:
    """Fold a written name to its lookup key.

    Case, surrounding space, Arabic diacritics and kashida, and Unicode
    composition all vary between one keyboard and the next without changing which
    emotion is meant.
    """
    text = unicodedata.normalize("NFC", str(name)).strip().lower()
    text = _HARAKAT.sub("", text.replace(KASHIDA, ""))
    return " ".join(text.split())


def _index() -> Dict[str, str]:
    """Every known spelling, in every language, mapped to its canonical name."""
    table: Dict[str, str] = {}
    for lang, mapping in NAMES.items():
        for canonical, localized in mapping.items():
            table[normalise(canonical)] = canonical
            table[normalise(localized)] = canonical
    for lang, mapping in ALIASES.items():
        for alias, canonical in mapping.items():
            table[normalise(alias)] = canonical
    return table


_INDEX = _index()


def canonical(name: str) -> Optional[str]:
    """The canonical (English) prototype name for *name*, or ``None``.

    Language-agnostic: ``anger``, ``raiva`` and ``غضب`` all resolve to ``anger``,
    because they are three names for one point in the core, not three points.
    """
    key = normalise(name)
    if key in _INDEX:
        return _INDEX[key]
    # A canonical name with no translation registered is still canonical.
    from emotion_algebra.prototypes import PROTOTYPES

    return key if key in PROTOTYPES else None


def localized(name: str, lang: str = "en") -> str:
    """Render a canonical prototype name in *lang*.

    Falls back to the canonical name when a language has no word registered for
    it — a missing translation should degrade to English, not to an exception.

    Raises
    ------
    UnsupportedLanguageError
        If *lang* has no profile.
    """
    from emotion_algebra.lang import profile

    profile(lang)
    key = canonical(name) or normalise(name)
    return NAMES.get(lang, {}).get(key, key)


def names_in(lang: str = "en") -> Dict[str, str]:
    """``{localized name: canonical name}`` for every prototype in *lang*.

    This is what :func:`emotion_algebra.readout.label` uses to answer in a
    language. Note that it maps *back* to the canonical name — the coordinates
    behind each label are the same ones in every language, and that is deliberate.
    """
    from emotion_algebra.lang import profile
    from emotion_algebra.prototypes import PROTOTYPES

    profile(lang)
    table = NAMES.get(lang, {})
    return {table.get(name, name): name for name in PROTOTYPES}
