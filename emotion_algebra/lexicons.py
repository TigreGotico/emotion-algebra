"""Lexicon utilities — word/phrase → emotion/sentiment/color/Hourglass lookups.

Two backends:

1. **Dict backend** (default, zero extra deps) — O(1) per-token lookup from
   the bundled ``word_emotion_lexicon.csv``.  Single-word matches only.

2. **Aho-Corasick backend** (requires ``[fast]`` extra: ``ahocorasick-ner``) —
   phrase-aware, greedy longest-match.  Enables multi-word entries such as
   "heart attack" or "cold shoulder" to be matched as single units.
   Automatically used by :func:`~emotion_algebra.text.score_text` and
   :func:`~emotion_algebra.text.from_text` when available.

The backend is selected lazily on first use and cached for the process lifetime.

Canonical lexicon schema (``word_emotion_lexicon.csv``):
  word            — lowercase token
  emotion         — Plutchik primary or ""
  pleasantness    — SenticNet introspection axis [-1,1] or ""
  attention       — SenticNet temper axis [-1,1] or ""
  aptitude        — SenticNet attitude axis [-1,1] or ""
  sensitivity     — SenticNet sensitivity axis [-1,1] or ""
  sentiment       — positive / negative / neutral or ""
  afinn_score     — integer or ""
  color           — color string or ""
  orientation     — orientation tag or ""
  subjectivity    — subjectivity tag or ""
"""
from __future__ import annotations

import csv
from os.path import dirname, join
from typing import Optional, List, Dict


# ---------------------------------------------------------------------------
# Lexicon loader
# ---------------------------------------------------------------------------

def load_lexicon() -> Dict[str, dict]:
    """Load the bundled word-emotion CSV into a ``dict`` keyed by word."""
    bucket: Dict[str, dict] = {}
    lexicon_path = join(dirname(__file__), "word_emotion_lexicon.csv")
    with open(lexicon_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("word", "").strip()
            if not word:
                continue
            bucket[word] = {
                "emotion":      row.get("emotion", "").strip(),
                "pleasantness": row.get("pleasantness", "").strip(),
                "attention":    row.get("attention", "").strip(),
                "aptitude":     row.get("aptitude", "").strip(),
                "sensitivity":  row.get("sensitivity", "").strip(),
                "sentiment":    row.get("sentiment", "").strip(),
                "afinn_score":  row.get("afinn_score", "").strip(),
                "color":        row.get("color", "").strip(),
                "orientation":  row.get("orientation", "").strip(),
                "subjectivity": row.get("subjectivity", "").strip(),
            }
    return bucket


LEXICON: Dict[str, dict] = load_lexicon()


# ---------------------------------------------------------------------------
# Field accessors
# ---------------------------------------------------------------------------

def get_word_emotion(word: str) -> Optional[str]:
    """Return the Plutchik emotion label for *word*, or ``None``."""
    if word in LEXICON:
        return LEXICON[word]["emotion"] or None
    return None


def get_color(word: str) -> Optional[str]:
    """Return the colour association for *word*, or ``None``."""
    if word in LEXICON:
        return LEXICON[word]["color"] or None
    return None


def get_sentiment(word: str) -> Optional[str]:
    """Return the sentiment label (positive/negative/neutral) for *word*, or ``None``."""
    if word in LEXICON:
        return LEXICON[word]["sentiment"] or None
    return None


def get_subjectivity(word: str) -> Optional[str]:
    """Return the subjectivity label for *word*, or ``None``."""
    if word in LEXICON:
        return LEXICON[word]["subjectivity"] or None
    return None


def get_orientation(word: str) -> Optional[str]:
    """Return the orientation label for *word*, or ``None``."""
    if word in LEXICON:
        return LEXICON[word]["orientation"] or None
    return None


def get_afinn_score(word: str) -> Optional[int]:
    """Return the AFINN-111 integer sentiment score for *word*, or ``None``."""
    if word in LEXICON:
        val = LEXICON[word]["afinn_score"]
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return None


def get_hourglass(word: str) -> Optional[Dict[str, float]]:
    """Return the four SenticNet Hourglass axis values for *word*, or ``None``.

    Returns
    -------
    dict with keys ``pleasantness``, ``attention``, ``aptitude``, ``sensitivity``
    (all floats in [-1, 1]), or ``None`` if the word has no SenticNet data.
    """
    entry = LEXICON.get(word)
    if entry is None:
        return None
    try:
        p = entry["pleasantness"]
        a = entry["attention"]
        ap = entry["aptitude"]
        s = entry["sensitivity"]
        if not any([p, a, ap, s]):
            return None
        return {
            "pleasantness": float(p) if p else 0.0,
            "attention":    float(a) if a else 0.0,
            "aptitude":     float(ap) if ap else 0.0,
            "sensitivity":  float(s) if s else 0.0,
        }
    except (ValueError, TypeError):
        return None


def get_float_emotion(word: str) -> Optional["FloatEmotion"]:
    """Return a :class:`~emotion_algebra.continuous.FloatEmotion` for *word*.

    Uses the SenticNet Hourglass columns.  Returns ``None`` if the word has
    no Hourglass data.

    Parameters
    ----------
    word:
        Lowercase token to look up.
    """
    axes = get_hourglass(word)
    if axes is None:
        return None
    try:
        from emotion_algebra.float_emotion import FloatEmotion
        return FloatEmotion(
            sensitivity=axes["sensitivity"],
            attention=axes["attention"],
            pleasantness=axes["pleasantness"],
            aptitude=axes["aptitude"],
        )
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Aho-Corasick phrase-aware backend
# ---------------------------------------------------------------------------

_ac_tagger = None                     # AhocorasickNER instance, built on first use
_ac_available: Optional[bool] = None  # None = not yet checked


def _check_ac_available() -> bool:
    """Return True if ``ahocorasick-ner`` is importable."""
    global _ac_available
    if _ac_available is None:
        try:
            import ahocorasick_ner  # noqa: F401
            _ac_available = True
        except ImportError:
            _ac_available = False
    return _ac_available


def _get_ac_tagger():
    """Return a lazily-built, module-level AhocorasickNER tagger.

    Built once from :data:`LEXICON` entries that have a non-empty emotion label.
    Subsequent calls return the cached instance.
    """
    global _ac_tagger
    if _ac_tagger is not None:
        return _ac_tagger

    from ahocorasick_ner import AhocorasickNER
    tagger = AhocorasickNER(case_sensitive=False)
    for word, data in LEXICON.items():
        emotion = data.get("emotion", "")
        if emotion:
            tagger.add_word(emotion, word)
    tagger.fit()
    _ac_tagger = tagger
    return _ac_tagger


def tag_emotions(text: str) -> List[Dict]:
    """Return a list of emotion tag dicts for *text*.

    Uses the Aho-Corasick backend if ``ahocorasick-ner`` is installed
    (``[fast]`` extra), otherwise falls back to per-token dict lookup.

    Parameters
    ----------
    text:
        Any string.  Case-insensitive.

    Returns
    -------
    list of dict
        Each dict has keys ``"word"``, ``"label"``, ``"start"``, ``"end"``.
    """
    if _check_ac_available():
        tagger = _get_ac_tagger()
        return list(tagger.tag(text, min_word_len=1))

    # Fallback: per-token dict lookup
    import re
    results = []
    for m in re.finditer(r'\b\w+\b', text.lower()):
        token = m.group()
        emotion = get_word_emotion(token)
        if emotion:
            results.append({
                "word": token,
                "label": emotion,
                "start": m.start(),
                "end": m.end(),
            })
    return results


if __name__ == "__main__":
    from pprint import pprint
    pprint(LEXICON)
