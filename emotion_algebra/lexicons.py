"""Lexicon utilities — word/phrase → emotion/sentiment/color lookups.

Two backends:

1. **Dict backend** (default, zero extra deps) — O(1) per-token lookup from
   the bundled ``word_emotion_lexicon.csv``.  Single-word matches only.

2. **Aho-Corasick backend** (requires ``[fast]`` extra: ``ahocorasick-ner``) —
   phrase-aware, greedy longest-match.  Enables multi-word entries such as
   "heart attack" or "cold shoulder" to be matched as single units.
   Automatically used by :func:`~emotion_algebra.text.score_text` and
   :func:`~emotion_algebra.text.from_text` when available.

The backend is selected lazily on first use and cached for the process lifetime.
"""
from os.path import dirname, join
from typing import Optional, List, Dict


def get_color(word: str) -> Optional[str]:
    """Return the colour association for *word* from the lexicon, or ``None``."""
    if word in LEXICON:
        return LEXICON[word]["color"] or None
    return None


def get_word_emotion(word: str) -> Optional[str]:
    """Return the emotion label for *word* from the lexicon, or ``None``."""
    if word in LEXICON:
        return LEXICON[word]["emotion"] or None
    return None


def get_sentiment(word: str) -> Optional[str]:
    """Return the sentiment label for *word* from the lexicon, or ``None``."""
    if word in LEXICON:
        return LEXICON[word]["sentiment"] or None
    return None


def get_subjectivity(word: str) -> Optional[str]:
    """Return the subjectivity label for *word* from the lexicon, or ``None``."""
    if word in LEXICON:
        return LEXICON[word]["subjectivity"] or None
    return None


def get_orientation(word: str) -> Optional[str]:
    """Return the orientation label for *word* from the lexicon, or ``None``."""
    if word in LEXICON:
        return LEXICON[word]["orientation"] or None
    return None


def load_lexicon() -> Dict[str, dict]:
    """Load the bundled word-emotion CSV into a ``dict``."""
    bucket: Dict[str, dict] = {}
    lexicon_path = join(dirname(__file__), "word_emotion_lexicon.csv")
    with open(lexicon_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.replace("\n", "")
            word, emotion, color, orientation, sentiment, subjectivity, source = line.split(",")
            bucket[word] = {
                "emotion": emotion,
                "color": color,
                "orientation": orientation,
                "sentiment": sentiment,
                "subjectivity": subjectivity,
                "source": source,
            }
    return bucket


LEXICON: Dict[str, dict] = load_lexicon()


# ---------------------------------------------------------------------------
# Aho-Corasick phrase-aware backend
# ---------------------------------------------------------------------------

_ac_tagger = None          # AhocorasickNER instance, built on first use
_ac_available: Optional[bool] = None  # None = not yet checked


def _check_ac_available() -> bool:
    """Return True if ahocorasick-ner is importable."""
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

    Built once from ``LEXICON`` entries that have a non-empty emotion label.
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
        In the dict-fallback mode ``"start"`` and ``"end"`` are approximate
        (character offsets of the matched token are not computed).
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
