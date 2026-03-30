"""Text-to-emotion utilities — v1.9.

Three tiers:

1. **Lexicon pipeline** (no extra deps) — :func:`from_text` and :func:`score_text`
   use :func:`~emotion_algebra.lexicons.tag_emotions`, which selects the fastest
   available backend automatically.

2. **Aho-Corasick backend** (``[fast]`` extra: ``ahocorasick-ner``) — phrase-aware,
   greedy longest-match.  Enables multi-word matches such as "heart attack" or
   "cold shoulder".  Activated transparently when the extra is installed.

3. **Mixed pipeline** — :func:`score_mixed` and :func:`from_mixed` combine word
   lexicon and emoji signals in a single pass.

For neural text-to-emotion use :class:`~emotion_algebra.deepmoji.DeepMojiONNXAdapter`
(``deepmoji-onnx`` canonical dependency).
"""
from __future__ import annotations

from typing import Optional, Dict, TYPE_CHECKING

from emotion_algebra.base import EmotionBase

if TYPE_CHECKING:
    from emotion_algebra.state import EmotionalState


def from_text(text: str) -> Optional[EmotionBase]:
    """Return the dominant emotion inferred from *text* via lexicon lookup.

    Uses :func:`~emotion_algebra.lexicons.tag_emotions` — automatically
    phrase-aware when ``ahocorasick-ner`` is installed (``[fast]`` extra).

    Parameters
    ----------
    text:
        Any string.  Case-insensitive.

    Returns
    -------
    Emotion or None
        Most common lexicon match, or ``None`` if no tokens match.
    """
    from emotion_algebra.lexicons import tag_emotions
    from emotion_algebra.emotions import get_emotion
    counts: Dict[str, int] = {}
    for match in tag_emotions(text):
        label = match["label"]
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return None
    best_label = max(counts, key=lambda k: counts[k])
    return get_emotion(best_label)


def score_text(text: str) -> "EmotionalState":
    """Return a full :class:`~emotion_algebra.state.EmotionalState` for *text*.

    All lexicon matches contribute with equal weight (weight=1.0 per match).
    Uses :func:`~emotion_algebra.lexicons.tag_emotions` — automatically
    phrase-aware when ``ahocorasick-ner`` is installed (``[fast]`` extra).

    Parameters
    ----------
    text:
        Any string.  Case-insensitive.

    Returns
    -------
    EmotionalState
        A neutral (zero-vector) state if no tokens match.
    """
    from emotion_algebra.lexicons import tag_emotions
    from emotion_algebra.emotions import get_emotion
    from emotion_algebra.state import EmotionalState
    state = EmotionalState()
    for match in tag_emotions(text):
        label = match["label"]
        if label:
            emo = get_emotion(label)
            if emo is not None:
                state.apply(emo)
    return state



def score_mixed(text: str) -> "EmotionalState":
    """Score *text* using both the word lexicon and emoji map in a single pass.

    Word tokens and emoji characters contribute equally (weight=1.0 each).
    This is the recommended entry point for general-purpose text that may
    contain both natural language and emoji.

    Parameters
    ----------
    text:
        Any string, e.g. ``"I'm so happy 😄🎉"``.

    Returns
    -------
    EmotionalState
        Zero-vector if no tokens or emoji match.
    """
    from emotion_algebra.emoji import score_emojis
    state = score_text(text)
    emoji_state = score_emojis(text)
    state._vector += emoji_state._vector
    return state


def from_mixed(text: str) -> Optional[EmotionBase]:
    """Return the dominant emotion from *text* using both word and emoji signals.

    Convenience wrapper around :func:`score_mixed`.

    Parameters
    ----------
    text:
        Any string.

    Returns
    -------
    Emotion or None
        ``None`` if neither the lexicon nor the emoji map produce any match.
    """
    return score_mixed(text).dominant()
