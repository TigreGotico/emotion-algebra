"""Text-to-emotion utilities — v1.8.

Four tiers:

1. **Lexicon pipeline** (no extra deps) — :func:`from_text` and :func:`score_text`
   use :func:`~emotion_algebra.lexicons.tag_emotions`, which selects the fastest
   available backend automatically.

2. **Aho-Corasick backend** (``[fast]`` extra: ``ahocorasick-ner``) — phrase-aware,
   greedy longest-match.  Enables multi-word matches such as "heart attack" or
   "cold shoulder".  Activated transparently when the extra is installed.

3. **Mixed pipeline** — :func:`score_mixed` and :func:`from_mixed` combine word
   lexicon and emoji signals in a single pass.

4. **HuggingFace bridge** (``[transformers]`` extra) — :class:`HFEmotionAdapter`
   wraps any HF text-classification pipeline with Plutchik-compatible labels.
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


class HFEmotionAdapter:
    """Bridge from a HuggingFace text-classification model to a named emotion.

    Wraps any ``transformers`` pipeline that outputs Plutchik-compatible labels
    (e.g. ``j-hartmann/emotion-english-distilroberta-base`` which outputs
    anger, disgust, fear, joy, neutral, sadness, surprise).

    Requires the ``[transformers]`` optional extra::

        pip install "emotion-algebra[transformers]"

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
    **pipeline_kwargs:
        Forwarded to ``transformers.pipeline()``.
    """

    def __init__(self, model_name: str, **pipeline_kwargs) -> None:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "HFEmotionAdapter requires the [transformers] extra: "
                "pip install 'emotion-algebra[transformers]'"
            ) from exc
        self._model_name = model_name
        self._pipe = hf_pipeline(
            "text-classification", model=model_name, top_k=None, **pipeline_kwargs
        )

    def from_text(self, text: str) -> Optional[EmotionBase]:
        """Classify *text* and return the dominant named emotion.

        Parameters
        ----------
        text:
            Input string.

        Returns
        -------
        Emotion or None
            ``None`` if the top label does not map to a known emotion.
        """
        results = self._pipe(text)
        scores = {r["label"].lower(): r["score"] for r in results[0]}
        return self.from_scores(scores)

    def from_scores(self, scores: Dict[str, float]) -> Optional[EmotionBase]:
        """Blend a ``{label: probability}`` distribution into a named emotion.

        Each label that maps to a known :class:`~emotion_algebra.plutchik.Emotion`
        contributes proportional to its score.

        Parameters
        ----------
        scores:
            Dict of ``{label: probability}`` as returned by a classifier.

        Returns
        -------
        Emotion or None
            The dominant named emotion of the blended state, or ``None`` if
            no labels matched.
        """
        from emotion_algebra.emotions import get_emotion
        from emotion_algebra.state import EmotionalState
        state = EmotionalState()
        for label, score in scores.items():
            emo = get_emotion(label.lower())
            if emo is not None:
                state.apply(emo, weight=float(score))
        return state.dominant()


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
