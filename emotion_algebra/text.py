"""Text-to-emotion utilities — v1.2.

Two tiers:

1. **Lexicon pipeline** (no extra deps beyond the bundled CSV) —
   :func:`from_text` and :func:`score_text` tokenize the input and look up
   each word in the bundled word-emotion lexicon.

2. **HuggingFace bridge** (requires the ``[transformers]`` optional extra) —
   :class:`HFEmotionAdapter` wraps any HF text-classification pipeline that
   outputs Plutchik-compatible labels.
"""
from __future__ import annotations

import re
from typing import Optional, Dict, TYPE_CHECKING

from emotion_algebra.base import EmotionBase

if TYPE_CHECKING:
    from emotion_algebra.state import EmotionalState


def from_text(text: str) -> Optional[EmotionBase]:
    """Return the dominant emotion inferred from *text* via lexicon lookup.

    Tokenizes *text*, looks up each token in the bundled word-emotion CSV,
    and returns the most frequently matched named emotion.

    Parameters
    ----------
    text:
        Any string.  Case-insensitive.

    Returns
    -------
    Emotion or None
        Most common lexicon match, or ``None`` if no tokens match.
    """
    from emotion_algebra.lexicons import get_word_emotion
    from emotion_algebra.emotions import get_emotion
    tokens = re.findall(r'\b\w+\b', text.lower())
    counts: Dict[str, int] = {}
    for token in tokens:
        label = get_word_emotion(token)
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return None
    best_label = max(counts, key=lambda k: counts[k])
    return get_emotion(best_label)


def score_text(text: str) -> "EmotionalState":
    """Return a full :class:`~emotion_algebra.state.EmotionalState` for *text*.

    All lexicon-matched tokens contribute with equal weight.  The result is a
    weighted 4-axis accumulation — richer than the single top-1 from
    :func:`from_text`.

    Parameters
    ----------
    text:
        Any string.  Case-insensitive.

    Returns
    -------
    EmotionalState
        A neutral (zero-vector) state if no tokens match.
    """
    from emotion_algebra.lexicons import get_word_emotion
    from emotion_algebra.emotions import get_emotion
    from emotion_algebra.state import EmotionalState
    state = EmotionalState()
    tokens = re.findall(r'\b\w+\b', text.lower())
    for token in tokens:
        label = get_word_emotion(token)
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
