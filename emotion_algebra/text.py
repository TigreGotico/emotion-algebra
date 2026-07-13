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

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, TYPE_CHECKING

from emotion_algebra.base import EmotionBase

if TYPE_CHECKING:
    from emotion_algebra.float_emotion import FloatEmotion
    from emotion_algebra.state import EmotionalState


#: Words that flip the polarity of a following emotion word.
#:
#: Negation is handled by the standard *contextual valence shifter* method
#: (Polanyi & Zaenen 2006; Taboada et al. 2011, *Lexicon-Based Methods for
#: Sentiment Analysis*, Computational Linguistics 37(2)): a negator flips the
#: emotion words inside a short following window rather than the whole sentence.
NEGATORS: frozenset = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "cannot", "can't", "cant", "won't", "wont", "isn't", "isnt", "aren't",
    "arent", "wasn't", "wasnt", "don't", "dont", "doesn't", "doesnt",
    "didn't", "didnt", "hardly", "barely", "scarcely", "without", "lack",
    "lacks", "lacking",
})

#: How far after a negator its scope reaches, in tokens.
#:
#: Taboada et al. use a small fixed window; beyond it, negation reliably stops
#: applying and a sentence-wide flip does more harm than good.
NEGATION_WINDOW: int = 3

#: Multipliers applied to an emotion word preceded by an intensifier/downtoner.
#:
#: Same lineage as the negators — Taboada et al. treat these as the other class
#: of valence shifter, scaling rather than flipping.
INTENSIFIERS: Dict[str, float] = {
    "slightly": 0.5, "somewhat": 0.6, "a_bit": 0.6, "kind_of": 0.6,
    "rather": 0.8, "fairly": 0.8, "pretty": 0.9,
    "very": 1.5, "really": 1.5, "so": 1.4, "quite": 1.2, "highly": 1.5,
    "extremely": 2.0, "incredibly": 2.0, "absolutely": 2.0, "totally": 1.8,
    "utterly": 2.0, "deeply": 1.7, "terribly": 1.8, "insanely": 2.0,
}

#: How far before an emotion word an intensifier reaches, in tokens.
INTENSIFIER_WINDOW: int = 2

_TOKEN_RE = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class EmotionSpan:
    """One lexicon hit, with the valence shifters that applied to it."""

    word: str
    label: str
    start: int
    end: int
    weight: float = 1.0
    negated: bool = False

    def __str__(self) -> str:
        shift = []
        if self.negated:
            shift.append("negated")
        if self.weight != 1.0:
            shift.append(f"×{self.weight:g}")
        suffix = f" ({', '.join(shift)})" if shift else ""
        return f"{self.word} → {self.label}{suffix}"


@dataclass(frozen=True)
class TextEmotionResult:
    """The full result of :func:`analyze` — spans, aggregate, and polarity.

    Attributes
    ----------
    text:
        The input, unchanged.
    spans:
        Every lexicon hit, in order, with its shifters resolved.
    aggregate:
        The summed :class:`~emotion_algebra.float_emotion.FloatEmotion` — the
        shifted contribution of every span.  Like
        :func:`~emotion_algebra.state.EmotionalState.apply`, this is an
        **unbounded accumulator**: enough matching words push an axis past the
        Hourglass ``[-3, 3]`` range, so it measures weight of evidence rather
        than a single emotion's intensity.  :attr:`polarity` is still clamped to
        ``[-1, 1]``, and :attr:`dominant` projects back onto a named emotion.
    """

    text: str
    spans: List[EmotionSpan] = field(default_factory=list)
    aggregate: "FloatEmotion" = None  # set in analyze()

    @property
    def polarity(self) -> float:
        """Cambria's four-axis sentiment score for the aggregate, in ``[-1, 1]``."""
        return self.aggregate.polarity

    @property
    def dominant(self) -> Optional[EmotionBase]:
        """The nearest named emotion to the aggregate, or ``None`` if nothing matched."""
        if not self.spans:
            return None
        from emotion_algebra.distance import closest_emotion
        return closest_emotion(self.aggregate.as_array)

    def __bool__(self) -> bool:
        return bool(self.spans)

    def __len__(self) -> int:
        return len(self.spans)

    def __repr__(self) -> str:
        return (
            f"TextEmotionResult(spans={len(self.spans)}, "
            f"dominant={self.dominant.name if self.dominant else None!r}, "
            f"polarity={self.polarity:+.2f})"
        )


def _shifters(text: str) -> tuple:
    """Return (token list, char-offset list) for valence-shifter lookup."""
    tokens, offsets = [], []
    for match in _TOKEN_RE.finditer(text.lower()):
        tokens.append(match.group())
        offsets.append(match.start())
    return tokens, offsets


def analyze(text: str) -> TextEmotionResult:
    """Analyze *text*, resolving negation and intensifiers.

    Unlike :func:`score_text`, which counts every lexicon hit at weight 1.0,
    this applies the two standard *contextual valence shifters*:

    * a :data:`NEGATORS` word flips the emotions in the next
      :data:`NEGATION_WINDOW` tokens ("not happy" reads as sadness, not joy);
    * an :data:`INTENSIFIERS` word scales the emotions in the next
      :data:`INTENSIFIER_WINDOW` tokens ("very happy" reads stronger, "slightly
      happy" weaker).

    Both follow Taboada et al. (2011).

    Negation flips the emotion *vector* — the span contributes its exact
    opposite.  Note that this does not always flip
    :attr:`~TextEmotionResult.polarity`: Cambria's polarity formula takes the
    *magnitude* of Attention, so negating a purely attention-axis emotion
    (anticipation ↔ surprise) leaves the polarity unchanged by construction.
    Polarity flips as you would expect on the hedonic axes.

    Parameters
    ----------
    text:
        Any string.  Case-insensitive.

    Returns
    -------
    TextEmotionResult
        Falsy when nothing matched.

    Examples
    --------
    >>> analyze("I am joyful").polarity > 0
    True
    >>> analyze("I am not joyful").polarity < 0
    True
    >>> analyze("I am very joyful").polarity > analyze("I am joyful").polarity
    True
    >>> analyze("I am slightly joyful").polarity < analyze("I am joyful").polarity
    True
    """
    from emotion_algebra.emotions import get_emotion
    from emotion_algebra.float_emotion import FloatEmotion
    from emotion_algebra.lexicons import tag_emotions

    tokens, offsets = _shifters(text)
    spans: List[EmotionSpan] = []
    vector = FloatEmotion(0, 0, 0, 0).as_array.copy()

    for match in tag_emotions(text):
        label = match.get("label")
        emotion = get_emotion(label) if label else None
        if emotion is None:
            continue

        start = int(match.get("start", 0))
        # Index of the first token of this match, so windows are measured in
        # tokens rather than characters.
        idx = next((i for i, off in enumerate(offsets) if off >= start), len(tokens))

        negated = any(
            tokens[i] in NEGATORS
            for i in range(max(0, idx - NEGATION_WINDOW), idx)
        )
        weight = 1.0
        for i in range(max(0, idx - INTENSIFIER_WINDOW), idx):
            if tokens[i] in INTENSIFIERS:
                weight *= INTENSIFIERS[tokens[i]]

        span = EmotionSpan(
            word=match.get("word", ""),
            label=label,
            start=start,
            end=int(match.get("end", start)),
            weight=weight,
            negated=negated,
        )
        spans.append(span)

        contribution = FloatEmotion.from_emotion(emotion).as_array * weight
        if negated:
            contribution = -contribution
        vector += contribution

    return TextEmotionResult(
        text=text, spans=spans, aggregate=FloatEmotion(*vector)
    )


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
