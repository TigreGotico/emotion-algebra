"""Emoji-to-emotion mapping — v1.6.

Three use cases:

1. **Free-form text scanning** — :func:`score_emojis` / :func:`from_emojis` extract
   Unicode emoji characters from any string and accumulate their emotional weights
   into an :class:`~emotion_algebra.state.EmotionalState`.

2. **DeepMoji-style classifier output** — :class:`DeepMojiAdapter` consumes a
   ``{emoji: probability}`` distribution (as produced by deepmoji and compatible
   models) and blends it into a named :class:`~emotion_algebra.plutchik.Emotion`.

3. **Direct lookup** — :func:`from_emoji` maps a single emoji character to its
   closest Plutchik primary emotion.

The canonical mapping is :data:`EMOJI_EMOTION_MAP`, a ``MappingProxyType`` keyed
on single Unicode emoji characters.  Emotion intensities follow the Hourglass of
Emotions axis: serenity (1) → joy (2) → ecstasy (3).

Scientific basis
----------------
Emoji-to-emotion assignments are grounded in:

- Plutchik (1980) — 8 primary emotions + dyad feelings
- Novak et al. (2015) "Sentiment of Emojis" — polarity and arousal crowd scores
- Felbo et al. (2017) "Using millions of emoji occurrences to learn…" — the 64
  DeepMoji emoji label set and their empirical sentiment clusters
"""
from __future__ import annotations

import re
import unicodedata
from types import MappingProxyType
from typing import Optional, Dict

from emotion_algebra.base import EmotionBase

# ---------------------------------------------------------------------------
# Canonical emoji → Plutchik primary emotion name
# ---------------------------------------------------------------------------
# Coverage: all 8 primaries across the intensity spectrum (mild → intense).
# Secondary/feeling emojis (e.g. ❤️ → love) map to the dominant primary
# component of that feeling.
# ---------------------------------------------------------------------------

_EMOJI_EMOTION_MAP_RAW: Dict[str, str] = {
    # --- JOY ----------------------------------------------------------------
    "😀": "joy",        # grinning
    "😃": "joy",        # grinning big eyes
    "😄": "joy",        # grinning smiling eyes
    "😁": "joy",        # beaming
    "😆": "ecstasy",    # laughing (intense joy)
    "😂": "ecstasy",    # tears of joy
    "🤣": "ecstasy",    # rofl
    "😊": "serenity",   # smiling (mild joy)
    "☺️": "serenity",
    "🥰": "joy",        # smiling hearts
    "😍": "joy",        # heart eyes
    "🤩": "ecstasy",    # star-struck
    "😎": "joy",        # cool
    "🥳": "joy",        # party
    "🎉": "joy",
    "🙌": "joy",
    "😋": "serenity",
    "😝": "joy",
    "💃": "joy",
    "🕺": "joy",

    # --- SADNESS ------------------------------------------------------------
    "😢": "sadness",
    "😭": "grief",      # intense sadness
    "😔": "pensiveness", # mild sadness
    "😞": "sadness",
    "😟": "sadness",
    "😕": "pensiveness",
    "☹️": "sadness",
    "🙁": "pensiveness",
    "💔": "sadness",
    "😿": "grief",
    "😥": "sadness",
    "😓": "sadness",
    "🥺": "pensiveness",

    # --- ANGER --------------------------------------------------------------
    "😠": "anger",
    "😡": "rage",       # intense anger
    "🤬": "rage",
    "💢": "anger",
    "😤": "annoyance",  # mild anger
    "👊": "anger",
    "✊": "anger",
    "🔥": "rage",

    # --- FEAR ---------------------------------------------------------------
    "😨": "fear",
    "😰": "fear",
    "😱": "terror",     # intense fear
    "😧": "fear",
    "😬": "apprehension", # mild fear
    "🙀": "fear",
    "😳": "apprehension",

    # --- DISGUST ------------------------------------------------------------
    "🤢": "disgust",
    "🤮": "loathing",   # intense disgust
    "😒": "disgust",
    "🙄": "disgust",
    "😑": "boredom",    # mild disgust / disengagement
    "😐": "boredom",

    # --- TRUST --------------------------------------------------------------
    "🙏": "trust",
    "😇": "trust",
    "🤝": "trust",
    "💙": "trust",
    "💗": "trust",
    "💕": "trust",
    "❤️": "trust",
    "🫶": "trust",
    "🫂": "trust",
    "💪": "admiration",  # intense trust/admiration

    # --- ANTICIPATION -------------------------------------------------------
    "🤔": "anticipation",
    "👀": "anticipation",
    "🧐": "anticipation",
    "🫤": "interest",    # mild anticipation
    "🤭": "anticipation",
    "😏": "anticipation",

    # --- SURPRISE -----------------------------------------------------------
    "😲": "surprise",
    "😯": "surprise",
    "🤯": "amazement",  # intense surprise
    "😮": "surprise",
    "🫨": "amazement",

    # --- FEELINGS (dyads) — mapped to dominant primary ----------------------
    # love = joy + trust → joy
    "💖": "joy",
    "💞": "joy",
    "💓": "joy",
    "💝": "joy",
    # delight = joy + surprise → joy
    "✨": "joy",
    # optimism = joy + anticipation → anticipation
    "🌟": "anticipation",
    "⭐": "anticipation",
    # submission = trust + fear → trust
    "🙇": "trust",
    # awe = surprise + fear → surprise
    "😦": "surprise",
    # disapproval = surprise + disgust → disgust
    "👎": "disgust",
    # remorse = sadness + disgust → sadness
    "😖": "sadness",
    "😣": "sadness",
    # contempt = anger + disgust → anger
    "😾": "anger",
    # aggressiveness = anger + anticipation → anger
    "⚔️": "anger",
    # envy = sadness + anger → sadness
    "😼": "sadness",
}

EMOJI_EMOTION_MAP: MappingProxyType = MappingProxyType(_EMOJI_EMOTION_MAP_RAW)
"""Immutable view of the canonical emoji → Plutchik emotion name mapping.

Values are emotion names accepted by
:func:`~emotion_algebra.emotions.get_emotion`.  Intensity variants (serenity,
joy, ecstasy; apprehension, fear, terror; etc.) are preserved where meaningful.

Use :func:`register_emoji` / :func:`unregister_emoji` to extend or override
mappings at runtime without modifying this constant.
"""

# Pre-compiled pattern matching any character in the map.
# Uses a character class over the exact key set for O(1)-per-char matching.
_EMOJI_RE = re.compile(
    "[" + re.escape("".join(_EMOJI_EMOTION_MAP_RAW.keys())) + "]"
)

# ---------------------------------------------------------------------------
# Runtime registry — user-defined overrides layered on top of the canonical map
# ---------------------------------------------------------------------------

_USER_REGISTRY: Dict[str, str] = {}


def register_emoji(emoji_char: str, emotion_name: str) -> None:
    """Register a custom emoji → emotion mapping at runtime.

    Overrides the canonical :data:`EMOJI_EMOTION_MAP` for *emoji_char*.
    The registration persists for the lifetime of the process.

    Parameters
    ----------
    emoji_char:
        A single Unicode emoji (or multi-codepoint sequence).
    emotion_name:
        Name of any emotion known to
        :func:`~emotion_algebra.emotions.get_emotion`.

    Raises
    ------
    ValueError
        If *emotion_name* is not a recognised emotion name.

    Examples
    --------
    >>> register_emoji("🤖", "trust")
    >>> from_emoji("🤖").name
    'trust'
    """
    from emotion_algebra.emotions import get_emotion
    if get_emotion(emotion_name) is None:
        raise ValueError(
            f"{emotion_name!r} is not a recognised emotion name. "
            "Use get_emotion() to verify valid names."
        )
    _USER_REGISTRY[emoji_char.strip()] = emotion_name


def unregister_emoji(emoji_char: str) -> bool:
    """Remove a runtime emoji registration.

    Has no effect on the canonical :data:`EMOJI_EMOTION_MAP`.

    Parameters
    ----------
    emoji_char:
        The emoji character to remove from the runtime registry.

    Returns
    -------
    bool
        ``True`` if the entry existed and was removed, ``False`` otherwise.
    """
    return _USER_REGISTRY.pop(emoji_char.strip(), None) is not None


def _lookup(emoji_char: str) -> Optional[str]:
    """Return the emotion name for *emoji_char*, checking user registry first."""
    return _USER_REGISTRY.get(emoji_char) or _EMOJI_EMOTION_MAP_RAW.get(emoji_char)


def from_emoji(emoji_char: str) -> Optional[EmotionBase]:
    """Return the :class:`~emotion_algebra.plutchik.Emotion` for *emoji_char*.

    Parameters
    ----------
    emoji_char:
        A single Unicode emoji character or multi-codepoint sequence (e.g.
        ``"❤️"``).  Whitespace is stripped before lookup.

    Returns
    -------
    Emotion or None
        ``None`` if the emoji is not in :data:`EMOJI_EMOTION_MAP`.
    """
    from emotion_algebra.emotions import get_emotion
    key = emoji_char.strip()
    label = _lookup(key)
    if label is None:
        return None
    return get_emotion(label)


def score_emojis(text: str) -> "EmotionalState":
    """Scan *text* for emoji characters and return a blended
    :class:`~emotion_algebra.state.EmotionalState`.

    Each emoji that appears in :data:`EMOJI_EMOTION_MAP` contributes equally
    (weight=1.0) to the state.  Repeated emojis accumulate.

    Parameters
    ----------
    text:
        Any string, including mixed emoji/word content.

    Returns
    -------
    EmotionalState
        Zero-vector state if no mapped emoji is found.
    """
    from emotion_algebra.state import EmotionalState
    from emotion_algebra.emotions import get_emotion
    state = EmotionalState()
    for char in text:
        label = _lookup(char)
        if label:
            emo = get_emotion(label)
            if emo is not None:
                state.apply(emo)
    return state


def from_emojis(text: str) -> Optional[EmotionBase]:
    """Return the dominant emotion inferred from emoji characters in *text*.

    Convenience wrapper around :func:`score_emojis` that returns a single
    named emotion via :meth:`~emotion_algebra.state.EmotionalState.dominant`.

    Parameters
    ----------
    text:
        Any string.

    Returns
    -------
    Emotion or None
        ``None`` if no mapped emoji appears in *text*.
    """
    state = score_emojis(text)
    return state.dominant()


class DeepMojiAdapter:
    """Map a DeepMoji-style ``{emoji: probability}`` distribution to a named emotion.

    DeepMoji (Felbo et al. 2017) and compatible models output a probability
    distribution over a fixed emoji label set.  This adapter converts that
    distribution into an :class:`~emotion_algebra.state.EmotionalState` by
    weighting each mapped emoji's emotion by its predicted probability.

    Usage::

        adapter = DeepMojiAdapter()

        # From a raw {emoji: score} dict (deepmoji / torchMoji output):
        scores = {"😂": 0.45, "😊": 0.30, "😭": 0.25}
        emotion = adapter.from_scores(scores)

        # From a ranked list of (emoji, score) tuples:
        emotion = adapter.from_ranked([("😂", 0.45), ("😊", 0.30)])

    Notes
    -----
    Emojis not present in :data:`EMOJI_EMOTION_MAP` are silently skipped.
    If no emoji in the input maps to a known emotion, returns ``None``.
    """

    def from_scores(self, scores: Dict[str, float]) -> Optional[EmotionBase]:
        """Blend *scores* into a named emotion.

        Parameters
        ----------
        scores:
            ``{emoji_char: probability}`` dict.  Probabilities need not sum to 1.

        Returns
        -------
        Emotion or None
            Dominant named emotion of the blended state, or ``None`` if no
            emoji in *scores* is in :data:`EMOJI_EMOTION_MAP`.
        """
        from emotion_algebra.state import EmotionalState
        from emotion_algebra.emotions import get_emotion
        state = EmotionalState()
        for emoji_char, score in scores.items():
            label = _lookup(emoji_char)
            if label:
                emo = get_emotion(label)
                if emo is not None:
                    state.apply(emo, weight=float(score))
        return state.dominant()

    def from_ranked(
        self, ranked: list, top_k: Optional[int] = None
    ) -> Optional[EmotionBase]:
        """Blend a ranked list of ``(emoji, score)`` tuples.

        Parameters
        ----------
        ranked:
            List of ``(emoji_char, probability)`` tuples, highest-score first.
        top_k:
            If given, only the first *top_k* entries are considered.

        Returns
        -------
        Emotion or None
        """
        subset = ranked[:top_k] if top_k is not None else ranked
        return self.from_scores(dict(subset))

    def score_state(self, scores: Dict[str, float]) -> "EmotionalState":
        """Like :meth:`from_scores` but returns the full blended
        :class:`~emotion_algebra.state.EmotionalState` instead of the dominant emotion.

        Useful when you need valence/arousal/axis breakdown rather than a
        single named result.

        Parameters
        ----------
        scores:
            ``{emoji_char: probability}`` dict.

        Returns
        -------
        EmotionalState
        """
        from emotion_algebra.state import EmotionalState
        from emotion_algebra.emotions import get_emotion
        state = EmotionalState()
        for emoji_char, score in scores.items():
            label = _lookup(emoji_char)
            if label:
                emo = get_emotion(label)
                if emo is not None:
                    state.apply(emo, weight=float(score))
        return state
