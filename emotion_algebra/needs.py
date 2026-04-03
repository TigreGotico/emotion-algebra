"""Need-to-emotion mapping — Max-Neef (1991) + Murray (1938) → Plutchik.

Maps fundamental human needs to the primary emotions that arise when those
needs are deficient.  Grounded in:

* **Max-Neef (1991)** — 9 fundamental needs: subsistence, protection, freedom,
  identity, participation, creation, understanding, idleness, affection.
* **Murray (1938)** — 17 psychogenic needs (subset): sentience, exhibition,
  nurturance, harm_avoidance, achievement.
* **Plutchik (1980)** — 8 primary emotions mapped to 4 Hourglass axes.

The mapping follows appraisal logic: a deficient need implies a blocked goal
(Scherer's "goal incongruence"), and the resulting emotion depends on which
CIA drive (Control / Identity / Arousal) is threatened:

    Control deficit   → fear / anger    (threat to safety / autonomy)
    Identity deficit  → sadness / disgust (threat to self-concept / belonging)
    Arousal deficit   → boredom / surprise (under/over-stimulation)

References
----------
Max-Neef, M. (1991). *Human Scale Development*.
Murray, H. A. (1938). *Explorations in Personality*.
Plutchik, R. (1980). *Emotion: A Psychoevolutionary Synthesis*.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emotion_algebra.base import EmotionBase
    from emotion_algebra.float_emotion import FloatEmotion

# ---------------------------------------------------------------------------
# Max-Neef need → primary Plutchik emotion when deficient
# ---------------------------------------------------------------------------

MAXNEEF_DEFICIT_EMOTIONS: dict[str, str] = {
    # Control needs — deficiency threatens safety/autonomy
    "subsistence":   "sadness",       # deprivation of nourishment → grief/loss
    "protection":    "fear",          # vulnerability → threat response
    "freedom":       "anger",         # constraint → obstacle to remove

    # Identity needs — deficiency threatens self-concept/belonging
    "identity":      "pensiveness",   # weakened self → introspection/melancholy
    "participation": "apprehension",  # exclusion → social uncertainty
    "creation":      "boredom",       # blocked creativity → disengagement
    "understanding": "distraction",   # confusion → disorientation

    # Arousal needs — deficiency threatens stimulation balance
    "idleness":      "annoyance",     # over-stimulation → irritability
    "affection":     "sadness",       # loneliness → loss/grief
}

# ---------------------------------------------------------------------------
# Murray psychogenic need → primary Plutchik emotion when deficient
# ---------------------------------------------------------------------------

MURRAY_DEFICIT_EMOTIONS: dict[str, str] = {
    # Murray (1938) — all 17 psychogenic needs
    # Each mapped to the primary Plutchik emotion that arises when the need is blocked.
    "achievement":    "annoyance",     # accomplish difficult tasks → frustration
    "affiliation":    "sadness",       # form friendships → loneliness/grief
    "aggression":     "anger",         # overcome opposition → rage at impotence
    "autonomy":       "anger",         # resist constraint → obstacle to remove
    "counteraction":  "annoyance",     # overcome weakness → frustrated determination
    "defendance":     "fear",          # defend self against criticism → threat
    "deference":      "apprehension",  # follow a superior → uncertainty without guidance
    "dominance":      "annoyance",     # control environment → frustrated agency
    "exhibition":     "pensiveness",   # make an impression → melancholy invisibility
    "harm_avoidance": "fear",          # avoid pain/danger → threat response
    "infavoidance":   "apprehension",  # avoid humiliation → social anxiety
    "nurturance":     "sadness",       # care for others → grief at helplessness
    "order":          "boredom",       # arrange/organise → disengaged chaos
    "play":           "boredom",       # fun/humour → dull disengagement
    "rejection":      "disgust",       # exclude inferior → revulsion unmet
    "sentience":      "boredom",       # aesthetic/sensory experience → dull deprivation
    "understanding":  "distraction",   # enquire/analyse → disoriented confusion
}

# Combined mapping
NEED_DEFICIT_EMOTIONS: dict[str, str] = {
    **MAXNEEF_DEFICIT_EMOTIONS,
    **MURRAY_DEFICIT_EMOTIONS,
}


def need_deficit_to_emotion(need_name: str) -> Optional["EmotionBase"]:
    """Return the Plutchik emotion that arises when *need_name* is deficient.

    Parameters
    ----------
    need_name:
        A Max-Neef or Murray need name (e.g. ``"subsistence"``, ``"sentience"``).

    Returns
    -------
    Emotion or None
        The primary emotion, or ``None`` if the need name is unknown.

    Examples
    --------
    >>> need_deficit_to_emotion("protection").name
    'fear'
    >>> need_deficit_to_emotion("creation").name
    'boredom'
    """
    from emotion_algebra.emotions import get_emotion
    emotion_name = NEED_DEFICIT_EMOTIONS.get(need_name)
    return get_emotion(emotion_name) if emotion_name else None


def need_deficit_to_float_emotion(need_name: str) -> Optional["FloatEmotion"]:
    """Return a continuous :class:`FloatEmotion` for a deficient need.

    Converts the named emotion to a ``FloatEmotion`` via
    :meth:`~emotion_algebra.float_emotion.FloatEmotion.from_emotion`.

    Parameters
    ----------
    need_name:
        A Max-Neef or Murray need name.

    Returns
    -------
    FloatEmotion or None
        Continuous 4-axis vector, or ``None`` if unknown need.

    Examples
    --------
    >>> fe = need_deficit_to_float_emotion("protection")
    >>> float(fe.as_array[0]) < 0  # fear → negative sensitivity
    True
    """
    from emotion_algebra.float_emotion import FloatEmotion
    emo = need_deficit_to_emotion(need_name)
    if emo is None:
        return None
    return FloatEmotion.from_emotion(emo)
