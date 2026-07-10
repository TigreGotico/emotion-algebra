"""Need-to-emotion mapping — Max-Neef (1991) + Murray (1938) → Plutchik.

Maps fundamental human needs to the primary emotions that arise when those
needs are deficient.  Grounded in:

* **Max-Neef (1991)** — 9 fundamental needs: subsistence, protection, freedom,
  identity, participation, creation, understanding, idleness, affection.
* **Murray (1938)** — 17 psychogenic needs: achievement, affiliation, aggression,
  autonomy, counteraction, defendance, deference, dominance, exhibition,
  harm_avoidance, infavoidance, nurturance, order, play, rejection, sentience,
  understanding.
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

import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport for Python < 3.11.

        Plain ``Enum`` formats ``str(member)`` as ``"ClassName.MEMBER"``;
        the real 3.11+ ``enum.StrEnum`` overrides this to return the value.
        Match that behavior so ``str(member)`` is version-portable — this
        module relies on it to build ``NEED_DEFICIT_EMOTIONS``.
        """

        def __str__(self) -> str:
            return str(self.value)
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emotion_algebra.base import EmotionBase
    from emotion_algebra.float_emotion import FloatEmotion


# ---------------------------------------------------------------------------
# CIA meta-drives (Control / Identity / Arousal)
# ---------------------------------------------------------------------------

class CIADrive(StrEnum):
    """Three meta-drives from the CIA framework."""
    CONTROL = "control"
    IDENTITY = "identity"
    AROUSAL = "arousal"


# ---------------------------------------------------------------------------
# Max-Neef 9 fundamental needs
# ---------------------------------------------------------------------------

class MaxNeefNeed(StrEnum):
    """Max-Neef's (1991) nine fundamental human needs."""
    SUBSISTENCE = "subsistence"
    PROTECTION = "protection"
    FREEDOM = "freedom"
    IDENTITY = "identity"
    PARTICIPATION = "participation"
    CREATION = "creation"
    UNDERSTANDING = "understanding"
    IDLENESS = "idleness"
    AFFECTION = "affection"


# ---------------------------------------------------------------------------
# Murray 17 psychogenic needs
# ---------------------------------------------------------------------------

class MurrayNeed(StrEnum):
    """Murray's (1938) seventeen psychogenic needs."""
    ACHIEVEMENT = "achievement"
    AFFILIATION = "affiliation"
    AGGRESSION = "aggression"
    AUTONOMY = "autonomy"
    COUNTERACTION = "counteraction"
    DEFENDANCE = "defendance"
    DEFERENCE = "deference"
    DOMINANCE = "dominance"
    EXHIBITION = "exhibition"
    HARM_AVOIDANCE = "harm_avoidance"
    INFAVOIDANCE = "infavoidance"
    NURTURANCE = "nurturance"
    ORDER = "order"
    PLAY = "play"
    REJECTION = "rejection"
    SENTIENCE = "sentience"
    UNDERSTANDING = "understanding"


# ---------------------------------------------------------------------------
# Need → deficit emotion mappings
# ---------------------------------------------------------------------------

MAXNEEF_DEFICIT_EMOTIONS: dict[MaxNeefNeed, str] = {
    # Control needs — deficiency threatens safety/autonomy
    MaxNeefNeed.SUBSISTENCE:   "sadness",       # deprivation → grief/loss
    MaxNeefNeed.PROTECTION:    "fear",           # vulnerability → threat
    MaxNeefNeed.FREEDOM:       "anger",          # constraint → obstacle

    # Identity needs — deficiency threatens self-concept/belonging
    MaxNeefNeed.IDENTITY:      "pensiveness",    # weakened self → melancholy
    MaxNeefNeed.PARTICIPATION: "apprehension",   # exclusion → uncertainty
    MaxNeefNeed.CREATION:      "boredom",        # blocked creativity → disengagement
    MaxNeefNeed.UNDERSTANDING: "distraction",    # confusion → disorientation

    # Arousal needs — deficiency threatens stimulation balance
    MaxNeefNeed.IDLENESS:      "annoyance",      # over-stimulation → irritability
    MaxNeefNeed.AFFECTION:     "sadness",         # loneliness → loss/grief
}

MURRAY_DEFICIT_EMOTIONS: dict[MurrayNeed, str] = {
    MurrayNeed.ACHIEVEMENT:    "annoyance",      # blocked accomplishment → frustration
    MurrayNeed.AFFILIATION:    "sadness",         # isolation → loneliness
    MurrayNeed.AGGRESSION:     "anger",           # impotence → rage
    MurrayNeed.AUTONOMY:       "anger",           # constraint → obstacle
    MurrayNeed.COUNTERACTION:  "annoyance",       # weakness → frustrated determination
    MurrayNeed.DEFENDANCE:     "fear",            # criticism → threat
    MurrayNeed.DEFERENCE:      "apprehension",    # no guidance → uncertainty
    MurrayNeed.DOMINANCE:      "annoyance",       # loss of control → frustration
    MurrayNeed.EXHIBITION:     "pensiveness",     # invisibility → melancholy
    MurrayNeed.HARM_AVOIDANCE: "fear",            # danger → threat
    MurrayNeed.INFAVOIDANCE:   "apprehension",    # humiliation risk → anxiety
    MurrayNeed.NURTURANCE:     "sadness",         # can't help → grief
    MurrayNeed.ORDER:          "boredom",          # chaos → disengagement
    MurrayNeed.PLAY:           "boredom",          # no fun → dullness
    MurrayNeed.REJECTION:      "disgust",          # contamination → revulsion
    MurrayNeed.SENTIENCE:      "boredom",          # sensory deprivation → dullness
    MurrayNeed.UNDERSTANDING:  "distraction",      # confusion → disorientation
}

# Combined mapping (StrEnum values work as plain strings)
NEED_DEFICIT_EMOTIONS: dict[str, str] = {
    **{str(k): v for k, v in MAXNEEF_DEFICIT_EMOTIONS.items()},
    **{str(k): v for k, v in MURRAY_DEFICIT_EMOTIONS.items()},
}


def need_deficit_to_emotion(need_name: str) -> Optional["EmotionBase"]:
    """Return the Plutchik emotion that arises when *need_name* is deficient.

    Parameters
    ----------
    need_name:
        A Max-Neef or Murray need name (e.g. ``"subsistence"``, ``"sentience"``).
        Accepts both :class:`MaxNeefNeed`/:class:`MurrayNeed` enum values
        and plain strings.

    Returns
    -------
    Emotion or None
        The primary emotion, or ``None`` if the need name is unknown.

    Examples
    --------
    >>> need_deficit_to_emotion(MaxNeefNeed.PROTECTION).name
    'fear'
    >>> need_deficit_to_emotion("creation").name
    'boredom'
    """
    from emotion_algebra.emotions import get_emotion
    emotion_name = NEED_DEFICIT_EMOTIONS.get(str(need_name))
    return get_emotion(emotion_name) if emotion_name else None


def need_deficit_to_float_emotion(need_name: str) -> Optional["FloatEmotion"]:
    """Return a continuous :class:`FloatEmotion` for a deficient need.

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
    >>> fe = need_deficit_to_float_emotion(MaxNeefNeed.PROTECTION)
    >>> float(fe.as_array[0]) < 0  # fear → negative sensitivity
    True
    """
    from emotion_algebra.float_emotion import FloatEmotion
    emo = need_deficit_to_emotion(need_name)
    if emo is None:
        return None
    return FloatEmotion.from_emotion(emo)
