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
(Scherer's "goal incongruence"), and *which* emotion results depends on which
CIA meta-drive — Control, Identity, or Arousal — the deficit threatens.  Each
drive can only produce emotions on certain Hourglass axes:

===========  ==========================  ==================================================
Drive        Hourglass axes it produces  Why
===========  ==========================  ==================================================
CONTROL      Sensitivity                 Safety, autonomy and mastery are threatened.  The
                                         coping check decides the pole: obstruction you can
                                         fight → annoyance/anger/rage; obstruction you
                                         cannot → apprehension/fear/terror.
IDENTITY     Pleasantness, Aptitude      Self-concept, worth and belonging are threatened →
                                         hedonic collapse (pensiveness/sadness/grief) or
                                         self-rejection (disgust/loathing).
AROUSAL      Attention, Aptitude         Stimulation is out of balance → disorientation
                                         (distraction/surprise) or disengagement (boredom).
===========  ==========================  ==================================================

Aptitude appears under two drives on purpose: it carries both *worth* (an
identity concern) and *engagement* (an arousal concern), and the deficit tables
use it for both.

:data:`NEED_DRIVES` records each need's drive, and
``test/test_needs.py::TestCIACoherence`` asserts every row of
:data:`MAXNEEF_DEFICIT_EMOTIONS` and :data:`MURRAY_DEFICIT_EMOTIONS` produces an
emotion on an axis its drive is allowed to produce.  The table and this
docstring therefore cannot drift apart.

References
----------
Max-Neef, M. (1991). *Human Scale Development*.
Murray, H. A. (1938). *Explorations in Personality*.
Plutchik, R. (1980). *Emotion: A Psychoevolutionary Synthesis*.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from emotion_algebra._compat import StrEnum

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

#: Hourglass axes each CIA drive is allowed to produce a deficit emotion on.
#: This is the machine-readable form of the table in the module docstring.
DRIVE_AXES: dict[CIADrive, frozenset] = {
    CIADrive.CONTROL:  frozenset({"sensitivity"}),
    CIADrive.IDENTITY: frozenset({"pleasantness", "aptitude"}),
    CIADrive.AROUSAL:  frozenset({"attention", "aptitude"}),
}

#: Which meta-drive each need belongs to.  A need is filed by *what the deficit
#: threatens*, not by which emotion it happens to produce — the emotion follows.
NEED_DRIVES: dict[str, CIADrive] = {
    # --- Max-Neef ---
    "subsistence":   CIADrive.IDENTITY,   # deprivation is felt as loss, not as acute threat
    "protection":    CIADrive.CONTROL,    # the acute-threat need proper
    "freedom":       CIADrive.CONTROL,    # constraint on the ability to act
    "identity":      CIADrive.IDENTITY,
    "participation": CIADrive.CONTROL,    # exclusion removes the ability to act through the group
    "creation":      CIADrive.AROUSAL,
    "understanding": CIADrive.AROUSAL,
    "idleness":      CIADrive.CONTROL,    # rest denied = an imposition to be resisted
    "affection":     CIADrive.IDENTITY,
    # --- Murray ---
    "achievement":    CIADrive.CONTROL,
    "affiliation":    CIADrive.IDENTITY,
    "aggression":     CIADrive.CONTROL,
    "autonomy":       CIADrive.CONTROL,
    "counteraction":  CIADrive.CONTROL,
    "defendance":     CIADrive.CONTROL,
    "deference":      CIADrive.CONTROL,
    "dominance":      CIADrive.CONTROL,
    "exhibition":     CIADrive.IDENTITY,
    "harm_avoidance": CIADrive.CONTROL,
    "infavoidance":   CIADrive.CONTROL,
    "nurturance":     CIADrive.IDENTITY,
    "order":          CIADrive.CONTROL,   # disorder threatens predictability
    "play":           CIADrive.AROUSAL,
    "rejection":      CIADrive.IDENTITY,
    "sentience":      CIADrive.AROUSAL,
    "understanding":  CIADrive.AROUSAL,
}

MAXNEEF_DEFICIT_EMOTIONS: dict[MaxNeefNeed, str] = {
    MaxNeefNeed.SUBSISTENCE:   "sadness",        # identity — deprivation → grief/loss
    MaxNeefNeed.PROTECTION:    "fear",           # control  — vulnerability → threat
    MaxNeefNeed.FREEDOM:       "anger",          # control  — constraint → obstacle
    MaxNeefNeed.IDENTITY:      "pensiveness",    # identity — weakened self → melancholy
    MaxNeefNeed.PARTICIPATION: "apprehension",   # control  — exclusion → social threat
    MaxNeefNeed.CREATION:      "boredom",        # arousal  — blocked creativity → disengagement
    MaxNeefNeed.UNDERSTANDING: "distraction",    # arousal  — confusion → disorientation
    MaxNeefNeed.IDLENESS:      "annoyance",      # control  — no rest → irritability at the imposition
    MaxNeefNeed.AFFECTION:     "sadness",        # identity — loneliness → loss/grief
}

MURRAY_DEFICIT_EMOTIONS: dict[MurrayNeed, str] = {
    MurrayNeed.ACHIEVEMENT:    "annoyance",      # control  — blocked accomplishment → frustration
    MurrayNeed.AFFILIATION:    "sadness",        # identity — isolation → loneliness
    MurrayNeed.AGGRESSION:     "anger",          # control  — impotence → rage
    MurrayNeed.AUTONOMY:       "anger",          # control  — constraint → obstacle
    MurrayNeed.COUNTERACTION:  "annoyance",      # control  — weakness → frustrated determination
    MurrayNeed.DEFENDANCE:     "fear",           # control  — criticism → threat
    MurrayNeed.DEFERENCE:      "apprehension",   # control  — no guidance → uncertainty
    MurrayNeed.DOMINANCE:      "annoyance",      # control  — loss of control → frustration
    MurrayNeed.EXHIBITION:     "pensiveness",    # identity — invisibility → melancholy
    MurrayNeed.HARM_AVOIDANCE: "fear",           # control  — danger → threat
    MurrayNeed.INFAVOIDANCE:   "apprehension",   # control  — humiliation risk → anxiety
    MurrayNeed.NURTURANCE:     "sadness",        # identity — can't help → grief
    # Chaos is a threat to predictability and control, so it belongs on the
    # Sensitivity axis. It previously mapped to "boredom" (disengagement),
    # which contradicted both its own CIA drive and the plain psychology —
    # a person deprived of order is anxious, not bored.
    MurrayNeed.ORDER:          "apprehension",   # control  — disorder → anxiety
    MurrayNeed.PLAY:           "boredom",        # arousal  — no fun → dullness
    MurrayNeed.REJECTION:      "disgust",        # identity — contamination → revulsion
    MurrayNeed.SENTIENCE:      "boredom",        # arousal  — sensory deprivation → dullness
    MurrayNeed.UNDERSTANDING:  "distraction",    # arousal  — confusion → disorientation
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
