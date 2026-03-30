"""Cognitive appraisal → primary emotion mapping — v1.5.

Implements a simplified, discrete version of Scherer's Component Process
Model (2001).  An :class:`Appraisal` describes how a subject evaluates an
event along five dimensions; :func:`appraisal_to_emotion` maps that pattern
to a primary :class:`~emotion_algebra.plutchik.Emotion`.

**Limitations**: Scherer's CPM defines graded, continuous appraisal checks
with probabilistic outcomes.  This module uses discrete categorical values and
returns a single best-fit emotion (first matching rule wins).  Treat the result
as a practical approximation, not a rigorous model.

References
----------
Scherer, K. R. (2001). Appraisal considered as a process of multilevel
sequential checking.  In K. R. Scherer, A. Schorr, & T. Johnstone (Eds.),
*Appraisal processes in emotion: Theory, methods, research* (pp. 92–120).
Oxford University Press.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from emotion_algebra.base import EmotionBase

NoveltyT     = Literal["expected", "unexpected"]
RelevanceT   = Literal["relevant", "irrelevant"]
CongruenceT  = Literal["congruent", "incongruent"]
AgencyT      = Literal["self", "other", "circumstance"]
CopingT      = Literal["high", "low"]


@dataclass
class Appraisal:
    """Structured cognitive evaluation of an event.

    All fields are optional; ``None`` means "unknown" or "not applicable".

    Attributes
    ----------
    novelty:
        Whether the event was expected or surprising.
    goal_relevance:
        Whether the event bears on the subject's active goals.
    goal_congruence:
        Whether the event helps (``"congruent"``) or hinders
        (``"incongruent"``) the goal.
    agency:
        Who is responsible — ``"self"``, ``"other"``, or ``"circumstance"``.
    coping_potential:
        Whether the subject feels able to deal with the event.
    """

    novelty: Optional[NoveltyT] = None
    goal_relevance: Optional[RelevanceT] = None
    goal_congruence: Optional[CongruenceT] = None
    agency: Optional[AgencyT] = None
    coping_potential: Optional[CopingT] = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "novelty": self.novelty,
            "goal_relevance": self.goal_relevance,
            "goal_congruence": self.goal_congruence,
            "agency": self.agency,
            "coping_potential": self.coping_potential,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Appraisal":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            novelty=data.get("novelty"),
            goal_relevance=data.get("goal_relevance"),
            goal_congruence=data.get("goal_congruence"),
            agency=data.get("agency"),
            coping_potential=data.get("coping_potential"),
        )


# ---------------------------------------------------------------------------
# Rule table — (novelty, relevance, congruence, agency, coping) → emotion
# None = wildcard.  Rules checked top-to-bottom; first match wins.
# Derived from Scherer (2001), Table 1 (simplified / discretised).
# ---------------------------------------------------------------------------

_RULES = [
    # novelty         relevance      congruence      agency           coping    emotion
    ("unexpected",    "irrelevant",  None,           None,            None,     "surprise"),
    (None,            "irrelevant",  None,           None,            None,     "boredom"),
    ("unexpected",    "relevant",    "congruent",    None,            None,     "joy"),
    (None,            "relevant",    "congruent",    "self",          None,     "joy"),
    (None,            "relevant",    "congruent",    "other",         None,     "trust"),
    (None,            "relevant",    "congruent",    "circumstance",  None,     "serenity"),
    (None,            "relevant",    "incongruent",  "other",         "high",   "anger"),
    (None,            "relevant",    "incongruent",  "other",         "low",    "fear"),
    (None,            "relevant",    "incongruent",  "self",          "high",   "disgust"),
    (None,            "relevant",    "incongruent",  "self",          "low",    "sadness"),
    (None,            "relevant",    "incongruent",  "circumstance",  "high",   "anticipation"),
    (None,            "relevant",    "incongruent",  "circumstance",  "low",    "sadness"),
    ("unexpected",    None,          None,           None,            None,     "surprise"),
    (None,            None,          "congruent",    None,            None,     "joy"),
    (None,            None,          "incongruent",  None,            "high",   "anger"),
    (None,            None,          "incongruent",  None,            "low",    "fear"),
]

_FIELDS = ("novelty", "goal_relevance", "goal_congruence", "agency", "coping_potential")


def _matches(rule: tuple, appraisal: Appraisal) -> bool:
    for rule_val, field in zip(rule[:5], _FIELDS):
        if rule_val is not None and getattr(appraisal, field) != rule_val:
            return False
    return True


def appraisal_to_emotion(appraisal: Appraisal) -> "EmotionBase":
    """Map a cognitive :class:`Appraisal` to the best-fit primary emotion.

    Checks rules top-to-bottom (first match wins).  Returns
    :class:`~emotion_algebra.plutchik.Neutrality` if no rule matches.

    Parameters
    ----------
    appraisal:
        A filled-in :class:`Appraisal` instance.

    Returns
    -------
    Emotion
        A named primary emotion, or :class:`~emotion_algebra.plutchik.Neutrality`.

    Examples
    --------
    >>> a = Appraisal(goal_relevance="relevant", goal_congruence="incongruent",
    ...               agency="other", coping_potential="high")
    >>> appraisal_to_emotion(a).name
    'anger'
    """
    from emotion_algebra.emotions import get_emotion
    from emotion_algebra.plutchik import Neutrality
    for rule in _RULES:
        if _matches(rule, appraisal):
            emo = get_emotion(rule[5])
            return emo if emo is not None else Neutrality()
    return Neutrality()
