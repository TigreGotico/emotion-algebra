"""Cognitive appraisal → primary emotion mapping — v2.0.

Implements Scherer's Component Process Model (2001) in two modes:

1. **Discrete** (v1 compat): categorical ``Appraisal`` fields with
   :func:`appraisal_to_emotion` (16-rule first-match table → named Emotion).
2. **Continuous** (v2): float-valued ``Appraisal`` fields with
   :func:`appraisal_to_float_emotion` (Scherer SEC → Hourglass 4D mapping
   → :class:`~emotion_algebra.float_emotion.FloatEmotion`).

The continuous mode also provides :func:`float_emotion_to_neuro_deltas` which
maps a ``FloatEmotion`` to (dopamine, serotonin, adrenaline) deltas — the
inverse of the Lövheim-inspired mapping used in LILACS's
``NeuroRepository.to_emotion_vector()``.

References
----------
Scherer, K. R. (2001). Appraisal considered as a process of multilevel
sequential checking.  In K. R. Scherer, A. Schorr, & T. Johnstone (Eds.),
*Appraisal processes in emotion: Theory, methods, research* (pp. 92–120).
Oxford University Press.

Cambria, E. et al. (2012). The Hourglass of Emotions.  *Cognitive
Behavioural Systems*, Springer.

Lövheim, H. (2012). A new three-dimensional model for emotions and
monoamine neurotransmitters.  *Medical Hypotheses*, 78(2), 341–348.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from emotion_algebra.base import EmotionBase

NoveltyT     = Literal["expected", "unexpected"]
RelevanceT   = Literal["relevant", "irrelevant"]
CongruenceT  = Literal["congruent", "incongruent"]
AgencyT      = Literal["self", "other", "circumstance"]
CopingT      = Literal["high", "low"]
PleasantnessT = Literal["pleasant", "unpleasant"]

# Mapping from categorical values to floats for continuous appraisal
_CATEGORICAL_TO_FLOAT: dict[str | None, float] = {
    # Novelty
    "unexpected": 1.0, "expected": 0.0,
    # Relevance
    "relevant": 1.0, "irrelevant": 0.0,
    # Congruence
    "congruent": 1.0, "incongruent": 0.0,
    # Agency
    "self": 1.0, "other": 0.5, "circumstance": 0.0,
    # Coping
    "high": 1.0, "low": 0.0,
    # Intrinsic pleasantness
    "pleasant": 1.0, "unpleasant": 0.0,
    # None = neutral / unknown
    None: 0.5,
}


@dataclass
class Appraisal:
    """Structured cognitive evaluation of an event.

    All fields are optional; ``None`` means "unknown" or "not applicable".
    Fields accept either categorical string values (for discrete rule matching)
    or float values in ``[0.0, 1.0]`` (for continuous appraisal).

    Attributes
    ----------
    novelty:
        Whether the event was expected (0.0) or surprising (1.0).
    goal_relevance:
        Whether the event bears on the subject's active goals (0.0–1.0).
    goal_congruence:
        Whether the event helps (1.0) or hinders (0.0) the goal.
    agency:
        Who is responsible — ``"self"`` (1.0), ``"other"`` (0.5),
        or ``"circumstance"`` (0.0).
    coping_potential:
        Whether the subject feels able to deal with the event (0.0–1.0).
    intrinsic_pleasantness:
        Hedonic tone of the stimulus itself, independent of goal
        congruence (Scherer's 2nd SEC check).
    """

    novelty: Optional[Union[NoveltyT, float]] = None
    goal_relevance: Optional[Union[RelevanceT, float]] = None
    goal_congruence: Optional[Union[CongruenceT, float]] = None
    agency: Optional[Union[AgencyT, float]] = None
    coping_potential: Optional[Union[CopingT, float]] = None
    intrinsic_pleasantness: Optional[Union[PleasantnessT, float]] = None

    def to_float(self) -> _FloatAppraisal:
        """Return a normalised copy with all fields as floats in [0, 1].

        Categorical values are mapped via the Scherer scale:
        ``"unexpected"`` → 1.0, ``"expected"`` → 0.0, ``None`` → 0.5, etc.
        Float values are passed through unchanged (clamped to [0, 1]).
        """
        def _convert(val: object) -> float:
            if val is None:
                return 0.5
            if isinstance(val, (int, float)):
                return max(0.0, min(1.0, float(val)))
            return _CATEGORICAL_TO_FLOAT.get(val, 0.5)

        return _FloatAppraisal(
            novelty=_convert(self.novelty),
            goal_relevance=_convert(self.goal_relevance),
            goal_congruence=_convert(self.goal_congruence),
            agency=_convert(self.agency),
            coping_potential=_convert(self.coping_potential),
            intrinsic_pleasantness=_convert(self.intrinsic_pleasantness),
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "Appraisal":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class _FloatAppraisal:
    """Internal: all-float version of Appraisal for continuous computation."""
    novelty: float = 0.5
    goal_relevance: float = 0.5
    goal_congruence: float = 0.5
    agency: float = 0.5
    coping_potential: float = 0.5
    intrinsic_pleasantness: float = 0.5


# ---------------------------------------------------------------------------
# Discrete rule table — backwards compat (v1)
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

    This is the **discrete** (v1) mapping — use
    :func:`appraisal_to_float_emotion` for the **continuous** (v2) mapping.

    Parameters
    ----------
    appraisal:
        A filled-in :class:`Appraisal` instance with categorical values.

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


# ---------------------------------------------------------------------------
# Continuous appraisal → FloatEmotion (v2)
# Scherer SEC checks → Cambria Hourglass axes
# ---------------------------------------------------------------------------

def appraisal_to_float_emotion(appraisal: Appraisal) -> "FloatEmotion":
    """Map a continuous :class:`Appraisal` to a 4-axis :class:`FloatEmotion`.

    Uses the theoretical mapping from Scherer's Sequential Evaluation Checks
    (SEC) to Cambria's Hourglass dimensions:

    * **Sensitivity** (anger ↔ fear): threat salience — high when goal is
      blocked (low congruence) and agent cannot cope (low coping potential).
    * **Attention** (vigilance ↔ surprise): novelty-driven engagement —
      high when the stimulus is unexpected and goal-relevant.
    * **Pleasantness** (joy ↔ sadness): hedonic valence — driven by goal
      congruence weighted by relevance, blended with intrinsic pleasantness.
    * **Aptitude** (trust ↔ disgust): competence/alignment — high when
      the agent can cope and the event is goal-congruent.

    All appraisal fields are normalised to ``[0, 1]`` via :meth:`Appraisal.to_float`
    before computation.  Output axes are centred at 0 and scaled to ``[-1, +1]``.

    Parameters
    ----------
    appraisal:
        An :class:`Appraisal` instance (categorical or float values).

    Returns
    -------
    FloatEmotion
        A continuous 4-axis emotion vector.

    Examples
    --------
    >>> a = Appraisal(novelty=0.8, goal_relevance=0.9, goal_congruence=0.7,
    ...               intrinsic_pleasantness=0.6)
    >>> fe = appraisal_to_float_emotion(a)
    >>> fe.pleasantness > 0  # congruent + pleasant → positive
    True
    """
    from emotion_algebra.float_emotion import FloatEmotion

    a = appraisal.to_float()

    # Scherer SEC → Hourglass axes
    # 1. Sensitivity: threat without coping capacity
    #    High when: relevant + incongruent + can't cope
    sensitivity = (1.0 - a.coping_potential) * a.goal_relevance * (1.0 - a.goal_congruence)

    # 2. Attention: novelty-driven engagement
    #    High when: unexpected + relevant
    attention = a.novelty * 0.6 + a.goal_relevance * 0.4

    # 3. Pleasantness: hedonic valence
    #    Goal congruence (weighted by relevance) + intrinsic pleasantness
    pleasantness = a.goal_congruence * a.goal_relevance * 0.7 + a.intrinsic_pleasantness * 0.3

    # 4. Aptitude: competence + alignment
    #    High when: can cope + goal-congruent
    aptitude = a.coping_potential * 0.6 + a.goal_congruence * 0.4

    # Centre at 0 and scale to [-1, +1]
    return FloatEmotion(
        sensitivity=(sensitivity - 0.5) * 2.0,
        attention=(attention - 0.5) * 2.0,
        pleasantness=(pleasantness - 0.5) * 2.0,
        aptitude=(aptitude - 0.5) * 2.0,
    )


# ---------------------------------------------------------------------------
# FloatEmotion → neurotransmitter deltas
# Inverse of the Lövheim-inspired mapping in NeuroRepository.to_emotion_vector()
# ---------------------------------------------------------------------------

def float_emotion_to_neuro_deltas(
    fe: "FloatEmotion",
    scale: float = 0.15,
) -> tuple[float, float, float]:
    """Convert a :class:`FloatEmotion` to neurotransmitter deltas.

    This is the inverse of the Lövheim-inspired mapping:

    * Dopamine  ← (sensitivity + attention) / 4 — arousal / salience
    * Serotonin ← positive valence — (pleasantness + aptitude) / 4
    * Adrenaline ← negative valence — when pleasantness + aptitude < 0

    The ``scale`` parameter controls how strongly a single appraisal event
    influences neurotransmitter levels.  Default ``0.15`` keeps individual
    turns from swamping the baseline.

    Parameters
    ----------
    fe:
        A :class:`~emotion_algebra.float_emotion.FloatEmotion`.
    scale:
        Scaling factor applied to all deltas (default ``0.15``).

    Returns
    -------
    tuple[float, float, float]
        ``(dopamine_delta, serotonin_delta, adrenaline_delta)``.

    Examples
    --------
    >>> from emotion_algebra import FloatEmotion
    >>> fe = FloatEmotion(sensitivity=0.5, attention=0.5, pleasantness=0.3, aptitude=0.3)
    >>> d, s, a = float_emotion_to_neuro_deltas(fe)
    >>> d > 0  # high arousal → dopamine
    True
    >>> s > 0  # positive valence → serotonin
    True
    """
    vec = fe.as_array  # [sensitivity, attention, pleasantness, aptitude]

    # Arousal → dopamine: attention is the primary driver (novelty/salience),
    # sensitivity contributes but can be negative (no threat = no arousal penalty)
    dopamine_delta = float(vec[1]) / 2.0 * scale  # attention-driven

    # Valence → serotonin (positive) or adrenaline (negative)
    net_valence = (float(vec[2]) + float(vec[3])) / 4.0 * scale
    serotonin_delta = max(0.0, net_valence)
    adrenaline_delta = max(0.0, -net_valence)

    return (dopamine_delta, serotonin_delta, adrenaline_delta)
