"""Cognitive appraisal → primary emotion mapping — v2.0.

Implements Scherer's Component Process Model (2001) in two modes:

1. **Discrete** (v1 compat): categorical ``Appraisal`` fields with
   :func:`appraisal_to_emotion` (16-rule first-match table → named Emotion).
2. **Continuous** (v2): float-valued ``Appraisal`` fields with
   :func:`appraisal_to_float_emotion` (Scherer SEC → Hourglass 4D mapping
   → :class:`~emotion_algebra.float_emotion.FloatEmotion`).

The continuous mode also provides :func:`float_emotion_to_neuro_deltas` which
maps a ``FloatEmotion`` to (dopamine, serotonin, adrenaline) deltas — the
inverse of the Lövheim-inspired mapping used by downstream
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

import warnings
from dataclasses import dataclass, fields
from typing import Optional, Union, Literal, TYPE_CHECKING

from emotion_algebra._compat import StrEnum


def _removal_version() -> str:
    """The version a thing deprecated *now* will be removed in: the next major.

    Computed from ``version.py`` so it can never go stale — never hardcode a
    version in a deprecation message.
    """
    from emotion_algebra.version import VERSION_MAJOR
    return f"{VERSION_MAJOR + 1}.0.0"

if TYPE_CHECKING:
    from emotion_algebra.base import EmotionBase
    from emotion_algebra.float_emotion import FloatEmotion
    from emotion_algebra.lovheim import LovheimPoint

NoveltyT     = Literal["expected", "unexpected"]
RelevanceT   = Literal["relevant", "irrelevant"]
CongruenceT  = Literal["congruent", "incongruent"]
AgencyT      = Literal["self", "other", "circumstance"]
CopingT      = Literal["high", "low"]
PleasantnessT = Literal["pleasant", "unpleasant"]

# Per-field categorical → float mappings (validates cross-field misuse)
_FIELD_CATEGORICAL: dict[str, dict[str, float]] = {
    "novelty":                {"unexpected": 1.0, "expected": 0.0},
    "goal_relevance":         {"relevant": 1.0, "irrelevant": 0.0},
    "goal_congruence":        {"congruent": 1.0, "incongruent": 0.0},
    "agency":                 {"self": 1.0, "other": 0.5, "circumstance": 0.0},
    "coping_potential":       {"high": 1.0, "low": 0.0},
    "intrinsic_pleasantness": {"pleasant": 1.0, "unpleasant": 0.0},
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

        Categorical values are mapped via per-field Scherer scales:
        ``"unexpected"`` → 1.0, ``"expected"`` → 0.0, ``None`` → 0.5, etc.
        Float values are passed through unchanged (clamped to [0, 1]).

        Raises
        ------
        ValueError
            If a string value is not valid for its field (e.g. ``agency="low"``).
        """
        def _convert(field_name: str, val: object) -> float:
            if val is None:
                return 0.5
            if isinstance(val, (int, float)):
                return max(0.0, min(1.0, float(val)))
            allowed = _FIELD_CATEGORICAL.get(field_name, {})
            if val not in allowed:
                raise ValueError(
                    f"Invalid value {val!r} for field {field_name!r}; "
                    f"expected one of {set(allowed)} or a float"
                )
            return allowed[val]

        return _FloatAppraisal(
            novelty=_convert("novelty", self.novelty),
            goal_relevance=_convert("goal_relevance", self.goal_relevance),
            goal_congruence=_convert("goal_congruence", self.goal_congruence),
            agency=_convert("agency", self.agency),
            coping_potential=_convert("coping_potential", self.coping_potential),
            intrinsic_pleasantness=_convert("intrinsic_pleasantness", self.intrinsic_pleasantness),
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

#: Coefficients of the SEC → Hourglass mapping, as **data** rather than magic
#: numbers buried in the expression.  Each entry names what it weighs and says
#: where it comes from: a Scherer prediction, or an explicit calibration
#: against the modal-emotion fixtures in ``test/test_appraisal.py``.
#:
#: Scherer's Component Process Model predicts the *direction* and *relative
#: ordering* of appraisal effects, not their numeric magnitudes — the CPM is a
#: qualitative theory.  Anything below marked ``calibrated`` is therefore a
#: fitted magnitude whose only job is to reproduce the orderings the theory does
#: predict; it is not a claim that Scherer published this number.  Tune these,
#: not the expression.
APPRAISAL_COEFFS: dict[str, float] = {
    # --- Sensitivity (anger ↔ fear) ---
    # Scherer 2001: high coping power → anger; low coping power → fear, given
    # the same obstruction.  Coping therefore sets the sign of this axis.
    "sensitivity_coping_gain": 2.0,        # calibrated: saturates the axis at maximal obstruction + full coping
    # Obstruction is aversive even with no coping belief either way, so a
    # maximal threat at neutral coping still lands on the fear pole rather than
    # at exactly zero.  Without this the axis has a hole at cp == 0.
    "sensitivity_threat_floor": 0.5,       # calibrated: fear at neutral coping, without swamping the coping sign

    # --- Attention (vigilance/interest ↔ surprise/amazement) ---
    "attention_relevance_gain": 0.4,           # calibrated: relevance alone is mild engagement
    "attention_novelty_relevance_gain": 1.2,   # calibrated: novelty × relevance → interest (the seeking response)
    "attention_surprise_bias": 0.3,            # calibrated: novelty without relevance → bare surprise

    # --- Pleasantness (joy ↔ sadness) ---
    # Scherer's SEC 2 (intrinsic pleasantness) is a separate check from SEC 3
    # (goal conduciveness), so they enter as separate terms rather than one.
    "pleasantness_congruence_gain": 0.7,   # calibrated: goal conduciveness dominates hedonic tone
    "pleasantness_intrinsic_gain": 0.3,    # calibrated: intrinsic pleasantness contributes even off-goal

    # --- Aptitude (trust ↔ disgust) ---
    "aptitude_coping_gain": 0.5,           # calibrated: felt competence is the largest contributor
    "aptitude_congruence_gain": 0.3,       # calibrated: a congruent world is a trustworthy one
    "aptitude_agency_gain": 0.2,           # calibrated: self-agency reads as competence
}


class Agency(StrEnum):
    """Who is responsible for the event — Scherer's causal-attribution check.

    The float values are the ones :meth:`Appraisal.to_float` maps to, so
    ``Appraisal(agency=Agency.OTHER)`` and ``Appraisal(agency="other")`` are
    the same appraisal.  Use the enum: it is the difference between an
    attribution the caller chose and one that was hard-coded by accident.

    Examples
    --------
    >>> a = Appraisal(goal_relevance="relevant", goal_congruence="incongruent",
    ...               agency=Agency.OTHER, coping_potential="high")
    >>> appraisal_to_emotion(a).name
    'anger'
    """

    SELF = "self"
    OTHER = "other"
    CIRCUMSTANCE = "circumstance"

    @property
    def as_float(self) -> float:
        """The ``[0, 1]`` value this attribution contributes to the Aptitude axis."""
        return _FIELD_CATEGORICAL["agency"][self.value]


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

    # Centre inputs at 0 (signed space) so neutral appraisals → zero vector
    cp  = a.coping_potential - 0.5       # >0 = can cope, <0 = can't
    gr  = a.goal_relevance - 0.5         # >0 = relevant
    gc  = a.goal_congruence - 0.5        # >0 = congruent, <0 = incongruent
    nov = a.novelty - 0.5                # >0 = unexpected
    ip  = a.intrinsic_pleasantness - 0.5 # >0 = pleasant
    ag  = a.agency - 0.5                 # >0 = self, 0 = other, <0 = circumstance

    # Obstruction: how blocked the goal is, scaled by how much it matters.
    # Centred inputs mean gr ∈ [-0.5, 0.5], so (gr + 0.5) ∈ [0, 1] reads as
    # "relevance weight" — an irrelevant goal cannot be obstructive at all.
    relevance_weight = max(0.0, gr + 0.5)
    obstruction = max(0.0, -gc) * relevance_weight

    C = APPRAISAL_COEFFS

    # 1. Sensitivity (positive = anger, negative = fear/terror).
    #    Scherer 2001: the anger/fear split is decided by the *coping* check —
    #    an obstructed goal you believe you can master produces anger, the same
    #    obstruction without that belief produces fear.  So coping supplies the
    #    SIGN, and obstruction supplies the MAGNITUDE.
    #
    #    The magnitude term is additive rather than a bare product with cp.
    #    A pure product is zero whenever coping is exactly neutral (cp = 0),
    #    which would mean a maximal, unmasterable threat appraised with neutral
    #    coping yields *no fear at all* — a hole with no basis in the CPM.
    #    THREAT_FLOOR keeps obstruction alone aversive: with no coping belief
    #    to lean on, an obstructed goal defaults toward the fear pole.
    sensitivity = (
        obstruction * cp * C["sensitivity_coping_gain"]
        - obstruction * C["sensitivity_threat_floor"]
    )

    # 2. Attention (positive = vigilance/interest, negative = surprise/amazement):
    #    Novelty drives engagement; goal-relevance steers its sign. A novel
    #    *and* relevant stimulus is curiosity/interest (positive pole, the
    #    seeking/vigilance response); novelty without relevance is bare surprise
    #    (negative pole). gr is centred in [-0.5, +0.5], so the novelty term
    #    flips from surprise to interest as relevance climbs past neutral.
    attention = (
        gr * C["attention_relevance_gain"]
        + nov * (gr * C["attention_novelty_relevance_gain"] - C["attention_surprise_bias"])
    )

    # 3. Pleasantness (positive = joy, negative = sadness):
    #    Goal congruence weighted by relevance, plus Scherer's separate
    #    intrinsic-pleasantness check (SEC 2), which contributes regardless of
    #    whether the stimulus bears on a goal at all.
    pleasantness = (
        gc * relevance_weight * C["pleasantness_congruence_gain"]
        + ip * C["pleasantness_intrinsic_gain"]
    )

    # 4. Aptitude (positive = trust, negative = disgust):
    #    Competence + alignment + agency. A congruent event you caused yourself
    #    reads as competence (trust); one caused by circumstance does not.
    aptitude = (
        cp * C["aptitude_coping_gain"]
        + gc * C["aptitude_congruence_gain"]
        + ag * C["aptitude_agency_gain"]
    )

    return FloatEmotion(
        sensitivity=max(-1.0, min(1.0, sensitivity)),
        attention=max(-1.0, min(1.0, attention)),
        pleasantness=max(-1.0, min(1.0, pleasantness)),
        aptitude=max(-1.0, min(1.0, aptitude)),
    )


# ---------------------------------------------------------------------------
# FloatEmotion → neurotransmitter deltas
# Inverse of the Lövheim-inspired mapping in NeuroRepository.to_emotion_vector()
# ---------------------------------------------------------------------------

def appraisal_to_lovheim(appraisal: Appraisal) -> "LovheimPoint":
    """Map a cognitive :class:`Appraisal` straight to a point in Lövheim's cube.

    Composition of :func:`appraisal_to_float_emotion` with
    :meth:`~emotion_algebra.lovheim.LovheimPoint.from_float_emotion` — the
    full cognition → affect → neurochemistry chain in one call.

    Parameters
    ----------
    appraisal:
        An :class:`Appraisal` instance.

    Returns
    -------
    LovheimPoint

    Examples
    --------
    >>> a = Appraisal(goal_relevance="relevant", goal_congruence="incongruent",
    ...               agency=Agency.OTHER, coping_potential="low")
    >>> appraisal_to_lovheim(a).closest_affect()
    'fear/terror'

    Note
    ----
    The continuous appraisal map produces mild, mixed vectors, and naming a
    weak vector by its single nearest corner is a coarse readout — the blend is
    the honest one.  Prefer :meth:`~emotion_algebra.lovheim.LovheimPoint.affect_blend`
    over :meth:`~emotion_algebra.lovheim.LovheimPoint.closest_affect` for
    appraisal output.
    """
    from emotion_algebra.lovheim import LovheimPoint
    return LovheimPoint.from_float_emotion(appraisal_to_float_emotion(appraisal))


def float_emotion_to_lovheim_deltas(
    fe: "FloatEmotion",
    scale: float = 0.15,
) -> tuple[float, float, float]:
    """Neurotransmitter deltas for *fe*, derived from Lövheim's cube.

    The emotion is projected into the cube
    (:meth:`~emotion_algebra.lovheim.LovheimPoint.from_float_emotion`) and the
    deltas are its **signed** displacement from
    :data:`~emotion_algebra.lovheim.BASELINE`, scaled.  This is the principled
    replacement for :func:`float_emotion_to_neuro_deltas`, whose hand-tuned
    coefficients only approximated the cube.

    Unlike the old function, serotonin and adrenaline here are *signed and
    independent*: a state can lower serotonin without raising adrenaline.  A
    neutral emotion produces all-zero deltas — no affect, no displacement.

    Parameters
    ----------
    fe:
        A :class:`~emotion_algebra.float_emotion.FloatEmotion`.
    scale:
        How strongly one event moves the levels.  Raw cube displacement is in
        ``[-0.5, +0.5]`` per axis; the default keeps a single turn from
        swamping the baseline.

    Returns
    -------
    tuple[float, float, float]
        ``(dopamine_delta, serotonin_delta, adrenaline_delta)``, each in
        ``[-scale, +scale]``.  "Adrenaline" is Lövheim's noradrenaline axis —
        see :mod:`emotion_algebra.lovheim`.

    Examples
    --------
    >>> from emotion_algebra import FloatEmotion
    >>> float_emotion_to_lovheim_deltas(FloatEmotion())  # neutral → no displacement
    (0.0, 0.0, 0.0)
    >>> d, s, a = float_emotion_to_lovheim_deltas(FloatEmotion(sensitivity=3.0))
    >>> d > 0  # rage sits on the high-dopamine face of the cube
    True
    """
    from emotion_algebra.lovheim import LovheimPoint

    dopamine, serotonin, adrenaline = LovheimPoint.from_float_emotion(fe).deltas_from_baseline()
    # Displacement is in [-0.5, 0.5]; ×2 puts a full-corner excursion at ±scale.
    return (dopamine * scale * 2.0, serotonin * scale * 2.0, adrenaline * scale * 2.0)


def float_emotion_to_neuro_deltas(
    fe: "FloatEmotion",
    scale: float = 0.15,
) -> tuple[float, float, float]:
    """Convert a :class:`FloatEmotion` to neurotransmitter deltas.

    .. deprecated::
        Superseded by :func:`float_emotion_to_lovheim_deltas`, which derives the
        same three numbers from Lövheim's cube instead of from hand-tuned
        coefficients.  This function keeps its original **non-negative**
        contract (serotonin and adrenaline are mutually exclusive), so switching
        is not a drop-in swap — the replacement returns *signed* deltas.

    * Dopamine  ← ``(|sensitivity| + |attention|) / 4`` — arousal/salience
      magnitude (both fear and anger are dopaminergic; the sign of sensitivity
      encodes approach/avoidance, not arousal strength)
    * Serotonin ← positive valence — ``(pleasantness + aptitude) / 4``
    * Adrenaline ← negative valence — when ``pleasantness + aptitude < 0``

    Parameters
    ----------
    fe:
        A :class:`~emotion_algebra.float_emotion.FloatEmotion`.
    scale:
        Scaling factor applied to all deltas (default ``0.15``).

    Returns
    -------
    tuple[float, float, float]
        ``(dopamine_delta, serotonin_delta, adrenaline_delta)``, each ``>= 0``.

    Examples
    --------
    >>> import warnings
    >>> from emotion_algebra import FloatEmotion
    >>> fe = FloatEmotion(sensitivity=0.5, attention=0.5, pleasantness=0.3, aptitude=0.3)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", DeprecationWarning)
    ...     d, s, a = float_emotion_to_neuro_deltas(fe)
    >>> d > 0 and s > 0
    True
    """
    warnings.warn(
        "float_emotion_to_neuro_deltas() is deprecated and will be removed in "
        f"{_removal_version()}; use float_emotion_to_lovheim_deltas(), which "
        "derives the deltas from Lövheim's cube. Note the replacement returns "
        "SIGNED deltas — serotonin and adrenaline are no longer mutually "
        "exclusive — so it is not a drop-in substitution.",
        DeprecationWarning,
        stacklevel=2,
    )

    vec = fe.as_array  # [sensitivity, attention, pleasantness, aptitude]

    # Arousal → dopamine: sensitivity (arousal) + attention (salience).
    # Both are magnitudes — a highly-aroused fear (sensitivity < 0) is just
    # as dopaminergic as a highly-aroused anger (sensitivity > 0).
    dopamine_delta = (abs(float(vec[0])) + abs(float(vec[1]))) / 4.0 * scale

    # Valence → serotonin (positive) or adrenaline (negative)
    net_valence = (float(vec[2]) + float(vec[3])) / 4.0 * scale
    serotonin_delta = max(0.0, net_valence)
    adrenaline_delta = max(0.0, -net_valence)

    return (dopamine_delta, serotonin_delta, adrenaline_delta)


# ---------------------------------------------------------------------------
# Appraisal -> the affect core (v3)
#
# The generative layer, mapped onto the descriptive one. Two of the core's axes
# ARE appraisal checks -- potency IS coping potential, unpredictability IS
# novelty -- so most of this map is an identity, not a fit. That is the whole
# argument for the GRID axes: the dimensions of felt emotion are the dimensions
# of appraisal, because appraisal is what constructs the feeling.
# ---------------------------------------------------------------------------

def appraisal_to_affect(appraisal: "Appraisal") -> "AffectState":
    """Map a cognitive :class:`Appraisal` onto the affect core.

    ==========================  ============================================
    core axis                   appraisal check
    ==========================  ============================================
    ``potency``                 **coping potential**, centred and signed.
                                *This is an identity, not a fit.*
    ``unpredictability``        **novelty**.  Also an identity.
    ``positivity``/``negativity``  goal congruence, tempered by intrinsic
                                pleasantness
    ``arousal``                 goal relevance -- how much is at stake
    ==========================  ============================================

    The crux case falls straight out.  Take one obstructing event and vary
    *nothing but coping potential*:

    >>> from emotion_algebra.appraisal import Appraisal, appraisal_to_affect
    >>> fight = Appraisal(goal_relevance=0.9, goal_congruence=0.0,
    ...                   coping_potential=0.9, novelty=0.2)
    >>> flight = Appraisal(goal_relevance=0.9, goal_congruence=0.0,
    ...                    coping_potential=0.1, novelty=0.8)
    >>> appraisal_to_affect(fight).potency > 0
    True
    >>> appraisal_to_affect(flight).potency < 0
    True
    >>> appraisal_to_affect(fight).valence < 0 and appraisal_to_affect(flight).valence < 0
    True

    Same event, same unpleasantness. Coping decides whether you fight or flee.
    """
    from emotion_algebra.affect import AffectState

    f = appraisal.to_float()

    # Coping potential in [0,1] -> potency in [-1,1]. An identity, re-centred.
    potency = 2.0 * f.coping_potential - 1.0

    # Novelty in [0,1] -> unpredictability in [0,1]. An identity.
    unpredictability = f.novelty

    # Goal congruence in [0,1] (0 = obstructive) -> signed hedonic tone, pulled
    # toward the stimulus's own pleasantness.
    hedonic = (2.0 * f.goal_congruence - 1.0) * 0.75 + (
        2.0 * f.intrinsic_pleasantness - 1.0
    ) * 0.25

    # Relevance is what is at stake, and stakes are what activate you.
    arousal = f.goal_relevance

    return AffectState(
        positivity=max(0.0, min(1.0, hedonic)),
        negativity=max(0.0, min(1.0, -hedonic)),
        potency=max(-1.0, min(1.0, potency)),
        arousal=max(0.0, min(1.0, arousal)),
        unpredictability=max(0.0, min(1.0, unpredictability)),
    )
