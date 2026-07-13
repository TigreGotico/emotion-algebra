"""The affect core — the empirically-grounded hub every other model converts through.

Coordinates follow Fontaine, Scherer, Roesch & Ellsworth (2007), *The world of
emotions is not two-dimensional*, Psychological Science 18(12):1050-7 — four
dimensions derived from 144 componential features across cultures:

    valence · potency/control · arousal · unpredictability

with one deliberate refinement: **valence is carried as two separable
non-negative channels**, ``positivity`` and ``negativity``.  Larsen, McGraw &
Cacioppo (2001) show that happiness and sadness genuinely co-activate in
predictably ambivalent situations, and a single signed axis cannot represent
that.  Signed ``valence`` remains available as their difference, so both readings
of an unresolved scientific question are on the table (see
:data:`~emotion_algebra.evidence` key ``valence.bipolarity``).

Why these axes and not the Hourglass's
--------------------------------------
The Hourglass packs anger and fear onto one signed *Sensitivity* axis, as
opposites.  They are not opposites.  Four independent appraisal programmes
(Smith & Ellsworth 1985; Roseman 1996; Scherer's SECs; Lerner & Keltner 2001,
with causal mediation) find that anger and fear are *both* negative and *both*
high-arousal, and are separated by **control / coping potential**.  They are
neighbours, not antipodes.

The consequence of getting this wrong is not subtle.  On the Hourglass axes::

    (rage + terror) / 2  ==  [0, 0, 0, 0]     ->  neutrality

Blend the two most intense negative states and you get calm.  Here, the same
blend lands on low potency-variance, negative, high-arousal — *distress* — which
is what it should be.

This is not a vector space
--------------------------
There is no ``__neg__``.  Sadness is not "minus joy": it has its own action
readiness (withdraw, help-seek), which is not "negative approach".  The
supported operations are **convex mixture**, **intensification** (a positive
scalar), **decay** (contraction toward baseline) and **distance** — that is a
convex cone with a metric, not a vector space.

Nor is it polar or conical.  Plutchik's cone needs a constant-radius circle to
separate quality (angle) from intensity (radius); the affect circumplex is an
*ellipse* (Stanisławski, Cieciuch & Strus 2021), so that decomposition does not
hold.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple

import numpy as np

#: The core coordinates, in canonical order.
#:
#: Graded ``SUPPORTED`` under the key ``"grid"`` — see :mod:`emotion_algebra.evidence`.
CORE_AXES: Tuple[str, ...] = (
    "positivity",
    "negativity",
    "potency",
    "arousal",
    "unpredictability",
)

#: Axes bounded to ``[0, 1]``.  Only ``potency`` is signed.
_UNIPOLAR = ("positivity", "negativity", "arousal", "unpredictability")


@dataclass(frozen=True)
class AffectState:
    """A point in the affect core.

    Parameters
    ----------
    positivity, negativity:
        Separable hedonic channels, each in ``[0, 1]``.  Both may be high at
        once — that is ambivalence, and it is the point of splitting them.
    potency:
        Control / coping, in ``[-1, 1]``.  **The axis that separates anger
        (high) from fear (low).**  The only signed coordinate.
    arousal:
        Activation, in ``[0, 1]``.
    unpredictability:
        How little the situation is expected, in ``[0, 1]``.

    Examples
    --------
    >>> anger = AffectState(negativity=0.8, potency=0.6, arousal=0.8, unpredictability=0.2)
    >>> fear  = AffectState(negativity=0.8, potency=-0.6, arousal=0.8, unpredictability=0.8)
    >>> anger.valence < 0 and fear.valence < 0        # both unpleasant
    True
    >>> anger.potency > 0 > fear.potency              # and separated by control
    True
    """

    positivity: float = 0.0
    negativity: float = 0.0
    potency: float = 0.0
    arousal: float = 0.0
    unpredictability: float = 0.0

    def __post_init__(self) -> None:
        for axis in CORE_AXES:
            v = getattr(self, axis)
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise TypeError(f"{axis} must be a number, got {type(v).__name__}")
            if not np.isfinite(v):
                raise ValueError(f"{axis} must be finite, got {v!r}")
        for axis in _UNIPOLAR:
            v = float(getattr(self, axis))
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{axis} must be in [0, 1], got {v!r}")
        if not -1.0 <= float(self.potency) <= 1.0:
            raise ValueError(f"potency must be in [-1, 1], got {self.potency!r}")

    # ------------------------------------------------------------------
    # Derived readings
    # ------------------------------------------------------------------

    @property
    def valence(self) -> float:
        """Signed hedonic tone, ``positivity - negativity``, in ``[-1, 1]``.

        The bipolar reading, offered as an exact projection of the two channels.
        It is lossy on purpose: it cannot tell ``(0.5, 0.5)`` — deeply mixed —
        from ``(0.0, 0.0)`` — flat.  :attr:`ambivalence` is what it drops.
        """
        return float(self.positivity - self.negativity)

    @property
    def ambivalence(self) -> float:
        """How much positive and negative are co-active, ``min(pos, neg)``.

        The quantity a signed-valence model destroys.  Graduation day scores
        high here and ~zero on :attr:`valence`.
        """
        return float(min(self.positivity, self.negativity))

    @property
    def intensity(self) -> float:
        """Overall affective magnitude — how *much* is going on, in ``[0, 1]``."""
        return float(
            max(self.positivity, self.negativity, abs(self.potency), self.arousal)
        )

    @property
    def as_array(self) -> np.ndarray:
        """The five coordinates, in :data:`CORE_AXES` order."""
        return np.array([getattr(self, a) for a in CORE_AXES], dtype=float)

    @property
    def is_origin(self) -> bool:
        """``True`` when every coordinate is exactly zero.

        This is a **mathematical** fact about the coordinates, not a
        psychological one. The origin is not "no emotion" — core affect is
        always on, and no organism sits here. For "nothing much is happening",
        use :func:`~emotion_algebra.homeostasis.at_rest`, which asks whether the
        state is near the *set point*.
        """
        return bool(np.allclose(self.as_array, 0.0))

    # ------------------------------------------------------------------
    # The legitimate operations.  There is no __neg__, __sub__ or __mul__ —
    # see the module docstring.
    # ------------------------------------------------------------------

    def blend(self, other: "AffectState", weight: float = 0.5) -> "AffectState":
        """Convex mixture with *other*: ``(1-w)·self + w·other``.

        Mixture — not addition — is the supported way to combine affect.  It
        keeps the result inside the space, and it is what Cowen & Keltner's
        continuous gradients between emotion categories look like.

        Raises
        ------
        ValueError
            If *weight* is outside ``[0, 1]`` (that would leave the cone).
        """
        if not 0.0 <= float(weight) <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight!r}")
        w = float(weight)
        mixed = (1.0 - w) * self.as_array + w * other.as_array
        return AffectState(**dict(zip(CORE_AXES, mixed)))

    def intensify(self, factor: float) -> "AffectState":
        """Scale magnitude by a non-negative *factor*, clamped back into the space.

        Raises
        ------
        ValueError
            If *factor* is negative — that would be negation by the back door.
        """
        if float(factor) < 0.0:
            raise ValueError(
                f"factor must be non-negative, got {factor!r}; "
                "the affect core has no negation (see the module docstring)"
            )
        scaled = self.as_array * float(factor)
        return AffectState(
            positivity=min(1.0, scaled[0]),
            negativity=min(1.0, scaled[1]),
            potency=max(-1.0, min(1.0, scaled[2])),
            arousal=min(1.0, scaled[3]),
            unpredictability=min(1.0, scaled[4]),
        )

    def decay(
        self,
        dt: float,
        half_life: float,
        toward: "AffectState" = None,
    ) -> "AffectState":
        """Relax toward the resting set point, halving the gap every *half_life*.

        **Not toward the origin.** Core affect is always on — there is no state
        of "no emotion" — so an organism at rest sits at a mildly positive,
        low-arousal *set point*, not at zero (Cacioppo & Berntson's positivity
        offset). Decaying to the origin would drive it to a state it never
        occupies.

        Parameters
        ----------
        toward:
            The set point. Defaults to
            :data:`~emotion_algebra.homeostasis.SET_POINT`.

        See Also
        --------
        emotion_algebra.homeostasis : set points, drives, and temperament.
        """
        from emotion_algebra.homeostasis import SET_POINT, relax

        return relax(self, dt, toward=SET_POINT if toward is None else toward,
                     half_life=half_life)

    def distance(self, other: "AffectState") -> float:
        """Euclidean distance in the core.

        Note the honest caveat: no study comparing Euclidean against angular
        distance on human similarity judgements was located, so this is a
        reasonable default rather than a validated choice.
        """
        return float(np.linalg.norm(self.as_array - other.as_array))

    def with_(self, **axes) -> "AffectState":
        """Return a copy with the named axes replaced."""
        unknown = set(axes) - set(CORE_AXES)
        if unknown:
            raise ValueError(f"unknown axes: {sorted(unknown)}")
        return replace(self, **axes)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        payload = {"type": "affect_state"}
        payload.update({a: float(getattr(self, a)) for a in CORE_AXES})
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "AffectState":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(**{a: float(data[a]) for a in CORE_AXES if a in data})

    def __repr__(self) -> str:
        parts = ", ".join(f"{a}={getattr(self, a):.3g}" for a in CORE_AXES)
        return f"AffectState({parts})"


def mixture(states: Iterable[AffectState], weights: Iterable[float] = None) -> AffectState:
    """Convex mixture of any number of states.

    Parameters
    ----------
    states:
        The states to combine.
    weights:
        Non-negative weights, normalized to sum to 1.  Uniform if omitted.

    Raises
    ------
    ValueError
        If *states* is empty, weights are negative, or the weights sum to zero.
    """
    states = list(states)
    if not states:
        raise ValueError("cannot take a mixture of no states")
    if weights is None:
        w = np.ones(len(states), dtype=float)
    else:
        w = np.asarray(list(weights), dtype=float)
        if w.shape[0] != len(states):
            raise ValueError("weights and states must be the same length")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    w = w / total
    stacked = np.stack([s.as_array for s in states])
    mixed = (stacked * w[:, None]).sum(axis=0)
    return AffectState(**dict(zip(CORE_AXES, mixed)))


#: The coordinate origin.
#:
#: **This is not "no emotion", and it is not rest.**  Core affect is always on
#: (Barrett & Bliss-Moreau 2009), and an organism at rest sits at a mildly
#: positive, low-arousal *set point*, not at zero — see
#: :data:`emotion_algebra.homeostasis.SET_POINT`.  The origin is a mathematical
#: reference that nothing occupies.  It is exported because it is the additive
#: identity of the coordinates, not because it is a state anyone is ever in.
ORIGIN = AffectState()
