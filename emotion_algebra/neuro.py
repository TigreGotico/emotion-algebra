"""Neuromodulators — mapped to computational roles, not to emotion names.

Why not Lövheim's cube
----------------------
Lövheim (2012) maps three monoamines onto the eight Tomkins affects, one per
cube corner.  The mapping has **never been empirically tested** — no study has
measured monoamine levels against discrete emotion reports in the same subjects —
and it appeared in *Medical Hypotheses*, which did not practise external peer
review.  For scale: the far narrower serotonin-*depression* hypothesis, studied
enormously more, did not survive umbrella review (Moncrieff et al. 2022).

But the error is not "neurochemistry".  The error is mapping neuromodulators
directly onto **discrete emotion labels**, which is a claim nobody knows how to
test.  Map them instead onto **computational roles** and you are standing on some
of the most replicated work in systems neuroscience:

======================  =====================================================
modulator               role
======================  =====================================================
**dopamine**            reward-prediction error; incentive salience; vigor
                        (Schultz, Dayan & Montague 1997).  Note: *wanting*,
                        not *liking* (Berridge & Robinson).
**noradrenaline**       arousal, adaptive gain, and **unexpected** uncertainty
                        (Aston-Jones & Cohen 2005; Yu & Dayan 2005)
**acetylcholine**       **expected** uncertainty; attentional precision
                        (Yu & Dayan 2005)
**serotonin**           patience, behavioural inhibition, aversive weighting,
                        effective time horizon (Doya 2002)
**cortisol**            sustained threat under low coping
**opioids/oxytocin**    hedonic *liking*; affiliation (in-group only — see
                        De Dreu et al. 2010)
**testosterone**        dominance and status seeking
======================  =====================================================

Doya (2002), *Metalearning and neuromodulation*, is the canonical computational
framing: dopamine ≈ TD error, serotonin ≈ discount factor, noradrenaline ≈
inverse temperature, acetylcholine ≈ learning rate.

And here is the payoff: **those roles land directly on the affect core's axes.**
Noradrenaline drives arousal and unpredictability.  Dopamine drives potency and
approach.  Serotonin drives inhibition and aversive weight.  The neurochemistry
and the empirical affect space agree, because both are describing the same
functional dimensions — which is exactly what a mapping to *emotion names* could
never show.

Lövheim's three monoamines are a subset of these, so his cube remains reachable
as a coordinate drop.  Nothing downstream breaks.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np

from emotion_algebra.affect import CORE_AXES, AffectState

#: The modulators, in canonical order.
MODULATORS: Tuple[str, ...] = (
    "dopamine",
    "noradrenaline",
    "serotonin",
    "acetylcholine",
    "cortisol",
    "opioids",
    "testosterone",
)

#: Every modulator sits in ``[0, 1]``, relative to the individual's own baseline.
BASELINE_LEVEL: float = 0.5

#: How each modulator loads onto the core axes.
#:
#: Each row is a *computational role* expressed in core coordinates, and each is
#: traceable to the citation in its evidence grade — never to an emotion label.
#: Magnitudes are calibrated; the literature fixes the signs and the orderings.
LOADINGS: Dict[str, Dict[str, float]] = {
    # Incentive salience and vigor: makes you want, act, and feel able to.
    "dopamine": {"potency": 0.55, "positivity": 0.45, "arousal": 0.30},
    # Arousal and *unexpected* uncertainty — the world just broke its own rules.
    "noradrenaline": {"arousal": 0.70, "unpredictability": 0.50, "potency": -0.10},
    # Patience and behavioural inhibition; weights aversive outcomes.
    "serotonin": {"negativity": -0.45, "potency": 0.25, "arousal": -0.30},
    # *Expected* uncertainty: known noise. Sharpens attention, does not alarm.
    "acetylcholine": {"unpredictability": 0.35, "arousal": 0.15},
    # Sustained threat you cannot cope with.
    "cortisol": {"negativity": 0.60, "potency": -0.50, "arousal": 0.35},
    # Hedonic liking and affiliation — the *pleasure* dopamine is not.
    "opioids": {"positivity": 0.65, "negativity": -0.25, "arousal": -0.15},
    # Status and dominance seeking.
    "testosterone": {"potency": 0.50, "arousal": 0.15},
}


@dataclass(frozen=True)
class NeuroState:
    """Neuromodulator levels, each in ``[0, 1]`` against the individual's baseline.

    ``0.5`` on every axis is baseline — the person's own resting chemistry, not
    an absolute concentration.

    Examples
    --------
    >>> NeuroState().is_baseline
    True
    >>> threat = NeuroState(noradrenaline=0.9, cortisol=0.9, dopamine=0.2)
    >>> threat.to_affect().potency < 0        # low coping
    True
    >>> threat.to_affect().arousal > 0.5      # but highly activated
    True
    """

    dopamine: float = BASELINE_LEVEL
    noradrenaline: float = BASELINE_LEVEL
    serotonin: float = BASELINE_LEVEL
    acetylcholine: float = BASELINE_LEVEL
    cortisol: float = BASELINE_LEVEL
    opioids: float = BASELINE_LEVEL
    testosterone: float = BASELINE_LEVEL

    def __post_init__(self) -> None:
        for m in MODULATORS:
            v = getattr(self, m)
            if isinstance(v, bool) or not isinstance(
                v, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{m} must be a number, got {type(v).__name__}")
            if not np.isfinite(v):
                raise ValueError(f"{m} must be finite, got {v!r}")
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"{m} must be in [0, 1], got {v!r}")

    # ------------------------------------------------------------------
    # Aliases and basics
    # ------------------------------------------------------------------

    @property
    def adrenaline(self) -> float:
        """Alias for :attr:`noradrenaline` — what downstream consumers call it."""
        return self.noradrenaline

    @property
    def as_array(self) -> np.ndarray:
        """Levels in :data:`MODULATORS` order."""
        return np.array([getattr(self, m) for m in MODULATORS], dtype=float)

    @property
    def is_baseline(self) -> bool:
        """``True`` when every modulator sits at the individual's baseline."""
        return bool(np.allclose(self.as_array, BASELINE_LEVEL))

    def deltas_from_baseline(self) -> Dict[str, float]:
        """Signed displacement from baseline, per modulator, each in ``[-0.5, 0.5]``."""
        return {m: float(getattr(self, m) - BASELINE_LEVEL) for m in MODULATORS}

    def with_(self, **levels) -> "NeuroState":
        """Return a copy with the named modulators replaced."""
        unknown = set(levels) - set(MODULATORS)
        if unknown:
            raise ValueError(f"unknown modulators: {sorted(unknown)}")
        return replace(self, **levels)

    # ------------------------------------------------------------------
    # The bridge
    # ------------------------------------------------------------------

    def to_affect(self) -> AffectState:
        """Project to the affect core via the computational roles in :data:`LOADINGS`.

        Each modulator contributes to the axes its *role* implies. This is a
        claim about function, and it is testable — unlike a claim that a
        neurotransmitter combination *is* an emotion.
        """
        axes = {a: 0.0 for a in CORE_AXES}
        for m in MODULATORS:
            delta = float(getattr(self, m)) - BASELINE_LEVEL  # [-0.5, 0.5]
            for axis, w in LOADINGS[m].items():
                axes[axis] += 2.0 * delta * w  # rescale delta to [-1, 1]

        from emotion_algebra.homeostasis import SET_POINT

        # Displacements are *from the resting set point*, not from the origin —
        # baseline chemistry means an organism at rest, and rest is not zero.
        return AffectState(
            positivity=_clamp(SET_POINT.positivity + axes["positivity"], 0.0, 1.0),
            negativity=_clamp(SET_POINT.negativity + axes["negativity"], 0.0, 1.0),
            potency=_clamp(SET_POINT.potency + axes["potency"], -1.0, 1.0),
            arousal=_clamp(SET_POINT.arousal + axes["arousal"], 0.0, 1.0),
            unpredictability=_clamp(
                SET_POINT.unpredictability + axes["unpredictability"], 0.0, 1.0
            ),
        )

    @classmethod
    def from_affect(cls, state: AffectState) -> "NeuroState":
        """Least-squares inverse of :meth:`to_affect`.

        The map is **lossy in both directions** and says so: seven modulators do
        not determine five axes, nor the reverse. This returns the minimum-norm
        chemistry consistent with the requested affect — the least dramatic
        explanation of the state, which is the honest default.
        """
        from emotion_algebra.homeostasis import SET_POINT

        target = state.as_array - SET_POINT.as_array  # displacement from rest

        # Loading matrix: rows = modulators, cols = core axes.
        M = np.zeros((len(MODULATORS), len(CORE_AXES)), dtype=float)
        for i, m in enumerate(MODULATORS):
            for axis, w in LOADINGS[m].items():
                M[i, CORE_AXES.index(axis)] = 2.0 * w

        # Minimum-norm least squares: M.T @ d = target, solved for d.
        deltas, *_ = np.linalg.lstsq(M.T, target, rcond=None)
        levels = np.clip(BASELINE_LEVEL + deltas, 0.0, 1.0)
        return cls(**dict(zip(MODULATORS, levels)))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        payload = {"type": "neuro_state"}
        payload.update({m: float(getattr(self, m)) for m in MODULATORS})
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "NeuroState":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(**{m: float(data[m]) for m in MODULATORS if m in data})

    def __repr__(self) -> str:
        moved = {
            m: getattr(self, m)
            for m in MODULATORS
            if abs(getattr(self, m) - BASELINE_LEVEL) > 1e-9
        }
        if not moved:
            return "NeuroState(baseline)"
        parts = ", ".join(f"{m}={v:.3g}" for m, v in moved.items())
        return f"NeuroState({parts})"


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


#: The resting chemistry.
BASELINE = NeuroState()
