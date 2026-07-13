"""Action readiness — what the organism is getting ready to *do*.

Frijda (1986) defines emotions as "changes in action readiness"; Frijda, Kuipers
& ter Schure (1989) show readiness modes are reliably assigned to emotion words
and are **predicted by appraisal**.  So readiness is derived here, not treated as
an independent generative axis — the evidence says it is downstream.

The fact this module exists to carry
------------------------------------
**Anger is negative in valence but approach-motivated.**  Carver & Harmon-Jones
(2009), *Psychological Bulletin* 135:183-204, and corroborated behaviourally by
Lerner & Keltner (2001): angry people behave like *approach-motivated* people,
not like fearful ones.

That single fact refutes any model which equates valence sign with
approach/avoidance — including every "positive = approach, negative = avoid"
sentiment system.  Motivational direction tracks **potency**, not valence:

* negative + **high** potency -> anger -> *approach*, attack
* negative + **low** potency  -> fear  -> *avoid*, flee
* negative + low potency + low arousal -> sadness -> *withdraw*

Valence tells you whether it is good.  Potency tells you what you are going to do
about it.
"""
from __future__ import annotations

from typing import Dict

from emotion_algebra._compat import StrEnum
from emotion_algebra.affect import AffectState


class Mode(StrEnum):
    """Frijda's modes of action readiness (the subset this core can support)."""

    #: Move toward, engage, pursue.  Anger and joy share this.
    APPROACH = "approach"
    #: Move against — attack, confront.  Approach with hostile intent.
    ANTAGONISM = "antagonism"
    #: Move away — flee, escape.  Active avoidance.
    AVOIDANCE = "avoidance"
    #: Disengage, give up, turn inward.  Passive; sadness lives here.
    WITHDRAWAL = "withdrawal"
    #: Orient, attend, gather information.  Surprise and interest.
    ATTENDING = "attending"
    #: Yield, defer, appease.
    SUBMISSION = "submission"
    #: Draw close, bond, care for.
    AFFILIATION = "affiliation"
    #: Push out, expel, reject.  Disgust.
    REJECTION = "rejection"
    #: Do nothing; nothing is demanded.
    REST = "rest"


def action_readiness(state: AffectState) -> Dict[str, float]:
    """Return the readiness of every :class:`Mode`, normalized to sum to 1.

    Derived from the core coordinates, following the appraisal-to-readiness
    correspondences Frijda reports.  Every rule below is a statement about
    *potency*, not valence — that is the point.

    Examples
    --------
    >>> from emotion_algebra.prototypes import prototype
    >>> r = action_readiness(prototype("anger"))
    >>> r["approach"] > r["avoidance"]        # anger approaches...
    True
    >>> prototype("anger").valence < 0        # ...while being unpleasant
    True
    """
    pos = state.positivity
    neg = state.negativity
    pot = state.potency
    ar = state.arousal
    unp = state.unpredictability

    empowered = max(0.0, pot)     # the sense that one can act
    helpless = max(0.0, -pot)     # the sense that one cannot

    scores = {
        # Eager pursuit. Scales with reward and activation — and, via the second
        # term, with *empowered hostility*, which is how anger gets here despite
        # negative valence.
        Mode.APPROACH: pos * ar + empowered * neg * ar * 0.5,
        # Approach + unpleasantness + power = move against it.
        Mode.ANTAGONISM: empowered * neg * ar,
        # Powerless + unpleasant + ACTIVATED = get away. Fear.
        # Arousal is what separates fleeing from giving up, so avoidance must be
        # gated on it: no floor term, or sadness would read as flight.
        Mode.AVOIDANCE: helpless * neg * ar * 1.3,
        # Powerless + unpleasant + DEACTIVATED = give up. Sadness.
        Mode.WITHDRAWAL: helpless * neg * (1.0 - ar),
        # The world is not as expected: find out more. Gated on not being
        # overwhelmed — an unpredictable threat you cannot handle makes you run,
        # not investigate. Surprise orients; terror does not.
        Mode.ATTENDING: unp * (0.35 + 0.65 * ar) * (1.0 - helpless * neg),
        # Powerless, but not fleeing: appease.
        Mode.SUBMISSION: helpless * neg * (1.0 - unp) * 0.5,
        # Safe and pleased: draw close.
        Mode.AFFILIATION: pos * (1.0 - ar) + pos * empowered * 0.5,
        # Unpleasant, but I have the power to expel it rather than flee it.
        Mode.REJECTION: empowered * neg * (1.0 - ar),
        # Nothing much is demanded. Falls away sharply as soon as anything is
        # going on — a linear term here would let rest outrank a live emotion
        # simply because the other modes are products of sub-unit numbers.
        Mode.REST: max(0.0, 1.0 - max(pos, neg, abs(pot), ar, unp)) ** 3,
    }

    total = sum(scores.values())
    if total <= 0.0:
        return {str(Mode.REST): 1.0, **{str(m): 0.0 for m in Mode if m is not Mode.REST}}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return {str(m): float(v / total) for m, v in ranked}


def dominant_tendency(state: AffectState) -> str:
    """The single most-ready action mode."""
    return next(iter(action_readiness(state)))
