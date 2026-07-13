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
        # Powerless + unpleasant + activated + UNCERTAIN = get away. Fear.
        #
        # Uncertainty is what separates flight from resignation, and this fell
        # out of the data rather than being designed in: fear is an *uncertain*
        # threat you cannot handle, so you run. Grief is a *certain* loss you
        # cannot handle, so you stop. Arousal alone does not separate them —
        # human norms put grief's arousal at 0.49, squarely in fear's range.
        # It is Lerner & Keltner's certainty dimension doing the work.
        Mode.AVOIDANCE: helpless * neg * ar * unp * 2.0,
        # Powerless + unpleasant + settled + CERTAIN = give up. Sadness, grief.
        Mode.WITHDRAWAL: helpless * neg * (1.0 - ar) * (1.0 - unp),
        # The world is not as expected: find out more. Gated on the situation not
        # being threatening — you investigate a surprise, you flee a threat.
        Mode.ATTENDING: unp * (0.35 + 0.65 * ar) * (1.0 - neg),
        # Powerless, but not fleeing: appease. Weighted below withdrawal — a
        # true submission display is a *social* act, and the core has no social
        # axis to condition it on, so it must not outrank the modes that do.
        Mode.SUBMISSION: helpless * neg * (1.0 - unp) * 0.35,
        # Safe and pleased: draw close.
        Mode.AFFILIATION: pos * (1.0 - ar) + pos * empowered * 0.5,
        # Unpleasant, but I have the power to expel it rather than flee it.
        # Distinguished from antagonism by arousal: you attack what enrages you,
        # you turn away from what merely revolts you.
        Mode.REJECTION: empowered * neg * (1.0 - ar) * 1.5,
        # Nothing much is demanded. Falls away sharply as soon as anything is
        # going on — a shallower term lets rest outrank a live but *mild* state
        # (disgust), simply because the other modes are products of sub-unit
        # numbers.
        Mode.REST: max(0.0, 1.0 - max(pos, neg, abs(pot), ar, unp)) ** 4,
    }

    total = sum(scores.values())
    if total <= 0.0:
        return {str(Mode.REST): 1.0, **{str(m): 0.0 for m in Mode if m is not Mode.REST}}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return {str(m): float(v / total) for m, v in ranked}


def dominant_tendency(state: AffectState) -> str:
    """The single most-ready action mode."""
    return next(iter(action_readiness(state)))
