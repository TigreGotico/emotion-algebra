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

What is earned, and what is not
-------------------------------
``scripts/robustness.py`` perturbs every guessed coefficient in this module by
±50% and re-checks the library's claims. The result is worth knowing before you
rely on this:

* **The DIRECTION is earned.**  "Anger approaches while being unpleasant" holds in
  **100%** of perturbations. So does "coping flips anger and fear". These follow
  from the structure of the model, not from the numbers.
* **The specific MODE is not.**  ``dominant_tendency(anger) == "antagonism"`` holds
  in only **55%**; ``fear -> "avoidance"`` in **52%**.  Approach and antagonism
  are neighbouring readings of the same drive, and which of them wins the argmax
  depends on coefficients nobody has fitted.

So: **trust the direction, and prefer the distribution.**  :func:`action_readiness`
returns all nine modes with their weights; :func:`dominant_tendency` is an argmax
convenience, and the argmax is the fragile part.  Treating its output as a finding
about people would be asserting a coefficient we chose.
"""
from __future__ import annotations

from typing import Dict

from emotion_algebra._compat import StrEnum
from emotion_algebra.affect import AffectState


#: How much of an empowered-hostile drive is counted as *approach* as well as
#: *antagonism*.
#:
#: Antagonism is a kind of approach — you move toward what you intend to attack —
#: so the same drive legitimately appears in both modes. One half is the neutral
#: split: it asserts no view about which reading dominates. Registered
#: ``CALIBRATED``; the robustness report says whether anything depends on it.
HOSTILE_APPROACH_SHARE: float = 0.5

#: The floor of the orienting response.
#:
#: An unexpected event turns your head even when you are entirely unaroused, so
#: ATTENDING ramps with arousal from a floor rather than from zero.
ORIENT_FLOOR: float = 0.35

#: Weight of each remaining rule, relative to the others. See the module
#: docstring for what each is doing, and ``provenance.py`` for what is behind
#: them (very little — they were tuned until the prototypes produced the
#: readiness modes the literature describes, which is calibration against our own
#: expectations).
FLIGHT_URGENCY: float = 2.0      # flight is the defining fear response
REJECTION_WEIGHT: float = 1.5    # you turn away from what merely revolts you
SUBMISSION_WEIGHT: float = 0.35  # below withdrawal: a social act, and we have no social axis
REST_FALLOFF: int = 4            # rest must not outrank a live but mild state


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
        #
        # HOSTILE_APPROACH_SHARE: an empowered, unpleasant, activated state is
        # BOTH approaching and antagonistic — antagonism is a *kind of* approach,
        # so the same drive is counted in both modes. Half is the neutral split:
        # it asserts no view about which reading dominates.
        Mode.APPROACH: pos * ar + empowered * neg * ar * HOSTILE_APPROACH_SHARE,
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
        Mode.AVOIDANCE: helpless * neg * ar * unp * FLIGHT_URGENCY,
        # Powerless + unpleasant + settled + CERTAIN = give up. Sadness, grief.
        Mode.WITHDRAWAL: helpless * neg * (1.0 - ar) * (1.0 - unp),
        # The world is not as expected: find out more. Gated on the situation not
        # being threatening — you investigate a surprise, you flee a threat.
        #
        # ORIENT_FLOOR: even a wholly unaroused surprise still turns your head.
        # Orienting does not require activation, so the arousal term ramps from a
        # floor rather than from zero.
        Mode.ATTENDING: unp * (ORIENT_FLOOR + (1.0 - ORIENT_FLOOR) * ar) * (1.0 - neg),
        # Powerless, but not fleeing: appease. Weighted below withdrawal — a
        # true submission display is a *social* act, and the core has no social
        # axis to condition it on, so it must not outrank the modes that do.
        Mode.SUBMISSION: helpless * neg * (1.0 - unp) * SUBMISSION_WEIGHT,
        # Safe and pleased: draw close. Affiliation is mostly a *calm* mode — you
        # bond when you are not braced — but confidence helps, which is the second
        # term. It is weighted at HOSTILE_APPROACH_SHARE for the same reason: a
        # neutral split, asserting nothing.
        Mode.AFFILIATION: pos * (1.0 - ar) + pos * empowered * HOSTILE_APPROACH_SHARE,
        # Unpleasant, but I have the power to expel it rather than flee it.
        # Distinguished from antagonism by arousal: you attack what enrages you,
        # you turn away from what merely revolts you.
        Mode.REJECTION: empowered * neg * (1.0 - ar) * REJECTION_WEIGHT,
        # Nothing much is demanded. Falls away sharply as soon as anything is
        # going on — a shallower term lets rest outrank a live but *mild* state
        # (disgust), simply because the other modes are products of sub-unit
        # numbers.
        Mode.REST: max(0.0, 1.0 - max(pos, neg, abs(pot), ar, unp)) ** REST_FALLOFF,
    }

    total = sum(scores.values())
    if total <= 0.0:
        return {str(Mode.REST): 1.0, **{str(m): 0.0 for m in Mode if m is not Mode.REST}}
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return {str(m): float(v / total) for m, v in ranked}


def dominant_tendency(state: AffectState) -> str:
    """The single most-ready action mode."""
    return next(iter(action_readiness(state)))
