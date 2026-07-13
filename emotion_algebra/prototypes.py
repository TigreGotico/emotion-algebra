"""Named emotions as regions of the affect core.

These are **prototypes, not basis vectors**.  Lindquist et al. (2012) and Siegel
et al. (2018) — two meta-analyses, neuroimaging and autonomic — find no
consistent, category-specific signature for discrete emotions.  So an emotion
name is a *label applied to a region*, and the honest readout is a distribution
over labels (:mod:`emotion_algebra.readout`), never an assignment to one.

Cowen & Keltner (2017) found the same shape from the other direction: many
distinguishable categories, but bridged by *continuous gradients* rather than
separated by sharp boundaries.  Prototypes-plus-gradients is exactly that.

Coordinates are calibrated to the appraisal profiles the literature reports for
each family, not fitted to a dataset:

* **anger family** — negative, high arousal, **high potency**, low
  unpredictability.  You know what happened and you can act on it.
* **fear family** — negative, high arousal, **low potency**, **high
  unpredictability**.  You don't and you can't.  (Smith & Ellsworth 1985;
  Lerner & Keltner 2001.)
* **sadness family** — negative, **low arousal**, low potency.  Loss already
  suffered; nothing to be done.
* **surprise family** — **unpredictability dominant**, valence near zero.
"""
from __future__ import annotations

from typing import Dict

from emotion_algebra.affect import AffectState

#: Named emotion prototypes in core coordinates.
#:
#: The 24 Plutchik names are kept because they are a useful, widely-understood
#: vocabulary — but here they are *labels over regions*, not axes.
PROTOTYPES: Dict[str, AffectState] = {
    # --- joy family: positive, approach, moderate-to-high arousal ---
    "serenity":     AffectState(positivity=0.35, potency=0.25, arousal=0.20, unpredictability=0.05),
    "joy":          AffectState(positivity=0.70, potency=0.35, arousal=0.55, unpredictability=0.10),
    "ecstasy":      AffectState(positivity=1.00, potency=0.45, arousal=0.90, unpredictability=0.15),

    # --- sadness family: negative, LOW arousal, low potency ---
    "pensiveness":  AffectState(negativity=0.30, potency=-0.20, arousal=0.15, unpredictability=0.10),
    "sadness":      AffectState(negativity=0.65, potency=-0.35, arousal=0.25, unpredictability=0.15),
    "grief":        AffectState(negativity=1.00, potency=-0.55, arousal=0.40, unpredictability=0.20),

    # --- anger family: negative, HIGH arousal, HIGH potency, low surprise ---
    "annoyance":    AffectState(negativity=0.30, potency=0.30, arousal=0.35, unpredictability=0.15),
    "anger":        AffectState(negativity=0.65, potency=0.60, arousal=0.70, unpredictability=0.15),
    "rage":         AffectState(negativity=1.00, potency=0.80, arousal=1.00, unpredictability=0.10),

    # --- fear family: negative, HIGH arousal, LOW potency, HIGH surprise ---
    "apprehension": AffectState(negativity=0.30, potency=-0.30, arousal=0.35, unpredictability=0.45),
    "fear":         AffectState(negativity=0.65, potency=-0.60, arousal=0.70, unpredictability=0.70),
    "terror":       AffectState(negativity=1.00, potency=-0.90, arousal=1.00, unpredictability=0.90),

    # --- trust family: positive, affiliative, calm ---
    "acceptance":   AffectState(positivity=0.30, potency=0.20, arousal=0.15, unpredictability=0.10),
    "trust":        AffectState(positivity=0.60, potency=0.30, arousal=0.25, unpredictability=0.10),
    "admiration":   AffectState(positivity=0.90, potency=0.35, arousal=0.45, unpredictability=0.15),

    # --- disgust family: negative, rejecting; disgust is potent (it expels) ---
    "boredom":      AffectState(negativity=0.25, potency=0.10, arousal=0.10, unpredictability=0.05),
    "disgust":      AffectState(negativity=0.60, potency=0.35, arousal=0.40, unpredictability=0.15),
    "loathing":     AffectState(negativity=0.90, potency=0.50, arousal=0.65, unpredictability=0.15),

    # --- anticipation family: engaged, forward-leaning, mildly positive ---
    "interest":     AffectState(positivity=0.35, potency=0.25, arousal=0.35, unpredictability=0.30),
    "anticipation": AffectState(positivity=0.50, potency=0.35, arousal=0.55, unpredictability=0.35),
    "vigilance":    AffectState(positivity=0.45, potency=0.45, arousal=0.80, unpredictability=0.35),

    # --- surprise family: unpredictability dominant, valence near zero ---
    "distraction":  AffectState(potency=-0.05, arousal=0.35, unpredictability=0.55),
    "surprise":     AffectState(potency=-0.15, arousal=0.65, unpredictability=0.85),
    "amazement":    AffectState(positivity=0.20, potency=-0.20, arousal=0.85, unpredictability=1.00),

    # --- states the Plutchik lexicon has no word for, but the space needs ---
    #: The midpoint of rage and terror. High-arousal negative with control
    #: cancelled out — which is what the Hourglass axes wrongly called
    #: "neutrality".
    "distress":     AffectState(negativity=0.85, potency=-0.05, arousal=0.85, unpredictability=0.50),
    #: Negative, low potency, low arousal, *and* self-directed — the state
    #: Tomkins has and Plutchik does not.
    "shame":        AffectState(negativity=0.70, potency=-0.65, arousal=0.35, unpredictability=0.15),
}


def prototype(name: str) -> AffectState:
    """Return the core prototype for *name*.

    Raises
    ------
    KeyError
        If *name* is not a known prototype.
    """
    key = name.lower().strip()
    if key not in PROTOTYPES:
        raise KeyError(f"unknown emotion prototype: {name!r}")
    return PROTOTYPES[key]
