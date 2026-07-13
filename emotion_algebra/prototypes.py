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
#:
#: **Provenance differs by axis, and that matters.**
#:
#: * ``positivity``/``negativity`` and ``arousal`` are **taken from human
#:   norms** — Warriner, Kuperman & Brysbaert (2013), *Norms of valence, arousal
#:   and dominance for 13,915 English lemmas*, Behav. Res. Methods 45:1191-1207,
#:   rescaled from their 1-9 scale.  Benchmarked at r = +0.89 (valence) and
#:   r = +0.69 (arousal) before this calibration; they are now taken directly.
#: * ``potency`` and ``unpredictability`` are **reasoned from the appraisal
#:   literature**, because Warriner does not measure them.  ``unpredictability``
#:   has no counterpart there at all, and ``potency`` — see below — is a
#:   *different construct* from Warriner's "dominance".
#:
#: Potency is appraised coping, not felt dominance
#: -----------------------------------------------
#: Benchmarking potency against Warriner's dominance gives only **r = +0.45**,
#: and the residuals are systematic: humans rate ``rage`` as *less* dominant than
#: ``anger`` (losing your temper is losing control), while the appraisal
#: literature has rage as the *higher*-coping state.
#:
#: These are two different things, and conflating them is a mistake the field
#: makes:
#:
#: * **appraised coping potential** (the *antecedent*): "can I do something about
#:   this?"  Anger is caused by a HIGH-coping appraisal; fear by a LOW-coping one.
#:   This is what Smith & Ellsworth (1985) and Lerner & Keltner (2001) measured,
#:   and it is what ``potency`` means here.
#: * **felt dominance** (the *consequent*): how in-control you feel while in the
#:   grip of the state.  Anger scores only moderately — being enraged is not
#:   being in control.  This is what Warriner's raters reported.
#:
#: They correlate, but they are not the same axis.  ``potency`` is the first.
#: The ordering the appraisal literature predicts — anger above fear — holds in
#: the human data too (anger +0.03 vs fear -0.42 on Warriner's scale), which is
#: the claim that matters; the absolute magnitudes are not transferable.
PROTOTYPES: Dict[str, AffectState] = {
    # --- joy family ---
    "serenity":     AffectState(positivity=0.69, potency=0.25, arousal=0.24, unpredictability=0.05),
    "joy":          AffectState(positivity=0.80, potency=0.35, arousal=0.57, unpredictability=0.10),
    # "ecstasy" in Warriner is dominated by the drug sense; "ecstatic" is the
    # emotion word, and is what is used here.
    "ecstasy":      AffectState(positivity=0.90, potency=0.45, arousal=0.74, unpredictability=0.15),

    # --- sadness family: negative, LOW arousal, low potency ---
    # "pensiveness" is not in Warriner; reasoned from its neighbours.
    "pensiveness":  AffectState(negativity=0.30, potency=-0.20, arousal=0.18, unpredictability=0.10),
    "sadness":      AffectState(negativity=0.65, potency=-0.35, arousal=0.23, unpredictability=0.15),
    "grief":        AffectState(negativity=0.67, potency=-0.55, arousal=0.49, unpredictability=0.20),

    # --- anger family: negative, high arousal, HIGH appraised coping, certain ---
    "annoyance":    AffectState(negativity=0.51, potency=0.30, arousal=0.39, unpredictability=0.15),
    "anger":        AffectState(negativity=0.62, potency=0.60, arousal=0.62, unpredictability=0.15),
    "rage":         AffectState(negativity=0.62, potency=0.80, arousal=0.70, unpredictability=0.10),

    # --- fear family: negative, high arousal, LOW coping, HIGH uncertainty ---
    "apprehension": AffectState(negativity=0.21, potency=-0.30, arousal=0.40, unpredictability=0.45),
    "fear":         AffectState(negativity=0.52, potency=-0.60, arousal=0.64, unpredictability=0.70),
    "terror":       AffectState(negativity=0.56, potency=-0.90, arousal=0.67, unpredictability=0.90),

    # --- trust family ---
    "acceptance":   AffectState(positivity=0.46, potency=0.20, arousal=0.41, unpredictability=0.10),
    "trust":        AffectState(positivity=0.56, potency=0.30, arousal=0.41, unpredictability=0.10),
    "admiration":   AffectState(positivity=0.65, potency=0.35, arousal=0.56, unpredictability=0.15),

    # --- disgust family: rejecting; disgust is potent (it expels) ---
    "boredom":      AffectState(negativity=0.56, potency=0.10, arousal=0.20, unpredictability=0.05),
    "disgust":      AffectState(negativity=0.42, potency=0.35, arousal=0.50, unpredictability=0.15),
    "loathing":     AffectState(negativity=0.65, potency=0.50, arousal=0.44, unpredictability=0.15),

    # --- anticipation family: engaged, forward-leaning ---
    "interest":     AffectState(positivity=0.42, potency=0.25, arousal=0.42, unpredictability=0.30),
    "anticipation": AffectState(positivity=0.06, potency=0.35, arousal=0.55, unpredictability=0.35),
    "vigilance":    AffectState(positivity=0.17, potency=0.45, arousal=0.33, unpredictability=0.35),

    # --- surprise family: unpredictability dominant ---
    "distraction":  AffectState(negativity=0.24, potency=-0.05, arousal=0.37, unpredictability=0.55),
    "surprise":     AffectState(positivity=0.61, potency=-0.15, arousal=0.70, unpredictability=0.85),
    # "amazement" is not in Warriner; "amazed" is.
    "amazement":    AffectState(positivity=0.64, potency=-0.20, arousal=0.62, unpredictability=1.00),

    # --- states the Plutchik lexicon has no word for, but the space needs ---
    #: The midpoint of rage and terror. High-arousal negative with coping
    #: cancelled out — which is what the Hourglass axes wrongly called
    #: "neutrality".
    "distress":     AffectState(negativity=0.75, potency=-0.05, arousal=0.70, unpredictability=0.50),
    #: Negative, low coping, self-directed — the state Tomkins has and Plutchik
    #: does not.
    "shame":        AffectState(negativity=0.59, potency=-0.65, arousal=0.55, unpredictability=0.15),
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
