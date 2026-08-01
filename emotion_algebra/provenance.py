"""Where every number in this library came from.

:mod:`emotion_algebra.evidence` grades the *theories* — it will tell you that
Plutchik's antipodal wheel is a ``METAPHOR`` and that Lövheim's cube was never
tested.  That is only half an audit.  It says nothing about the fact that
``NEGATIVITY_BIAS = 1.5`` was chosen by hand.

There are 162 hand-chosen numbers in this package.  This module records where
each one came from, and a test asserts that **every** module-level constant is
registered — so a new magic number turns CI red.

    >>> from emotion_algebra import provenance
    >>> provenance.of("homeostasis.NEGATIVITY_BIAS").trustworthy
    False
    >>> len(provenance.guessed())          # not backed by data or a publication
    19
    >>> provenance.counts()["assumed"]     # a bare number with no reason at all
    0

The uncomfortable summary
-------------------------
Most of these numbers are **not** fitted to anything, and could not be: the data
that would fit them is not obtainable.  The GRID per-emotion coordinates are, by
the source paper's own footnote, unpublished and available only on request from
the author.  Smith & Ellsworth's appraisal table and Frijda's action-readiness
table are paywalled.

So this module does not pretend the numbers are earned.  It records that they are
not, names each one, and — together with ``scripts/robustness.py`` — measures
**which of the library's claims actually depend on them.**  A claim that only
holds for the coefficients someone happened to pick is not a claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from emotion_algebra._compat import StrEnum


class Provenance(StrEnum):
    """Where a number came from, from strongest to weakest."""

    #: Derived from a dataset. The record names the script that did it, so the
    #: number can be regenerated and checked.
    FITTED = "fitted"

    #: Copied verbatim from a named published table or formula.
    PUBLISHED = "published"

    #: A judgement call, with a stated rationale. The literature typically fixes
    #: the *sign* or the *ordering*; the magnitude is ours.
    CALIBRATED = "calibrated"

    #: No basis. Not a citation, not a fit, not a rationale — just a number.
    #: **Every one of these is a bug.** They must be justified, derived, or
    #: deleted, and a test asserts there are none left.
    ASSUMED = "assumed"


#: Provenances that may be cited as support for a number.
TRUSTWORTHY = frozenset({Provenance.FITTED, Provenance.PUBLISHED})


@dataclass(frozen=True)
class Constant:
    """The provenance of one number."""

    key: str
    value: Any
    provenance: Provenance
    cite: str
    note: str = ""

    @property
    def trustworthy(self) -> bool:
        """``True`` when the number rests on data or a published source."""
        return self.provenance in TRUSTWORTHY

    def __str__(self) -> str:
        return f"[{self.provenance.upper()}] {self.key} = {self.value!r}"


_REGISTRY: Dict[str, Constant] = {}


def register_constant(
    key: str,
    value: Any,
    provenance: Provenance,
    cite: str,
    note: str = "",
) -> Constant:
    """Record where a number came from.

    Parameters
    ----------
    key:
        Dotted name, e.g. ``"homeostasis.NEGATIVITY_BIAS"``.
    value:
        The number itself (or a dict/tuple of them, for a table).
    provenance:
        Its :class:`Provenance`.
    cite:
        The script that fitted it, the table it was copied from, or — for
        ``ASSUMED`` — a plain statement that there is no source.
    note:
        What is actually known. For a ``CALIBRATED`` number this is usually
        "the literature fixes the sign; the magnitude is ours".

    Raises
    ------
    ValueError
        If the key is already registered, or the citation is empty.
    """
    if key in _REGISTRY:
        raise ValueError(f"constant already registered: {key!r}")
    if not cite.strip():
        raise ValueError(f"{key!r} needs a citation — even 'no source' is a citation")
    entry = Constant(
        key=key, value=value, provenance=Provenance(provenance),
        cite=cite.strip(), note=note.strip(),
    )
    _REGISTRY[key] = entry
    return entry


def of(key: str) -> Constant:
    """The provenance record for *key*.

    Raises
    ------
    KeyError
        If *key* is not registered.
    """
    if key not in _REGISTRY:
        raise KeyError(f"unregistered constant: {key!r}")
    return _REGISTRY[key]


def all_constants() -> Dict[str, Constant]:
    """Every registered number."""
    return dict(_REGISTRY)


def by_provenance(provenance: Provenance) -> List[Constant]:
    """Every constant at *provenance*, sorted by key."""
    return sorted(
        (c for c in _REGISTRY.values() if c.provenance == Provenance(provenance)),
        key=lambda c: c.key,
    )


def counts() -> Dict[str, int]:
    """How many constants sit at each provenance."""
    return {p.value: len(by_provenance(p)) for p in Provenance}


def guessed() -> List[Constant]:
    """Every number that is **not** backed by data or a publication.

    These are the ones ``scripts/robustness.py`` perturbs, to find out which of
    the library's claims depend on them.
    """
    return sorted(
        (c for c in _REGISTRY.values() if not c.trustworthy),
        key=lambda c: c.key,
    )


def report() -> str:
    """The full audit, grouped by provenance."""
    order = [
        Provenance.FITTED,
        Provenance.PUBLISHED,
        Provenance.CALIBRATED,
        Provenance.ASSUMED,
    ]
    n = len(_REGISTRY)
    lines = [
        "emotion-algebra — where the numbers came from",
        "=" * 45,
        "",
        f"{n} registered constants.  {len(guessed())} of them are not backed by "
        f"data or a publication.",
        "",
    ]
    for p in order:
        entries = by_provenance(p)
        if not entries:
            continue
        lines.append(f"{p.upper()}  ({len(entries)})")
        for c in entries:
            lines.append(f"  {c.key} = {c.value!r}")
            lines.append(f"      source: {c.cite}")
            if c.note:
                lines.append(f"      known : {c.note}")
        lines.append("")
    if not by_provenance(Provenance.ASSUMED):
        lines.append("No ASSUMED constants remain.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The register.
#
# Ordered by module.  Read the ASSUMED section first — those are the ones with
# nothing behind them at all.
# ---------------------------------------------------------------------------

_WARRINER = (
    "FITTED by scripts/validate/calibrate_prototypes.py from Warriner, Kuperman "
    "& Brysbaert (2013), Norms of valence, arousal and dominance for 13,915 "
    "English lemmas, Behav. Res. Methods 45:1191-1207."
)

register_constant(
    "prototypes.valence",
    "26 emotions x positivity/negativity",
    Provenance.FITTED,
    cite=_WARRINER,
    note="Taken directly from the human means, rescaled from the 1-9 scale. "
         "Correlating these back against Warriner would be circular; any honest "
         "evaluation uses held-out data.",
)

register_constant(
    "prototypes.arousal",
    "26 emotions x arousal",
    Provenance.FITTED,
    cite=_WARRINER,
    note="As above.",
)

register_constant(
    "prototypes.potency",
    "26 emotions x potency",
    Provenance.CALIBRATED,
    cite="Reasoned from the appraisal literature (Smith & Ellsworth 1985; "
         "Roseman 1996; Lerner & Keltner 2001). Their numeric tables are "
         "paywalled and could not be obtained, so the values are not fitted.",
    note="The literature fixes the ORDERING (anger above fear) and that ordering "
         "holds in human data too. The MAGNITUDES are ours. Note potency means "
         "appraised COPING, which is not PAD's felt dominance — they correlate "
         "at only r=0.46. See evidence key 'pad.dominance'.",
)

register_constant(
    "prototypes.unpredictability",
    "26 emotions x unpredictability",
    Provenance.CALIBRATED,
    cite="Reasoned from the appraisal literature. Warriner measures no such "
         "axis, and the GRID coordinates that would fit it are unpublished "
         "(Fontaine et al. 2007, footnote 3: available on request from the "
         "author).",
    note="The LEAST supported axis in the library. Nothing validates these 26 "
         "numbers. The ordering (fear and surprise high, anger low) is what the "
         "certainty literature implies; the magnitudes are invented.",
)

_FRIJDA = (
    "Frijda, Kuipers & ter Schure (1989) rate 30 emotion words against 16 "
    "readiness modes — the table that would fit these is paywalled (403). The "
    "coefficients were tuned until the prototypes produced the readiness modes "
    "the literature describes, which is calibration against our own "
    "expectations, not against data."
)

_NEURO_NOTE = (
    "The literature fixes the SIGN and the ORDERING of each loading — that "
    "dopamine drives approach, that noradrenaline drives arousal and unexpected "
    "uncertainty. It fixes no magnitude. No dataset maps neuromodulators onto "
    "these axes, so none of these weights is fitted."
)

register_constant(
    "neuro.LOADINGS",
    "7 modulators x axis weights (19 numbers)",
    Provenance.CALIBRATED,
    cite="Signs and roles from Schultz, Dayan & Montague (1997); Aston-Jones & "
         "Cohen (2005); Yu & Dayan (2005); Doya (2002). Magnitudes: ours.",
    note=_NEURO_NOTE,
)

register_constant(
    "neuro.BASELINE_LEVEL",
    0.5,
    Provenance.CALIBRATED,
    cite="A convention, not a measurement: the midpoint of [0, 1].",
    note="Every modulator is expressed RELATIVE to the individual's own "
         "baseline, so 0.5 is a coordinate choice rather than a concentration.",
)

register_constant(
    "appraisal.APPRAISAL_COEFFS",
    "10 SEC->axis gains",
    Provenance.CALIBRATED,
    cite="Scherer's Component Process Model predicts the DIRECTION and relative "
         "ordering of appraisal effects, not their magnitudes — the CPM is a "
         "qualitative theory.",
    note="Each magnitude was fitted to reproduce the orderings the theory does "
         "predict, against the modal-emotion fixtures in test/test_appraisal.py. "
         "That is calibration against our own expectations, not against data.",
)

register_constant(
    "homeostasis.SET_POINT",
    "5 axis values",
    Provenance.CALIBRATED,
    cite="The positivity offset is ESTABLISHED in DIRECTION (Cacioppo & "
         "Berntson's evaluative space model): at rest, positivity exceeds "
         "negativity. No portable magnitude exists — the studies located use "
         "incompatible scales (Ito et al. 1998 vs later work), so nothing "
         "transfers.",
    note="Direction: evidenced. Magnitude: ours. This is why an agent at rest "
         "explores rather than sitting inert.",
)

register_constant(
    "homeostasis.NEGATIVITY_BIAS",
    1.5,
    Provenance.CALIBRATED,
    cite="NO SOURCE. 'Bad is stronger than good' (Baumeister, Bratslavsky, "
         "Finkenauer & Vohs 2001) is a NARRATIVE REVIEW and reports no ratio. "
         "The familiar '~2x' is Kahneman & Tversky's loss-aversion lambda "
         "(~2.25), fitted to MONETARY GAMBLES — a different domain, and not "
         "established for affective weighting.",
    note="The direction is well-evidenced: negative events weigh more. The "
         "magnitude 1.5 is invented, and citing the negativity-bias literature "
         "for it would be borrowing a number that literature does not contain.",
)

register_constant(
    "homeostasis.DEFAULT_HALF_LIFE",
    300.0,
    Provenance.CALIBRATED,
    cite="NO SOURCE. Emotion-duration data exists (Verduyn & Lavrijsen 2015 "
         "report sadness as longest-lasting) but the per-emotion durations are "
         "paywalled and could not be verified.",
    note="Seconds. An arbitrary default, exposed as a parameter precisely "
         "because it is arbitrary — set it to whatever your tick rate needs. "
         "Nothing in the library depends on this value being right.",
)

register_constant(
    "homeostasis.REST_TOLERANCE",
    0.1,
    Provenance.CALIBRATED,
    cite="A threshold on the core's own metric, not a measured quantity.",
    note="How close to the set point counts as 'at rest'. 0.1 is roughly 5% of "
         "the space's diameter.",
)

register_constant(
    "readout.TEMPERATURE",
    0.25,
    Provenance.CALIBRATED,
    cite="No source. Chosen so a prototype reads as itself with a clear margin "
         "while its true neighbours stay visibly non-zero.",
    note="Controls only how SHARP the label distribution is, never which label "
         "wins. The argmax is invariant to it.",
)

register_constant(
    "pad._DOMINANCE_WEIGHTS",
    {"sensitivity": 0.60, "aptitude": 0.30, "attention": 0.10},
    Provenance.CALIBRATED,
    cite="Anchored on the most-replicated fact about dominance: anger is "
         "dominant, fear is submissive (Mehrabian 1996), so Sensitivity carries "
         "it and is the only signed term.",
    note="Only used by the Hourglass VIEW, which is graded METAPHOR. The core "
         "needs no such reconstruction: it has a potency axis.",
)

register_constant(
    "tendency.HOSTILE_APPROACH_SHARE",
    0.5,
    Provenance.CALIBRATED,
    cite="A neutral split, not a measurement. Antagonism is a KIND of approach — "
         "you move toward what you intend to attack — so an empowered-hostile "
         "drive legitimately appears in both modes.",
    note="One half asserts no view about which reading dominates. Any value in "
         "(0, 1) preserves the ordering the literature actually constrains.",
)

register_constant(
    "tendency.ORIENT_FLOOR",
    0.35,
    Provenance.CALIBRATED,
    cite="No source. An unexpected event turns your head even when you are "
         "entirely unaroused, so ATTENDING ramps from a floor rather than zero.",
    note="The floor's EXISTENCE is the claim; its height is ours.",
)

register_constant(
    "tendency.FLIGHT_URGENCY",
    2.0,
    Provenance.CALIBRATED,
    cite=_FRIJDA,
    note="Flight is the defining response of the whole fear family, so avoidance "
         "is weighted above the modes it competes with. The ORDERING is the "
         "claim; the factor is ours.",
)

register_constant(
    "tendency.REJECTION_WEIGHT",
    1.5,
    Provenance.CALIBRATED,
    cite=_FRIJDA,
    note="Separates rejection from antagonism by arousal: you attack what "
         "enrages you, you turn away from what merely revolts you.",
)

register_constant(
    "tendency.SUBMISSION_WEIGHT",
    0.35,
    Provenance.CALIBRATED,
    cite=_FRIJDA,
    note="Held BELOW withdrawal deliberately. A submission display is a SOCIAL "
         "act and the core has no social axis to condition it on, so it must not "
         "outrank the modes that the core can actually justify.",
)

register_constant(
    "tendency.REST_FALLOFF",
    4,
    Provenance.CALIBRATED,
    cite="No source. An exponent chosen so that rest falls away sharply once "
         "anything is happening.",
    note="A shallower exponent let REST outrank a live but MILD state (disgust), "
         "purely because the other modes are products of sub-unit numbers. This "
         "corrects an artefact of the scoring's shape, not a claim about people.",
)

register_constant(
    "tendency.WEIGHTS",
    "coefficients of the 9 action-readiness rules",
    Provenance.CALIBRATED,
    cite="Frijda, Kuipers & ter Schure (1989) rate 30 emotion words against 16 "
         "readiness modes — the table that would fit these is paywalled (403).",
    note="These were tuned until the prototypes produced the readiness modes the "
         "literature describes. That is fitting to our own expectations. The "
         "robustness report says which of the resulting claims survive "
         "perturbation and which do not.",
)

register_constant(
    "neural.PROBE",
    "73 x 5 linear probe",
    Provenance.FITTED,
    cite="FITTED on GoEmotions (Demszky et al. 2020) for four axes, and on "
         "EmoBank's (Buechel & Hahn 2017) human arousal ratings for arousal. "
         "Held-out scores are reported in emotion_algebra/neural.py.",
    note="The only genuinely fitted machinery in the library besides the "
         "prototype valence/arousal.",
)

register_constant(
    "neural.SATURATION_CAPS",
    {"punctuation": 5, "length": 50},
    Provenance.CALIBRATED,
    cite="Squashing constants, not measurements: they bound a count so one "
         "20-exclamation-mark message cannot dominate the feature.",
    note="The probe is fitted DOWNSTREAM of these, so their exact values are "
         "absorbed into the learned weights. The robustness report confirms "
         "nothing depends on them.",
)

register_constant(
    "lang.SATURATION_CAPS",
    {"punctuation": 5, "length": 50, "kashida": 10},
    Provenance.CALIBRATED,
    cite="A squashing constant, not a measurement: it bounds the kashida count "
         "so one heavily-stretched word cannot saturate the channel, mirroring "
         "the punctuation caps in neural.SATURATION_CAPS.",
    note="Arabic has no capital letters, so the SHOUTING channel is realised by "
         "kashida (tatweel, U+0640) stretching instead. That the two are the "
         "same channel is a CLAIM about how emphasis is written, and the "
         "multilingual evaluation is what tests it. The bound itself is "
         "arbitrary and any probe fitted downstream absorbs it.",
)

register_constant(
    "multilingual.PROBE",
    "393 x 5 linear probe",
    Provenance.FITTED,
    cite="FITTED on ENGLISH ONLY, by scripts/validate/fit_multilingual_probe.py: "
         "valence/potency/unpredictability on GoEmotions (Demszky et al. 2020) "
         "label prototypes, arousal on EmoBank (Buechel & Hahn 2017) human "
         "ratings. Held out on EmoBank: valence r=+0.56, arousal r=+0.50.",
    note="Fitted on English and applied to Portuguese and Arabic ZERO-SHOT, "
         "relying on the encoder's cross-lingual alignment. That the mapping "
         "crosses is not assumed — it is measured on XED and reported per "
         "language in docs/evidence.md, INCLUDING where it fails. It transfers "
         "to Portuguese (potency d=+0.46, the dominant axis) and only PARTIALLY "
         "to Arabic (potency d=+0.28; unpredictability dominates instead). No "
         "in-language fitting is possible: no human-rated Arabic VAD lexicon "
         "exists, and the open Portuguese norms are Brazilian, not European.",
)
