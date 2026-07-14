"""Evidence grades — how much to trust each part of this library.

Affective science does not speak with one voice, and the models this library
implements sit at wildly different levels of empirical support.  Some are
meta-analytically replicated across cultures; one was published in a journal
that did not practise external peer review.  A library that presents them all in
the same typeface is lying by omission.

So every construct here carries a machine-readable :class:`Grade` and its
citation.  Ask the library how much to trust its own parts:

    >>> from emotion_algebra.evidence import grade_of, Grade
    >>> grade_of("circumplex") is Grade.ESTABLISHED
    True
    >>> grade_of("plutchik.antipodal") is Grade.METAPHOR
    True

The grades are not opinions about how *useful* a model is — Plutchik's wheel is
extremely useful as a labelling convention, and this library ships it.  They are
claims about what the evidence supports.  A ``METAPHOR`` construct may be the
right tool for a job; it just may not be cited as a finding about how emotion
works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from emotion_algebra._compat import StrEnum


class Grade(StrEnum):
    """How well-supported a construct is, from strongest to weakest."""

    #: Replicated, cross-cultural, and/or meta-analytic.  Safe to build on.
    ESTABLISHED = "established"

    #: Good primary evidence, but thin or same-group replication.
    SUPPORTED = "supported"

    #: A live scientific conflict.  Where a construct is CONTESTED, this
    #: library implements *both* readings and lets the caller choose, rather
    #: than silently picking a winner.
    CONTESTED = "contested"

    #: Proposed, plausible, and never empirically tested.  Usable, but it may
    #: not be cited as evidence for anything.
    SPECULATIVE = "speculative"

    #: A design device.  Either empirically disconfirmed, or never intended as
    #: an empirical claim in the first place.  Often still the most convenient
    #: way to *talk* about emotion — which is why it is shipped.
    METAPHOR = "metaphor"


#: Grades that may be cited as support for a claim about how emotion works.
CITABLE = frozenset({Grade.ESTABLISHED, Grade.SUPPORTED})


@dataclass(frozen=True)
class Evidence:
    """The evidential standing of one construct."""

    key: str
    grade: Grade
    cite: str
    note: str = ""

    @property
    def citable(self) -> bool:
        """``True`` when this construct may be cited as evidence for a claim."""
        return self.grade in CITABLE

    def __str__(self) -> str:
        return f"[{self.grade.upper()}] {self.key} — {self.cite}"


_REGISTRY: Dict[str, Evidence] = {}


def register(key: str, grade: Grade, cite: str, note: str = "") -> Evidence:
    """Record the evidential standing of a construct.

    Parameters
    ----------
    key:
        Dotted name of the construct, e.g. ``"plutchik.antipodal"``.
    grade:
        Its :class:`Grade`.
    cite:
        The citation the grade rests on.
    note:
        What the evidence actually says, and what it does not.

    Returns
    -------
    Evidence
    """
    if key in _REGISTRY:
        raise ValueError(f"evidence already registered for {key!r}")
    entry = Evidence(key=key, grade=Grade(grade), cite=cite, note=note)
    _REGISTRY[key] = entry
    return entry


def evidence(key: str, grade: Grade, cite: str, note: str = ""):
    """Class/function decorator: attach an :class:`Evidence` record.

    The decorated object gains an ``__evidence__`` attribute, so a grade travels
    with the construct rather than living only in prose.
    """
    entry = register(key, grade, cite, note)

    def _decorate(obj):
        obj.__evidence__ = entry
        return obj

    return _decorate


def grade_of(key: str) -> Optional[Grade]:
    """Return the :class:`Grade` of *key*, or ``None`` if it is not registered."""
    entry = _REGISTRY.get(key)
    return entry.grade if entry else None


def get(key: str) -> Optional[Evidence]:
    """Return the full :class:`Evidence` record for *key*, or ``None``."""
    return _REGISTRY.get(key)


def all_evidence() -> Dict[str, Evidence]:
    """Every registered construct, keyed by name."""
    return dict(_REGISTRY)


def by_grade(grade: Grade) -> list:
    """Every construct at *grade*, sorted by key."""
    return sorted(
        (e for e in _REGISTRY.values() if e.grade == Grade(grade)),
        key=lambda e: e.key,
    )


def report() -> str:
    """Render the whole evidence table, strongest grade first."""
    order = [
        Grade.ESTABLISHED,
        Grade.SUPPORTED,
        Grade.CONTESTED,
        Grade.SPECULATIVE,
        Grade.METAPHOR,
    ]
    lines = ["emotion-algebra — evidence grades", "=" * 34, ""]
    for grade in order:
        entries = by_grade(grade)
        if not entries:
            continue
        lines.append(f"{grade.upper()}  ({len(entries)})")
        for e in entries:
            lines.append(f"  {e.key}")
            lines.append(f"      cite: {e.cite}")
            if e.note:
                lines.append(f"      note: {e.note}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The grades.
#
# Ordered strongest to weakest.  Each note says what the evidence shows and,
# where it matters more, what it does not.
# ---------------------------------------------------------------------------

register(
    "circumplex",
    Grade.ESTABLISHED,
    "Russell (1980), J. Pers. Soc. Psychol. 39(6):1161-78; "
    "Watson & Tellegen (1985), Psychol. Bull. 98(2):219-35.",
    "Valence x arousal is the best-replicated structure in affective science — "
    "it recurs across self-report, similarity ratings, languages and cultures. "
    "Watson & Tellegen's PA/NA is the same space at a 45-degree rotation.",
)

register(
    "appraisal.control_separates_anger_from_fear",
    Grade.ESTABLISHED,
    "Smith & Ellsworth (1985), J. Pers. Soc. Psychol. 48:813-38; "
    "Roseman (1996), Cognition & Emotion 10:241-77; "
    "Lerner & Keltner (2001), J. Pers. Soc. Psychol. 81:146-59.",
    "Four independent programmes converge: valence and arousal cannot separate "
    "anger from fear (both negative, both high-arousal); control/coping "
    "potential, and secondarily certainty, can and do. Lerner & Keltner show "
    "control MEDIATES the divergent risk judgements — causal, not correlational.",
)

register(
    "categories.no_discrete_signatures",
    Grade.ESTABLISHED,
    "Lindquist et al. (2012), Behav. Brain Sci. 35(3); "
    "Siegel et al. (2018), Psychol. Bull. 144(4).",
    "Two meta-analyses, different modalities (neuroimaging; autonomic), same "
    "null: discrete emotion categories have no consistent, specific signature. "
    "This is why emotion names are a READOUT in this library, never a basis.",
)

register(
    "neuro.dopamine_reward_prediction_error",
    Grade.ESTABLISHED,
    "Schultz, Dayan & Montague (1997), Science 275:1593-9; "
    "Berridge & Robinson (1998) on wanting vs liking.",
    "Phasic dopamine encodes reward-prediction error — among the most "
    "replicated results in systems neuroscience. Note it tracks INCENTIVE "
    "SALIENCE (wanting), not hedonic pleasure (liking); conflating the two is "
    "the classic error.",
)

register(
    "neuro.noradrenaline_arousal_and_unexpected_uncertainty",
    Grade.ESTABLISHED,
    "Aston-Jones & Cohen (2005), Annu. Rev. Neurosci. 28:403-50; "
    "Yu & Dayan (2005), Neuron 46(4):681-92.",
    "Locus-coeruleus noradrenaline sets arousal and adaptive gain, and signals "
    "UNEXPECTED uncertainty (a change in the world's rules) — as distinct from "
    "acetylcholine's expected uncertainty (known noise).",
)

register(
    "grid",
    Grade.SUPPORTED,
    "Fontaine, Scherer, Roesch & Ellsworth (2007), Psychol. Sci. 18(12):1050-7, "
    "'The world of emotions is not two-dimensional'.",
    "Four dimensions — valence, potency/control, arousal, unpredictability — "
    "derived from 144 componential features across cultures; a 2-D solution was "
    "statistically insufficient. Partially self-replicated by the same group; "
    "no fully independent replication located, hence SUPPORTED not ESTABLISHED.",
)

register(
    "tendency.action_readiness",
    Grade.SUPPORTED,
    "Frijda (1986), The Emotions; Frijda, Kuipers & ter Schure (1989), "
    "J. Pers. Soc. Psychol. 57:212-28.",
    "Action readiness is empirically dissociable from valence/arousal and is "
    "PREDICTED BY appraisal — which is why this library derives it rather than "
    "treating it as an independent generative axis.",
)

register(
    "tendency.anger_is_approach",
    Grade.SUPPORTED,
    "Carver & Harmon-Jones (2009), Psychol. Bull. 135:183-204; "
    "corroborated behaviourally by Lerner & Keltner (2001).",
    "Anger is negative-valence but APPROACH-motivated. This breaks any model "
    "that equates valence sign with approach/avoidance. The specific EEG "
    "asymmetry mechanism drew published rebuttal; the behavioural claim did not.",
)

register(
    "pad.dominance",
    Grade.SUPPORTED,
    "Mehrabian (1996), Curr. Psychol. 14(4):261-92; Osgood et al. (1957).",
    "Dominance has weak psychometrics as a general-purpose third factor, but it "
    "earns its keep precisely where valence and arousal fail: anger vs fear. "
    "Treated here as the same construct as appraisal's coping potential — a "
    "strong hypothesis, not a demonstrated identity; no direct psychometric "
    "comparison of the two was located.",
)

register(
    "neuro.serotonin_patience_and_inhibition",
    Grade.SUPPORTED,
    "Doya (2002), Neural Networks 15(4-6):495-506, 'Metalearning and "
    "neuromodulation'; Cools, Roiser & Dayan on serotonin and aversive "
    "processing.",
    "Serotonin tracks patience, behavioural inhibition and effective time "
    "horizon (a discount factor). Note the serotonin-DEPRESSION hypothesis is a "
    "separate and much weaker claim — see Moncrieff et al. (2022), Mol. "
    "Psychiatry — and is NOT what this grade endorses.",
)

register(
    "neuro.acetylcholine_expected_uncertainty",
    Grade.SUPPORTED,
    "Yu & Dayan (2005), Neuron 46(4):681-92.",
    "Acetylcholine signals EXPECTED uncertainty — known, estimable noise — and "
    "sets attentional precision.",
)

register(
    "neuro.cortisol_sustained_threat",
    Grade.SUPPORTED,
    "HPA-axis stress literature (Sapolsky and successors).",
    "Cortisol tracks sustained threat under low coping; slow relative to the "
    "monoamines, which is why it is modelled with a longer time constant.",
)

register(
    "valence.bipolarity",
    Grade.CONTESTED,
    "FOR bipolar: Russell (1980); Russell & Carroll (1999). "
    "AGAINST: Cacioppo & Berntson's evaluative space model; "
    "Larsen, McGraw & Cacioppo (2001), J. Pers. Soc. Psychol. 81:684-96.",
    "Larsen et al. show happiness and sadness CO-ACTIVATE in predictably "
    "ambivalent situations (graduation day) — which a single signed axis cannot "
    "represent. On an average day affect does behave bipolarly. Unresolved. "
    "This library therefore carries separable positivity/negativity channels AND "
    "exposes signed valence as their difference, so both readings are available.",
)

register(
    "neuro.panksepp_primary_systems",
    Grade.CONTESTED,
    "Panksepp, Affective Neuroscience (1998); "
    "critical commentary: Barrett (2005).",
    "Seven primary-process systems (SEEKING, RAGE, FEAR, LUST, CARE, "
    "PANIC/GRIEF, PLAY) resting on real causal manipulation — stimulation, "
    "lesion, pharmacology — but almost entirely in animals. The leap from "
    "'conserved subcortical circuit in rats' to 'the signature of a named human "
    "emotion' is exactly the leap Barrett disputes. Better evidenced than the "
    "Lovheim cube; still contested.",
)

register(
    "neuro.oxytocin_affiliation",
    Grade.CONTESTED,
    "Kosfeld et al. (2005), Nature 435:673-6; "
    "but see De Dreu et al. (2010), Science 328:1408-11.",
    "Oxytocin increases trust — but in-group only; De Dreu shows it can raise "
    "out-group derogation. It is NOT a 'niceness' dial, and modelling it as one "
    "would be a misreading of the evidence.",
)

register(
    "neuro.testosterone_dominance",
    Grade.CONTESTED,
    "Status/dominance literature; replication record is mixed.",
    "Associated with status-seeking and dominance behaviour; effect sizes and "
    "replication are inconsistent.",
)

register(
    "lovheim.cube",
    Grade.SPECULATIVE,
    "Lovheim (2012), Medical Hypotheses 78(2):341-8.",
    "NEVER EMPIRICALLY TESTED. No study has measured monoamine levels against "
    "discrete emotion reports in the same subjects. The venue did not practise "
    "external peer review — an Elsevier panel found Medical Hypotheses was "
    "publishing 'baseless, speculative, non-testable' material and removed the "
    "editor in 2010. Compare: the far narrower serotonin-depression hypothesis "
    "did not survive umbrella review (Moncrieff et al. 2022). A 3-monoamine to "
    "8-discrete-emotion mapping is a much stronger claim on far less evidence. "
    "Shipped because downstream consumes it, and because the cube is a useful "
    "labelling convention. It is not a model of neurochemistry.",
)

register(
    "plutchik.antipodal",
    Grade.METAPHOR,
    "Plutchik (1980); tested by Smith & Schneider (2009), "
    "Sociol. Methods Res. 37(4).",
    "Smith & Schneider ran over 2,000 statistical tests and report the "
    "emotion-wheel theory 'receives no empirical support'. The opposite-pairs "
    "structure is borrowed from the colour wheel. Its flagship pair is refuted "
    "directly: anger and fear are not opposites but NEIGHBOURS, both negative "
    "and high-arousal, differing on control. Retained as a labelling convention "
    "— which is what it always was.",
)

register(
    "plutchik.cone",
    Grade.METAPHOR,
    "Plutchik (1980); premise disconfirmed by Stanislawski, Cieciuch & Strus "
    "(2021), Pers. Individ. Dif.",
    "The cone needs a constant-radius circle to decompose quality (angle) from "
    "intensity (radius). The affect circumplex is an ELLIPSE, not a circle — "
    "arousal deviates systematically — so the decomposition does not hold. "
    "This is why this library does not use a polar or conical geometry.",
)

register(
    "hourglass",
    Grade.METAPHOR,
    "Cambria, Livingstone & Hussain (2012), in Cognitive Behavioural Systems, "
    "Springer LNCS.",
    "Self-described as 'a derivative of Plutchik's wheel', constructed to "
    "compute a polarity score for sentiment analysis. No factor-analytic "
    "derivation from ratings data. Its own polarity formula takes the ABSOLUTE "
    "VALUE of Attention and Sensitivity — an admission that those axes are not "
    "hedonically bipolar, which contradicts the bipolar geometry the model "
    "otherwise assumes. Excellent engineering; not a finding.",
)

register(
    "text.english_only",
    Grade.ESTABLISHED,
    "Felbo et al. (2017), EMNLP — DeepMoji, trained on English tweets; "
    "Demszky et al. (2020), ACL — GoEmotions, English; "
    "Buechel & Hahn (2017), EACL — EmoBank, English; "
    "Warriner, Kuperman & Brysbaert (2013), Behav. Res. Methods — English norms.",
    "Not a claim about emotion — a claim about this library. EVERY dataset "
    "behind the text layer is English, so the text layer reads English. The "
    "typographic cues are English orthography besides: Arabic asks questions "
    "with U+061F and has no capital letters at all, so two of the eight cues "
    "are not weak on Arabic, they are structurally dead. Reading another "
    "language here does not raise; it returns a confident wrong number. It is "
    "therefore refused. Nothing else in the library is language-bound.",
)

register(
    "prototypes.cross_lingual_transfer",
    Grade.CONTESTED,
    "Fontaine, Scherer, Roesch & Ellsworth (2007), Psychol. Sci. 18(12):1050-57.",
    "GRID supports the four DIMENSIONS replicating across languages and "
    "cultures. It does not establish that the PER-TERM POSITIONS are stable — "
    "that 'raiva' or 'ghadab' sits exactly where 'anger' sits is an assumption, "
    "not a finding. It cannot currently be checked either: there is no "
    "human-rated Arabic valence/arousal/dominance lexicon in existence, and the "
    "one openly available Portuguese norm set is Brazilian, not European. The "
    "machine-translated 'Arabic' sentiment lexicons that do exist encode English "
    "raters' judgements, not Arabic speakers', and are not used here.",
)
