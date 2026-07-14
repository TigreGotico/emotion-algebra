# Evidence

Affective science does not speak with one voice. Some of the models in this
library are replicated across cultures and meta-analyses. One was published in a
journal that did not practise external peer review. **A library that presents them
all in the same typeface is lying by omission.**

So every construct carries a machine-readable grade and its citation:

```python
from emotion_algebra import evidence

evidence.grade_of("circumplex")          # Grade.ESTABLISHED
evidence.grade_of("lovheim.cube")        # Grade.SPECULATIVE

print(evidence.report())                 # the whole table, with citations

evidence.get("lovheim.cube").citable     # False
```

## The grades

| Grade | Meaning |
| --- | --- |
| `ESTABLISHED` | Replicated, cross-cultural, and/or meta-analytic. Safe to build on. |
| `SUPPORTED` | Good primary evidence, but thin or same-group replication. |
| `CONTESTED` | A live scientific conflict. **Both readings are implemented**; the library does not pick a winner. |
| `SPECULATIVE` | Proposed, plausible, never empirically tested. Usable — but it may not be cited as evidence for anything. |
| `METAPHOR` | A design device. Either empirically disconfirmed, or never intended as an empirical claim. Often still the most convenient way to *talk* about emotion, which is why it ships. |

A grade is **not** a judgement of usefulness. Plutchik's wheel is graded
`METAPHOR` and is extremely useful — as a vocabulary. The grade says what the
evidence supports, not what the construct is good for.

`ESTABLISHED` and `SUPPORTED` are `CITABLE`. The rest are not.

## What holds up

**`ESTABLISHED`**

- **The circumplex** (Russell 1980; Watson & Tellegen 1985). Valence × arousal is
  the best-replicated structure in affective science — it recurs across
  self-report, similarity ratings, languages and cultures.
- **Control separates anger from fear** (Smith & Ellsworth 1985; Roseman 1996;
  Scherer's SECs; Lerner & Keltner 2001). Four independent programmes converge.
  Lerner & Keltner show control *mediates* the divergent risk judgements — causal,
  not correlational.
- **Discrete emotions have no consistent signature** (Lindquist et al. 2012;
  Siegel et al. 2018). Two meta-analyses, different modalities (neuroimaging;
  autonomic physiology), the same null. This is why emotion names are a **readout**
  in this library, never a basis.
- **Dopamine encodes reward-prediction error** (Schultz, Dayan & Montague 1997).
  Note it tracks *wanting*, not *liking* — conflating the two is the classic error.
- **Noradrenaline signals arousal and unexpected uncertainty** (Aston-Jones &
  Cohen 2005; Yu & Dayan 2005).

**`SUPPORTED`**

- **The GRID four dimensions** (Fontaine, Scherer, Roesch & Ellsworth 2007) —
  valence, potency, arousal, unpredictability, from 144 componential features
  across cultures. A 2-D solution was statistically insufficient. Partially
  self-replicated by the same group; no fully independent replication located,
  hence `SUPPORTED` rather than `ESTABLISHED`.
- **Action readiness** (Frijda 1986; Frijda, Kuipers & ter Schure 1989) — and it is
  *predicted by* appraisal, which is why this library derives it rather than
  treating it as an independent axis.
- **Anger is approach-motivated** despite negative valence (Carver & Harmon-Jones
  2009). This breaks any model tying approach to positive valence.
- **Serotonin, acetylcholine, cortisol** in their computational roles (Doya 2002;
  Yu & Dayan 2005).
- **PAD's dominance** — weak psychometrics as a general third factor, but it earns
  its keep precisely where valence and arousal fail: anger vs fear.

## What is genuinely unresolved

**`CONTESTED` — and both readings ship.**

- **Is valence bipolar?** Russell says yes. Cacioppo & Berntson's evaluative space
  model says positivity and negativity are separable systems that can co-activate —
  and Larsen, McGraw & Cacioppo (2001) show happiness and sadness *do* co-activate
  in predictably ambivalent situations (graduation day). On an average day, affect
  behaves bipolarly. **Unresolved.** So the core carries separable
  `positivity`/`negativity` channels *and* exposes signed `valence` as their
  difference. You get both.
- **Panksepp's primary-process systems** — real causal manipulation (stimulation,
  lesion, pharmacology), but almost entirely in animals. The leap to "the signature
  of a named human emotion" is exactly what Barrett disputes.
- **Oxytocin and affiliation** — Kosfeld et al. (2005) show it increases trust; De
  Dreu et al. (2010) show it can raise out-group derogation. It is **not** a
  "niceness" dial, and modelling it as one would misread the evidence.

## What does not hold up

**`SPECULATIVE`**

- **Lövheim's cube.** **Never empirically tested** — no study has measured monoamine
  levels against discrete emotion reports in the same subjects. The venue,
  *Medical Hypotheses*, did not practise external peer review; an Elsevier panel
  found it was publishing "baseless, speculative, non-testable" material and
  removed the editor in 2010. For scale, the far narrower serotonin-*depression*
  hypothesis did not survive umbrella review (Moncrieff et al. 2022), and a
  3-monoamine → 8-discrete-emotion mapping is a much stronger claim on far less
  evidence.

  It ships because downstream consumers use it and because the cube is a useful
  labelling convention. **It is not a model of neurochemistry.** For a defensible
  one, see [neurochemistry](neurochemistry.md).

**`METAPHOR`**

- **Plutchik's antipodal wheel.** Smith & Schneider (2009) ran over 2,000
  statistical tests and report the emotion-wheel theory *"receives no empirical
  support"*. The opposite-pairs structure is borrowed from the colour wheel, and
  its flagship pair is refuted directly: anger and fear are **neighbours**, both
  negative and high-arousal, differing on control.
- **Plutchik's cone.** It needs a constant-radius circle to separate quality
  (angle) from intensity (radius). The affect circumplex is an **ellipse**
  (Stanisławski, Cieciuch & Strus 2021), so the decomposition does not hold. This
  is why the library does not use a polar or conical geometry.
- **Cambria's Hourglass.** Self-described as "a derivative of Plutchik's wheel",
  constructed to compute a polarity score for sentiment analysis. No
  factor-analytic derivation from ratings data. And its own polarity formula takes
  the **absolute value** of Attention and Sensitivity — a formal admission that
  those axes are not hedonically bipolar, which contradicts the bipolar geometry
  the model otherwise assumes. Excellent engineering; not a finding.

## Provisional findings

Some things this library has measured but does not consider established. They are
marked in the code, and they should not be cited.

**Appraised coping is not felt dominance** (`PROVISIONAL`). Our `potency` axis
means *appraised coping potential* — "can I act on this?", the antecedent
judgement that Smith & Ellsworth and Lerner & Keltner measured. PAD's *dominance*
means how in-control you feel *while in the grip* of the state. They correlate at
only **r = 0.46**, and the residuals look systematic: humans rate `rage` as *less*
dominant than `anger` (being enraged is not being in control), while the appraisal
literature has rage as the *higher*-coping state.

The direction is stable; the magnitude is not established. It rests on 25 emotion
terms, three of which needed hand-picked substitutes for word-sense confounds
(Warriner's *ecstasy* is the drug). **Do not cite it.**

## Arguing with a grade

Every grade carries its citation. If you think one is wrong, the source is right
there:

```python
print(evidence.get("plutchik.antipodal").cite)
print(evidence.get("plutchik.antipodal").note)
```

Open an issue. This is the part of the library most worth being wrong about in
public.


---

# Where the numbers came from

The grades above cover the **theories**. They say nothing about the fact that
`NEGATIVITY_BIAS = 1.5` was chosen by hand.

So every constant in the library carries its own provenance:

```python
from emotion_algebra import provenance

provenance.of("homeostasis.NEGATIVITY_BIAS").provenance   # Provenance.CALIBRATED
print(provenance.report())                                # the full audit
```

| Provenance | Meaning | Count |
| --- | --- | --- |
| `FITTED` | Derived from a dataset; the record names the script that did it | **3** |
| `PUBLISHED` | Copied verbatim from a named table | **0** |
| `CALIBRATED` | A judgement call, with a stated rationale | **19** |
| `ASSUMED` | A bare number with no reason — **every one is a bug** | **0** |

**19 of 22 constants are not backed by data or a publication.** A test enforces
that a new magic number cannot be added without saying what is behind it.

## Why they are not fitted

Because the data is not obtainable, and we will not fabricate it:

- **GRID per-emotion coordinates** — Fontaine et al. (2007) state in their own
  footnote 3 that the numbers are *not published in the article* and "can be
  requested from the first author". No open repository has them. Eyeballing their
  Figure 1 would be fabrication.
- **Smith & Ellsworth (1985)** appraisal table, **Frijda et al. (1989)**
  action-readiness table — paywalled, 403 everywhere.

## One constant whose usual citation is wrong

`NEGATIVITY_BIAS = 1.5` is easy to justify by pointing at "bad is stronger than
good" (Baumeister et al. 2001). **That paper is a narrative review and reports no
ratio.** The familiar "bad counts about twice as much" is Kahneman & Tversky's
loss-aversion λ ≈ 2.25 — fitted to **monetary gambles**, a different domain, and
never established for affective weighting.

The direction is well-evidenced. The magnitude is ours. The library says so.

<a name="robustness"></a>
# Robustness — which claims survive if the guesses are wrong?

You cannot validate a constant you have no data for. You **can** measure which of
your conclusions depend on it.

`scripts/robustness.py` perturbs **every** guessed constant by ±50% (1000 runs)
and re-derives every claim the library makes.

```
  anger and fear separate on potency                            100.0%  EARNED
  valence/arousal CANNOT separate anger from fear               100.0%  EARNED
  rage + terror -> distress, not neutrality                     100.0%  EARNED
  anger approaches while being unpleasant                       100.0%  EARNED
  coping alone flips anger and fear                             100.0%  EARNED
  L&K: anger judges risk lower than fear                        100.0%  EARNED
  L&K: anger patterns with happiness, not fear                  100.0%  EARNED
  rest is mildly positive (the positivity offset)               100.0%  EARNED
  every state relaxes home                                      100.0%  EARNED
  bad news lands harder than good (negativity bias)             100.0%  EARNED
  coping chemistry flips anger and fear under identical threat  100.0%  EARNED
  every prototype names itself                                  100.0%  EARNED
  sadness -> withdrawal                                          96.5%  EARNED
  grief -> withdrawal                                            91.0%  EARNED
  at the set point, nothing is demanded (rest)                   88.1%  fragile
  surprise -> attending                                          87.1%  fragile
  shame -> withdrawal                                            75.0%  fragile
  joy -> affiliation                                             61.0%  fragile
  disgust -> rejection                                           57.5%  fragile
  anger -> antagonism                                            55.9%  fragile
  fear -> avoidance                                              50.0%  fragile
  rest is NOT the origin                                         45.8%  NOT EARNED — an artefact of a chosen number
  - at the set point, nothing is demanded (rest)  (88%)
  - surprise -> attending  (87%)
  - shame -> withdrawal  (75%)
  - joy -> affiliation  (61%)
  - disgust -> rejection  (57%)
  - anger -> antagonism  (56%)
  - fear -> avoidance  (50%)
  - rest is NOT the origin  (46%)
```

## What this means

**The science is earned.** Every load-bearing claim — anger and fear separating on
potency, `rage + terror → distress`, anger approaching while unpleasant, both
Lerner & Keltner predictions, the neurochemical coping flip — holds in **100%** of
perturbations. Those follow from the *structure* of the model, not from any number
we picked. That is the strongest statement this library can make about itself.

**Some outputs are not.** The specific action-readiness *labels* are
coefficient-dependent: `anger → antagonism` survives only 55%, `fear → avoidance`
52%. Approach and antagonism are neighbouring readings of the same drive, and
which wins the argmax is a matter of coefficients nobody has fitted. **The
direction is robust; the label is not.** Prefer `action_readiness()` — the full
distribution — over `dominant_tendency()`.

**And one claim is a genuine artefact.** "The origin is not rest" survives only
**45%**: it holds *because* we chose a set point further from the origin than the
rest tolerance. The concept — core affect is always on, and rest is not a blank
state (Barrett & Bliss-Moreau 2009) — is evidenced. The numerical assertion is
not. Both are now labelled as what they are.

# Open gaps

Standing, and not currently fixable:

- **Potency and unpredictability are not fitted.** Pending the GRID data.
- **English only.** Every dataset behind the text layer (DeepMoji, GoEmotions,
  EmoBank, Warriner) is English. The *core* — appraisal → affect → tendency — is
  language-agnostic; `neural.py` is not.
- **The distance metric is unvalidated.** No Euclidean-vs-angular comparison
  against human similarity judgements exists here; Russell's (1980) similarity
  matrix was not obtainable.
- **Ambivalence is representable but never validated** against human mixed-emotion
  reports.
