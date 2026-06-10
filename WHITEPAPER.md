# emotion-algebra: A Computational Framework for Affective Arithmetic

**Version**: 1.7
**Date**: 2026-03-30
**License**: Apache 2.0

---

## Abstract

We present **emotion-algebra**, a Python library that treats emotions as first-class algebraic objects. Built on Cambria's Hourglass of Emotions (2012) and Plutchik's Wheel (1980), the library defines a signed-integer lattice over four affective axes and implements a complete operator algebra over that lattice. The result is a composable, introspectable, and numerically grounded affective representation suitable for dialogue systems, NPC AI, affective computing research, and classifier post-processing. This paper describes the theoretical foundations, the design decisions that depart from or extend the source models, and the practical APIs for text analysis, emoji mapping, cognitive appraisal, and continuous-space embedding.

---

## 1. Introduction

Computational systems that reason about affect typically face one of two failure modes: representations that are too coarse (positive/negative/neutral) to capture the richness of human emotional experience, or representations that are so fine-grained (e.g. raw VAD vectors from a regression model) that they resist symbolic manipulation, introspection, and rule-based combination.

emotion-algebra occupies the middle ground. It encodes a 24-point discrete lattice — the named emotions of the Hourglass of Emotions — with sufficient algebraic structure to support arithmetic, geometry, temporal dynamics, and multi-signal fusion, while remaining fully introspectable: every value carries a human-readable name, a dimension, and a clear scientific lineage.

The library serves three user groups:

1. **Application developers** who need a structured, composable affect layer for chatbots, game NPCs, or recommendation engines — without writing their own emotion representation.
2. **NLP researchers** who want to post-process classifier output (word embeddings, emoji distributions, transformer logits) into an interpretable affective space.
3. **Affective computing researchers** who want a reference implementation of Plutchik/Cambria semantics in Python, with tests and documented deviations from the original models.

---

## 2. Theoretical Foundations

### 2.1 Cambria's Hourglass of Emotions (2012)

The Hourglass of Emotions (Cambria, Livingstone, & Hussain, 2012) extends Plutchik's wheel into a four-dimensional signed-integer space. Each dimension (called an *axis*) represents an independent affective component:

| Axis | Positive pole (flow > 0) | Negative pole (flow < 0) |
|------|--------------------------|--------------------------|
| **Sensitivity** | Rage (3) → Anger (2) → Annoyance (1) | Apprehension (−1) → Fear (−2) → Terror (−3) |
| **Attention** | Vigilance (3) → Anticipation (2) → Interest (1) | Distraction (−1) → Surprise (−2) → Amazement (−3) |
| **Pleasantness** | Ecstasy (3) → Joy (2) → Serenity (1) | Pensiveness (−1) → Sadness (−2) → Grief (−3) |
| **Aptitude** | Admiration (3) → Trust (2) → Acceptance (1) | Boredom (−1) → Disgust (−2) → Loathing (−3) |

The integer magnitude encodes *intensity*: ±1 is mild, ±2 is primary, ±3 is intense. Values beyond ±3 are supported as hyper-intensity offsets (e.g. "hyper rage" = Sensitivity(5)).

This design makes the Hourglass a metrizable space: distances, clusters, and projections are well-defined operations over the four-dimensional integer lattice.

### 2.2 Plutchik's Wheel of Emotions (1980)

Plutchik's wheel (1980) provides:

- **Eight primary emotions** at three intensity levels each — the named emotions of the Hourglass axes.
- **Dyadic feelings** (*primary dyads*) — named combinations of two adjacent primary emotions (e.g. joy + trust = love, anger + anticipation = aggressiveness).
- **Opposite pairs** — each primary emotion has a diametrically opposite counterpart (joy ↔ sadness, anger ↔ fear, etc.).

In emotion-algebra, `Feeling` encodes named dyads; `CompositeEmotion` encodes the algebraic result of cross-axis combination without necessarily matching a named dyad.

### 2.3 Russell's Circumplex Model (1980)

The Russell Circumplex (1980) organises affect into a 2D valence × arousal plane with six qualitative regions. emotion-algebra projects the Hourglass axes onto this plane:

- **Valence** = Pleasantness axis component only (Posner et al. 2005: arousal and hedonics are orthogonal).
- **Arousal** = `max(|flow|)` across all active axes — axis-independent activation intensity.
- **Type** = one of: `"excited positive"`, `"calm positive"`, `"excited negative"`, `"calm negative"`, `"activated neutral"`, `"neutral"`.

The key design decision here — that `anger.valence == 0` despite anger being colloquially "negative" — is scientifically grounded: anger is a Sensitivity-axis emotion; its hedonic content is zero. Only Pleasantness-axis emotions carry hedonic information.

### 2.4 Scherer's Component Process Model (2001)

Scherer's CPM (2001) models emotion as the output of sequential cognitive appraisal across five dimensions: novelty, goal-relevance, goal-congruence, agency, and coping potential. emotion-algebra implements a simplified 16-rule deterministic mapping from appraisal patterns to primary emotions (`appraisal.py`), enabling causally grounded emotion inference without a full appraisal simulator.

### 2.5 Emoji as Affective Signals (Felbo et al. 2017)

Felbo et al. (2017) demonstrated that emoji co-occurrence in social media text provides a weak but scalable supervision signal for sentiment and emotion classification. The DeepMoji model outputs a probability distribution over 64 emoji. emotion-algebra provides `EMOJI_EMOTION_MAP` — a curated mapping of ~90 Unicode emoji to Plutchik primary emotions — and `DeepMojiAdapter`, which converts DeepMoji-style `{emoji: probability}` distributions into named emotions or `EmotionalState` vectors.

---

## 3. Design Decisions and Departures from Source Models

### 3.1 Integer lattice, not continuous

The Hourglass model specifies integer flow values. emotion-algebra preserves this: `Emotion.emotional_flow` is always an `int`. This makes every named emotion a finite lattice point, enabling exact equality, hashing, and the standard algebraic operators (`+`, `-`, `//`, `<<`, `>>`).

For research use cases requiring continuous-valued representations (e.g. embedding projection, regression model output), `FloatEmotion` provides float-precision arithmetic over the same 4-axis space, with a `from_embedding()` constructor that projects arbitrary-dimension vectors via linear map.

### 3.2 Feeling vs CompositeEmotion

The Hourglass model does not distinguish between Plutchik's named dyads (feelings) and arbitrary two-axis composites. We introduce this distinction explicitly:

- `Feeling` — a named cultural label (Plutchik's primary dyads: love, submission, awe, etc.). Immutable once constructed.
- `CompositeEmotion` — the algebraic result of cross-axis `+` or `*`. May or may not have a Feeling name.

This prevents the library from silently discarding information: `joy + trust` returns `Feeling("love")` because that dyad is named; `joy + anger` returns an unnamed `CompositeEmotion` because no canonical name exists for that combination.

### 3.3 Valence is Pleasantness-only

A frequently confused property of the Hourglass model is that anger and fear have zero hedonic valence. Colloquially, these emotions are "negative," but scientifically, their negativity is *arousal-based* (high Sensitivity activation), not *hedonic*. Posner et al. (2005) make this explicit: valence and arousal are orthogonal dimensions.

emotion-algebra enforces this: `anger.valence == 0`, `fear.valence == 0`. Only Pleasantness-axis emotions have non-zero valence. This design decision is documented in `SPECIFICATION.md` and called out explicitly in the API reference.

### 3.4 kind is approximate

`Emotion.kind` and `CompositeEmotion.kind` return hand-curated psychological category labels (`"event related"`, `"social"`, `"cathected"`, etc.) derived from Plutchik's prose descriptions. These are *not* derived from the Hourglass axes and should be treated as approximate. Two entries (`"gloat"`, `"frivolity"`) are explicitly unclassified. See `AUDIT.md` issue A-001.

---

## 4. Architecture

```
emotion_algebra/
│
├── base.py            EmotionBase ABC — as_array, as_matrix, emotional_flow,
│                      valence, arousal, type, __int__, __float__, comparisons
│
├── plutchik.py        Emotion, Neutrality, EmotionalDimension
│                      EMOTION_AXES, DIMENSIONS
│
├── composite_emotions.py  CompositeEmotion, CompositeDimension
│                          COMPOSITE_EMOTIONS, COMPOSITE_EMOTIONS_NAMES
│
├── feelings.py        Feeling, FEELINGS, FEELING_NAMES
│                      get_feeling, get_feeling_from_emotions
│
├── emotions.py        EMOTIONS registry (MappingProxyType, 24 entries)
│                      get_emotion, emotion_to_dimension, POSITIVE/NEGATIVE_EMOTIONS
│
├── state.py           EmotionalState (mutable 4-axis float accumulator)
│                      EmotionTimeline (snapshot sequence)
│
├── distance.py        emotion_distance, closest_emotion, emotion_clusters
│
├── text.py            from_text, score_text (lexicon)
│                      score_mixed, from_mixed (word + emoji)
│                      HFEmotionAdapter (HuggingFace bridge)
│
├── emoji.py           EMOJI_EMOTION_MAP (~90 emoji)
│                      from_emoji, score_emojis, from_emojis
│                      register_emoji, unregister_emoji
│                      DeepMojiAdapter
│
├── appraisal.py       Appraisal (Scherer CPM dataclass)
│                      appraisal_to_emotion (16-rule table)
│
├── float_emotion.py   FloatEmotion (continuous space, from_embedding)
│
├── lexicons.py        word_emotion_lexicon.csv loader
│                      get_word_emotion, get_sentiment, get_color, etc.
│
└── __init__.py        Public API + EmotionAnalyzer facade
```

All global registries (`EMOTIONS`, `FEELINGS`, `EMOJI_EMOTION_MAP`, `BEHAVIOURS`, `REACTIONS`) are wrapped in `MappingProxyType` — immutable after module load.

---

## 5. Multi-signal Emotion Inference

A key design goal is composability across signal sources. The same `EmotionalState` accumulator can receive contributions from any combination of:

| Source | Entry point | Weight |
|--------|-------------|--------|
| Word lexicon | `score_text(text)` | 1.0 per matched token |
| Emoji characters | `score_emojis(text)` | 1.0 per matched emoji |
| Word + emoji (unified) | `score_mixed(text)` | 1.0 each |
| DeepMoji distribution | `DeepMojiAdapter.score_state(scores)` | probability × emotion |
| HuggingFace classifier | `HFEmotionAdapter.from_scores(scores)` | probability × emotion |
| Cognitive appraisal | `appraisal_to_emotion(appraisal)` | discrete |
| Manual | `state.apply(emotion, weight)` | explicit |

```python
state = EmotionalState()
state.apply(score_text(transcript))                  # word signal
state.apply(score_emojis(transcript))                # emoji signal
state.apply(hf_adapter.from_scores(logits), w=0.5)  # transformer signal
state.apply(appraisal_to_emotion(appraisal), w=2.0) # appraisal (high confidence)
state.decay(0.9)
dominant = state.dominant()
```

This architecture separates signal acquisition (how to get affect data) from signal integration (how to combine it) and from interpretation (what emotion is dominant).

---

## 6. Emoji-Emotion Mapping

### Scientific basis

The `EMOJI_EMOTION_MAP` was constructed by cross-referencing three sources:

1. **Plutchik's wheel** — ensures every entry maps to a named Plutchik primary or intensity variant.
2. **Novak et al. (2015)** — crowd-sourced emoji sentiment and arousal scores, used to assign intensity levels (serenity vs joy vs ecstasy for positive-pleasantness emoji).
3. **Felbo et al. (2017)** — the 64-emoji DeepMoji label set was used to ensure the canonical map covers all emoji likely to appear in DeepMoji output.

### Intensity preservation

The map preserves Hourglass intensity levels where semantically unambiguous:

- 😊 → `serenity` (Pleasantness +1, mild positive)
- 😄 → `joy` (Pleasantness +2, primary)
- 😂 → `ecstasy` (Pleasantness +3, intense)

This allows `score_emojis` output to carry arousal information, not just polarity.

### Runtime extensibility

`register_emoji(char, emotion_name)` allows per-process customisation without modifying the canonical map. User registrations are checked before the canonical map in `_lookup()`.

---

## 7. Evaluation and Coverage

The library ships with 812 unit tests covering:

| Module | Coverage |
|--------|----------|
| `plutchik.py` | 91% |
| `feelings.py` | 98% |
| `composite_emotions.py` | 98% |
| `state.py` | 96% |
| `distance.py` | 100% |
| `appraisal.py` | 100% |
| `float_emotion.py` | 97% |
| `emoji.py` | 100% |
| `text.py` | 78% (HFEmotionAdapter requires `[transformers]` extra) |
| **Overall** | **94%** |

All tests are deterministic and require no external services. The `[transformers]` extra paths are guarded by lazy imports and exercised via mocks in the full test suite.

---

## 8. Limitations and Non-Goals

### 8.1 Discrete lattice is lossy

The 24-point integer lattice cannot represent affect at arbitrary precision. Regression models (VAD predictors, embedding-based classifiers) output continuous values; `closest_emotion()` and `FloatEmotion.from_embedding()` provide projection paths, but information is lost in the discretisation.

### 8.2 kind labels are approximate

`Emotion.kind` and `CompositeEmotion.kind` are manually curated from Plutchik's prose, not derived from the Hourglass axes. They should not be used as ground truth for psychological classification.

### 8.3 Single valence axis

A full PAD (Pleasure–Arousal–Dominance) model would add a third independent axis. Dominance is out of scope for this library — it would require restructuring the entire 4-axis representation.

### 8.4 No audio/video

emotion-algebra is text and emoji only. For multimodal affect recognition, use a dedicated model and bridge via `HFEmotionAdapter` or `DeepMojiAdapter`.

### 8.5 Not clinical

The library models categorical emotions per Plutchik and Cambria. It is not designed for clinical diagnostic use and makes no claims about psychological validity beyond its stated scientific sources.

---

## 9. Use Cases

| Domain | Pattern |
|--------|---------|
| **Dialogue / chatbot mood tracking** | `EmotionalState` + `decay()` + `dominant()` per turn |
| **NPC emotional AI** | `Appraisal` → `appraisal_to_emotion()` → `EmotionalState.apply()` |
| **Sentiment-aware recommendation** | `emotion_distance()`, `closest_emotion()` for similarity ranking |
| **Social media analysis** | `score_mixed()` — handles both prose and emoji |
| **DeepMoji post-processing** | `DeepMojiAdapter.from_scores()` → named emotion |
| **HuggingFace post-processing** | `HFEmotionAdapter.from_scores()` → named emotion |
| **Affective computing research** | `FloatEmotion`, `from_embedding()`, `EmotionTimeline` |
| **Game NPC scripting** | CLI REPL, `info` mode, expression evaluator |

---

## 10. References

- Cambria, E., Livingstone, A., & Hussain, A. (2012). The Hourglass of Emotions. In *Cognitive Behavioural Systems*, LNCS 7403. Springer.
- Felbo, B., Mislove, A., Søgaard, A., Rahwan, I., & Lehmann, S. (2017). Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm. *EMNLP 2017*.
- Novak, P. K., Smailović, J., Sluban, B., & Mozetič, I. (2015). Sentiment of emojis. *PLOS ONE*, 10(12).
- Plutchik, R. (1980). A general psychoevolutionary theory of emotion. In R. Plutchik & H. Kellerman (Eds.), *Emotion: Theory, research, and experience* (Vol. 1, pp. 3–33). Academic Press.
- Posner, J., Russell, J. A., & Peterson, B. S. (2005). The circumplex model of affect: An integrative approach. *Development and Psychopathology*, 17(3), 715–734.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
- Scherer, K. R. (2001). Appraisal considered as a process of multilevel sequential checking. In K. R. Scherer, A. Schorr, & T. Johnstone (Eds.), *Appraisal processes in emotion* (pp. 92–120). Oxford University Press.

---

## Experimental Validation Plan

The claims in this paper fall into two classes. **Internal-consistency claims** (the lattice is composable and introspectable; operators obey their algebraic laws) are formal properties that can be verified by property-based testing without human data — the existing suite already covers most of these. **External-validity claims** (text and emoji affect scores correspond to human emotional experience; appraisal and need-deficit mappings are psychologically plausible) require comparison against human-annotated data and competitive baselines. This section lays out experiments for both classes and flags where new data or human evaluation is needed.

A guiding principle: emotion-algebra is a *representation and post-processing layer*, not a classifier in its own right. Its value proposition is interpretability and composability at a small or no cost in accuracy relative to opaque continuous baselines. The plan is therefore designed to test that trade-off directly, rather than to claim state-of-the-art classification.

### E1 — Operator-algebra property tests (internal consistency)

- **Hypothesis.** The signed-integer lattice and its operators form a well-behaved algebra: intensity shifts (`<<`, `>>`) are monotone and bounded; negation maps each emotion to its Plutchik opposite; addition of opposite-axis components composes without information loss; scalar operations preserve axis identity and `type`.
- **Data.** None required — closed-form over the 24-point lattice and its composites. Enumerable in full.
- **Method.** Extend the existing deterministic unit tests with property-based tests (e.g. Hypothesis) over randomly sampled lattice points and operator sequences, asserting invariants: involution of negation (`-(-e) == e`), opposite-pair table consistency, shift monotonicity and clamping at ±3 (and hyper-intensity beyond), `Feeling`/`CompositeEmotion` naming determinism, and idempotence/commutativity where claimed.
- **Baseline.** The current curated unit tests (812 reported here; concretely `test/test_algebra.py`, `test/test_feeling_operators.py`, `test/test_composite.py`, `test/test_blend.py`, `test/test_distance.py`, `test/test_float_emotion.py`) already assert specific cases such as `annoyance + 1 == anger`, `-anger == fear`, `-joy == sadness`. Property tests generalise these from examples to universally quantified invariants.
- **Metric.** Pass/fail of each invariant; number of operator-sequence samples exercised; line/branch coverage of `plutchik.py`, `composite_emotions.py`, `feelings.py`, `float_emotion.py`, `distance.py`.
- **Success criterion.** 100% of stated invariants hold over ≥10⁵ sampled operator sequences; no counterexamples; coverage of the algebra modules remains ≥95%.
- **New data / human eval needed?** No.

### E2 — Introspectability audit (internal consistency)

- **Hypothesis.** Every value the library can produce — named emotion, `Feeling`, `CompositeEmotion`, `FloatEmotion`, and `EmotionalState.dominant()` output — exposes a human-readable name, an axis/dimension, and a derivable scientific lineage, with the only documented exceptions being the explicitly-flagged approximate `kind` labels (`"gloat"`, `"frivolity"`; AUDIT.md A-001).
- **Data.** None required — exhaustive enumeration of the registries plus a sampled enumeration of composites.
- **Method.** Enumerate `EMOTIONS`, `FEELINGS`, `COMPOSITE_EMOTIONS`, and `EMOJI_EMOTION_MAP`; assert each entry has non-empty name, valid dimension, and resolvable provenance. For composites, sample cross-axis sums and assert every result is either a named `Feeling` or a labelled `CompositeEmotion` (never silently dropped). Cross-check the `kind` approximations against the AUDIT.md exception list.
- **Baseline.** None (this is a structural property, not a comparison).
- **Metric.** Fraction of producible values that are fully introspectable; count of undocumented unnamed/unclassified outputs (target: zero beyond the AUDIT.md exceptions).
- **Success criterion.** Zero undocumented introspection gaps; the only unclassified entries are the two flagged in AUDIT.md.
- **New data / human eval needed?** No.

### E3 — Text affect scoring vs human annotations (external validity)

- **Hypothesis.** Emotions scored by `from_text` / `score_text` agree with human emotion labels at a level competitive with standard lexicon and classifier baselines, while remaining categorically interpretable.
- **Data.** Public labelled affect corpora — categorical (e.g. ISEAR's seven emotions; the SemEval-2018 Task 1 affect-in-tweets emotion set) and dimensional (e.g. EmoBank valence–arousal; the NRC VAD lexicon for word-level checks). Map Plutchik primaries to dataset label sets via a fixed, pre-registered crosswalk (documented before scoring to avoid post-hoc tuning). Datasets are pulled through the standard HF datasets cache where available.
- **Method.** Score each item, take `EmotionalState.dominant()` (or top-k) as the predicted category and the Pleasantness component as predicted valence / `arousal` as predicted arousal. Evaluate categorical agreement on the label crosswalk and correlation on the dimensional corpora. Report per-emotion and macro figures.
- **Baseline.** (a) Coarse pos/neg/neutral sentiment (VADER or the bundled sentiment lexicon) projected onto valence sign; (b) a VAD-regression model (e.g. a small fine-tuned or off-the-shelf VAD predictor) for the dimensional comparison; (c) a transformer emotion classifier consumed through `HFEmotionAdapter` as an upper-reference, to quantify the interpretability/accuracy trade-off.
- **Metric.** Macro-F1 and per-class F1 for categorical agreement; Cohen's/Krippendorff's measures against the human consensus label; Pearson/Spearman correlation between predicted and gold valence and arousal.
- **Success criterion.** Categorical macro-F1 within a pre-registered margin (target: ≥0.85× the transformer reference) while strictly beating the coarse pos/neg/neutral baseline; valence correlation ≥0.5 and arousal correlation positive and significant. Falling short on arousal would itself be an informative result given the lexicon's polarity bias.
- **New data / human eval needed?** No new annotation if the public corpora and a fixed crosswalk suffice; the label crosswalk should be reviewed by an annotator with affect-science background and is the main subjective design choice to flag.

### E4 — Emoji affect scoring and intensity preservation (external validity)

- **Hypothesis.** `score_emojis` / `from_emoji` recovers both polarity and *intensity ordering* (serenity < joy < ecstasy) consistent with crowd-sourced emoji sentiment, and `DeepMojiAdapter` faithfully converts DeepMoji-style distributions into the matching named emotions.
- **Data.** Novak et al. (2015) emoji sentiment ranking (already a construction source — used here only for held-out validation of intensity ordering, not refitting); a labelled emoji-bearing tweet set for end-to-end checks; synthetic DeepMoji-style probability distributions with known dominant emoji for adapter conformance.
- **Method.** For each mapped emoji, compare emotion-algebra's assigned valence sign and intensity rank against the Novak sentiment/arousal scores (Spearman rank correlation on intensity). For the adapter, feed distributions and assert the recovered dominant emotion matches the argmax mapping and that mixed distributions accumulate proportionally to probability.
- **Baseline.** A flat polarity-only emoji lexicon (collapsing all positive emoji to +1) — tests whether intensity preservation adds measurable signal over polarity alone.
- **Metric.** Spearman correlation of predicted vs crowd intensity ordering; sign-agreement rate; adapter exact-match rate on dominant emotion; proportionality error on mixed distributions.
- **Success criterion.** Intensity-ordering Spearman ≥0.6 and significantly above the flat-polarity baseline; adapter dominant-emotion exact match ≥0.95 on synthetic distributions.
- **New data / human eval needed?** No for the Novak comparison; a small labelled emoji-tweet set may need sourcing for the end-to-end check.

### E5 — Appraisal mapping plausibility (external validity)

- **Hypothesis.** The 16-rule `appraisal_to_emotion` table reproduces the emotion that Scherer-CPM theory predicts for a given appraisal pattern, and human raters agree the produced emotion is the most plausible for that pattern.
- **Data.** Canonical appraisal-pattern → emotion mappings from the CPM literature for the covered emotions (theory-derived gold). For human validation: a set of short scenario vignettes, each hand-coded into an `Appraisal`, with crowd ratings of the most fitting emotion.
- **Method.** Run each gold appraisal pattern through `appraisal_to_emotion` and check against the theory-predicted emotion. For vignettes, compare the library's output to the modal human-rated emotion.
- **Baseline.** A random/majority-class emotion assignment over the same label set.
- **Metric.** Exact-match accuracy vs theory; agreement (e.g. Cohen's κ) with modal human rating on vignettes.
- **Success criterion.** ≥0.9 exact match against the theory-derived table (this part is essentially a specification conformance test); human-agreement κ meaningfully above chance and above the random baseline.
- **New data / human eval needed?** **Yes** — the vignette set and its crowd ratings are new data and require human evaluation. The theory-conformance half needs none. Note the table covers a subset of emotions; vignettes must stay within that subset.

### E6 — Need-deficit mapping plausibility (external validity)

- **Hypothesis.** The Max-Neef / Murray need → deficit-emotion mappings (e.g. unmet protection → fear, unmet subsistence → sadness, unmet freedom → anger, unmet creation → boredom) match human intuition about how an unmet need feels.
- **Data.** The library's own need→emotion table as the system under test (see `test/test_needs.py`); for human validation, a forced-choice survey pairing each need-deficit description with candidate emotions.
- **Method.** Existing unit tests already pin the table to specific assertions (e.g. `protection` deficit → fear, `freedom` → anger). The new experiment asks human raters which emotion best fits each described deficit and compares the modal human choice to the table.
- **Baseline.** Random emotion assignment; optionally a generic "most needs feel bad → sadness" constant baseline.
- **Metric.** Agreement rate / κ between table and modal human choice; per-need confusion matrix.
- **Success criterion.** Table-vs-human agreement significantly above both baselines on a majority of needs; documented disagreements treated as candidate table revisions rather than failures.
- **New data / human eval needed?** **Yes** — the forced-choice survey is new human-eval data. The table-conformance portion is already covered by `test/test_needs.py`.

### E7 — Classifier post-processing A/B (external validity, end-use)

- **Hypothesis.** Routing a black-box emotion/sentiment classifier's output through emotion-algebra (via `HFEmotionAdapter` / `DeepMojiAdapter`, optional `EmotionalState` smoothing and `decay`) preserves or improves downstream task performance while adding interpretability and temporal stability — i.e. interpretability is not bought with accuracy.
- **Data.** A multi-turn conversational or streaming-text affect dataset with turn-level or document-level emotion labels (e.g. a dialogue emotion corpus), so that `EmotionalState` accumulation and `decay` are meaningfully exercised.
- **Method.** A/B comparison. **Arm A:** raw classifier argmax per turn. **Arm B:** identical classifier piped through `HFEmotionAdapter.from_scores` into an `EmotionalState` with decay, taking `dominant()` per turn. Same underlying model in both arms — the only variable is the post-processing layer.
- **Baseline.** Arm A (raw classifier) is the baseline; additionally compare against a naïve exponential-moving-average smoother over raw logits to isolate the contribution of the algebraic structure beyond simple smoothing.
- **Metric.** Turn-level macro-F1; temporal stability / label-churn rate across adjacent turns; a held-out human judgement of which arm's emotion trajectory is more coherent.
- **Success criterion.** Arm B's F1 is non-inferior to Arm A (within a pre-registered margin) **and** shows lower label churn / higher rated trajectory coherence — demonstrating the interpretability layer is free or beneficial. A pure accuracy drop beyond the margin would be a negative result worth reporting.
- **New data / human eval needed?** Possibly — public dialogue-emotion corpora may suffice for the quantitative metrics; the trajectory-coherence judgement is a small human eval to flag.

### Summary of what requires new data or human evaluation

| Experiment | New data | Human eval |
|---|---|---|
| E1 Operator-algebra properties | No | No |
| E2 Introspectability audit | No | No |
| E3 Text affect vs human labels | Reuse public corpora | Crosswalk review only |
| E4 Emoji scoring & intensity | Maybe (emoji-tweet set) | No |
| E5 Appraisal plausibility | Yes (vignettes) | Yes |
| E6 Need-deficit plausibility | Yes (survey) | Yes |
| E7 Post-processing A/B | Maybe (dialogue corpus) | Small (trajectory coherence) |

E1 and E2 are the internal-consistency backbone and can run today against the existing suite. E3, E4, and E7 lean on public datasets and standard baselines. E5 and E6 are the only experiments that strictly require fresh human-rated data to validate the appraisal and need-deficit tables; until those run, those two mappings should continue to be described as theory-derived and approximate.
