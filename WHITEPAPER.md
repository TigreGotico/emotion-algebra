# emotion-algebra: A Computational Framework for Affective Arithmetic

**Version**: 1.7
**Date**: 2026-03-30
**License**: Apache 2.0

---

## Abstract

We present **emotion-algebra**, a Python library that treats emotions as first-class algebraic objects. Built on Cambria's Hourglass of Emotions (2012) and Plutchik's Wheel (1980), the library defines a signed integer lattice over four affective axes and implements a complete operator algebra over that lattice. The result is a composable, introspectable, and numerically grounded affective representation suitable for dialogue systems, NPC AI, affective computing research, and classifier post-processing. We describe the theoretical foundations, the design decisions that depart from or extend the source models, and the practical APIs for text analysis, emoji mapping, cognitive appraisal, and continuous-space embedding.

---

## 1. Introduction

Computational systems that reason about affect typically face one of two failure modes: representations that are too coarse (positive/negative/neutral) to capture the richness of human emotional experience, or representations that are so fine-grained (e.g. raw VAD vectors from a regression model) that they resist symbolic manipulation, introspection, and rule-based combination.

emotion-algebra occupies the middle ground. It encodes a 24-point discrete lattice — the named emotions of the Hourglass of Emotions — with sufficient algebraic structure to support arithmetic, geometry, temporal dynamics, and multi-signal fusion, while remaining fully introspectable: every value has a human-readable name, a dimension, and a clear scientific lineage.

The library serves three user groups:

1. **Application developers** who need a structured, composable affect layer for chatbots, game NPCs, or recommendation engines — without writing their own emotion representation.
2. **NLP researchers** who want to post-process classifier output (word embeddings, emoji distributions, transformer logits) into an interpretable affective space.
3. **Affective computing researchers** who want a reference implementation of Plutchik/Cambria semantics in Python, with tests and documented deviations from the original models.

---

## 2. Theoretical Foundations

### 2.1 Cambria's Hourglass of Emotions (2012)

The Hourglass of Emotions (Cambria, Livingstone, & Hussain, 2012) extends Plutchik's wheel into a four-dimensional signed integer space. Each dimension (called an *axis*) represents an independent affective component:

| Axis | Positive pole (flow > 0) | Negative pole (flow < 0) |
|------|--------------------------|--------------------------|
| **Sensitivity** | Rage (3) → Anger (2) → Annoyance (1) | Apprehension (−1) → Fear (−2) → Terror (−3) |
| **Attention** | Vigilance (3) → Anticipation (2) → Interest (1) | Distraction (−1) → Surprise (−2) → Amazement (−3) |
| **Pleasantness** | Ecstasy (3) → Joy (2) → Serenity (1) | Pensiveness (−1) → Sadness (−2) → Grief (−3) |
| **Aptitude** | Admiration (3) → Trust (2) → Acceptance (1) | Boredom (−1) → Disgust (−2) → Loathing (−3) |

The integer magnitude encodes *intensity*: ±1 is mild, ±2 is primary, ±3 is intense. Values beyond ±3 are supported as hyper-intensity offsets (e.g. "hyper rage" = Sensitivity(5)).

This design makes the Hourglass a metrizable space: distances, clusters, and projections are well-defined operations over the 4-dimensional integer lattice.

### 2.2 Plutchik's Wheel of Emotions (1980)

Plutchik's wheel (1980) provides:

- **Eight primary emotions** at three intensity levels each — the named emotions of the Hourglass axes.
- **Dyadic feelings** (*primary dyads*) — named combinations of two adjacent primary emotions (e.g. joy + trust = love, anger + anticipation = aggressiveness).
- **Opposite pairs** — each primary emotion has a diametrically opposite counterpart (joy ↔ sadness, anger ↔ fear, etc.).

In emotion-algebra, `Feeling` encodes named dyads; `CompositeEmotion` encodes the algebraic result of cross-axis combination without necessarily matching a named dyad.

### 2.3 Russell's Circumplex Model (1980)

The Russell Circumplex (1980) organises affect into a 2D valence × arousal plane with six qualitative regions. emotion-algebra maps the Hourglass axes onto this plane:

- **Valence** = Pleasantness axis component only (Posner et al. 2005: arousal and hedonics are orthogonal).
- **Arousal** = `max(|flow|)` across all active axes — axis-independent activation intensity.
- **Type** = one of: `"excited positive"`, `"calm positive"`, `"excited negative"`, `"calm negative"`, `"activated neutral"`, `"neutral"`.

The key design decision here — that `anger.valence == 0` despite anger being colloquially "negative" — is scientifically grounded: anger is a Sensitivity-axis emotion; its hedonic content is zero. Only Pleasantness-axis emotions carry hedonic information.

### 2.4 Scherer's Component Process Model (2001)

Scherer's CPM (2001) models emotion as the output of sequential cognitive appraisal across five dimensions: novelty, goal-relevance, goal-congruence, agency, and coping potential. emotion-algebra implements a simplified 16-rule deterministic mapping from appraisal patterns to primary emotions (`appraisal.py`), enabling causally grounded emotion inference without requiring a full appraisal simulator.

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

The library ships with 713 unit tests covering:

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

All tests are deterministic and require no external services. The `[transformers]` extra paths are guarded by lazy imports and tested via mock in the full test suite.

---

## 8. Limitations and Non-Goals

### 8.1 Discrete lattice is lossy

The 24-point integer lattice cannot represent affect at arbitrary precision. Regression models (VAD predictors, embedding-based classifiers) output continuous values; `closest_emotion()` and `FloatEmotion.from_embedding()` provide projection paths, but information is lost.

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

*This whitepaper describes emotion-algebra v1.7. Generated with Claude Sonnet 4.6.*
