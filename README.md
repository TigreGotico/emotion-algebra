# emotion-algebra

Signed integer arithmetic over a 4-axis affective space, grounded in Plutchik's Wheel of Emotions (1980) and Cambria's Hourglass of Emotions (2012). Emotions are first-class mathematical objects; every operator returns a typed result.

```python
from emotion_algebra.emotions import get_emotion

anger = get_emotion("anger")

anger + 1          # → rage          (intensity up)
anger - 1          # → annoyance     (intensity down)
-anger             # → fear          (opposite pole)
anger >> 2         # → hyper anger
anger * fear       # → CompositeEmotion (cross-axis product)
```

---

## Installation

```bash
pip install emotion-algebra
```

Optional extras:

| Extra | Deps | Enables |
|-------|------|---------|
| `[lexicon]` | pandas | `lexicons.py` — word→emotion CSV lookup |
| `[fast]` | ahocorasick-ner | Phrase-aware Aho-Corasick backend for `score_text`/`from_text` |
| `[viz]` | matplotlib | `viz.py` — Plutchik wheel, circumplex, timeline, Lövheim cube |
| `[all]` | all of the above | everything |

`deepmoji-onnx` is a **core dependency** — neural text→emoji→emotion scoring via `DeepMojiONNXAdapter` is always available.

---

## The model

### Four axes (Cambria 2012)

| Axis | Positive pole | Negative pole | Hedonic? |
|------|--------------|---------------|----------|
| Sensitivity | rage → anger → annoyance | apprehension → fear → terror | No |
| Attention | vigilance → anticipation → interest | distraction → surprise → amazement | No |
| Pleasantness | ecstasy → joy → serenity | pensiveness → sadness → grief | **Yes** |
| Aptitude | admiration → trust → acceptance | boredom → disgust → loathing | **Yes** |

Each axis has three integer intensity levels: ±1 (mild), ±2 (primary), ±3 (intense).

### Valence, polarity, and arousal

- **Valence** — the Pleasantness axis alone, the hedonic axis. `anger.valence == 0`; `joy.valence == 2`.
- **Polarity** — the overall sentiment score, using Cambria's published four-axis
  formula `(P + |At| − |S| + Ap) / 3`. `trust.valence == 0` but `trust.polarity == +0.22`.
- **Arousal** — `abs(emotional_flow)`. Axis-independent activation intensity.
- **Type** — Russell Circumplex quadrant, classified on **polarity**.

Valence and polarity are deliberately different quantities — see
[docs/valence_arousal.md](docs/valence_arousal.md).

### Feelings (Plutchik dyads)

Adding two emotions on *different* axes builds a `CompositeEmotion`. The named
dyad is a `Feeling`, looked up from its two components:

```python
from emotion_algebra.emotions import get_emotion
from emotion_algebra.feelings import get_feeling_from_emotions

joy   = get_emotion("joy")
trust = get_emotion("trust")

joy + trust                              # → CompositeEmotion (the raw cross-axis sum)
get_feeling_from_emotions("joy", "trust")  # → 'love'          (the named dyad)
```

Ten names are claimed by both the Feeling and CompositeEmotion lineages. Use
`resolve()` — it prefers the Feeling and the preference is explicit, not an
accident of import order. See [docs/taxonomy.md](docs/taxonomy.md).

---

## Feature overview

| Feature | API | Module |
|---------|-----|--------|
| Core emotion algebra | `Emotion`, `+`, `-`, `*`, `<<`, `>>` | `plutchik.py` |
| Named feelings (dyads) | `Feeling`, `get_feeling()` | `feelings.py` |
| Multi-axis composites | `CompositeEmotion` | `composite_emotions.py` |
| Continuous space | `FloatEmotion`, `from_embedding()` | `float_emotion.py` |
| Stateful accumulation | `EmotionalState`, `EmotionTimeline` | `state.py` |
| Word lexicon | `from_text()`, `score_text()` | `text.py` |
| Emoji mapping | `from_emoji()`, `score_emojis()`, `DeepMojiAdapter` | `emoji.py` |
| Mixed word+emoji | `score_mixed()`, `from_mixed()` | `text.py` |
| Geometry | `emotion_distance()`, `closest_emotion()` | `distance.py` |
| Cognitive appraisal | `Appraisal`, `appraisal_to_emotion()` | `appraisal.py` |
| Continuous appraisal | `appraisal_to_float_emotion()`, `appraisal_to_lovheim()` | `appraisal.py` |
| Emotion blending | `FloatEmotion.blend(joy, trust)` | `float_emotion.py` |
| Need-deficit emotions | `need_deficit_to_emotion()`, `CIADrive`, `MaxNeefNeed`, `MurrayNeed` | `needs.py` |
| Behavioural reactions | `REACTIONS`, `REACTION_TO_EMOTION_MAP` | `behaviour.py` |
| **Neurochemistry** | `LovheimPoint`, `closest_affect()`, `affect_blend()` | `lovheim.py` |
| **PAD / VAD interop** | `to_pad()`, `from_pad()`, `pad_distance()` | `pad.py` |
| **Name resolution** | `resolve()`, `is_ambiguous()`, `collision_table()` | `taxonomy.py` |
| **Similarity** | `emotion_similarity()` (distance or cosine) | `distance.py` |
| **Serialization** | `to_dict()`/`from_dict()`, `to_json()`/`from_json()`, versioned schema | `serialization.py` |
| **Text analysis** | `analyze()` — spans, negation, intensifiers | `text.py` |
| **Plots** | `plot_wheel()`, `plot_circumplex()`, `plot_timeline()`, `plot_lovheim()` | `viz.py` |
| CLI | `python -m emotion_algebra` | `__main__.py` |

---

## Highlights

### Neurochemistry — Lövheim's cube

Three monoamine axes whose eight corners are Tomkins' basic affects
(Lövheim 2012). Bidirectionally bridged to the Hourglass space, so an agent's
emotional state has a live neurochemical readout:

```python
from emotion_algebra import EmotionalState, get_emotion

state = EmotionalState()
state.apply(get_emotion("ecstasy"))

state.to_lovheim()                  # LovheimPoint(serotonin=1, dopamine=1, noradrenaline=0)
state.to_lovheim().closest_affect() # 'enjoyment/joy'
state.to_lovheim().affect_blend()   # exact distribution over all eight affects
```

The module documents **two properties it provably cannot have** — it cannot reach
positive Aptitude (Tomkins has no trust affect), and the bridge is not injective
(Plutchik's opposites are exact negatives, and Lövheim puts them at *adjacent*
corners). Both are theorems, and both are locked by tests.
See [docs/lovheim.md](docs/lovheim.md).

### Text analysis with valence shifters

```python
from emotion_algebra.text import analyze

analyze("I am joyful").dominant.name            # 'joy'
analyze("I am not joyful").dominant.name        # 'sadness'    (negation)
analyze("I am very joyful").dominant.name       # 'ecstasy'    (intensifier)
analyze("I am slightly joyful").dominant.name   # 'serenity'   (downtoner)
```

Windowed negation and intensifier handling, following Taboada et al. (2011).

### It is an actual algebra

The laws are written down in [docs/laws.md](docs/laws.md) and machine-checked
with `hypothesis` — identity, involution of negation, associativity (and exactly
where it fails), metric axioms, conversion coherence.

---

## Quick reference

```python
from emotion_algebra import (
    get_emotion, get_feeling,
    EmotionalState, EmotionTimeline,
    from_text, score_text, score_mixed, from_mixed,
    from_emoji, score_emojis, DeepMojiAdapter,
    register_emoji, unregister_emoji,
    emotion_distance, closest_emotion, emotion_clusters,
    Appraisal, appraisal_to_emotion, appraisal_to_float_emotion,
    float_emotion_to_neuro_deltas,
    FloatEmotion,
    CIADrive, MaxNeefNeed, MurrayNeed,
    need_deficit_to_emotion, need_deficit_to_float_emotion,
)

# --- Emotion properties ---
e = get_emotion("joy")
e.emotional_flow   # 2
e.valence          # 2    (pleasantness axis)
e.arousal          # 2
e.type             # "excited positive"
e.opposite_emotion # sadness
e.as_array         # np.array([0, 0, 2, 0])

# --- State accumulation ---
state = EmotionalState()
state.apply(get_emotion("joy"), weight=0.8)
state.apply(get_emotion("trust"), weight=0.5)
state.decay(0.9)
state.dominant()   # → Emotion or None

# --- Text analysis ---
from_text("rage and fury")          # → Emotion (lexicon)
score_mixed("I'm so happy 😄🎉")   # → EmotionalState (words + emoji)
from_mixed("grief 😭")             # → dominant Emotion

# --- Emoji ---
from_emoji("😊")                   # → Emotion("serenity")
register_emoji("🤖", "trust")      # custom mapping

# --- DeepMoji bridge ---
adapter = DeepMojiAdapter()
adapter.from_scores({"😂": 0.6, "😭": 0.4})  # → Emotion
adapter.score_state({"😂": 0.6, "😭": 0.4})  # → EmotionalState

# --- Geometry ---
a, b = get_emotion("anger"), get_emotion("joy")
emotion_distance(a, b)             # Euclidean distance in 4D Hourglass space
closest_emotion([2, 0, 1, 0])     # nearest named Emotion to a float vector

# --- Cognitive appraisal (Scherer CPM) ---
# Discrete (v1): categorical fields → named Emotion
a = Appraisal(goal_relevance="relevant", goal_congruence="incongruent",
              agency="other", coping_potential="low")
appraisal_to_emotion(a)           # → fear

# Continuous (v2): float fields → FloatEmotion → neuro deltas
a = Appraisal(novelty=0.8, goal_relevance=0.9, goal_congruence=0.3,
              coping_potential=0.2, intrinsic_pleasantness=0.4)
fe = appraisal_to_float_emotion(a)
d, s, adr = float_emotion_to_neuro_deltas(fe)  # (dopamine, serotonin, adrenaline)

# --- Emotion blending ---
joy   = get_emotion("joy")
trust = get_emotion("trust")
FloatEmotion.blend(joy, trust)                 # → pleasantness + aptitude
FloatEmotion.blend(joy, trust, weights=[0.7, 0.3], scale=0.8)

# --- Need-deficit emotions (Max-Neef + Murray) ---
need_deficit_to_emotion(MaxNeefNeed.PROTECTION)  # → fear
need_deficit_to_emotion(MurrayNeed.NURTURANCE)   # → sadness
need_deficit_to_float_emotion("freedom")          # → FloatEmotion (anger axis)

# --- Continuous space ---
FloatEmotion(sensitivity=1.5, pleasantness=-0.8)
FloatEmotion.from_embedding(np.array([0.3, -0.1, 0.7, 0.2]))
```

---

## CLI

```bash
# Info about an emotion, feeling, or dimension
python -m emotion_algebra info anger
python -m emotion_algebra info love
python -m emotion_algebra info pleasantness

# Evaluate an expression
python -m emotion_algebra "joy + trust"
python -m emotion_algebra "rage - 1"

# Single emoji
python -m emotion_algebra 😊

# Interactive REPL (all 24 emotions pre-loaded)
python -m emotion_algebra
```

---

## Scientific references

- Plutchik, R. (1980). *A general psychoevolutionary theory of emotion.* In R. Plutchik & H. Kellerman (Eds.), *Emotion: Theory, research, and experience* (Vol. 1, pp. 3–33).
- Cambria, E., Livingstone, A., & Hussain, A. (2012). *The Hourglass of Emotions.* In A. Esposito et al. (Eds.), *Cognitive Behavioural Systems*, LNCS 7403.
- Russell, J. A. (1980). *A circumplex model of affect.* Journal of Personality and Social Psychology, 39(6), 1161–1178.
- Posner, J., Russell, J. A., & Peterson, B. S. (2005). *The circumplex model of affect: An integrative approach.* Development and Psychopathology, 17(3), 715–734.
- Felbo, B., Mislove, A., Søgaard, A., Rahwan, I., & Lehmann, S. (2017). *Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm.* EMNLP 2017.
- Scherer, K. R. (2001). *Appraisal considered as a process of multilevel sequential checking.* In K. R. Scherer et al. (Eds.), *Appraisal processes in emotion* (pp. 92–120).
- Max-Neef, M. (1991). *Human Scale Development: Conception, Application and Further Reflections.* Apex Press.
- Murray, H. A. (1938). *Explorations in Personality.* Oxford University Press.
- Lövheim, H. (2012). *A new three-dimensional model for emotions and monoamine neurotransmitters.* Medical Hypotheses, 78(2), 341–348.

---

## License

Apache 2.0
