# emotion_data

Python library implementing **emotion algebra** — emotions as first-class mathematical objects.

## Model provenance

The library blends two distinct academic frameworks:

| Model | Authors | Role in this library |
|---|---|---|
| **Plutchik's Wheel of Emotions** (1980) | Robert Plutchik | Named dyad composition (joy+trust→love), opposite pairs, 8-primary structure |
| **Hourglass of Emotions** (2012) | Cambria, Livingstone, Hussain | 4-axis signed-integer scale (PASA), `emotional_flow`, `EmotionalDimension` |

**Known departures and limitations:**
- `Emotion.type` and `CompositeEmotion.type` use hand-crafted heuristics, not model derivations.
- `EmotionalDimension.valence` is a simplification (sensitivity→−1, attention→+1, others→0).
- Valence is a single signed integer. Full PAD (Pleasure–Arousal–Dominance) is out of scope.
- Behaviours map directly from emotions; the cognitive appraisal layer (Lazarus 1991) is omitted.

## Installation

```bash
pip install emotion_data            # core (numpy only)
pip install "emotion_data[lexicon]" # + word-emotion CSV lookup
pip install "emotion_data[tagging]" # + ParallelDots API tagging
```

## Algebra

Emotions support arithmetic operators that move along the Hourglass intensity axis.

### Intensity arithmetic

| Expression | Result | Notes |
|---|---|---|
| `annoyance + 1` | `anger` | primary → secondary |
| `anger + 1` | `rage` | secondary → tertiary |
| `rage + 1` | `mega rage` | offset = 1 |
| `rage + 3` | `hyper rage` | offset = 3 |
| `anger - 1` | `annoyance` | down one step |
| `serenity - 1` | `Neutrality` | below primary → neutral |

```python
from emotion_data.emotions import EMOTIONS
from copy import copy

anger = copy(EMOTIONS["anger"])
print(anger + 1)   # rage
print(anger - 1)   # annoyance
print(-anger)      # fear
```

### Negation (opposite emotion)

```python
-anger      # fear
-joy        # sadness
-trust      # disgust
-rage       # terror
```

### Composition (Feeling dyads)

Adding two emotions from **different** dimensions produces a named `Feeling`:

```python
from emotion_data.emotions import EMOTIONS
from copy import copy

joy   = copy(EMOTIONS["joy"])
trust = copy(EMOTIONS["trust"])

love = joy + trust
print(love.name)   # "love"
print(type(love))  # <class 'emotion_data.feelings.Feeling'>
```

Selected dyads:

| Expression | Feeling |
|---|---|
| `joy + trust` | love |
| `joy + surprise` | delight |
| `anticipation + trust` | hope |
| `fear + sadness` | despair |
| `anticipation + joy` | optimism |

## Emotion Taxonomy

The Hourglass of Emotions defines four axes, each with three intensity levels per polarity.

| Dimension | Intense (+) | Secondary (+) | Primary (+) | Primary (−) | Secondary (−) | Intense (−) |
|---|---|---|---|---|---|---|
| **pleasantness** | ecstasy | joy | serenity | pensiveness | sadness | grief |
| **aptitude** | admiration | trust | acceptance | boredom | disgust | loathing |
| **sensitivity** | rage | anger | annoyance | apprehension | fear | terror |
| **attention** | vigilance | anticipation | interest | distraction | surprise | amazement |

## 4D Emotion Vectors

Every emotion has a 4D vector representation (one value per Hourglass axis):

```python
from emotion_data.emotions import EMOTIONS
from copy import copy
import numpy as np

joy = copy(EMOTIONS["joy"])
print(joy.emotion_vector)  # [Neutrality, Neutrality, joy, Neutrality]
print(joy.as_array)        # np.array([0, 0, 2, 0])
```

## Comparison

```python
rage  = copy(EMOTIONS["rage"])
anger = copy(EMOTIONS["anger"])

rage > anger    # True  (flow 3 > 2)
anger > annoyance  # True  (flow 2 > 1)
```

## Key Classes

| Class | File | Role |
|---|---|---|
| `Emotion` | `emotion_data/plutchik.py` | Core algebra carrier |
| `Neutrality` | `emotion_data/plutchik.py` | Identity element |
| `EmotionalDimension` | `emotion_data/plutchik.py` | Hourglass axis |
| `Feeling` | `emotion_data/feelings.py` | Named two-emotion dyad |
| `CompositeEmotion` | `emotion_data/composite_emotions.py` | Cross-axis composite |
| `Behaviour` | `emotion_data/behaviour.py` | Adaptive reaction |
| `BehavioralReaction` | `emotion_data/behaviour.py` | Cognitive appraisal mapping |
