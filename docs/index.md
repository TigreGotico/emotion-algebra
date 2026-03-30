# emotion_algebra

Python library implementing **emotion algebra** — emotions as first-class mathematical objects,
grounded in Cambria's Hourglass of Emotions and Plutchik's Wheel.

## Documentation

| Page | Contents |
|------|----------|
| [Taxonomy](taxonomy.md) | All 24 named emotions, axes, intensity levels |
| [Algebra](algebra.md) | Operators: intensity arithmetic, negation, composition |
| [Valence, Arousal & Type](valence_arousal.md) | Scientific properties; Russell Circumplex |
| [Feelings](feelings.md) | Named dyads, Feeling vs CompositeEmotion |
| [API Reference](api_reference.md) | All classes, properties, operators |

## Model provenance

| Model | Authors | Role |
|-------|---------|------|
| **Hourglass of Emotions** (2012) | Cambria, Livingstone, Hussain | 4-axis signed-integer representation (PASA), `emotional_flow` |
| **Plutchik's Wheel** (1980) | Robert Plutchik | Named dyads, opposite pairs, 8-primary structure |
| **Russell's Circumplex** (1980) | James Russell | `type` classification via valence × arousal quadrants |

**Key design decisions (per [SPECIFICATION.md](../SPECIFICATION.md)):**
- `Emotion.valence` = Pleasantness axis only — anger/fear valence = 0 (arousal ⊥ hedonics, Posner 2005)
- `Emotion.arousal` = `|emotional_flow|` — axis-independent activation intensity
- `Emotion.type` uses Russell's Circumplex (6 categories)
- Cross-axis `+` returns `CompositeEmotion`; `Feeling` is a separate named construct

## Installation

```bash
pip install emotion_algebra              # core (numpy only)
pip install "emotion_algebra[lexicon]"   # + word→emotion CSV lookup (pandas)
pip install "emotion_algebra[tagging]"   # + ParallelDots API tagging
```

## Quick start

```python
from emotion_algebra.emotions import EMOTIONS
from copy import copy

anger = copy(EMOTIONS["anger"])
print(anger + 1)       # rage
print(anger - 1)       # annoyance
print(-anger)          # fear
print(anger.valence)   # 0  (Sensitivity axis — not hedonic)
print(anger.arousal)   # 2
print(anger.type)      # "activated neutral"

joy = copy(EMOTIONS["joy"])
print(joy.valence)     # 2  (Pleasantness +2)
print(joy.type)        # "excited positive"

composite = joy + copy(EMOTIONS["trust"])
print(composite.name)  # "love"
print(composite.type)  # "excited positive"
```

## Key classes

| Class | Module | Role |
|-------|--------|------|
| `Emotion` | `plutchik.py` | Core algebra carrier — single-axis lattice point |
| `Neutrality` | `plutchik.py` | Identity element (`e + Neutrality() == e`) |
| `EmotionalDimension` | `plutchik.py` | One Hourglass axis |
| `CompositeEmotion` | `composite_emotions.py` | Multi-axis state (result of cross-axis `+`) |
| `Feeling` | `feelings.py` | Named cultural label for a dyad (Plutchik) |
| `Behaviour` / `BehavioralReaction` | `behaviour.py` | Adaptive reaction mapping |
