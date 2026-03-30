# Feelings

A `Feeling` is a **named cultural label** for a multi-axis emotional state —
Plutchik's primary dyads and extensions.

## Feeling vs CompositeEmotion

| | `CompositeEmotion` | `Feeling` |
|-|--------------------|-----------|
| Created by | `emotion1 + emotion2` (cross-axis) | `Feeling("love")` or `get_feeling()` |
| Type | algebraic representation | psychological/cultural label |
| Named dyad lookup | `COMPOSITE_EMOTIONS_NAMES` | `FEELING_NAMES` |
| Inherits | `Emotion` (operator algebra) | standalone |

The `+` operator always returns `CompositeEmotion`.  `Feeling` is constructed
explicitly.

## Named dyads (selection)

| Expression | Feeling name |
|------------|-------------|
| `joy + trust` | love |
| `joy + surprise` | delight |
| `anticipation + trust` | hope |
| `fear + sadness` | despair |
| `anticipation + joy` | optimism |
| `sadness + disgust` | remorse |
| `trust + fear` | submission |
| `anticipation + fear` | anxiety |

Full list: `FEELING_NAMES` in `emotion_data/feelings.py`.

## Creating a Feeling

```python
from emotion_data.feelings import FEELINGS, Feeling
from copy import copy

love = copy(FEELINGS["love"])
print(love.name)     # "love"
print(love.valence)  # 2  (joy's Pleasantness +2)
print(love.arousal)  # 2  (max |flow| across components)
print(love.type)     # "excited positive"
```

## Feeling properties

| Property | Definition |
|----------|-----------|
| `valence` | sum of `e.valence` for all component emotions (Pleasantness contributions only) |
| `arousal` | `max(e.arousal)` across components |
| `type` | Russell Circumplex classification using `valence` and `arousal` |
| `emotional_flow` | sum of all component flows (signed net activation) |
| `emotion_vector` | 4-element list `[sensitivity, attention, pleasantness, aptitude]` |

## Opposite feelings

`OPPOSITE_FEELINGS_NAMES` maps each feeling to its psychological opposite:

```python
from emotion_data.feelings import FEELINGS, OPPOSITE_FEELINGS_NAMES

print(OPPOSITE_FEELINGS_NAMES["love"])       # "remorse"
print(OPPOSITE_FEELINGS_NAMES["optimism"])   # "disapproval"
```
