# Emotion Algebra

All arithmetic operates on the signed integer axis of the Hourglass model.
Each emotion occupies one of 24 named lattice points across four axes.

## Intensity arithmetic

Moving along an axis upgrades or downgrades intensity:

```
annoyance (+1) → anger (+2) → rage (+3) → mega rage (offset 1) → hyper rage (offset ≥3)
```

| Operator | Example | Result |
|----------|---------|--------|
| `+ int` | `annoyance + 1` | `anger` |
| `- int` | `anger - 1` | `annoyance` |
| `+ int` past ±3 | `rage + 1` | `mega rage` |
| `/ int` | `rage / 3` | `annoyance` |
| `// int` | `rage // 2` | `annoyance` |
| `<< int` | `anger << 1` | `annoyance` |
| `>> int` | `annoyance >> 1` | `anger` |

## Negation — opposite pole

`-emotion` flips the sign on the same axis, returning the symmetric opposite:

| Expression | Result | Axis |
|------------|--------|------|
| `-anger` | `fear` | Sensitivity |
| `-joy` | `sadness` | Pleasantness |
| `-trust` | `disgust` | Aptitude |
| `-rage` | `terror` | Sensitivity |
| `-ecstasy` | `grief` | Pleasantness |

## Same-axis addition

Two emotions on the same axis sum their flows:

```python
annoyance + annoyance  # flow 1+1 = 2 → anger
anger + anger          # flow 2+2 = 4 → mega rage (offset 1)
joy + serenity         # flow 2+1 = 3 → ecstasy
```

## Cross-axis composition

Two emotions on **different** axes produce a `CompositeEmotion`:

```python
from emotion_algebra.emotions import EMOTIONS
from copy import copy

joy   = copy(EMOTIONS["joy"])
trust = copy(EMOTIONS["trust"])

composite = joy + trust   # CompositeEmotion spanning Pleasantness + Aptitude
composite.name            # → "love" (named dyad from COMPOSITE_EMOTIONS_NAMES)
composite.valence         # → 2  (joy's Pleasantness flow)
composite.arousal         # → 2  (max |flow| across components)
composite.type            # → "excited positive"
```

The result is always a `CompositeEmotion`.  To get a named `Feeling` use
`Feeling("love")` explicitly — see [feelings.md](feelings.md).

## `*` operator

`emotion1 * emotion2` (different axes) is an alias for cross-axis `+` and
returns `CompositeEmotion`.

## Comparison

All comparison operators compare `emotional_flow` values:

```python
rage > anger      # True  (3 > 2)
anger > annoyance # True  (2 > 1)
fear < apprehension  # True  (-2 < -1)
```

Comparing emotions from different axes is numerically valid but
psychologically meaningless — use with care.

## Hyper-intensity

Flows beyond ±3 generate hyper-emotions with an `intensity_offset`:

| `intensity_offset` | Prefix |
|--------------------|--------|
| 1 | `mega` |
| 2 | `extreme` |
| ≥3 | `hyper` |

```python
rage + 1    # "mega rage"   (offset=1)
rage + 2    # "extreme rage" (offset=2)
rage + 3    # "hyper rage"  (offset=3)
```
