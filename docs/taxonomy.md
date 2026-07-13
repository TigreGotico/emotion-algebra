> **This page documents a [legacy view](legacy.md).** It describes the
> Plutchik/Hourglass layer, which is kept as a *vocabulary* and graded
> `METAPHOR`. For the model the library actually computes with, see
> [the core](theory.md) and [its laws](core-laws.md).

# Emotion Taxonomy

The Hourglass of Emotions (Cambria et al. 2012) defines four axes, each with
three intensity levels per polarity (±1, ±2, ±3).

## Axes and emotions

| Axis | flow +3 | +2 | +1 | −1 | −2 | −3 |
|------|---------|----|----|----|----|----|
| **Pleasantness** | ecstasy | joy | serenity | pensiveness | sadness | grief |
| **Attention** | vigilance | anticipation | interest | distraction | surprise | amazement |
| **Sensitivity** | rage | anger | annoyance | apprehension | fear | terror |
| **Aptitude** | admiration | trust | acceptance | boredom | disgust | loathing |

Positive pole = approach / active / desirable.
Negative pole = avoidance / suppressed / aversive.

**Sensitivity axis direction**: anger (+) is an approach response (high reactivity,
outward activation); fear (−) is an avoidance response (defensive, inward).
Per Öhman (1986), this is not a hedonic distinction — both poles are arousal states.

## Intensity levels

| `|flow|` | Level | Example |
|-----------|-------|---------|
| 1 | basic (primary) | annoyance, apprehension |
| 2 | mild (secondary) | anger, fear |
| 3 | intense (tertiary) | rage, terror |
| 0 | neutral | Neutrality |

## All 24 named emotions

```
Pleasantness: ecstasy, joy, serenity, pensiveness, sadness, grief
Attention:    vigilance, anticipation, interest, distraction, surprise, amazement
Sensitivity:  rage, anger, annoyance, apprehension, fear, terror
Aptitude:     admiration, trust, acceptance, boredom, disgust, loathing
```

## Accessing emotions

```python
from emotion_algebra.emotions import EMOTIONS, POSITIVE_EMOTIONS, NEGATIVE_EMOTIONS
from copy import copy

anger = copy(EMOTIONS["anger"])

# POSITIVE_EMOTIONS: only Pleasantness > 0 (serenity, joy, ecstasy)
# NEGATIVE_EMOTIONS: only Pleasantness < 0 (pensiveness, sadness, grief)
print([e.name for e in POSITIVE_EMOTIONS])
```

## Two lineages, one name

Ten names are claimed by **both** dyad lineages, and they mean different things:

* a **Feeling** is a Plutchik dyad — two *primaries* combined;
* a **CompositeEmotion** is an Hourglass compound — the same pairing at maximum
  intensity.

So `love` is `joy + trust` as a Feeling but `ecstasy + admiration` as a
CompositeEmotion: the same dyad, a very different intensity.

| Name | Feeling (Plutchik dyad) | CompositeEmotion (Hourglass compound) |
|---|---|---|
| aggressiveness | anger + anticipation | rage + vigilance |
| anxiety | anticipation + fear | terror + vigilance |
| awe | fear + surprise | terror + amazement |
| contempt | disgust + anger | rage + loathing |
| disapproval | surprise + sadness | grief + amazement |
| envy | sadness + anger | grief + admiration |
| love | joy + trust | ecstasy + admiration |
| optimism | anticipation + joy | ecstasy + vigilance |
| remorse | sadness + disgust | grief + loathing |
| submission | trust + fear | terror + admiration |

### Resolving a name

`emotion_algebra.taxonomy` is the single sanctioned lookup. It **prefers the
Feeling** by default — the lower-intensity reading is the one a plain English
name usually means — and the preference is explicit rather than a side-effect of
import order:

```python
from emotion_algebra import resolve, is_ambiguous, describe_collision

resolve("love")                      # Feeling(joy + trust)          [default]
resolve("love", prefer="composite")  # CompositeEmotion(ecstasy + admiration)

is_ambiguous("love")                 # True   (case-insensitive)
is_ambiguous("joy")                  # False

describe_collision("love")
# {'name': 'love', 'feeling': ['joy', 'trust'], 'composite': ['ecstasy', 'admiration']}
```

An unknown name returns `None`; an unknown `prefer` raises `ValueError`. Every
collision's resolution is locked by a test, so the answer cannot drift.
