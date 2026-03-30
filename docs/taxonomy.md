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
