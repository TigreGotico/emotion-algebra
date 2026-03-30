# emotion-algebra

Signed integer arithmetic over a 4-axis affective space, based on Plutchik's Wheel of Emotions (1980) and Cambria's Hourglass of Emotions (2012).

```python
from emotion_algebra.emotions import get_emotion

anger = get_emotion("anger")   # sensitivity axis, flow +2
fear  = get_emotion("fear")    # sensitivity axis, flow -2

anger + 1          # → rage   (flow +3)
anger - 1          # → annoyance (flow +1)
-anger             # → fear   (opposite)
anger * fear       # → CompositeEmotion  (cross-axis product)
```

---

## Installation

```bash
pip install emotion-algebra
```

Optional extras:

```bash
pip install "emotion-algebra[lexicon]"   # pandas — NRC/Emolex CSV loading
pip install "emotion-algebra[deepmoji]"  # torch + torchMoji — text → emotion
pip install "emotion-algebra[tagging]"   # paralleldots — cloud NLP tagging
```

---

## The model

### Axes (Cambria 2012)

| Axis | Positive pole (flow > 0) | Negative pole (flow < 0) | Hedonic? |
|---|---|---|---|
| Sensitivity | rage → anger → annoyance | apprehension → fear → terror | No (activation) |
| Attention | vigilance → anticipation → interest | distraction → surprise → amazement | No (activation) |
| Pleasantness | ecstasy → joy → serenity | pensiveness → sadness → grief | Yes |
| Aptitude | admiration → trust → acceptance | boredom → disgust → loathing | Yes |

Each axis has three intensity levels: ±1 (basic), ±2 (secondary), ±3 (tertiary).

### Valence and arousal (Russell 1980)

- **Valence** = Pleasantness axis component only. `anger.valence == 0`; `joy.valence == 2`.
- **Arousal** = `abs(emotional_flow)`. `rage.arousal == 3`.
- **Type** = Russell Circumplex quadrant: `"excited positive"`, `"calm positive"`, `"excited negative"`, `"calm negative"`, `"activated neutral"`, `"neutral"`.

### Feelings (Plutchik dyads)

A `Feeling` is a named combination of two primary emotions from different axes:

```python
from emotion_algebra.emotions import get_emotion
joy   = get_emotion("joy")
trust = get_emotion("trust")
joy + trust   # → Feeling("love")
```

### Composite emotions

Cross-axis arithmetic produces a `CompositeEmotion` — an unnamed blend preserving full 4D vector information:

```python
anger * fear   # CompositeEmotion  (sensitivity × sensitivity, different poles)
rage + trust   # CompositeEmotion  (sensitivity + aptitude)
```

---

## Quick reference

```python
from emotion_algebra.emotions import get_emotion, get_dimension, EMOTIONS
from emotion_algebra.feelings import get_feeling, FEELINGS
from emotion_algebra.plutchik import Neutrality

# All 24 named emotions
list(EMOTIONS.keys())

# All 38 named feelings
list(FEELINGS.keys())

# Emotion properties
e = get_emotion("joy")
e.emotional_flow   # 2
e.valence          # 2   (pleasantness axis)
e.arousal          # 2
e.type             # "excited positive"
e.dimension.name   # "pleasantness"
e.opposite_emotion # sadness

# Arithmetic
e + 1    # ecstasy  (flow +3)
e - 1    # serenity (flow +1)
e - 2    # Neutrality
-e       # sadness
abs(e)   # Neutrality()
```

See [`examples/`](examples/) for runnable scripts covering every operator and class.

---

## Scientific references

- Plutchik, R. (1980). *A general psychoevolutionary theory of emotion.* In R. Plutchik & H. Kellerman (Eds.), *Emotion: Theory, research, and experience* (Vol. 1, pp. 3–33).
- Cambria, E., Livingstone, A., & Hussain, A. (2012). *The Hourglass of Emotions.* In A. Esposito et al. (Eds.), *Cognitive Behavioural Systems*, LNCS 7403.
- Russell, J. A. (1980). *A circumplex model of affect.* Journal of Personality and Social Psychology, 39(6), 1161–1178.
- Posner, J., Russell, J. A., & Peterson, B. S. (2005). *The circumplex model of affect: An integrative approach.* Development and Psychopathology, 17(3), 715–734.

---

## License

Apache 2.0
