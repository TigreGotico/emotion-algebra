# Valence, Arousal, and Type

## Valence

`emotion.valence` returns the **Pleasantness axis component only**.

This follows from Cambria et al. (2012) — Pleasantness is the hedonic axis.
Arousal (reactivity) and hedonics are orthogonal in all validated models
(Posner et al. 2005, Russell 1980).

| Emotion | Axis | `valence` | Reason |
|---------|------|-----------|--------|
| `ecstasy` | Pleasantness +3 | **+3** | maximum pleasant |
| `joy` | Pleasantness +2 | **+2** | secondary pleasant |
| `serenity` | Pleasantness +1 | **+1** | basic pleasant |
| `pensiveness` | Pleasantness −1 | **−1** | basic unpleasant |
| `sadness` | Pleasantness −2 | **−2** | secondary unpleasant |
| `grief` | Pleasantness −3 | **−3** | maximum unpleasant |
| `anger` | Sensitivity +2 | **0** | arousal, not hedonic |
| `fear` | Sensitivity −2 | **0** | arousal, not hedonic |
| `anticipation` | Attention +2 | **0** | engagement, not hedonic |
| `trust` | Aptitude +2 | **0** | competence, not hedonic |

`POSITIVE_EMOTIONS` contains only Pleasantness>0 emotions: `[serenity, joy, ecstasy]`.

## Arousal

`emotion.arousal` = `abs(emotional_flow)` — activation intensity, axis-independent.

Maps to the arousal dimension of Russell's (1980) Circumplex:

| Emotion | `arousal` |
|---------|-----------|
| `serenity` / `pensiveness` | 1 |
| `joy` / `sadness` / `anger` / `fear` | 2 |
| `ecstasy` / `grief` / `rage` / `terror` | 3 |
| `Neutrality` | 0 |

## Type — Russell Circumplex

`emotion.type` classifies the emotion using `valence` and `arousal`:

| Category | Condition | Examples |
|----------|-----------|---------|
| `"excited positive"` | valence > 0, arousal > 1 | joy, ecstasy |
| `"excited negative"` | valence < 0, arousal > 1 | sadness, grief |
| `"calm positive"` | valence > 0, arousal ≤ 1 | serenity |
| `"calm negative"` | valence < 0, arousal ≤ 1 | pensiveness |
| `"activated neutral"` | valence == 0, arousal > 0 | anger, fear, anticipation, trust |
| `"neutral"` | arousal == 0 | Neutrality |

The same classification applies to `CompositeEmotion.type` and `Feeling.type`.

## EmotionalDimension.valence

`dimension.valence` is the hedonic sign of the positive pole of that axis:

| Axis | `valence` | Justification |
|------|-----------|---------------|
| `pleasantness` | **+1** | explicitly hedonic |
| `aptitude` | **+1** | competence is socially desirable |
| `sensitivity` | **0** | approach vs avoidance — orthogonal to hedonics |
| `attention` | **0** | engagement — orthogonal to hedonics |

## Polarity — the four-axis score

`valence` is **not** the library's sentiment score. It is the Pleasantness axis
alone. The sentiment score is `polarity`, and it is Cambria's published formula
over all four axes:

```
polarity = (Pleasantness + |Attention| − |Sensitivity| + Aptitude) / 3
```

evaluated on axes normalized to `[−1, 1]` and clamped to `[−1, 1]`.

```python
get_emotion("trust").valence     # 0    — trust is not on the Pleasantness axis
get_emotion("trust").polarity    # +0.33 — but it is clearly positive sentiment
```

Keeping the two separate is the point: `valence` answers "how pleasant", while
`polarity` answers "how positive overall", and for anything off the Pleasantness
axis those differ. `_circumplex_type` classifies on **polarity**.

### The |Attention| quirk is intentional

Attention enters the formula as a *magnitude*, so **both of its poles score
positive** — anticipation and surprise alike. Negating a purely attention-axis
emotion therefore leaves its polarity unchanged.

That is a property of the published formula, and it is locked by
`test_both_attention_poles_score_positive`. The reading is defensible: engagement,
in either direction, is not hedonically negative the way threat (|Sensitivity|,
which *subtracts*) is.
