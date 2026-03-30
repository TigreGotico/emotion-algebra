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
