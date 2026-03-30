# API Reference

## `emotion_data.plutchik`

### `Emotion`

Core algebraic type. Represents a single named emotion on one Hourglass axis.

| Attribute / Property | Type | Description |
|----------------------|------|-------------|
| `name` | `str` | Canonical name, prefixed with intensity label when hyper |
| `emotional_flow` | `int` | Signed axis position: ±1/±2/±3, 0 for Neutrality |
| `valence` | `int` | Pleasantness component only; 0 for non-Pleasantness axes |
| `arousal` | `int` | `abs(emotional_flow)` — activation intensity |
| `type` | `str` | Russell Circumplex category |
| `intensity` | `str` | `"basic"` / `"mild"` / `"intense"` / hyper prefix |
| `dimension` | `EmotionalDimension` | The axis this emotion belongs to |
| `intensity_offset` | `int` | Extra intensity beyond ±3 |
| `is_primary` | `bool` | flow ±1 |
| `is_secondary` | `bool` | flow ±2 |
| `is_tertiary` | `bool` | flow ±3 |
| `is_hyper` | `bool` | `intensity_offset > 0` |
| `is_composite` | `bool` | Always `False` on plain `Emotion` |
| `emotion_vector` | `list[Emotion]` | `[sensitivity, attention, pleasantness, aptitude]` |
| `as_array` | `np.ndarray` | shape `(4,)`, integer flows |
| `as_matrix` | `np.ndarray` | shape `(2, 2)` |
| `kind` | `str` | Hand-curated psychological category (not model-derived) |
| `base_emotion` | `Emotion` | Primary (±1) form of this emotion |
| `opposite_emotion` | `Emotion` | Symmetric opposite pole |
| `parent_emotion` | `Emotion \| None` | One step toward primary |

**Operators**:

| Operator | Other type | Returns |
|----------|-----------|---------|
| `+ int` | `int` | `Emotion` (flow shift) |
| `- int` | `int` | `Emotion` |
| `* / //` | `int` | `Emotion` |
| `+ Emotion` (same axis) | `Emotion` | `Emotion` (sum flows) |
| `+ Emotion` (diff axis) | `Emotion` | `CompositeEmotion` |
| `* Emotion` (diff axis) | `Emotion` | `CompositeEmotion` (alias for `+`) |
| `-` (unary) | — | opposite pole `Emotion` |
| `<< int` | `int` | downgrade (flow − n) |
| `>> int` | `int` | upgrade (flow + n) |
| `< <= > >= == !=` | `Emotion / int` | `bool` (compares `emotional_flow`) |
| `bool()` | — | `True` when `emotional_flow != 0` |

### `Neutrality`

Identity element: `e + Neutrality() == e` for all `e`.
`emotional_flow = 0`, `valence = 0`, `arousal = 0`, `bool() == False`.

### `EmotionalDimension`

One of four Hourglass axes.

| Property | Type | Description |
|----------|------|-------------|
| `axis` | `str` | `"pleasantness"` / `"attention"` / `"sensitivity"` / `"aptitude"` |
| `valence` | `int` | Hedonic sign of positive pole: Pleasantness/Aptitude=+1, others=0 |
| `kind` | `str` | `"hedonic"` or `"activation"` |
| `basic_emotion` / `basic_opposite` | `Emotion` | flow ±1 |
| `mild_emotion` / `mild_opposite` | `Emotion` | flow ±2 |
| `intense_emotion` / `intense_opposite` | `Emotion` | flow ±3 |

### `_circumplex_type(valence, arousal) → str`

Module-level helper. Classifies a `(valence, arousal)` pair into a Russell
Circumplex category. Used by `Emotion.type`, `CompositeEmotion.type`,
`Feeling.type`.

---

## `emotion_data.composite_emotions`

### `CompositeEmotion`

Multi-axis emotion spanning two Hourglass dimensions.

| Property | Returns | Description |
|----------|---------|-------------|
| `components` | `list[Emotion]` | Single-axis emotions, one per active axis |
| `emotional_flow` | `int` | Signed sum of component flows |
| `valence` | `int` | Sum of Pleasantness-axis component flows |
| `arousal` | `int` | `max(abs(e.emotional_flow) for e in components)` |
| `type` | `str` | Russell Circumplex |
| `name` | `str` | Named composite from `COMPOSITE_EMOTIONS_NAMES`, or descriptive fallback |
| `emotion_vector` | `list[Emotion]` | 4-element vector |
| `dimensions` | `list[EmotionalDimension]` | One per component |

### `COMPOSITE_EMOTIONS_NAMES`

`dict[str, list[str]]` — 16 named two-axis composites from Plutchik (1980).
Examples: `"love": ["ecstasy", "admiration"]`, `"aggressiveness": ["rage", "vigilance"]`.

---

## `emotion_data.feelings`

### `Feeling`

Named cultural label for a multi-axis emotional state.

| Property | Returns | Description |
|----------|---------|-------------|
| `emotions` | `list[Emotion]` | Component emotions |
| `name` | `str` | Named dyad from `FEELING_NAMES`, or descriptive fallback |
| `emotional_flow` | `int` | Signed sum of component flows |
| `valence` | `int` | Sum of `e.valence` for all components (Pleasantness only) |
| `arousal` | `int` | `max(e.arousal)` across components |
| `type` | `str` | Russell Circumplex |
| `emotion_vector` | `list[Emotion]` | 4-element vector |
| `dimensions` | `list[EmotionalDimension]` | One per component |

### `FEELING_NAMES`

`MappingProxyType[str, list[str]]` — 40 named feelings mapped to two-emotion
component lists.

### `get_feeling_from_emotions(emotion1, emotion2) → str | None`

Return the feeling name for a pair of emotion names, or `None` if no match.

---

## `emotion_data.emotions`

### `EMOTIONS`

`MappingProxyType[str, Emotion]` — all 24 named emotions, keyed by name.

### `POSITIVE_EMOTIONS` / `NEGATIVE_EMOTIONS`

Lists of `Emotion` where `valence > 0` / `valence < 0`.
Under the spec, these contain only Pleasantness-axis emotions:
- `POSITIVE_EMOTIONS`: `[serenity, joy, ecstasy]`
- `NEGATIVE_EMOTIONS`: `[pensiveness, sadness, grief]`

### `get_emotion(name) → Emotion | None`

Look up an emotion by name.

### `emotion_to_dimension(name) → EmotionalDimension | list | None`

Return the axis for a named emotion.

---

## `emotion_data.behaviour`

### `Behaviour` / `BehavioralReaction`

Maps emotions to adaptive behavioural responses.
`BEHAVIOURS`, `REACTIONS`, `REACTION_TO_EMOTION_MAP` are all
`MappingProxyType` (immutable).

**Note**: The behaviour→emotion mapping skips cognitive appraisal (Lazarus 1991)
and is not derived from either Plutchik or Cambria — it is a convenience layer only.
