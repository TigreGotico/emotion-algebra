# API Reference

## `emotion_algebra.plutchik`

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

## `emotion_algebra.composite_emotions`

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

## `emotion_algebra.feelings`

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

## `emotion_algebra.emotions`

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

## `emotion_algebra.behaviour`

### `Behaviour` / `BehavioralReaction`

Maps emotions to adaptive behavioural responses.
`BEHAVIOURS`, `REACTIONS`, `REACTION_TO_EMOTION_MAP` are all
`MappingProxyType` (immutable).

**Note**: The behaviour→emotion mapping skips cognitive appraisal (Lazarus 1991)
and is not derived from either Plutchik or Cambria — it is a convenience layer only.

---

## `emotion_algebra.state`

### `EmotionalState`

Mutable 4-axis float accumulator. Represents a running affective state that can be
updated, decayed, and queried. — `state.py:EmotionalState`

| Method / Property | Returns | Description |
|-------------------|---------|-------------|
| `apply(emotion, weight=1.0)` | `self` | Add `weight * emotion.as_array` to internal vector |
| `decay(factor=0.9)` | `self` | Multiply all axes by `factor` (simulates fading) |
| `reset()` | `self` | Zero all axes |
| `dominant()` | `Emotion \| None` | `closest_emotion(vector)`, or `None` if zero |
| `snapshot()` | `np.ndarray` | Copy of current 4-axis float vector |
| `valence()` | `float` | Pleasantness axis value |
| `arousal()` | `float` | `max(abs(v))` across all axes |
| `to_dict() / from_dict()` | `dict / EmotionalState` | Serialization |
| `__add__(other)` | `EmotionalState` | Vector addition |
| `__mul__(scalar)` | `EmotionalState` | Scalar multiplication |

```python
state = EmotionalState()
state.apply(get_emotion("joy"), weight=0.8).apply(get_emotion("trust"), weight=0.5)
state.decay(0.9)
state.dominant()   # → Emotion or None
state.valence()    # → float
```

### `EmotionTimeline`

Ordered sequence of `EmotionalState` snapshots. — `state.py:EmotionTimeline`

| Method | Returns | Description |
|--------|---------|-------------|
| `append(state)` | `self` | Record a snapshot of `state` |
| `drift()` | `np.ndarray` | `snapshots[-1] - snapshots[0]` |
| `dominant_sequence()` | `list[Emotion \| None]` | `closest_emotion` for each snapshot |
| `to_dict() / from_dict()` | `dict / EmotionTimeline` | Serialization |

---

## `emotion_algebra.distance`

### `emotion_distance(a, b) → float`

Euclidean distance between two `EmotionBase` objects in the 4-axis Hourglass space.
`a.as_array` and `b.as_array` are cast to float before computation. — `distance.py:emotion_distance`

### `closest_emotion(vector) → Emotion`

Return the named `Emotion` in `EMOTIONS` nearest to `vector` (list or ndarray of 4 floats)
by Euclidean distance. — `distance.py:closest_emotion`

### `emotion_clusters(threshold=1.5) → list[list[Emotion]]`

Greedy seed-based clustering of all 24 named emotions by Hourglass distance. — `distance.py:emotion_clusters`

---

## `emotion_algebra.appraisal`

### `Appraisal`

Dataclass encoding Scherer's Component Process Model (CPM) appraisal dimensions. — `appraisal.py:Appraisal`

| Field | Type | Values |
|-------|------|--------|
| `novelty` | `str \| None` | `"expected"` / `"unexpected"` |
| `goal_relevance` | `str \| None` | `"relevant"` / `"irrelevant"` |
| `goal_congruence` | `str \| None` | `"congruent"` / `"incongruent"` |
| `agency` | `str \| None` | `"self"` / `"other"` / `"circumstance"` |
| `coping_potential` | `str \| None` | `"high"` / `"low"` |

### `appraisal_to_emotion(appraisal) → Emotion`

Maps an `Appraisal` to a primary emotion via a 16-rule table derived from Scherer (2001).
Returns `Neutrality()` if no rule matches. — `appraisal.py:appraisal_to_emotion`

```python
a = Appraisal(goal_relevance="relevant", goal_congruence="incongruent",
              agency="other", coping_potential="low")
appraisal_to_emotion(a)   # → fear
```

---

## `emotion_algebra.float_emotion`

### `FloatEmotion`

Continuous-valued point in the 4-axis Hourglass space. Subclasses `EmotionBase`;
all arithmetic is float-precision. — `float_emotion.py:FloatEmotion`

| Property | Returns | Description |
|----------|---------|-------------|
| `name` | `str` | Explicit name or `"~" + closest_emotion(vector).name` |
| `as_array` | `np.ndarray[float]` | 4-element float vector |
| `as_matrix` | `np.ndarray[float]` | 2×2 matrix |
| `valence` | `float` | Pleasantness axis value |
| `arousal` | `float` | `max(abs(v))` across all axes |
| `emotional_flow` | `float` | Sum of all axis values |
| `type` | `str` | Russell Circumplex category |

**Constructors**:

| Method | Description |
|--------|-------------|
| `FloatEmotion(sensitivity, attention, pleasantness, aptitude)` | Direct construction |
| `FloatEmotion.from_embedding(vec, projection_matrix=None)` | Project ndarray into Hourglass space |
| `FloatEmotion.from_emotion(emotion)` | Cast a discrete `Emotion` to float |

Supports full arithmetic with `FloatEmotion`, `EmotionBase`, `int`, and `float` operands.

---

## `emotion_algebra.emoji`

### `EMOJI_EMOTION_MAP`

`MappingProxyType[str, str]` — ~90 Unicode emoji → Plutchik emotion name.
Covers all 8 primary emotion axes at multiple intensity levels. — `emoji.py`

### `from_emoji(emoji_char) → Emotion | None`

Single emoji lookup. Strips whitespace; checks user registry before canonical map. — `emoji.py:from_emoji`

### `score_emojis(text) → EmotionalState`

Scan `text` character-by-character; each mapped emoji contributes weight=1.0 to the returned state. — `emoji.py:score_emojis`

### `from_emojis(text) → Emotion | None`

`score_emojis(text).dominant()`. Returns `None` if no mapped emoji found. — `emoji.py:from_emojis`

### `register_emoji(emoji_char, emotion_name) → None`

Add a runtime mapping that overrides `EMOJI_EMOTION_MAP`. Validates `emotion_name`; raises `ValueError` if unknown. — `emoji.py:register_emoji`

### `unregister_emoji(emoji_char) → bool`

Remove a runtime registration. Returns `True` if removed, `False` if absent. Canonical map is unaffected. — `emoji.py:unregister_emoji`

### `DeepMojiAdapter`

Converts `{emoji: probability}` distributions (as produced by DeepMoji / torchMoji) to named emotions. — `emoji.py:DeepMojiAdapter`

| Method | Returns | Description |
|--------|---------|-------------|
| `from_scores(scores)` | `Emotion \| None` | Blend distribution → dominant emotion |
| `from_ranked(ranked, top_k=None)` | `Emotion \| None` | `[(emoji, score)]` input variant |
| `score_state(scores)` | `EmotionalState` | Full blended state (no reduction to dominant) |

---

## `emotion_algebra.text`

### `from_text(text) → Emotion | None`

Tokenise `text`, look up each word in the bundled NRC/EmoLex-style CSV, return the most frequent matched emotion. — `text.py:from_text`

### `score_text(text) → EmotionalState`

All matching tokens apply to an `EmotionalState` with equal weight. Richer than `from_text`. — `text.py:score_text`

### `score_mixed(text) → EmotionalState`

Single-pass scoring combining word lexicon and emoji map. Recommended entry point for general text. — `text.py:score_mixed`

### `from_mixed(text) → Emotion | None`

`score_mixed(text).dominant()`. — `text.py:from_mixed`

### `DeepMojiONNXAdapter`

Canonical neural text-to-emotion engine wrapping `deepmoji-onnx`. Core dependency — always available. — `deepmoji.py:DeepMojiONNXAdapter`

| Method | Returns | Description |
|--------|---------|-------------|
| `analyze(text)` | `Emotion \| None` | DeepMoji ONNX → top-k emoji → dominant emotion |
| `score(text)` | `EmotionalState` | DeepMoji ONNX → full 4-axis accumulation |
| `top_emoji_scores(text)` | `dict[str, float]` | Raw `{emoji: probability}` from model |

Constructor: `DeepMojiONNXAdapter(variant='fp32', cache_dir=None, top_k=10)`

---

## `emotion_algebra` (top-level)

### `EmotionAnalyzer`

Stateless facade exposing all sub-module entry points as `@staticmethod` methods. — `__init__.py:EmotionAnalyzer`

All functions described above are also importable directly from `emotion_algebra`:

```python
from emotion_algebra import (
    get_emotion, get_feeling, get_dimension,
    EmotionalState, EmotionTimeline,
    emotion_distance, closest_emotion, emotion_clusters,
    from_text, score_text, score_mixed, from_mixed,
    from_emoji, score_emojis, from_emojis,
    register_emoji, unregister_emoji, DeepMojiAdapter,
    Appraisal, appraisal_to_emotion,
    FloatEmotion, EMOJI_EMOTION_MAP,
)
```
