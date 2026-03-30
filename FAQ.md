# FAQ — emotion_algebra

## What is emotion_algebra?

A Python library implementing **emotion algebra** based on Plutchik's Wheel of Emotions and the Hourglass of Emotions. Emotions are first-class mathematical objects with operator overloading.

## What are the minimum dependencies?

Core package requires only `numpy`. All other dependencies are optional extras.

## How do I install just the core?

```bash
pip install emotion_algebra
```

## What optional extras are available?

| Extra | Deps | Enables |
|---|---|---|
| `[lexicon]` | pandas | `lexicons.py` — word→emotion CSV lookup |
| `[transformers]` | transformers>=4.0, torch | `text.py` — HuggingFace pipeline tagging |

## How do I score text that contains both words and emoji?

Use `score_mixed` / `from_mixed` — they run both the word lexicon and emoji map in a single pass:

```python
from emotion_algebra.text import score_mixed, from_mixed

state = score_mixed("I'm so happy 😄🎉")   # EmotionalState — both signals blended
emotion = from_mixed("grief and sorrow 😭") # dominant Emotion
```

`EmotionAnalyzer.score_mixed(text)` and `EmotionAnalyzer.analyze_mixed(text)` expose the same API.

## How do I add custom emoji mappings?

```python
from emotion_algebra.emoji import register_emoji, unregister_emoji

register_emoji("🤖", "trust")    # 🤖 now maps to trust
unregister_emoji("🤖")           # remove — canonical map restored
```

`register_emoji` validates the emotion name and raises `ValueError` for unknown names.
Custom registrations layer on top of `EMOJI_EMOTION_MAP` and affect all functions
(`from_emoji`, `score_emojis`, `DeepMojiAdapter`).

## How do I map emojis to emotions?

Three entry points in `emotion_algebra.emoji`:

```python
from emotion_algebra.emoji import from_emoji, score_emojis, from_emojis, DeepMojiAdapter

from_emoji("😊")               # → Emotion("serenity")
from_emojis("Best day! 😄🎉") # → Emotion("joy")  (dominant)
score_emojis("😭😭😊")        # → EmotionalState  (full 4-axis accumulation)

# DeepMoji / torchMoji output:
adapter = DeepMojiAdapter()
adapter.from_scores({"😂": 0.45, "😊": 0.30, "😭": 0.25})  # → Emotion
adapter.from_ranked([("😂", 0.45), ("😊", 0.30)], top_k=1)  # → Emotion
adapter.score_state({"😂": 0.6, "😊": 0.4})                 # → EmotionalState
```

The canonical map is `EMOJI_EMOTION_MAP` — ~90 emojis covering all 8 Plutchik primaries
and their intensity variants (serenity/joy/ecstasy, apprehension/fear/terror, etc.).

## How does intensity arithmetic work?

Emotions sit on a signed integer axis per dimension. Adding an integer moves up the axis:

```
annoyance (1) → anger (2) → rage (3) → mega rage (offset 1) → hyper rage (offset 3+)
```

## How do I get the opposite emotion?

Use negation: `-anger == fear`, `-joy == sadness`.

## How does joy + trust produce love?

Per the spec, cross-axis `+` always returns a `CompositeEmotion`, never a `Feeling` directly.
To get a named `Feeling`, use `Feeling("love")` or `get_feeling()` explicitly.

## What is the difference between Feeling and CompositeEmotion?

`CompositeEmotion` is the algebraic representation (a point in H spanning two axes).
`Feeling` is a cultural/psychological label for a named multi-axis experience (Plutchik's dyads).
The `+` operator always returns `CompositeEmotion`; `Feeling` is a separate named construct.

## What is valence and how is it computed?

`valence` = the **Pleasantness axis component** only. Anger (Sensitivity axis) has `valence=0`.
This is correct: arousal (reactivity) and hedonics are orthogonal in all validated models
(Posner et al. 2005). Only Pleasantness-axis emotions carry hedonic information.

## What is arousal?

`arousal = abs(emotional_flow)` — activation intensity, axis-independent. Range 0–3+.
Maps to Russell's (1980) arousal dimension.

## What Python versions are supported?

Python 3.10 and above.

## Where are the tests?

Five test files in `test/`:
- `test_plutchik.py` — `Emotion`, `Neutrality`, `EmotionalDimension` (all operators, classification, vectors)
- `test_feelings.py` — `Feeling`, `FEELINGS`, factory functions, arithmetic
- `test_composite.py` — `CompositeEmotion`, `CompositeDimension` (construction, dimension properties, operators)
- `test_behaviour.py` — `Behaviour`, `BehavioralReaction`, `BEHAVIOURS`, `REACTIONS`
- `test_emotions_module.py` — `emotions.py`, `lexicons.py`, `EmotionAnalyzer`

Run: `python -m pytest test/ --cov=emotion_algebra --cov-config=.coveragerc`

## What is the coverage target?

93%+ on core modules. `deepmoji.py` and `tag.py` are excluded (removed; optional HF adapter paths
require the `[transformers]` extra and are skipped in the default test run).

## What does `int(emotion)` return?

Net activation scalar — the sum of `emotional_flow` across all component axes. This is **not** a
hedonic score. `int(love)` returns 4 because joy (flow=2) and trust (flow=2) each contribute 2.
Use `valence` for positive/negative polarity.

## What changed in the Phase 3 architectural fixes?

| Fix | Details |
|---|---|
| `get_emotion` import shadow | `lexicons.get_emotion` renamed `get_word_emotion`; `EmotionAnalyzer.emotion()` now returns `Emotion` not a string |
| `CompositeEmotion.emotional_flow` | Was `np.linalg.norm` (always ≥ 0); now `sum(component.emotional_flow)` — signed |
| `copy()` aliased state | All operator `copy(self)` calls replaced with `deepcopy(self)` |
| `Emotion.__bool__` | Now reflects presence (non-zero intensity), not valence — fear is truthy |
| `Emotion.valence` | Returns `int` (+1/−1/0), not `bool`; `POSITIVE_EMOTIONS` filter fixed |
| `np.matrix` deprecated | `as_matrix` now returns `np.ndarray`; `__mul__` uses `np.matmul` |
| Dead `__sub__` branch | `self.name - other` (TypeError) replaced with `NotImplemented` |
| `string_to_emotion` duplication | `Feeling` now calls `Emotion.string_to_emotion` directly |
| `acknowledgement` spelling | `OPPOSITE_FEELINGS_NAMES` key normalised to match `FEELING_NAMES` |
| Global dict mutation | `EMOTIONS`, `FEELINGS`, `BEHAVIOURS`, `REACTIONS` wrapped in `MappingProxyType` |

## Known bugs fixed in production pass (Phase 2)

- `Emotion.__bool__` was returning `numpy.bool_` (not Python `bool`), causing `TypeError` in Python 3.11+.
- `_get_behaviours()` used `if e:` to guard emotion lookup, silently skipping negative-flow emotions (fear, terror). Fixed to `if e is not None:`.
- `emotion_to_dimension()` had the same `if emotion:` guard bug.
- `BehavioralReaction.from_data({})` crashed with `KeyError` on missing `'behaviour'` key.

## What is the emotional_flow property?

Signed integer intensity: ±1 primary, ±2 secondary, ±3 tertiary, 0 neutral. Used as the scalar operand in all arithmetic.

## What is Neutrality?

The identity element: `emotion + Neutrality() == emotion`. Also returned when arithmetic reduces flow to zero.
