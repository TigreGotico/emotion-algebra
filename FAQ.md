# FAQ — emotion_data

## What is emotion_data?

A Python library implementing **emotion algebra** based on Plutchik's Wheel of Emotions and the Hourglass of Emotions. Emotions are first-class mathematical objects with operator overloading.

## What are the minimum dependencies?

Core package requires only `numpy`. All other dependencies are optional extras.

## How do I install just the core?

```bash
pip install emotion_data
```

## What optional extras are available?

| Extra | Deps | Enables |
|---|---|---|
| `[lexicon]` | pandas | `lexicons.py` — word→emotion CSV lookup |
| `[deepmoji]` | torch, torchMoji | `deepmoji.py` — emoji-based emotion tagging |
| `[tagging]` | paralleldots | `tag.py` — ParallelDots API tagging |

## How does intensity arithmetic work?

Emotions sit on a signed integer axis per dimension. Adding an integer moves up the axis:

```
annoyance (1) → anger (2) → rage (3) → mega rage (offset 1) → hyper rage (offset 3+)
```

## How do I get the opposite emotion?

Use negation: `-anger == fear`, `-joy == sadness`.

## How does joy + trust produce love?

When two emotions from **different** Hourglass dimensions are added, a `Feeling` dyad is created. The pair is looked up in `FEELING_NAMES` and the named feeling is returned.

## What Python versions are supported?

Python 3.10 and above.

## Where are the tests?

Five test files in `test/`:
- `test_plutchik.py` — `Emotion`, `Neutrality`, `EmotionalDimension` (all operators, classification, vectors)
- `test_feelings.py` — `Feeling`, `FEELINGS`, factory functions, arithmetic
- `test_composite.py` — `CompositeEmotion`, `CompositeDimension` (construction, dimension properties, operators)
- `test_behaviour.py` — `Behaviour`, `BehavioralReaction`, `BEHAVIOURS`, `REACTIONS`
- `test_emotions_module.py` — `emotions.py`, `lexicons.py`, `EmotionAnalyzer`

Run: `python -m pytest test/ --cov=emotion_data --cov-config=.coveragerc`

## What is the coverage target?

90%+ on core modules. `deepmoji.py` and `tag.py` are excluded (optional external integrations).

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
