> **Plutchik / Hourglass model.** Stateful accumulation over the Hourglass
> axes. For the affect core's dynamics, set points, drives, temperament,
> see [building an agent](agents.md).

# State & Timeline

## EmotionalState

`EmotionalState` (`state.py:EmotionalState`) is a mutable 4-axis float accumulator representing an agent's running affective state. It is the primary output of multi-signal analysis pipelines. Each call to `apply()` adds a weighted emotion vector to the running total, and `decay()` fades that total toward zero over time.

```python
from emotion_algebra.state import EmotionalState
from emotion_algebra.emotions import get_emotion

state = EmotionalState()
state.apply(get_emotion("joy"), weight=0.8)
state.apply(get_emotion("fear"), weight=0.3)
state.decay(factor=0.9)        # fade all axes by 10%
state.valence()                # float, Pleasantness axis
state.arousal()                # float, max(|axis|)
state.dominant()               # → Emotion or None
state.snapshot()               # np.ndarray shape (4,), copy
```
### apply / decay / reset

- `apply(emotion, weight=1.0)`: adds `weight * emotion.as_array` to the internal vector. Returns `self` for chaining.
- `decay(factor=0.9)`: multiplies all axes by `factor`. This simulates emotional fading over time.
- `reset()`: zeroes all axes.

### Serialization

A state serializes to a plain dictionary, so you can persist it as JSON and restore it later.

```python
d = state.to_dict()
restored = EmotionalState.from_dict(d)
```
### Arithmetic

States support vector arithmetic directly, so you can combine or scale two running states without unpacking their axes by hand.

```python
combined = state_a + state_b   # → new EmotionalState (vector sum)
scaled   = state * 0.5         # → new EmotionalState
assert state_a == state_b      # → compares snapshots
```
## EmotionTimeline

`EmotionTimeline` (`state.py:EmotionTimeline`) records a sequence of `EmotionalState` snapshots, enabling temporal analysis. Append a snapshot after each event, and the timeline tracks how the state moves over the whole sequence.

```python
from emotion_algebra.state import EmotionTimeline

timeline = EmotionTimeline()
timeline.append(state_t0)
timeline.append(state_t1)
timeline.append(state_t2)

timeline.drift()               # np.ndarray, state[-1] - state[0]
timeline.dominant_sequence()   # list[Emotion | None], per-snapshot dominant
```
### Use cases

| Goal | Method |
|------|--------|
| Track mood shift over a conversation | `append()` after each turn, then inspect `drift()` |
| Find the most frequent dominant emotion | `Counter(timeline.dominant_sequence())` |
| Persist state across sessions | `timeline.to_dict()` then JSON then `from_dict()` |

## Worked example: dialogue mood tracker

This example tracks how a conversation's mood moves across three turns, decaying the running state a little after each one.

```python
from emotion_algebra.state import EmotionalState, EmotionTimeline
from emotion_algebra import affect_from_text

timeline = EmotionTimeline()
turns = [
    "I'm so excited about this!",
    "But I'm also a bit worried 😟",
    "Everything turned out great 🎉",
]
state = EmotionalState()
for turn in turns:
    state.apply(affect_from_text(turn))
    state.decay(0.85)
    timeline.append(state)

print(timeline.dominant_sequence())
print("Net drift:", timeline.drift())
```
---
[← Building an agent](agents.md) · [Home](index.md) · [The laws →](core-laws.md)
