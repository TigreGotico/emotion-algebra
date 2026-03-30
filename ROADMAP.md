# ROADMAP — emotion-algebra

Post-v1.0 feature direction. Organized by milestone; each section opens with the user story that motivates it.

---

## v1.1 — Emotional State

> *"As a dialogue system, I want to track a mutable emotional state across conversation turns and observe how it evolves."*

> *"As a game developer, I want an NPC to accumulate emotional reactions to events over time, then behave according to its current mood."*

### Features

**`EmotionalState`** — a mutable, weighted snapshot of the full 4-axis space.

- Holds a float weight per axis slot (not just integer flow steps)
- `state.apply(emotion, weight=1.0)` — blend an `Emotion` into the state
- `state.decay(factor=0.9)` — exponential decay toward neutral
- `state.dominant()` → the `Emotion` or `Feeling` with the highest current weight
- `state.snapshot()` → immutable `EmotionVector` (numpy array) for logging/ML

**`EmotionTimeline`** — ordered sequence of `EmotionalState` snapshots.

- `timeline.append(state)` — record a moment
- `timeline.drift()` → net displacement vector from first to last snapshot
- `timeline.plot()` — ASCII or matplotlib sparkline per axis (optional dep)

**Serialization** — `Emotion`, `Feeling`, `CompositeEmotion` gain `to_dict()` / `from_dict()`.

```python
state = EmotionalState()
state.apply(get_emotion("anger"), weight=0.8)
state.apply(get_emotion("fear"),  weight=0.3)
state.decay(0.5)
state.dominant()   # → apprehension (blended, decayed)
```

---

## v1.2 — Text Integration

> *"As an NLP developer, I want to map arbitrary text to an emotion without requiring a GPU or external API."*

> *"As a content moderator, I want to score the emotional tone of a user message using only the installed library."*

### Features

**Lexicon-first pipeline** (no new deps beyond existing `[lexicon]` extra):

- `from_text(text) -> Emotion | Feeling | None` — tokenize, look up each token in
  `word_emotion_lexicon.csv`, aggregate by majority/weighted vote
- `score_text(text) -> EmotionalState` — returns a full weighted state, not just top-1

**HuggingFace bridge** (new optional extra `[transformers]`):

- Thin adapter: `HFEmotionAdapter(model_name)` wraps any HF pipeline that outputs
  Plutchik-compatible labels (e.g. `j-hartmann/emotion-english-distilroberta-base`)
- `adapter.from_text(text) -> Emotion` — calls the model, maps label → `Emotion` via
  `EMOTIONS` lookup, returns a typed object
- Replaces the removed DeepMoji integration with a maintained, model-agnostic bridge

```python
adapter = HFEmotionAdapter("j-hartmann/emotion-english-distilroberta-base")
adapter.from_text("I can't believe they did that!")   # → outrage (CompositeEmotion)
```

---

## v1.3 — Distance and Similarity

> *"As a recommendation engine, I want to find emotions or content closest to a user's current mood."*

> *"As a researcher, I want to measure how emotionally similar two pieces of text are."*

### Features

**`emotion_distance(a, b) -> float`** — Euclidean distance in the 4-axis signed integer space.

**`closest_emotion(vector) -> Emotion`** — nearest named emotion to an arbitrary 4D point.

**`EmotionCluster`** — group emotions by proximity; useful for simplifying a complex
`CompositeEmotion` into the nearest named `Feeling`.

```python
emotion_distance(get_emotion("rage"), get_emotion("anger"))   # → 1.0
emotion_distance(get_emotion("joy"),  get_emotion("sadness")) # → 4.0
closest_emotion([0, 0, 2.3, 0.8])   # → joy
```

---

## v1.4 — CLI

> *"As a developer, I want to explore the algebra interactively from the terminal without writing Python."*

### Features

**`python -m emotion_algebra`** — REPL-style CLI:

```
$ python -m emotion_algebra "joy + trust"
love  (Feeling, valence=4, arousal=2, type=excited positive)

$ python -m emotion_algebra "rage - 2"
annoyance  (Emotion, sensitivity axis, flow=+1)

$ python -m emotion_algebra info anger
anger  |  sensitivity  |  flow=+2  |  valence=0  |  arousal=2  |  type=activated neutral
opposite: fear   |   intense: rage   |   mild: annoyance
```

**`emotion-algebra` entry point** in `pyproject.toml` `[project.scripts]`.

---

## v1.5 — Appraisal Layer

> *"As a cognitive scientist, I want to model not just what emotion is felt, but why — what event properties triggered it."*

### Features

**`Appraisal`** dataclass — structured cognitive evaluation of an event, following
Scherer's Component Process Model (2001):

| Dimension | Values |
|---|---|
| Novelty | `expected` / `unexpected` |
| Goal relevance | `relevant` / `irrelevant` |
| Goal congruence | `congruent` / `incongruent` |
| Agency | `self` / `other` / `circumstance` |
| Coping potential | `high` / `low` |

**`appraisal_to_emotion(appraisal) -> Emotion`** — deterministic mapping from appraisal
pattern to primary emotion (per Scherer 2001, Table 1).

```python
a = Appraisal(novelty="unexpected", goal_congruence="incongruent", agency="other")
appraisal_to_emotion(a)   # → anger
```

This separates *what happened* (appraisal) from *what is felt* (emotion), enabling
richer NPC AI and explainable affective systems.

---

## v2.0 — Continuous Space

> *"As an ML researcher, I want to work in a continuous emotion space, not a discrete integer lattice."*

### Features

- `FloatEmotion` — same 4-axis model with `float` flow values; compatible with `EmotionBase`
- `emotion_vector` returns `np.ndarray` of floats; all arithmetic preserves float precision
- `from_embedding(vec: np.ndarray) -> FloatEmotion` — project a high-dimensional
  embedding into the 4-axis Hourglass space via a learned linear map
- Enables direct integration with sentence transformers and VAD regression models

---

## Use Case Summary

| Use case | Key feature |
|---|---|
| Dialogue / chatbot mood tracking | `EmotionalState`, `decay()`, `dominant()` |
| NPC emotional AI in games | `EmotionalState`, `appraisal_to_emotion()` |
| Sentiment-aware content recommendation | `emotion_distance()`, `closest_emotion()` |
| Text emotion classification (no GPU) | `from_text()` lexicon pipeline |
| Text emotion classification (transformer) | `HFEmotionAdapter` |
| Interactive exploration / prototyping | CLI (`python -m emotion_algebra`) |
| Affective computing research | `FloatEmotion`, `from_embedding()` |
| Cognitive / appraisal modelling | `Appraisal`, `appraisal_to_emotion()` |

---

## Non-Goals

- **Full PAD model** — Dominance axis is out of scope; it would require restructuring the
  entire 4-axis Hourglass representation.
- **Audio/video affect recognition** — out of scope; use a dedicated multimodal model and
  bridge via `HFEmotionAdapter` or `Emotion.from_scores()`.
- **Therapy or clinical use** — this library models Plutchik/Cambria categorical emotions,
  not clinical diagnostic criteria.
