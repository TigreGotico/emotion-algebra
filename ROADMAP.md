# emotion_algebra — Revival Roadmap

## Why revive

The only open-source Python library implementing **emotion algebra** — emotions as first-class mathematical objects with operator overloading (`joy + trust == love`, `-anger == fear`, `rage + 1 == "hyper rage"`). Built on two academic theories (Plutchik's Wheel + Hourglass of Emotions). No comparable library exists. Natural fit for OVOS/HiveMind to modulate conversational tone based on detected emotional state.

---

## Phase 1 — Stabilize (week 1)

**Goal:** Clean, installable core with zero broken dependencies.

- [ ] Replace `setup.py` with `pyproject.toml`; add `python_requires = ">=3.10"`
- [ ] Split into core package and optional integrations:
  - Core (no external deps): `plutchik.py`, `composite_emotions.py`, `feelings.py`, `emotions.py`, `behaviour.py`
  - Optional `[lexicon]`: `lexicons.py` + `word_emotion_lexicon.csv`
  - Optional `[deepmoji]`: `deepmoji.py` (isolate — 2169 lines, archived upstream model)
  - Optional `[tagging]`: `tag.py` (ParallelDots API — unreliable, external)
- [ ] Write unit tests for the algebra — it's fully deterministic:
  - Intensity arithmetic: `annoyance + 1 == anger`, `anger + 1 == rage`
  - Negation: `-anger == fear`, `-joy == sadness`
  - Composition: `joy + trust` returns a `Feeling` named `"love"`
  - Vector representation: 4D emotion vectors
  - Boundary conditions: offset/hyperintensity caps
- [ ] CI: GitHub Actions for Python 3.10 / 3.11 / 3.12

---

## Phase 2 — Modernize (week 2)

**Goal:** Typed, documented, ergonomic API.

- [ ] Add type hints to `Emotion`, `Feeling`, `CompositeEmotion`, `Dimension`, `Behaviour`
- [ ] Add docstrings with examples to all public classes
- [ ] Replace `numpy` matrix operations with explicit typed methods (reduce numpy dependency to optional)
- [ ] Expose clean top-level API in `__init__.py`:
  ```python
  from emotion_algebra import EMOTIONS, FEELINGS, emotion, feeling
  ```
- [ ] Add `Emotion.from_text(word)` that looks up word in lexicon → returns closest `Emotion`
- [ ] Write `docs/index.md` with algebra examples as the lead

---

## Phase 3 — Integrations (optional, future)

**Goal:** Connect to modern emotion detection backends.

- [ ] Replace DeepMoji integration with a maintained HuggingFace model (e.g. `j-hartmann/emotion-english-distilroberta-base`) that outputs Plutchik-compatible labels
- [ ] Add `Emotion.from_scores(dict)` — converts probability distribution from any classifier to an `Emotion` or `Feeling`
- [ ] Add OVOS integration: `EmotionParser` that accepts a voice utterance and returns an `Emotion` to influence TTS expressiveness
- [ ] Publish to PyPI as `emotion-algebra`

---

## Core algebra to preserve (regression-test before any refactor)

| Expression | Expected result |
|---|---|
| `annoyance + 1` | `anger` |
| `anger + 1` | `rage` |
| `rage + 5` | `Emotion("hyper rage")` |
| `-anger` | `fear` |
| `-joy` | `sadness` |
| `joy + trust` | `Feeling("love")` |
| `joy + surprise` | `Feeling("delight")` |
| `Emotion("joy").emotion_vector` | 4D numpy array |
| `Emotion("joy").valence` | `True` |
| `Emotion("rage") > Emotion("anger")` | `True` |
