# Text & Emoji Analysis

Three pipelines for inferring emotions from natural language and emoji. All return either a named `Emotion` or a full `EmotionalState`.

---

## 1. Word lexicon pipeline

No external dependencies. Uses the bundled NRC/EmoLex-style CSV (`word_emotion_lexicon.csv`).

```python
from emotion_algebra.text import from_text, score_text

from_text("fury and rage")        # → Emotion (most frequent lexicon match)
score_text("joy and happiness")   # → EmotionalState (all matches accumulated)
```

`from_text` returns the single dominant emotion. `score_text` returns the full accumulated state — richer when the text contains multiple emotions.

---

## 2. Emoji map pipeline

~90 Unicode emoji → Plutchik emotion, covering all 8 primaries and intensity variants.

```python
from emotion_algebra.emoji import from_emoji, score_emojis, from_emojis

from_emoji("😊")                 # → Emotion("serenity")
from_emoji("😡")                 # → Emotion("rage")
score_emojis("Best day! 😄🎉")  # → EmotionalState
from_emojis("Best day! 😄🎉")   # → Emotion (dominant)
```

`score_emojis` scans the string character-by-character; each mapped emoji contributes weight=1.0. Non-emoji characters are ignored.

### Custom mappings

```python
from emotion_algebra.emoji import register_emoji, unregister_emoji

register_emoji("🤖", "trust")    # override or extend EMOJI_EMOTION_MAP
unregister_emoji("🤖")           # remove; canonical map is restored
```

User registrations take priority over the canonical `EMOJI_EMOTION_MAP` and affect all functions (`from_emoji`, `score_emojis`, `DeepMojiAdapter`).

---

## 3. Mixed pipeline (recommended)

`score_mixed` / `from_mixed` process both word tokens and emoji characters in a single pass.

```python
from emotion_algebra.text import score_mixed, from_mixed

score_mixed("I'm so happy 😄🎉")   # → EmotionalState (words + emoji blended)
from_mixed("grief and sorrow 😭")  # → dominant Emotion
```

This is the recommended entry point for unstructured text that may contain both natural language and emoji.

---

## 4. DeepMoji / classifier bridge

`DeepMojiAdapter` (`emoji.py:DeepMojiAdapter`) consumes `{emoji: probability}` distributions as produced by DeepMoji (Felbo et al. 2017) and compatible models.

```python
from emotion_algebra.emoji import DeepMojiAdapter

adapter = DeepMojiAdapter()

# From a raw probability dict
scores = {"😂": 0.45, "😊": 0.30, "😭": 0.25}
adapter.from_scores(scores)              # → Emotion (dominant)
adapter.score_state(scores)             # → EmotionalState (full blend)

# From a ranked list (e.g. top-k output)
ranked = [("😂", 0.45), ("😊", 0.30)]
adapter.from_ranked(ranked, top_k=1)    # → Emotion
```

Emojis not in `EMOJI_EMOTION_MAP` (or the user registry) are silently skipped.

---

## 5. Neural engine — DeepMojiONNXAdapter

`DeepMojiONNXAdapter` (`deepmoji.py:DeepMojiONNXAdapter`) is the canonical neural text-to-emotion engine. It runs the DeepMoji ONNX model, converts top-k emoji probabilities through `DeepMojiAdapter`, and returns `Emotion` / `EmotionalState`.

`deepmoji-onnx` is a **core dependency** — no extra required. The model is downloaded from HuggingFace on first use and cached to `~/.cache/deepmoji`.

```python
from emotion_algebra.deepmoji import DeepMojiONNXAdapter

adapter = DeepMojiONNXAdapter()                        # downloads model once
emotion = adapter.analyze("I'm absolutely furious!")   # → Emotion or None
state   = adapter.score("Best day ever! 😄")           # → EmotionalState
scores  = adapter.top_emoji_scores("rainy days")       # → {emoji: prob, ...}
```

---

## Signal comparison

| Pipeline | Input | Deps | Best for |
|----------|-------|------|----------|
| `from_text` / `score_text` | words | none (lexicon) | Clean English prose |
| `from_emoji` / `score_emojis` | emoji chars | none | Social media, chat |
| `score_mixed` / `from_mixed` | words + emoji | none | General unstructured text |
| `DeepMojiAdapter` | `{emoji: prob}` | none | Raw emoji probability dicts |
| `DeepMojiONNXAdapter` | text | `deepmoji-onnx` (core) | Neural inference, highest accuracy |
