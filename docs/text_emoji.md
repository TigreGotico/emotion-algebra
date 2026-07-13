# Text & emoji

Getting emotion out of language. There are three paths, they are not equally good,
and this page tells you which to use.

## Short answer

**Use `affect_from_text`.** It is the only path that recovers `potency` — the axis
that separates a user who will escalate from one who will quietly leave.

```python
from emotion_algebra import affect_from_texts, dominant

angry, afraid = affect_from_texts([
    "This is the third time your app has lost my work. Fix it.",
    "I don't know if I'm doing this right and I'm scared I've broken something.",
])

angry.valence,  angry.potency    # -0.43, +0.16   -> 'annoyance'
afraid.valence, afraid.potency   # -0.47, -0.43   -> 'apprehension'
```

Near-identical valence. **Opposite potency.**

> **Trust valence and potency. Do not trust arousal.** See the numbers below.

## The three paths

### 1. The neural probe — `emotion_algebra.neural`

DeepMoji (Felbo et al. 2017) was trained on **1.2 billion tweets** to predict which
emoji a message carried. A 325-float linear probe, fitted on GoEmotions and
shipped with the package, maps its 64-emoji output onto the affect core.

```python
from emotion_algebra import affect_from_text, affect_from_texts

affect_from_text("I am furious about this")        # potency +0.66
affect_from_text("I'm terrified something broke")  # potency -0.53
```

Batch with `affect_from_texts` — the DeepMoji forward pass dominates the cost, so
one call with a hundred strings is far cheaper than a hundred calls.

The model weights (~90 MB) download on first use and are cached.

### 2. The word lexicon — `emotion_algebra.text`

A bag of words over `word_emotion_lexicon.csv`, with windowed negation and
intensifier handling (Taboada et al. 2011).

```python
from emotion_algebra.text import analyze

analyze("I am not joyful").dominant.name        # 'sadness'    (negation)
analyze("I am very joyful").dominant.name       # 'ecstasy'    (intensifier)
analyze("I am slightly joyful").dominant.name   # 'serenity'   (downtoner)
```

Good for: transparency, offline use, no model download, and word-level spans you
can show a user.

**Bad for potency.** It has no syntax and no word senses. On *"I'm scared I've
broken something"* it fires on **broken → anger** and reports an angry,
approach-motivated user — the exact opposite of the truth. That failure is not a
bug to be patched; it is what bag-of-words costs.

### 3. Emoji — `emotion_algebra.emoji`

Direct emoji→emotion mapping, and a neural adapter.

```python
from emotion_algebra import from_emoji, score_emojis, DeepMojiONNXAdapter

from_emoji("😡")
score_emojis("shipping today 🎉🎉")
```

## How they score

Measured on **EmoBank** (Buechel & Hahn 2017), held out — a different corpus, at
sentence level, never seen during fitting.

| axis | word lexicon | DeepMoji probe |
| --- | --- | --- |
| valence | +0.331 | **+0.466** |
| arousal | **+0.129** | +0.028 |

**Read that honestly.** The probe is much better at valence and **worse at
arousal — near zero.** Emoji usage carries hedonic tone far more than activation,
which is not obvious in advance but is plainly true.

### Arousal from text is unsolved here

Neither path is good enough. If your application depends on arousal from raw
text, this library will not give it to you, and it will not pretend otherwise.

The scripts that produce these numbers are in `scripts/validate/`, and they say
the same thing.

## Which should I use?

| If you need… | Use |
| --- | --- |
| **potency** (angry vs afraid) | `affect_from_text` — nothing else recovers it |
| the best **valence** | `affect_from_text` |
| **arousal** | neither; it isn't there |
| word-level **spans** to highlight | `text.analyze` |
| **no model download**, full transparency | `text.analyze` |
| emoji-heavy input | `emoji` / `affect_from_text` (DeepMoji was trained on exactly this) |
