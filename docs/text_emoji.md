# Text & emoji

Getting emotion out of language.

## Text

```python
from emotion_algebra import affect_from_texts, dominant

angry, afraid = affect_from_texts([
    "This is the third time your app has lost my work. Fix it.",
    "I don't know if I'm doing this right and I'm scared I've broken something.",
])

angry.valence,  angry.potency    # -0.43, +0.16   -> 'annoyance'
afraid.valence, afraid.potency   # -0.47, -0.43   -> 'apprehension'
```

Near-identical valence. **Opposite potency.** One user will escalate; the other
will quietly leave.

Batch with `affect_from_texts` — the forward pass dominates the cost, so one call
with a hundred strings is far cheaper than a hundred calls.

## How it works

**DeepMoji** (Felbo et al. 2017) was trained on **1.2 billion tweets** to predict
which emoji a message carried. It has no theory of emotion at all — which is
exactly why it is worth listening to. It is a representation of how people
*actually write* when they feel things.

A 325-float linear probe, fitted on **GoEmotions** (Demszky et al. 2020, 43,410
human-labelled comments) and shipped with the package, maps its 64-emoji output
onto the affect core.

The model weights (~90 MB) download on first use and are cached.

```python
from emotion_algebra import affect_from_features

# If you already have DeepMoji features, skip the encoder entirely.
affect_from_features(features)      # (n, 64) -> list[AffectState]
```

## What it is good at, and what it is not

Measured on **EmoBank** (Buechel & Hahn 2017), held out — a different corpus, at
sentence level, never seen during fitting.

| axis | held-out correlation |
| --- | --- |
| **valence** | **+0.42** |
| **arousal** | **+0.35** |
| **potency** | recovers the angry/frightened distinction (see above) |

### Emoji tell you how someone feels. Punctuation tells you how loudly.

Arousal is the interesting one. A probe built on the **emoji distribution alone**
scores +0.16. Eight **typographic cues** — exclamation density, capitals ratio,
stretched vowels (`sooo`), length, ellipses, intensifiers — score **+0.32 on their
own**, twice as well. Together they reach +0.35.

Emoji encode hedonic tone; **emphasis encodes activation**. SHOUTING is arousal.

So the probe gets both, and the typographic rows are wired to arousal and nothing
else — letting them touch valence measurably degraded it.

One subtlety worth knowing about: DeepMoji reads `!` as *excitement*, and
excitement looks positive to it. Left alone, both *"this is wonderful!!!"* and
*"this is unacceptable!!!"* drift toward **neutral valence** — nonsense in one
direction and dangerously wrong in the other. So the encoder is shown the text
with emphasis stripped, while the typographic cues see it intact:

```python
affect_from_text("this is wonderful")             # arousal 0.52, valence +0.56
affect_from_text("this is wonderful!!!")          # arousal 0.95, valence +0.56
affect_from_text("THIS IS COMPLETELY UNACCEPTABLE!!!")   # arousal 0.90, valence -0.51
```

Emphasis changes how *loud* something is. It does not change whether it is good or
bad. That is now true of the model as well as of people.

> Honest ceiling: EmoBank's arousal ratings have low inter-annotator agreement, so
> +0.35 from 72 linear features is decent but not a solved problem. A fine-tuned
> transformer would do better and cost far more.

## Emoji

```python
from emotion_algebra import from_emoji, score_emojis, EMOJI_EMOTION_MAP

from_emoji("😡")
score_emojis("shipping today 🎉🎉")
```

A direct emoji→emotion map, extensible with `register_emoji`. Note that
`affect_from_text` already handles emoji well — DeepMoji was trained on exactly
this signal — so reach for the map when you want an explicit, auditable lookup
rather than a learned one.

## Why there is no word lexicon

There used to be one. It mapped **`terrified` to positive sentiment**, `happy` to
*anticipation*, and `miserable` to *anger* — and on *"I'm scared I've broken
something"* it fired on **`broken` → anger**, reporting an angry, approach-motivated
user. The exact opposite of the truth.

A bag of words has no syntax and no word senses, and its labels were unsourced. It
failed this library's own evidence standard more badly than anything the library
criticises, so it was removed rather than shipped with a warning label.
