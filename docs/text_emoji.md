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

## The language boundary

The text layer reads **English**. Hand it anything else and it raises
`UnsupportedLanguageError`.

That refusal is deliberate, and it is worth being precise about why, because
"unsupported" usually means *degraded* and here it means something worse.

DeepMoji was trained on English tweets. The eight typographic cues around it are
English orthography — `?` is the question mark, capital letters are shouting,
`([a-z])\1{2,}` is a stretched word, and *very / really / totally* are the
intensifiers. Give that pipeline a sentence of Arabic and none of it holds:

| cue | on Arabic |
|---|---|
| `?` count | Arabic asks questions with **`؟`** (U+061F). Reads **0.0 on every Arabic question ever written.** |
| CAPS ratio | Arabic is **caseless**. Not "rarely capitalised" — the category does not exist. Reads **0.0 forever.** |
| `([a-z])\1{2,}` | Matches **no Arabic at all.** |
| intensifier list | Contains **no Arabic words.** |

And the emphasis-stripping guard — the thing that stops `!` being read as
excitement and quietly neutralising the valence of *"this is unacceptable!!!"* —
is an ASCII regex. On Arabic it matches nothing, so the guard silently does
nothing and the bug it exists to prevent comes straight back.

**None of this raises.** It returns an `AffectState`: five plausible floats, with
no indication that four of the eight arousal cues were structurally dead and the
encoder was reading a language it has never seen. An affect reading is consumed
as *evidence* by everything downstream of it — the tendency it implies, the drive
it creates, the reply it shapes. A wrong one does not degrade that decision, it
corrupts it, silently, forever.

So the text layer refuses. **A confident wrong number is worse than no number.**

### What is *not* refused

Only the `text → AffectState` arrow is language-bound. Everything else in the
library — appraisal, tendency, homeostasis, the neuromodulator readout, blending,
the distance metric — operates on five floats and has never seen a word. If you
can produce an `AffectState` for Portuguese or Arabic by some other means, the
entire rest of the library works on it unchanged.

### Language profiles

`emotion_algebra.lang` describes what each language's typography actually *does*.
The eight features are defined as **functional channels** — emphasis,
questioning, shouting, elongation, length, trailing-off, intensification — and
each `LanguageProfile` says how its language realises them.

```python
from emotion_algebra import typographic_features, CHANNELS

features, available = typographic_features("لماذا؟؟", lang="ar")
features[CHANNELS.index("questioning")]   # 0.4 — read with Arabic's eyes
```

The interesting case is shouting. Arabic cannot capitalise, but Arabic writers do
stretch words for emphasis with the **kashida** (tatweel, U+0640): `مرحبـــــا` is
the written equivalent of raising your voice. Same channel, same feature index,
different orthography.

Where a language realises a channel with *nothing at all*, the profile says so,
and `typographic_features` returns an **availability mask** alongside the values.
A missing channel is then a known absence rather than a silent zero — which is
the difference between *missing data* and *evidence of calm*, and they are not
the same thing.

Profiles exist for `en`, `pt` and `ar`. Only `en` has an encoder behind it. The
other two describe the typography correctly and are what the multilingual encoder
will be built on; until that encoder is fitted **and evaluated**, the honest
answer to "what does this Arabic sentence feel like" is that this library does
not yet know.
