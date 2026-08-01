# Text & emoji

Getting emotion out of language.

## Text

```python
from emotion_algebra import affect_from_texts, dominant

angry, afraid = affect_from_texts([
    "This is the third time your app has lost my work. Fix it.",
    "I don't know if I'm doing this right and I'm scared I've broken something.",
])

angry.valence,  angry.potency    # -0.43, +0.16   -> 'disgust'
afraid.valence, afraid.potency   # -0.47, -0.43   -> 'apprehension'
```
Near-identical valence. **Opposite potency.** One user will escalate. The other
will quietly leave.

Batch with `affect_from_texts`, the forward pass dominates the cost, so one call
with a hundred strings is far cheaper than a hundred calls.

## How it works

**DeepMoji** (Felbo et al. 2017) was trained on **1.2 billion tweets** to predict
which emoji a message carried. It has no theory of emotion at all, which is
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

Measured on **EmoBank** (Buechel & Hahn 2017), held out, a different corpus, at
sentence level, never seen during fitting.

| axis | held-out correlation |
| --- | --- |
| **valence** | **+0.42** |
| **arousal** | **+0.35** |
| **potency** | recovers the angry/frightened distinction (see above) |

### Emoji tell you how someone feels. Punctuation tells you how loudly.

Arousal is the interesting one. A probe built on the **emoji distribution alone**
scores +0.16. Eight **typographic cues**, exclamation density, capitals ratio,
stretched vowels (`sooo`), length, ellipses, intensifiers, score **+0.32 on their
own**, twice as well. Together they reach +0.35.

Emoji encode hedonic tone. **emphasis encodes activation**. SHOUTING is arousal.

So the probe gets both, and the typographic rows are wired to arousal and nothing
else, letting them touch valence measurably degraded it.

One subtlety worth knowing about: DeepMoji reads `!` as *excitement*, and
excitement looks positive to it. Left alone, both *"this is wonderful!!!"* and
*"this is unacceptable!!!"* drift toward **neutral valence**, nonsense in one
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
`affect_from_text` already handles emoji well, DeepMoji was trained on exactly
this signal, so reach for the map when you want an explicit, auditable lookup
rather than a learned one.

## Why there is no word lexicon

There used to be one. It mapped **`terrified` to positive sentiment**, `happy` to
*anticipation*, and `miserable` to *anger*, and on *"I'm scared I've broken
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
English orthography, `?` is the question mark, capital letters are shouting,
`([a-z])\1{2,}` is a stretched word, and *very / really / totally* are the
intensifiers. Give that pipeline a sentence of Arabic and none of it holds:

| cue | on Arabic |
|---|---|
| `?` count | Arabic asks questions with **`؟`** (U+061F). Reads **0.0 on every Arabic question ever written.** |
| CAPS ratio | Arabic is **caseless**. Not "rarely capitalised", the category does not exist. Reads **0.0 forever.** |
| `([a-z])\1{2,}` | Matches **no Arabic at all.** |
| intensifier list | Contains **no Arabic words.** |

And the emphasis-stripping guard, the thing that stops `!` being read as
excitement and quietly neutralising the valence of *"this is unacceptable!!!"*, is an ASCII regex. On Arabic it matches nothing, so the guard silently does
nothing and the bug it exists to prevent comes straight back.

**None of this raises.** It returns an `AffectState`: five plausible floats, with
no indication that four of the eight arousal cues were structurally dead and the
encoder was reading a language it has never seen. An affect reading is consumed
as *evidence* by everything downstream of it, the tendency it implies, the drive
it creates, the reply it shapes. A wrong one does not degrade that decision, it
corrupts it, silently, forever.

So the text layer refuses. **A confident wrong number is worse than no number.**

### What is *not* refused

Only the `text → AffectState` arrow is language-bound. Everything else in the
library, appraisal, tendency, homeostasis, the neuromodulator readout, blending,
the distance metric, operates on five floats and has never seen a word. If you
can produce an `AffectState` for Portuguese or Arabic by some other means, the
entire rest of the library works on it unchanged.

### Language profiles

`emotion_algebra.lang` describes what each language's typography actually *does*.
The eight features are defined as **functional channels**, emphasis,
questioning, shouting, elongation, length, trailing-off, intensification, and
each `LanguageProfile` says how its language realises them.

```python
from emotion_algebra import typographic_features, CHANNELS

features, available = typographic_features("لماذا؟؟", lang="ar")
features[CHANNELS.index("questioning")]   # 0.4, read with Arabic's eyes
```
The interesting case is shouting. Arabic cannot capitalise, but Arabic writers do
stretch words for emphasis with the **kashida** (tatweel, U+0640): `مرحبـــــا` is
the written equivalent of raising your voice. Same channel, same feature index,
different orthography.

Where a language realises a channel with *nothing at all*, the profile says so,
and `typographic_features` returns an **availability mask** alongside the values.
A missing channel is then a known absence rather than a silent zero, which is
the difference between *missing data* and *evidence of calm*, and they are not
the same thing.

## Portuguese and Arabic, experimental

```bash
pip install emotion-algebra[multilingual]
```
```python
from emotion_algebra import affect_from_text

affect_from_text("Não sei se estou a fazer isto bem", lang="pt")
```
English still goes through DeepMoji, unchanged. Portuguese and Arabic go through
`paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0), and **the probe behind them
was fitted on English gold and has never seen a word of either language.**

That sounds like cheating. It works because the encoder is *distilled so a
sentence and its translation land in the same place*:

|  | en~pt | en~ar | pt~ar |
|---|---|---|---|
| *"your app has lost my work"* | 0.87 | 0.78 | 0.95 |

...against **0.15** for the angry-English/frightened-Portuguese mismatch. If a
direction fitted in the English region of that space is the same direction in the
Arabic region, the mapping crosses for free.

**If.** That is a claim, and it is measured, not assumed.

### What actually transfers

Gold is [XED](https://github.com/Helsinki-NLP/XED) (CC-BY-4.0). The test is the
one the library lives or dies by: **do anger and fear still separate, and do they
separate on _potency_?**

| lang | n | accuracy | shuffled control | potency *d* | separating axis |
|---|---|---|---|---|---|
| en | 1200 | 0.695 | 0.467 | **+0.77** | potency |
| pt | 1099 | 0.656 | 0.529 | **+0.46** | potency |
| ar | 794 | 0.668 | 0.637 | +0.28 | **unpredictability** |

**Portuguese transfers.** Potency is the dominant axis, anger sits on the
high-potency side, and the result survives the shuffled-label control. The
complaint and the goodbye stay distinguishable.

**Arabic transfers only partially, and this is the honest headline.** Potency
keeps the right *sign* but attenuates by nearly two-thirds, and
**unpredictability dominates instead**. Held-out accuracy is only three points
above the majority baseline. So:

> **The anger/fear-on-potency distinction is NOT established for Arabic.**

The Arabic reading is usable and the numbers are not noise, but the axis that
justifies this library's existence is not the one doing the work there, and it
would be dishonest to ship it as though it were.

Two caveats, and they cut in opposite directions. The XED labels for *both*
languages are **projected** across subtitle alignments rather than
human-annotated, so this is weak gold, and the Portuguese is *Brazilian*. And
the Arabic text carries visible tokenisation damage (words run together:
`أنيكبيرجداًفيالسن`), which plausibly depresses the Arabic result rather than
reflecting a true failure of transfer. It is a lead, not an excuse.

### Why the prototypes are not translated

Only the *coordinates'* names could be. The coordinates themselves stay canonical
and language-neutral, because **there is no human-rated Arabic
valence/arousal/dominance lexicon in existence.** The Arabic entries in the
widely-used NRC lexicon family are *machine translations* of English sentiment
scores: English raters' judgements in Arabic clothing. Fitting Arabic prototypes
on them would launder an English opinion into an Arabic-looking number, which is
precisely the failure this library exists to name. The only open Portuguese norms
are Brazilian.

`prototypes.cross_lingual_transfer` is graded `CONTESTED` for exactly this reason:
GRID supports the four *dimensions* replicating across cultures, but not the
*per-term positions*.

### A note on the encoder

On English, the one language with real gold, the multilingual encoder scores
**+0.56 valence / +0.50 arousal** against DeepMoji's **+0.42 / +0.35**, on the
same held-out EmoBank split. It is, on that measure, the better English encoder.

DeepMoji remains the English path anyway. Its value here was never accuracy: it
is that a model trained on 1.2 billion tweets, with no theory of emotion, reached
for *potency* on its own to tell anger from fear. That is an independent witness,
and independent witnesses are worth more than a tenth of a correlation
coefficient.

## Emotion names in other languages

`dominant()` and `label()` take a `lang=`, and the CLI takes `--lang`:

```python
from emotion_algebra import prototype, dominant

mixed = prototype("fúria").blend(prototype("رعب"), 0.5)

dominant(mixed, lang="en")   # 'distress'
dominant(mixed, lang="pt")   # 'angústia'
dominant(mixed, lang="ar")   # 'كرب'
```
**This translates names. It does not translate coordinates**, and the distinction
is the whole point.

`raiva` returns the same point in the core as `anger` because this library
*assumes* it is the same point, not because it has checked. It cannot check.
There is no human-rated Arabic affective lexicon to check against, and the open
Portuguese norms are Brazilian. So `prototypes.cross_lingual_transfer` is graded
`CONTESTED`, and a translated label is a **label**, never a finding about
Portuguese or Arabic emotion terms.

The probabilities are byte-identical in every language. Only the keys move.

### What this looks like when it half-works

The library's founding example, in Arabic:

| message | potency | label |
|---|---|---|
| *"I'm scared I've broken something"* | **−0.488** | `خوف` (fear) |
| *"third time your app has lost my work"* | **+0.005** | `قبول` (acceptance) |

The **ordering is right**, the complaint sits above the goodbye on potency, which
is the property everything downstream depends on. But the Arabic complaint's
potency never climbs high enough to reach *anger*, so it lands on the wrong label.

That is exactly the attenuation the evaluation measured (`d = +0.28` for Arabic,
against `+0.46` for Portuguese). **The direction survives the crossing. The
calibration does not.** It is written down here rather than smoothed over, because
a user who trusts an Arabic label without knowing this would be trusting something
the evidence does not support.

---
[← Valence & arousal](valence_arousal.md) · [Home](index.md) · [The models →](models.md)
