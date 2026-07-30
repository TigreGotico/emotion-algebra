# The models

This library does not implement *an* emotion model. It implements **several**, each faithfully, to its own author's specification, grades them by evidence, and
maps between them.

That is the point. Affective science does not agree with itself, and a library
that picks one theory and hides the rest is making a scientific claim it cannot
support. This one gives you the models, tells you what each is worth, and lets you
move between them without pretending the moves are free.

```python
from emotion_algebra import conversion_views, convert, fidelity, explain_loss

conversion_views()
# ['core', 'circumplex', 'hourglass', 'lovheim', 'neuro', 'pad', 'plutchik']
```
## The catalogue

| Model | Author | Grade | Reach for it when… |
| --- | --- | --- | --- |
| **[Affect core](theory.md)** | Fontaine, Scherer, Roesch & Ellsworth (2007) | `SUPPORTED` | you are computing. It is the hub. |
| **[Circumplex](interop.md)** | Russell (1980) | `ESTABLISHED` | you want the classic valence × arousal plane. |
| **[PAD / VAD](interop.md)** | Mehrabian & Russell (1974) | `SUPPORTED` | you are talking to the rest of the field. NRC-VAD, dimensional regressors. |
| **[Plutchik's wheel](taxonomy.md)** | Plutchik (1980) | `METAPHOR` | you want a *vocabulary*: 24 named emotions, 40 dyads, intensity ladders. |
| **[Hourglass](algebra.md)** | Cambria, Livingstone & Hussain (2012) | `METAPHOR` | you need SenticNet interop or a signed-integer lattice. |
| **[Lövheim's cube](lovheim.md)** | Lövheim (2012) | `SPECULATIVE` | something downstream speaks his three monoamines. |
| **[Neuromodulators](neurochemistry.md)** | Schultz. Doya. Yu & Dayan | `SUPPORTED` | your agent needs learning rates, exploration, drives. |

**A grade is not a verdict on usefulness.** Plutchik's wheel is graded `METAPHOR`
and is the most useful *vocabulary* in the field, it simply is not a finding about
the structure of emotion. Use the right tool, knowing what it is.

Full grade table with citations: [evidence](evidence.md).

## Why the core is the hub

Conversions route through one model rather than pairwise: N models need 2N maps
instead of N².

```
   Hourglass ─┐                    ┌─ PAD
   Plutchik ──┼──►  AFFECT CORE  ◄─┼─ circumplex
   Lövheim ───┘                    └─ NeuroState
```
The core got the job for two reasons: it is the best-evidenced, and it is the most
**expressive**. It is the only model here with a *potency* axis, and without
potency you cannot represent the difference between anger and fear at all. A hub
that lacked it would silently destroy that distinction on every conversion passing
through.

Every pair is reachable, **always**. A test asserts the graph is total.

## Every conversion declares its cost

```python
fidelity("pad", "core")        # Fidelity.LOSSY
fidelity("hourglass", "pad")   # Fidelity.HEURISTIC   -- the weakest leg wins

print(explain_loss("circumplex", "core"))
# potency and unpredictability. This is why the circumplex cannot tell
# anger from fear: they differ on potency, and it has no potency axis.
```
| Fidelity | Meaning |
| --- | --- |
| `EXACT` | Bijective on its subspace. |
| `LOSSY` | Information provably discarded, and `explain_loss` names it. |
| `HEURISTIC` | Calibrated, not derived. The numbers are a judgement call. |

A conversion that loses information is fine. One that loses it **silently** is not.

## Plutchik's wheel

Faithful to the source, arithmetic and all.

```python
from emotion_algebra.emotions import get_emotion
from emotion_algebra.feelings import get_feeling_from_emotions
from emotion_algebra import resolve, describe_collision

anger = get_emotion("anger")
anger + 1                                   # rage        (intensity up)
anger - 1                                   # annoyance   (intensity down)
-anger                                      # fear        (Plutchik's "opposite")

get_feeling_from_emotions("joy", "trust")   # 'love'      (the named dyad)
```
**Know what you are getting.** `-anger == fear` is *Plutchik's claim*, implemented
faithfully. It is also the claim Smith & Schneider (2009) tested across more than
2,000 statistical tests and found unsupported: anger and fear are **neighbours**,
both unpleasant and both aroused, differing on control.

So the arithmetic is correct *for Plutchik's model*, and Plutchik's model is not
correct about people. Both things are true, and the library says both. The core has
no `__neg__` for exactly this reason.

### Name collisions

Ten names, `love`, `optimism`, `awe`, `contempt`, `remorse`, and five more, are
claimed by **both** dyad lineages. A `Feeling` is a Plutchik dyad (two *primaries*). A `CompositeEmotion` is an Hourglass compound (the same pairing at maximum
intensity). So `love` is `joy + trust` as a Feeling but `ecstasy + admiration` as a
CompositeEmotion, the same dyad, a very different intensity.

`resolve()` is the sanctioned lookup, and its preference is explicit rather than an
accident of import order:

```python
resolve("love")                      # Feeling(joy + trust)              [default]
resolve("love", prefer="composite")  # CompositeEmotion(ecstasy + admiration)

describe_collision("love")
# {'name': 'love', 'feeling': ['joy', 'trust'],
#  'composite': ['ecstasy', 'admiration']}
```
Every collision's resolution is locked by a test, so the answer cannot drift.

## The Hourglass

Four signed axes, Sensitivity, Attention, Pleasantness, Aptitude, each `[-3, 3]`,
with Cambria's published polarity formula:

```
polarity = (P + |At| − |S| + Ap) / 3
```
Implemented as published, **including the `abs()` terms**, which are worth
understanding before you rely on the model.

`−|S|` means **both poles of Sensitivity reduce polarity**: anger and fear alike.
That is a formal statement, inside the model's own arithmetic, that the axis is not
hedonically bipolar. But the model's *geometry* treats it as bipolar, with anger at
`+3` and fear at `−3`. Those two commitments are in tension, and the consequence is
concrete:

```
(rage + terror) / 2  ==  [0, 0, 0, 0]   ->  neutrality
```
Blend the two most intense negative states and the Hourglass reports **calm**.

This is why `hourglass → core` is graded `HEURISTIC`: Sensitivity conflates negative
valence with potency, and **that conflation cannot be undone**. Splitting it back
apart is a judgement call, and `explain_loss` says so out loud.

Reach for the Hourglass when you need SenticNet interop or a signed-integer lattice.
Do not reach for it to decide whether someone is angry or afraid.

## Lövheim's cube

Three monoamines. Eight Tomkins affects at the corners. Implemented faithfully,
graded `SPECULATIVE`.

```python
from emotion_algebra import convert, prototype

convert(prototype("rage"), "core", "lovheim")   # -> LovheimPoint
```
Never empirically tested, and published in a venue that did not practise external
peer review. It is here because things downstream speak his three monoamines, and
letting them interoperate without endorsing the model is precisely what a
multi-model library is for.

For a neurochemical layer you can defend, see [neurochemistry](neurochemistry.md):
the same idea, but modulators mapped to *computational roles* rather than to emotion
names.

## Adding your own

```python
from emotion_algebra.projection import register_view, Fidelity

register_view(
    "my_model",
    to_core=lambda x: ...,      # -> AffectState
    from_core=lambda s: ...,    # -> your type
    fidelity=Fidelity.LOSSY,
    loses="what your model cannot represent",   # required unless EXACT
)
```
A lossy view that does not declare its loss raises `ValueError`. That is deliberate,
and it is the one rule this library will not bend.

---
[← Text & emoji](text_emoji.md) · [Home](index.md) · [Emotion taxonomy →](taxonomy.md)
