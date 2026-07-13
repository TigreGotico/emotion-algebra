# Plutchik, the Hourglass, and Lövheim

If you came here for `joy + trust == love` and `-anger == fear`, they are all
still here. This page explains what they're for, and what they can't do.

## They are views, not the model

Every legacy model is registered as a **view** over the affect core. You can
convert to any of them, and from any of them, and every conversion tells you what
it destroys:

```python
from emotion_algebra import convert, fidelity, explain_loss, conversion_views

conversion_views()
# ['core', 'circumplex', 'hourglass', 'lovheim', 'neuro', 'pad', 'plutchik']

fidelity("plutchik", "core")     # Fidelity.LOSSY
print(explain_loss("plutchik", "core"))
```

They are kept because they are a **genuinely useful, widely understood
vocabulary**. They are not kept as claims about how emotion works.

**Use the wheel to talk. Use the core to compute.**

## Plutchik's wheel

```python
from emotion_algebra import resolve
from emotion_algebra.emotions import get_emotion
from emotion_algebra.feelings import get_feeling_from_emotions

anger = get_emotion("anger")
anger + 1                                   # rage        (intensity up)
anger - 1                                   # annoyance   (intensity down)
-anger                                      # fear        (the wheel's "opposite")

get_feeling_from_emotions("joy", "trust")   # 'love'      (the named dyad)
resolve("love")                             # Feeling(joy + trust)
```

**Graded `METAPHOR`.** Smith & Schneider (2009) ran over 2,000 statistical tests
and report the emotion-wheel theory *"receives no empirical support"*. The
opposite-pairs structure is borrowed from the colour wheel.

Its flagship pair is refuted directly. `-anger == fear` is the trick every emotion
library performs, and it is the least defensible thing any of them do: anger and
fear are **neighbours**, not opposites — both unpleasant, both aroused, differing
on control.

So in this library `-anger` is a **lexical fact about a wheel**, not a law of the
model. `AffectState` has no `__neg__`, and a test asserts its absence.

## The Hourglass

Four signed axes (Sensitivity, Attention, Pleasantness, Aptitude), each `[-3, 3]`,
with a polarity formula:

```
polarity = (P + |At| − |S| + Ap) / 3
```

**Graded `METAPHOR`.** It is self-described as "a derivative of Plutchik's wheel",
built to compute a sentiment score. No factor-analytic derivation from data exists.

And look at that formula. `−|S|` — the **absolute value**. Both poles of the
Sensitivity axis *reduce* polarity, anger and fear alike. That is a formal
admission, inside the model's own arithmetic, that the axis is not hedonically
bipolar — which contradicts the bipolar geometry the model otherwise assumes.

The consequence is not subtle. On Hourglass axes:

```
(rage + terror) / 2  ==  [0, 0, 0, 0]   ->  neutrality
```

Blend the two most intense negative states a person can have, and get **calm**.
That is what conflating "negative valence" with "low control" onto one signed axis
costs you.

The conversion `hourglass → core` is graded `HEURISTIC` for exactly this reason:
Sensitivity conflates two things the evidence separates, and **that conflation
cannot be undone**. Splitting it back into valence and potency is a judgement call,
and the library says so:

```python
print(explain_loss("hourglass", "core"))
```

## Lövheim's cube

Three monoamines, eight Tomkins affects at the corners.

**Graded `SPECULATIVE`.** Never empirically tested; published in a venue that did
not practise external peer review. It ships because downstream consumers expect it
and because it is a serviceable labelling convention.

```python
from emotion_algebra import convert, prototype

convert(prototype("rage"), "core", "lovheim")   # -> LovheimPoint
```

For a neurochemical layer you can actually defend, see
[neurochemistry](neurochemistry.md) — same idea, but modulators mapped to
*computational roles* rather than to emotion names.

## The name collisions

Ten names are claimed by **both** dyad lineages — `love`, `optimism`, `awe`,
`contempt`, `remorse`, and five more. A Feeling is a Plutchik dyad (two
*primaries*); a CompositeEmotion is an Hourglass compound (the same pairing at
maximum intensity). So `love` is `joy + trust` as a Feeling but
`ecstasy + admiration` as a CompositeEmotion — the same dyad, a very different
intensity.

`resolve()` is the single sanctioned lookup. It **prefers the Feeling**, and the
preference is explicit rather than an accident of import order:

```python
from emotion_algebra import resolve, is_ambiguous, describe_collision

resolve("love")                      # Feeling(joy + trust)            [default]
resolve("love", prefer="composite")  # CompositeEmotion(ecstasy + admiration)

is_ambiguous("love")                 # True
describe_collision("love")
# {'name': 'love', 'feeling': ['joy', 'trust'],
#  'composite': ['ecstasy', 'admiration']}
```

Every collision's resolution is locked by a test, so the answer cannot drift.

## Should I use these?

**Yes, if** you want a familiar vocabulary, you're rendering something for humans,
or you're interoperating with a system that speaks Plutchik.

**No, if** you're computing anything that depends on the *structure* being right —
distances, blends, decisions, or anything where anger and fear must not be
confused with each other's opposites.

The core is there for that.
