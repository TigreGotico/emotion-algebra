# emotion-algebra documentation

A library for representing, reasoning about, and computing with emotion.

There are two ways in. Pick whichever fits.

---

## "I want to use it"

| | |
| --- | --- |
| **[Quickstart](quickstart.md)** | Five minutes, hands-on. Everything you need to be productive. |
| **[Building an agent](agents.md)** | Set points, drives, moods, temperament, the pattern for anything that has to *feel* over time. |
| **[Text & emoji](text_emoji.md)** | Getting emotion out of language, and which method to trust. |
| **[Interop](interop.md)** | PAD, VAD, the circumplex, and converting to whatever your other tools speak. |
| **[CLI](cli.md)** | `emotion-algebra` from the shell. |
| **[API reference](api_reference.md)** | Every public symbol. |

## "I want to know if I can trust it"

| | |
| --- | --- |
| **[The model](theory.md)** | Why these five coordinates, and not the ones every other library uses. |
| **[Evidence](evidence.md)** | Every construct, its grade, and its citation. The library's own epistemics. |
| **[The laws](core-laws.md)** | The algebra, formally, including the laws it deliberately *refuses*. |
| **[Appraisal](appraisal.md)** | Where emotions come from. The generative layer. |
| **[Neurochemistry](neurochemistry.md)** | Neuromodulators as computational roles. |
| **[The models](models.md)** | Every model this library implements, its grade, and how they map. |

---

## The one idea

If you read nothing else, read this.

Anger and fear are **both unpleasant** and **both highly aroused**. So a model
with only valence and arousal, which is almost every model, *cannot tell them
apart*.

What separates them is **control**. Anger is what you feel when something is wrong
and you can act on it. Fear is what you feel when you can't.

That axis, `potency`, is why this library exists:

```python
from emotion_algebra import prototype, dominant

anger = prototype("anger")   # unpleasant, aroused, potency +0.60
fear  = prototype("fear")    # unpleasant, aroused, potency -0.60

dominant(anger.blend(fear, 0.5))
# 'distress'
```
Libraries built on Plutchik's wheel place anger and fear at *opposite ends of one
axis*, so that blend comes out as the **zero vector**, perfect neutrality. Blend
the two most intense negative states a person can have, and get *calm*.

Everything else in these docs follows from taking that seriously.

---

## The five coordinates

```python
AffectState(
    positivity,        # [0, 1]   how good it feels
    negativity,        # [0, 1]   how bad it feels     (yes, both at once)
    potency,           # [-1, 1]  how in-control you feel
    arousal,           # [0, 1]   how activated you are
    unpredictability,  # [0, 1]   how unexpected it is
)
```
Derived for free:

- **`valence`** = `positivity − negativity`, the familiar single number, when you
  want it.
- **`ambivalence`** = `min(positivity, negativity)`, what that single number
  destroys. Graduation day is genuinely happy *and* sad, and a signed axis cannot
  say so.

## What the library will not do

- **No negation.** Sadness is not "minus joy", it has its own pull (withdraw,
  seek help), and that is not "negative approach".
- **No addition.** Emotions do not sum. They **mix**. A convex mixture stays inside
  the space. Addition leaves it.
- **No pretending.** Every conversion between models declares what it destroys,
  and every construct declares how well-evidenced it is.
